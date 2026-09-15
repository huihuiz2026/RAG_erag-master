# -*-coding:utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
import re
import traceback
import json
import random
import numpy as np
from langchain_chroma import Chroma
import chromadb
# 新版 langchain (>=0.1.0) 已将 PromptTemplate 移至 langchain_core.prompts
# 旧写法 `from langchain import PromptTemplate` 会报 ImportError
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from py2neo import Graph
from model import RagEmbedding, RagLLM, QwenLLM
# 从 model.py 导入 DashScope 配置常量，避免在 utils.py 中重复定义
from model import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, DASHSCOPE_LLM_MODEL
from prompt_cfg import rule_template, finance_template, keyword_prompt


llm = RagLLM()
embedding_model = RagEmbedding()
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
zhidu_db = Chroma("zhidu_db", 
                embedding_model.get_embedding_fun(), 
                client=chroma_client)
graph = Graph("bolt://localhost:7687", user='neo4j', password='zhao1051hui+-',name='neo4j')

def run_chat(prompt, history=[]):
    """多轮对话接口：使用 DashScope 的 qwen 模型，兼容 OpenAI SDK 调用方式

    - 支持流式输出（stream=True）
    - 支持多轮历史对话（history 是 [(user_msg, assistant_msg), ...] 形式）
    """
    # 使用 DashScope 提供的 OpenAI 兼容接口（与 model.py 中 RagLLM 共用同一套配置）
    client = OpenAI(
        base_url=DASHSCOPE_BASE_URL,        # https://dashscope.aliyuncs.com/compatible-mode/v1
        api_key=DASHSCOPE_API_KEY,          # 从环境变量 DASHSCOPE_API_KEY 读取
    )

    history_msg = []
    for idx, msg in enumerate(history):
        if idx == 0:
            continue
        history_msg.append({"role": "user", "content": msg[0]})
        history_msg.append({"role": "assistant", "content": msg[1]})
    # print(history_msg)

    chat_completion = client.chat.completions.create(
        messages=history_msg+[
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        max_tokens=4096,  # 最大生成的token数量。
        stream=True,      # 开启流式输出
        model=DASHSCOPE_LLM_MODEL,   # 默认 qwen-plus，可在 model.py 中改为 qwen-max / qwen-turbo
        temperature=0.1,  # 控制生成文本的随机性。越低越确定，越高越随机。
        top_p=0.9,
    )
    return chat_completion

def run_rag_pipline(query, context_query, k=3, context_query_type="query", 
                    stream=True, prompt_template=rule_template,
                    temperature=0.1):
    if context_query_type == "vector":
        related_docs = zhidu_db.similarity_search_by_vector(context_query, k=k)
    elif context_query_type == "query":
        related_docs = zhidu_db.similarity_search(context_query, k=k)
    elif context_query_type == "doc":
        related_docs = context_query
    else:
        related_docs = zhidu_db.similarity_search(context_query, k=k)
    context = "\n".join([f"上下文{i+1}: {doc.page_content} \n" \
                        for i, doc in enumerate(related_docs)])
                        
    prompt = PromptTemplate(
                        input_variables=["question","context"],
                        template=prompt_template,)
    llm_prompt = prompt.format(question=query, context=context)
    
    if stream:
        response = llm(llm_prompt, stream=True)
        return (response, context)
    else:
        response = llm(llm_prompt, stream=False, temperature=temperature)
        return (response, context)

def parse_query(query, max_keywords=3):
    
    prompt_template = PromptTemplate(
                input_variables=["query_str", "max_keywords"],
                template=keyword_prompt,
            )
    
    final_prompt = prompt_template.format(max_keywords='3',
                                          query_str=query)
    response = llm(final_prompt)
    keywords = response.split('\n')[0].split('^')
    return keywords

def get_node(keyword, node_type):
    query = f"""
    MATCH (n:{node_type})
    where n.name CONTAINS "{keyword}"
    RETURN n.name as name
    """
    fetch_node = None
    print(query)
    results = graph.run(query)
    
    for record in results:
        return record['name']
    return fetch_node

def gen_contexts(investor_condition, 
                 company_condition, 
                 even_type_condition,
                 query_level=1,
                 exclude_content=False):
    if query_level == 1:
        query = f"""
        MATCH (i:Investor)-[:INVEST]->(c:Company)-[r:HAPPEN]->(e)
        WHERE 1=1 {investor_condition} {company_condition} {even_type_condition}
        RETURN i.name as investor,  c.name as company_name, e.name as even_type, r as relation
        """
    else:
        # query_level=2 的语义是"去掉投资者维度，只看公司-事件"，MATCH 里没有 i 变量，
        # 因此必须丢弃 investor_condition；否则当 query_level=1 查不到结果回退到这里时，
        # WHERE 子句里残留的 `and i.name = "..."` 会触发 Cypher 报错：
        #   ClientError [Statement.SyntaxError] Variable `i` not defined
        query = f"""
        MATCH (c:Company)-[r:HAPPEN]->(e)
        WHERE 1=1 {company_condition} {even_type_condition}
        RETURN c.name as company_name, e.name as even_type, r as relation
        """
    print(query)
    results = graph.run(query)
    contexts = []
    seen = set()    # 用 set 记录已出现的 context 字符串，做去重
    for record in results:
        context = ''
        record = dict(record)
        if 'investor' in record:
            context += f"{record['investor']} 投资了 {record['company_name']} \n"
        context = context + f"{record['company_name']} 发生了 {record['even_type']} \n 详细如下："
        for key, value in dict(record['relation']).items():
            if exclude_content:
                if key in ["title", "content"]:
                    continue
            context = context + f"\n  {key}: {value}"

        # 去重：相同的 context 字符串只保留一次
        # 原因：图数据库中 (投资者,公司) 和 (公司,事件) 可能有多条边，
        # Cypher 匹配会产生笛卡尔积，导致格式化后的字符串重复出现
        if context not in seen:
            seen.add(context)
            contexts.append(context)
    return contexts

def get_even_detail(keyword, exclude_content=False):
    investor = get_node(keyword, "Investor")
    company = get_node(keyword, "Company")
    even_type = get_node(keyword, "EventType")
    
    investor_condition = ""
    company_condition = ""
    even_type_condition = ""
    if investor:
        investor_condition = f' and i.name = "{investor}"'
    if company:
        company_condition = f' and c.name = "{company}"'
    if even_type:
        even_type_condition = f' and e.name = "{even_type}"'
        
    print(f"investor={investor_condition} company={company_condition} even_type={even_type_condition}")
    if investor_condition or company_condition or even_type_condition:
        contexts = gen_contexts(investor_condition, 
                                company_condition, 
                                even_type_condition,
                                query_level=1,
                                exclude_content=exclude_content)
        if len(contexts) == 0:
            contexts = gen_contexts(investor_condition, 
                                    company_condition, 
                                    even_type_condition,
                                    query_level=2,
                                    exclude_content=exclude_content)
        return contexts
    else:
        return []

def graph_rag_pipline(query, exclude_content=True, stream=True, temperature=0.1):

    keywords = parse_query(query, max_keywords=3)
    contexts = []
    ignore_words = ['公司', '分析', '投资']
    for keyword in keywords:
        if keyword in ignore_words:
            continue
        contexts.extend(get_even_detail(keyword=keyword, exclude_content=exclude_content))
    
    prompt = PromptTemplate(
                        input_variables=["question", "context"],
                        template=finance_template,)
    context = "\n========================\n".join(contexts)
    llm_prompt = prompt.format(question=query, context=context)
    print(llm_prompt)
    
    if stream:
        response = llm(llm_prompt, stream=True)
        return (response, context)
    else:
        response = llm(llm_prompt, stream=False, temperature=temperature)
        # 修复原代码 `return` 无返回值（返回 None）导致 Agent 拿不到答案的 Bug
        return (response, context)


def extract_tables_and_remainder(text):
    pattern = r'<table.*?>.*?</table>'
    tables = re.findall(pattern, text, re.DOTALL)
    remainder = re.sub(pattern, '', text, flags=re.DOTALL).strip()
    return tables, remainder


