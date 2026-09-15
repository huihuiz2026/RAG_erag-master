# e-rag 项目 RAG 效果优化方案

> 本方案针对当前仓库 `RAG_erag-master/rag_erag/` 的具体代码现状提出，所有改造点都对应到实际文件、实际函数与实际痛点，不讨论与本项目无关的通用理论。
>
> 当前技术栈：**Gradio + FastAPI** 前后端 · **DashScope qwen-plus / text-embedding-v3** 模型 · **Chroma** 向量库 · **Neo4j** 图数据库 · **ReAct Agent** 路由

---

## 目录

- [一、优先级 P0（最容易见效，建议先做）](#一优先级-p0最容易见效建议先做)
- [二、优先级 P1（中等改造，效果明显）](#二优先级-p1中等改造效果明显)
- [三、优先级 P2（架构升级，长期收益）](#三优先级-p2架构升级长期收益)
- [四、优先级 P3（锦上添花）](#四优先级-p3锦上添花)
- [五、推荐落地路线（按周排）](#五推荐落地路线按周排)
- [六、一句话总结](#六一句话总结)

---

## 一、优先级 P0（最容易见效，建议先做）

### 1. 加入 Rerank 重排（最大幅度提升精度）

**针对文件**：`utils.py` 的 `run_rag_pipline`

**当前问题**：
```python
related_docs = zhidu_db.similarity_search(context_query, k=k)  # k=3
```
直接拿 Embedding 召回的 top-3 进 LLM，Embedding 召回的精度上限并不高，3 段里常常掺杂噪声。规章制度类文档中相似句子非常多，召回容易"串味"。

**改造方案**：召回阶段放大到 30，再用重排模型精排到 top 3~5。
- DashScope 自带 `gte-rerank` 系列 API，与现有 `OpenAI` 兼容客户端共用同一套鉴权，零成本接入
- 或者本地部署 `BAAI/bge-reranker-v2-m3`（多语言、显存友好）

```python
# 伪代码
candidates = zhidu_db.similarity_search(query, k=30)
reranked = dashscope_rerank(query, [d.page_content for d in candidates], top_n=3)
related_docs = [candidates[i] for i in reranked.indices]
```

**预期收益**：只加这一步就能让答案质量提升一个台阶，是 ROI 最高的单项改造。

---

### 2. ReAct 路由改成 Function Calling（Agent 稳定性）

**针对文件**：`router.py` 的 `Agent.text_completion` + `parse_latest_plugin_call`

**当前问题**：
```python
def parse_latest_plugin_call(self, text):
    i = text.rfind('\nAction:')
    j = text.rfind('\nAction Input:')
    ...
```
用字符串切片解析 LLM 输出，**只要 qwen-plus 一次输出格式漂移（少一个回车、多输出一句 Thought），`plugin_name` 就会为空**，整个请求被打回到"对不起，我不能回答这个问题"。这是隐形 bug 高发地带。

**改造方案**：qwen-plus 原生支持 OpenAI function calling，把三个工具改成 tools schema 传进去，DashScope 直接返回结构化 `tool_calls`，不再字符串解析。

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_guizha",
        "description": "查询公司规章制度（考勤/工时/请假/差旅）",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
}, ...]
completion = client.chat.completions.create(
    model="qwen-plus", messages=..., tools=tools, tool_choice="auto"
)
```

**附带好处**：可以一次返回多个 `tool_call`，未来想"既查制度又查图谱"也变得简单。

---

### 3. 给 chunk 加元数据，并改成结构化切分

**针对**：索引侧脚本（`zhidu_db` 集合现存数据）

**当前问题**：规章制度文档大概率是 PDF / Word，规章制度有非常清晰的结构（`第x章 / 第x条 / 附件x`）。如果是定长切分，会导致跨条款的内容被切散。

**改造方案**：
- 按 `章/节/条` 结构化切，每个 chunk 自带 metadata：
  ```python
  metadata = {
      "doc_name": "差旅管理办法",
      "chapter": "第三章",
      "article": "第七条",
      "effective_date": "2024-01-01"
  }
  ```
- 在 `run_rag_pipline` 把 metadata 拼进 prompt（"以下内容来自《差旅管理办法》第七条"），LLM 输出可以带出处
- 检索时支持 `where={"doc_name": "差旅管理办法"}` 过滤，让"差旅一节怎么规定"能精准命中

---

## 二、优先级 P1（中等改造，效果明显）

### 4. 混合检索（BM25 + Vector + RRF）

**针对文件**：`utils.py` 的 `zhidu_db.similarity_search`

**当前问题**：规章制度里有大量**专有名词、表格代号、文件编号**（如"国办发〔2013〕35号"），向量模型对这类离散 token 召回偏差大，BM25 反而精准。

**改造方案**：
- Chroma 不带 BM25，本地可以用 `rank_bm25` 维护一个内存 BM25 索引
- 用 LangChain 的 `EnsembleRetriever` 做 RRF 融合，约 10 行代码
- 或者把向量库换成支持 hybrid 的 Milvus / Qdrant / Weaviate

---

### 5. Query 改写 + 多轮上下文化

**针对文件**：`router.py` 的 `text_completion`

**当前问题（致命）**：Agent 完全没用 `history`，每一轮都是孤立分类。
用户问"差旅怎么报销？" → "那海外呢？"，第二轮会被分到 `other`，或者直接拿"那海外呢"去检索——肯定查不到。

**改造方案**：在 `text_completion` 入口加一步 Query Contextualization。
```python
rewritten = self.model(
    f"基于以下对话历史，把最后一句改写为独立完整的问题：\n{history}\n用户最新：{text}"
)
# 然后用 rewritten 去走分类和检索
```

**进阶**：叠一个 HyDE / Multi-Query —— 让 LLM 生成 3 个不同表述的查询同时检索，召回再合并，对模糊提问效果显著。

---

### 6. 实体抽取 + 实体消歧（图 RAG 的命门）

**针对文件**：`utils.py` 的 `parse_query` 和 `get_node`

**当前问题（多个）**：
1. `parse_query` 让 LLM 用 `^` 拼接关键词：
   ```python
   keywords = response.split('\n')[0].split('^')
   ```
   只取了第一行，但 qwen 输出经常前面有解释性文字，结果是空数组或脏关键词。
2. `get_node` 用 `name CONTAINS "{keyword}"` 模糊匹配：
   ```cypher
   MATCH (n:Investor) WHERE n.name CONTAINS "苹果" RETURN n.name
   ```
   遇到"苹果"会同时命中"苹果公司""苹果手机供应商"，但代码 `return record['name']` 拿第一条，结果完全是运气。
3. `ignore_words = ['公司', '分析', '投资']` 是硬编码黑名单，扩展性差。

**改造方案**：
- 关键词抽取改为 **function calling 输出 JSON schema**，强制
  ```json
  {"entities": [{"name":"阿里巴巴","type":"Company"}, ...]}
  ```
- `get_node` 改为：先 exact match，未命中再 fuzzy；多个候选时用 query embedding 与节点 name embedding 算相似度选最优
- 维护一份**实体别名表**（"阿里" → "阿里巴巴集团控股有限公司"），在 Cypher 之前先标准化

---

### 7. 图查询深度自适应

**针对文件**：`utils.py` 的 `gen_contexts`

**当前问题**：只查两种固定模板（带/不带 Investor），多跳关系完全用不上。

**改造方案**：
- Cypher 可变长路径 `MATCH (n)-[*1..3]-(m)`，对开放性问题更友好
- 或让 LLM 先判断"问的是 1 跳/2 跳/3 跳"再选模板
- **进阶 Text2Cypher**：把 Neo4j 的 schema（节点标签、关系类型）丢给 qwen，让它生成 Cypher（用 few-shot 提示限制范围），灵活度大幅提升

---

## 三、优先级 P2（架构升级，长期收益）

### 8. 父子文档检索（ParentDocument Retriever）

**针对**：索引和 `run_rag_pipline`

**当前问题**：精度和上下文长度是矛盾的——chunk 太小召回精度高但 LLM 看不懂全貌，chunk 太大召回噪声多。

**改造方案**：用小 chunk（200 字）建索引保证精度，但召回后返回它对应的**父 chunk**（800 字）给 LLM。LangChain 的 `ParentDocumentRetriever` 直接支持。规章制度场景非常契合。

---

### 9. 多向量索引（Query–Doc Asymmetry）

**针对**：索引侧

**当前问题**：用户问"出差能坐高铁一等座吗"，文档里写的是"乘坐铁路一等席的人员范围"，向量距离不会近。

**改造方案**：在索引阶段让 LLM 给每个 chunk **生成 3-5 个假设问题**（HyQ），对这些假设问题做 embedding 入库；检索时用问题向量召回，再回到原文。索引一次性成本，检索零额外开销。

---

### 10. 表格特殊处理（当前埋了坑）

**针对文件**：`utils.py` 的 `extract_tables_and_remainder` 和 `ui.py`

**当前问题**：现在的逻辑是 **LLM 生成完之后才在生成文本里正则抠 `<table>`**——这非常奇怪。LLM 一般不会自己输出完整 HTML 表格，除非 context 里就有，那其实想抠的是 **context 里的表格**。但 `extract_tables_and_remainder` 接受 `rag_response[1]`（context），又用 `re.sub` 把表格从 context 里删掉，导致传给前端"参考信息"里反而没了表格——前后矛盾。

**改造方案**：
- 索引阶段把 PDF 里的表格识别出来，转成 Markdown table 或 JSON，单独存一份 collection
- 检索时表格走专门通道（如 NL2SQL 或简单的列名匹配），文本走 chroma
- UI 端"参考表格信息"直接展示原表 HTML，不要从 LLM 输出里抠

---

### 11. 自动评估闭环

**针对**：整个项目

**改造方案**：建一份 50-100 条的金标 QA 集（覆盖三类问题），用 **RAGAS** 自动跑：
- `context_precision`：召回的 context 是否相关
- `context_recall`：标准答案需要的信息是否都被召回
- `faithfulness`：生成是否忠于 context
- `answer_relevancy`：回答是否切题

每次改完一个 P0/P1 跑一次，量化收益。**这是工程化最关键的一步——没有评估就无法迭代**。

---

### 12. 接入观测（Langfuse / DashScope 自带）

**针对**：整个项目

**当前问题**：`print(query)`、`print(llm_prompt)` 散落在 `utils.py`，生产环境根本没法 trace。

**改造方案**：接 Langfuse（开源、支持 OpenAI SDK 自动埋点），一行 wrapper 就能看到每次请求的 prompt / 召回 / 耗时 / token，定位 badcase 效率提升 10×。

---

## 四、优先级 P3（锦上添花）

### 13. 拒答与置信度

让 LLM 同时输出
```json
{"answer":"...", "confidence": 0.85, "citations":["chunk_id_1","chunk_id_3"]}
```
confidence < 0.5 时前端显示"建议咨询 HR 确认"，而不是硬给答案。

### 14. 缓存

- query embedding 缓存（同样的问题不要重复打 embedding API）
- 完整 QA 结果缓存（Redis，TTL 1h），对热门问题极其有效
- DashScope 的 prompt cache（system prompt 几 K 字符，每轮都重传，浪费 token）

### 15. 真正的多步 ReAct

现在 `text_completion` 调用一次工具就返回，不是真 ReAct。可以做成 loop：
```
Question → Action: get_guizha → Observation →
Thought: 信息不够 → Action: get_finance → Observation →
Final Answer
```
让 Agent 自主决定是否要补查。和 P0-#2 的 function calling 天然契合（多轮 tool_call 即可）。

---

## 五、推荐落地路线（按周排）

| 周次 | 任务 | 涉及文件 | 预期收益 |
|---|---|---|---|
| 第 1 周 | P0-#1 接入 DashScope rerank | `utils.py` | 检索精度 +20~40% |
| 第 1 周 | P0-#2 改 function calling | `router.py` `model.py` | 路由准确率 95% → 99% |
| 第 2 周 | P0-#3 + P1-#4 结构化切 + 混合检索 | 索引脚本 + `utils.py` | 召回率 +30% |
| 第 2 周 | P1-#5 Query 改写 + 多轮上下文 | `router.py` `ui.py` | 多轮可用 |
| 第 3 周 | P1-#6 #7 图 RAG 实体消歧 + Text2Cypher | `utils.py` | 图 RAG 从"碰运气"变可靠 |
| 第 3-4 周 | P2-#11 接入 RAGAS 评估 | 新建 `eval/` | 后续所有迭代有量化依据 |
| 长期 | P2-#9 #10 多向量 + 表格独立通道 | 索引重构 | 复杂查询场景 |

---

## 六、一句话总结

> **这个项目当前的瓶颈不在 LLM 选型，而在"召回质量 + Agent 稳定性 + 评估缺失"三件事上。**
> 三件事都不需要换模型、不需要换数据库，只需要在现有 Chroma + Neo4j + DashScope 框架里加：
>
> 1. **Rerank**
> 2. **Function Calling**
> 3. **RAGAS 评估**
>
> 这三招做完，效果会和现在判若两个系统。

---

*文档版本：v1.0 ｜ 创建日期：2026-06-26*