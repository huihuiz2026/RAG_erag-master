import os
import warnings
warnings.filterwarnings('ignore')

from typing import Any, List, Optional
from openai import OpenAI
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun

# ====== DashScope 配置 ======
# API Key 从环境变量 DASHSCOPE_API_KEY 读取（用户已在系统环境变量配置）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# DashScope 提供的 OpenAI 兼容接口地址
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 默认对话模型：qwen-plus（性价比高）；可改为 qwen-max / qwen-turbo
DASHSCOPE_LLM_MODEL = "qwen-plus"
# 默认 embedding 模型：text-embedding-v3（1024 维，与 Milvus 集合 dim=1024 对应）
DASHSCOPE_EMBED_MODEL = "text-embedding-v3"


class RagLLM(object):
    """直接可调用的 LLM 封装（非 LangChain 风格），通过 DashScope 调用 qwen 模型"""
    client: Optional[Any] = None

    def __init__(self):
        super().__init__()
        self.client = OpenAI(
            base_url=DASHSCOPE_BASE_URL,
            api_key=DASHSCOPE_API_KEY,
        )

    def __call__(self, prompt: str, **kwargs: Any):
        # qwen 是对话模型，必须用 chat.completions 接口，不能用旧的 completions
        completion = self.client.chat.completions.create(
            model=kwargs.get('model', DASHSCOPE_LLM_MODEL),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get('temperature', 0.1),
            top_p=kwargs.get('top_p', 0.9),
            max_tokens=kwargs.get('max_tokens', 4096),
            stream=kwargs.get('stream', False),
        )
        if kwargs.get('stream', False):
            return completion
        return completion.choices[0].message.content


class QwenLLM(LLM):
    """LangChain 风格的 LLM 子类，可以接入 LangChain 的 Chain/Agent 等"""
    client: Optional[Any] = None

    def __init__(self):
        super().__init__()
        self.client = OpenAI(
            base_url=DASHSCOPE_BASE_URL,
            api_key=DASHSCOPE_API_KEY,
        )

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        completion = self.client.chat.completions.create(
            model=kwargs.get('model', DASHSCOPE_LLM_MODEL),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get('temperature', 0.1),
            top_p=kwargs.get('top_p', 0.9),
            max_tokens=kwargs.get('max_tokens', 4096),
            stream=kwargs.get('stream', False),
            stop=stop,
        )
        return completion.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "dashscope_qwen_plus"


# ====== Embedding ======
# 用 DashScope 官方 Embedding（text-embedding-v3，输出 1024 维）
from langchain_community.embeddings import DashScopeEmbeddings


class RagEmbedding(object):
    def __init__(self,
                 model_name: str = DASHSCOPE_EMBED_MODEL,
                 dashscope_api_key: Optional[str] = None):
        # 优先用显式传入的 key，否则用环境变量里的
        self.embedding = DashScopeEmbeddings(
            model=model_name,
            dashscope_api_key=dashscope_api_key or DASHSCOPE_API_KEY,
        )

    def get_embedding_fun(self):
        return self.embedding