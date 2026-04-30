#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     knowledge_retrieval_llamaindex.py
   @Description:   使用 LlamaIndex 进行知识检索（Milvus + Ollama）
                   - 句子窗口检索（自动扩展上下文）
                   - HyDE（假设性文档嵌入）
                   - 相似度过滤 + 长上下文重排序
-------------------------------------------------
"""

import os
from typing import Optional, List

from dotenv import load_dotenv
from pymilvus import connections

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    PromptTemplate,
    ResponseMode
)
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    LongContextReorder,
)
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

load_dotenv()

# =====================
# Config
# =====================
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:0.5b")
MILVUS_DB_PATH = os.environ.get("MILVUS_DB_PATH", "./milvus.db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "llamaindex_docs")


def connect_milvus():
    """连接 Milvus"""
    connections.connect(
        alias="default",
        uri=os.path.abspath(MILVUS_DB_PATH)
    )


def get_embedding_model():
    """获取 Ollama Embedding 模型"""
    return OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_llm():
    """获取 Ollama LLM 模型"""
    return Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
    )


class LlamaIndexRAG:
    """
    LlamaIndex RAG Pipeline

    支持特性：
    - 句子窗口检索（Sentence Window Retrieval）：自动扩展上下文窗口
    - HyDE（Hypothetical Document Embeddings）：假设性文档嵌入增强检索
    - 相似度过滤：过滤低相关性结果
    - 长上下文重排序：优化 Lost in the Middle 现象
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        use_hyde: bool = True,
        similarity_top_k: int = 5,
        similarity_cutoff: float = 0.3,
    ):
        self.collection_name = collection_name
        self.use_hyde = use_hyde
        self.similarity_top_k = similarity_top_k
        self.similarity_cutoff = similarity_cutoff

        # 初始化 models
        self.emb_model = get_embedding_model()
        self.llm = get_llm()
        Settings.embed_model = self.emb_model
        Settings.llm = self.llm

        # 加载已有索引
        self.index = self._load_index()

        # 自定义 Prompt
        self._build_prompt()

        # 构建 Query Engine
        self.query_engine = self._build_query_engine()

    def _load_index(self) -> VectorStoreIndex:
        """从 Milvus 加载已有的向量索引"""
        print(f"[*] 从 Milvus 加载索引: {self.collection_name}")

        vector_store = MilvusVectorStore(
            uri=os.path.abspath(MILVUS_DB_PATH),
            collection_name=self.collection_name,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )
        return index

    def _build_prompt(self):
        """自定义 System Prompt"""
        qa_prompt = PromptTemplate(
            """你是一个严谨的文档问答助手。

## 角色设定
你只能基于下面的上下文信息来回答用户的问题。
如果上下文中没有包含足够的信息来回答该问题，请直接回复：
"抱歉，本地知识库中暂无该问题相关的信息。"

## 要求
- 回答要简洁、准确、有依据
- 可以引用上下文中的具体内容
- 不要编造或猜测上下文中没有的信息

## 上下文信息
---------------------
{context_str}
---------------------

## 用户问题
{query_str}
"""
        )
        # 保存 prompt 以便后续 update_prompts 使用
        self.qa_prompt = qa_prompt

    def _build_query_engine(self):
        """构建增强版 Query Engine"""
        # ---- 1. Retriever ----
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.similarity_top_k,
        )

        # ---- 2. Post-processors ----
        postprocessors = []

        # 句子窗口替换：把 window 替换回原始文本（给 LLM 更完整的上下文）
        postprocessor_window = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )
        postprocessors.append(postprocessor_window)

        # 过滤低相似度结果
        postprocessor_sim = SimilarityPostprocessor(
            similarity_cutoff=self.similarity_cutoff
        )
        postprocessors.append(postprocessor_sim)

        # 长上下文重排序：把最相关的放在中间（Lost in the Middle 现象优化）
        postprocessor_reorder = LongContextReorder()
        postprocessors.append(postprocessor_reorder)

        # ---- 3. Response Synthesizer ----
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT_ACCUMULATE,
            llm=self.llm,
        )

        # ---- 4. 组装 base Query Engine ----
        base_query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=postprocessors,
        )

        # 应用自定义 Prompt
        base_query_engine.update_prompts(
            {"response_synthesizer:prompt_template": self.qa_prompt}
        )

        # ---- 5. HyDE 包装（可选） ----
        if self.use_hyde:
            print("[*] 启用 HyDE (假设性文档嵌入)")
            hyde = HyDEQueryTransform(
                llm=self.llm,
                include_original=True,
            )
            query_engine = TransformQueryEngine(
                base_query_engine,
                hyde,
            )
        else:
            query_engine = base_query_engine

        return query_engine

    def query(self, question: str, stream: bool = False):
        """
        执行 RAG 查询

        Args:
            question: 用户问题
            stream: 是否流式输出

        Returns:
            回答字符串或生成器
        """
        if stream:
            streaming_response = self.query_engine.query(question)
            for text in streaming_response.response_gen:
                yield text
        else:
            response = self.query_engine.query(question)
            return str(response)


# =====================
# CLI 入口
# =====================
def main():

    connect_milvus()

    rag = LlamaIndexRAG(
        collection_name=COLLECTION_NAME,
        use_hyde=True,
        similarity_top_k=5,
        similarity_cutoff=0.3,
    )

    while True:
        q = input("\n请输入问题（输入 'exit' 退出）: ").strip()
        if q.lower() == "exit":
            print("再见!")
            break

        print("\n思考中...\n")
        answer = rag.query(q, stream=False)

        print(f"\n回答:\n{answer}\n")


if __name__ == "__main__":
    main()
