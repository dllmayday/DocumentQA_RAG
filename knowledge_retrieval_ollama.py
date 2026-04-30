#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import List, Set

from pymilvus import connections, Collection, utility
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

MILVUS_DB_PATH = "./milvus.db"
OLLAMA_BASE_URL = "http://localhost:11434"


# =====================
# Milvus connect
# =====================
def connect_milvus():
    connections.connect(
        alias="default",
        uri=os.path.abspath(MILVUS_DB_PATH)
    )


# =====================
# Models（只初始化一次）
# =====================
_emb = None
_llm = None


def get_embeddings():
    global _emb
    if _emb is None:
        _emb = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=OLLAMA_BASE_URL
        )
    return _emb


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model="qwen2.5:0.5b",
            base_url=OLLAMA_BASE_URL
        )
    return _llm


# =====================
# Collection cache（关键优化）
# =====================
_collection_cache = {}


def get_collection(name: str):
    if name in _collection_cache:
        return _collection_cache[name]

    col = Collection(name, using="default")
    col.load()

    _collection_cache[name] = col
    return col


# =====================
# Vector search
# =====================
def search(col_name: str, query_vec, k=5):

    col = get_collection(col_name)

    res = col.search(
        data=[query_vec],
        anns_field="embedding",
        param={
            "metric_type": "COSINE",
            "params": {"ef": 128}   # ✔ 比 64 更稳
        },
        limit=k,
        output_fields=["doc_id", "text"]
    )

    doc_ids, texts = [], []

    for hit in res[0]:
        doc_ids.append(hit.entity.get("doc_id"))
        texts.append(hit.entity.get("text"))

    return doc_ids, texts


# =====================
# RAG pipeline
# =====================
class RAGPipeline:

    def __init__(self, query: str):
        self.query = query
        self.emb = get_embeddings()
        self.llm = get_llm()

    # ---------------------
    # child retrieval
    # ---------------------
    def child_retriever(self) -> Set[str]:
        qv = self.emb.embed_query(self.query)
        ids, _ = search("child_chunk", qv)
        return set(ids)

    # ---------------------
    # parent retrieval
    # ---------------------
    def get_parent(self, doc_ids: List[str]):

        col = get_collection("parent_chunk")

        out = []

        for did in doc_ids:
            res = col.query(
                expr=f'doc_id == "{did}"',
                output_fields=["text"]
            )
            if res:
                out.append(res[0]["text"])

        return out

    # ---------------------
    # LLM answer
    # ---------------------
    def ask_llm(self, docs: List[str]):

        prompt = ChatPromptTemplate.from_template("""
你是一个严谨的文档问答助手。

只能使用以下文档回答问题：

{context}

问题：
{question}

如果没有答案，请回答：
“抱歉，本地知识库中暂无该问题相关的信息。”
""")
        chain = prompt | self.llm

        # ✅ 渲染查看最终 prompt
        formatted = prompt.format_messages(
            context="\n\n".join(docs),
            question=self.query
        )
        print(formatted)   # ← 这里会显示完整填充后的消息

        return chain.invoke({
            "context": "\n\n".join(docs),
            "question": self.query
        }).content

    # ---------------------
    # main pipeline
    # ---------------------
    def run(self):

        print("▶ search child")

        child_ids = self.child_retriever()

        # ✔ fallback（避免空结果）
        if not child_ids:
            print("⚠ child empty fallback")
            child_ids = set()

        print("▶ fetch parent")

        docs = self.get_parent(list(child_ids))

        if not docs:
            return "抱歉，本地知识库中暂无该问题相关的信息。"

        print("▶ LLM")

        return self.ask_llm(docs)


# =====================
# CLI
# =====================
def main():

    connect_milvus()

    while True:
        q = input("Q: ").strip()
        if q.lower() == "exit":
            break

        rag = RAGPipeline(q)
        print("\n" + rag.run() + "\n")


if __name__ == "__main__":
    main()