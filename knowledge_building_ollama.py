#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import uuid
import queue
import threading
from typing import List
import argparse

from dotenv import load_dotenv
from pymilvus import (
    connections, Collection,
    FieldSchema, CollectionSchema,
    DataType, utility
)

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings 
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# =====================
# Config
# =====================
OLLAMA_BASE_URL = "http://localhost:11434"
MILVUS_DB_PATH = os.environ.get("MILVUS_DB_PATH", "./milvus.db")

# =====================
# Model
# =====================
def get_emb():
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_BASE_URL
    )

def get_llm():
    return ChatOllama(
        model="qwen2.5:0.5b",
        base_url=OLLAMA_BASE_URL
    )

# =====================
# Milvus
# =====================
def connect():
    connections.connect(
        alias="default",
        uri=os.path.abspath(MILVUS_DB_PATH)
    )

def create_collection(name, dim=768):
    if name in utility.list_collections():
        return Collection(name)

    schema = CollectionSchema([
        FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema("doc_id", DataType.VARCHAR, max_length=64),
        FieldSchema("text", DataType.VARCHAR, max_length=65535),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=dim),
    ])

    col = Collection(name=name, schema=schema)

    col.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type": "AUTOINDEX"
        }
    )

    return col

# =====================
# Embedding（带截断）
# =====================
def embed_batch(emb, texts: List[str], batch_size=32):
    vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        # ⚠️ 防止超长崩溃
        batch = [t[:2000] for t in batch]

        vecs = emb.embed_documents(batch)
        vectors.extend(vecs)

    return vectors

# =====================
# Summary Worker（异步）
# =====================
class SummaryWorker(threading.Thread):

    def __init__(self, q, emb):
        super().__init__(daemon=True)
        self.q = q
        self.emb = emb
        self.llm = get_llm()

        self.prompt = ChatPromptTemplate.from_template(
            "请用100字总结：\n{doc}"
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    def run(self):
        col = create_collection("summary")
        col.load()

        while True:
            item = self.q.get()

            if item is None:
                break

            docs, ids = item

            texts = [d.page_content[:2000] for d in docs]

            summaries = self.chain.batch(
                [{"doc": t} for t in texts]
            )

            vectors = embed_batch(self.emb, summaries)

            col.insert([
                [str(uuid.uuid4()) for _ in summaries],
                ids,
                summaries,
                vectors
            ])

            col.flush()
            self.q.task_done()

# =====================
# Ingestor
# =====================
class Ingestor:

    def __init__(self):
        self.emb = get_emb()

        # ✔ splitter 只初始化一次（关键优化）
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50
        )

        # ✔ summary 队列
        self.q = queue.Queue(maxsize=1000)

        self.worker = SummaryWorker(self.q, self.emb)
        self.worker.start()

    def insert(self, name, docs, ids):

        col = create_collection(name)
        col.load()

        texts = [d.page_content for d in docs]
        vectors = embed_batch(self.emb, texts)

        col.insert([
            [str(uuid.uuid4()) for _ in docs],
            ids,
            texts,
            vectors
        ])

        col.flush()

    def load_docs(self, folder):

        docs = []

        for root, _, files in os.walk(folder):
            for f in files:
                path = os.path.join(root, f)

                if f.endswith(".pdf"):
                    docs += PyPDFLoader(path).load()
                elif f.endswith(".txt"):
                    docs += TextLoader(path).load()

        return docs

    def run(self, folder):

        docs = self.load_docs(folder)

        # ========= parent =========
        parent_chunks = self.parent_splitter.split_documents(docs)
        doc_ids = [str(uuid.uuid4()) for _ in parent_chunks]

        print("▶ parent")
        self.insert("parent_chunk", parent_chunks, doc_ids)

        # ========= child =========
        print("▶ child")

        sub_docs = []
        sub_ids = []

        for d, pid in zip(parent_chunks, doc_ids):
            parts = self.child_splitter.split_documents([d])
            for p in parts:
                sub_docs.append(p)
                sub_ids.append(pid)

        self.insert("child_chunk", sub_docs, sub_ids)

        # ========= summary（异步） =========
        # print("▶ summary (async)")
        # self.q.put((parent_chunks, doc_ids))

    def wait(self):
        self.q.join()

# =====================
# main
# =====================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simple command line parser with one variable.")
    parser.add_argument("--doc_folder", type=str, help="Query string to be processed.")
    args = parser.parse_args()
    doc_folder = args.doc_folder

    connect()

    ing = Ingestor()

    ing.run(doc_folder)

    # print("▶ waiting summary...")
    # ing.wait()

    print("✔ DONE")