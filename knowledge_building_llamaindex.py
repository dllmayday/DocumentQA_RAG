#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     knowledge_building_llamaindex.py
   @Description:   使用 LlamaIndex 构建知识库（Milvus + Ollama）
                   - 语义感知分块（替代固定大小分块）
                   - 句子窗口索引（提升召回质量）
                   - 支持增量更新
-------------------------------------------------
"""

import os
import argparse
from typing import List, Optional

from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    Document,
    Settings
)
from llama_index.core.node_parser import (
    SentenceWindowNodeParser,
    SemanticSplitterNodeParser
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import connections, utility

load_dotenv()

# =====================
# Config
# =====================
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
MILVUS_DB_PATH = os.environ.get("MILVUS_DB_PATH", "./milvus.db")

# nomic-embed-text 输出维度
NOMIC_DIM = 768


def connect_milvus():
    """连接 Milvus (local file mode)"""
    connections.connect(
        alias="default",
        uri=os.path.abspath(MILVUS_DB_PATH)
    )


def get_embedding_model():
    """获取 LlamaIndex 的 Ollama Embedding 模型"""
    return OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_sentence_window_splitter():
    """
    句子窗口分块器：
    - 将文档拆成句子级别的 node
    - 检索时自动带回 window_size 个相邻句子作为上下文
    - 非常适合精确问答场景
    """
    return SentenceWindowNodeParser(
        window_size=3,          # 每个句子前后各带 3 句
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )


def get_semantic_splitter(emb_model):
    """
    语义分块器：
    - 基于 embedding 相似度在语义边界处切分
    - 比 RecursiveCharacterTextSplitter 效果好很多
    """
    return SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=emb_model,
    )


def build_index(
    doc_folder: str,
    collection_name: str = "llamaindex_docs",
    use_sentence_window: bool = True,
    rebuild: bool = False
):
    """
    构建 LlamaIndex 向量索引并存入 Milvus

    Args:
        doc_folder: 文档文件夹路径
        collection_name: Milvus collection 名称
        use_sentence_window: 是否使用句子窗口分块（推荐）
        rebuild: 是否重建已有 collection
    """
    print(f"[*] 连接 Milvus...")
    connect_milvus()

    # 如果要重建，先删旧 collection
    if rebuild and collection_name in utility.list_collections():
        from pymilvus import Collection
        Collection(name=collection_name).drop()
        print(f"[!] 已删除旧 collection: {collection_name}")

    print(f"[*] 加载文档: {doc_folder}")
    reader = SimpleDirectoryReader(
        input_dir=doc_folder,
        required_exts=[".txt", ".pdf"],
        recursive=True,
    )
    documents = reader.load_data()
    print(f"[+] 共加载 {len(documents)} 个文档")

    if not documents:
        print("[!] 未找到任何文档!")
        return None

    # 初始化 embedding 模型
    emb_model = get_embedding_model()
    Settings.embed_model = emb_model

    # 选择分块策略
    if use_sentence_window:
        print("[*] 使用 Sentence Window 分块策略")
        parser = get_sentence_window_splitter()
    else:
        print("[*] 使用 Semantic Splitter 分块策略")
        parser = get_semantic_splitter(emb_model)

    nodes = parser.get_nodes_from_documents(documents)
    print(f"[+] 生成了 {len(nodes)} 个节点")

    # 创建 Milvus vector store
    vector_store = MilvusVectorStore(
        uri=os.path.abspath(MILVUS_DB_PATH),
        collection_name=collection_name,
        dim=NOMIC_DIM,
        overwrite=rebuild,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 构建索引
    print(f"[*] 正在构建向量索引并存入 Milvus...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
    )

    print(f"[+] 索引构建完成! Collection: {collection_name}")
    return index


# =====================
# main
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LlamaIndex Knowledge Building")
    parser.add_argument("--doc_folder", type=str, required=True,
                        help="文档文件夹路径")
    parser.add_argument("--collection", type=str,
                        default="llamaindex_docs",
                        help="Milvus collection 名称 (default: llamaindex_docs)")
    parser.add_argument("--use_semantic_split", action="store_true",
                        help="使用语义分块（默认使用句子窗口分块）")
    parser.add_argument("--rebuild", action="store_true",
                        help="强制重建已有索引")

    args = parser.parse_args()

    build_index(
        doc_folder=args.doc_folder,
        collection_name=args.collection,
        use_sentence_window=(not args.use_semantic_split),
        rebuild=args.rebuild,
    )
