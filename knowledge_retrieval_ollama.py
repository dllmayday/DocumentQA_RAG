#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     knowledge_retrieval_ollama.py
   @Author:        Luyao.zhang
   @Date:          2023/12/29
   @Description:   RAG retrieval with Ollama local models
-------------------------------------------------
"""
import os
from dotenv import load_dotenv
from typing import List
from langchain_milvus import Milvus
from pymilvus import connections, Collection
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

load_dotenv(".env")

# ------------ Ollama Configuration -----------
OLLAMA_BASE_URL = "http://localhost:11434"

def get_embeddings_model(model: str = "nomic-embed-text"):
    """获取 Ollama 嵌入模型"""
    return OllamaEmbeddings(model=model, base_url=OLLAMA_BASE_URL)

def get_chat_model(model: str = "qwen2.5:0.5b"):
    """获取 Ollama 聊天模型"""
    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1
    )

# ------------ Milvus -----------
# Use Milvus Lite (local file-based)
milvus_db_path = os.environ.get("MILVUS_DB_PATH", "./milvus.db")
connection_args = {"uri": milvus_db_path}


class RAG_pipeline(object):
    """
    构建RAG Pipeline (Ollama Version)
    """

    def __init__(self, milvus_connection_args, embeddings_model, query: str):
        """
        Args:
            milvus_connection_args: milvus连接信息
            embeddings_model: text嵌入模型
            query: 用户输入的请求
        """
        self.milvus_connection_args = milvus_connection_args
        self.embeddings_model = embeddings_model
        self.query = query

    def _ensure_connection(self):
        """确保 Milvus 连接"""
        try:
            connections.connect(**self.milvus_connection_args)
        except:
            pass

    def child_chunk_retriever(
            self,
            child_collection_name: str = "child_chunk"):
        """
        对child_chunk设置检索器retriever
        Returns:
            doc_id组成的列表
        """
        self._ensure_connection()
        
        # 构建用于索引child chunk的向量数据库
        vectorstore = Milvus(
            connection_args=self.milvus_connection_args,
            collection_name=child_collection_name,
            embedding_function=self.embeddings_model,
            enable_dynamic_field=True,
        )

        # Vectorstore retrieves the child chunks
        child_chunk_res = vectorstore.similarity_search(
            query=self.query, k=3)
        
        # 只返回doc_id组成的列表
        res_id_list = []
        for single_res in child_chunk_res:
            if "doc_id" in single_res.metadata:
                res_id_list.append(single_res.metadata["doc_id"])
        return res_id_list

    def summary_retriever(
            self,
            summary_collection_name: str = "summary"):
        """
        对summary_chunk设置检索器retriever
        Returns:
            doc_id组成的列表
        """
        self._ensure_connection()
        
        # 构建用于索引summary的向量数据库
        vectorstore = Milvus(
            connection_args=self.milvus_connection_args,
            collection_name=summary_collection_name,
            embedding_function=self.embeddings_model,
            enable_dynamic_field=True,
        )

        # Vectorstore retrieves the summary chunks
        summary_chunk_res = vectorstore.similarity_search(
            query=self.query, k=3)
        
        # 只返回doc_id组成的列表
        res_id_list = []
        for single_res in summary_chunk_res:
            if "doc_id" in single_res.metadata:
                res_id_list.append(single_res.metadata["doc_id"])
        return res_id_list

    def hypothetical_retriever(
            self,
            hypothetical_query_collection="hypothetical_query"
    ):
        """
        对hypothetical_query设置检索器retriever
        Returns:
            doc_id组成的列表
        """
        self._ensure_connection()
        
        try:
            vectorstore = Milvus(
                connection_args=self.milvus_connection_args,
                collection_name=hypothetical_query_collection,
                embedding_function=self.embeddings_model,
                enable_dynamic_field=True,
            )
            
            hypo_res = vectorstore.similarity_search(
                query=self.query, k=3)
            
            res_id_list = []
            for single_res in hypo_res:
                if "doc_id" in single_res.metadata:
                    res_id_list.append(single_res.metadata["doc_id"])
            return res_id_list
        except Exception as e:
            print(f"Warning: hypothetical_query collection not available: {e}")
            return []

    def get_parent_document(
            self,
            doc_id_list: List[str],
            parent_chunk: str = "parent_chunk"):
        """
        根据doc_id列表从parent_chunk集合中获取完整的文档内容
        """
        self._ensure_connection()
        
        # Use pymilvus directly with URI for Milvus Lite
        uri = self.milvus_connection_args.get("uri", "./milvus.db")
        connections.connect(uri=uri)
        collection = Collection(name=parent_chunk)
        collection.load()
        
        parent_chunk_list = []
        for doc_id in doc_id_list:
            # 向量查询
            try:
                retrieved_res = collection.query(
                    expr=f"doc_id in ['{doc_id}']",
                    offset=0,
                    limit=10,
                    output_fields=["text", "page_content"],
                    consistency_level="Strong"
                )
                if retrieved_res:
                    # 尝试获取文本内容
                    text = retrieved_res[0].get("text") or retrieved_res[0].get("page_content", "")
                    parent_chunk_list.append(text)
            except Exception as e:
                print(f"Warning: Failed to retrieve doc_id {doc_id}: {e}")
                continue

        return parent_chunk_list

    def build_prompt_get_answer(self, docs: List[str], stream=True):
        """
        构建prompt并返回LLM的答案
        """
        template = """
        你是一个文档问答机器人，请仅仅根据下面指定文档列表中的多个文档来回答提出的问题，不能依赖自己的任何先验知识，如果在指定的文档中没有找到问题的答案，
        请回答:'抱歉，本地知识库中暂无该问题相关的信息。'
        {context}
        问题：{question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        model = get_chat_model()
        
        if stream:
            llm_chain = prompt | model
            answer = llm_chain.stream(
                {"question": self.query, "context": docs})
            for ret in answer:
                yield ret.content
        else:
            chain = prompt | model
            answer = chain.invoke({"question": self.query, "context": docs})
            return answer

    def build_rag(self, stream=True):
        """
        构建RAG Pipeline
        Returns:
        """
        child_list = self.child_chunk_retriever()
        summary_list = self.summary_retriever()
        hypo_list = self.hypothetical_retriever()
        
        # 取交集（优先使用多个检索结果的交集）
        if child_list and summary_list:
            intersection_list = list(set(child_list) & set(summary_list))
            if intersection_list:
                doc_id_list = intersection_list
            else:
                # 如果没有交集，使用child_list
                doc_id_list = child_list
        else:
            doc_id_list = child_list if child_list else summary_list
        
        # 如果hypo_list有结果，也加入考虑
        if hypo_list:
            doc_id_list = list(set(doc_id_list) | set(hypo_list))
        
        parent_chunk_list = self.get_parent_document(doc_id_list)
        
        if stream:
            stream_res = self.build_prompt_get_answer(
                parent_chunk_list, stream=stream)
            for res in stream_res:
                print(res, end="", flush=True)
        else:
            answer = self.build_prompt_get_answer(
                parent_chunk_list, stream=stream)
            return answer


def main():
    while True:
        input_prompt = "\n请输入内容（输入 'exit' 退出程序）: "
        query = input(input_prompt)
        if query.lower() == "exit":
            print("程序退出。")
            break
        
        # 使用 Ollama 本地模型
        embeddings = get_embeddings_model()
        rag = RAG_pipeline(
            milvus_connection_args=connection_args,
            embeddings_model=embeddings,
            query=query
        )
        print("\n回答: ", end="")
        rag.build_rag(stream=True)
        print()  # 换行
        print("\nUsing Ollama local models")


if __name__ == '__main__':
    main()
