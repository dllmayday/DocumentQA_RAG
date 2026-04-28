#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     knowledge_building_ollama.py
   @Author:        Luyao.zhang
   @Date:          2023/12/29
   @Description:   Knowledge building with Ollama local models
-------------------------------------------------
"""
import os
import argparse
from dotenv import load_dotenv
from typing import List
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import connections

import uuid
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
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
    return ChatOllama(model=model, base_url=OLLAMA_BASE_URL)

# ------------ Milvus -----------
# Use Milvus Lite (local file-based)
milvus_db_path = os.environ.get("MILVUS_DB_PATH", "./milvus.db")
connection_args = {"uri": milvus_db_path}


class UpdateTxtFile2Milvus(object):
    """
    从本地更新txt文件夹到milvus数据库，需要同时更新到4个milvus集合，分别为：
        parent_chunk: 用来存储parent document以及对应的uuid
        child_chunk: 用来存储child chunk以及对应的parent document的uuid
        summary：用来存储summary以及对应的parent document的uuid
        hypothetical_query: 用来存储 hypothetical queries以及对应的parent document的uuid
    """
    def __init__(
            self,
            document_folder: str,
            milvus_connection_args,
            embedding_model):
        self.document_folder = document_folder
        self.milvus_connection_args = milvus_connection_args
        self.embedding_model = embedding_model

    def get_pdf_txt_files_list(self):
        pdf_list = []
        txt_list = []
        for root, _, files in os.walk(self.document_folder):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_list.append(os.path.join(root, file))
                elif file.endswith(".txt"):
                    txt_list.append(os.path.join(root, file))
        return pdf_list, txt_list

    def load_split_doc(self, chunk_size: int = 400):
        """
        处理pdf以及txt格式的文档，并将文档按照指定的chunk size分成多个大chunk，并对每个大chunk生成对应的uuid
        Args:
            chunk_size:
        Returns:
        """
        chunk_doc_list = []
        # get pdf and txt document list
        pdf_list, txt_list = self.get_pdf_txt_files_list()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)

        # load pdf document
        pdf_loaders = [
            PyPDFLoader(pdf_file) for pdf_file in sorted(pdf_list)
        ]
        pdf_docs = []
        for loader in pdf_loaders:
            pdf_docs.extend(loader.load())
        pdf_docs = text_splitter.split_documents(pdf_docs)
        # load txt document
        txt_loaders = [
            TextLoader(
                txt_file) for txt_file in sorted(txt_list)]
        txt_docs = []
        for loader in txt_loaders:
            txt_docs.extend(loader.load())
        txt_docs = text_splitter.split_documents(txt_docs)
        # merge
        chunk_doc_list.extend(pdf_docs)
        chunk_doc_list.extend(txt_docs)

        doc_ids = [str(uuid.uuid4()) for _ in chunk_doc_list]
        return chunk_doc_list, doc_ids


    def build_and_write_parent_chunk_colle(
            self,
            docs: List,
            doc_ids: List,
            parent_collection_name: str = "parent_chunk"):
        """
        构建parent_chunk集合，并写入数据
        该集合的作用是用来通过doc_id索引到对应的parent document
        Args:
            docs:List 由load_split_txt_doc函数生成的doc list
            doc_ids:List 由load_split_txt_doc函数生成的doc_ids list
            parent_collection_name: 存储parent chunk的集合
        Returns:

        """
        # Create Milvus connection for Lite mode
        try:
            connections.connect(**self.milvus_connection_args)
        except:
            pass  # Connection may already exist
        
        # 构建用于索引parent chunk的向量数据库
        # enable_dynamic_field=True 自动处理额外字段（如 producer）
        vectorstore = Milvus(
            connection_args=self.milvus_connection_args,
            collection_name=parent_collection_name,
            embedding_function=self.embedding_model,
            enable_dynamic_field=True,
        )
        
        # 添加文档（确保 metadata 中包含 doc_id）
        docs_list = []
        for i, doc in enumerate(docs):
            _id = doc_ids[i]
            doc.metadata["doc_id"] = _id
            docs_list.append(doc)

        vectorstore.add_documents(docs_list)

    def build_and_write_child_chunk_colle(
            self,
            docs: List,
            doc_ids: List,
            child_collection_name: str = "child_chunk",
            child_chunk_size=200,
            child_chunk_oversize=50):
        """
        构建child_chunk集合，并写入数据
        Returns:

        """
        # Create Milvus connection for Lite mode
        try:
            connections.connect(**self.milvus_connection_args)
        except:
            pass  # Connection may already exist
        
        # 构建用于索引child chunk的向量数据库
        # enable_dynamic_field=True 自动处理额外字段（如 producer）
        vectorstore = Milvus(
            connection_args=self.milvus_connection_args,
            collection_name=child_collection_name,
            embedding_function=self.embedding_model,
            enable_dynamic_field=True,
        )
        
        # The splitter to use to create smaller chunks
        child_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_oversize
        )

        sub_docs = []
        for i, doc in enumerate(docs):
            _id = doc_ids[i]
            _sub_docs = child_text_splitter.split_documents([doc])
            for _doc in _sub_docs:
                _doc.metadata["doc_id"] = _id
            sub_docs.extend(_sub_docs)

        vectorstore.add_documents(sub_docs)

    def build_and_write_summary_colle(
            self,
            docs: List,
            doc_ids: List,
            summary_collection_name: str = "summary"):
        # Create Milvus connection for Lite mode
        try:
            connections.connect(**self.milvus_connection_args)
        except:
            pass  # Connection may already exist
        
        # 使用 Ollama 本地模型生成摘要
        summary_chain = (
            {"doc": lambda x: x.page_content}
            | ChatPromptTemplate.from_template("你是一个负责总结文档的助手，请以不超过100字总结给出的文档:\n\n{doc}")
            | get_chat_model()
            | StrOutputParser()
        )

        # batch summary
        summaries = summary_chain.batch(docs)

        # 构建用于索引summary的向量数据库
        # enable_dynamic_field=True 自动处理额外字段（如 producer）
        vectorstore = Milvus(
            connection_args=self.milvus_connection_args,
            collection_name=summary_collection_name,
            embedding_function=self.embedding_model,
            enable_dynamic_field=True,
        )
        
        summary_docs = [
            Document(page_content=s, metadata={"doc_id": doc_ids[i]})
            for i, s in enumerate(summaries)
        ]

        vectorstore.add_documents(summary_docs)

    def build_and_write_hypothetical_colle(
            self,
            docs: List,
            doc_ids: List,
            hypo_collection_name: str = "hypothetical_query",
    ):
        # 构建prompt并返回答案
        def _build_prompt_return_answer(chunk) -> List[str]:
            template = """
            你是一个善于提出问题的机器人，请根据下面指定的文档来提出2个假设的问题，以下是对要生成的2个假设问题的规范：
                每个假设问题的长度不超过20个汉字；
                生成的假设问题要明确，可以直接通过假设问题去思考，不要生成还要结合上下文才能理解的问题
                每个假设问题都应有明确的主谓宾，不应该包含没有上下文语境的词汇，比如'该法律'、'这个法律'要替换成明确的法律名称；
                假设问题的数据类型是python的字符串，输出的形式是python列表，该列表包含生成的3个假设问题。

            文档：{context}
            """
            prompt = ChatPromptTemplate.from_template(template)
            model = get_chat_model()
            chain = prompt | model
            answer = chain.invoke({"context": chunk})
            # answer_list = json.loads(answer)
            try:
                answer_list = eval(answer)
                return answer_list
            except BaseException:
                return []

        # Create Milvus connection for Lite mode
        try:
            connections.connect(**self.milvus_connection_args)
        except:
            pass  # Connection may already exist
    
        # 构建用于索引hypothetical_query的向量数据库
        vectorstore = Milvus(
            connection_args=self.milvus_connection_args,
            collection_name=hypo_collection_name,
            embedding_function=self.embedding_model,
            enable_dynamic_field=True,
        )
        
        hypo_docs = []
        for i, single_chunk in enumerate(docs):
            answer_list = _build_prompt_return_answer(single_chunk)
            for single_hypo_q in answer_list:
                hypo_docs.extend([
                    Document(page_content=single_hypo_q, metadata={"doc_id": doc_ids[i]})
                ])

        vectorstore.add_documents(hypo_docs)

    def update_txt_file(self):
        """
        更新txt文件夹到对应的4个milvus数据库
        若待上传的文件过多，则需上传较长时间
        """
        # 对txt文件夹中的所有txt文件进行切分并生成对应的doc_id
        docs, doc_ids = self.load_split_doc()
        print(f"doc_ids: {doc_ids}")
        # 同时更新如下集合数据
        # parent_chunk
        print(f"Writing data --> 'parent_chunk' collection")
        self.build_and_write_parent_chunk_colle(docs=docs, doc_ids=doc_ids)
        print(f"Completed writing to --> 'parent_chunk' collection")
        # child_chunk
        print(f"Writing data --> 'child_chunk' collection")
        self.build_and_write_child_chunk_colle(docs=docs, doc_ids=doc_ids)
        print(f"Completed writing to --> 'child_chunk' collection")
        # summary
        print(f"Writing data --> 'summary' collection")
        self.build_and_write_summary_colle(docs=docs, doc_ids=doc_ids)
        print(f"Completed writing to --> 'summary' collection")
        # hypothetical query
        # print(f"Writing data --> 'hypothetical query' collection")
        # self.build_and_write_hypothetical_colle(docs=docs, doc_ids=doc_ids)
        # print(f"Completed writing to --> 'hypothetical query' collection")
        print("UPDATED ALL TXT FILES TO parent_chunk, child_chunk, summary")
        return

def main():
    parser = argparse.ArgumentParser(description="Simple command line parser with one variable.")
    parser.add_argument("--doc_folder", type=str, help="Query string to be processed.")
    args = parser.parse_args()
    doc_folder = args.doc_folder

    # 使用 Ollama 本地嵌入模型
    embeddings = get_embeddings_model()

    # 只连接一次
    milvus_path = os.path.abspath("./milvus.db")
    print("Milvus DB Path:", milvus_path)
    print("Using Ollama local models")

    update_txt = UpdateTxtFile2Milvus(
        document_folder=doc_folder,
        milvus_connection_args=connection_args,
        embedding_model=embeddings)

    update_txt.update_txt_file()

if __name__ == '__main__':
    main()
