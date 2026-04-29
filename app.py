#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     app.py
   @Description:   FastAPI Web Interface for RAG Agent
-------------------------------------------------
"""
import os
import sys
from dotenv import load_dotenv
from typing import List, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import os

# 延迟导入 RAG 模块
RAG_pipeline = None
get_embeddings_model = None
connection_args = None

def _import_rag_modules():
    """延迟导入 RAG 模块"""
    global RAG_pipeline, get_embeddings_model, connection_args
    if RAG_pipeline is None:
        try:
            from knowledge_retrieval_ollama import RAG_pipeline as _rag, get_embeddings_model as _emb, connection_args as _conn
            RAG_pipeline = _rag
            get_embeddings_model = _emb
            connection_args = _conn
        except ImportError as e:
            raise ImportError(f"缺少必要的依赖模块: {e}\n请运行: pip install -r requirements.txt") from e

load_dotenv(".env")

# ------------ FastAPI App Setup -----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("🚀 RAG Agent 服务已启动")
    yield
    print("👋 RAG Agent 服务已关闭")

app = FastAPI(
    title="RAG Agent",
    description="基于 Ollama 的本地知识库问答系统",
    version="1.0.0",
    lifespan=lifespan
)

# ------------ Pydantic Models -----------
class ChatRequest(BaseModel):
    query: str
    stream: bool = True

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []

# ------------ API Endpoints -----------
@app.get("/", response_class=HTMLResponse)
async def home():
    """渲染主页"""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "RAG Agent"}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    非流式聊天接口
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")
    
    try:
        _import_rag_modules()
        embeddings = get_embeddings_model()
        rag = RAG_pipeline(
            milvus_connection_args=connection_args,
            embeddings_model=embeddings,
            query=request.query
        )
        
        # 获取回复
        answer = rag.build_prompt_get_answer(
            _get_context_for_query(rag, request.query),
            stream=False
        )
        
        return {
            "answer": str(answer.content if hasattr(answer, 'content') else answer),
            "query": request.query
        }
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/stream")
async def chat_stream(query: str):
    """
    流式聊天接口 (纯文本流)
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")
    
    async def generate():
        try:
            _import_rag_modules()
            embeddings = get_embeddings_model()
            rag = RAG_pipeline(
                milvus_connection_args=connection_args,
                embeddings_model=embeddings,
                query=query
            )
            
            context_docs = _get_context_for_query(rag, query)
            
            # 流式生成响应
            for chunk in rag.build_prompt_get_answer(context_docs, stream=True):
                yield chunk
            
            #yield "[DONE]"
        except ImportError as e:
            yield f"错误: 缺少依赖 - {e}"
        except Exception as e:
            yield f"错误: {str(e)}"
    
    return StreamingResponse(generate(), media_type="text/plain")

def _get_context_for_query(rag: RAG_pipeline, query: str) -> List[str]:
    """
    获取查询上下文
    """
    # 执行检索
    child_list = rag.child_chunk_retriever()
    summary_list = rag.summary_retriever()
    hypo_list = rag.hypothetical_retriever()
    
    # 取交集
    if child_list and summary_list:
        intersection_list = list(set(child_list) & set(summary_list))
        if intersection_list:
            doc_id_list = intersection_list
        else:
            doc_id_list = child_list
    else:
        doc_id_list = child_list if child_list else summary_list
    
    if hypo_list:
        doc_id_list = list(set(doc_id_list) | set(hypo_list))
    
    # 获取父文档
    parent_chunk_list = rag.get_parent_document(doc_id_list)
    return parent_chunk_list

# ------------ Run Server -----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
