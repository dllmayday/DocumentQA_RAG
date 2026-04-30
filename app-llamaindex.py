#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     app.py
   @Description:   FastAPI Web Interface for RAG Agent (LlamaIndex 版本)
-------------------------------------------------
"""

import os
from dotenv import load_dotenv
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# =========================
# RAG 引擎导入
# =========================
from knowledge_retrieval_ollama import RAGPipeline, connect_milvus as old_connect_milvus
from knowledge_retrieval_llamaindex import LlamaIndexRAG, connect_milvus as li_connect_milvus

load_dotenv(".env")

# =========================
# 引擎选择配置（默认使用 LlamaIndex）
# 可选值: "llamaindex" | "ollama"
# =========================
RAG_ENGINE = os.environ.get("RAG_ENGINE", "llamaindex")

print(f"[*] 当前 RAG 引擎: {RAG_ENGINE}")


# =========================
# FastAPI 生命周期
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[*] RAG Agent 服务已启动")
    if RAG_ENGINE == "llamaindex":
        li_connect_milvus()
    else:
        old_connect_milvus()
    yield
    print("[*] RAG Agent 服务已关闭")


app = FastAPI(
    title="RAG Agent",
    description="基于 Ollama + Milvus + LlamaIndex 的本地知识库问答系统",
    version="3.0.0",
    lifespan=lifespan
)


# =========================
# Pydantic Models
# =========================
class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    query: str


# =========================
# Web 页面
# =========================
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


# =========================
# 健康检查
# =========================
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "RAG Agent",
        "version": "2.0.0"
    }


# =========================
# 非流式 Chat（核心接口）
# =========================
@app.post("/api/chat")
async def chat(request: ChatRequest, engine: str = None):

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")

    # 选择使用的引擎
    selected_engine = engine or RAG_ENGINE

    try:
        if selected_engine == "llamaindex":
            rag = LlamaIndexRAG()
            answer = rag.query(request.query)
        else:
            rag = RAGPipeline(request.query)
            answer = rag.run()

        return ChatResponse(
            answer=str(answer),
            query=request.query
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# 伪流式接口（逐字符输出）
# =========================
@app.get("/api/chat/stream")
async def chat_stream(query: str, engine: str = None):

    if not query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")

    # 选择使用的引擎
    selected_engine = engine or RAG_ENGINE

    async def generate():
        try:
            if selected_engine == "llamaindex":
                rag = LlamaIndexRAG()
                result = str(rag.query(query))
                for ch in result:
                    yield ch
            else:
                rag = RAGPipeline(query)
                result = rag.run()
                for ch in result:
                    yield ch

        except Exception as e:
            yield f"错误: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")


# =========================
# 启动入口
# =========================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )