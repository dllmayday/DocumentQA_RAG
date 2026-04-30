# DocumentQA_RAG

## Environment Setup
### Install Milvus
Please refer to the Docker method of installing Milvus online.

### Download Repository
```shell
git clone https://github.com/JackDance/DocumentQA_RAG
```
### Install Python Packages
```shell
conda create -n your_conda_envir_name python=3.13 -y
conda activate your_conda_envir_name
```
```shell
pip install  requirements.txt
```



## Modify Configuration
Open `.env`file and modify the following values
```shell
OPENAI_API_KEY1="your openai api key"
EMB_OPENAI_API_BASE="your openai embedding api base"
CHAT_OPENAI_API_BASE="your chat openai api base"
MILVUS_HOST="milvus host ip"
MILVUS_PORT="milvus port, default is 19530"
```

## Knowledge Building
```shell
cd DocumentQA_RAG
python knowledge_building.py --doc_folder "your local document folder path"
```
## Knowledge Retrieval
```shell
cd DocumentQA_RAG
python knowledge_retrieval.py
```
## Sample Result

https://github.com/JackDance/DocumentQA_RAG/assets/46999456/d01d9fa0-8b6d-473d-a23f-f46dcc8a85c9


## 基于本地ollama部署模型搭建知识库

本地ollama部署搭建环境参考[ollama-doc](https://docs.ollama.com/). 本此使用了qwen2.5：0.5b的模型(设备性能太垃圾大的根本跑不起，随着这个模型也挺差的凑活用吧)

### Knowledge Building
```shell
cd DocumentQA_RAG
python knowledge_retrieval_ollama.py --doc_folder "your local document folder path"
```
### Knowledge Retrieval

#### Python Entrance
```shell
cd DocumentQA_RAG
python knowledge_retrieval_ollama.py
```
#### Web Entrance
```
python app.py
```
或
```
uvicorn app:app --host 0.0.0.0 --port 8000
```
启动后可用通过浏览器访问知识库
http://0.0.0.0:8000/