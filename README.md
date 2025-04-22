# Langchain Quickstart

```shell
python -m venv venv
source venv/bin/activate 
pip install langchain openai chromadb tiktoken unstructured dotenv langchain-community

cat > .env << EOF
OPENAI_API_KEY=sk-proj-xxxx
EOF

pip install -r requirements.txt
```

## 运行文档帮助

```shell
python rag_hardware_qa.py
```
