from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from qa_core import run_langchain_qa, check_token
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

class LoginRequest(BaseModel):
    password: str

class AskRequest(BaseModel):
    query: str
    model_type: str = "deepseek"
    chroma_dir: str = "chroma_db_zhipuai"

@app.post("/login")
def login(data: LoginRequest):
    """
    简单登录接口，校验密码，返回 token。
    """
    password = os.getenv("QA_PASSWORD", "changeme")
    if data.password == password:
        return {"token": os.getenv("QA_API_TOKEN", "changeme")}
    raise HTTPException(status_code=401, detail="Invalid password")


def get_token(request: Request) -> str:
    """
    从 Authorization header 获取 token。
    """
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    return auth.split(" ", 1)[1]

@app.post("/ask")
def ask(data: AskRequest, token: str = Depends(get_token)):
    """
    问答接口，需携带 Bearer token。
    """
    if not check_token(token):
        raise HTTPException(status_code=401, detail="Invalid token")
    try:
        result = run_langchain_qa(data.query, data.model_type, data.chroma_dir)
        return {"result": result["result"], "sources": [doc.metadata for doc in result["source_documents"]]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 