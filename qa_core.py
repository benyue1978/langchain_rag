from typing import Any, Dict
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

DEFAULT_CHROMA_DIR = "chroma_db_openai"
EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_MODEL = "gpt-3.5-turbo"
DEEPSEEK_MODEL = "deepseek-chat"
TEMPERATURE = 0
TOP_K_RESULTS = 5


def run_langchain_qa(query: str, model_type: str = "openai", chroma_dir: str = DEFAULT_CHROMA_DIR) -> Dict[str, Any]:
    """
    执行一次问答，返回答案和参考来源。

    Args:
        query: 用户问题
        model_type: 使用的模型类型
        chroma_dir: Chroma 数据库目录
    Returns:
        包含 result（答案字符串）和 source_documents（参考文档列表）
    """
    # Embeddings
    if "zhipuai" in chroma_dir.lower():
        from embeddings import ZhipuAIEmbeddings
        embeddings = ZhipuAIEmbeddings(
            model="embedding-3",
            dimensions=2048
        )
    else:
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            show_progress_bar=True
        )
    # 向量存储
    vectorstore = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
    # LLM
    if model_type == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set")
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=TEMPERATURE,
            streaming=False
        )
    elif model_type == "deepseek":
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY not set")
        llm = ChatDeepSeek(
            model=DEEPSEEK_MODEL,
            temperature=TEMPERATURE,
            streaming=False
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    # Prompt
    prompt_template = """你是一个硬件和体系专家，请根据用户的问题，从文档中检索相关信息并生成回答。

问题: {question}

相关文档内容:
{context}

请根据以上信息生成专业、准确的回答，并在回答最后给出相应文档名、章节和页码。如果文档中没有相关信息，用你最好理解回答，请明确说明文档中没有。

回答:"""
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    result = qa_chain.invoke({"query": query})
    return result

def check_token(token: str) -> bool:
    """
    校验 API Token 是否有效。
    Args:
        token: 待校验的 token
    Returns:
        True 表示有效，False 表示无效
    """
    # 这里简单从环境变量读取，实际可扩展为数据库或其它机制
    valid_token = os.getenv("QA_API_TOKEN", "changeme")
    return token == valid_token 