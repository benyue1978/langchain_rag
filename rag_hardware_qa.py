import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 系统配置
CHROMA_DIR = "chroma_db_hardware"
PDF_FILES = ["data/r01uh0368ej0220_rl78f13_hardware.pdf", "data/r01us0015ej0230-rl78-software.pdf"]

# 模型配置
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI 嵌入模型
CHAT_MODEL = "gpt-3.5-turbo"  # OpenAI 对话模型
TEMPERATURE = 0  # 温度参数：0表示最确定性的回答，1表示最具创造性
CHUNK_SIZE = 500  # 文档分块大小
CHUNK_OVERLAP = 50  # 分块重叠大小
TOP_K_RESULTS = 5  # 检索时返回的最相关文档数量

def load_and_process_documents(pdf_paths: List[str]) -> List[Any]:
    """加载并处理PDF文档
    
    Args:
        pdf_paths: PDF文件路径列表
        
    Returns:
        加载的文档列表
    """
    all_docs = []
    for path in pdf_paths:
        try:
            loader = UnstructuredPDFLoader(path)
            docs = loader.load()
            print(f"✅ 成功加载文档: {path}")
            all_docs.extend(docs)
        except Exception as e:
            print(f"❌ 加载文档失败 {path}: {str(e)}")
    return all_docs

def split_documents(documents: List[Any]) -> List[Any]:
    """分割文档为较小的块
    
    Args:
        documents: 要分割的文档列表
        
    Returns:
        分割后的文档块列表
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"]  # 优化中文分割
    )
    return splitter.split_documents(documents)

def init_vectorstore() -> Chroma:
    """初始化向量存储
    
    使用 OpenAI 的 text-embedding-ada-002 模型生成文档嵌入。
    首次运行时会创建新的向量存储，后续运行会直接加载已存在的存储。
    
    Returns:
        Chroma向量存储实例
    """
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        show_progress_bar=True
    )
    
    if Path(CHROMA_DIR).exists():
        print(f"🟡 加载已存在的 Chroma DB: {CHROMA_DIR}")
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )

    print(f"🟢 初始化新的 Chroma DB: {CHROMA_DIR}")
    documents = load_and_process_documents(PDF_FILES)
    chunks = split_documents(documents)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"✅ 向量存储初始化完成，共处理 {len(chunks)} 个文档块")
    return db

def run_qa_interface(vectorstore: Chroma) -> None:
    """运行问答接口
    
    使用 ChatGPT (gpt-3.5-turbo) 模型处理用户查询，
    通过向量相似度搜索找到相关文档内容。
    
    Args:
        vectorstore: 向量存储实例
    """
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K_RESULTS}
    )
    
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=TEMPERATURE,
        streaming=True  # 启用流式输出
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    print("\n🤖 硬件文档问答系统")
    print("- 输入问题并按回车")
    print("- 输入 'exit' 或 'quit' 退出")
    print("- 输入 'help' 获取帮助")
    
    while True:
        query = input("\n> ").strip()
        
        if not query:
            continue
            
        if query.lower() in ["exit", "quit"]:
            print("👋 感谢使用！")
            break
            
        if query.lower() == "help":
            print("\n📖 帮助信息:")
            print("- 您可以询问有关硬件的任何问题")
            print("- 系统会从文档中检索相关信息并生成回答")
            print("- 每个回答都会显示信息来源")
            continue
            
        try:
            result = qa_chain.invoke({"query": query})
            print(f"\n🧠 回答:\n{result['result']}")
            print("\n📄 参考来源:")
            for doc in result["source_documents"]:
                print(f" - {doc.metadata.get('source', '未知来源')}")
        except Exception as e:
            print(f"❌ 处理问题时出错: {str(e)}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        api_key = input("请输入您的OpenAI API密钥: ").strip()
        os.environ["OPENAI_API_KEY"] = api_key
        
    try:
        vectorstore = init_vectorstore()
        run_qa_interface(vectorstore)
    except Exception as e:
        print(f"❌ 系统错误: {str(e)}")