import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 系统配置
CHROMA_DIR = "chroma_db_hardware"

# 模型配置
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI 嵌入模型
CHUNK_SIZE = 500  # 文档分块大小
CHUNK_OVERLAP = 50  # 分块重叠大小

def get_pdf_files(data_dir: str) -> List[str]:
    """获取指定目录下的所有PDF文件
    
    Args:
        data_dir: PDF文件所在目录
        
    Returns:
        PDF文件路径列表
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"目录不存在: {data_dir}")
    
    pdf_files = []
    for file in data_path.glob("**/*.pdf"):
        pdf_files.append(str(file))
    
    if not pdf_files:
        raise ValueError(f"目录中没有找到PDF文件: {data_dir}")
        
    return pdf_files

def get_processed_files(chroma_dir: str) -> Set[str]:
    """获取已经处理过的文件列表
    
    Args:
        chroma_dir: Chroma数据库目录
        
    Returns:
        已处理文件路径集合
    """
    if not Path(chroma_dir).exists():
        return set()
        
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        show_progress_bar=True
    )
    
    db = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings
    )
    
    # 从元数据中获取已处理的文件路径
    processed = set()
    for doc in db.get()["metadatas"]:
        if doc and "source" in doc:
            processed.add(doc["source"])
    
    return processed

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

def create_or_update_vectorstore(data_dir: str) -> None:
    """创建或更新向量存储
    
    使用 OpenAI 的 text-embedding-ada-002 模型生成文档嵌入。
    如果向量存储已存在，只处理新增的文档。
    
    Args:
        data_dir: PDF文件所在目录
    """
    # 获取所有PDF文件
    pdf_files = get_pdf_files(data_dir)
    print(f"📁 发现 {len(pdf_files)} 个PDF文件")
    
    # 获取已处理的文件
    processed_files = get_processed_files(CHROMA_DIR)
    print(f"💾 已处理 {len(processed_files)} 个文件")
    
    # 找出新增的文件
    new_files = [f for f in pdf_files if f not in processed_files]
    if not new_files:
        print("✨ 没有新的文件需要处理")
        return
    
    print(f"🆕 发现 {len(new_files)} 个新文件")
    
    # 初始化嵌入模型
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        show_progress_bar=True
    )
    
    # 处理新文件
    documents = load_and_process_documents(new_files)
    chunks = split_documents(documents)
    
    # 创建或更新向量存储
    if Path(CHROMA_DIR).exists():
        print("🔄 更新现有向量数据库")
        db = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        db.add_documents(chunks)
    else:
        print("🆕 创建新的向量数据库")
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
    
    print(f"✅ 处理完成，新增 {len(chunks)} 个文档块")

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="创建或更新文档向量数据库")
    parser.add_argument(
        "--datadir",
        type=str,
        default="data",
        help="PDF文件所在目录（默认：./data）"
    )
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 检查OpenAI API密钥
    if not os.getenv("OPENAI_API_KEY"):
        api_key = input("请输入您的OpenAI API密钥: ").strip()
        os.environ["OPENAI_API_KEY"] = api_key
    
    try:
        create_or_update_vectorstore(args.datadir)
    except Exception as e:
        print(f"❌ 系统错误: {str(e)}") 