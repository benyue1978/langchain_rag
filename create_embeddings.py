import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set, Union
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from embeddings import ZhipuAIEmbeddings

# 加载环境变量
load_dotenv()

# 系统配置
DEFAULT_CHROMA_DIR = "chroma_db_openai"  # 默认Chroma数据库目录

# 模型配置
OPENAI_MODEL = "text-embedding-ada-002"  # OpenAI 嵌入模型
ZHIPUAI_MODEL = "embedding-3"  # 智谱AI 嵌入模型
CHUNK_SIZE = 500  # 文档分块大小
CHUNK_OVERLAP = 50  # 分块重叠大小

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md", ".csv", ".xls", ".xlsx", ".ppt", ".pptx"]

def get_embeddings(model_provider: str) -> Embeddings:
    """获取指定提供商的嵌入模型
    
    Args:
        model_provider: 模型提供商，可选值：'openai' 或 'zhipuai'
        
    Returns:
        嵌入模型实例
        
    Raises:
        ValueError: 当提供商名称无效时
    """
    if model_provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            api_key = input("请输入您的OpenAI API密钥: ").strip()
            os.environ["OPENAI_API_KEY"] = api_key
        return OpenAIEmbeddings(
            model=OPENAI_MODEL,
            show_progress_bar=True
        )
    elif model_provider == "zhipuai":
        if not os.getenv("ZHIPUAI_API_KEY"):
            api_key = input("请输入您的智谱AI API密钥: ").strip()
            os.environ["ZHIPUAI_API_KEY"] = api_key
        return ZhipuAIEmbeddings(
            model=ZHIPUAI_MODEL,
            dimensions=2048  # 显式指定2048维度
        )
    else:
        raise ValueError(f"不支持的模型提供商: {model_provider}")

def get_supported_files(data_dir: str) -> List[str]:
    """获取指定目录下的所有支持的文件
    
    Args:
        data_dir: PDF文件所在目录
        
    Returns:
        PDF文件路径列表
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"目录不存在: {data_dir}")
    
    supported_files = []
    for file in data_path.glob("**/*"):
        if file.suffix in SUPPORTED_EXTENSIONS:
            supported_files.append(str(file))
    
    if not supported_files:
        raise ValueError(f"目录中没有找到支持的文件: {data_dir}")
        
    return supported_files

def get_processed_files(chroma_dir: str) -> Set[str]:
    """获取已经处理过的文件列表
    
    Args:
        chroma_dir: Chroma数据库目录
        
    Returns:
        已处理文件路径集合
    """
    if not Path(chroma_dir).exists():
        return set()
        
    db = Chroma(
        persist_directory=chroma_dir
    )
    
    # 从元数据中获取已处理的文件路径
    processed = set()
    for doc in db.get()["metadatas"]:
        if doc and "source" in doc:
            processed.add(doc["source"])
    
    return processed

def load_and_process_documents(file_paths: List[str]) -> List[Any]:
    """加载并处理PDF文档
    
    Args:
        pdf_paths: PDF文件路径列表
        
    Returns:
        加载的文档列表
    """
    all_docs = []
    for path in file_paths:
        try:
            print(f"🔍 加载文档: {path}")
            loader = UnstructuredLoader(path)
            docs = loader.load()
            #print(f"🔍 加载文档: {docs}") # 打印文档内容
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

def create_or_update_vectorstore(data_dir: str, model_provider: str = "openai", chroma_dir: str = DEFAULT_CHROMA_DIR) -> None:
    """创建或更新向量存储
    
    使用指定的嵌入模型生成文档嵌入。
    如果向量存储已存在，只处理新增的文档。
    
    Args:
        data_dir: PDF文件所在目录
        model_provider: 模型提供商，可选值：'openai' 或 'zhipuai'
        chroma_dir: Chroma数据库目录路径
    """
    # 获取所有支持的文件
    supported_files = get_supported_files(data_dir)
    print(f"📁 发现 {len(supported_files)} 个支持的文件")
    
    # 获取已处理的文件
    processed_files = get_processed_files(chroma_dir)
    print(f"💾 已处理 {len(processed_files)} 个文件")
    
    # 找出新增的文件
    new_files = [f for f in supported_files if f not in processed_files]
    if not new_files:
        print("✨ 没有新的文件需要处理")
        return
    
    print(f"🆕 发现 {len(new_files)} 个新文件")
    
    # 初始化嵌入模型
    embeddings = get_embeddings(model_provider)
    
    # 处理新文件
    documents = load_and_process_documents(new_files)
    chunks = split_documents(documents)
    
    # 创建或更新向量存储
    if Path(chroma_dir).exists():
        print("🔄 更新现有向量数据库")
        db = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embeddings
        )
        db.add_documents(chunks)
    else:
        print("🆕 创建新的向量数据库")
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=chroma_dir
        )
    
    print(f"✅ 处理完成，新增 {len(chunks)} 个文档块")

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="创建或更新文档向量数据库")
    parser.add_argument(
        "--datadir",
        type=str,
        default="data",
        help="文件所在目录（默认：./data）"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["openai", "zhipuai"],
        default="openai",
        help="使用的嵌入模型提供商（默认：openai）"
    )
    parser.add_argument(
        "--chromadir",
        type=str,
        default=DEFAULT_CHROMA_DIR,
        help=f"指定Chroma数据库目录路径（默认：{DEFAULT_CHROMA_DIR}）"
    )
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    try:
        create_or_update_vectorstore(args.datadir, args.model, args.chromadir)
    except Exception as e:
        print(f"❌ 系统错误: {str(e)}") 