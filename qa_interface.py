import os
import argparse
from typing import Dict, Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models.base import BaseChatModel
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 系统配置
CHROMA_DIR = "chroma_db_hardware"

# 模型配置
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI 嵌入模型
OPENAI_MODEL = "gpt-3.5-turbo"  # OpenAI 对话模型
DEEPSEEK_MODEL = "deepseek-chat"  # DeepSeek 对话模型
TEMPERATURE = 0  # 温度参数：0表示最确定性的回答，1表示最具创造性
TOP_K_RESULTS = 5  # 检索时返回的最相关文档数量

def get_llm(model_type: str = "openai") -> BaseChatModel:
    """获取语言模型实例
    
    Args:
        model_type: 模型类型，可选 'openai' 或 'deepseek'
        
    Returns:
        语言模型实例
    """
    if model_type == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            api_key = input("请输入您的OpenAI API密钥: ").strip()
            os.environ["OPENAI_API_KEY"] = api_key
            
        return ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=TEMPERATURE,
            streaming=True,  # 启用流式输出
        )
    elif model_type == "deepseek":
        if not os.getenv("DEEPSEEK_API_KEY"):
            api_key = input("请输入您的DeepSeek API密钥: ").strip()
            os.environ["DEEPSEEK_API_KEY"] = api_key
            
        return ChatDeepSeek(
            model=DEEPSEEK_MODEL,
            temperature=TEMPERATURE,
            streaming=True,  # 启用流式输出
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def init_qa_system(model_type: str = "openai") -> Dict[str, Any]:
    """初始化问答系统
    
    Args:
        model_type: 模型类型，可选 'openai' 或 'deepseek'
    
    Returns:
        包含向量存储和QA链的字典
    """
    # 初始化嵌入模型
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        show_progress_bar=True
    )
    
    # 加载向量存储
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    
    # 初始化检索器
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K_RESULTS}
    )
    
    # 初始化语言模型
    llm = get_llm(model_type)
    
    # 创建提示模板
    prompt_template = """你是一个硬件工程师，请根据用户的问题，从文档中检索相关信息并生成回答。

问题: {question}

相关文档内容:
{context}

请根据以上信息生成专业、准确的回答，并在回答最后给出相应文档名、章节和页码。如果文档中没有相关信息，请明确说明。

回答:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # 创建QA链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return {
        "vectorstore": vectorstore,
        "qa_chain": qa_chain
    }

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="硬件文档问答系统")
    parser.add_argument(
        "--model",
        type=str,
        choices=["openai", "deepseek"],
        default="openai",
        help="选择使用的模型 (默认: openai)"
    )
    return parser.parse_args()

def format_source_info(doc: Any) -> str:
    """格式化文档来源信息
    
    Args:
        doc: 文档对象
        
    Returns:
        格式化后的来源信息字符串
    """
    source = doc.metadata.get('source', '未知来源')
    return source

def run_qa_interface(model_type: str = "openai") -> None:
    """运行问答接口
    
    Args:
        model_type: 模型类型，可选 'openai' 或 'deepseek'
    """
    print(f"\n🤖 硬件文档问答系统 (使用 {model_type} 模型)")
    print("- 输入问题并按回车")
    print("- 输入 'exit' 或 'quit' 退出")
    print("- 输入 'help' 获取帮助")
    
    qa_system = init_qa_system(model_type)
    qa_chain = qa_system["qa_chain"]
    
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
            
            # 获取去重后的参考来源
            if result["source_documents"]:
                sources = {format_source_info(doc) for doc in result["source_documents"]}
                print("\n📄 参考来源:")
                for source in sorted(sources):  # 排序以保持稳定的输出顺序
                    print(f" - {source}")
        except Exception as e:
            print(f"❌ 处理问题时出错: {str(e)}")

if __name__ == "__main__":
    args = parse_args()
    try:
        run_qa_interface(args.model)
    except Exception as e:
        print(f"❌ 系统错误: {str(e)}") 