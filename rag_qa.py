import os
import warnings
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langsmith import Client
import datetime
import sys
from langchain.schema.runnable import RunnableLambda

# 加载环境变量
load_dotenv()

# 确保设置了必要的环境变量
assert os.getenv("OPENAI_API_KEY"), "请设置 OPENAI_API_KEY 环境变量"
assert os.getenv("LANGSMITH_API_KEY"), "请设置 LANGSMITH_API_KEY 环境变量"

# 初始化 LangSmith 客户端
client = Client()

# 配置日志级别
logging.getLogger("pypdf").setLevel(logging.ERROR)

# 过滤警告信息
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def init_qa_system(silent=False):
    """初始化QA系统
    
    Args:
        silent (bool): 是否静默初始化（不显示提示信息）
        
    Returns:
        qa_chain: 初始化好的QA链
    """
    if not silent:
        print("🔄 正在初始化 QA 系统...")
        
    # 初始化文档加载器和分割器
    loader = PyPDFLoader("氮氧传感器性能及其控制策略研究-from SY.pdf")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    # 加载和分割文档
    pages = loader.load()
    documents = text_splitter.split_documents(pages)
    
    # 创建向量存储
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
    
    # 创建QA链
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    template = """你是一个专业的氮氧传感器专家。使用以下上下文来回答问题。如果你不知道答案，就说"抱歉，我无法回答这个问题"。
    不要试图编造答案。尽量使用中文回答。

    上下文: {context}
    聊天历史: {chat_history}
    人类: {question}

    助手:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template,
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,
        return_generated_question=False,
        output_key="answer",
        verbose=False
    )
    
    if not silent:
        print("\n✨ QA 系统初始化完成！")
        
    return qa_chain

def create_evaluation_dataset():
    """创建评估数据集"""
    dataset_name = f"nox_sensor_qa_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset = client.create_dataset(dataset_name)
    
    examples = [
        {
            "inputs": {"question": "什么是氮氧传感器？"},
            "outputs": {"answer": "氮氧传感器是一种用于检测和测量氮氧化物浓度的传感器装置。"}
        },
        {
            "inputs": {"question": "氮氧传感器的主要应用场景是什么？"},
            "outputs": {"answer": "主要应用于汽车尾气排放控制系统，用于监测和控制氮氧化物的排放。"}
        },
        {
            "inputs": {"question": "氮氧传感器的工作原理是什么？"},
            "outputs": {"answer": "基于电化学原理，通过测量氧离子的浓度差来检测氮氧化物含量。"}
        },
        {
            "inputs": {"question": "氮氧传感器的性能指标有哪些？"},
            "outputs": {"answer": "主要包括响应时间、测量精度、温度特性和使用寿命等指标。"}
        },
        {
            "inputs": {"question": "如何提高氮氧传感器的控制精度？"},
            "outputs": {"answer": "通过优化传感器结构、改进控制算法和提高信号处理能力等方式。"}
        }
    ]
    
    client.create_examples(dataset_id=dataset.id, examples=examples)
    return dataset_name

def run_evaluation(qa_chain):
    """运行评估测试集"""
    dataset_name = create_evaluation_dataset()
    
    def construct_chain():
        input_mapper = RunnableLambda(
            lambda x: {"question": x["question"], "chat_history": []}
        )
        return input_mapper | qa_chain
    
    print("评估中...")
    client.run_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=construct_chain,
        project_metadata={"tags": ["nox_sensor_qa_eval"]},
        verbose=False
    )
    print("完成")

def interactive_mode(qa_chain):
    """交互式问答模式"""
    print("\n💬 开始交互式问答...\n")
    
    while True:
        question = input("❓ 请输入您的问题（输入 'quit' 或 'exit' 退出）: ").strip()
        
        if question.lower() in ['quit', 'exit']:
            print("\n👋 感谢使用！再见！")
            break
            
        if not question:
            print("⚠️ 请输入有效的问题！")
            continue
            
        try:
            result = qa_chain.invoke({"question": question})
            answer = result['answer']
            source_docs = result['source_documents']
            
            print("\n💡 答案:", answer)
            
            if source_docs:
                print("\n📚 参考来源:")
                for i, doc in enumerate(source_docs[:2], 1):
                    print(f"{i}. {doc.page_content[:150]}...")
            
            print("\n" + "-"*50 + "\n")
            
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")
            print("请稍后重试或联系系统管理员。\n")

def main():
    # 根据命令行参数选择模式
    is_eval_mode = len(sys.argv) > 1 and sys.argv[1] == "--evaluate"
    
    # 初始化QA系统
    qa_chain = init_qa_system(silent=is_eval_mode)
    
    # 运行相应的模式
    if is_eval_mode:
        run_evaluation(qa_chain)
    else:
        interactive_mode(qa_chain)

if __name__ == "__main__":
    main() 