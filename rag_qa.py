"""基于RAG的问答系统模块

此模块实现了一个基于检索增强生成(RAG)的问答系统，专注于氮氧化物传感器相关问题。
系统使用PDF文档作为知识库，通过向量存储和语言模型提供准确的答案。
"""

import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
from uuid import uuid4
from dotenv import load_dotenv
import time
from datetime import datetime

# 加载环境变量
load_dotenv()

from chinese_text_splitter import ChineseTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langsmith import Client

from pdf_processor import PDFProcessor, PDFProcessorError
from langsmith.evaluation import RunEvaluator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QASystemConfig:
    """问答系统配置

    Attributes:
        pdf_path: PDF 文档路径
        chunk_size: 文本分块大小
        chunk_overlap: 文本分块重叠大小
        temperature: 语言模型温度参数
        model_name: OpenAI 模型名称
    """
    pdf_path: str | Path
    chunk_size: int = 500
    chunk_overlap: int = 50
    temperature: float = 0.0
    model_name: str = "gpt-3.5-turbo"

class RAGQASystem:
    """基于 RAG 的问答系统
    
    此类实现了一个专门用于处理氮氧传感器相关问题的问答系统。
    系统使用 PDF 文档作为知识库，通过向量检索和语言模型生成答案。
    
    Attributes:
        config: 系统配置
        qa_chain: 问答链
        memory: 对话记忆
    """
    
    def __init__(self, config: QASystemConfig):
        """初始化问答系统
        
        Args:
            config: 系统配置对象
            
        Raises:
            ValueError: 当配置参数无效时
            PDFProcessorError: 当 PDF 处理失败时
        """
        self.config = config
        self.qa_chain = None
        self.memory = None
        
        # 验证配置
        self._validate_config()
        
        # 初始化系统
        self._init_system()
    
    def _validate_config(self) -> None:
        """验证配置参数
        
        Raises:
            ValueError: 当配置参数无效时
        """
        if self.config.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if self.config.chunk_overlap < 0:
            raise ValueError("chunk_overlap 必须大于等于 0")
        if self.config.temperature < 0 or self.config.temperature > 1:
            raise ValueError("temperature 必须在 0 到 1 之间")
    
    def _init_system(self) -> None:
        """初始化问答系统
        
        此方法执行以下步骤：
        1. 加载和处理 PDF 文档
        2. 分割文本
        3. 创建向量存储
        4. 设置问答链
        
        Raises:
            PDFProcessorError: 当 PDF 处理失败时
        """
        try:
            # 加载 PDF
            pdf_processor = PDFProcessor()
            pages = pdf_processor.load_pdf(self.config.pdf_path)
            
            # 合并所有页面内容
            text = "\n\n".join(page.content for page in pages)
            
            # 分割文本
            text_splitter = ChineseTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            texts = text_splitter.split_text(text)
            
            # 创建向量存储
            embeddings = OpenAIEmbeddings()
            persist_dir = Path("chroma_db")
            
            # 确保持久化目录存在
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # 初始化或加载向量存储
            if (persist_dir / "chroma.sqlite3").exists():
                logger.info("从持久化目录加载现有向量存储")
                vectorstore = Chroma(
                    persist_directory=str(persist_dir),
                    embedding_function=embeddings,
                    collection_name="nox_sensor_docs"
                )
            else:
                logger.info("创建新的向量存储")
                vectorstore = Chroma.from_texts(
                    texts=texts,
                    embedding=embeddings,
                    persist_directory=str(persist_dir),
                    collection_name="nox_sensor_docs"
                )
                vectorstore.persist()
                logger.info("向量存储已持久化到磁盘")
            
            # 设置提示模板
            template = """你是一个专门研究氮氧传感器的专家。请基于以下背景信息，
            专业且准确地回答问题。如果无法从背景信息中找到答案，请明确说明。

            背景信息：
            {context}

            问题：{question}

            请用中文回答。
            """
            
            QA_PROMPT = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # 创建基础问答链
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(
                    temperature=self.config.temperature,
                    model_name=self.config.model_name
                ),
                retriever=vectorstore.as_retriever(),
                combine_docs_chain_kwargs={"prompt": QA_PROMPT}
            )
            
            logger.info("问答系统初始化完成")
            
        except PDFProcessorError as e:
            logger.error(f"PDF 处理失败: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"系统初始化失败: {str(e)}")
            raise
    
    def ask(self, question: str) -> str:
        """向系统提问
        
        Args:
            question: 问题文本
            
        Returns:
            系统的回答
            
        Raises:
            ValueError: 当问答链未初始化时
            Exception: 当问答过程发生错误时
        """
        if not self.qa_chain:
            raise ValueError("问答系统未正确初始化")
        
        try:
            # 使用新版本的LangChain调用方式
            result = self.qa_chain.invoke({
                "question": question,
                "chat_history": []
            })
            return result["answer"]
        except Exception as e:
            logger.error(f"问答失败: {str(e)}")
            raise

def create_qa_system(pdf_path: str | Path) -> RAGQASystem:
    """创建问答系统的便捷函数
    
    Args:
        pdf_path: PDF 文档路径
        
    Returns:
        初始化好的问答系统
    """
    config = QASystemConfig(pdf_path=pdf_path)
    return RAGQASystem(config)

def interactive_mode(qa_system: RAGQASystem) -> None:
    """交互式问答模式
    
    Args:
        qa_system: 初始化好的问答系统
    """
    print("欢迎使用氮氧传感器问答系统！")
    print("输入 'q' 或 'quit' 退出")
    
    while True:
        question = input("\n请输入您的问题: ").strip()
        if question.lower() in ['q', 'quit']:
            break
            
        try:
            answer = qa_system.ask(question)
            print(f"\n回答: {answer}")
        except Exception as e:
            print(f"\n错误: {str(e)}")

def clean_langsmith():
    client = Client()
    projects = client.list_projects()
    for project in projects:
        client.delete_project(project_name=project.name)
        time.sleep(1)
    datasets = client.list_datasets()
    for dataset in datasets:
        client.delete_dataset(dataset_name=dataset.name)
        time.sleep(1)
    print("LangSmith 数据已清理")

def create_evaluation_dataset(dataset_name: str) -> str:
    """创建评估数据集
    
    Returns:
        数据集名称
        
    Raises:
        Exception: 当创建数据集失败时
    """
    try:
        client = Client()
        if client.has_dataset(dataset_name=dataset_name):
            return dataset_name
        
        dataset = client.create_dataset(dataset_name)
        
        # 创建评估样例
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
        logger.info(f"成功创建评估数据集: {dataset_name}")
        return dataset_name
        
    except Exception as e:
        logger.error(f"创建评估数据集失败: {str(e)}")
        raise

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """评估答案的正确性
    
    使用 GPT-4 作为评判，比较生成的答案与参考答案的正确性。
    """
    from langchain_openai import ChatOpenAI
    
    eval_instructions = "你是一位专门评估氮氧传感器相关答案的专家教授。"
    user_content = f"""你正在评估以下问题的答案：
    问题：{inputs['question']}
    
    标准答案：
    {reference_outputs['answer']}
    
    待评估答案：
    {outputs['answer']}
    
    请仅回复 CORRECT 或 INCORRECT：
    评分：
    """
    
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    response = llm.invoke(user_content).content.strip()
    
    return {
        "score": 1.0 if response == "CORRECT" else 0.0,
        "reasoning": f"GPT-4 评估结果: {response}"
    }

def conciseness(outputs: dict, reference_outputs: dict) -> dict:
    """评估答案的简洁性
    
    如果生成的答案长度小于参考答案长度的2倍，则认为是简洁的。
    """
    answer_length = len(outputs["answer"])
    ref_length = len(reference_outputs["answer"])
    score = 1.0 if answer_length < 2 * ref_length else ref_length / answer_length
    
    return {
        "score": score,
        "reasoning": f"答案长度: {answer_length}字，参考答案长度: {ref_length}字。得分说明：答案长度小于参考答案2倍为满分。"
    }

def run_custom_eval(qa_system: RAGQASystem, dataset_name: str, project_name: str) -> None:
    """运行自定义评估
    
    Args:
        qa_system: QA系统实例
        dataset_name: 数据集名称
        project_name: 项目名称
    """
    client = Client()
    
    def ls_target(inputs: dict) -> dict:
        """目标函数，用于生成答案"""
        return {"answer": qa_system.ask(inputs["question"])}
    
    # 运行评估
    experiment_results = client.evaluate(
        ls_target,  # 评估目标
        data=dataset_name,  # 评估数据集
        evaluators=[correctness, conciseness],  # 评估器
        experiment_prefix=project_name  # 实验名称前缀
    )
    
    print(f"\n评估完成：结果已写入 Project = {project_name}")
    return experiment_results

if __name__ == "__main__":
    # 检查是否为评估模式
    is_eval_mode = len(sys.argv) > 1 and sys.argv[1] == "--evaluate"

    is_clean_mode = len(sys.argv) > 1 and sys.argv[1] == "--clean"

    print("Tracing enabled:", os.getenv("LANGCHAIN_TRACING_V2"))
    
    # 设置 OpenAI API 密钥
    if not os.getenv("OPENAI_API_KEY"):
        api_key = input("请输入您的 OpenAI API 密钥: ").strip()
        os.environ["OPENAI_API_KEY"] = api_key
    
    # 检查 LangSmith API 密钥（评估模式需要）
    if is_eval_mode and not os.getenv("LANGSMITH_API_KEY"):
        api_key = input("请输入您的 LangSmith API 密钥: ").strip()
        os.environ["LANGSMITH_API_KEY"] = api_key
    
    try:
        if is_clean_mode:
            clean_langsmith()
            exit()

        # 创建问答系统
        qa_system = create_qa_system("氮氧传感器性能及其控制策略研究-from SY.pdf")
        
        # 根据模式选择运行方式
        if is_eval_mode:
            create_evaluation_dataset("nox_sensor_qa_eval_v1")
            project_name = f"nox_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_custom_eval(qa_system, "nox_sensor_qa_eval_v1", project_name)        
        else:
            interactive_mode(qa_system)
        
    except Exception as e:
        print(f"系统错误: {str(e)}")