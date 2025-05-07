import os
import argparse
import signal
import sys
import logging
from typing import Dict, Any
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models.base import BaseChatModel
from dotenv import load_dotenv
from qa_core import run_langchain_qa

# 配置日志级别为 ERROR
logging.basicConfig(level=logging.ERROR)

# 加载环境变量
load_dotenv()

# 系统配置
DEFAULT_CHROMA_DIR = "chroma_db_openai"  # 默认Chroma数据库目录
HISTORY_FILE = os.path.expanduser("~/.qa_history")  # 保存在用户主目录下

# 确保历史文件目录存在
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)


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
    parser.add_argument(
        "--chromadir",
        type=str,
        default=DEFAULT_CHROMA_DIR,
        help=f"指定Chroma数据库目录路径 (默认: {DEFAULT_CHROMA_DIR})"
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

def signal_handler(signum, frame):
    """处理中断信号"""
    print("\n\n👋 感谢使用！")
    sys.exit(0)

def create_keybindings() -> KeyBindings:
    """创建自定义键绑定
    
    Returns:
        KeyBindings 实例
    """
    kb = KeyBindings()
    
    @kb.add('enter')
    def _(event):
        """处理回车键：直接提交查询"""
        buff = event.current_buffer
        if buff.text.strip():  # 如果有内容才提交
            buff.validate_and_handle()
    
    @kb.add('escape', 'enter')  # Alt + Enter
    def _(event):
        """处理 Alt+Enter：插入换行"""
        event.current_buffer.insert_text('\n')
    
    return kb

def run_qa_interface(model_type: str = "openai", chroma_dir: str = DEFAULT_CHROMA_DIR) -> None:
    """运行问答接口
    
    Args:
        model_type: 模型类型，可选 'openai' 或 'deepseek'
        chroma_dir: Chroma数据库目录路径
    """
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    # 定义提示样式
    style = Style.from_dict({
        'prompt': '#00aa00 bold',
        'continuation': '#666666',
    })
    
    # 初始化提示会话
    session = PromptSession(
        history=FileHistory(HISTORY_FILE),
        auto_suggest=AutoSuggestFromHistory(),
        enable_history_search=True,
        key_bindings=create_keybindings(),
        style=style,
        multiline=True,  # 启用多行模式
        prompt_continuation=lambda width, line_number, is_soft_wrap: '... > ',
    )
    
    print(f"\n🤖 硬件文档问答系统 (使用 {model_type} 模型)")
    print(f"📁 使用数据库目录: {chroma_dir}")
    print("- 按回车提交问题")
    print("- 按 Alt+回车 换行继续输入")
    print("- 输入 'exit' 或 'quit' 退出")
    print("- 输入 'help' 获取帮助")
    print("- 按 Ctrl+C 随时退出")
    print("- 使用上下箭头键浏览历史记录")
    print("- 使用 Ctrl+R 搜索历史记录")
    
    while True:
        try:
            # 使用 prompt_toolkit 获取多行输入
            query = session.prompt(
                ">>> ",
                style=style
                ).strip()
            
            if not query:
                continue
                
            if query.lower() in ["exit", "quit"]:
                print("👋 感谢使用！")
                break
                
            if query.lower() == "help":
                print("\n📖 帮助信息:")
                print("- 您可以询问有关硬件的任何问题")
                print("- 按回车提交问题")
                print("- 按 Alt+回车 换行继续输入")
                print("- 系统会从文档中检索相关信息并生成回答")
                print("- 每个回答都会显示信息来源")
                print("- 按 Ctrl+C 随时退出")
                print("- 使用上下箭头键浏览历史记录")
                print("- 使用 Ctrl+R 搜索历史记录")
                continue
                
            try:
                result = run_langchain_qa(query, model_type, chroma_dir)
                print(f"\n🧠 回答:\n{result['result']}")
                
                # 获取去重后的参考来源
                if result["source_documents"]:
                    sources = {format_source_info(doc) for doc in result["source_documents"]}
                    print("\n📄 参考来源:")
                    for source in sorted(sources):  # 排序以保持稳定的输出顺序
                        print(f" - {source}")
            except Exception as e:
                print(f"❌ 处理问题时出错: {str(e)}")
        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None)
        except EOFError:  # 处理 Ctrl+D
            print("\n👋 感谢使用！")
            break

if __name__ == "__main__":
    args = parse_args()
    try:
        run_qa_interface(args.model, args.chromadir)
    except Exception as e:
        print(f"❌ 系统错误: {str(e)}") 