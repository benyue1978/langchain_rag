# 文档智能问答系统

基于 LangChain 和 OpenAI/智谱 AI 实现的文档智能问答系统，支持中文交互，可以根据用户提问从 PDF 文档中检索相关信息并生成专业的回答。

## 功能特点

- 支持多种类型（pdf, doc, docx, txt, md, csv, xls, xlsx, ppt, pptx, epub）文档的自动加载和处理
- 支持多种嵌入模型：
  - OpenAI text-embedding-ada-002
  - 智谱 AI embedding-3（2048维）
- 支持增量添加新文档，避免重复处理
- 支持多种对话模型：
  - ChatGPT (gpt-3.5-turbo)
  - DeepSeek Chat
- 提供文档来源追踪
- 完全支持中文

## 系统要求

- Python 3.8+
- OpenAI API 密钥（使用 OpenAI 模型时）
- 智谱 AI API 密钥（使用智谱 AI 模型时）
- DeepSeek API 密钥（使用 DeepSeek 模型时）

## 安装步骤

1. 克隆项目并进入目录：

```bash
git clone git@codeup.aliyun.com:zkz/tools/zkz-ai-rag.git
cd zkz-ai-rag
```

1. 安装依赖：

```bash
python -m venv venv
source venv/bin/activate # MacOS
venv/Scripts/activate # Windows
# Windows还需要先安装VS C++
# Windows上为了处理word等文件，还需要安装pandoc https://github.com/jgm/pandoc/releases
# 为了处理doc文件，需要安装libreoffice：
# MacOS: brew install --cask libreoffice
# Windows: https://www.libreoffice.org/download/download-libreoffice/ 安装之后把C:\Program Files\LibreOffice\program加入Path环境变量
# 然后把doc批量转换为docx - doc-conver.ps1
pip install -r requirements.txt
```

1. 配置环境变量：
   - 参考`.env.sample`，创建 `.env` 文件
   - 根据需要添加相应的 API 密钥：

```bash
# OpenAI（可选）
OPENAI_API_KEY=your-openai-api-key-here

# 智谱 AI（可选）
ZHIPUAI_API_KEY=your-zhipuai-api-key-here

# DeepSeek（可选）
DEEPSEEK_API_KEY=your-deepseek-api-key-here
```

## 使用方法

### 1. 准备文档

将需要处理的 PDF 文档放入 `data` 目录（或其他指定目录）中，支持子目录。

### 2. 创建向量数据库

运行 `create_embeddings.py` 脚本来处理文档并创建向量数据库：

```bash
# 使用默认的 OpenAI embeddings
python create_embeddings.py

# 使用智谱 AI embeddings
python create_embeddings.py --chromadir chroma_db_zhipuai

# 或指定自定义目录
python create_embeddings.py --datadir /path/to/pdf/files
```

特点：

- 自动检测并处理目录中的所有支持的文件
- 支持增量更新，只处理新添加的文档
- 保留现有的向量数据库内容
- 根据数据库目录名自动选择嵌入模型

### 3. 启动问答系统

运行 `qa_interface.py` 开始问答交互：

```bash
# 使用默认的 OpenAI 模型
python qa_interface.py

# 使用 DeepSeek 模型
python qa_interface.py --model deepseek

# 使用智谱 AI embeddings（通过指定数据库目录）
python qa_interface.py --chromadir chroma_db_zhipuai
```

使用说明：

- 输入问题并按回车
- 输入 'exit' 或 'quit' 退出
- 输入 'help' 获取帮助

支持的模型：

- OpenAI (默认): 使用 gpt-3.5-turbo 模型
- DeepSeek: 使用 deepseek-chat 模型
- 智谱 AI: 使用 embedding-3 模型进行文本嵌入

注意：使用不同的模型需要对应的 API 密钥：

- OpenAI 模型需要设置 `OPENAI_API_KEY` 环境变量
- DeepSeek 模型需要设置 `DEEPSEEK_API_KEY` 环境变量
- 智谱 AI 需要设置 `ZHIPUAI_API_KEY` 环境变量

### 示例问答

```txt
🤖 硬件文档问答系统
- 输入问题并按回车
- 输入 'exit' 或 'quit' 退出
- 输入 'help' 获取帮助

> RL78/F13 的 LED 驱动功能是什么？

🧠 回答：
RL78/F13的LED驱动功能包括8位D/A转换器和内置比较器。该芯片支持2.7至5.5V的电源电压，并在-40至+105°C的环境温度下正常工作。此外，RL78/F13还具有LIN集成功能，适用于一般汽车电气应用（如电机控制、车门控制、车灯控制等）和摩托车发动机控制。

文档名: RL78/F13, F14
章节: CHAPTER 1 OVERVIEW
页码: 8, 9

📄 参考来源：
 - data/r01uh0368ej0220_rl78f13_hardware.pdf
```

## 项目结构

```dir
.
├── README.md               # 项目说明文档
├── requirements.txt        # 项目依赖
├── .env                   # 环境变量配置
├── data/                  # PDF文档目录
├── create_embeddings.py   # 文档处理和向量化脚本
├── qa_interface.py        # 问答交互接口
├── embeddings.py          # 嵌入模型实现
└── chroma_db/            # 向量数据库存储目录
```

## 注意事项

1. 首次运行需要配置相应的 API 密钥
2. PDF 文档需要是可复制的文本格式
3. 向量数据库会占用一定磁盘空间
4. 建议定期备份向量数据库目录
5. 智谱 AI embeddings 使用2048维向量，可能需要更多存储空间

## 常见问题

1. **找不到文件**
   - 确认文件放在正确的目录中
   - 检查文件权限

2. **API 密钥错误**
   - 检查 `.env` 文件中的密钥是否正确
   - 确认 API 密钥未过期
   - 确保使用了正确的环境变量名

3. **内存使用过高**
   - 可以调整 `CHUNK_SIZE` 参数
   - 分批处理大型文档
   - 注意智谱 AI embeddings 的向量维度较大

## 国内使用推荐智谱AI+Deepseek

```shell
# 做文档索引
python create_embeddings.py --model zhipuai --chromadir chroma_db_zhipuai
# 做查询
python qa_interface.py --chromadir chroma_db_zhipuai --model deepseek
```
