# 硬件文档智能问答系统

基于 LangChain 和 OpenAI 实现的硬件文档智能问答系统，支持中文交互，可以根据用户提问从 PDF 文档中检索相关信息并生成专业的回答。

## 功能特点

- 支持 PDF 文档的自动加载和处理
- 使用 OpenAI 的 text-embedding-ada-002 模型生成文档向量
- 支持增量添加新文档，避免重复处理
- 使用 ChatGPT (gpt-3.5-turbo) 生成专业的回答
- 提供文档来源追踪
- 完全支持中文

## 系统要求

- Python 3.8+
- OpenAI API 密钥

## 安装步骤

1. 克隆项目并进入目录：
```bash
git clone [your-repo-url]
cd [your-repo-name]
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
   - 创建 `.env` 文件
   - 添加 OpenAI API 密钥：
```bash
OPENAI_API_KEY=your-api-key-here
```

## 使用方法

### 1. 准备文档

将需要处理的 PDF 文档放入 `data` 目录（或其他指定目录）中。

### 2. 创建向量数据库

运行 `create_embeddings.py` 脚本来处理文档并创建向量数据库：

```bash
# 使用默认的 ./data 目录
python create_embeddings.py

# 或指定自定义目录
python create_embeddings.py --datadir /path/to/pdf/files
```

特点：
- 自动检测并处理目录中的所有 PDF 文件
- 支持增量更新，只处理新添加的文档
- 保留现有的向量数据库内容

### 3. 启动问答系统

运行 `qa_interface.py` 开始问答交互：

```bash
python qa_interface.py
```

使用说明：
- 输入问题并按回车
- 输入 'exit' 或 'quit' 退出
- 输入 'help' 获取帮助

### 示例问答

```
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

```
.
├── README.md               # 项目说明文档
├── requirements.txt        # 项目依赖
├── .env                   # 环境变量配置
├── data/                  # PDF文档目录
├── create_embeddings.py   # 文档处理和向量化脚本
├── qa_interface.py        # 问答交互接口
└── chroma_db_hardware/    # 向量数据库存储目录
```

## 注意事项

1. 首次运行需要配置 OpenAI API 密钥
2. PDF 文档需要是可复制的文本格式
3. 向量数据库会占用一定磁盘空间
4. 建议定期备份 `chroma_db_hardware` 目录

## 常见问题

1. **找不到 PDF 文件**
   - 确认文件放在正确的目录中
   - 检查文件权限

2. **API 密钥错误**
   - 检查 `.env` 文件中的密钥是否正确
   - 确认 API 密钥未过期

3. **内存使用过高**
   - 可以调整 `CHUNK_SIZE` 参数
   - 分批处理大型文档
