# LangChain RAG QA 系统实现

基于 LangChain 框架实现的检索增强生成（RAG）问答系统，支持与 PDF 文档进行智能对话。

## 已完成任务

- [x] 基础环境配置
  - [x] 环境变量配置（OpenAI API Key, LangSmith API Key）
  - [x] 日志和警告配置
- [x] PDF 文档处理
  - [x] 实现 PDF 加载器
  - [x] 文本分割器配置
- [x] 向量存储实现
  - [x] OpenAI Embeddings 配置
  - [x] Chroma 向量数据库集成
- [x] QA 系统核心功能
  - [x] 会话记忆实现
  - [x] 自定义提示模板
  - [x] ConversationalRetrievalChain 配置
- [x] 交互式问答模式
  - [x] 用户输入处理
  - [x] 答案展示
  - [x] 来源文档显示
- [x] 评估系统
  - [x] 评估数据集创建
  - [x] 运行评估功能

## 进行中任务

- [ ] PDF 文档处理优化
  - [ ] 文档预处理（去除无用内容）
  - [ ] 中文分词优化
  - [ ] PDF 解析质量提升

## 未来任务

- [ ] 系统功能增强
  - [ ] 实现文档动态加载
  - [ ] 添加文档管理功能
  - [ ] 支持文档元数据
- [ ] 用户体验优化
  - [ ] 添加进度条显示
  - [ ] 优化错误提示
  - [ ] 添加结果导出功能
- [ ] 性能优化
  - [ ] 向量存储缓存机制
  - [ ] 批量处理优化
  - [ ] 异步处理支持
- [ ] 测试完善
  - [ ] 单元测试编写
  - [ ] 集成测试实现
  - [ ] 性能测试
- [ ] 部署相关
  - [ ] Docker 容器化
  - [ ] CI/CD 流程配置
  - [ ] 监控告警配置

## 相关文件

- ✅ `rag_qa.py`: 主要实现文件，包含核心 QA 系统逻辑
- ⏳ `requirements.txt`: 项目依赖文件（待创建）
- ⏳ `README.md`: 项目说明文档（待创建）
- ⏳ `tests/`: 测试目录（待创建）
- ⏳ `.env.example`: 环境变量示例文件（待创建）

## 实现细节

### 架构设计
- 使用 LangChain 框架作为基础
- 采用 RAG (Retrieval-Augmented Generation) 架构
- 使用 Chroma 作为向量数据库
- 使用 OpenAI GPT-4 作为基础模型

### 数据流
1. PDF 文档加载 → 文本分割 → 向量化存储
2. 用户提问 → 相关文档检索 → 上下文组装 → 模型回答

### 技术组件
- LangChain
- OpenAI API
- Chroma DB
- PyPDF Loader
- LangSmith（用于评估）

### 环境配置
- Python 3.8+
- OpenAI API Key
- LangSmith API Key
- 必要的 Python 包依赖
