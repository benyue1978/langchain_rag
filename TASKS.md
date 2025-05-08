# LangChain RAG QA 系统实现

基于 LangChain 框架实现的检索增强生成（RAG）问答系统，支持与 PDF 文档进行智能对话。

## 已完成任务

- [x] 基础环境配置
  - [x] 环境变量配置（OpenAI API Key, LangSmith API Key, ZhipuAI API Key）
  - [x] 日志和警告配置
  - [x] 日志级别优化（设置为 ERROR）
- [x] PDF 文档处理
  - [x] 实现 PDF 加载器
  - [x] 文本分割器配置
  - [x] 文档预处理（去除无用内容）
    - [x] 移除页眉页脚
    - [x] 清理多余空白
    - [x] 修复错误的换行
    - [x] 提取文档结构
  - [x] 中文分词优化
    - [x] 实现基于 jieba 的中文分词器
    - [x] 支持句子级别分割
    - [x] 智能处理长句子
    - [x] 保持语义完整性
  - [x] PDF 解析质量提升
    - [x] 智能识别和处理章节结构
    - [x] 清理图表标题和参考文献
    - [x] 规范化特殊字符和标点
- [x] 向量存储实现
  - [x] 多种 Embeddings 支持
    - [x] OpenAI Embeddings 配置
    - [x] 智谱 AI Embeddings 实现（2048维）
    - [x] 自动模型选择机制
  - [x] Chroma 向量数据库集成
  - [x] 持久化存储配置
- [x] QA 系统核心功能
  - [x] 会话记忆实现
  - [x] 自定义提示模板
  - [x] ConversationalRetrievalChain 配置
  - [x] 错误处理和日志记录
- [x] 交互式问答模式
  - [x] 用户输入处理
  - [x] 答案展示
  - [x] 错误提示优化
- [x] 评估系统
  - [x] 评估数据集创建
  - [x] 运行评估功能
  - [x] LangSmith 集成
- [x] 代码质量改进
  - [x] 类型注解完善
  - [x] 文档字符串补充
  - [x] 代码结构优化
  - [x] 错误处理增强
- [x] 提取 run_langchain_qa 到单独文件，命令行接口调用新方法
- [x] 用 FastAPI 实现 /ask /login 接口，支持无 UI 的 token 验证

## 进行中任务

- [ ] 前端静态页面
  - [ ] 实现最简 HTML 前端，放入 FastAPI static
  - [ ] 前后端联调
- [ ] Docker 化
  - [ ] 编写 Dockerfile 打包后端与前端
  - [ ] 支持一键部署
- [ ] 系统功能增强
  - [ ] 实现文档动态加载
  - [ ] 添加文档管理功能
  - [ ] 支持文档元数据
  - [ ] 优化检索策略
  - [ ] 支持更多嵌入模型
- [ ] 性能优化
  - [ ] 向量存储缓存机制
  - [ ] 批量处理优化
  - [ ] 异步处理支持
  - [ ] 大维度向量优化


## 未来任务

- [ ] 用户体验优化
  - [ ] 添加进度条显示
  - [ ] 添加结果导出功能
  - [ ] 支持多种输出格式
  - [ ] 模型切换界面优化
- [ ] 测试完善
  - [ ] 单元测试编写
  - [ ] 集成测试实现
  - [ ] 性能测试
  - [ ] 多模型兼容性测试
- [ ] 部署相关
  - [ ] Docker 容器化
  - [ ] CI/CD 流程配置
  - [ ] 监控告警配置
- [ ] 功能扩展
  - [ ] 支持更多文档格式
  - [ ] 多语言支持
  - [ ] API 接口开发
  - [ ] 更多 AI 模型集成

## 相关文件

- ✅ `rag_qa.py`: 主要实现文件，包含核心 QA 系统逻辑
- ✅ `pdf_processor.py`: PDF 处理器，提供专门的 PDF 文档处理功能
- ✅ `chinese_text_splitter.py`: 中文文本分割器，基于 jieba 分词实现
- ✅ `embeddings.py`: 嵌入模型实现，支持多种模型
- ✅ `qa_core.py`: 核心问答逻辑，供 CLI 和 API 共用
- ✅ `fastapi_app.py`: FastAPI Web API 实现，含 /ask /login
- ⏳ `requirements.txt`: 项目依赖文件（待更新）
- ⏳ `README.md`: 项目说明文档（待更新）
- ⏳ `tests/`: 测试目录（待创建）
- ⏳ `.env.example`: 环境变量示例文件（待创建）

## 实现细节

### 架构设计
- 使用 LangChain 框架作为基础
- 采用 RAG (Retrieval-Augmented Generation) 架构
- 使用 Chroma 作为向量数据库
- 支持多种 AI 模型和 embeddings

### 数据流
1. PDF 文档加载 → 文本分割 → 向量化存储
2. 用户提问 → 相关文档检索 → 上下文组装 → 模型回答
3. 评估反馈 → 性能优化 → 系统改进

### 技术组件
- LangChain
- OpenAI API
- 智谱 AI API
- DeepSeek API
- Chroma DB
- PyMuPDF
- jieba 分词
- LangSmith（用于评估）

### 环境配置
- Python 3.8+
- OpenAI API Key（可选）
- 智谱 AI API Key（可选）
- DeepSeek API Key（可选）
- LangSmith API Key
- 必要的 Python 包依赖
