from langchain_chroma import Chroma

chroma_dir = "chroma_db_zhipuai"  # 或你的实际数据库目录
vectorstore = Chroma(persist_directory=chroma_dir)

# 获取所有文档及其元数据
all_docs = vectorstore.get()
metadatas = all_docs["metadatas"]
documents = all_docs["documents"]

for meta, doc in zip(metadatas, documents):
    if meta and "source" in meta and meta["source"] == "data/19 检验/3-首件、巡检检验规范/ZKZ.C321首件检验管理办法A1.docx":
        print("---- 文档块 ----")
        print("元数据:", meta)
        print("内容:", doc[:500])  # 只显示前500字符，防止内容过长