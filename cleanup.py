from langchain_chroma import Chroma

chroma_dir = "chroma_db_zhipuai"
vectorstore = Chroma(persist_directory=chroma_dir)

# 获取所有 metadatas
all_metas = vectorstore.get()["metadatas"]

# 找出所有 .doc/.xls 文件
to_delete = set()
for meta in all_metas:
    if meta and "source" in meta and (meta["source"].endswith(".docx") or meta["source"].endswith(".xlsx")):
        to_delete.add(meta["source"])

# 批量删除
if to_delete:
    vectorstore.delete(where={"source": {"$in": list(to_delete)}})
    print(f"已删除 {len(to_delete)} 个 .docx/.xlsx 文件的向量")
else:
    print("没有需要删除的 .docx/.xlsx 文件")