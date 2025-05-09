from langchain_unstructured import UnstructuredLoader

# Excel
loader = UnstructuredLoader('data/19 检验/2、4、5、7、8-检验记录/4-QCB0首件检验记录/首件检验记录ZKZQC-B0-24101001.xlsx')
docs = loader.load()
print(docs)
