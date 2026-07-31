from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader("./books", glob="*.pdf", loader_cls=PyPDFLoader)

# load all thing at once

docs = loader.load()

# print(len(docs))
# print(docs[214].page_content)
# print(docs[214].metadata)


for doc in loader.load():
    print(doc.page_content)
    print(doc.metadata)

# load one by one - lazy loading

# print("Lazy loading")

# for doc in loader.lazy_load():
#     print(doc.page_content)
#     print(doc.metadata)