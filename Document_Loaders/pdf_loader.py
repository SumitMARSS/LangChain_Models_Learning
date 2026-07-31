from langchain_community.document_loaders import PyPDFLoader

loadPdf = PyPDFLoader("SearchingFeature.pdf")

loadedDocs = loadPdf.load()

#print(loadedDocs)
print(len(loadedDocs))
print(loadedDocs[0].page_content)
print(loadedDocs[0].metadata)