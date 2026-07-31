from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path="data.csv")


docs = loader.load()
print(len(docs))
print(docs[1].page_content)