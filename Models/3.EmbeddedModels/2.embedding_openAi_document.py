from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

document = [
    "What is the capital of India?",
    "What is the capital of USA?",
    "What is the capital of UK?"
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)
query_result = embeddings.embed_documents(document)
print(str(query_result))