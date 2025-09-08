from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)
query_result = embeddings.embed_query("What is the capital of India?")
print(str(query_result))