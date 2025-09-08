from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
print("Key exists:", os.getenv("OPENAI_API_KEY") is not None)

llm = OpenAI(model="gpt-3.5-turbo")


response = llm.invoke("What is LangChain?")
print("Response from llm is ",response)