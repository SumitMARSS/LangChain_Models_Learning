from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


loader = TextLoader("cricket.txt", encoding="utf-8")

docs = loader.load()

cricket_content = docs[0].page_content


llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    input_variables=["text"],
    template="You are a helpful assistant. Give a summary within 50 words for given : {text}"
)

chain = prompt | model | parser

result = chain.invoke({"text": cricket_content})


print(docs[0].page_content)
print()


print(result)

# print(docs)
# print()
# print(type(docs))
# print()
# print(docs[0].page_content)
# print()
# print(docs[0].metadata)