from langchain_community.document_loaders import WebBaseLoader

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


url = "https://impact.indiaai.gov.in/"
loader = WebBaseLoader([url])

docs = loader.load()


# print(len(docs))
# print(docs[0].page_content)

llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    input_variables=["text", "question"],
    template="Give answer for this {question} based on : {text}"
)

chain = prompt | model | parser

ans = chain.invoke({"text": docs[0].page_content, "question": "What is the purpose of this website? Give answer in 50 words."})
print(ans)