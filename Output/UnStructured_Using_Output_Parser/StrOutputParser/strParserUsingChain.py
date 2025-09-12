from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm = llm)


template1 = PromptTemplate(
    template="Write a detailed blog post about {topic} ",
    input_variables=["topic"]
)


template2 = PromptTemplate(
    # template="Convert {text} into this {language}.",
    template="Summarise {text} within 3 points  .",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

# chain.invoke({"topic":"Artificial Intelligence", "language":"hindi"})
result = chain.invoke({"topic":"Artificial Intelligence"})

print(result)