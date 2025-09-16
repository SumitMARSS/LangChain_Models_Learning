from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm = llm)


prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="Write a joke on given {topic}."
)

prompt2 = PromptTemplate(
    input_variables=['text'],
    template="Give 2 line explanation on given text \n {text}."
)

parser = StrOutputParser()

chain1 = RunnableSequence(prompt1, model, parser);
response = chain1.invoke({"topic": "Unemployment in India"})
print(response)

chain2 = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

response = chain2.invoke({"topic": "Unemployment in India"})
print(response)


print("\n\n")
