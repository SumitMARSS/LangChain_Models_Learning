from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a joke on {topic}."
)

prompt2 = PromptTemplate(
    input_variables=["text"],
    template="Explain the following joke {text}."
)

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)

print(chain.invoke({"topic": "Unemployment in India"}))