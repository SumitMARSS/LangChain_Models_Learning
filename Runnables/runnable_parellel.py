from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
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
    template="Generate a tweet on {topic}."
)

prompt2 = PromptTemplate(
    input_variables=["topic"],
    template="Generate a post on {topic}."
)

chain1 = RunnableSequence(prompt, model, parser)
chain2 = RunnableSequence(prompt2, model, parser)

chain3 = RunnableParallel({
    "tweet": chain1,
    "post": chain2
})
result = chain3.invoke({"topic": "Unemployment in India"})
print(result)
print()
print("Tweet:", result["tweet"])
print()
print("Post:", result["post"])