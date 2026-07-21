from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
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

chain1 = RunnableSequence(prompt, model, parser)

chain2 = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": RunnableSequence(prompt2, model, parser)
})

chain3 = RunnableSequence(chain1, chain2) # combining the joke generation and explanation into a single sequence
result = chain3.invoke({"topic": "Unemployment in India"})
print(result)

print("Joke:", result["joke"])
print("Explanation:", result["explanation"])