from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
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


chain1 = RunnableSequence(prompt, model, parser)

def word_counter(str):
    return len(str.split())

chain2 = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(word_counter)
})

chain3 = RunnableSequence(chain1, chain2) # combining the joke generation and explanation into a single sequence
result = chain3.invoke({"topic": "Cricket"})
print(result)

print("Joke:", result["joke"])
print("Word Count:", result["word_count"])