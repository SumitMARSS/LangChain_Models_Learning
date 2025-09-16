from langchain.schema.runnable import RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


# basic example of lambda runnables

def word_counter(text):
    return len(text.split())

# runnable_word_counter = RunnableLambda(word_counter)

# print(runnable_word_counter.invoke("Hello world, this is a test"))


# example use case of lambda runnables in parellel runnables
llm1 = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

llm2 = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V3.1",
    task="text-generation",
)


model1 = ChatHuggingFace(llm = llm1)
model2 = ChatHuggingFace(llm = llm2)


prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="Write a joke about {topic}."
)

parser = StrOutputParser()

joke_chain = RunnableSequence(prompt1, model1, parser)

parellel_runnable = RunnableParallel({
    "joke": RunnablePassthrough(),
    "joke_word_count": RunnableLambda(word_counter)
})


# parellel_runnable = RunnableParallel({
#     "joke": RunnablePassthrough(),
#     "joke_word_count": RunnableLambda(lambda text: len(text.split())
# })

final_chain = RunnableSequence(joke_chain, parellel_runnable)
response = final_chain.invoke({"topic": "Unemployment in India"})
print("Final Output", response)
print("\n")
print("Joke",response['joke'])
print("\n")
print("Joke Word Count", response['joke_word_count'])