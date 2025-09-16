from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# basiic example of runnables

pass_through = RunnablePassthrough()
print(pass_through.invoke("Hello World"))
print(pass_through.invoke(2))

# use case of runnables passthrough in parellel runnables

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

prompt2 = PromptTemplate(
    input_variables=['text'],
    template="Generate a explanation about \n {text} within 30 words."
)

parser = StrOutputParser()

joke_chain = RunnableSequence(prompt1, model1, parser)

parellel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": RunnableSequence(prompt2, model2, parser)
})

final_chain = RunnableSequence(joke_chain, parellel_chain)


response = final_chain.invoke({"topic": "Unemployment in India"})
print("Final Output", response)
print("\n")
print("Joke",response['joke'])
print("\n")
print("Explain", response['explanation'])


print("\n\n")
