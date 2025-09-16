from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence
from dotenv import load_dotenv

load_dotenv()


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
    template="Write a tweet about {topic}."
)

prompt2 = PromptTemplate(
    input_variables=['topic'],
    template="Generate a linkedin blog about \n {topic} within 30 words."
)

parser = StrOutputParser()

parellel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1, model1, parser) ,
    "linkedinBlog": RunnableSequence(prompt2, model2, parser)
})



response = parellel_chain.invoke({"topic": "Unemployment in India"})
print("Final Output", response)
print("\n")
print("Tweet",response['tweet'])
print("\n")
print("LinkedinBlog", response['linkedinBlog'])


print("\n\n")
