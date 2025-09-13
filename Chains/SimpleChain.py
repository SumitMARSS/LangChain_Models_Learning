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


template = PromptTemplate(
    input_variables=["topic"],
    template="Write 5 fact about given {topic}."
)

parser = StrOutputParser()

chain = template | model | parser

# graph chain visualization
chain.get_graph().print_ascii()
response = chain.invoke({"topic": "Cricket"})
print(response)