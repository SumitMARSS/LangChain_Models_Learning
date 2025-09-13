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
    input_variables=["topic"],
    template="Write a detailed report on given {topic}."
)

template2 = PromptTemplate(
    input_variables=['report'],
    template="Give 2 most important point from given report \n {report}."
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser


response = chain.invoke({"topic": "Unemployment in India"})
print(response)


print("\n\n")

# graph chain visualization
chain.get_graph().print_ascii()

