from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm = llm)


template1 = PromptTemplate(
    template="Write a detailed blog post about {topic} within 20 words.",
    input_variables=["topic"]
)

prompt1 = template1.format(topic="Artificial Intelligence")

template2 = PromptTemplate(
    template="Convert {text} into this {language}.",
    input_variables=["language"]
)

result1 = model.invoke(prompt1)
print(result1.content)
print("--------------------------------------------------")
print("Translating the above content into Hindi")
prompt2 = template2.format(language="hindi", text=result1.content)
result2 = model.invoke(prompt2)
print(result2.content)