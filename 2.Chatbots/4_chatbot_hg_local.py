from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from dotenv import load_dotenv
load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    model_kwargs={"temperature":0, "max_new_tokens":112}
)


model = ChatHuggingFace(llm = llm)
response = model.invoke("What is the capital of India?")
print(response.content)