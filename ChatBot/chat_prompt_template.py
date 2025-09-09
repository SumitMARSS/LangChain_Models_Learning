from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm = llm)

# this doesn't work as of now - maybe unstability in langchain - check later

# chatTemplate = ChatPromptTemplate.from_messages([
#     SystemMessage(content="You are a {domain} expert."),
#     HumanMessage(content="Explain in simpel terms, what is {topic}")
# ])


chatTemplate = ChatPromptTemplate([
    ('system', "You are a {domain} expert."),
    ('human', "Explain in simple terms, what is {topic}. LImit  your response to 20 words.")
])
    

prompt = chatTemplate.invoke({'domain': 'cricket', 'topic': 'wide'})

print("Prompt:", prompt)

model_response = model.invoke(prompt)
print("Model Response:", model_response.content)