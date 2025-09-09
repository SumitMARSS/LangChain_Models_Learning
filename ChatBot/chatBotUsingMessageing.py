# version 3 - using langchain messages - HumanMessage, AIMessage, SystemMessage


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm = llm)


chat_history = [
    SystemMessage(content="You are a helpful AI assistant.")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chat. Goodbye!")
        break
    # response = model.invoke(user_input)
    response = model.invoke(chat_history)  # passing the chat history to the model so that it can understand the context
    print(f"Bot: {response.content}")
    chat_history.append(AIMessage(content=response.content))    
    

print("\nChat History:", chat_history)

