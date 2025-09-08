
# Basic ChatBot using LangChain and HuggingFace as it forgot everything after each response - no memory (stateless)



# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv

# load_dotenv()


# llm = HuggingFaceEndpoint(
#     repo_id= "meta-llama/Llama-3.1-8B-Instruct",
#     task="text-generation",
# )
# model = ChatHuggingFace(llm = llm)

# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ['exit', 'quit']:
#         print("Exiting the chat. Goodbye!")
#         break
#     response = model.invoke(user_input)
#     print(f"Bot: {response.content}")





# Solution for memory part - keeping the chat history in a list and passing it to the model each time
# Their is multiple ways to do this - history buffer summary, conversation buffer etc - check langchain docs for more details


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm = llm)


chat_history = []
# in chat history send content - like - you - ... , bot - ... so that llm can understand the context


while True:
    user_input = input("You: ")
    chat_history.append(f"You: {user_input}")
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chat. Goodbye!")
        break
    # response = model.invoke(user_input)
    response = model.invoke(chat_history)  # passing the chat history to the model so that it can understand the context
    print(f"Bot: {response.content}")
    chat_history.append(f"Bot: {response.content}")    
    

print("\nChat History:", "\n".join(chat_history))

