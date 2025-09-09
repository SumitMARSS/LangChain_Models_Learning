import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

DB_NAME = "support_chat.db"

# ------------------ DB Functions ------------------

def init_db():
    """Create DB only if not exists"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def flush_db():
    """Delete all old messages from chat history"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()
    print("✅ Chat history cleared!")

def save_message(role, content):
    """Insert message into DB"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (role, content, timestamp) VALUES (?, ?, ?)",
              (role, content, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def load_history():
    """Fetch all old messages"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT role, content FROM chat_history ORDER BY id ASC")
    messages = c.fetchall()
    conn.close()
    return messages

def get_message_objects():
    """Convert DB messages to HumanMessage / AIMessage"""
    history = load_history()
    msg_objs = []
    for role, content in history:
        if role == "human":
            msg_objs.append(HumanMessage(content=content))
        elif role == "ai":
            msg_objs.append(AIMessage(content=content))
    return msg_objs


# ------------------ LLM Setup ------------------

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)


# ------------------ Prompt Template - one way but each time in message system message is going - takes high cost ------------------

# chat_template = ChatPromptTemplate.from_messages([
#     ("system", "You are a cricket expert helping people with their queries."),
#     MessagesPlaceholder("history"),
#     ("human", "{customer_query}, limit your response to 20 words.")
# ])


# # ------------------ Chat Flow ------------------

# def chat_with_customer(query):
#     history_msgs = get_message_objects()
#     if not history_msgs:
#         history_msgs.insert(0, SystemMessage(content="You are a cricket expert helping people with their queries."))
#     prompt = chat_template.invoke({
#         "history": history_msgs,
#         "customer_query": query
#     })
#     print("Prompt:", prompt)
#     # Call the model
#     response = model.invoke(prompt)

#     # Save conversation
#     save_message("human", query)
#     save_message("ai", response.content)

#     return response.content



# ------------------ Prompt Template - better way (only first time system message is sent after that only message history) ------------------


# Case 1: First message → only system + human
first_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a cricket expert helping people with their queries."),
    ("human", "{customer_query}, limit your response to 20 words.")
])

# Case 2: Subsequent messages → use history + new query
chat_template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("history"),
    ("human", "{customer_query}, limit your response to 20 words.")
])


# ------------------ Chat Flow ------------------

def chat_with_customer(query):
    history_msgs = get_message_objects()

    if not history_msgs:
        # First message → no history, send minimal prompt
        prompt = first_prompt.invoke({"customer_query": query})
    else:
        # Subsequent → full history via placeholder
        prompt = chat_template.invoke({
            "history": history_msgs,
            "customer_query": query
        })

    print("Prompt:", prompt)

    # Call the model
    response = model.invoke(prompt)

    # Save new conversation
    save_message("human", query)
    save_message("ai", response.content)

    return response.content

# ------------------ Main Loop ------------------

if __name__ == "__main__":
    init_db()
    print("Cricket Support Chatbot (type 'exit' to end, 'flush' to clear chat history)\n")

    while True:
        customer_query = input("You: ")

        if customer_query.lower() == "exit":
            print("Chat ended. See you next time!")
            break
        elif customer_query.lower() == "flush":
            flush_db()
            continue

        ai_response = chat_with_customer(customer_query)
        print("AI:", ai_response)


