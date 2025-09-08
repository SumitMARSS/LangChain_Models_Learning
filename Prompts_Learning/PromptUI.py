import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)


model = ChatHuggingFace(llm=llm)

##########################################    static  prompt   ###################################################

# Streamlit UI
# st.title("📝 Text Summarizer with Hugging Face + LangChain")
# st.write("Enter any text below and get a summarized version using LLaMA-3.1.")

# # User input
# user_input = st.text_area("✍️ Enter your text to summarize:")

# if st.button("Summarize"):
#     if user_input.strip():
#         # Get response
#         response = model.invoke(user_input)

#         # Show output
#         st.subheader("Summary")
#         st.write(response.content)
#     else:
#         st.warning("⚠️ Please enter some text before summarizing.")



##########################################    dynamic  prompt   ###################################################

prompt_template = PromptTemplate(
    input_variables=["query"],
    template="""
You are a knowledgeable cricket assistant. 
- Only answer questions about **cricketers** (e.g., Virat Kohli, MS Dhoni, Rohit Sharma, Jasprit Bumrah, etc.).  
- If the question is NOT about cricketers, politely respond with: 
  "❌ I can only answer questions related to cricketers."  

Now, here is the user's question:  
{query}
"""
)

# Streamlit UI
st.title("🏏 Cricketer Q&A Bot (LangChain + Hugging Face)")
st.write("Ask me anything about cricketers, and I’ll answer. If you ask something else, I’ll politely refuse 🙂")

# User input
user_input = st.text_area("✍️ Enter your question about a cricketer:")

if st.button("Ask"):
    if user_input.strip():
        # Format prompt dynamically
        formatted_prompt = prompt_template.format(query=user_input)

        # Get response
        response = model.invoke(formatted_prompt)

        # Show output
        st.subheader("Summary")
        st.write(response.content)
    else:
        st.warning("⚠️ Please enter a question about a cricketer.")




