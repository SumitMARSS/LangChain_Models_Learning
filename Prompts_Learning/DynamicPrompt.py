import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

# ---------------- Streamlit UI ----------------
st.title("📄 Research Paper Explainer (LangChain + HuggingFace)")
st.write("Select options below to generate a customized explanation of a research paper.")

# 1. Dropdown: Research Paper Topics
paper = st.selectbox(
    "📑 Choose a research paper:",
    [
        "Attention Is All You Need (Transformer, 2017)",
        "BERT: Pre-training of Deep Bidirectional Transformers (2018)",
        "GPT-3: Language Models are Few-Shot Learners (2020)",
        "Vision Transformers (ViT, 2020)",
        "AlphaFold: Protein Structure Prediction (2021)",
        "Diffusion Models: Denoising Diffusion Probabilistic Models (2020)",
    ]
)

# 2. Dropdown: Level of Explanation
level = st.selectbox(
    "🎓 Choose explanation level:",
    ["Beginner Friendly", "Intermediate", "Advanced"]
)

# 3. Dropdown: Style of Explanation
style = st.selectbox(
    "🛠️ Choose explanation style:",
    ["Mathematical way", "Code way", "Conceptual way"]
)

# 4. Dropdown: Limit Paragraphs
length = st.selectbox(
    "✂️ Choose response length:",
    ["1-2 paragraphs", "2-5 paragraphs", "Detailed (5+ paragraphs)"]
)

# Dynamic Prompt
prompt_template = load_prompt("currentPrompt.json") 
# laod the prompt template from JSON file - reusable if it get once dumped into json - see PromptGenerator.py


if st.button("🔍 Generate Explanation"):
    with st.spinner("Generating explanation..."):
        formatted_prompt = prompt_template.format(
            paper=paper,
            level=level,
            style=style,
            length=length,
        )

        response = model.invoke(formatted_prompt)

        st.subheader(" Generated Explanation")
        st.write(response.content)
