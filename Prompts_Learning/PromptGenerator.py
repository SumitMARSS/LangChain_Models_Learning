from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["paper", "level", "style", "length"],
    template="""
        You are an expert AI research assistant. 
        Explain the research paper "{paper}" in the following way:

        - Audience level: {level}  
        - Explanation style: {style}  
        - Response length: {length}  

        Be clear, structured, and stay strictly focused on the paper.
    """,
    validate_template=True
)

prompt_template.save("currentPrompt.json")  # Save the prompt template to a JSON file