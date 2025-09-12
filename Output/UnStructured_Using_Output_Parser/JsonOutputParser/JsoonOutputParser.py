from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm = llm)


parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age, city of a fictional person {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


#Without using chains

# prompt = template.format()
# print(prompt)
# response = model.invoke(prompt)
# print(response)
# final_result = parser.parse(response.content)
# print(final_result)
# print() 




# Using chains


chain = template | model | parser
response = chain.invoke({})
print(response)
