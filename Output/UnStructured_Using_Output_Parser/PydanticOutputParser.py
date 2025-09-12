from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm = llm)


class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    city: str = Field(description="The city where the person lives")    


parser = PydanticOutputParser(pydantic_object=Person)


template = PromptTemplate(
    input_variables=["input"],
    template="""Give name, age, and city of a fictional person from {input}.
                Return ONLY a valid JSON object with the following schema:
                {format_instructions}
                Do not add any explanation, code, or text outside the JSON.
            """,
    partial_variables={"format_instructions": parser.get_format_instructions()},
)



chain = template | model | parser

response = chain.invoke({"input": "Spain"})
print(response)