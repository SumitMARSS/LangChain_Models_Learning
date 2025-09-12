from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm = llm)


schemas = [
    ResponseSchema(name="fact1", description="fact 1 about the topic"),
    ResponseSchema(name="fact2", description="fact 2 about the topic"),
    ResponseSchema(name="fact3", description="fact 3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schemas)

template = PromptTemplate(
    template="Give me 3 fact about {topic} \n {format_instructions}",
    input_variables=['topic'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


chain = template | model | parser

response = chain.invoke({'topic': "Black holes"})
print(response)