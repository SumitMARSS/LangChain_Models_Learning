from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal 
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from dotenv import load_dotenv

load_dotenv()


llm1 = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

llm2 = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V3.1",
    task="text-generation",
)


model1 = ChatHuggingFace(llm = llm1)
model2 = ChatHuggingFace(llm = llm2)


class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="The sentiment of the feedback, either 'positive' or 'negative'")


parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    input_variables=["feedback"],
    template="Classify the following feedback, as 'positive' or 'negative'\n {feedback} \n {formal_instruction}" ,
    partial_variables={"formal_instruction": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model1 | parser2

prompt2 = PromptTemplate(
    input_variables=["feedback"],
    template="Provide a detailed response to the following positive feedback:\n {feedback} "
)

prompt3 = PromptTemplate(
    input_variables=["feedback"],
    template="Provide a detailed response to the following negative feedback, including an apology and a solution:\n {feedback} "
)

# branch chain start - content to be write inside renunable branch  


# branch_chain = RunnableBranch(
#     (condition1, chain1),
#     (condition2, chain2),
#     default chain
# )



branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model1 | parser1),
    (lambda x: x.sentiment == 'negative', prompt3 | model1 | parser1),
    RunnableLambda(lambda x: "Could not classify the sentiment." )
)

chain = classifier_chain | branch_chain
result = chain.invoke({"feedback": "This is the worst phone i got ever!"})
print(result)



# graph chain visualization
chain.get_graph().print_ascii()
