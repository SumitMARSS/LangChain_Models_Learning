from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableBranch
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


prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="Write a report about {topic}."
)

prompt2 = PromptTemplate(
    input_variables=['report'],
    template="Summarize this report within 200 words \n {report}."
)

parser = StrOutputParser()

report_generate = RunnableSequence(prompt1, model1, parser)


# branch_chain = RunnableBranch(
#     (condition, runnable),
#     default
# )



branch_chain = RunnableBranch(
    (lambda x:len(x.split()) > 500, RunnableSequence(prompt2, model2, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_generate, branch_chain)

response = final_chain.invoke({"topic": "Unemployment in India"})
print("Final Output", response)
print("\n")






# graph chain visualization
final_chain.get_graph().print_ascii()
