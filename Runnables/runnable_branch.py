from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a detailed report on {topic}."
)

prompt2 = PromptTemplate(
    input_variables=["text"],
    template="Summaries the following report within 100 words: {text}."
)


report_generation_chain = RunnableSequence(prompt, model, parser)

conditional_chain = RunnableBranch(
    (lambda x: len(x.split()) > 100, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
) 

final_chain = RunnableSequence(report_generation_chain, conditional_chain) # combining the report generation and conditional logic into a single sequence
result = final_chain.invoke({"topic": "Cricket"})
print()
print(result)
print()
