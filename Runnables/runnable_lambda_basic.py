from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv


def word_counter(str):
    return len(str.split())


chain = RunnableLambda(word_counter)
print(chain.invoke("This is a test string to count the number of words."))



## any custome logic can be implemented in the lambda function and can be used in the RunnableLambda class. -> can be treated as a custom runnable.