from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_chroma import Chroma
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm = llm)

###############################################################
## step - 1.a  - Indexing (Document Ingestion)
###############################################################

video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    yt_api = YouTubeTranscriptApi()
    transcript_list = yt_api.fetch(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(chunk.text for chunk in transcript_list)
    #print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

# print("Transcript ", len(transcript_list))
# print(transcript_list[0])
# print(transcript_list)


###############################################################
## step - 1.b  - Indexing (Text Splitting)
###############################################################

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(transcript)

print("Number of chunks created: ", len(chunks))

###############################################################################
## step - 1.b  - Indexing (Embedding Generation and Storing in Vector Store)
###############################################################################

model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
embeddings = model.encode(chunks)
vector_store = Chroma(documents = chunks, embedding = embeddings, persist_directory = "./chroma_db")

###############################################################################
## step - 2  - ## Step 2 - Retrieval
###############################################################################

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

## retriever.invoke("What is the main topic of the video?")

###############################################################################
## step - 3  - ## Step 3 - Augumentation
###############################################################################

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)


context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
print("Context: ", context_text)

final_prompt = prompt.format(context=context_text, question=question)
print("Final Prompt: ", final_prompt)


ans = model.invoke(final_prompt)
print("Answer: ", ans)