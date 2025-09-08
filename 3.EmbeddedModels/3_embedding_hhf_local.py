# for query embedding

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv  
load_dotenv()  

# Initialize the HuggingFaceEmbeddings model with a local model path - too big like 900MB

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = HuggingFaceEndpointEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2")
# Example usage
text = "This is a sample text for embedding."
vector = embeddings.embed_query(text)
print(vector)




# for document embedding

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv  
load_dotenv()  

# Initialize the HuggingFaceEmbeddings model with a local model path - too big like 900MB

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = HuggingFaceEndpointEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2")
# Example usage
document = [
    "This is a sample text for embedding.",
    "Here is another piece of text to embed.",
    "Embeddings are useful for various NLP tasks."
]
vector = embeddings.embed_documents(document)
print(vector)

