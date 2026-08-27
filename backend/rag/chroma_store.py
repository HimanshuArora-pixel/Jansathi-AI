import os
import chromadb
from chromadb.config import Settings
from chromadb.config import Settings

# Initialize ChromaDB client
PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
client = chromadb.PersistentClient(path=PERSIST_DIRECTORY, settings=Settings(anonymized_telemetry=False))

# Use HuggingFace Serverless Inference API for embeddings to save RAM
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Requires HF_TOKEN environment variable
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    embedding_function = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name="BAAI/bge-small-en-v1.5"
    )
else:
    print("Warning: HF_TOKEN not set. Vector search will be disabled.")
    embedding_function = None

def get_collection(collection_name: str = "jansaathi_legal_kb"):
    """
    Returns the ChromaDB collection for the given name.
    Creates it if it doesn't exist.
    """
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"} # Cosine similarity works best for text embeddings
    )

def add_documents(collection_name: str, documents: list[str], metadatas: list[dict], ids: list[str]):
    """
    Embeds and adds documents to the ChromaDB collection.
    """
    if embedding_function is None:
        print("Vector search disabled on Render. Skipping adding documents to Chroma.")
        return
        
    collection = get_collection(collection_name)
    embeddings = embedding_function.embed_documents(documents)
    
    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Added {len(documents)} documents to {collection_name}")

def search_documents(collection_name: str, query: str, n_results: int = 5, where: dict = None):
    """
    Searches the ChromaDB collection using the query string.
    """
    if embedding_function is None:
        return {} # Vector search disabled on Render, BM25 will handle retrieval
        
    collection = get_collection(collection_name)
    query_embedding = embedding_function.embed_query(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where
    )
    
    return results
