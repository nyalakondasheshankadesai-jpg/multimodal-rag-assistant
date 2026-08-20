import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def get_embedding_model():
    """Use a local Hugging Face model; embeddings never require an API key."""
    model_name = os.getenv('HF_EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')
    return HuggingFaceEmbedding(model_name=model_name)
