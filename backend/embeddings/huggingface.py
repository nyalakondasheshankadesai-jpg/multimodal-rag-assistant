import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def get_huggingface_embedding():
    hf_model = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    return HuggingFaceEmbedding(model_name=hf_model)
