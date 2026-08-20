import os
from llama_index.llms.ollama import Ollama

def get_ollama_llm():
    model = os.getenv("OLLAMA_MODEL", "llama3")
    timeout = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))
    return Ollama(model=model, request_timeout=timeout)
