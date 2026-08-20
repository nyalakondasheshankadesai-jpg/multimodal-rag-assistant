import os
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini

def get_llm():
    provider = os.getenv('LLM_PROVIDER', 'gemini').lower()
    
    if provider == 'gemini':
        return Gemini(model=os.getenv('GEMINI_MODEL', 'models/gemini-3.5-flash-lite'), api_key=os.getenv('GEMINI_API_KEY', os.getenv('GOOGLE_API_KEY', 'dummy-key')))
    elif provider == 'ollama':
        return Ollama(
            model=os.getenv('OLLAMA_MODEL', 'llama3'),
            request_timeout=float(os.getenv('OLLAMA_TIMEOUT', '120.0')),
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
