import os
from llama_index.llms.gemini import Gemini

def get_gemini_llm():
    model = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
    return Gemini(model=model)
