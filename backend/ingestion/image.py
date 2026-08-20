import os
import base64
import requests
import pytesseract
from PIL import Image
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

def extract_image_text(file_path: str) -> str:
    vision_provider = os.getenv("VISION_PROVIDER", "gemini").lower()
    
    if vision_provider == "local":
        print("Using local OCR (pytesseract)")
        return pytesseract.image_to_string(Image.open(file_path))
        
    with open(file_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    prompt = "1. Extract all text from this image exactly as written. 2. Provide a detailed description of the visual contents."
    
    if vision_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        ext = os.path.splitext(file_path)[1].lower()
        mime_type = "image/png" if ext == ".png" else "image/webp" if ext == ".webp" else "image/jpeg"
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime_type, "data": base64_image}}
                ]
            }]
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        raise ValueError(f"Unsupported VISION_PROVIDER: {vision_provider}")

def process_single_image(file_path: str, doc_name: str, ingestion_id: str, knowledge_id: str | None = None):
    print(f"Extracting text from image {doc_name}...")
    knowledge_id = knowledge_id or doc_name
    text = extract_image_text(file_path)
    doc = Document(text=text)
    
    nodes = []
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    doc_nodes = text_splitter.get_nodes_from_documents([doc])
    
    for node in doc_nodes:
        node.metadata["repo_name"] = knowledge_id
        node.metadata["ingestion_id"] = ingestion_id
        node.metadata["file_name"] = doc_name
        node.metadata["file_path"] = doc_name
        node.metadata["source_type"] = "image"
        nodes.append(node)
        
    print(f"Successfully chunked image {doc_name} into {len(nodes)} nodes.")
    return nodes
