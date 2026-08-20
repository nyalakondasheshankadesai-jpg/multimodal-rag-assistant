import os
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

def process_text_document(file_path: str, doc_name: str, ingestion_id: str, knowledge_id: str | None = None):
    """
    Reads a plain text or markdown file, creates a Document, and splits it into chunks.
    """
    knowledge_id = knowledge_id or doc_name
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    doc = Document(
        text=text,
        metadata={
            "repo_name": knowledge_id,
            "ingestion_id": ingestion_id,
            "file_name": doc_name,
            "file_path": doc_name,
            "source_type": "document"
        }
    )
    
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents([doc])
    
    # Add metadata to each node explicitly if needed
    for node in nodes:
        node.metadata["repo_name"] = knowledge_id
        node.metadata["ingestion_id"] = ingestion_id
        node.metadata["file_name"] = doc_name
        node.metadata["file_path"] = doc_name
        node.metadata["source_type"] = "document"
        
    return nodes
