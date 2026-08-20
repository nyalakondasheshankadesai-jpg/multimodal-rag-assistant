import os
import pdfplumber
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

def process_pdf_document(file_path: str, doc_name: str, ingestion_id: str, knowledge_id: str | None = None):
    print(f"Extracting PDF with layout preservation: {doc_name}")
    knowledge_id = knowledge_id or doc_name
    nodes = []
    
    with pdfplumber.open(file_path) as pdf:
        text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        
        for i, page in enumerate(pdf.pages):
            # Extract text preserving layout
            text = page.extract_text(layout=True)
            if not text:
                continue
                
            # Extract tables
            tables = page.extract_tables()
            for table in tables:
                if table:
                    table_str = "\\n".join([(" | ".join([str(cell) if cell else "" for cell in row])) for row in table])
                    text += "\\n\\n[TABLE]\\n" + table_str + "\\n[/TABLE]\\n"
                    
            doc = Document(text=text)
            doc_nodes = text_splitter.get_nodes_from_documents([doc])
            
            for node in doc_nodes:
                node.metadata["repo_name"] = knowledge_id
                node.metadata["ingestion_id"] = ingestion_id
                node.metadata["file_name"] = doc_name
                node.metadata["file_path"] = doc_name
                node.metadata["source_type"] = "document"
                node.metadata["extension"] = ".pdf"
                node.metadata["page_start"] = i + 1
                node.metadata["page_end"] = i + 1
                node.metadata["page"] = i + 1
                node.metadata["start_char"] = getattr(node, 'start_char_idx', None)
                node.metadata["end_char"] = getattr(node, 'end_char_idx', None)
                
                nodes.append(node)
                
    print(f"Successfully chunked {doc_name} into {len(nodes)} nodes.")
    return nodes
