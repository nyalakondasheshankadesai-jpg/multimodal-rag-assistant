import os
import re

main_path = r"c:\Users\shash\Desktop\Anti_gravity_files\RAG PROJECT\backend\main.py"
with open(main_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the import from ingestion
content = content.replace(
    "from ingestion import clone_repository, process_documents, get_nodes_from_documents, extract_local_zip, process_single_document, process_single_image, DATA_DIR",
    "from ingestion import clone_repository, process_documents, get_nodes_from_documents, extract_local_zip, process_pdf_document, process_docx_document, process_single_image, DATA_DIR"
)

# We need to change `process_single_document` to either pdf or docx, and add hash checking.
# This requires significant changes to main.py. Let's just rewrite the relevant parts manually.
