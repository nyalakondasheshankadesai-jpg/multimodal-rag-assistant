import os
from llama_index.core import Document
from ingestion import get_nodes_from_documents
from rag import create_index_from_nodes, query_codebase

def test():
    print("Ingesting dummy data...")
    doc = Document(text="def cook_turkey():\n    print('Cooking turkey!')", metadata={"file_path": "turkey.py"})
    ingestion_id = "test-ingestion"
    nodes = get_nodes_from_documents([doc], repo_name="test-repo", ingestion_id=ingestion_id)
    create_index_from_nodes(nodes)
    print("Ingestion complete.")
    
    print("\nTesting Query:")
    res = query_codebase("how to cook a turkey in the python code", repo_name="test-repo")
    print("Answer:", res["answer"])
    for s in res["sources"]:
        print("Score:", s["score"])

if __name__ == "__main__":
    test()
