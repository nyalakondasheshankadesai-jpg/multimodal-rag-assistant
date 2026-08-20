import os
import shutil
from git import Repo
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from . import DATA_DIR, REPO_DIR

def clear_repo_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if os.path.exists(REPO_DIR):
        try:
            def onerror(func, path, exc_info):
                import stat
                if not os.access(path, os.W_OK):
                    os.chmod(path, stat.S_IWUSR)
                    func(path)
                else:
                    raise
            shutil.rmtree(REPO_DIR, onerror=onerror)
        except Exception as e:
            print(f"Error removing old repo: {e}")

def clone_repository(repo_url: str):
    """Clones a github repo to a local directory, clearing it first if it exists."""
    clear_repo_dir()
    print(f"Cloning {repo_url} into {REPO_DIR}...")
    Repo.clone_from(repo_url, REPO_DIR)
    print("Cloning complete.")

def process_documents():
    """Reads documents from the cloned repo and applies smart chunking."""
    if not os.path.exists(REPO_DIR):
        raise FileNotFoundError(f"Repository directory {REPO_DIR} does not exist. Clone or upload a repo first.")
        
    exclude = [".git", "node_modules", "__pycache__", "venv", "env", ".env", "dist", "build"]
    
    def file_filter(file_path):
        for ex in exclude:
            if ex in file_path:
                return False
        valid_exts = [
            ".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".txt", ".json", 
            ".html", ".css", ".java", ".go", ".rs", ".cpp", ".c", ".h",
            ".php", ".rb", ".swift", ".kt", ".kts", ".scala", ".sh", 
            ".sql", ".yaml", ".yml", ".xml", ".vue"
        ]
        return any(file_path.endswith(ext) for ext in valid_exts)
        
    reader = SimpleDirectoryReader(
        input_dir=REPO_DIR, 
        recursive=True,
        file_extractor={}, 
    )
    documents = reader.load_data()
    
    filtered_docs = [doc for doc in documents if file_filter(doc.metadata.get("file_path", ""))]
    print(f"Loaded {len(filtered_docs)} valid documents from the repository.")
    
    return filtered_docs

def get_nodes_from_documents(documents, repo_name: str, ingestion_id: str):
    """Splits documents into nodes (chunks) based on file type."""
    nodes = []
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    
    for doc in documents:
        file_path = doc.metadata.get("file_path", "")
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == ".py":
                splitter = CodeSplitter(language="python", chunk_lines=40, chunk_lines_overlap=15, max_chars=1500)
                doc_nodes = splitter.get_nodes_from_documents([doc])
            elif ext in [".js", ".ts", ".jsx", ".tsx"]:
                splitter = CodeSplitter(language="javascript", chunk_lines=40, chunk_lines_overlap=15, max_chars=1500)
                doc_nodes = splitter.get_nodes_from_documents([doc])
            else:
                doc_nodes = text_splitter.get_nodes_from_documents([doc])
            
            for node in doc_nodes:
                node.metadata["repo_name"] = repo_name
                node.metadata["ingestion_id"] = ingestion_id
                node.metadata["file_name"] = os.path.basename(file_path)
                node.metadata["file_path"] = file_path
                node.metadata["extension"] = ext
                node.metadata["source_type"] = "code"
                
                # Robust Line number extraction using character offsets
                start_char = getattr(node, 'start_char_idx', None)
                if start_char is not None and doc.text:
                    start_line = doc.text.count('\\n', 0, start_char) + 1
                    end_char = getattr(node, 'end_char_idx', None)
                    if end_char is not None:
                        end_line = doc.text.count('\\n', 0, end_char) + 1
                    else:
                        end_line = start_line + node.text.count('\\n')
                        
                    node.metadata["start_line"] = start_line
                    node.metadata["end_line"] = end_line
                    
                if ext == ".py":
                    node.metadata["language"] = "python"
                elif ext in [".js", ".ts", ".jsx", ".tsx"]:
                    node.metadata["language"] = "javascript"
                else:
                    node.metadata["language"] = "text"
            
            nodes.extend(doc_nodes)
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            doc_nodes = text_splitter.get_nodes_from_documents([doc])
            
            for node in doc_nodes:
                node.metadata["repo_name"] = repo_name
                node.metadata["ingestion_id"] = ingestion_id
                node.metadata["file_name"] = os.path.basename(file_path)
                node.metadata["file_path"] = file_path
                node.metadata["extension"] = ext
                node.metadata["source_type"] = "code"
                node.metadata["language"] = "text"
                
            nodes.extend(doc_nodes)
            
    print(f"Successfully chunked into {len(nodes)} nodes.")
    return nodes
