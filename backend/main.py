from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
import uvicorn
import os
import time
import hashlib
import uuid
import psutil
import threading
from pathlib import Path
from urllib.parse import urlparse

from ingestion.github import clone_repository, process_documents, get_nodes_from_documents
from ingestion.zip import extract_local_zip
from ingestion.pdf import process_pdf_document
from ingestion.docx import process_docx_document
from ingestion.image import process_single_image
from ingestion.text import process_text_document

from rag import (
    create_index_from_nodes,
    query_codebase,
    get_ingested_repos,
    get_active_ingestion_id,
    set_active_ingestion,
    cleanup_old_versions,
    delete_ingestion_version,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
UPLOAD_DIR = os.path.join(DATA_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="AI Technical Documentation Engineer API")

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp"}
ingestion_lock = threading.Lock()

class IngestRequest(BaseModel):
    repo_url: str

class QueryRequest(BaseModel):
    question: str
    repo_name: str = None

status = {
    "is_ingesting": False,
    "last_ingested_repo": None,
    "error": None,
    "progress_message": "",
    "progress_percentage": 0.0
}
query_cache = {}


def begin_ingestion() -> None:
    """Reserve the single local ingestion workspace before a task is queued."""
    if not ingestion_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An ingestion process is already running.")
    status.update({
        "is_ingesting": True,
        "error": None,
        "progress_message": "Preparing ingestion...",
        "progress_percentage": 0.0,
    })


def finish_ingestion() -> None:
    status["is_ingesting"] = False
    if ingestion_lock.locked():
        ingestion_lock.release()


def new_ingestion_id() -> str:
    return f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"


def save_upload(file: UploadFile, destination: str) -> None:
    """Save an uploaded file with an explicit size limit."""
    total = 0
    with open(destination, "wb") as buffer:
        while chunk := file.file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Upload exceeds the 100 MB limit.")
            buffer.write(chunk)


def repository_display_name(repo_url: str) -> str:
    path = urlparse(repo_url).path.rstrip("/")
    name = Path(path).name
    return name[:-4] if name.endswith(".git") else name


def validate_github_url(repo_url: str) -> None:
    parsed = urlparse(repo_url)
    if parsed.scheme != "https" or parsed.hostname not in {"github.com", "www.github.com"}:
        raise HTTPException(status_code=400, detail="Provide an HTTPS URL for a GitHub repository.")
    if len([part for part in parsed.path.split("/") if part]) < 2:
        raise HTTPException(status_code=400, detail="Provide a complete GitHub repository URL.")

def update_embedding_progress(current, total):
    global status
    status["progress_message"] = f"Generating Embeddings ({current}/{total} chunks)..."
    status["progress_percentage"] = float(current) / float(total) if total > 0 else 1.0

def complete_indexing(nodes, knowledge_id: str, ingestion_id: str, display_name: str) -> None:
    if not nodes:
        raise ValueError("No readable content was found to index.")
    status["progress_message"] = "Generating Embeddings..."
    create_index_from_nodes(nodes, progress_callback=update_embedding_progress)
    set_active_ingestion(knowledge_id, ingestion_id, display_name=display_name)
    cleanup_old_versions(knowledge_id, ingestion_id)
    query_cache.clear()


def background_ingest_document(file_path: str, filename: str, knowledge_id: str):
    global status
    ingestion_id = None
    try:
        status["progress_message"] = "Parsing document..."
        ingestion_id = new_ingestion_id()
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.webp']:
            nodes = process_single_image(file_path, filename, ingestion_id, knowledge_id)
        elif ext == '.pdf':
            nodes = process_pdf_document(file_path, filename, ingestion_id, knowledge_id)
        elif ext == '.docx':
            nodes = process_docx_document(file_path, filename, ingestion_id, knowledge_id)
        else:
            nodes = process_text_document(file_path, filename, ingestion_id, knowledge_id)

        complete_indexing(nodes, knowledge_id, ingestion_id, filename)
        status["last_ingested_repo"] = f"Document: {filename}"
        status["progress_message"] = "Ingestion Complete!"
        status["progress_percentage"] = 1.0
        print("Document Ingestion Pipeline Completed Successfully.")
            
    except Exception as e:
        if ingestion_id:
            try:
                delete_ingestion_version(knowledge_id, ingestion_id)
            except Exception as cleanup_error:
                print(f"Failed to remove partial document ingestion: {cleanup_error}")
        status["error"] = str(e)
        status["progress_message"] = "Ingestion Failed"
        print(f"Document ingestion failed: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        finish_ingestion()


def background_ingest_repository(repo_url: str, knowledge_id: str, display_name: str, zip_path: str | None = None):
    ingestion_id = None
    try:
        status["progress_message"] = "Downloading repository..." if not zip_path else "Extracting ZIP archive..."
        if zip_path:
            extract_local_zip(zip_path)
        else:
            clone_repository(repo_url)
        status["progress_message"] = "Parsing repository files..."
        ingestion_id = new_ingestion_id()
        documents = process_documents()
        nodes = get_nodes_from_documents(documents, repo_name=knowledge_id, ingestion_id=ingestion_id)
        complete_indexing(nodes, knowledge_id, ingestion_id, display_name)
        status["last_ingested_repo"] = f"Repository: {display_name}"
        status["progress_message"] = "Ingestion Complete!"
        status["progress_percentage"] = 1.0
    except Exception as e:
        if ingestion_id:
            try:
                delete_ingestion_version(knowledge_id, ingestion_id)
            except Exception as cleanup_error:
                print(f"Failed to remove partial repository ingestion: {cleanup_error}")
        status["error"] = str(e)
        status["progress_message"] = "Ingestion Failed"
        print(f"Repository ingestion failed: {e}")
    finally:
        if zip_path and os.path.exists(zip_path):
            os.remove(zip_path)
        finish_ingestion()

@app.post("/api/ingest/document")
async def ingest_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "")
    ext = os.path.splitext(filename)[1].lower()
    if not filename or ext not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Supported file types: {', '.join(sorted(ALLOWED_DOCUMENT_EXTENSIONS))}")

    begin_ingestion()
    knowledge_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{knowledge_id}{ext}")
    try:
        save_upload(file, temp_path)
        background_tasks.add_task(background_ingest_document, temp_path, filename, knowledge_id)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        finish_ingestion()
        raise
    return {"message": f"Started document ingestion for {filename} in the background.", "uuid": knowledge_id}


@app.post("/api/ingest")
async def ingest_github_repository(request: IngestRequest, background_tasks: BackgroundTasks):
    validate_github_url(request.repo_url)
    begin_ingestion()
    knowledge_id = str(uuid.uuid4())
    display_name = repository_display_name(request.repo_url)
    background_tasks.add_task(background_ingest_repository, request.repo_url, knowledge_id, display_name)
    return {"message": f"Started repository ingestion for {display_name} in the background.", "uuid": knowledge_id}


@app.post("/api/ingest/local")
async def ingest_local_zip(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload a ZIP archive.")

    begin_ingestion()
    knowledge_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{knowledge_id}.zip")
    try:
        save_upload(file, temp_path)
        background_tasks.add_task(background_ingest_repository, "", knowledge_id, filename, temp_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        finish_ingestion()
        raise
    return {"message": f"Started ZIP ingestion for {filename} in the background.", "uuid": knowledge_id}


@app.get("/api/uuid_mapping")
async def get_uuid_mapping():
    from rag import get_knowledge_display_names
    return get_knowledge_display_names()
@app.get("/api/status")
async def get_status():
    return status

@app.get("/api/repos")
async def list_repos():
    return {"repos": get_ingested_repos()}

@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    if status["is_ingesting"]:
        raise HTTPException(status_code=400, detail="Cannot answer questions while ingestion is in progress.")
        
    repo_version = get_active_ingestion_id(request.repo_name) if request.repo_name else "default"
    cache_key = hashlib.sha256(f"{request.repo_name}_{repo_version}_{request.question}".encode('utf-8')).hexdigest()
    
    if cache_key in query_cache:
        return query_cache[cache_key]
        
    try:
        result = query_codebase(request.question, repo_name=request.repo_name)
        query_cache[cache_key] = result
        if len(query_cache) > 500:
            query_cache.pop(next(iter(query_cache)))
        return result
    except Exception as e:
        if "429" in str(e):
            raise HTTPException(status_code=429, detail="Quota exceeded for LLM/Embedding provider.")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/knowledge/{knowledge_id}")
async def remove_knowledge(knowledge_id: str):
    if status["is_ingesting"]:
        raise HTTPException(status_code=400, detail="Cannot delete knowledge while ingestion is in progress.")
    from rag import delete_knowledge
    try:
        delete_knowledge(knowledge_id)
        query_cache.clear()
        return {"message": f"Successfully deleted {knowledge_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/diagnostics")
async def get_diagnostics():
    memory = psutil.virtual_memory()
    return {
        "cpu_usage_percent": psutil.cpu_percent(),
        "ram_usage_percent": memory.percent,
        "ram_used_gb": round(memory.used / (1024 ** 3), 2),
        "ram_total_gb": round(memory.total / (1024 ** 3), 2),
        "providers": {
            "EMBEDDING_PROVIDER": "huggingface (local)",
            "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "ollama"),
            "VISION_PROVIDER": os.getenv("VISION_PROVIDER", "local"),
        },
    }

@app.get("/api/test_embedding")
async def test_embedding():
    try:
        from embeddings.factory import get_embedding_model
        model = get_embedding_model()
        model.get_text_embedding("test")
        return {"status": "success"}
    except Exception as e:
        if "429" in str(e):
            raise HTTPException(status_code=429, detail="Quota exceeded")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
