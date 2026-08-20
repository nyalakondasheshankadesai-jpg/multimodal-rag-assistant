import os
import qdrant_client
import json
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.schema import NodeWithScore
from typing import List, Optional
from llama_index.core import PromptTemplate
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

load_dotenv(override=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
QDRANT_DB_DIR = os.path.join(DATA_DIR, "qdrant_db")
from embeddings.factory import get_embedding_model
from llm.factory import get_llm



def get_collection_name():
    model = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    sanitized = model.replace("/", "_").replace("-", "_").replace(".", "_").lower()
    return f"rag_{sanitized}"


qa_prompt_str = (
    "You are an expert AI Technical Documentation Engineer.\n"
    "Context information from the codebase is below. It includes file paths and line numbers in the metadata.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and no prior knowledge, "
    "answer the query clearly and concisely. If you do not know the answer based on the context, "
    "say 'I don't know'. Do not hallucinate.\n"
    "Query: {query_str}\n"
    "Answer: "
)
QA_PROMPT = PromptTemplate(qa_prompt_str)

def get_qdrant_client():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    client = qdrant_client.QdrantClient(path=QDRANT_DB_DIR)
    return client

def get_repos_registry():
    repos_file = os.path.join(DATA_DIR, "repos.json")
    if os.path.exists(repos_file):
        try:
            with open(repos_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def set_active_ingestion(repo_name: str, ingestion_id: str, display_name: str = None):
    repos = get_repos_registry()
    repos[repo_name] = {
        "active_ingestion_id": ingestion_id,
        "display_name": display_name or repo_name,
    }
    with open(os.path.join(DATA_DIR, "repos.json"), "w") as f:
        json.dump(repos, f)


def get_knowledge_display_names():
    """Return stable knowledge IDs mapped to names suitable for the UI."""
    return {
        knowledge_id: details.get("display_name", knowledge_id)
        for knowledge_id, details in get_repos_registry().items()
    }

def delete_knowledge(repo_name: str):
    """Deletes a repository or document from the registry and Qdrant."""
    repos = get_repos_registry()
    if repo_name in repos:
        del repos[repo_name]
        with open(os.path.join(DATA_DIR, "repos.json"), "w") as f:
            json.dump(repos, f)
            
    client = get_qdrant_client()
    if client.collection_exists(get_collection_name()):
        try:
            client.delete(
                collection_name=get_collection_name(),
                points_selector=Filter(
                    must=[
                        FieldCondition(key="repo_name", match=MatchValue(value=repo_name))
                    ]
                )
            )
        except Exception as e:
            print(f"Failed to delete {repo_name} from Qdrant: {e}")

def get_ingested_repos():
    return list(get_repos_registry().keys())

def get_active_ingestion_id(repo_name: str) -> Optional[str]:
    repos = get_repos_registry()
    return repos.get(repo_name, {}).get("active_ingestion_id")

def cleanup_old_versions(repo_name: str, active_ingestion_id: str):
    """Deletes all chunks for the repo that don't match the active ingestion ID (atomic update cleanup)."""
    client = get_qdrant_client()
    if client.collection_exists(get_collection_name()):
        try:
            client.delete(
                collection_name=get_collection_name(),
                points_selector=Filter(
                    must=[
                        FieldCondition(key="repo_name", match=MatchValue(value=repo_name))
                    ],
                    must_not=[
                        FieldCondition(key="ingestion_id", match=MatchValue(value=active_ingestion_id))
                    ]
                )
            )
            print(f"Cleaned up old chunks for repo: {repo_name}")
        except Exception as e:
            print(f"Cleanup failed: {e}")


def delete_ingestion_version(repo_name: str, ingestion_id: str):
    """Remove chunks from an ingestion that did not complete successfully."""
    client = get_qdrant_client()
    if client.collection_exists(get_collection_name()):
        client.delete(
            collection_name=get_collection_name(),
            points_selector=Filter(
                must=[
                    FieldCondition(key="repo_name", match=MatchValue(value=repo_name)),
                    FieldCondition(key="ingestion_id", match=MatchValue(value=ingestion_id)),
                ]
            ),
        )

def create_index_from_nodes(nodes, progress_callback=None):
    Settings.llm = get_llm()
    Settings.embed_model = get_embedding_model()
    """Inserts nodes into Qdrant. Atomic commit is handled by returning success before cleanup."""
    client = get_qdrant_client()
    
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=get_collection_name(),
        enable_hybrid=True
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("Building Vector Store Index with Qdrant...")
    index = VectorStoreIndex([], storage_context=storage_context)
    
    provider = "huggingface"
    model_name = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        
    for node in nodes:
        node.metadata["embedding_provider"] = provider
        node.metadata["embedding_model"] = model_name
        
    batch_size = 20
    total_nodes = len(nodes)
    
    for i in range(0, total_nodes, batch_size):
        batch = nodes[i:i+batch_size]
        index.insert_nodes(batch)
        if progress_callback:
            progress_callback(min(i + batch_size, total_nodes), total_nodes)
            
    print("Index built and stored in Qdrant.")
    return index

def get_query_engine(repo_name: str = None):
    Settings.llm = get_llm()
    Settings.embed_model = get_embedding_model()
    client = get_qdrant_client()
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=get_collection_name(),
        enable_hybrid=True
    )
    
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
        top_n=5
    )
    
    filter_conditions = []
    if repo_name:
        filter_conditions.append(ExactMatchFilter(key="repo_name", value=repo_name))
        active_id = get_active_ingestion_id(repo_name)
        if active_id:
            filter_conditions.append(ExactMatchFilter(key="ingestion_id", value=active_id))
            
    filters = MetadataFilters(filters=filter_conditions) if filter_conditions else None
    
    query_engine = index.as_query_engine(
        similarity_top_k=20,
        sparse_top_k=20,
        filters=filters,
        vector_store_query_mode="hybrid",
        node_postprocessors=[reranker],
        response_mode="compact",
        text_qa_template=QA_PROMPT
    )
    
    # Removed HyDEQueryTransform to improve exact code matches
    return query_engine

def query_codebase(question: str, repo_name: str = None):
    engine = get_query_engine(repo_name=repo_name)
    try:
        response = engine.query(question)
    except Exception as e:
        if "429" in str(e) or "Too Many Requests" in str(e):
            raise Exception("429 Too Many Requests: Rate limit exceeded for LLM provider. Please wait and try again.")
        raise e
    
    sources = []
    for source_node in response.source_nodes:
        metadata = source_node.node.metadata
        sources.append({
            "file": metadata.get("file_name", "Unknown File"),
            "path": metadata.get("file_path", "Unknown Path"),
            "source_type": metadata.get("source_type", "code"),
            "start_line": metadata.get("start_line"),
            "end_line": metadata.get("end_line"),
            "page": metadata.get("page"),
            "content": source_node.node.get_content()[:200] + "...",
            "score": float(source_node.score) if source_node.score is not None else None
        })
        
    return {
        "answer": response.response,
        "sources": sources
    }
