from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import requests
import time

API_URL = "http://localhost:8000/api"


class ErrorResponse:
    status_code = 500
    text = ""

    def json(self):
        return {}


def safe_get(url, **kwargs):
    try:
        resp = requests.get(url, **kwargs)
        if resp.status_code == 429 or (resp.text and "429" in resp.text):
            st.error("OpenAI/Gemini quota exceeded. Please switch to local providers in the Settings.")
        return resp
    except Exception as e:
        if "429" in str(e) or "Too Many Requests" in str(e):
            st.error("OpenAI/Gemini quota exceeded. Please switch to local providers in the Settings.")
            return ErrorResponse()
        raise e

def safe_post(url, **kwargs):
    try:
        resp = requests.post(url, **kwargs)
        if resp.status_code == 429 or (resp.text and "429" in resp.text):
            st.error("OpenAI/Gemini quota exceeded. Please switch to local providers in the Settings.")
        return resp
    except Exception as e:
        if "429" in str(e) or "Too Many Requests" in str(e):
            st.error("OpenAI/Gemini quota exceeded. Please switch to local providers in the Settings.")
            return ErrorResponse()
        raise e


st.set_page_config(page_title="AI Multi-Modal RAG Engineer", layout="wide")

st.title("🤖 AI Multi-Modal RAG Engineer")
st.markdown("Ask technical questions backed by source code and document citations!")

# Check if currently ingesting
try:
    current_status = safe_get(f"{API_URL}/status").json()
except:
    current_status = {"is_ingesting": False}

# Fetch available knowledge base items
try:
    repos_response = safe_get(f"{API_URL}/repos").json()
    available_repos = repos_response.get("repos", [])
except Exception:
    available_repos = []

# Fetch uuid mapping
try:
    mapping_response = safe_get(f"{API_URL}/uuid_mapping").json()
except Exception:
    mapping_response = {}


# Sidebar for Ingestion
with st.sidebar:
    page = st.radio("Navigation", ["Chat & Knowledge", "System Diagnostics", "Settings"])
    st.divider()

if page == "System Diagnostics":
    st.header("System Diagnostics")
    try:
        diag = safe_get(f"{API_URL}/diagnostics").json()
        col1, col2 = st.columns(2)
        col1.metric("CPU Usage", f"{diag.get('cpu_usage_percent', 0)}%")
        col2.metric("RAM Usage", f"{diag.get('ram_usage_percent', 0)}%")
        
        st.write(f"**RAM:** {diag.get('ram_used_gb', 0)} GB / {diag.get('ram_total_gb', 0)} GB")
        
        st.subheader("Configured Providers")
        providers = diag.get('providers', {})
        st.write(f"**Embedding Provider:** {providers.get('EMBEDDING_PROVIDER', 'Unknown')}")
        st.write(f"**LLM Provider:** {providers.get('LLM_PROVIDER', 'Unknown')}")
        st.write(f"**Vision Provider:** {providers.get('VISION_PROVIDER', 'Unknown')}")
    except Exception as e:
        st.error(f"Could not load diagnostics: {e}")

elif page == "Settings":
    st.header("Settings")
    if st.button("Test Embedding Model"):
        try:
            resp = safe_get(f"{API_URL}/test_embedding")
            if resp.status_code == 200:
                st.success("Embedding model is working perfectly!")
            elif resp.status_code != 429:
                st.error(f"Error testing embedding: {resp.text}")
        except Exception as e:
            st.error(f"Failed to connect: {e}")

elif page == "Chat & Knowledge":
    with st.sidebar:
        st.header("1. Add Knowledge")
    
    ingest_type = st.radio("Ingestion Method", ["GitHub URL", "Local ZIP Upload", "Document / Image"])
    
    if ingest_type == "GitHub URL":
        repo_url = st.text_input("GitHub Repo URL", placeholder="https://github.com/user/repo")
        if st.button("Ingest via GitHub"):
            if repo_url:
                try:
                    response = safe_post(f"{API_URL}/ingest", json={"repo_url": repo_url})
                    if response.status_code == 200:
                        st.success("Ingestion started in the background!")
                        st.rerun()
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
            else:
                st.warning("Please enter a valid URL.")
                
    elif ingest_type == "Local ZIP Upload":
        uploaded_file = st.file_uploader("Upload Codebase (ZIP)", type="zip")
        if st.button("Ingest via ZIP"):
            if uploaded_file is not None:
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/zip")}
                    response = safe_post(f"{API_URL}/ingest/local", files=files)
                    if response.status_code == 200:
                        st.success("ZIP Ingestion started in the background!")
                        st.rerun()
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
            else:
                st.warning("Please upload a ZIP file first.")
                
    elif ingest_type == "Document / Image":
        uploaded_doc = st.file_uploader("Upload Document / Image", type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "webp"])
        if st.button("Ingest Document"):
            if uploaded_doc is not None:
                try:
                    files = {"file": (uploaded_doc.name, uploaded_doc.getvalue(), uploaded_doc.type)}
                    response = safe_post(f"{API_URL}/ingest/document", files=files)
                    if response.status_code == 200:
                        st.success("Document Ingestion started in the background!")
                        st.rerun()
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
            else:
                st.warning("Please upload a document first.")
            
    st.divider()
    
    st.header("Status")
    status_placeholder = st.empty()
    progress_bar = st.empty()
    
    # Auto-polling loop if ingesting
    if current_status.get("is_ingesting"):
        while True:
            try:
                status = safe_get(f"{API_URL}/status").json()
                if status.get("is_ingesting"):
                    status_placeholder.info(f"🔄 {status.get('progress_message', 'Ingestion in progress...')}")
                    if status.get("progress_percentage", 0) > 0:
                        progress_bar.progress(status["progress_percentage"])
                    time.sleep(1.5)
                elif status.get("error"):
                    status_placeholder.error(f"❌ Error: {status['error']}")
                    break
                else:
                    status_placeholder.success(f"✅ Ready! Last ingested: {status['last_ingested_repo']}")
                    progress_bar.empty()
                    time.sleep(1)
                    st.rerun()
            except Exception:
                status_placeholder.error("Backend offline.")
                break
    else:
        if current_status.get("error"):
            status_placeholder.error(f"❌ Error: {current_status['error']}")
        elif current_status.get("last_ingested_repo"):
            status_placeholder.success(f"✅ Ready! Last ingested: {current_status['last_ingested_repo']}")
        else:
            status_placeholder.write("Ready to ingest.")

    st.divider()
    st.header("📚 Knowledge Base")
    if available_repos:
        for repo in available_repos:
            col1, col2 = st.columns([0.8, 0.2])
            display_name = mapping_response.get(repo, repo)
            col1.write(f"✓ {display_name}")
            if col2.button("🗑️", key=f"del_{repo}"):
                try:
                    requests.delete(f"{API_URL}/knowledge/{repo}")
                    st.rerun()
                except:
                    st.error("Failed to delete.")
    else:
        st.write("No knowledge ingested yet.")

    # Main Chat Interface
    st.header("2. Ask Questions")

    selected_repo = None
    if available_repos:
        selected_repo = st.selectbox("Search within Context (Metadata Filter):", ["Everything"] + available_repos)
        if selected_repo == "Everything":
            selected_repo = None
    else:
        st.info("No knowledge base found. Please ingest a repository or document first.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    def render_citations(sources):
        for source in sources:
            source_type = source.get('source_type', 'code')
            file_name = source['file']
            score = source.get('score')
            score_label = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
            
            if source_type == "document":
                page = source.get('page')
                loc_str = f"Page {page}" if page else ""
                st.markdown(f"**📄 {file_name}** {loc_str} *(Score: {score_label})*")
            elif source_type == "image":
                st.markdown(f"**🖼️ {file_name}** *(Score: {score_label})*")
            else:
                lines_str = f"Lines {source['start_line']}-{source['end_line']}" if source.get('start_line') and source.get('end_line') else ""
                st.markdown(f"**💻 {file_name}** {lines_str} *(Score: {score_label})*")
                
            st.code(source["content"], language="text" if source_type in ("document", "image") else "python")


    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("View Sources"):
                    render_citations(msg["sources"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the knowledge base..."):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking and retrieving context..."):
                try:
                    payload = {"question": prompt}
                    if selected_repo:
                        payload["repo_name"] = selected_repo
                        
                    response = safe_post(f"{API_URL}/ask", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])
                        
                        st.markdown(answer)
                        if sources:
                            with st.expander("View Sources"):
                                render_citations(sources)
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        error_msg = f"Error: {response.json().get('detail', 'Unknown error')}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    error_msg = f"Failed to connect to backend: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
