import time
import requests
import sys

API_URL = "http://localhost:8000/api"

# Default fallback if no arg provided
REPO_NAME = "Prompt-Based-text-classification-" 

QUESTIONS = [
    "What is the main purpose of this repository?",
    "How does the application start?",
    "Are there any external APIs or dependencies used?",
    "how to cook a turkey in the python code", # Nonsense question to test Confidence Cut-off
]

def evaluate(repo_name):
    print(f"Starting Evaluation for Repository: {repo_name}")
    print("Ensure your backend (FastAPI) is running at http://localhost:8000")
    print("-" * 60)
    
    total_time = 0
    
    for i, question in enumerate(QUESTIONS):
        print(f"\n[Q{i+1}] {question}")
        start_time = time.time()
        
        payload = {"question": question, "repo_name": repo_name}
        
        try:
            response = requests.post(f"{API_URL}/ask", json=payload)
            latency = time.time() - start_time
            total_time += latency
            
            if response.status_code == 200:
                data = response.json()
                sources = data.get("sources", [])
                
                if sources:
                    scores = [s.get("score", -999) for s in sources if s.get("score") is not None]
                    max_score = f"{max(scores):.2f}" if scores else "N/A"
                else:
                    max_score = "N/A (Cut-off triggered or no sources)"
                    
                print(f"  Latency    : {latency:.3f}s")
                print(f"  Max Score  : {max_score}")
                print(f"  Answer     : {data.get('answer', '')[:120].strip()}...")
                
            else:
                print(f"  Error      : HTTP {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"  Request failed: {e}")
            
    print("\n" + "-" * 60)
    print(f"Total Evaluation Time: {total_time:.3f}s")
    print("Note: Running this script again immediately will be extremely fast due to the Query Cache!")

if __name__ == "__main__":
    target_repo = sys.argv[1] if len(sys.argv) > 1 else REPO_NAME
    evaluate(target_repo)
