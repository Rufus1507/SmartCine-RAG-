import os
import sys
import json
import time

# Ensure console output is in UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Force flush on print
import builtins
def print(*args, **kwargs):
    builtins.print(*args, flush=True, **kwargs)

# Add workspace directory to path
workspace_dir = r"c:\Users\Admin\Desktop\4\DAP391m\code"
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

from chatbot.data_loader import load_data, load_faiss_index, load_embedder_model, load_keyword_dict, load_aliases
from chatbot.llm_client import get_llm_client
from chatbot.chains.rag_chain import run_rag_pipeline
from chatbot.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

def main():
    print("🎬 Startup initialization...")
    start_init = time.time()
    
    # Initialize exactly the same objects as the startup path in Q2
    try:
        df = load_data()
        keyword_dict = load_keyword_dict()
        aliases_dict = load_aliases()
        faiss_index = load_faiss_index()
        embedder_model = load_embedder_model()
        
        # Load local LLM client using same configuration logic as app startup
        llm = get_llm_client(
            provider="Local LLM",
            api_key=LLM_API_KEY,
            model_name=LLM_MODEL,
            base_url=LLM_BASE_URL
        )
        print(f"✔️ Initialization completed in {time.time() - start_init:.2f}s")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        sys.exit(1)

    # 2. Load test questions
    questions_path = os.path.join(workspace_dir, "eval", "test_questions.json")
    results_path = os.path.join(workspace_dir, "eval", "results_raw.json")
    
    if not os.path.exists(questions_path):
        print(f"❌ Test questions file not found at {questions_path}")
        sys.exit(1)
        
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
        
    print(f"📋 Loaded {len(questions)} test questions.")
    
    results = []
    
    # 3. Execution Loop
    for idx, q in enumerate(questions):
        q_id = q["id"]
        query = q["question"]
        print(f"👉 Running [{idx+1}/{len(questions)}] {q_id}: '{query}' ...")
        
        t_start = time.time()
        try:
            # Execute run_rag_pipeline with debug=True, empty chat_history, and empty last_filters
            res = run_rag_pipeline(
                llm=llm,
                user_input=query,
                df=df,
                keyword_dict=keyword_dict,
                aliases_dict=aliases_dict,
                faiss_index=faiss_index,
                embedder_model=embedder_model,
                chat_history=[],
                last_filters={},
                stream=False,
                debug=True
            )
            
            # Unpack 6-item tuple
            answer_result, filtered_df, intent, filters, detected, trace = res
            
            # Consume stream/generator if needed
            if not isinstance(answer_result, str):
                try:
                    answer_result = "".join(list(answer_result))
                except Exception:
                    answer_result = str(answer_result)
                    
            # Convert filtered_df to list of dicts with only: title, imdb_id
            movies = []
            if not filtered_df.empty:
                for _, row in filtered_df.iterrows():
                    movies.append({
                        "title": row.get("Title"),
                        "imdb_id": row.get("imdb_id")
                    })
                    
            results.append({
                "id": q_id,
                "question": query,
                "difficulty": q.get("difficulty"),
                "category": q.get("category"),
                "answer_result": answer_result,
                "trace": trace,
                "movies": movies
            })
            duration = time.time() - t_start
            print(f"   [{idx+1}/{len(questions)}] {q_id} done in {duration:.2f}s")
            
        except Exception as e:
            duration = time.time() - t_start
            print(f"❌ [{idx+1}/{len(questions)}] {q_id} FAILED in {duration:.2f}s: {e}")
            results.append({
                "id": q_id,
                "question": query,
                "difficulty": q.get("difficulty"),
                "category": q.get("category"),
                "error": str(e),
                "answer_result": f"Error during pipeline execution: {e}",
                "trace": None,
                "movies": []
            })
            
    # 5. Save raw results to eval/results_raw.json
    try:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Saved raw results to {results_path}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

if __name__ == "__main__":
    main()
