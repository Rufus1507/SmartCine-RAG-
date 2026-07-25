import os
import sys
import json
import time

# Thêm thư mục gốc vào path
eval_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(eval_dir)
sys.path.append(workspace_dir)

# Đảm bảo in UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Force flush on print
import builtins
def print(*args, **kwargs):
    builtins.print(*args, flush=True, **kwargs)

from chatbot.data_loader import load_data, load_faiss_index, load_embedder_model
from chatbot.llm_client import get_llm_client
from chatbot.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
from eval.traditional_rag import run_traditional_rag_pipeline

def main():
    print("🎬 [Traditional RAG Harness - test_questions] Khởi tạo hệ thống...")
    start_init = time.time()

    try:
        df = load_data()
        embedder_model = load_embedder_model()
        traditional_index_path = os.path.join(workspace_dir, "data", "traditional_context.index")
        if not os.path.exists(traditional_index_path):
            raise FileNotFoundError(f"Chỉ mục FAISS RAG truyền thống chưa được tạo tại {traditional_index_path}")
        import faiss
        traditional_index = faiss.read_index(traditional_index_path)
        llm = get_llm_client(
            provider="Local LLM",
            api_key=LLM_API_KEY,
            model_name=LLM_MODEL,
            base_url=LLM_BASE_URL
        )
        print(f"✔️ Hoàn thành khởi tạo trong {time.time() - start_init:.2f}s")
    except Exception as e:
        print(f"❌ Khởi tạo thất bại: {e}")
        sys.exit(1)

    questions_path = os.path.join(workspace_dir, "eval", "test_questions.json")
    results_path   = os.path.join(workspace_dir, "eval", "traditional_test_results_raw.json")

    if not os.path.exists(questions_path):
        print(f"❌ Không tìm thấy bộ câu hỏi tại {questions_path}")
        sys.exit(1)

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"📋 Đã tải {len(questions)} câu hỏi kiểm thử từ test_questions.json.")
    results = []

    for idx, q in enumerate(questions):
        q_id   = q["id"]
        query  = q["question"]
        print(f"👉 [Traditional RAG] [{idx+1}/{len(questions)}] {q_id}: '{query[:60]}...' ...")

        t_start = time.time()
        try:
            answer_result, retrieved_df = run_traditional_rag_pipeline(
                query=query,
                llm=llm,
                df=df,
                index=traditional_index,
                model=embedder_model,
                top_k=5
            )
            movies = []
            if not retrieved_df.empty:
                for _, row in retrieved_df.iterrows():
                    movies.append({
                        "title":         row.get("Title"),
                        "imdb_id":       row.get("imdb_id"),
                        "rating":        row.get("Rating"),
                        "year":          row.get("Year"),
                        "genres":        row.get("genres"),
                        "directors":     row.get("directors"),
                        "final_context": row.get("final_context")
                    })

            results.append({
                "id":            q_id,
                "question":      query,
                "difficulty":    q.get("difficulty"),
                "category":      q.get("category"),
                "answer_result": answer_result,
                "movies":        movies
            })
            duration = time.time() - t_start
            print(f"   ✔️  [{idx+1}/{len(questions)}] {q_id} hoàn thành trong {duration:.2f}s")

        except Exception as e:
            duration = time.time() - t_start
            print(f"❌ [{idx+1}/{len(questions)}] {q_id} LỖI trong {duration:.2f}s: {e}")
            results.append({
                "id":            q_id,
                "question":      query,
                "difficulty":    q.get("difficulty"),
                "category":      q.get("category"),
                "error":         str(e),
                "answer_result": f"Error: {e}",
                "movies":        []
            })

    try:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Đã lưu kết quả thô vào {results_path}")
    except Exception as e:
        print(f"❌ Không thể lưu kết quả: {e}")

if __name__ == "__main__":
    main()
