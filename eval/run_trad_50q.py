import os
import sys
import json
import time
import faiss

# Project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

sys.stdout.reconfigure(encoding='utf-8')

import builtins
def print(*args, **kwargs):
    builtins.print(*args, flush=True, **kwargs)

from chatbot.data_loader import load_data, load_embedder_model
from chatbot.llm_client import get_llm_client
from chatbot.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
from eval.traditional_rag import run_traditional_rag_pipeline

def main():
    print("🎬 [Traditional RAG - 50Q Benchmark] Khởi tạo hệ thống...")
    start_init = time.time()

    try:
        df = load_data()
        embedder_model = load_embedder_model()

        # Tải chỉ mục FAISS truyền thống (traditional_context.index)
        traditional_index_path = os.path.join(ROOT_DIR, "data", "traditional_context.index")
        if not os.path.exists(traditional_index_path):
            raise FileNotFoundError(
                f"⚠️  Chỉ mục FAISS RAG truyền thống chưa được tạo.\n"
                f"   Hãy chạy: python eval/generate_traditional_embeddings.py\n"
                f"   Đường dẫn thiếu: {traditional_index_path}"
            )
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

    questions_path = os.path.join(ROOT_DIR, "eval", "benchmark_subset_50.json")
    results_path   = os.path.join(ROOT_DIR, "eval", "traditional_50q_results.json")

    if not os.path.exists(questions_path):
        print(f"❌ Không tìm thấy bộ câu hỏi tại {questions_path}")
        sys.exit(1)

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"📋 Đã tải {len(questions)} câu hỏi từ benchmark_subset_50.json")
    results = []
    total = len(questions)

    for idx, q in enumerate(questions):
        q_id  = q["id"]
        query = q["question"]
        print(f"👉 [{idx+1:02d}/{total}] {q_id} (level={q['level']}, {q['difficulty']}): {query[:65]}...")

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
                        "title":    row.get("Title"),
                        "imdb_id":  row.get("imdb_id"),
                        "rating":   row.get("Rating"),
                        "year":     row.get("Year"),
                        "genres":   row.get("genres"),
                        "directors": row.get("directors"),
                    })

            results.append({
                "id":            q_id,
                "question":      query,
                "level":         q.get("level"),
                "difficulty":    q.get("difficulty"),
                "category":      q.get("category"),
                "answer_result": answer_result,
                "movies":        movies,
                "latency_s":     round(time.time() - t_start, 2)
            })
            print(f"   ✔️  Xong trong {time.time() - t_start:.2f}s")

        except Exception as e:
            duration = time.time() - t_start
            print(f"   ❌ Lỗi trong {duration:.2f}s: {e}")
            results.append({
                "id":            q_id,
                "question":      query,
                "level":         q.get("level"),
                "difficulty":    q.get("difficulty"),
                "category":      q.get("category"),
                "error":         str(e),
                "answer_result": f"Error: {e}",
                "movies":        [],
                "latency_s":     round(duration, 2)
            })

    # Lưu kết quả
    try:
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Đã lưu {len(results)} kết quả vào {results_path}")
    except Exception as e:
        print(f"❌ Không thể lưu kết quả: {e}")

if __name__ == "__main__":
    main()
