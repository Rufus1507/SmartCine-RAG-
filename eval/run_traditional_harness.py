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

import argparse

def main():
    parser = argparse.ArgumentParser(description="Traditional RAG Harness for evaluation")
    parser.add_argument("--start", type=int, default=0, help="Index of question to start from (0-based)")
    parser.add_argument("--limit", type=int, default=None, help="Number of questions to run")
    parser.add_argument("--dry-run", action="store_true", help="Print questions without running")
    args = parser.parse_args()

    print("🎬 [Traditional RAG Harness] Khởi tạo hệ thống...")
    start_init = time.time()
    
    try:
        df = load_data()
        embedder_model = load_embedder_model()
        
        # Tải chỉ mục FAISS truyền thống thực tế (traditional_context.index)
        traditional_index_path = os.path.join(workspace_dir, "data", "traditional_context.index")
        if not os.path.exists(traditional_index_path):
            raise FileNotFoundError(f"Chỉ mục FAISS RAG truyền thống chưa được tạo tại {traditional_index_path}")
        import faiss
        traditional_index = faiss.read_index(traditional_index_path)
        
        # Load LLM client
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
        
    questions_path = os.path.join(workspace_dir, "eval", "hq_questions.json")
    results_path = os.path.join(workspace_dir, "eval", "traditional_results_raw.json")
    
    if not os.path.exists(questions_path):
        print(f"❌ Không tìm thấy bộ câu hỏi tại {questions_path}")
        sys.exit(1)
        
    with open(questions_path, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
        
    start_idx = args.start
    end_idx = len(all_questions) if args.limit is None else min(start_idx + args.limit, len(all_questions))
    questions = all_questions[start_idx:end_idx]
    
    print(f"📋 Đã tải {len(all_questions)} câu hỏi kiểm thử. Sẽ chạy {len(questions)} câu từ vị trí {start_idx} đến {end_idx-1}.")
    
    if args.dry_run:
        print("🔍 Chế độ Dry-run. Danh sách câu hỏi sẽ chạy:")
        for idx, q in enumerate(questions):
            real_idx = start_idx + idx
            print(f"  [{real_idx+1}] {q['id']}: {q['question']}")
        return

    results = []
    if start_idx > 0 and os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
                results = existing_results[:start_idx]
                print(f"💾 Đã nạp {len(results)} kết quả hiện có để chạy tiếp.")
        except Exception as e:
            print(f"⚠️ Không thể nạp kết quả cũ: {e}. Chạy lại từ đầu.")
            results = []
            
    for idx, q in enumerate(questions):
        real_idx = start_idx + idx
        q_id = q["id"]
        query = q["question"]
        print(f"👉 [Traditional RAG] Đang chạy [{real_idx+1}/{len(all_questions)}] {q_id}: '{query}' ...")
        
        t_start = time.time()
        try:
            # Chạy RAG truyền thống
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
                        "title": row.get("Title"),
                        "imdb_id": row.get("imdb_id"),
                        "final_context": row.get("final_context")
                    })
                    
            duration = time.time() - t_start
            results.append({
                "id": q_id,
                "question": query,
                "difficulty": q.get("difficulty"),
                "category": q.get("category"),
                "answer_result": answer_result,
                "movies": movies,
                "latency_s": round(duration, 2)
            })
            print(f"   [{real_idx+1}/{len(all_questions)}] {q_id} hoàn thành trong {duration:.2f}s")
            
        except Exception as e:
            duration = time.time() - t_start
            print(f"❌ [{real_idx+1}/{len(all_questions)}] {q_id} LỖI trong {duration:.2f}s: {e}")
            results.append({
                "id": q_id,
                "question": query,
                "difficulty": q.get("difficulty"),
                "category": q.get("category"),
                "error": str(e),
                "answer_result": f"Error during traditional pipeline execution: {e}",
                "movies": [],
                "latency_s": round(duration, 2)
            })
            
        # Lưu kết quả tạm thời sau mỗi câu
        try:
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as save_err:
            print(f"⚠️ Không thể lưu kết quả tạm thời: {save_err}")
            
    print(f"\n💾 Đã chạy xong và lưu toàn bộ kết quả thô của RAG truyền thống vào {results_path}")

if __name__ == "__main__":
    main()
