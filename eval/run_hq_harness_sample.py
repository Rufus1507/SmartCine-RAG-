"""
eval/run_hq_harness_sample.py — Bản rút gọn để verify rate limiter (Task 5).

Chạy 5 câu từ hq_questions_sample.json, log timestamp trước/sau mỗi cặp
(intent_chain + answer_chain) để xác nhận khoảng cách ~5s giữa các lần gọi.
"""
import os
import sys
import json
import time
import logging

# ── UTF-8 & flush ─────────────────────────────────────────────────────────────
sys.stdout.reconfigure(encoding="utf-8")
import builtins
def print(*args, **kwargs):
    builtins.print(*args, flush=True, **kwargs)

# ── Logging: hiển thị DEBUG của rate_limiter ──────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
# Tắt log ồn ào của thư viện ngoài
for noisy in ("httpx", "httpcore", "openai", "sentence_transformers", "faiss"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ── Path setup ────────────────────────────────────────────────────────────────
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

from chatbot.data_loader import load_data, load_faiss_index, load_embedder_model, load_keyword_dict, load_aliases
from chatbot.llm_client import get_llm_client
from chatbot.chains.rag_chain import run_rag_pipeline
from chatbot.config import GEMINI_DEFAULT_KEY, GEMINI_DEFAULT_MODEL

def main():
    print("🎬 Initializing (sample run — 5 câu)...")
    start_init = time.time()

    try:
        df = load_data()
        keyword_dict = load_keyword_dict()
        aliases_dict = load_aliases()
        faiss_index = load_faiss_index()
        embedder_model = load_embedder_model()

        llm = get_llm_client(
            provider="Gemini API",
            api_key=GEMINI_DEFAULT_KEY,
            model_name=GEMINI_DEFAULT_MODEL,
        )
        print(f"✔️  Init done in {time.time() - start_init:.2f}s | model={GEMINI_DEFAULT_MODEL}")
    except Exception as e:
        print(f"❌ Init failed: {e}")
        sys.exit(1)

    questions_path = os.path.join(workspace_dir, "eval", "hq_questions_sample.json")
    results_path   = os.path.join(workspace_dir, "eval", "hq_sample_results.json")

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"📋 Loaded {len(questions)} sample questions.\n")

    results = []
    prev_call_ts = None  # Timestamp của lần gọi LLM cuối cùng

    for idx, q in enumerate(questions):
        q_id   = q["id"]
        query  = q["question"]

        print(f"── [{idx+1}/{len(questions)}] {q_id}: '{query}'")
        t_before = time.monotonic()
        wall_before = time.strftime("%H:%M:%S")

        if prev_call_ts is not None:
            gap = t_before - prev_call_ts
            print(f"   ⏱  Khoảng cách kể từ lần gọi LLM trước: {gap:.2f}s")

        try:
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
                debug=True,
            )

            answer_result, filtered_df, intent, filters, detected, trace = res

            if not isinstance(answer_result, str):
                try:
                    answer_result = "".join(list(answer_result))
                except Exception:
                    answer_result = str(answer_result)

            movies = []
            if not filtered_df.empty:
                for _, row in filtered_df.iterrows():
                    movies.append({"title": row.get("Title"), "imdb_id": row.get("imdb_id")})

            t_after = time.monotonic()
            prev_call_ts = t_after
            duration = t_after - t_before

            print(f"   ✅ done | wall={wall_before} | elapsed={duration:.2f}s | intent={intent}")
            results.append({
                "id": q_id, "question": query,
                "difficulty": q.get("difficulty"), "category": q.get("category"),
                "answer_result": answer_result, "intent": intent,
                "movies": movies, "elapsed_s": round(duration, 2),
            })

        except Exception as e:
            t_after = time.monotonic()
            prev_call_ts = t_after
            duration = t_after - t_before
            print(f"   ❌ FAILED in {duration:.2f}s: {e}")
            results.append({
                "id": q_id, "question": query,
                "error": str(e), "elapsed_s": round(duration, 2),
            })

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Saved {len(results)} results → {results_path}")
    errors = sum(1 for r in results if "error" in r)
    print(f"✅ {len(results) - errors}/{len(results)} thành công | ❌ {errors} lỗi")

if __name__ == "__main__":
    main()
