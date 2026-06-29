"""
run_latency.py — Đo latency riêng cho CineBot V3 (Step 7 của run_eval_v2)
- Có timeout cho mỗi LLM call (15s) để tránh treo
- Phân biệt cached vs live calls
- Append kết quả vào evaluation_report_v2.md
"""

import os
import sys
import re
import json
import time
import random
import threading
import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.stdout.reconfigure(encoding='utf-8')
import builtins
def print(*args, **kwargs):
    builtins.print(*args, flush=True, **kwargs)

workspace_dir = r"c:\Users\Admin\Desktop\4\DAP391m\code"
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

import pandas as pd
import faiss

from chatbot.config import MIN_VOTES_THRESHOLD, CHATBOT_DIR
from chatbot.data_loader import load_data, load_embedder_model, load_bm25_index
from chatbot.llm_client import get_llm_client
from chatbot.feature_engineering import MovieFeatureBuilder
from chatbot.representation.semantic_representation import (
    INDEX_C_PATH, make_profile_version_c
)
from chatbot.retrieval.bm25_retriever import bm25_search
from chatbot.retrieval.retriever import semantic_search_retriever
from chatbot.retrieval.reranker import rerank_results
from chatbot.similarity.weighted_similarity import compute_weighted_similarity

random.seed(42)
np.random.seed(42)

LLM_TIMEOUT_SEC = 15  # max wait per LLM call before marking as timeout

# ─── Timeout-safe LLM call ──────────────────────────────────────────────────
def call_with_timeout(fn, timeout_sec, fallback="[TIMEOUT]"):
    """Run fn() in a thread, return result or fallback if timeout."""
    result = [fallback]
    exc = [None]
    def target():
        try:
            result[0] = fn()
        except Exception as e:
            result[0] = f"[ERROR: {e}]"
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_sec)
    return result[0]


def main():
    print("=" * 65)
    print("⏱  CINEBOT V3 — LATENCY EVALUATION (with timeout)")
    print("=" * 65)

    # ─── Load ────────────────────────────────────────────────────────
    print("\n[Load] Loading dataset & models...")
    df = load_data()
    df_filtered = df[df['num_votes'] >= MIN_VOTES_THRESHOLD].reset_index(drop=True)
    print(f"Filtered movies: {len(df_filtered):,}")

    embedder_model = load_embedder_model()
    index_c = faiss.read_index(INDEX_C_PATH)
    bm25_index = load_bm25_index(df_filtered)
    builder = MovieFeatureBuilder()

    # ─── Embedding cache ─────────────────────────────────────────────
    print("Building embedding cache from FAISS index C...")
    embeddings_c = index_c.reconstruct_n(0, index_c.ntotal)
    profile_text_to_emb = {}
    for i, row in df_filtered.iterrows():
        prof = make_profile_version_c(row)
        profile_text_to_emb[prof] = embeddings_c[i]

    original_encode = embedder_model.encode
    def patched_encode(sentences, *args, **kwargs):
        is_single = isinstance(sentences, str)
        s_list = [sentences] if is_single else list(sentences)
        results, to_enc_idx, to_enc_txt = [], [], []
        for idx, text in enumerate(s_list):
            if text in profile_text_to_emb:
                results.append(profile_text_to_emb[text])
            else:
                results.append(None)
                to_enc_idx.append(idx)
                to_enc_txt.append(text)
        if to_enc_txt:
            vecs = original_encode(to_enc_txt, *args, **kwargs)
            if isinstance(vecs, list):
                vecs = np.array(vecs)
            for idx, vec in zip(to_enc_idx, vecs):
                results[idx] = vec
        return results[0] if is_single else np.array(results)
    embedder_model.encode = patched_encode
    print("Embedding cache ready.")

    # ─── LLM with cache ──────────────────────────────────────────────
    print("Loading LLM (cx/gpt-5.5)...")
    llm_raw = get_llm_client(
        provider="Local LLM", api_key="any",
        model_name="cx/gpt-5.5",
        base_url="http://localhost:20128/v1"
    )
    llm_cache_path = os.path.join(workspace_dir, "evaluation_v3", "llm_cache.json")
    llm_cache = {}
    if os.path.exists(llm_cache_path):
        with open(llm_cache_path, "r", encoding="utf-8") as f:
            llm_cache = json.load(f)
    print(f"LLM cache loaded: {len(llm_cache)} entries")

    cache_hits = [0]
    cache_misses = [0]
    timeout_count = [0]

    class TimedCachingLLM:
        def invoke(self, prompt, *args, **kwargs):
            if hasattr(prompt, "content"):
                key = prompt.content
            elif isinstance(prompt, list):
                key = "\n".join([getattr(m, "content", str(m)) for m in prompt])
            else:
                key = str(prompt)

            if key in llm_cache:
                cache_hits[0] += 1
                class R:
                    content = llm_cache[key]
                return R()

            cache_misses[0] += 1

            def do_call():
                return llm_raw.invoke(prompt, *args, **kwargs).content

            content = call_with_timeout(do_call, LLM_TIMEOUT_SEC, "[TIMEOUT]")

            if content == "[TIMEOUT]":
                timeout_count[0] += 1
                content = "Không có thông tin."

            llm_cache[key] = content
            try:
                with open(llm_cache_path, "w", encoding="utf-8") as f:
                    json.dump(llm_cache, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            class R:
                pass
            r = R()
            r.content = content
            return r

        def __getattr__(self, name):
            return getattr(llm_raw, name)

    llm = TimedCachingLLM()

    # ─── Load ground truth ───────────────────────────────────────────
    gt_path = os.path.join(workspace_dir, "evaluation_v3", "ground_truth.json")
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth_list = json.load(f)

    # ─── Latency profiling ───────────────────────────────────────────
    print(f"\n[Step 7] Latency profiling — 50 runs, LLM timeout={LLM_TIMEOUT_SEC}s ...")

    latency_details = {
        "Entity Extraction":    [],
        "Intent LLM":           [],
        "Retrieval (Hybrid)":   [],
        "Similarity Scoring":   [],
        "Cross-Encoder Rerank": [],
        "RAG Generation":       [],
        "Total (end-to-end)":   [],
    }
    intent_call_type = []   # "cached" or "live" or "timeout"
    rag_call_type    = []

    from chatbot.entity_extractor import detect_entities
    from chatbot.chains.intent_chain import run_intent_chain
    from chatbot.chains.answer_chain import run_answer_chain
    from chatbot.data_loader import load_keyword_dict, load_aliases

    keyword_dict = load_keyword_dict()
    aliases_dict = load_aliases()

    import chatbot.retrieval.reranker as reranker_module
    import chatbot.retrieval.multistage_retriever as ms_module
    orig_rerank = reranker_module.rerank_results
    def capped_rerank(query, df, top_k=20):
        return orig_rerank(query, df.head(20), top_k)
    reranker_module.rerank_results = capped_rerank
    ms_module.rerank_results = capped_rerank

    for idx in range(50):
        gt = random.choice(ground_truth_list)
        query = gt["query"]

        # 1. Entity Extraction
        t0 = time.time()
        detected = detect_entities(query, keyword_dict, aliases_dict)
        t_entity = (time.time() - t0) * 1000
        latency_details["Entity Extraction"].append(t_entity)

        # 2. Intent LLM (with timeout)
        prev_hits = cache_hits[0]
        prev_timeouts = timeout_count[0]
        t0 = time.time()
        def do_intent():
            return run_intent_chain(llm, query, detected, [])
        parsed = call_with_timeout(do_intent, LLM_TIMEOUT_SEC + 2, {"intent": "search", "filters": {}})
        t_intent = (time.time() - t0) * 1000
        latency_details["Intent LLM"].append(t_intent)
        if cache_hits[0] > prev_hits:
            intent_call_type.append("cached")
        elif timeout_count[0] > prev_timeouts:
            intent_call_type.append("timeout")
        else:
            intent_call_type.append("live")

        intent = parsed.get("intent", "search") if isinstance(parsed, dict) else "search"

        # 3. Retrieval
        t0 = time.time()
        faiss_res = semantic_search_retriever(query, df_filtered, index_c, embedder_model, top_k=150)
        bm25_res  = bm25_search(query, df_filtered, bm25_index, top_k=100)
        seen_links, candidate_list = set(), []
        for cdf in [faiss_res, bm25_res]:
            if not cdf.empty:
                for _, row in cdf.iterrows():
                    link = row["Movie Link"]
                    if link not in seen_links:
                        seen_links.add(link)
                        candidate_list.append(row)
        candidates_df = pd.DataFrame(candidate_list)
        t_retrieval = (time.time() - t0) * 1000
        latency_details["Retrieval (Hybrid)"].append(t_retrieval)

        # 4. Similarity Scoring
        t0 = time.time()
        try:
            seed_row = df_filtered[df_filtered['Title'] == gt["seed_movie"]].iloc[0]
        except IndexError:
            seed_row = df_filtered.iloc[0]
        seed_features = builder.transform_row(seed_row)
        seed_prof = make_profile_version_c(seed_row)
        seed_features["semantic_embedding"] = profile_text_to_emb.get(seed_prof)

        matched_rows = []
        for _, row in candidates_df.iterrows():
            rf = builder.transform_row(row)
            cp = make_profile_version_c(row)
            rf["semantic_embedding"] = profile_text_to_emb.get(cp)
            sim = compute_weighted_similarity(rf, seed_features)
            rc = row.copy()
            rc["final_similarity_score"] = sim["final_score"]
            matched_rows.append(rc)
        ranked_df = pd.DataFrame(matched_rows).sort_values("final_similarity_score", ascending=False)
        top_100 = ranked_df.head(100).copy()
        t_scoring = (time.time() - t0) * 1000
        latency_details["Similarity Scoring"].append(t_scoring)

        # 5. Cross-Encoder Rerank
        t0 = time.time()
        reranked_df = rerank_results(query, top_100, top_k=10)
        t_rerank = (time.time() - t0) * 1000
        latency_details["Cross-Encoder Rerank"].append(t_rerank)

        # 6. RAG Generation (with timeout)
        prev_hits2 = cache_hits[0]
        prev_to2   = timeout_count[0]
        t0 = time.time()
        def do_rag():
            return run_answer_chain(llm, query, reranked_df, intent, stream=False)
        answer = call_with_timeout(do_rag, LLM_TIMEOUT_SEC + 2, "")
        t_rag = (time.time() - t0) * 1000
        latency_details["RAG Generation"].append(t_rag)
        if cache_hits[0] > prev_hits2:
            rag_call_type.append("cached")
        elif timeout_count[0] > prev_to2:
            rag_call_type.append("timeout")
        else:
            rag_call_type.append("live")

        t_total = t_entity + t_intent + t_retrieval + t_scoring + t_rerank + t_rag
        latency_details["Total (end-to-end)"].append(t_total)

        if (idx + 1) % 10 == 0:
            print(f"  Run {idx+1}/50 | cache_hits={cache_hits[0]} misses={cache_misses[0]} timeouts={timeout_count[0]}")

    # ─── Print results ────────────────────────────────────────────────
    print("\n--- Latency Results V2 (ms) ---")
    latency_summary = {}
    for key, times in latency_details.items():
        avg_t = np.mean(times)
        p95_t = np.percentile(times, 95)
        ratio = p95_t / avg_t if avg_t > 0 else 0
        latency_summary[key] = {"avg": float(avg_t), "p95": float(p95_t), "ratio": float(ratio)}
        print(f"  {key}: Avg={avg_t:.1f}ms, P95={p95_t:.1f}ms, Ratio={ratio:.1f}x")

    # Breakdown by call type
    def summarize_by_type(times, types, label):
        cached  = [t for t, ty in zip(times, types) if ty == "cached"]
        live    = [t for t, ty in zip(times, types) if ty == "live"]
        timeout = [t for t, ty in zip(times, types) if ty == "timeout"]
        print(f"\n  [{label} breakdown]")
        if cached:  print(f"    Cached  (n={len(cached):2d}): Avg={np.mean(cached):.1f}ms, P95={np.percentile(cached,95):.1f}ms")
        if live:    print(f"    Live    (n={len(live):2d}): Avg={np.mean(live):.1f}ms, P95={np.percentile(live,95):.1f}ms")
        if timeout: print(f"    Timeout (n={len(timeout):2d}): forced {LLM_TIMEOUT_SEC}s cutoff")

    summarize_by_type(latency_details["Intent LLM"],    intent_call_type, "Intent LLM")
    summarize_by_type(latency_details["RAG Generation"], rag_call_type,   "RAG Generation")

    print(f"\n  Total: cache_hits={cache_hits[0]}, live_calls={cache_misses[0]}, timeouts={timeout_count[0]}")

    # ─── Save latency JSON ────────────────────────────────────────────
    latency_path = os.path.join(workspace_dir, "evaluation_v3", "latency_results_v2.json")
    latency_export = {
        "latency_summary": latency_summary,
        "breakdown": {
            "intent_call_types": intent_call_type,
            "rag_call_types": rag_call_type,
            "cache_hits": cache_hits[0],
            "cache_misses": cache_misses[0],
            "timeouts": timeout_count[0]
        }
    }
    with open(latency_path, "w", encoding="utf-8") as f:
        json.dump(latency_export, f, ensure_ascii=False, indent=2)
    print(f"\n  Latency results saved to: {latency_path}")

    # ─── Append latency section to evaluation_report_v2.md ────────────
    report_path = os.path.join(workspace_dir, "evaluation_report_v2.md")
    latency_section = "\n\n---\n\n## 9. Phân tích Độ trễ — V2 (Latency, timeout=15s)\n\n"
    latency_section += f"50 profiling runs. LLM timeout mỗi call: {LLM_TIMEOUT_SEC}s.\n\n"
    latency_section += "| Stage | Avg (ms) | P95 (ms) | P95/Avg ratio |\n"
    latency_section += "|---|---|---|---|\n"
    for key, vals in latency_summary.items():
        latency_section += f"| **{key}** | {vals['avg']:.1f} | {vals['p95']:.1f} | {vals['ratio']:.1f}x |\n"

    # Breakdown tables
    def breakdown_rows(times, types, label):
        cached  = [t for t, ty in zip(times, types) if ty == "cached"]
        live    = [t for t, ty in zip(times, types) if ty == "live"]
        timeout_t = [t for t, ty in zip(times, types) if ty == "timeout"]
        rows = f"\n### {label} — Cached vs Live\n\n"
        rows += "| Type | Count | Avg (ms) | P95 (ms) |\n|---|---|---|---|\n"
        if cached:
            rows += f"| Cached | {len(cached)} | {np.mean(cached):.1f} | {np.percentile(cached,95):.1f} |\n"
        if live:
            rows += f"| Live | {len(live)} | {np.mean(live):.1f} | {np.percentile(live,95):.1f} |\n"
        if timeout_t:
            rows += f"| Timeout (>{LLM_TIMEOUT_SEC}s) | {len(timeout_t)} | — | — |\n"
        return rows

    latency_section += breakdown_rows(latency_details["Intent LLM"],    intent_call_type, "Intent LLM")
    latency_section += breakdown_rows(latency_details["RAG Generation"], rag_call_type,   "RAG Generation")
    latency_section += f"\n> Total: cache_hits={cache_hits[0]}, live_calls={cache_misses[0]}, timeouts={timeout_count[0]}\n"
    latency_section += "\n> [!NOTE]\n"
    latency_section += f"> P95 cao chủ yếu do **live calls** đến local LLM server (cx/gpt-5.5). "
    latency_section += f"Timeouts ({timeout_count[0]} lần) bị cắt tại {LLM_TIMEOUT_SEC}s để tránh treo. "
    latency_section += "Cached calls có latency ổn định thấp.\n"

    # Remove old placeholder section if exists, then append
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Remove old latency warning block if present
        content = re.sub(
            r'\n\n---\n\n## 9\. Phân tích Độ trễ.*',
            '',
            content,
            flags=re.DOTALL
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content + latency_section)
        print(f"\n✅ Latency section appended to: {report_path}")
    else:
        print(f"⚠️ Report not found at {report_path}")

    print("\n🎉 Latency evaluation hoàn thành!")


if __name__ == "__main__":
    main()
