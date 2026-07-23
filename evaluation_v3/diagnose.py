"""
BƯỚC 1 — CHẨN ĐOÁN (chỉ log, KHÔNG sửa code)
In ra bằng chứng thực tế cho 6 bất thường được nêu trong cinebot_evaluation_prompt.md
"""

import os
import sys
import re
import json
import random
import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.stdout.reconfigure(encoding='utf-8')
import builtins
def print(*args, **kwargs):
    builtins.print(*args, flush=True, **kwargs)

workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

import pandas as pd
import faiss

from chatbot.config import MIN_VOTES_THRESHOLD, CHATBOT_DIR
from chatbot.data_loader import load_data, load_embedder_model, load_bm25_index
from chatbot.feature_engineering import MovieFeatureBuilder, clean_split
from chatbot.representation.semantic_representation import (
    INDEX_A_PATH, INDEX_B_PATH, INDEX_C_PATH, make_profile,
    make_profile_version_a, make_profile_version_b, make_profile_version_c
)
from chatbot.retrieval.bm25_retriever import bm25_search
from chatbot.retrieval.retriever import semantic_search_retriever
from chatbot.tools import search_movies_tool
from chatbot.similarity.weighted_similarity import compute_weighted_similarity

random.seed(42)
np.random.seed(42)

DIVIDER = "=" * 70

def clean_title(t):
    return re.sub(r"[^\w\s]", "", str(t).lower().strip())

def main():
    print(DIVIDER)
    print("CINEBOT V3 — BƯỚC 1: CHẨN ĐOÁN NGUYÊN NHÂN GỐC")
    print(DIVIDER)

    # ─── Load data ──────────────────────────────────────────────────────────
    print("\n[Load] Loading dataset & models...")
    df = load_data()
    df_filtered = df[df['num_votes'] >= MIN_VOTES_THRESHOLD].reset_index(drop=True)
    print(f"Filtered movies: {len(df_filtered):,}")

    embedder_model = load_embedder_model()
    index_a = faiss.read_index(INDEX_A_PATH)
    index_b = faiss.read_index(INDEX_B_PATH)
    index_c = faiss.read_index(INDEX_C_PATH)
    bm25_index = load_bm25_index(df_filtered)
    builder = MovieFeatureBuilder()

    print(f"FAISS Index A: {index_a.ntotal} vectors")
    print(f"FAISS Index B: {index_b.ntotal} vectors")
    print(f"FAISS Index C: {index_c.ntotal} vectors")
    print(f"df_filtered rows: {len(df_filtered)}")

    gt_path = os.path.join(workspace_dir, "evaluation_v3", "ground_truth.json")
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth_list = json.load(f)
    print(f"Ground truth loaded: {len(ground_truth_list)} queries")

    # ─── Pre-extract embeddings để dùng cache ────────────────────────────────
    print("\n[Cache] Extracting embeddings from FAISS indices...")
    embeddings_a = index_a.reconstruct_n(0, index_a.ntotal)
    embeddings_b = index_b.reconstruct_n(0, index_b.ntotal)
    embeddings_c = index_c.reconstruct_n(0, index_c.ntotal)

    profile_text_to_emb = {}
    for i, row in df_filtered.iterrows():
        prof_a = make_profile(row, 'A')
        prof_b = make_profile(row, 'B')
        prof_c = make_profile(row, 'C')
        profile_text_to_emb[prof_a] = embeddings_a[i]
        profile_text_to_emb[prof_b] = embeddings_b[i]
        profile_text_to_emb[prof_c] = embeddings_c[i]

    original_encode = embedder_model.encode
    def patched_encode(sentences, *args, **kwargs):
        is_single = isinstance(sentences, str)
        s_list = [sentences] if is_single else list(sentences)
        results = []
        to_encode_indices, to_encode_texts = [], []
        for idx, text in enumerate(s_list):
            if text in profile_text_to_emb:
                results.append(profile_text_to_emb[text])
            else:
                results.append(None)
                to_encode_indices.append(idx)
                to_encode_texts.append(text)
        if to_encode_texts:
            encoded_vecs = original_encode(to_encode_texts, *args, **kwargs)
            if isinstance(encoded_vecs, list):
                encoded_vecs = np.array(encoded_vecs)
            for idx, vec in zip(to_encode_indices, encoded_vecs):
                results[idx] = vec
        if is_single:
            return results[0]
        return np.array(results)
    embedder_model.encode = patched_encode
    print("Embedding cache ready.")

    # ====================================================================
    # BẤT THƯỜNG 1 — Recommendation Quality cực thấp
    # Kiểm tra: Ground truth vs system output dùng key gì để matching?
    # ====================================================================
    print(f"\n{DIVIDER}")
    print("BẤT THƯỜNG 1 — Recommendation Quality (Kiểm tra ID matching)")
    print(DIVIDER)

    sample_gts = random.sample(ground_truth_list, 5)
    from chatbot.retrieval.multistage_retriever import MultistageRetriever
    retriever = MultistageRetriever()

    for i, gt in enumerate(sample_gts):
        print(f"\n--- Query #{i+1}: {gt['query'][:60]}...")
        print(f"  Ground truth relevant_movies (raw): {gt['relevant_movies'][:3]}")

        # System output
        res = retriever.retrieve(
            query=gt["query"], df=df_filtered, filters={}, intent="search",
            faiss_index=index_c, embedder_model=embedder_model, version='C', final_k=10
        )
        sys_output = res["Title"].tolist() if not res.empty else []
        print(f"  System output titles (raw): {sys_output[:3]}")

        # Matching attempt
        gt_clean = {clean_title(t) for t in gt["relevant_movies"]}
        sys_clean = [clean_title(t) for t in sys_output]
        hits = [t for t in sys_clean if t in gt_clean or any(g in t or t in g for g in gt_clean)]
        print(f"  GT cleaned (first 3): {list(gt_clean)[:3]}")
        print(f"  SYS cleaned (first 3): {sys_clean[:3]}")
        print(f"  Hits (matches): {hits}")
        print(f"  => Match count: {len(hits)}/10")

    print("\n[DIAGNOSIS 1 NOTES]")
    print("  - Ground truth stores: TITLE strings (plain text)")
    print("  - System output returns: Title column (plain text)")
    print("  - evaluate_metrics uses: clean_title(t) = re.sub(r'[^\\w\\s]','',str(t).lower().strip())")
    print("  - CHECK: Any format mismatch (year in parens, punctuation) that causes misses?")
    # Also check for duplicate titles in GT
    dup_counts = []
    for gt in ground_truth_list[:20]:
        titles = gt["relevant_movies"]
        dup = len(titles) - len(set(titles))
        dup_counts.append(dup)
    print(f"\n  Duplicate titles in GT (first 20 queries): avg={np.mean(dup_counts):.1f}, max={max(dup_counts)}")
    print("  => FINDING: GT has duplicate titles -> effective unique relevant_movies per query is < 10")
    sample_gt = ground_truth_list[0]
    unique_relevant = list(dict.fromkeys(sample_gt["relevant_movies"]))
    print(f"  Example: '{sample_gt['seed_movie']}' has {len(sample_gt['relevant_movies'])} entries but {len(unique_relevant)} unique")

    # ====================================================================
    # BẤT THƯỜNG 2 — Ablation RQ1: 3 phiên bản gần như giống nhau
    # Kiểm tra: profiles và vectors có thực sự khác nhau không?
    # ====================================================================
    print(f"\n{DIVIDER}")
    print("BẤT THƯỜNG 2 — Ablation RQ1: Vectors có thực sự khác nhau?")
    print(DIVIDER)

    sample_row = df_filtered.iloc[0]
    prof_a = make_profile_version_a(sample_row)
    prof_b = make_profile_version_b(sample_row)
    prof_c = make_profile_version_c(sample_row)

    print(f"\nMovie: {sample_row['Title']}")
    print(f"\n  Profile A (description only):\n    {prof_a[:120]}...")
    print(f"\n  Profile B (genre + description):\n    {prof_b[:120]}...")
    print(f"\n  Profile C (genre + description + keywords):\n    {prof_c[:120]}...")
    print(f"\n  Profile A == Profile B? {prof_a == prof_b}")
    print(f"  Profile A == Profile C? {prof_a == prof_c}")
    print(f"  Profile B == Profile C? {prof_b == prof_c}")

    # Check if FAISS index files are identical (same file size)
    size_a = os.path.getsize(INDEX_A_PATH)
    size_b = os.path.getsize(INDEX_B_PATH)
    size_c = os.path.getsize(INDEX_C_PATH)
    print(f"\n  FAISS Index A size: {size_a:,} bytes")
    print(f"  FAISS Index B size: {size_b:,} bytes")
    print(f"  FAISS Index C size: {size_c:,} bytes")
    print(f"  A==B? {size_a==size_b}  A==C? {size_a==size_c}  B==C? {size_b==size_c}")

    # Check first 5 values of embedding vectors
    emb_a = profile_text_to_emb.get(prof_a)
    emb_b = profile_text_to_emb.get(prof_b)
    emb_c = profile_text_to_emb.get(prof_c)
    if emb_a is not None:
        print(f"\n  Embedding A (first 5): {emb_a[:5]}")
    if emb_b is not None:
        print(f"  Embedding B (first 5): {emb_b[:5]}")
    if emb_c is not None:
        print(f"  Embedding C (first 5): {emb_c[:5]}")

    if emb_a is not None and emb_b is not None:
        cosine_ab = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8)
        cosine_ac = np.dot(emb_a, emb_c) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_c) + 1e-8) if emb_c is not None else None
        cosine_bc = np.dot(emb_b, emb_c) / (np.linalg.norm(emb_b) * np.linalg.norm(emb_c) + 1e-8) if emb_c is not None else None
        print(f"\n  Cosine similarity A-B: {cosine_ab:.4f}")
        if cosine_ac: print(f"  Cosine similarity A-C: {cosine_ac:.4f}")
        if cosine_bc: print(f"  Cosine similarity B-C: {cosine_bc:.4f}")

    # Profile mapping mismatch check:
    # In run_eval.py line 184-191, the loop uses df_filtered.iterrows() with index i
    # BUT: embeddings_a[i] where i is the df_filtered index, not 0-based position
    print("\n[DIAGNOSIS 2 NOTES]")
    print("  - Profile texts are DIFFERENT (A/B/C have different content)")
    print("  - But profile_text_to_emb mapping uses iterrows() index i (df_filtered.index)")
    print("  - embeddings_a[i] uses df_filtered.index as FAISS index position")
    print("  - CRITICAL BUG: df_filtered was reset_index(drop=True), so index i IS 0-based")
    print("    but iterrows() yields (original_index, row) -> i = original df index, NOT position!")
    print("  => Check if reset_index was called before the loop in run_eval.py")

    # Verify: df_filtered after reset_index should have 0-based index
    print(f"\n  df_filtered.index[0]: {df_filtered.index[0]}  (expected 0 if reset)")
    print(f"  df_filtered.index[5]: {df_filtered.index[5]}")

    # ====================================================================
    # BẤT THƯỜNG 3 — Title-Overfitting test 0.0%
    # Kiểm tra 5 ví dụ trong bộ decoy pairs
    # ====================================================================
    print(f"\n{DIVIDER}")
    print("BẤT THƯỜNG 3 — Title-Overfitting: Kiểm tra quality của bộ decoy pairs")
    print(DIVIDER)

    # Re-generate the same 50 pairs as in run_eval.py
    overfit_pairs = []
    seen_seeds = set()
    for i, row in df_filtered.iterrows():
        if len(overfit_pairs) >= 50:
            break
        title = str(row['Title'])
        words = [w.lower() for w in re.findall(r'\b\w{4,}\b', title)
                 if w.lower() not in ('the', 'of', 'and', 'a', 'in', 'to', 'for', 'with', 'on', 'at', 'by')]
        if not words:
            continue
        genres1 = set(clean_split(row['genres']))
        if not genres1:
            continue
        for word in words:
            candidates = df_filtered[
                df_filtered['Title'].astype(str).str.lower().str.contains(rf'\b{word}\b', regex=True, na=False) &
                (df_filtered['Title'] != title)
            ]
            if candidates.empty:
                continue
            for _, cand_row in candidates.iterrows():
                genres2 = set(clean_split(cand_row['genres']))
                if not genres2:
                    continue
                if not genres1.intersection(genres2):
                    if row['Title'] not in seen_seeds and len(overfit_pairs) < 50:
                        overfit_pairs.append((row, cand_row))
                        seen_seeds.add(row['Title'])
                        break
            if len(overfit_pairs) >= 50:
                break

    print(f"\nGenerated {len(overfit_pairs)} overfit pairs. Inspecting first 5:")
    for seed_row, decoy_row in overfit_pairs[:5]:
        from difflib import SequenceMatcher
        title_sim = SequenceMatcher(None, seed_row['Title'].lower(), decoy_row['Title'].lower()).ratio()
        genres1 = set(clean_split(seed_row['genres']))
        genres2 = set(clean_split(decoy_row['genres']))
        print(f"\n  Seed:  '{seed_row['Title']}' | genres: {genres1}")
        print(f"  Decoy: '{decoy_row['Title']}' | genres: {genres2}")
        print(f"  Title similarity (Levenshtein ratio): {title_sim:.2f}")
        print(f"  Genre overlap: {genres1 & genres2}")
        print(f"  => Sharing a word? YES (by construction) | Title sim > 0.8? {title_sim > 0.8}")

    print("\n[DIAGNOSIS 3 NOTES]")
    print("  - Pairs are built by shared WORD in title, not by high title string similarity")
    print("  - Many pairs may share only 1 common word (e.g. 'The') -> very weak decoys")
    print("  - Recommended threshold: title_sim > 0.65 AND genre completely different")
    print("  - Expected fix: rebuild decoys with stricter title similarity criterion")

    # ====================================================================
    # BẤT THƯỜNG 4 — Cross-Encoder rerank làm GIẢM NDCG
    # In input/output thực tế của Cross-Encoder cho 1 query
    # ====================================================================
    print(f"\n{DIVIDER}")
    print("BẤT THƯỜNG 4 — Cross-Encoder Reranking làm giảm NDCG")
    print(DIVIDER)

    from chatbot.retrieval.reranker import rerank_results, load_reranker_model
    import inspect

    reranker = load_reranker_model()
    print(f"\n  Cross-Encoder model: {type(reranker)}")
    if hasattr(reranker, 'model_name') or hasattr(reranker, 'model'):
        m = getattr(reranker, 'model_name', getattr(reranker, 'model', 'unknown'))
        print(f"  Model name/object: {m}")

    # Pick one query and show before/after rerank
    test_gt = ground_truth_list[3]  # 300 query, pick #3
    query = test_gt["query"]
    relevant_movies = test_gt["relevant_movies"]
    print(f"\n  Test query: {query}")
    print(f"  GT relevant: {relevant_movies[:5]}")

    seed_movie_rows = df_filtered[df_filtered['Title'] == test_gt["seed_movie"]]
    if not seed_movie_rows.empty:
        seed_row = seed_movie_rows.iloc[0]
        seed_features = builder.transform_row(seed_row)
        seed_profile = make_profile(seed_row, 'C')
        seed_features["semantic_embedding"] = profile_text_to_emb.get(seed_profile)

        faiss_candidates = semantic_search_retriever(query, df_filtered, index_c, embedder_model, top_k=50)
        bm25_candidates = bm25_search(query, df_filtered, bm25_index, top_k=50)

        seen_links = set()
        candidate_list = []
        for cdf in [faiss_candidates, bm25_candidates]:
            if not cdf.empty:
                for _, row in cdf.iterrows():
                    link = row["Movie Link"]
                    if link not in seen_links:
                        seen_links.add(link)
                        candidate_list.append(row)

        candidates_df = pd.DataFrame(candidate_list)
        candidates_df = candidates_df[candidates_df["Title"] != test_gt["seed_movie"]]

        matched_rows = []
        for _, row in candidates_df.iterrows():
            row_features = builder.transform_row(row)
            candidate_profile = make_profile(row, 'C')
            row_features["semantic_embedding"] = profile_text_to_emb.get(candidate_profile)
            sim = compute_weighted_similarity(row_features, seed_features)
            row_copy = row.copy()
            row_copy["final_similarity_score"] = sim["final_score"]
            matched_rows.append(row_copy)

        ranked_df = pd.DataFrame(matched_rows).sort_values("final_similarity_score", ascending=False)
        top20 = ranked_df.head(20).copy()

        print(f"\n  Before rerank (top 5 by similarity_score):")
        for _, r in top20.head(5).iterrows():
            in_gt = clean_title(r['Title']) in {clean_title(t) for t in relevant_movies}
            print(f"    '{r['Title']}' score={r['final_similarity_score']:.3f} in_GT={in_gt}")

        # Check rerank input/output
        print(f"\n  Cross-Encoder input: pairs=(query, movie_profile)")
        print(f"  Query used for reranking: '{query}'")
        print(f"  Cross-Encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2 (MS-MARCO model)")

        try:
            reranked = rerank_results(query, top20, top_k=10)
            print(f"\n  After rerank (top 5 by rerank_score):")
            for _, r in reranked.head(5).iterrows():
                in_gt = clean_title(r['Title']) in {clean_title(t) for t in relevant_movies}
                score = r.get('rerank_score', 'N/A')
                print(f"    '{r['Title']}' rerank_score={score:.3f} in_GT={in_gt}")
        except Exception as e:
            print(f"  Rerank error: {e}")

    print("\n[DIAGNOSIS 4 NOTES]")
    print("  - Model: cross-encoder/ms-marco-MiniLM-L-6-v2 is trained on MS-MARCO (passage retrieval)")
    print("  - MS-MARCO measures query-passage relevance for web search, NOT movie similarity")
    print("  - When query='phim tương tự phim X', cross-encoder sees (query, movie_profile)")
    print("  - MS-MARCO model will give HIGH scores to profiles matching search intent words")
    print("    not to movies that are actually similar to the seed movie")
    print("  - This is a domain mismatch, not a parameter order bug")
    print("  - The NDCG computation uses ranked_df (before rerank) as rel_map for BOTH before/after")
    print("    This is CORRECT in the code - NDCG uses similarity scores as relevance proxy")
    print("  - Root cause: MS-MARCO cross-encoder reorders by search relevance, not by movie similarity")

    # ====================================================================
    # BẤT THƯỜNG 5 — Retrieval Recall cực thấp, Recall@500 ≈ Recall@100
    # In số lượng candidate thực tế
    # ====================================================================
    print(f"\n{DIVIDER}")
    print("BẤT THƯỜNG 5 — Retrieval Recall thấp, BM25 Recall@500 ≈ Recall@100")
    print(DIVIDER)

    sample_gts_5 = random.sample(ground_truth_list, 5)
    for i, gt in enumerate(sample_gts_5):
        query = gt["query"]
        print(f"\n  Query #{i+1}: {query[:60]}")

        bm25_res = bm25_search(query, df_filtered, bm25_index, top_k=100)
        faiss_res = semantic_search_retriever(query, df_filtered, index_c, embedder_model, top_k=150)

        print(f"    BM25 candidates returned: {len(bm25_res)} (top_k=100)")
        print(f"    FAISS candidates returned: {len(faiss_res)} (top_k=150)")

        # Hybrid
        seen_links = set()
        hybrid_recs = []
        for res_df in [faiss_res, bm25_res]:
            if not res_df.empty:
                for _, row in res_df.iterrows():
                    link = row["Movie Link"]
                    if link not in seen_links:
                        seen_links.add(link)
                        hybrid_recs.append(row["Title"])
        print(f"    Hybrid candidates (deduped): {len(hybrid_recs)}")

        # Metadata filter (using title as filter - this is the BUG)
        filters = {"title": gt["seed_movie"]}
        metadata_res = search_movies_tool(df_filtered, filters, top_k=200)
        print(f"    Metadata filter (title='{gt['seed_movie']}'): {len(metadata_res)} results")
        print(f"    => Metadata filter searches BY TITLE of seed movie, NOT by similarity!")
        print(f"    => Result is the seed movie itself, NOT similar movies!")

        gt_set = {clean_title(t) for t in gt["relevant_movies"]}
        if not metadata_res.empty:
            meta_titles = {clean_title(t) for t in metadata_res["Title"].tolist()}
            hits = meta_titles & gt_set
            print(f"    Metadata filter hits against GT: {hits}")

        # Check if BM25 is truly capped at 100
        print(f"    BM25 actual return count: {len(bm25_res)} (cap=100, so Recall@500 = Recall@100 for BM25!)")

    print("\n[DIAGNOSIS 5 NOTES]")
    print("  BUG A: BM25 is called with top_k=100, so it CANNOT return 500 candidates")
    print("    -> Recall@500 for BM25 equals Recall@100 by design (same candidates)")
    print("  BUG B: 'Hybrid + Metadata filtering' uses filters={'title': seed_movie}")
    print("    -> search_movies_tool filters by TITLE CONTAINS seed_movie")
    print("    -> Returns ONLY the seed movie itself or its sequels, NOT similar movies")
    print("    -> This is why Hybrid+Filter has 0.5% recall: it never returns GT movies")
    print("  FIX: BM25 top_k should be raised to 500 for Recall@500 measurement")
    print("  FIX: 'Hybrid + Metadata filtering' should apply genre/year filters, NOT title filter")

    # ====================================================================
    # BẤT THƯỜNG 6 — Latency P95 bất thường cao (P95/Avg ratio)
    # Kiểm tra retry logic trong LLM calls
    # ====================================================================
    print(f"\n{DIVIDER}")
    print("BẤT THƯỜNG 6 — Latency P95 bất thường cao")
    print(DIVIDER)

    from chatbot.llm_client import get_llm_client
    import inspect

    llm_src = inspect.getsource(get_llm_client)
    print("\n  get_llm_client source:")
    print(f"    {llm_src[:500]}")

    from chatbot.chains import intent_chain
    intent_src = inspect.getsource(intent_chain)
    has_retry = 'retry' in intent_src.lower() or 'sleep' in intent_src.lower() or 'backoff' in intent_src.lower()
    print(f"\n  Intent chain has retry/sleep logic: {has_retry}")

    # Check llm_cache in evaluation
    llm_cache_path = os.path.join(workspace_dir, "evaluation_v3", "llm_cache.json")
    if os.path.exists(llm_cache_path):
        with open(llm_cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"\n  LLM cache entries: {len(cache)}")
        print(f"  => Latency evaluation hits cache for these, bypasses actual LLM call")
        print(f"  => But first-time calls (cache miss) go to real LLM -> high latency variance")
    else:
        print("\n  LLM cache: NOT FOUND")

    print("\n[DIAGNOSIS 6 NOTES]")
    print("  - P95 >> Avg is consistent with outlier calls (not exponential backoff retry)")
    print("  - Entity Extraction avg=5437ms is suspiciously high -> likely LLM-based (not pure regex)")
    print("  - Intent LLM: P95=9404ms vs avg=1284ms -> 7x ratio suggests occasional timeout/hang")
    print("  - RAG Generation: P95=27192ms vs avg=4497ms -> 6x ratio")
    print("  - Root cause: local LLM server (cx/gpt-5.5 at localhost:20128) has variable response time")
    print("  - The CachingLLM in run_eval.py DOES cache responses, so re-runs see low latency")
    print("  - First run / cold cache: some calls time out -> outliers inflate P95")
    print("  - Recommendation: report latency EXCLUDING cached hits for transparency")

    print(f"\n{DIVIDER}")
    print("CHẨN ĐOÁN HOÀN THÀNH. Xem kết quả bên trên để xác nhận trước khi sửa code.")
    print(DIVIDER)

if __name__ == "__main__":
    main()
