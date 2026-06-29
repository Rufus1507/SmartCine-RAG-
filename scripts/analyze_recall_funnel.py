"""
Script phân tích funnel Recall@10 theo từng Stage của pipeline.
Trả lời 3 câu hỏi trong task.md phản hồi lần 4:
  1a. Funnel analysis: số candidate giảm dần qua Stage 2, 3, 4, Top10
  1b. Chi tiết điểm số của candidates BỊ LOẠI khỏi Top 10
  1c. Mô phỏng graph_weight thay đổi (0.05 -> 0.15) tác động tới Recall@10
  3.  Ngưỡng rating >= 7.0 loại bỏ bao nhiêu phim personnel
"""
import os
import sys
import json
import re
import numpy as np
import pandas as pd
from collections import deque

sys.stdout.reconfigure(encoding='utf-8')
chatbot_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(chatbot_dir)

from chatbot.data_loader import load_data, load_faiss_index, load_embedder_model
from chatbot.graph.build_movie_graph import load_or_build_graph
from chatbot.graph.graph_query import find_movies_by_collab_path, clean_name
from chatbot.retrieval.multistage_retriever import MultistageRetriever
from chatbot.feature_engineering import MovieFeatureBuilder
from chatbot.similarity.weighted_similarity import compute_weighted_similarity, DEFAULT_WEIGHTS
from chatbot.representation.semantic_representation import make_profile
from chatbot.retrieval.retriever import semantic_search_retriever
from chatbot.data_loader import load_bm25_index
from chatbot.retrieval.bm25_retriever import bm25_search
from chatbot.retrieval.reranker import rerank_results

def clean_title(t):
    return re.sub(r"[^\w\s]", "", str(t).lower().strip())

def evaluate_metrics(recommendations, ground_truth):
    gt_clean = {clean_title(t) for t in ground_truth}
    if not gt_clean:
        return {"precision@5": 0.0, "precision@10": 0.0, "recall@10": 0.0, "mrr": 0.0}
    hits_5, hits_10, mrr_val = 0, 0, 0.0
    for idx, rec in enumerate(recommendations[:10]):
        rec_clean = clean_title(rec)
        matched = any(gt == rec_clean or gt in rec_clean or rec_clean in gt for gt in gt_clean)
        if matched:
            if idx < 5: hits_5 += 1
            hits_10 += 1
            if mrr_val == 0.0: mrr_val = 1.0 / (idx + 1)
    return {
        "precision@5": hits_5 / 5.0,
        "precision@10": hits_10 / 10.0,
        "recall@10": hits_10 / len(gt_clean),
        "mrr": mrr_val
    }

def build_stage2_to_top10(query, seed_movie, ground_truth, df, faiss_index, embedder_model,
                           bm25_index, G, builder, retriever, title_map, weights_to_use=None):
    """Chạy pipeline từ Stage 1 → Top 10, trả về intermediate results."""
    if weights_to_use is None:
        weights_to_use = DEFAULT_WEIGHTS.copy()

    base_row, is_sim = retriever._get_base_movie(df, query, {"title": seed_movie})
    if not is_sim or base_row is None:
        return None

    ref_title = base_row["Title"]

    # Stage 1: Graph candidates
    graph_results = find_movies_by_collab_path(G, ref_title, max_hops=3, max_neighbors_per_hop=20)
    graph_rows = []
    for res in graph_results[:300]:
        idx = title_map.get(res["Title"].lower())
        if idx is not None:
            row_copy = df.iloc[idx].copy()
            row_copy["graph_path_explanation"] = res["graph_path_explanation"]
            row_copy["graph_path_type"] = res.get("graph_path_type", "personnel")
            graph_rows.append(row_copy)

    n_graph_candidates = len(graph_rows)

    faiss_cands = semantic_search_retriever(query, df, faiss_index, embedder_model, top_k=150)
    bm25_cands = bm25_search(query, df, bm25_index, top_k=100)

    seen_links = set()
    candidate_list = []
    for cands_df_src in ([pd.DataFrame(graph_rows)] if graph_rows else []) + [faiss_cands, bm25_cands]:
        if not cands_df_src.empty:
            for _, row in cands_df_src.iterrows():
                link = row.get("Movie Link", "")
                if link not in seen_links:
                    seen_links.add(link)
                    candidate_list.append(row)
                    if len(candidate_list) >= 500: break
        if len(candidate_list) >= 500: break

    candidates_all = pd.DataFrame(candidate_list)
    stage2_df = candidates_all.head(200).copy()
    n_after_stage2 = len(stage2_df)

    # Count graph in stage 2
    graph_links = {r.get("Movie Link", "") for r in graph_rows}
    n_graph_in_stage2 = sum(1 for _, row in stage2_df.iterrows() if row.get("Movie Link", "") in graph_links)

    # Stage 3: Weighted Similarity
    ref_features = builder.transform_row(base_row)
    base_profile = make_profile(base_row, 'C')
    if embedder_model is not None:
        ref_features["semantic_embedding"] = embedder_model.encode([base_profile], convert_to_numpy=True)[0]
    ref_features["graph_score"] = 1.0

    candidate_profiles = [make_profile(row, 'C') for _, row in stage2_df.iterrows()]
    if embedder_model is not None:
        cand_embs = embedder_model.encode(candidate_profiles, convert_to_numpy=True)
    else:
        cand_embs = [None] * len(stage2_df)

    matched_rows = []
    for idx_c, (_, row) in enumerate(stage2_df.iterrows()):
        row_features = builder.transform_row(row)
        if embedder_model is not None and idx_c < len(cand_embs):
            row_features["semantic_embedding"] = cand_embs[idx_c]
        if "graph_path_explanation" in row and pd.notna(row.get("graph_path_explanation")):
            p_type = row.get("graph_path_type", "personnel")
            row_features["graph_score"] = 1.0 if p_type == "personnel" else 0.0
        else:
            row_features["graph_score"] = 0.0
        sim_bd = compute_weighted_similarity(row_features, ref_features, weights=weights_to_use.copy())
        row_copy = row.copy()
        row_copy["final_similarity_score"] = sim_bd["final_score"]
        row_copy["content_similarity"] = sim_bd["content_score"]
        row_copy["genre_similarity"] = sim_bd["genre_score"]
        row_copy["actor_similarity"] = sim_bd["actor_score"]
        row_copy["director_similarity"] = sim_bd["director_score"]
        row_copy["graph_score_val"] = sim_bd["graph_score"]
        matched_rows.append(row_copy)

    ranked_df = pd.DataFrame(matched_rows).sort_values("final_similarity_score", ascending=False)
    top100_df = ranked_df.head(100).copy()
    n_graph_in_stage3 = sum(1 for _, row in top100_df.iterrows() if row.get("Movie Link", "") in graph_links)

    # Stage 4: Rerank
    query_rerank = f"Phim tương tự như {ref_title} có thể loại, đạo diễn hoặc nội dung hấp dẫn."
    reranked_df = rerank_results(query_rerank, top100_df, top_k=20)
    n_graph_in_stage4 = sum(1 for _, row in reranked_df.iterrows() if row.get("Movie Link", "") in graph_links)

    final_top10 = reranked_df.head(10)
    n_graph_in_top10 = sum(1 for _, row in final_top10.iterrows() if row.get("Movie Link", "") in graph_links)
    recs = [row["Title"] for _, row in final_top10.iterrows()]
    metrics = evaluate_metrics(recs, ground_truth)

    return {
        "seed": seed_movie,
        "ref_title": ref_title,
        "n_graph_candidates": n_graph_candidates,
        "n_after_stage2": n_after_stage2,
        "n_graph_in_stage2": n_graph_in_stage2,
        "n_graph_in_stage3_top100": n_graph_in_stage3,
        "n_graph_in_stage4_top20": n_graph_in_stage4,
        "n_graph_in_top10": n_graph_in_top10,
        "recall@10": metrics["recall@10"],
        "precision@10": metrics["precision@10"],
        "mrr": metrics["mrr"],
        "recs": recs,
        "top100_df": top100_df,
        "graph_links": graph_links,
        "ground_truth": ground_truth,
    }


def main():
    print("=" * 70)
    print("PHÂN TÍCH FUNNEL RECALL@10 - CINEBOT V3 GRAPH RAG")
    print("=" * 70)

    print("\n📦 Loading dữ liệu và model...")
    df = load_data()
    faiss_index = load_faiss_index()
    embedder_model = load_embedder_model()
    bm25_index = load_bm25_index(df)
    G = load_or_build_graph(df)
    builder = MovieFeatureBuilder()
    retriever = MultistageRetriever()
    title_map = {str(t).lower(): idx for idx, t in enumerate(df["Title"])}

    eval_set_path = os.path.join(chatbot_dir, "evaluation_v3", "multihop_eval_set.json")
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    print(f"✅ Loaded {len(eval_set)} câu hỏi kiểm thử.\n")

    # ===================================================================
    # PHẦN 1A: FUNNEL ANALYSIS (5 câu hỏi đại diện)
    # ===================================================================
    print("=" * 70)
    print("PHẦN 1A: FUNNEL ANALYSIS — SỐ GRAPH CANDIDATE QUA TỪNG STAGE")
    print("=" * 70)
    # 5 câu đại diện: 0=Inception, 3=Pulp Fiction, 5=The Godfather, 7=Forrest Gump, 11=Inglourious Basterds
    rep_indices = [0, 3, 5, 7, 11]
    funnel_results = []
    all_dropped = []

    for qi in rep_indices:
        item = eval_set[qi]
        print(f"  Đang phân tích: [{qi+1}/19] {item['seed_movie']}...")
        res = build_stage2_to_top10(
            item["query"], item["seed_movie"], item["relevant_movies"],
            df, faiss_index, embedder_model, bm25_index, G, builder, retriever, title_map
        )
        if res is None:
            continue
        funnel_results.append(res)

        # Thu thập candidates bị loại (có trong GT, nhưng không vào top10)
        gt_clean = {clean_title(t) for t in item["relevant_movies"]}
        recs_clean = {clean_title(r) for r in res["recs"]}
        top100_df = res["top100_df"]
        graph_links = res["graph_links"]
        for _, row in top100_df.iterrows():
            t = row.get("Title", "")
            link = row.get("Movie Link", "")
            t_clean = clean_title(t)
            in_gt = any(g == t_clean or g in t_clean or t_clean in g for g in gt_clean)
            in_top10 = any(g == t_clean or g in t_clean or t_clean in g for g in recs_clean)
            if in_gt and not in_top10:
                all_dropped.append({
                    "seed": item["seed_movie"],
                    "candidate": t,
                    "is_graph": link in graph_links,
                    "content_sim": row.get("content_similarity", 0),
                    "genre_sim": row.get("genre_similarity", 0),
                    "actor_sim": row.get("actor_similarity", 0),
                    "director_sim": row.get("director_similarity", 0),
                    "graph_score": row.get("graph_score_val", 0),
                    "final_score": row.get("final_similarity_score", 0),
                })

    print()
    print(f"{'Seed Movie':<25} {'GraphGen':>8} {'S2(200)':>8} {'Graph@S2':>9} {'Graph@S3':>9} {'Graph@S4':>9} {'Graph@T10':>10} {'Recall':>8}")
    print("-" * 95)
    for fd in funnel_results:
        print(f"{fd['seed']:<25} {fd['n_graph_candidates']:>8} {fd['n_after_stage2']:>8} "
              f"{fd['n_graph_in_stage2']:>9} {fd['n_graph_in_stage3_top100']:>9} "
              f"{fd['n_graph_in_stage4_top20']:>9} {fd['n_graph_in_top10']:>10} {fd['recall@10']:>8.2%}")

    # ===================================================================
    # PHẦN 1B: CHI TIẾT ĐIỂM SỐ CANDIDATES GROUND TRUTH BỊ LOẠI
    # ===================================================================
    print("\n" + "=" * 70)
    print("PHẦN 1B: ĐIỂM SỐ CHI TIẾT GT CANDIDATES BỊ LOẠI KHỎI TOP 10")
    print("=" * 70)

    if all_dropped:
        print(f"\n{'Seed':<22} {'Candidate':<30} {'Cont':>6} {'Genre':>6} {'Actor':>6} {'Dir':>6} {'Graph':>6} {'Final':>6} {'IsGraph':>8}")
        print("-" * 100)
        for ex in all_dropped[:10]:
            print(f"{ex['seed'][:21]:<22} {ex['candidate'][:29]:<30} "
                  f"{ex['content_sim']:>6.3f} {ex['genre_sim']:>6.3f} {ex['actor_sim']:>6.3f} "
                  f"{ex['director_sim']:>6.3f} {ex['graph_score']:>6.2f} {ex['final_score']:>6.3f} "
                  f"{'✓' if ex['is_graph'] else '✗':>8}")
    else:
        print("(Không tìm thấy GT candidate bị loại trong 5 câu đại diện)")
        print("→ Cho thấy các GT candidate đã không vào Top100 của Stage 3 (bị loại ở đây)")

    # ===================================================================
    # PHẦN 1C: MÔ PHỎNG TRỌNG SỐ GRAPH 0.05 → 0.15
    # ===================================================================
    print("\n" + "=" * 70)
    print("PHẦN 1C: MÔ PHỎNG TRỌNG SỐ GRAPH: 0.05 → 0.15")
    print("=" * 70)
    print("(Chạy lại Stage 3-4 với trọng số mới, KHÔNG thay đổi code production)")

    new_weights = DEFAULT_WEIGHTS.copy()
    new_weights["graph"] = 0.15
    new_weights["content"] = 0.25

    sim_orig = []
    sim_new = []

    for qi, item in enumerate(eval_set):
        print(f"  [{qi+1:2d}/19] {item['seed_movie']:<30}", end=" -> ")
        res_orig = build_stage2_to_top10(
            item["query"], item["seed_movie"], item["relevant_movies"],
            df, faiss_index, embedder_model, bm25_index, G, builder, retriever, title_map,
            weights_to_use=DEFAULT_WEIGHTS.copy()
        )
        res_new = build_stage2_to_top10(
            item["query"], item["seed_movie"], item["relevant_movies"],
            df, faiss_index, embedder_model, bm25_index, G, builder, retriever, title_map,
            weights_to_use=new_weights.copy()
        )
        r_orig = res_orig["recall@10"] if res_orig else 0.0
        r_new  = res_new["recall@10"]  if res_new  else 0.0
        sim_orig.append(res_orig or {"recall@10": 0.0, "precision@10": 0.0, "mrr": 0.0})
        sim_new.append(res_new or {"recall@10": 0.0, "precision@10": 0.0, "mrr": 0.0})
        print(f"Recall orig={r_orig:.2%}  new={r_new:.2%}")

    avg_o = {k: np.mean([r.get(k, 0) for r in sim_orig]) for k in ["precision@10", "recall@10", "mrr"]}
    avg_n = {k: np.mean([r.get(k, 0) for r in sim_new])  for k in ["precision@10", "recall@10", "mrr"]}

    print("\n📊 KẾT QUẢ SO SÁNH TRỌNG SỐ GRAPH")
    print(f"{'Chỉ số':<15} {'graph=0.05 (hiện tại)':>22} {'graph=0.15 (mô phỏng)':>22} {'Delta':>10}")
    print("-" * 75)
    for k in ["precision@10", "recall@10", "mrr"]:
        delta = avg_n[k] - avg_o[k]
        print(f"{k:<15} {avg_o[k]:>22.4f} {avg_n[k]:>22.4f} {delta:>+10.4f}")

    # ===================================================================
    # PHẦN 3: NGƯỠNG RATING >= 5.0 — BAO NHIÊU PHIM BỊ LOẠI
    # ===================================================================
    print("\n" + "=" * 70)
    print("PHẦN 3: TÁC ĐỘNG NGƯỠNG RATING >= 5.0 TRÊN GROUND TRUTH")
    print("=" * 70)

    def find_all_personnel_candidates(seed, max_hops=3):
        seed_node = f"Movie:{seed}" if G.has_node(f"Movie:{seed}") else None
        if not seed_node:
            for node, data in G.nodes(data=True):
                if data.get("type") == "Movie" and clean_name(node).lower() == seed.lower():
                    seed_node = node; break
        if not seed_node: return {}

        def get_nb(u):
            raw = []
            if G.has_node(u):
                for v in G.successors(u):
                    for key in G[u][v]:
                        if G[u][v][key].get("type") in ("DIRECTED", "ACTED_IN", "COLLAB_WITH"):
                            raw.append(v)
                for v in G.predecessors(u):
                    for key in G[v][u]:
                        if G[v][u][key].get("type") in ("DIRECTED", "ACTED_IN", "COLLAB_WITH"):
                            raw.append(v)
            return list(dict.fromkeys(raw))

        queue = deque([(seed_node, 0)])
        visited = {seed_node}
        candidates = {}
        while queue:
            curr, hops = queue.popleft()
            if hops >= max_hops: continue
            for nb in get_nb(curr):
                if nb in visited: continue
                visited.add(nb)
                nb_type = G.nodes[nb].get("type")
                if nb_type == "Movie":
                    rating = G.nodes[nb].get("rating", 0.0) or 0.0
                    candidates[clean_name(nb)] = rating
                queue.append((nb, hops + 1))
        return candidates

    print(f"\n{'Seed Movie':<28} {'Tổng có p.path':>14} {'>= 5.0':>7} {'< 5.0':>7} {'% bị loại':>11}")
    print("-" * 80)
    total_all, total_kept, total_dropped = 0, 0, 0
    for item in eval_set:
        seed = item["seed_movie"]
        cands = find_all_personnel_candidates(seed)
        n_all = len(cands)
        n_kept = sum(1 for r in cands.values() if r >= 5.0)
        n_dropped = n_all - n_kept
        pct = (n_dropped / n_all * 100) if n_all > 0 else 0.0
        total_all += n_all; total_kept += n_kept; total_dropped += n_dropped
        print(f"{seed:<28} {n_all:>14,} {n_kept:>7,} {n_dropped:>7,} {pct:>10.1f}%")

    overall_pct = (total_dropped / total_all * 100) if total_all > 0 else 0.0
    print("-" * 80)
    print(f"{'TỔNG CỘNG':<28} {total_all:>14,} {total_kept:>7,} {total_dropped:>7,} {overall_pct:>10.1f}%")
    print()
    if overall_pct > 30:
        print("⚠️  CẢNH BÁO: Trên 30% phim có personnel path bị loại do rating < 5.0.")
        print("   → Đề xuất xem xét hạ ngưỡng hoặc điều chỉnh đồ thị.")
    else:
        print(f"✅ Tỷ lệ bị loại ({overall_pct:.1f}%) < 30% — ngưỡng 5.0 rất hợp lý.")

    print("\n✅ Phân tích hoàn tất.")

if __name__ == "__main__":
    main()
