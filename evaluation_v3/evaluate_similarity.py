import os
import sys
# Configure console to output UTF-8 to avoid Windows charmap encoding errors
sys.stdout.reconfigure(encoding='utf-8')
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import faiss
import torch
from sentence_transformers import SentenceTransformer

# Add workspace dir to path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(workspace_dir)

from chatbot.config import MIN_VOTES_THRESHOLD, CHATBOT_DIR
from chatbot.data_loader import load_data, load_embedder_model
from chatbot.representation.semantic_representation import (
    INDEX_A_PATH, INDEX_B_PATH, INDEX_C_PATH, make_profile
)
from chatbot.retrieval.multistage_retriever import MultistageRetriever

# Define ground truth similar movies for the 6 target movies
GROUND_TRUTH = {
    "Iron Man": {
        "title": "Iron Man",
        "matches": [
            "Iron Man 2", "Iron Man 3", "The Avengers", "Avengers: Age of Ultron",
            "Captain America: Civil War", "Avengers: Infinity War", "Avengers: Endgame",
            "Captain America: The Winter Soldier", "Captain America: The First Avenger",
            "Thor", "Man of Steel", "The Incredible Hulk", "Spider-Man: Homecoming"
        ]
    },
    "Avengers: Endgame": {
        "title": "Avengers: Endgame",
        "matches": [
            "Avengers: Infinity War", "The Avengers", "Avengers: Age of Ultron",
            "Captain America: Civil War", "Guardians of the Galaxy", "Guardians of the Galaxy Vol. 2",
            "Thor: Ragnarok", "Iron Man", "Captain America: The Winter Soldier",
            "Doctor Strange", "Black Panther", "Thor", "Captain Marvel"
        ]
    },
    "The Dark Knight": {
        "title": "The Dark Knight",
        "matches": [
            "The Dark Knight Rises", "Batman Begins", "Batman", "Batman Returns",
            "Joker", "Batman v Superman: Dawn of Justice", "Watchmen", "The Prestige",
            "Shutter Island", "Inception", "Man of Steel", "V for Vendetta", "The Dark Knight Returns"
        ]
    },
    "Interstellar": {
        "title": "Interstellar",
        "matches": [
            "Inception", "The Martian", "Gravity", "Arrival", "Contact", "Ad Astra",
            "2001: A Space Odyssey", "First Man", "Sunshine", "Moon", "The Fountain",
            "Prometheus", "Oblivion"
        ]
    },
    "The Martian": {
        "title": "The Martian",
        "matches": [
            "Interstellar", "Gravity", "Apollo 13", "Moon", "Prometheus", "Contact",
            "First Man", "Ad Astra", "Arrival", "Cast Away", "Apollo 11", "Sunshine"
        ]
    },
    "Inception": {
        "title": "Inception",
        "matches": [
            "Interstellar", "Shutter Island", "Memento", "The Matrix", "Tenet",
            "The Prestige", "Coherence", "Eternal Sunshine of the Spotless Mind",
            "The Dark Knight", "Minority Report", "Source Code", "Looper", "The Matrix Reloaded"
        ]
    }
}

def clean_title(t):
    return re.sub(r"[^\w\s]", "", str(t).lower().strip())

def evaluate_metrics(recommendations: list, ground_truth: list) -> dict:
    gt_clean = {clean_title(t) for t in ground_truth}
    
    hits = 0
    dcg = 0.0
    mrr = 0.0
    
    # Calculate Precision and DCG
    for idx, rec in enumerate(recommendations[:10]):
        rank = idx + 1
        rec_clean = clean_title(rec)
        
        # Check matching
        matched = False
        for gt in gt_clean:
            if gt == rec_clean or gt in rec_clean or rec_clean in gt:
                matched = True
                break
                
        if matched:
            hits += 1
            dcg += 1.0 / np.log2(rank + 1)
            if mrr == 0.0:
                mrr = 1.0 / rank
                
    precision_10 = hits / 10.0
    recall_10 = hits / len(ground_truth)
    
    # Ideal DCG
    idcg = 0.0
    for idx in range(min(10, len(ground_truth))):
        idcg += 1.0 / np.log2(idx + 2)
        
    ndcg_10 = dcg / idcg if idcg > 0 else 0.0
    
    return {
        "precision@10": precision_10,
        "recall@10": recall_10,
        "mrr": mrr,
        "ndcg": ndcg_10
    }

def main():
    print("============================================================")
    print("🎬 EVALUATION FRAMEWORK V3 — SIMILARITY & HYBRID RAG")
    print("============================================================")
    
    # Load dataset
    df = load_data()
    df_filtered = df[df['num_votes'] >= MIN_VOTES_THRESHOLD].reset_index(drop=True)
    print(f"Loaded {len(df):,} movies, {len(df_filtered):,} filtered movies.")
    
    embedder_model = load_embedder_model()
    retriever = MultistageRetriever()
    
    # Check if index files exist, if not wait/raise
    if not (os.path.exists(INDEX_A_PATH) and os.path.exists(INDEX_B_PATH) and os.path.exists(INDEX_C_PATH)):
        print("⚠️ Waiting for representation index files to be built...")
        # Fallback to load normal indices if not present yet
        INDEX_A = None
        INDEX_B = None
        INDEX_C = None
    else:
        INDEX_A = faiss.read_index(INDEX_A_PATH)
        INDEX_B = faiss.read_index(INDEX_B_PATH)
        INDEX_C = faiss.read_index(INDEX_C_PATH)
        
    # We will evaluate 4 systems:
    # 1. Legacy / Current Similarity System (description semantic index only, no weights)
    # 2. Weighted Similarity (Version A: Description)
    # 3. Weighted Similarity (Version B: Genre + Description)
    # 4. Weighted Similarity (Version C: Genre + Description + Keywords)
    
    system_metrics = {
        "Legacy System": [],
        "Weighted Sim (Version A)": [],
        "Weighted Sim (Version B)": [],
        "Weighted Sim (Version C)": []
    }
    
    detailed_results = {}
    
    for key, gt_info in GROUND_TRUTH.items():
        target_title = gt_info["title"]
        matches = gt_info["matches"]
        print(f"\nEvaluating: '{target_title}'...")
        detailed_results[target_title] = {}
        
        # Get target row
        target_rows = df[df['Title'].astype(str).str.lower() == target_title.lower()]
        if target_rows.empty:
            target_rows = df[df['Title'].astype(str).str.contains(target_title, case=False, na=False)]
        if target_rows.empty:
            print(f"❌ Target movie '{target_title}' not found in database!")
            continue
        target_row = target_rows.iloc[0]
        
        # --- 1. Current Legacy System ---
        # Heuristic semantic search using standard index + cross encoder
        from chatbot.retrieval.similar_movie_retriever import find_similar_movies_v2
        legacy_index = faiss.read_index(os.path.join(CHATBOT_DIR, "description_embeddings.index"))
        
        legacy_df, _ = find_similar_movies_v2(df, legacy_index, embedder_model, f"phim giống phim {target_title}", {})
        legacy_recs = legacy_df["Title"].tolist() if not legacy_df.empty else []
        legacy_metrics = evaluate_metrics(legacy_recs, matches)
        system_metrics["Legacy System"].append(legacy_metrics)
        detailed_results[target_title]["Legacy System"] = legacy_recs[:10]
        
        # --- 2. Weighted System (A, B, C) ---
        versions = [('A', INDEX_A), ('B', INDEX_B), ('C', INDEX_C)]
        for ver, idx in versions:
            sys_name = f"Weighted Sim (Version {ver})"
            
            if idx is None:
                # If index is missing, skip or fallback
                system_metrics[sys_name].append({"precision@10": 0.0, "recall@10": 0.0, "mrr": 0.0, "ndcg": 0.0})
                detailed_results[target_title][sys_name] = []
                continue
                
            # Perform retrieval using MultistageRetriever
            res_df = retriever.retrieve(
                query=f"phim giống phim {target_title}",
                df=df,
                filters={},
                intent="search",
                faiss_index=idx,
                embedder_model=embedder_model,
                version=ver,
                final_k=10
            )
            recs = res_df["Title"].tolist() if not res_df.empty else []
            metrics = evaluate_metrics(recs, matches)
            system_metrics[sys_name].append(metrics)
            detailed_results[target_title][sys_name] = recs[:10]
            
    # Compute averages
    report_lines = []
    report_lines.append("# 📊 Báo cáo Đánh giá CineBot V3 (Weighted Similarity & Multi-stage Hybrid Retrieval)\n")
    report_lines.append(f"- **Thời gian chạy**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("### 📈 So sánh các Hệ thống (Average Metrics over 6 Test Movies)\n")
    report_lines.append("| Hệ thống | Precision@10 | Recall@10 | MRR | NDCG@10 |\n")
    report_lines.append("|---|---|---|---|---|\n")
    
    print("\n--- RESULTS SUMMARY ---")
    for sys_name, metrics_list in system_metrics.items():
        if not metrics_list:
            continue
        avg_p = np.mean([m["precision@10"] for m in metrics_list])
        avg_r = np.mean([m["recall@10"] for m in metrics_list])
        avg_mrr = np.mean([m["mrr"] for m in metrics_list])
        avg_ndcg = np.mean([m["ndcg"] for m in metrics_list])
        
        print(f"{sys_name}: P@10={avg_p:.3f}, R@10={avg_r:.3f}, MRR={avg_mrr:.3f}, NDCG={avg_ndcg:.3f}")
        report_lines.append(f"| **{sys_name}** | {avg_p*100:.1f}% | {avg_r*100:.1f}% | {avg_mrr:.3f} | {avg_ndcg:.3f} |\n")
        
    report_lines.append("\n### 🔍 Chi tiết gợi ý 10 phim tương tự hàng đầu (Top 10 Recommendations Comparison)\n")
    for target, systems in detailed_results.items():
        report_lines.append(f"#### Phim gốc: **{target}**\n")
        for sys_name, recs in systems.items():
            recs_str = ", ".join([f"_{r}_" for r in recs]) if recs else "Không tìm thấy"
            report_lines.append(f"- **{sys_name}**: {recs_str}\n")
            
    # Write before_vs_after_report.md
    report_path = os.path.join(workspace_dir, "evaluation_v3", "before_vs_after_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
        
    print(f"\nBáo cáo trước vs sau đã được lưu tại: {report_path}")

if __name__ == "__main__":
    import re
    main()
