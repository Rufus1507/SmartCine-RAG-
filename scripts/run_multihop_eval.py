import os
import sys
import json
import time
import re
import pandas as pd
import numpy as np

# Đảm bảo in UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào path để import dạng 'from chatbot.xyz'
chatbot_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(chatbot_dir)

from chatbot.data_loader import load_data, load_faiss_index, load_embedder_model
from chatbot.retrieval.retrieval_router import route_retrieval
from chatbot.retrieval.multistage_retriever import MultistageRetriever

def clean_title(t):
    return re.sub(r"[^\w\s]", "", str(t).lower().strip())

def evaluate_metrics(recommendations: list, ground_truth: list) -> dict:
    gt_clean = {clean_title(t) for t in ground_truth}
    if not gt_clean:
        return {"precision@5": 0.0, "precision@10": 0.0, "recall@10": 0.0, "mrr": 0.0}

    hits_5 = 0
    hits_10 = 0
    mrr_val = 0.0
    for idx, rec in enumerate(recommendations[:10]):
        rec_clean = clean_title(rec)
        matched = False
        for gt in gt_clean:
            if gt == rec_clean or gt in rec_clean or rec_clean in gt:
                matched = True
                break
        if matched:
            if idx < 5:
                hits_5 += 1
            hits_10 += 1
            if mrr_val == 0.0:
                mrr_val = 1.0 / (idx + 1)
                
    precision_5 = hits_5 / 5.0
    precision_10 = hits_10 / 10.0
    recall_10 = hits_10 / len(gt_clean) if len(gt_clean) > 0 else 0.0
    
    return {
        "precision@5": precision_5,
        "precision@10": precision_10,
        "recall@10": recall_10,
        "mrr": mrr_val
    }

def main():
    print("🎬 Tải dữ liệu phim và mô hình...")
    df = load_data()
    faiss_index = load_faiss_index()
    embedder_model = load_embedder_model()
    
    eval_set_path = os.path.join(chatbot_dir, "evaluation_v3", "multihop_eval_set.json")
    if not os.path.exists(eval_set_path):
        print(f"❌ File dataset không tồn tại: {eval_set_path}")
        return
        
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
        
    print(f"📊 Đã tải bộ dữ liệu gồm {len(eval_set)} câu hỏi kiểm thử.")
    
    results_with_graph = []
    results_without_graph = []
    
    retriever = MultistageRetriever()
    
    print("\n🚀 Bắt đầu đánh giá...")
    for idx, item in enumerate(eval_set):
        query = item["query"]
        seed_movie = item["seed_movie"]
        ground_truth = item["relevant_movies"]
        
        # Thiết lập bộ lọc mặc định cho phim seed
        filters = {"title": seed_movie}
        intent = "search"
        
        # 1. Chạy với Graph RAG (mặc định trong route_retrieval)
        t0 = time.time()
        res_with, _ = route_retrieval(
            query=query,
            df=df,
            filters=filters,
            intent=intent,
            faiss_index=faiss_index,
            embedder_model=embedder_model,
            final_k=10
        )
        latency_with = time.time() - t0
        
        rec_with = [row["Title"] for _, row in res_with.iterrows()]
        metrics_with = evaluate_metrics(rec_with, ground_truth)
        metrics_with["latency"] = latency_with
        results_with_graph.append(metrics_with)
        
        # 2. Chạy KHÔNG có Graph RAG (chỉ dùng multistage semantic/hybrid/filter thông thường)
        t0 = time.time()
        res_without = retriever.retrieve(
            query=query,
            df=df,
            filters=filters,
            intent=intent,
            faiss_index=faiss_index,
            embedder_model=embedder_model,
            final_k=10,
            graph_candidates=None  # Không truyền graph candidates
        )
        latency_without = time.time() - t0
        
        rec_without = [row["Title"] for _, row in res_without.iterrows()]
        metrics_without = evaluate_metrics(rec_without, ground_truth)
        metrics_without["latency"] = latency_without
        results_without_graph.append(metrics_without)
        
        print(f"Processed {idx+1}/{len(eval_set)}: '{seed_movie}' -> With Graph Recall@10: {metrics_with['recall@10']:.2%}, Without Graph: {metrics_without['recall@10']:.2%}")

    # Tính toán giá trị trung bình
    avg_with = {k: np.mean([r[k] for r in results_with_graph]) for k in ["precision@5", "precision@10", "recall@10", "mrr", "latency"]}
    avg_without = {k: np.mean([r[k] for r in results_without_graph]) for k in ["precision@5", "precision@10", "recall@10", "mrr", "latency"]}
    
    print("\n📈 KẾT QUẢ ĐÁNH GIÁ TRUNG BÌNH:")
    print("-----------------------------------------------------------------")
    print(f"| Chỉ số         | Không có Graph RAG (Baseline) | Có Graph RAG         |")
    print("-----------------------------------------------------------------")
    print(f"| Precision@5    | {avg_without['precision@5']:>27.2%} | {avg_with['precision@5']:>20.2%} |")
    print(f"| Precision@10   | {avg_without['precision@10']:>27.2%} | {avg_with['precision@10']:>20.2%} |")
    print(f"| Recall@10      | {avg_without['recall@10']:>27.2%} | {avg_with['recall@10']:>20.2%} |")
    print(f"| MRR            | {avg_without['mrr']:>27.4f} | {avg_with['mrr']:>20.4f} |")
    print(f"| Latency (giây) | {avg_without['latency']:>27.4f}s | {avg_with['latency']:>20.4f}s |")
    print("-----------------------------------------------------------------")
    
    # Lưu kết quả
    output_results = {
        "baseline_without_graph": avg_without,
        "with_graph_rag": avg_with,
        "detail_with_graph": results_with_graph,
        "detail_without_graph": results_without_graph
    }
    
    results_save_path = os.path.join(chatbot_dir, "evaluation_v3", "multihop_eval_results.json")
    with open(results_save_path, "w", encoding="utf-8") as f:
        json.dump(output_results, f, ensure_ascii=False, indent=2)
    print(f"\n✔️ Đã lưu kết quả chi tiết tại: {results_save_path}")

if __name__ == "__main__":
    main()
