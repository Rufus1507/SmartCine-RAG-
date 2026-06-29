import os
import sys
import json
import numpy as np

# Đảm bảo in UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào path để import dạng 'from chatbot.xyz'
chatbot_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(chatbot_dir)

from chatbot.data_loader import load_data
from chatbot.graph.build_movie_graph import load_or_build_graph
from chatbot.graph.graph_query import find_movies_by_collab_path
from chatbot.retrieval.multistage_retriever import MultistageRetriever

def main():
    print("🎬 Loading data and graph...")
    df = load_data()
    G = load_or_build_graph(df)
    
    eval_set_path = os.path.join(chatbot_dir, "evaluation_v3", "multihop_eval_set.json")
    if not os.path.exists(eval_set_path):
        print(f"❌ File dataset không tồn tại: {eval_set_path}")
        return
        
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
        
    print(f"📊 Đã tải bộ dữ liệu gồm {len(eval_set)} câu hỏi kiểm thử.")
    
    retriever = MultistageRetriever()
    queries_stats = []
    
    total_personnel = 0
    total_shared = 0
    total_candidates_all = 0
    
    for idx, item in enumerate(eval_set):
        query = item["query"]
        seed_movie = item["seed_movie"]
        
        # Trích xuất base movie và tìm ứng viên đồ thị
        # Giả lập logic tìm kiếm như route_retrieval
        base_row, is_similar = retriever._get_base_movie(df, query, {"title": seed_movie})
        if not is_similar or base_row is None:
            print(f"⚠️ Warning: Could not find base movie for '{seed_movie}' in query '{query}'")
            continue
            
        reference_movie_title = base_row["Title"]
        graph_results = find_movies_by_collab_path(G, reference_movie_title, max_hops=3, max_neighbors_per_hop=20)
        
        # Đếm số lượng loại đường đi
        num_personnel = 0
        num_shared = 0
        for res in graph_results:
            p_type = res.get("graph_path_type", "personnel")
            if p_type == "personnel":
                num_personnel += 1
            else:
                num_shared += 1
                
        total_candidates = num_personnel + num_shared
        ratio = num_personnel / total_candidates if total_candidates > 0 else 0.0
        
        queries_stats.append({
            "query": query,
            "seed_movie": seed_movie,
            "resolved_movie_title": reference_movie_title,
            "num_personnel": num_personnel,
            "num_shared_attribute": num_shared,
            "total_candidates": total_candidates,
            "personnel_ratio": ratio
        })
        
        total_personnel += num_personnel
        total_shared += num_shared
        total_candidates_all += total_candidates
        
        print(f"[{idx+1}/{len(eval_set)}] Seed: '{seed_movie}' -> Personnel: {num_personnel}, Shared: {num_shared}, Ratio: {ratio:.2%}")
        
    avg_personnel_ratio = np.mean([q["personnel_ratio"] for q in queries_stats]) if queries_stats else 0.0
    
    stats_summary = {
        "summary": {
            "total_queries": len(queries_stats),
            "total_personnel_candidates": total_personnel,
            "total_shared_attribute_candidates": total_shared,
            "total_candidates": total_candidates_all,
            "avg_personnel_candidates": total_personnel / len(queries_stats) if queries_stats else 0.0,
            "avg_shared_attribute_candidates": total_shared / len(queries_stats) if queries_stats else 0.0,
            "avg_total_candidates": total_candidates_all / len(queries_stats) if queries_stats else 0.0,
            "avg_personnel_ratio": avg_personnel_ratio
        },
        "queries": queries_stats
    }
    
    output_path = os.path.join(chatbot_dir, "personnel_ratio_stats.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, ensure_ascii=False, indent=2)
        
    print(f"\n✔️ Đã lưu kết quả đo đạc tại: {output_path}")

if __name__ == "__main__":
    main()
