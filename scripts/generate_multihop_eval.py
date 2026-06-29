import os
import sys
import json
import pandas as pd

# Đảm bảo in UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào path để import dạng 'from chatbot.xyz'
chatbot_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(chatbot_dir)

from chatbot.data_loader import load_data
from chatbot.graph.build_movie_graph import load_or_build_graph
from chatbot.graph.graph_query import find_movies_by_collab_path, clean_name

def main():
    print("🎬 Loading data and graph for generating eval set...")
    df = load_data()
    G = load_or_build_graph(df, force_rebuild=False)
    
    # 20 phim phổ biến trong đồ thị
    popular_titles = [
        "Inception", "The Dark Knight", "Interstellar", "Pulp Fiction", "The Matrix",
        "The Godfather", "Fight Club", "Forrest Gump", "Se7en", "Gladiator",
        "Avatar", "Titanic", "Inglourious Basterds", "Django Unchained", "The Departed",
        "The Prestige", "Memento", "Saving Private Ryan", "The Silence of the Lambs", "Goodfellas"
    ]
    
    eval_set = []
    
    for title in popular_titles:
        # Kiểm tra xem phim có trong graph không
        movie_node = f"Movie:{title}"
        if not G.has_node(movie_node):
            # Tìm phim gần đúng
            movie_lower = title.lower()
            found = False
            for node, data in G.nodes(data=True):
                if data.get("type") == "Movie" and clean_name(node).lower() == movie_lower:
                    movie_node = node
                    title = clean_name(node)
                    found = True
                    break
            if not found:
                continue
                
        # Tìm các phim liên kết qua graph (max_hops=3, limit=15)
        linked = find_movies_by_collab_path(G, title, max_hops=3, max_neighbors_per_hop=15)
        if not linked:
            continue
            
        relevant_movies = [m["Title"] for m in linked[:10]]
        
        # Tạo 3 dạng câu hỏi multi-hop tự nhiên phong phú cho mỗi phim
        query_templates = [
            f"Tìm phim giống như {title} có cùng diễn viên hoặc đạo diễn liên quan",
            f"Gợi ý phim tương tự {title} qua liên kết nhân sự và thể loại",
            f"Phim nào liên quan đến {title} qua các mối quan hệ đạo diễn, diễn viên từng hợp tác?"
        ]
        
        # Chọn mẫu câu ngẫu nhiên hoặc tuần tự
        query = query_templates[len(eval_set) % len(query_templates)]
        
        eval_set.append({
            "query": query,
            "seed_movie": title,
            "relevant_movies": relevant_movies
        })
        
    print(f"Generated {len(eval_set)} eval queries.")
    
    output_path = os.path.join(chatbot_dir, "evaluation_v3", "multihop_eval_set.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Saved eval set to: {output_path}")

if __name__ == "__main__":
    main()
