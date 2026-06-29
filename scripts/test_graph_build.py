import os
import sys

# Đảm bảo in UTF-8 trên Windows để không bị lỗi UnicodeEncodeError
sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào path để import dạng 'from chatbot.xyz'
chatbot_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(chatbot_dir)

from chatbot.data_loader import load_data
from chatbot.graph.build_movie_graph import load_or_build_graph

def main():
    print("🎬 Tải dữ liệu để build graph...")
    df = load_data()
    
    print("🛠️ Đang tải hoặc xây dựng đồ thị (Graph RAG)...")
    G = load_or_build_graph(df, force_rebuild=True)
    
    print("\n================ STATISTICS ================")
    print(f"Tổng số Nodes: {G.number_of_nodes()}")
    print(f"Tổng số Edges: {G.number_of_edges()}")
    
    # Đếm số lượng node theo từng loại
    node_types = {}
    for node, data in G.nodes(data=True):
        ntype = data.get("type", "Unknown")
        node_types[ntype] = node_types.get(ntype, 0) + 1
        
    print("\nSố lượng node theo phân loại:")
    for ntype, count in node_types.items():
        print(f"  - {ntype}: {count}")
        
    # Đếm số lượng edge theo từng loại
    edge_types = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        etype = data.get("type", "Unknown")
        edge_types[etype] = edge_types.get(etype, 0) + 1
        
    print("\nSố lượng edge theo phân loại:")
    for etype, count in edge_types.items():
        print(f"  - {etype}: {count}")
        
    # In thử 1 ví dụ multi-hop với phim đầu tiên hoặc phim phổ biến
    print("\n================ MULTI-HOP TEST ================")
    # Tìm phim có rating cao hoặc có nhiều vote
    movie_candidates = df[df['num_votes'] >= 5000].sort_values(by="Rating", ascending=False)
    if not movie_candidates.empty:
        test_movie = movie_candidates.iloc[0]["Title"]
    else:
        test_movie = df.iloc[0]["Title"]
        
    print(f"Chọn phim test: '{test_movie}'")
    
    if G.has_node(test_movie):
        print(f"\nQuan hệ trực tiếp của phim '{test_movie}':")
        for neighbor in G.neighbors(test_movie):
            for key in G[test_movie][neighbor]:
                etype = G[test_movie][neighbor][key].get("type")
                ntype = G.nodes[neighbor].get("type")
                print(f"  -> {etype} -> {neighbor} ({ntype})")
                
        print(f"\nĐạo diễn & Diễn viên liên quan trực tiếp tới '{test_movie}':")
        for u in G.predecessors(test_movie):
            ntype = G.nodes[u].get("type")
            for key in G[u][test_movie]:
                etype = G[u][test_movie][key].get("type")
                print(f"  <- {etype} <- {u} ({ntype})")
                
        print("\nMulti-hop path: Các diễn viên đã từng hợp tác với đạo diễn của phim này:")
        directors = [u for u in G.predecessors(test_movie) if G.nodes[u].get("type") == "Director"]
        for d in directors:
            collaborators = []
            if G.has_node(d):
                for neighbor in G.neighbors(d):
                    for key in G[d][neighbor]:
                        if G[d][neighbor][key].get("type") == "COLLAB_WITH":
                            weight = G[d][neighbor][key].get("weight", 1)
                            collaborators.append((neighbor, weight))
            collaborators = sorted(collaborators, key=lambda x: x[1], reverse=True)[:5]
            print(f"  Đạo diễn '{d}' từng hợp tác nhiều nhất với:")
            for c, w in collaborators:
                print(f"    - Diễn viên '{c}' (số lần hợp tác: {w})")
    else:
        print(f"❌ Không tìm thấy phim '{test_movie}' trong đồ thị.")

if __name__ == "__main__":
    main()
