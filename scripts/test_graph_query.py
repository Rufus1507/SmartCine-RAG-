import os
import sys

# Đảm bảo in UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào path để import dạng 'from chatbot.xyz'
chatbot_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(chatbot_dir)

from chatbot.data_loader import load_data
from chatbot.graph.build_movie_graph import load_or_build_graph
from chatbot.graph.graph_query import find_collaborators_of_movie, find_movies_by_collab_path, explain_path

def main():
    print("🎬 Loading data and graph...")
    df = load_data()
    G = load_or_build_graph(df, force_rebuild=False)
    
    # Lấy một phim phổ biến để test
    test_movies = ["Inception", "The Dark Knight", "Interstellar", "Pulp Fiction", "The Matrix"]
    test_movie = None
    for m in test_movies:
        if G.has_node(m):
            test_movie = m
            break
            
    if not test_movie:
        # Fallback to first movie in G of type Movie
        for node, data in G.nodes(data=True):
            if data.get("type") == "Movie":
                test_movie = node
                break
                
    print(f"\n🎯 Selected test movie: '{test_movie}'")
    
    print("\n1. Testing find_collaborators_of_movie:")
    collabs = find_collaborators_of_movie(G, test_movie)
    print(f"Found {len(collabs)} collaborators.")
    for c in collabs[:5]:
        print(f"  - {c['name']} ({c['type']}): collaborator of {c['collaborator_of']} with weight {c['weight']}")
        
    print("\n2. Testing find_movies_by_collab_path (max_hops=3):")
    linked_movies = find_movies_by_collab_path(G, test_movie, max_hops=3, max_neighbors_per_hop=5)
    print(f"Found {len(linked_movies)} linked movies.")
    for m in linked_movies[:5]:
        print(f"  - Movie: {m['Title']} (Rating: {m['Rating']}, Year: {m['Year']})")
        print(f"    Link explanation: {m['graph_path_explanation']}")
        
    # Testing explain_path for a specific pair
    if len(linked_movies) > 0:
        other_movie = linked_movies[0]['Title']
        print(f"\n3. Testing explain_path between '{test_movie}' and '{other_movie}':")
        explanation, path_type = explain_path(G, test_movie, other_movie)
        print(f"  Explanation: {explanation} ({path_type})")


if __name__ == "__main__":
    main()
