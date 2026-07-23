import os
import json
import pickle
import networkx as nx
import pandas as pd
from tqdm import tqdm
from chatbot.config import CHATBOT_DIR
from chatbot.feature_engineering import clean_split

# Đường dẫn cache cho đồ thị
GRAPH_CACHE_PATH = os.path.join(CHATBOT_DIR, "movie_graph.pkl")

# Cache đồ thị trong bộ nhớ
_loaded_graph = None

def build_movie_graph(df: pd.DataFrame, vocab_data: dict, actor_metadata: dict, director_metadata: dict) -> nx.MultiDiGraph:

    """
    Xây dựng đồ thị NetworkX MultiDiGraph từ dữ liệu phim với ID node có tiền tố để tránh đụng độ tên.
    
    Args:
        df: DataFrame chứa danh sách các phim.
        vocab_data: Từ điển chứa vocabulary của actors, directors, countries...
        actor_metadata: Metadata chứa thông tin Tier của diễn viên.
        director_metadata: Metadata chứa thông tin Tier của đạo diễn.
        
    Returns:
        nx.MultiDiGraph: Đồ thị đa hướng có chứa các node và edge quan hệ.
    """
    G = nx.MultiDiGraph()
    
    # Không loại phim theo vote/rating: phim thiếu vote hoặc rating vẫn phải có thể được recommend.
    df_filtered = df.reset_index(drop=True)
    
    print(f"Building graph with {len(df_filtered)} movies...")
    
    # 1. Thêm các node thể loại từ từ vựng (Genre)
    from chatbot.feature_engineering.movie_feature_builder import PARENT_GENRES
    for genre in PARENT_GENRES:
        G.add_node(f"Genre:{genre}", type="Genre", name=genre)
        
    # 2. Thêm các node quốc gia từ vocabularies
    countries = vocab_data.get("countries", [])
    for country in countries:
        G.add_node(f"Country:{country}", type="Country", name=country)
        
    # Tạo cấu trúc lưu thông tin hợp tác (collab) giữa đạo diễn và diễn viên
    collab_counter = {}
    
    # 3. Duyệt qua từng bộ phim để thêm node Movie, Actor, Director và các cạnh liên kết
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Đang xây dựng graph nodes và edges"):
        movie_title = str(row.get("Title", "")).strip()
        if not movie_title:
            continue
            
        movie_node = f"Movie:{movie_title}"
        
        # Thêm node Movie
        G.add_node(
            movie_node,
            type="Movie",
            title=movie_title,
            year=row.get("Year"),
            rating=row.get("Rating"),
            num_votes=row.get("num_votes"),
            has_oscar=row.get("has_oscar"),
            has_awards=row.get("has_awards"),
            has_nomination=row.get("has_nomination"),
            decade=row.get("decade")
        )
        
        # Lấy danh sách đạo diễn, diễn viên, thể loại, quốc gia
        movie_genres = clean_split(row.get("genres"))
        movie_directors = clean_split(row.get("directors"))
        movie_actors = clean_split(row.get("stars"))
        movie_countries = clean_split(row.get("countries_origin"))
        
        # Thêm edge HAS_GENRE
        for g in movie_genres:
            from chatbot.feature_engineering.movie_feature_builder import GENRE_HIERARCHY
            mapped_genres = GENRE_HIERARCHY.get(g, [g] if g in PARENT_GENRES else [])
            for mg in mapped_genres:
                genre_node = f"Genre:{mg}"
                if G.has_node(genre_node):
                    G.add_edge(movie_node, genre_node, key="HAS_GENRE", type="HAS_GENRE")
                    
        # Thêm edge PRODUCED_IN
        for c in movie_countries:
            country_node = f"Country:{c}"
            if G.has_node(country_node):
                G.add_edge(movie_node, country_node, key="PRODUCED_IN", type="PRODUCED_IN")
                
        # Thêm node và edge cho Đạo diễn (Director)
        for d in movie_directors:
            if not d:
                continue
            d_node = f"Director:{d}"
            d_meta = director_metadata.get(d, {})
            d_tier = d_meta.get("director_tier", "Tier D")
            
            G.add_node(d_node, type="Director", director_name=d, director_tier=d_tier)
            G.add_edge(d_node, movie_node, key="DIRECTED", type="DIRECTED")
            
        # Thêm node và edge cho Diễn viên (Actor)
        for a in movie_actors:
            if not a:
                continue
            a_node = f"Actor:{a}"
            a_meta = actor_metadata.get(a, {})
            a_tier = a_meta.get("actor_tier", "Tier D")
            
            G.add_node(a_node, type="Actor", actor_name=a, actor_tier=a_tier)
            G.add_edge(a_node, movie_node, key="ACTED_IN", type="ACTED_IN")
            
        # Thu thập thông tin hợp tác (COLLAB_WITH) giữa đạo diễn và diễn viên
        for d in movie_directors:
            if not d:
                continue
            for a in movie_actors:
                if not a:
                    continue
                pair = (d, a)
                collab_counter[pair] = collab_counter.get(pair, 0) + 1
                
    # 4. Thêm các cạnh COLLAB_WITH giữa đạo diễn và diễn viên
    for (d, a), weight in collab_counter.items():
        d_node = f"Director:{d}"
        a_node = f"Actor:{a}"
        if G.has_node(d_node) and G.has_node(a_node):
            G.add_edge(d_node, a_node, key="COLLAB_WITH", type="COLLAB_WITH", weight=weight)
            G.add_edge(a_node, d_node, key="COLLAB_WITH", type="COLLAB_WITH", weight=weight)
            
    return G

def load_or_build_graph(df: pd.DataFrame, force_rebuild: bool = False) -> nx.MultiDiGraph:
    """
    Tải đồ thị từ cache nếu tồn tại, ngược lại sẽ xây dựng mới và lưu vào cache.
    """
    global _loaded_graph
    
    if not force_rebuild and _loaded_graph is not None:
        return _loaded_graph
        
    from chatbot.feature_engineering.movie_feature_builder import VOCAB_PATH, ACTOR_METADATA_PATH, DIRECTOR_METADATA_PATH
    
    if not force_rebuild and os.path.exists(GRAPH_CACHE_PATH):
        print(f"Loading movie graph from cache: {GRAPH_CACHE_PATH}...")
        try:
            with open(GRAPH_CACHE_PATH, "rb") as f:
                G = pickle.load(f)
            print(f"Successfully loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            cached_movie_count = sum(
                1 for _, data in G.nodes(data=True)
                if data.get("type") == "Movie"
            )
            expected_movie_count = (
                df["Title"].dropna().astype(str).str.strip().loc[lambda s: s.ne("")]
                .nunique()
                if "Title" in df.columns else 0
            )
            if expected_movie_count and cached_movie_count < expected_movie_count:
                print(
                    "Graph cache is missing movies compared to current dataset "
                    f"({cached_movie_count:,}/{expected_movie_count:,}). "
                    "Rebuilding without vote filtering..."
                )
            else:
                _loaded_graph = G
                return G
        except Exception as e:
            print(f"Could not load graph from cache: {e}. Rebuilding graph...")
            
    # Đọc vocab và metadata
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    with open(ACTOR_METADATA_PATH, "r", encoding="utf-8") as f:
        actor_metadata = json.load(f)
    with open(DIRECTOR_METADATA_PATH, "r", encoding="utf-8") as f:
        director_metadata = json.load(f)
        
    G = build_movie_graph(df, vocab_data, actor_metadata, director_metadata)
    
    print(f"Saving movie graph to cache: {GRAPH_CACHE_PATH}...")
    try:
        with open(GRAPH_CACHE_PATH, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Successfully saved graph cache!")
    except Exception as e:
        print(f"Error saving graph cache: {e}")
        
    _loaded_graph = G
    return G
