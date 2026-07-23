import re
import unicodedata
import pandas as pd
from chatbot.config import FINAL_TOP_K
from chatbot.tools import search_movies_tool
from chatbot.retrieval.bm25_retriever import bm25_search
from chatbot.data_loader import load_bm25_index
from chatbot.retrieval.hybrid_search import hybrid_search
from chatbot.retrieval.reranker import rerank_results

def is_similar_movie_query(query: str, filters: dict) -> bool:
    """
    Xác định xem truy vấn có phải là yêu cầu tìm phim tương tự hay không.
    """
    similar_patterns = [
        r'(?:phim\s+)?(giống|tương\s+tự|tựa\s+như|tựa\s+với|như|liên\s+quan|lien\s+quan)\s+(?:phim\s+)?',
        r'similar\s+to',
        r'like\s+'
    ]
    # Chuẩn hóa Unicode NFC
    query_nfc = unicodedata.normalize('NFC', query.lower())
    for pat in similar_patterns:
        if re.search(pat, query_nfc):
            return True
            
    words_in_msg = set(re.findall(r'\b\w+\b', query_nfc))
    if filters.get("title") and not words_in_msg.isdisjoint({"giống", "giong", "tương tự", "tuong tu", "như", "nhu", "tựa", "tua", "liên", "lien", "quan"}):
        return True
        
    return False

def extract_title_from_query(query: str, df: pd.DataFrame) -> str:
    """
    Trích xuất tên phim từ câu truy vấn tiếng Việt bằng cách quét cụm từ sau 'phim '
    và đối chiếu với tiêu đề phim trong DataFrame.
    """
    query_nfc = unicodedata.normalize('NFC', query.lower())
    # Regex tìm các ký tự sau từ "phim "
    match = re.search(r'phim\s+([^,.?]+)', query_nfc, re.IGNORECASE)
    if match:
        potential_title = match.group(1).strip()
        # Thử khớp chính xác trước
        title_match = df[df['Title'].astype(str).str.lower() == potential_title]
        if not title_match.empty:
            return title_match.iloc[0]['Title']
            
        # Thử quét các cụm từ con (giảm dần từ dài tới ngắn) để bỏ đi các từ nối tiếng Việt
        words = potential_title.split()
        for i in range(len(words), 0, -1):
            sub_title = " ".join(words[:i])
            # Bỏ dấu hai chấm ở cuối nếu có
            sub_title_clean = sub_title.rstrip(":")
            title_match = df[df['Title'].astype(str).str.lower() == sub_title_clean]
            if not title_match.empty:
                return title_match.iloc[0]['Title']
    return None

def is_director_filmography_query(query: str, filters: dict, df: pd.DataFrame = None) -> bool:
    """
    Phat hien truy van multi-hop dang:
    'Phim khac cua dao dien X' hoac 'Dao dien cua phim X da lam gi?'
    Pattern nay can duoc xu ly qua Graph RAG de dam bao dung route multi-hop.
    """
    movie_title = filters.get("title")
    if not movie_title and df is not None:
        movie_title = extract_title_from_query(query, df)
        
    if not movie_title:
        return False
        
    # Chuẩn hóa Unicode NFC
    query_nfc = unicodedata.normalize('NFC', query.lower())
    # Pattern: "dao dien cua phim X" hoac "phim khac cua dao dien" hoac "X da lam phim gi khac"
    multihop_patterns = [
        r'\u0111\u1ea1o\s*di\u1ec5n\s+c\u1ee7a\s+phim',        # "dao dien cua phim X"
        r'phim\s*(n\u00e0o|kh\u00e1c)\s+c\u1ee7a\s+\u0111\u1ea1o',   # "phim khac cua dao dien"
        r'\u0111\u00e3\s+t\u1eebng\s+l\u00e0m.*phim',             # "X da tung lam nhung phim"
        r'l\u00e0m\s+nh\u1eefng\s+phim\s*(n\u00e0o|kh\u00e1c)',    # "lam nhung phim nao khac"
        r'directed\s+(other|any|what)',       # English equivalents
        r'other\s+(films?|movies?)\s+by',
    ]
    for pat in multihop_patterns:
        if re.search(pat, query_nfc, re.IGNORECASE):
            return True
    return False

def route_retrieval(
    query: str,
    df: pd.DataFrame,
    filters: dict,
    intent: str,
    faiss_index,
    embedder_model,
    final_k: int = FINAL_TOP_K,
    trace: dict = None
) -> tuple[pd.DataFrame, str]:
    """
    Định tuyến truy vấn (Retrieval Router) sang hệ thống Multi-stage Hybrid Retrieval V3.
    Bổ sung thêm bước lấy candidate từ Graph RAG (NetworkX) nếu là truy vấn tìm phim tương tự.
    Trả về tuple: (DataFrame kết quả, tên luồng định tuyến)
    """
    import numpy as np
    from chatbot.retrieval.multistage_retriever import MultistageRetriever

    # Task 3 FIX: copy ngay đầu hàm để tránh mutate object filters của caller.
    # rag_chain.py dùng {**last_filters, **new_filters} để merge context đa lượt;
    # nếu object bị mutate tại đây, filter tạm (vd: title) sẽ rò rỉ sang lượt tiếp theo.
    filters = filters.copy()

    retriever = MultistageRetriever()

    # Task 4 FIX: Đảm bảo key "stage0_graph" luôn tồn tại trong trace trước khi truy cập.
    # Nếu nhánh similar-movie hoặc filmography không được kích hoạt, key này sẽ không được
    # tạo ra → gây KeyError tiềm ẩn ở downstream code đọc trace["stage0_graph"].
    if trace is not None:
        trace.setdefault("stage0_graph", {"called": False, "candidates": []})

    graph_candidates = None
    
    # 1. Kiểm tra nếu là truy vấn phim tương tự
    if is_similar_movie_query(query, filters):
        # Trích xuất phim gốc
        base_row, is_similar = retriever._get_base_movie(df, query, filters)
        if is_similar and base_row is not None:
            reference_movie_title = base_row["Title"]
            
            # Nạp và truy vấn đồ thị phim
            try:
                from chatbot.graph.build_movie_graph import load_or_build_graph
                from chatbot.graph.graph_query import find_movies_by_collab_path
                
                G = load_or_build_graph(df)
                # Tìm phim có đường đi quan hệ trên đồ thị (max_hops=3)
                # Giới hạn max_hops=2 để chỉ lấy phim có chung nhân sự trực tiếp 1-hop
                # (Phim gốc → Diễn viên/Đạo diễn → Phim khác).
                # max_hops=3 trước đây cho phép chuỗi 2-hop nhân sự không liên quan nội dung
                # (ví dụ: Interstellar → Jessica Chastain → Director X → phim của Director X).
                graph_results = find_movies_by_collab_path(G, reference_movie_title, max_hops=2, max_neighbors_per_hop=20)
                
                if trace is not None:
                    trace["stage0_graph"]["called"] = True
                    serializable_graph = []
                    for res in graph_results:
                        clean_res = {}
                        for k, v in res.items():
                            if isinstance(v, (np.integer, np.floating)):
                                clean_res[k] = v.item()
                            else:
                                clean_res[k] = v
                        serializable_graph.append(clean_res)
                    trace["stage0_graph"]["candidates"] = serializable_graph
                
                graph_rows = []
                # Tối ưu hóa: Tạo map tra cứu tiêu đề O(1) thay vì quét DataFrame O(N) trong vòng lặp
                title_map = {str(t).lower(): idx for idx, t in enumerate(df["Title"])}
                
                # Giới hạn số lượng ứng viên đồ thị ở mức 300 để tránh quá tải
                for res in graph_results[:300]:
                    title = res["Title"]
                    explanation = res["graph_path_explanation"]
                    p_type = res.get("graph_path_type", "personnel")
                    
                    idx = title_map.get(title.lower())
                    if idx is not None:
                        row_copy = df.iloc[idx].copy()
                        row_copy["graph_path_explanation"] = explanation
                        row_copy["graph_path_type"] = p_type
                        row_copy["graph_hop_count"] = res.get("hop_count", 2)  # Số hop, mặc định 2 nếu không có thông tin
                        graph_rows.append(row_copy)
                        
                if graph_rows:
                    graph_candidates = pd.DataFrame(graph_rows)
            except Exception as e:
                print(f"Error getting candidates from Graph RAG: {e}")
    
    # 2. P3 FIX: Kiểm tra nếu là truy vấn multi-hop filmography của đạo diễn/diễn viên
    # Ví dụ: "Đạo diễn của Alien: Romulus đã từng làm những phim kinh dị nào khác?"
    # LLM parse ra director=Fede Alvarez → exact_filter_shortcut vô tình đúng nhưng
    # không phải graph multi-hop. Cần trigger graph từ seed movie để đảm bảo route chính xác.
    elif is_director_filmography_query(query, filters, df):
        movie_title = filters.get("title") or extract_title_from_query(query, df)
        if movie_title:
            # Lưu lại vào filters để đồng bộ với phần còn lại của pipeline
            filters["title"] = movie_title
            try:
                from chatbot.graph.build_movie_graph import load_or_build_graph
                from chatbot.graph.graph_query import find_movies_by_collab_path
                
                G = load_or_build_graph(df)
                # Tìm phim có đường đi quan hệ từ phim seed qua graph (movie -> director -> other movies)
                graph_results = find_movies_by_collab_path(G, movie_title, max_hops=2, max_neighbors_per_hop=30)
                
                if trace is not None:
                    trace["stage0_graph"]["called"] = True
                    serializable_graph = []
                    for res in graph_results:
                        clean_res = {}
                        for k, v in res.items():
                            if isinstance(v, (np.integer, np.floating)):
                                clean_res[k] = v.item()
                            else:
                                clean_res[k] = v
                        serializable_graph.append(clean_res)
                    trace["stage0_graph"]["candidates"] = serializable_graph
                
                graph_rows = []
                title_map = {str(t).lower(): idx for idx, t in enumerate(df["Title"])}
                for res in graph_results[:300]:
                    title = res["Title"]
                    idx = title_map.get(title.lower())
                    if idx is not None:
                        row_copy = df.iloc[idx].copy()
                        row_copy["graph_path_explanation"] = res["graph_path_explanation"]
                        row_copy["graph_path_type"] = res.get("graph_path_type", "personnel")
                        row_copy["graph_hop_count"] = res.get("hop_count", 2)
                        graph_rows.append(row_copy)
                if graph_rows:
                    graph_candidates = pd.DataFrame(graph_rows)
            except Exception as e:
                print(f"Error getting graph candidates for filmography query: {e}")
                


                
    local_trace = trace if trace is not None else {}
    result = retriever.retrieve(
        query=query,
        df=df,
        filters=filters,
        intent=intent,
        faiss_index=faiss_index,
        embedder_model=embedder_model,
        final_k=final_k,
        graph_candidates=graph_candidates,
        trace=local_trace
    )
    actual_route = local_trace.get("actual_route", "multistage_hybrid")
    return result, actual_route
