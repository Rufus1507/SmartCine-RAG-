import re
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
    query_lower = query.lower()
    for pat in similar_patterns:
        if re.search(pat, query_lower):
            return True
            
    # Kiểm tra xem từ khóa có chứa từ chỉ định tương đồng không
    words_in_msg = set(re.findall(r'\b\w+\b', query_lower))
    if filters.get("title") and not words_in_msg.isdisjoint({"giống", "giong", "tương tự", "tuong tu", "như", "nhu", "tựa", "tua", "liên", "lien", "quan"}):
        return True
        
    return False

def route_retrieval(
    query: str,
    df: pd.DataFrame,
    filters: dict,
    intent: str,
    faiss_index,
    embedder_model,
    final_k: int = FINAL_TOP_K
) -> tuple[pd.DataFrame, str]:
    """
    Định tuyến truy vấn (Retrieval Router) sang hệ thống Multi-stage Hybrid Retrieval V3.
    Bổ sung thêm bước lấy candidate từ Graph RAG (NetworkX) nếu là truy vấn tìm phim tương tự.
    Trả về tuple: (DataFrame kết quả, tên luồng định tuyến)
    """
    from chatbot.retrieval.multistage_retriever import MultistageRetriever
    retriever = MultistageRetriever()
    
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
                graph_results = find_movies_by_collab_path(G, reference_movie_title, max_hops=3, max_neighbors_per_hop=20)
                
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
                        graph_rows.append(row_copy)
                        
                if graph_rows:
                    graph_candidates = pd.DataFrame(graph_rows)
            except Exception as e:
                print(f"⚠️ Lỗi khi lấy candidates từ Graph RAG: {e}")
                
    result = retriever.retrieve(
        query=query,
        df=df,
        filters=filters,
        intent=intent,
        faiss_index=faiss_index,
        embedder_model=embedder_model,
        final_k=final_k,
        graph_candidates=graph_candidates
    )
    return result, "multistage_hybrid"
