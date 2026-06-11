import re
import pandas as pd
from chatbot.config import FINAL_TOP_K
from chatbot.tools import search_movies_tool
from chatbot.bm25_retriever import bm25_search
from chatbot.data_loader import load_bm25_index
from chatbot.hybrid_search import hybrid_search
from chatbot.reranker import rerank_results

def is_similar_movie_query(query: str, filters: dict) -> bool:
    """
    Xác định xem truy vấn có phải là yêu cầu tìm phim tương tự hay không.
    """
    similar_patterns = [
        r'(?:phim\s+)?(giống|tương\s+tự|tựa\s+như|tựa\s+với|như)\s+(?:phim\s+)?',
        r'similar\s+to',
        r'like\s+'
    ]
    query_lower = query.lower()
    for pat in similar_patterns:
        if re.search(pat, query_lower):
            return True
            
    # Kiểm tra xem từ khóa có chứa từ chỉ định tương đồng không
    words_in_msg = set(re.findall(r'\b\w+\b', query_lower))
    if filters.get("title") and not words_in_msg.isdisjoint({"giống", "giong", "tương tự", "tuong tu", "như", "nhu", "tựa", "tua"}):
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
    Định tuyến truy vấn (Retrieval Router) đến luồng xử lý phù hợp nhất.
    Trả về tuple: (DataFrame kết quả, tên luồng định tuyến)
    """
    # 1. Kiểm tra xem có phải truy vấn phim tương đồng (Similar Movie) hay không
    if is_similar_movie_query(query, filters):
        from chatbot.similar_movie_retriever import find_similar_movies_v2
        similar_df, found = find_similar_movies_v2(df, faiss_index, embedder_model, query, filters)
        if found:
            return similar_df.head(final_k), "similar_movie_v2"
        
    # 2. Kiểm tra nếu chỉ tìm theo Quốc gia (Country Search)
    is_country_only = filters.get("country") and not any(filters.get(k) for k in [
        "title", "genre", "director", "star", "year_min", "year_max", "rating_min"
    ])
    if is_country_only:
        # Route: Metadata Filter only
        result = search_movies_tool(df, filters, top_k=final_k)
        return result, "metadata_only"
        
    # 3. Kiểm tra tìm theo thực thể cụ thể (Title, Actor, Director)
    is_entity_search = any(filters.get(k) for k in ["title", "director", "star"])
    if is_entity_search or intent == "info":
        # Route: BM25 only (Tìm kiếm từ khóa chính xác trên tên phim/diễn viên/đạo diễn)
        bm25_index = load_bm25_index(df)
        candidates = bm25_search(query, df, bm25_index, top_k=100)
        result = search_movies_tool(candidates, filters, top_k=final_k)
        return result, "bm25_only"
        
    # 4. Kiểm tra tìm theo Thể loại (Genre Search)
    is_genre_search = filters.get("genre") and not any(filters.get(k) for k in ["title", "director", "star"])
    if is_genre_search:
        # Route: Metadata Filter + BM25
        bm25_index = load_bm25_index(df)
        candidates = bm25_search(query, df, bm25_index, top_k=100)
        if candidates.empty:
            result = search_movies_tool(df, filters, top_k=final_k)
        else:
            result = search_movies_tool(candidates, filters, top_k=final_k)
            if result.empty:
                result = search_movies_tool(df, filters, top_k=final_k)
        return result, "metadata_plus_bm25"
        
    # 5. Mặc định là tìm kiếm nội dung / ngữ nghĩa (Content Search)
    # Route: BM25 + FAISS (Hybrid Search V2)
    # Chạy Hybrid Search để lấy các ứng viên và xếp hạng theo RRF
    result = hybrid_search(query, df, filters, faiss_index, embedder_model, final_k=final_k)
    return result, "hybrid_rrf"
