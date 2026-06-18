import pandas as pd
from chatbot.retrieval.retriever import semantic_search_retriever
from chatbot.tools import search_movies_tool
from chatbot.config import SEMANTIC_TOP_K, FINAL_TOP_K
from chatbot.retrieval.bm25_retriever import bm25_search
from chatbot.data_loader import load_bm25_index

def rrf_merge(bm25_df: pd.DataFrame, faiss_df: pd.DataFrame, k: int = 60) -> pd.DataFrame:
    """
    Kết hợp kết quả tìm kiếm của BM25 và FAISS dùng Reciprocal Rank Fusion (RRF).
    Deduplicate các kết quả theo cột 'Movie Link'.
    """
    if bm25_df.empty and faiss_df.empty:
        return pd.DataFrame()
    if bm25_df.empty:
        return faiss_df.copy()
    if faiss_df.empty:
        return bm25_df.copy()
        
    rrf_scores = {}
    
    # Tính điểm RRF cho các phim từ BM25
    for rank, link in enumerate(bm25_df["Movie Link"].values):
        rrf_scores[link] = rrf_scores.get(link, 0.0) + 1.0 / (k + (rank + 1))
        
    # Tính điểm RRF cho các phim từ FAISS
    for rank, link in enumerate(faiss_df["Movie Link"].values):
        rrf_scores[link] = rrf_scores.get(link, 0.0) + 1.0 / (k + (rank + 1))
        
    # Lấy danh sách link phim duy nhất
    all_links = list(rrf_scores.keys())
    
    # Tạo từ điển map link sang dòng dữ liệu
    link_to_row = {}
    for _, row in bm25_df.iterrows():
        link_to_row[row["Movie Link"]] = row
    for _, row in faiss_df.iterrows():
        link_to_row[row["Movie Link"]] = row
        
    # Sắp xếp các link phim theo điểm RRF giảm dần
    sorted_links = sorted(all_links, key=lambda l: rrf_scores[l], reverse=True)
    
    merged_rows = [link_to_row[link] for link in sorted_links]
    merged_df = pd.DataFrame(merged_rows)
    merged_df["rrf_score"] = [rrf_scores[link] for link in sorted_links]
    return merged_df

def hybrid_search(
    query: str, 
    df: pd.DataFrame, 
    filters: dict, 
    index, 
    model, 
    semantic_k: int = 100, 
    bm25_k: int = 100, 
    final_k: int = FINAL_TOP_K
) -> pd.DataFrame:
    """
    Tìm kiếm lai nâng cấp (Hybrid Search V2):
    1. Chạy BM25 lấy Top bm25_k (100) phim.
    2. Chạy FAISS lấy Top semantic_k (100) phim.
    3. Ghép kết quả và tính điểm RRF để lấy ra tối đa 150 phim hàng đầu.
    4. Áp dụng Metadata Filters (Pandas) trên các ứng viên.
    5. Fallback chạy lọc trên toàn bộ DB nếu rỗng.
    """
    has_metadata_filters = any(filters.get(k) for k in [
        "genre", "director", "star", "title", "year_min", "year_max", "rating_min", "country"
    ])
    
    if query:
        # Nạp chỉ mục BM25 qua cached data loader
        bm25_index = load_bm25_index(df)
        
        # 1. Tìm kiếm ngữ nghĩa FAISS
        faiss_candidates = pd.DataFrame()
        if index is not None and model is not None:
            faiss_candidates = semantic_search_retriever(query, df, index, model, top_k=semantic_k)
            
        # 2. Tìm kiếm từ khóa BM25
        bm25_candidates = pd.DataFrame()
        if bm25_index is not None:
            bm25_candidates = bm25_search(query, df, bm25_index, top_k=bm25_k)
            
        # 3. Trộn kết quả qua RRF
        candidates = rrf_merge(bm25_candidates, faiss_candidates, k=60)
        
        if candidates.empty:
            return search_movies_tool(df, filters, top_k=final_k)
            
        # 4. Áp dụng bộ lọc trên các ứng viên
        filtered_candidates = search_movies_tool(candidates, filters, top_k=final_k)
        
        # Fallback nếu bộ lọc loại bỏ toàn bộ ứng viên
        if filtered_candidates.empty and has_metadata_filters:
            return search_movies_tool(df, filters, top_k=final_k)
            
        return filtered_candidates
    else:
        # Lọc trực tiếp nếu không có query tìm kiếm
        return search_movies_tool(df, filters, top_k=final_k)
