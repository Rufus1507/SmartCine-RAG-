import re
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi

def clean_tokenize(text: str) -> list[str]:
    """
    Làm sạch chuỗi văn bản và tách thành danh sách các từ (tokens).
    """
    if not text or not isinstance(text, str):
        return []
    text = text.lower()
    # Thay thế các ký tự đặc biệt bằng khoảng trắng
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()

def build_bm25_index(df: pd.DataFrame) -> BM25Okapi:
    """
    Xây dựng chỉ mục BM25Okapi từ các trường: Title, genres, directors, stars, countries_origin.
    Sử dụng cột tfidf_text tiền xử lý để tối ưu hóa hiệu năng nếu có sẵn.
    """
    if "tfidf_text" in df.columns:
        corpus = df["tfidf_text"].fillna("").apply(clean_tokenize).tolist()
    else:
        corpus = []
        for _, row in df.iterrows():
            title = str(row.get("Title", ""))
            genres = str(row.get("genres", ""))
            directors = str(row.get("directors", ""))
            stars = str(row.get("stars", ""))
            countries = str(row.get("countries_origin", ""))
            
            # Ghép các thuộc tính và tăng trọng số cho Title (lặp lại 2 lần)
            doc = f"{title} {title} {genres} {directors} {stars} {countries}"
            corpus.append(clean_tokenize(doc))
        
    return BM25Okapi(corpus)

def bm25_search(query: str, df: pd.DataFrame, bm25_index: BM25Okapi, top_k: int = 100) -> pd.DataFrame:
    """
    Tìm kiếm từ khóa bằng BM25, trả về DataFrame các phim có điểm số cao nhất.
    """
    if df.empty or bm25_index is None or not query:
        return pd.DataFrame()
        
    tokenized_query = clean_tokenize(query)
    if not tokenized_query:
        return pd.DataFrame()
        
    scores = bm25_index.get_scores(tokenized_query)
    
    # Lấy các index có score cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    result = df.iloc[top_indices].copy()
    result["bm25_score"] = scores[top_indices]
    
    # Chỉ giữ lại các phim có điểm tương hợp > 0
    result = result[result["bm25_score"] > 0]
    return result
