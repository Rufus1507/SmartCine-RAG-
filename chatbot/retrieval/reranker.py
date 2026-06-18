import streamlit as st
import pandas as pd
from sentence_transformers import CrossEncoder

@st.cache_resource
def load_reranker_model() -> CrossEncoder:
    """
    Nạp và lưu trữ mô hình Cross-Encoder để xếp hạng lại (Reranking).
    Mô hình mặc định: cross-encoder/ms-marco-MiniLM-L-6-v2
    """
    try:
        # Tải mô hình thông qua sentence-transformers
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return model
    except Exception as e:
        print(f"⚠️ Không thể tải mô hình Cross-Encoder: {e}")
        return None

def rerank_results(query: str, candidates_df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """
    Xếp hạng lại các phim ứng viên bằng Cross-Encoder.
    Tạo profile phim kết hợp các trường: Title, Genre, Director, Stars, Description.
    """
    if candidates_df.empty or not query:
        return candidates_df
        
    model = load_reranker_model()
    if model is None:
        return candidates_df.head(top_k)
        
    # Tạo profile cho từng phim ứng viên
    movie_profiles = []
    for _, row in candidates_df.iterrows():
        if "final_context" in row and pd.notna(row["final_context"]):
            profile = str(row["final_context"])
        else:
            title = str(row.get("Title", ""))
            genres = str(row.get("genres", ""))
            directors = str(row.get("directors", ""))
            stars = str(row.get("stars", ""))
            desc = str(row.get("description", ""))
            profile = f"Title: {title} | Genres: {genres} | Directors: {directors} | Stars: {stars} | Description: {desc}"
        movie_profiles.append(profile)
        
    # Tạo cặp (query, doc) để đưa vào mô hình Cross-Encoder
    pairs = [(query, mp) for mp in movie_profiles]
    
    try:
        scores = model.predict(pairs)
        
        result = candidates_df.copy()
        result["rerank_score"] = scores
        # Sắp xếp theo score giảm dần
        result = result.sort_values(by="rerank_score", ascending=False)
        return result.head(top_k).copy()
    except Exception as e:
        print(f"⚠️ Lỗi trong quá trình Rerank: {e}")
        return candidates_df.head(top_k).copy()
