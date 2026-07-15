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

def rerank_results(query: str, candidates_df: pd.DataFrame, top_k: int = 20,
                   base_movie_profile: str = None) -> pd.DataFrame:
    """
    Xếp hạng lại các phim ứng viên bằng Cross-Encoder.
    - Với truy vấn tìm phim tương tự: dùng base_movie_profile thay vì câu hỏi thô
      để cross-encoder so sánh nội dung phim với phim gốc (chính xác hơn "có phim tương tự X không?")
    - Tạo profile phim: Description trước (quan trọng nhất), sau đó Title, Genre, Director, Stars.
    """
    if candidates_df.empty or not query:
        return candidates_df
        
    model = load_reranker_model()
    if model is None:
        return candidates_df.head(top_k)
    
    # Xác định query cho cross-encoder
    # Với similar_to queries: dùng profile phim gốc để so sánh nội dung trực tiếp
    rerank_query = base_movie_profile if base_movie_profile else query
        
    # Tạo profile cho từng phim ứng viên
    # Đặt Description trước vì cross-encoder ms-marco-MiniLM được train trên passage ranking
    # → phần đầu của passage có ảnh hưởng lớn hơn đến score
    movie_profiles = []
    for _, row in candidates_df.iterrows():
        desc = str(row.get("description", ""))
        title = str(row.get("Title", ""))
        genres = str(row.get("genres", ""))
        directors = str(row.get("directors", ""))
        stars = str(row.get("stars", ""))
        # Dùng final_context nếu có (cho aggregation results), ngược lại build từ fields
        if "final_context" in row and pd.notna(row["final_context"]) and len(str(row["final_context"])) > 20:
            profile = str(row["final_context"])
        else:
            # Đặt description ở đầu để cross-encoder tập trung vào nội dung
            profile = f"Description: {desc} | Title: {title} | Genres: {genres} | Directors: {directors} | Stars: {stars}"
        movie_profiles.append(profile)
        
    # Tạo cặp (query, doc) để đưa vào mô hình Cross-Encoder
    pairs = [(rerank_query, mp) for mp in movie_profiles]
    
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
