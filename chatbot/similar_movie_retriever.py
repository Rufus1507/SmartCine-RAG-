import os
import re
import pandas as pd
import numpy as np
import faiss
import streamlit as st
from chatbot.config import PROFILE_INDEX_PATH, MIN_VOTES_THRESHOLD
from chatbot.reranker import rerank_results
from chatbot.tools import get_movie_detail_tool, search_movies_tool

@st.cache_resource
def load_profile_faiss_index():
    """
    Nạp tệp chỉ mục FAISS chứa vector profile phim.
    """
    if not os.path.exists(PROFILE_INDEX_PATH):
        return None
    return faiss.read_index(PROFILE_INDEX_PATH)

def make_profile(row):
    title = str(row.get('Title', '')).strip()
    genre = str(row.get('genres', '')).strip()
    director = str(row.get('directors', '')).strip()
    stars = str(row.get('stars', '')).strip()
    description = str(row.get('description', '')).strip()
    
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if genre:
        parts.append(f"Genre: {genre}")
    if director:
        parts.append(f"Director: {director}")
    if stars:
        parts.append(f"Stars: {stars}")
    if description:
        parts.append(f"Description: {description}")
        
    return "\n".join(parts)

def generate_similarity_reason(row, base_row):
    """
    Tạo lý do giải thích sự tương đồng giữa phim row và base_row.
    """
    reasons = []
    
    # So khớp thể loại
    base_genres = set([g.strip().lower() for g in str(base_row.get('genres', '')).split(',') if g.strip()])
    row_genres = set([g.strip().lower() for g in str(row.get('genres', '')).split(',') if g.strip()])
    common_genres = base_genres.intersection(row_genres)
    if common_genres:
        reasons.append(f"cùng thể loại ({', '.join(list(common_genres)[:3])})")
        
    # So khớp đạo diễn
    base_dirs = set([d.strip().lower() for d in str(base_row.get('directors', '')).split(',') if d.strip()])
    row_dirs = set([d.strip().lower() for d in str(row.get('directors', '')).split(',') if d.strip()])
    common_dirs = base_dirs.intersection(row_dirs)
    if common_dirs:
        # Chuyển viết hoa chữ cái đầu cho tên đạo diễn để đẹp hơn
        common_dirs_cap = [d.title() for d in common_dirs]
        reasons.append(f"được chỉ đạo bởi cùng đạo diễn ({', '.join(common_dirs_cap)})")
        
    # So khớp diễn viên chính
    base_stars = set([s.strip().lower() for s in str(base_row.get('stars', '')).split(',') if s.strip()])
    row_stars = set([s.strip().lower() for s in str(row.get('stars', '')).split(',') if s.strip()])
    common_stars = base_stars.intersection(row_stars)
    if common_stars:
        common_stars_cap = [s.title() for s in common_stars]
        reasons.append(f"có sự góp mặt của diễn viên ({', '.join(common_stars_cap[:3])})")
        
    if not reasons:
        reasons.append("có cốt truyện và phong cách nghệ thuật tương đồng")
        
    return "Phim " + ", ".join(reasons) + "."

def find_similar_movies_v2(df: pd.DataFrame, index, model, user_input: str, filters: dict) -> tuple[pd.DataFrame, bool]:
    """
    Tìm phim tương tự V2 sử dụng profile và Cross-Encoder Reranker.
    """
    # Trích xuất tên phim cần tìm tương tự
    similar_patterns = [
        r'(?:phim\s+)?(?:giống|tương\s+tự|tựa\s+như|tựa\s+với|như)\s+(?:phim\s+)?([^,.?]+)',
        r'(?:tương\s+tự|tựa)\s+với\s+(?:phim\s+)?([^,.?]+)',
        r'(?:phim\s+)?tựa\s+(?:bộ\s+|phim\s+)?([^,.?]+)',
        r'similar\s+to\s+([^,.?]+)',
        r'like\s+([^,.?]+)'
    ]
    
    candidate_title = None
    for pat in similar_patterns:
        match = re.search(pat, user_input, re.IGNORECASE)
        if match:
            candidate_title = match.group(1).strip()
            break
            
    words_in_msg = set(re.findall(r'\b\w+\b', user_input.lower()))
    if not candidate_title and filters.get("title") and not words_in_msg.isdisjoint({"giống", "giong", "tương tự", "tuong tu", "như", "nhu", "tựa", "tua"}):
        candidate_title = filters["title"]
        
    if not candidate_title:
        return pd.DataFrame(), False
        
    candidate_title = re.sub(r'^(bộ\s+phim|phim|bộ|cái|con|những|các|tựa|tựa\s+phim)\s+', '', candidate_title, flags=re.IGNORECASE).strip()
    
    # Lấy thông tin chi tiết phim gốc
    base_movie = get_movie_detail_tool(df, candidate_title)
    if base_movie.empty:
        # Fallback tìm kiếm tương đối theo tên phim gốc
        title_matches = df[df['Title'].astype(str).str.contains(candidate_title, case=False, na=False)]
        if not title_matches.empty:
            base_movie = title_matches.iloc[[0]]
        else:
            return pd.DataFrame(), False
            
    base_row = base_movie.iloc[0]
    base_title = base_row['Title']
    
    # Nạp chỉ mục profile FAISS
    profile_index = load_profile_faiss_index()
    if profile_index is None:
        # Fallback dùng description index thông thường nếu chưa tạo profile index
        return pd.DataFrame(), False
        
    # Tạo profile cho phim gốc để embed
    base_profile = make_profile(base_row)
    base_vector = model.encode([base_profile], convert_to_numpy=True).astype('float32')
    
    # Lọc danh sách phim ứng viên có num_votes >= MIN_VOTES_THRESHOLD
    df_filtered = df[df['num_votes'] >= MIN_VOTES_THRESHOLD].reset_index(drop=True)
    
    # Tìm kiếm FAISS profile
    k_search = min(150, len(df_filtered))
    distances, indices = profile_index.search(base_vector, k_search)
    
    matched_rows = []
    scores = []
    
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1 or idx >= len(df_filtered):
            continue
        row = df_filtered.iloc[idx].copy()
        
        # Bỏ qua chính bộ phim gốc đang tìm kiếm tương tự
        if str(row['Title']).lower() == base_title.lower():
            continue
            
        # Tính điểm tương đồng dựa trên L2 distance
        # Khoảng cách L2 nhỏ hơn nghĩa là độ tương đồng lớn hơn
        sim_score = float(1.0 / (1.0 + dist))
        row['similarity_score'] = f"{sim_score * 100:.1f}%"
        row['similarity_score_val'] = sim_score
        
        # Sinh lý do tương đồng
        row['similarity_reason'] = generate_similarity_reason(row, base_row)
        
        matched_rows.append(row)
        
    if not matched_rows:
        return pd.DataFrame(), True
        
    candidates_df = pd.DataFrame(matched_rows)
    
    # Áp dụng thêm bộ lọc metadata nếu có
    filters_copy = filters.copy()
    filters_copy["title"] = None
    candidates_df = search_movies_tool(candidates_df, filters_copy)
    
    if candidates_df.empty:
        return pd.DataFrame(), True
        
    # Rerank bằng Cross-Encoder
    query_rerank = f"Phim tương tự như {base_title} có thể loại, đạo diễn hoặc nội dung hấp dẫn."
    reranked_df = rerank_results(query_rerank, candidates_df, top_k=20)
    
    # Loại bỏ cột phụ trung gian
    if 'similarity_score_val' in reranked_df.columns:
        reranked_df = reranked_df.drop(columns=['similarity_score_val'])
        
    return reranked_df, True
