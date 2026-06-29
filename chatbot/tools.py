import unicodedata
import pandas as pd
import numpy as np

from chatbot.config import (
    COL_TITLE, COL_GENRE, COL_DIRECTOR, COL_STARS, COL_YEAR, COL_RATING, COL_OVERVIEW,
    COL_OSCAR, COL_AWARDS, COL_NOMINATION, COL_DURATION, COL_METASCORE
)
COL_VOTES = "num_votes"

def normalize_genre(genre_str: str) -> str:
    if not genre_str:
        return genre_str
    genre_lower = genre_str.strip().lower()
    
    mapping = {
        "science fiction": "Sci-Fi",
        "sci fi": "Sci-Fi",
        "sci-fi": "Sci-Fi",
        "khoa học viễn tưởng": "Sci-Fi",
        "viễn tưởng": "Sci-Fi",
        "hành động": "Action",
        "action": "Action",
        "hài": "Comedy",
        "hài hước": "Comedy",
        "comedy": "Comedy",
        "kinh dị": "Horror",
        "horror": "Horror",
        "kịch tính": "Drama",
        "chính kịch": "Drama",
        "drama": "Drama",
        "lãng mạn": "Romance",
        "tình cảm": "Romance",
        "romance": "Romance",
        "hoạt hình": "Animation",
        "animation": "Animation",
        "phiêu lưu": "Adventure",
        "adventure": "Adventure",
        "tội phạm": "Crime",
        "hình sự": "Crime",
        "crime": "Crime",
        "bí ẩn": "Mystery",
        "mystery": "Mystery",
        "giật gân": "Thriller",
        "thriller": "Thriller",
        "thần thoại": "Fantasy",
        "fantasy": "Fantasy",
        "lịch sử": "History",
        "history": "History",
        "chiến tranh": "War",
        "war": "War",
        "tài liệu": "Documentary",
        "documentary": "Documentary",
        "gia đình": "Family",
        "family": "Family",
        "nhạc": "Music",
        "ca nhạc": "Music",
        "âm nhạc": "Music",
        "music": "Music",
        "miền tây": "Western",
        "western": "Western"
    }
    return mapping.get(genre_lower, genre_str)

def search_movies_tool(df: pd.DataFrame, filters: dict, top_k: int = 5) -> pd.DataFrame:
    """
    Filter movies from dataframe based on parsed filters.
    This tool does not call LLM.
    """
    try:
        if df.empty:
            return pd.DataFrame()
            
        result = df.copy()

        # Xác định cột votes an toàn
        votes_col = COL_VOTES if COL_VOTES in result.columns else ("Votes" if "Votes" in result.columns else None)
        if votes_col and not filters.get("title"):
            try:
                result = result[result[votes_col] >= 1000]
            except Exception:
                pass

        # Bắt buộc có country (áp dụng cho mọi truy vấn)
        if "countries_origin" in result.columns:
            try:
                result = result[
                    result["countries_origin"].astype(str).str.strip().ne('') &
                    result["countries_origin"].notna()
                ]
            except Exception:
                pass

        # Lọc thể loại (Genre)
        if filters.get("genre") and COL_GENRE in result.columns:
            try:
                genre_val = normalize_genre(filters["genre"])
                result = result[result[COL_GENRE].astype(str).str.contains(
                    genre_val, case=False, na=False
                )]
            except Exception:
                pass

        # Lọc đạo diễn (Director)
        if filters.get("director") and COL_DIRECTOR in result.columns:
            try:
                result = result[result[COL_DIRECTOR].astype(str).str.contains(
                    filters["director"], case=False, na=False
                )]
            except Exception:
                pass

        # Lọc diễn viên (Stars)
        if filters.get("star") and COL_STARS in result.columns:
            try:
                result = result[result[COL_STARS].astype(str).str.contains(
                    filters["star"], case=False, na=False
                )]
            except Exception:
                pass

        # Lọc quốc gia (Country)
        if filters.get("country") and "countries_origin" in result.columns:
            try:
                from chatbot.data_loader import load_country_aliases
                country_aliases = load_country_aliases()
                country_query = str(filters["country"]).strip().lower()
                standard_country = country_aliases.get(country_query)
                if not standard_country:
                    query_stripped = ''.join(c for c in unicodedata.normalize('NFD', country_query) if unicodedata.category(c) != 'Mn')
                    for k, v in country_aliases.items():
                        k_stripped = ''.join(c for c in unicodedata.normalize('NFD', k) if unicodedata.category(c) != 'Mn')
                        if k_stripped == query_stripped:
                            standard_country = v
                            break
                if not standard_country:
                    standard_country = filters["country"]
                
                result = result[result["countries_origin"].astype(str).str.contains(
                    standard_country, case=False, na=False
                )]
            except Exception:
                pass

        # Lọc tên phim (Title)
        if filters.get("title") and COL_TITLE in result.columns:
            try:
                result = result[result[COL_TITLE].astype(str).str.contains(
                    filters["title"], case=False, na=False
                )]
            except Exception:
                pass

        # Lọc năm phát hành tối thiểu (Year min)
        if filters.get("year_min") and COL_YEAR in result.columns:
            try:
                result = result[result[COL_YEAR] >= float(filters["year_min"])]
            except Exception:
                pass

        # Lọc năm phát hành tối đa (Year max)
        if filters.get("year_max") and COL_YEAR in result.columns:
            try:
                result = result[result[COL_YEAR] <= float(filters["year_max"])]
            except Exception:
                pass

        # Lọc điểm số đánh giá tối thiểu (Rating min)
        if filters.get("rating_min") and COL_RATING in result.columns:
            try:
                result = result[result[COL_RATING] >= float(filters["rating_min"])]
            except Exception:
                pass

        # Lọc giải thưởng Oscar
        if filters.get("has_oscar") and COL_OSCAR in result.columns:
            try:
                result = result[result[COL_OSCAR] == 1]
            except Exception:
                pass

        # Lọc giải thưởng nói chung
        if filters.get("has_awards") and COL_AWARDS in result.columns:
            try:
                result = result[result[COL_AWARDS] == 1]
            except Exception:
                pass

        # Lọc thời lượng tối thiểu (phút)
        if filters.get("duration_min") and COL_DURATION in result.columns:
            try:
                result = result[result[COL_DURATION] >= float(filters["duration_min"])]
            except Exception:
                pass

        # Lọc thời lượng tối đa (phút)
        if filters.get("duration_max") and COL_DURATION in result.columns:
            try:
                result = result[result[COL_DURATION] <= float(filters["duration_max"])]
            except Exception:
                pass

        # Lọc Metascore tối thiểu
        if filters.get("meta_score_min") and COL_METASCORE in result.columns:
            try:
                result = result[result[COL_METASCORE] >= float(filters["meta_score_min"])]
            except Exception:
                pass

        # Sắp xếp kết quả
        sort_by = filters.get("sort_by")
        sort_order = filters.get("sort_order", "desc")
        ascending = (sort_order == "asc")

        if sort_by == "votes" and votes_col:
            try:
                result = result.sort_values(votes_col, ascending=ascending)
            except Exception:
                pass
        elif sort_by == "year" and COL_YEAR in result.columns:
            try:
                result = result.sort_values(COL_YEAR, ascending=ascending)
            except Exception:
                pass
        elif sort_by == "metascore" and COL_METASCORE in result.columns:
            try:
                result = result.sort_values(COL_METASCORE, ascending=ascending)
            except Exception:
                pass
        else:
            # Mặc định sắp xếp theo rating giảm dần
            if COL_RATING in result.columns:
                try:
                    result = result.sort_values(COL_RATING, ascending=ascending if sort_by == "rating" else False)
                except Exception:
                    pass
                    
        return result.head(top_k).copy()
    except Exception:
        return pd.DataFrame()

def semantic_search_tool(query: str, df: pd.DataFrame, index, model, top_k: int = 100) -> pd.DataFrame:
    """
    Perform semantic search on description column using FAISS index and sentence transformer model.
    This tool does not call LLM.
    """
    try:
        if df.empty or index is None or model is None or not query:
            return df.copy()
            
        q_emb = model.encode([query]).astype('float32')
        _, indices = index.search(q_emb, top_k)
        # Chỉ lấy các index hợp lệ nằm trong khoảng dòng của df
        valid_indices = [idx for idx in indices[0] if 0 <= idx < len(df)]
        return df.iloc[valid_indices].copy()
    except Exception:
        return df.copy()

def get_movie_detail_tool(df: pd.DataFrame, title: str) -> pd.DataFrame:
    """
    Get detailed information of a movie by its title.
    This tool does not call LLM.
    """
    try:
        if df.empty or not title or COL_TITLE not in df.columns:
            return pd.DataFrame()
            
        # Tìm chính xác trước
        match = df[df[COL_TITLE].astype(str).str.lower() == title.strip().lower()]
        if match.empty:
            # Khớp chuỗi con nếu không có phim khớp chính xác
            match = df[df[COL_TITLE].astype(str).str.contains(title.strip(), case=False, na=False)]
            
        if not match.empty:
            # Sắp xếp các kết quả tìm thấy theo số lượt vote (độ phổ biến) giảm dần
            votes_col = "num_votes" if "num_votes" in match.columns else ("Votes" if "Votes" in match.columns else None)
            if votes_col:
                try:
                    if votes_col == "Votes":
                        # Làm sạch cột Votes thô để phục vụ sắp xếp
                        def clean_votes(val):
                            if pd.isna(val):
                                return 0
                            val_str = str(val).strip().upper()
                            if not val_str:
                                return 0
                            try:
                                if val_str.endswith('K'):
                                    return int(float(val_str[:-1]) * 1000)
                                elif val_str.endswith('M'):
                                    return int(float(val_str[:-1]) * 1000000)
                                val_str = val_str.replace(',', '')
                                return int(float(val_str))
                            except Exception:
                                return 0
                        temp_votes = match["Votes"].apply(clean_votes)
                        match = match.loc[temp_votes.sort_values(ascending=False).index]
                    else:
                        match = match.sort_values(by=votes_col, ascending=False)
                except Exception:
                    pass
            
        return match.head(1).copy()
    except Exception:
        return pd.DataFrame()

def recommend_by_actor_tool(df: pd.DataFrame, actor: str, top_k: int = 5) -> pd.DataFrame:
    """
    Recommend movies featuring a specific actor, sorted by rating.
    This tool does not call LLM.
    """
    try:
        if df.empty or not actor or COL_STARS not in df.columns:
            return pd.DataFrame()
            
        result = df[df[COL_STARS].astype(str).str.contains(actor, case=False, na=False)]
        
        # Bỏ qua các phim có ít lượt vote để đảm bảo chất lượng
        votes_col = COL_VOTES if COL_VOTES in result.columns else ("Votes" if "Votes" in result.columns else None)
        if votes_col:
            result = result[result[votes_col] >= 1000]
            
        if COL_RATING in result.columns:
            result = result.sort_values(COL_RATING, ascending=False)
            
        return result.head(top_k).copy()
    except Exception:
        return pd.DataFrame()

def recommend_by_director_tool(df: pd.DataFrame, director: str, top_k: int = 5) -> pd.DataFrame:
    """
    Recommend movies directed by a specific director, sorted by rating.
    This tool does not call LLM.
    """
    try:
        if df.empty or not director or COL_DIRECTOR not in df.columns:
            return pd.DataFrame()
            
        result = df[df[COL_DIRECTOR].astype(str).str.contains(director, case=False, na=False)]
        
        # Bỏ qua các phim có ít lượt vote
        votes_col = COL_VOTES if COL_VOTES in result.columns else ("Votes" if "Votes" in result.columns else None)
        if votes_col:
            result = result[result[votes_col] >= 1000]
            
        if COL_RATING in result.columns:
            result = result.sort_values(COL_RATING, ascending=False)
            
        return result.head(top_k).copy()
    except Exception:
        return pd.DataFrame()

def compare_movies_tool(df: pd.DataFrame, movie_titles: list) -> pd.DataFrame:
    """
    Compare multiple movies based on their titles.
    This tool does not call LLM.
    """
    try:
        if df.empty or not movie_titles or COL_TITLE not in df.columns:
            return pd.DataFrame()
            
        matched_dfs = []
        for title in movie_titles:
            if not title or not isinstance(title, str):
                continue
            # Tìm chính xác trước
            match = df[df[COL_TITLE].astype(str).str.lower() == title.strip().lower()]
            if match.empty:
                # Khớp chuỗi con
                match = df[df[COL_TITLE].astype(str).str.contains(title.strip(), case=False, na=False)]
            if not match.empty:
                matched_dfs.append(match.head(1))
                
        if not matched_dfs:
            return pd.DataFrame()
            
        result = pd.concat(matched_dfs)
        if COL_TITLE in result.columns:
            result = result.drop_duplicates(subset=[COL_TITLE])
        return result.copy()
    except Exception:
        return pd.DataFrame()

# ============================================================
# LANGCHAIN TOOL WRAPPERS & ADVANCED RETRIEVAL (similar movies)
# ============================================================
import re
from langchain_core.tools import tool
from chatbot.config import SEMANTIC_TOP_K

def find_similar_movies(df: pd.DataFrame, index, model, user_input: str, filters: dict) -> tuple[pd.DataFrame, bool]:
    """
    Truy vấn các phim tương tự phim được hỏi trong user_input:
    1. Nhận diện tên phim mẫu bằng regex hoặc filter title.
    2. Lấy thông tin chi tiết phim gốc.
    3. Thực hiện Semantic Search bằng mô tả của phim gốc.
    4. Loại bỏ phim gốc khỏi kết quả và áp dụng bộ lọc phụ.
    """
    if index is None or model is None:
        return pd.DataFrame(), False

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
        
    if candidate_title:
        candidate_title = re.sub(r'^(bộ\s+phim|phim|bộ|cái|con|những|các|tựa|tựa\s+phim)\s+', '', candidate_title, flags=re.IGNORECASE).strip()
        
        base_movie = get_movie_detail_tool(df, candidate_title)
        if not base_movie.empty:
            base_title = base_movie[COL_TITLE].values[0]
            base_desc = base_movie[COL_OVERVIEW].values[0] if COL_OVERVIEW in base_movie.columns else ""
            
            if base_desc:
                filtered_df = semantic_search_tool(base_desc, df, index, model, top_k=SEMANTIC_TOP_K)
                if COL_TITLE in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[COL_TITLE].str.lower() != base_title.lower()]
                filters_copy = filters.copy()
                filters_copy["title"] = None
                filtered_df = search_movies_tool(filtered_df, filters_copy)
                return filtered_df, True
                
        title_matches = df[df[COL_TITLE].astype(str).str.contains(candidate_title, case=False, na=False)]
        if not title_matches.empty:
            filtered_df = search_movies_tool(title_matches, filters)
            return filtered_df, True
        else:
            filtered_df = semantic_search_tool(candidate_title, df, index, model, top_k=SEMANTIC_TOP_K)
            filters_copy = filters.copy()
            filters_copy["title"] = None
            filtered_df = search_movies_tool(filtered_df, filters_copy)
            return filtered_df, True
            
    return pd.DataFrame(), False

@tool
def movie_search_tool(df: pd.DataFrame, filters: dict, top_k: int = 5) -> pd.DataFrame:
    """
    Tìm kiếm và lọc phim trong cơ sở dữ liệu (DataFrame) theo các bộ lọc:
    thể loại (genre), đạo diễn (director), diễn viên (star), tên phim (title), năm, điểm số, và sắp xếp.
    Không gọi LLM.
    """
    return search_movies_tool(df, filters, top_k)

@tool
def movie_info_tool(df: pd.DataFrame, title: str) -> pd.DataFrame:
    """
    Truy vấn thông tin chi tiết của một bộ phim cụ thể trong cơ sở dữ liệu dựa trên tiêu đề (title).
    Không gọi LLM.
    """
    return get_movie_detail_tool(df, title)
