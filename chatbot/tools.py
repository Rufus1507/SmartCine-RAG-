import pandas as pd
import numpy as np

# Mapping tên cột mặc định trong CSV của CineBot
COL_TITLE    = "Title"               # tên phim
COL_GENRE    = "genres"              # thể loại
COL_DIRECTOR = "directors"           # đạo diễn
COL_STARS    = "stars"               # danh sách diễn viên
COL_YEAR     = "Year"                # năm phát hành
COL_RATING   = "Rating"              # điểm IMDB
COL_OVERVIEW = "description"         # mô tả phim
COL_VOTES    = "num_votes"           # số lượt vote đã làm sạch

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

        # Lọc thể loại (Genre)
        if filters.get("genre") and COL_GENRE in result.columns:
            try:
                result = result[result[COL_GENRE].astype(str).str.contains(
                    filters["genre"], case=False, na=False
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
