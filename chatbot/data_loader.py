import os
import ast
import json
import pandas as pd
import streamlit as st
import faiss
import torch
from sentence_transformers import SentenceTransformer
from chatbot.config import (
    MOVIE_DATA_PATH, KEYWORD_DICT_PATH, ALIASES_PATH, INDEX_PATH,
    CHATBOT_DIR, COL_GENRE, COL_DIRECTOR, COL_STARS, COL_RATING, COL_YEAR
)
from chatbot.retrieval.bm25_retriever import build_bm25_index


@st.cache_data
def load_data():
    """
    Nạp dữ liệu từ file Parquet và làm sạch định dạng.
    """
    df = pd.read_parquet(MOVIE_DATA_PATH)
    
    # Làm sạch imdb_id: bỏ dấu gạch chéo cuối
    if 'imdb_id' in df.columns:
        df['imdb_id'] = df['imdb_id'].astype(str).str.strip().str.rstrip('/')
        
    # Rename các cột cho tương thích ngược với logic cũ
    df = df.rename(columns={
        "title": "Title",
        "year": "Year",
        "rating": "Rating",
        "Movie_Link": "Movie Link",
        "votes": "Votes",
        "languages": "Languages"
    })
    
    # Làm sạch các cột danh sách chuỗi (genres, directors, stars, Languages, countries_origin)
    def clean_list_column(val):
        if pd.isna(val):
            return ""
        val_str = str(val).strip()
        if not val_str:
            return ""
        if val_str.startswith("[") and val_str.endswith("]"):
            try:
                lst = ast.literal_eval(val_str)
                lst = [item for item in lst if item and item != "None"]
                return ", ".join(lst)
            except Exception:
                pass
        # Hỗ trợ phân tách bằng dấu đứng "|" cho dữ liệu mới
        if "|" in val_str:
            lst = [item.strip() for item in val_str.split("|") if item.strip() and item.strip() != "None"]
            return ", ".join(lst)
        return val_str
        
    df[COL_GENRE] = df[COL_GENRE].apply(clean_list_column)
    df[COL_DIRECTOR] = df[COL_DIRECTOR].apply(clean_list_column)
    df[COL_STARS] = df[COL_STARS].apply(clean_list_column)
    if "Languages" in df.columns:
        df["Languages"] = df["Languages"].apply(clean_list_column)
    if "countries_origin" in df.columns:
        df["countries_origin"] = df["countries_origin"].apply(clean_list_column)
    
    # Ép kiểu dữ liệu số
    df[COL_RATING] = pd.to_numeric(df[COL_RATING], errors="coerce")
    df[COL_YEAR]   = pd.to_numeric(df[COL_YEAR],   errors="coerce")
    
    # Chuẩn hoá cột decade từ string (ví dụ '2010s') sang số (ví dụ 2010)
    if 'decade' in df.columns:
        def clean_decade(val):
            if pd.isna(val):
                return None
            val_str = str(val).strip()
            if val_str.endswith('s'):
                val_str = val_str[:-1]
            try:
                return float(val_str)
            except Exception:
                return None
        df['decade'] = df['decade'].apply(clean_decade)
    
    # Chuẩn hoá lượt vote
    def clean_votes(val):
        if pd.isna(val):
            return 0
        if isinstance(val, (int, float)):
            return int(val)
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
            
    df['num_votes'] = df['Votes'].apply(clean_votes)
    
    # Tái tạo cột final_context động tương thích ngược (tối ưu hóa tốc độ nạp)
    def format_val(val):
        if pd.isna(val) or val == "":
            return None
        val_str = str(val).strip()
        if val_str.lower() in ['nan', 'none']:
            return None
        return val_str

    def clean_num_str(val):
        if pd.isna(val) or val == "":
            return None
        try:
            v_num = float(val)
            if v_num.is_integer():
                return str(int(v_num))
            return str(v_num)
        except Exception:
            return str(val).strip()

    records = df.to_dict('records')
    contexts = []
    for row in records:
        parts = []
        
        title = format_val(row.get('Title'))
        if title:
            parts.append(f"Title: {title}")
            
        desc = format_val(row.get('description'))
        if desc:
            parts.append(f"Description: {desc}")
            
        genres = format_val(row.get('genres'))
        if genres:
            parts.append(f"Genres: {genres}")
            
        directors = format_val(row.get('directors'))
        if directors:
            parts.append(f"Directors: {directors}")
            
        writers = format_val(row.get('writers'))
        if writers:
            parts.append(f"Writers: {writers}")
            
        stars = format_val(row.get('stars'))
        if stars:
            parts.append(f"Stars: {stars}")
            
        rating = clean_num_str(row.get('Rating'))
        if rating:
            parts.append(f"Rating: {rating}")
            
        votes = clean_num_str(row.get('Votes'))
        if votes:
            parts.append(f"Votes: {votes}")
            
        meta = clean_num_str(row.get('meta_score'))
        if meta:
            parts.append(f"Meta Score: {meta}")
            
        year = clean_num_str(row.get('Year'))
        if year:
            parts.append(f"Year: {year}")
            
        duration = clean_num_str(row.get('duration_min'))
        if duration:
            parts.append(f"Duration: {duration} minutes")
            
        country = format_val(row.get('countries_origin'))
        if country:
            parts.append(f"Country: {country}")
            
        language = format_val(row.get('Languages'))
        if language:
            parts.append(f"Language: {language}")
            
        prod = format_val(row.get('production_company'))
        if prod:
            parts.append(f"Production Company: {prod}")
            
        awards = format_val(row.get('awards_content'))
        if awards:
            parts.append(f"Awards: {awards}")
            
        imdb_id = format_val(row.get('imdb_id'))
        if imdb_id:
            parts.append(f"IMDb ID: {imdb_id}")
            
        if not parts:
            contexts.append("")
            continue
            
        result = parts[0]
        for part in parts[1:]:
            if result.endswith('.') or result.endswith('!') or result.endswith('?'):
                result += " " + part
            else:
                result += ". " + part
        contexts.append(result)
        
    df['final_context'] = contexts
    return df

@st.cache_data
def load_keyword_dict():
    """
    Nạp từ điển thực thể và từ khóa đặc trưng.
    """
    with open(KEYWORD_DICT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_aliases():
    """
    Nạp từ điển bí danh / tên viết tắt.
    """
    try:
        with open(ALIASES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@st.cache_resource
def load_faiss_index():
    """
    Nạp tệp chỉ mục FAISS chứa vector mô tả phim.
    """
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Chỉ mục FAISS chưa được tạo.")
    return faiss.read_index(INDEX_PATH)

@st.cache_resource
def load_embedder_model():
    """
    Nạp mô hình SentenceTransformer sinh vector nhúng.
    """
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Chỉ mục FAISS chưa được tạo.")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

@st.cache_resource
def load_bm25_index(df):
    """
    Xây dựng hoặc nạp chỉ mục BM25Okapi được lưu trong bộ nhớ cache.
    """
    return build_bm25_index(df)

@st.cache_data
def load_country_aliases():
    """
    Nạp từ điển bí danh / dịch tên quốc gia.
    """
    country_aliases_path = os.path.join(CHATBOT_DIR, "country_aliases.json")
    try:
        with open(country_aliases_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

