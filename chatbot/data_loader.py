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
    Nạp dữ liệu từ file gộp movie_master.csv và làm sạch định dạng.
    """
    try:
        df = pd.read_csv(MOVIE_DATA_PATH, encoding='utf-8', low_memory=False)
        df.columns = df.columns.str.replace(r'^\xef\xbb\xbf', '', regex=True)
    except Exception:
        try:
            df = pd.read_csv(MOVIE_DATA_PATH, encoding='latin-1', low_memory=False)
            df.columns = df.columns.str.replace(r'^\xef\xbb\xbf', '', regex=True)
        except Exception:
            df = pd.read_csv(MOVIE_DATA_PATH, low_memory=False)
            
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

