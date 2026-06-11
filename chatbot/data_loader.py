import os
import ast
import json
import pandas as pd
import streamlit as st
import faiss
import torch
from sentence_transformers import SentenceTransformer
from chatbot.config import (
    IMDB_DATA_PATH, ADVANCED_DATA_PATH, KEYWORD_DICT_PATH, ALIASES_PATH, INDEX_PATH,
    CHATBOT_DIR, COL_GENRE, COL_DIRECTOR, COL_STARS, COL_RATING, COL_YEAR
)
from chatbot.bm25_retriever import build_bm25_index

@st.cache_data
def load_data():
    """
    Nạp dữ liệu từ 2 file CSV, thực hiện kết hợp (merge) và làm sạch định dạng.
    """
    try:
        df1 = pd.read_csv(IMDB_DATA_PATH, encoding='latin-1', low_memory=False)
        df1.columns = df1.columns.str.replace(r'^\xef\xbb\xbf', '', regex=True)
    except Exception:
        df1 = pd.read_csv(IMDB_DATA_PATH, low_memory=False)
        
    try:
        df2 = pd.read_csv(ADVANCED_DATA_PATH, encoding='latin-1', low_memory=False)
        df2.columns = df2.columns.str.replace(r'^\xef\xbb\xbf', '', regex=True)
    except Exception:
        df2 = pd.read_csv(ADVANCED_DATA_PATH, low_memory=False)
        
    # Merge 2 bảng thông qua link phim
    df = pd.merge(df1, df2, left_on="Movie Link", right_on="link", how="inner")
    
    # Làm sạch các cột danh sách chuỗi (genres, directors, stars)
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
        return val_str
        
    df[COL_GENRE] = df[COL_GENRE].apply(clean_list_column)
    df[COL_DIRECTOR] = df[COL_DIRECTOR].apply(clean_list_column)
    df[COL_STARS] = df[COL_STARS].apply(clean_list_column)
    
    # Ép kiểu dữ liệu số
    df[COL_RATING] = pd.to_numeric(df[COL_RATING], errors="coerce")
    df[COL_YEAR]   = pd.to_numeric(df[COL_YEAR],   errors="coerce")
    
    # Chuẩn hoá lượt vote
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

