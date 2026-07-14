import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
import faiss
import torch
from sentence_transformers import SentenceTransformer
from chatbot.config import CHATBOT_DIR

REPRESENTATION_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_A_PATH = os.path.join(CHATBOT_DIR, "representation_a.index")
INDEX_B_PATH = os.path.join(CHATBOT_DIR, "representation_b.index")
INDEX_C_PATH = os.path.join(CHATBOT_DIR, "representation_c.index")

# Load content keywords from keyword_dict.json
@st.cache_resource
def load_content_keywords():
    keyword_dict_path = os.path.join(CHATBOT_DIR, "keyword_dict.json")
    try:
        with open(keyword_dict_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return {k for k, v in d.items() if v == "search_content"}
    except Exception:
        return set()

def extract_keywords_from_text(text: str, kw_set: set) -> str:
    if not isinstance(text, str) or pd.isna(text):
        return ""
    # Clean text
    words = re.findall(r'\b\w+\b', text.lower())
    matched = [w for w in words if w in kw_set]
    return ", ".join(list(set(matched)))

def make_profile_version_a(row) -> str:
    """Version A: Description"""
    desc = str(row.get('description', '')).strip()
    if pd.isna(row.get('description')) or desc.lower() == 'nan':
        return ""
    return desc

def make_profile_version_b(row) -> str:
    """Version B: Genre + Description"""
    genre = str(row.get('genres', '')).strip()
    desc = str(row.get('description', '')).strip()
    if pd.isna(row.get('description')) or desc.lower() == 'nan':
        desc = ""
    parts = []
    if genre:
        parts.append(f"Genre: {genre}")
    if desc:
        parts.append(f"Description: {desc}")
    return "\n".join(parts)

def make_profile_version_c(row, kw_set=None) -> str:
    """Version C: Genre + Description + Keywords"""
    genre = str(row.get('genres', '')).strip()
    desc = str(row.get('description', '')).strip()
    if pd.isna(row.get('description')) or desc.lower() == 'nan':
        desc = ""
        
    if kw_set is None:
        kw_set = load_content_keywords()
        
    keywords = extract_keywords_from_text(desc, kw_set)
    
    parts = []
    if genre:
        parts.append(f"Genre: {genre}")
    if desc:
        parts.append(f"Description: {desc}")
    if keywords:
        parts.append(f"Keywords: {keywords}")
    return "\n".join(parts)

def make_profile(row, version: str) -> str:
    if version.upper() == 'A':
        return make_profile_version_a(row)
    elif version.upper() == 'B':
        return make_profile_version_b(row)
    elif version.upper() == 'C':
        return make_profile_version_c(row)
    else:
        raise ValueError(f"Unknown version: {version}")

def generate_embeddings_for_version(df_filtered: pd.DataFrame, version: str, model, output_path: str):
    """
    Generates embeddings and builds a FAISS index for a specific profile version.
    """
    print(f"Generating embeddings for Version {version}...")
    kw_set = load_content_keywords() if version.upper() == 'C' else None
    
    profiles = []
    for _, row in df_filtered.iterrows():
        if version.upper() == 'C':
            profiles.append(make_profile_version_c(row, kw_set))
        else:
            profiles.append(make_profile(row, version))
            
    # Run embedding
    embeddings = model.encode(
        profiles,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Save FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, output_path)
    print(f"Saved Version {version} index to {output_path}")
