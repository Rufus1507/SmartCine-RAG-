import os
import sys
import pandas as pd
import numpy as np
import time
import torch
from sentence_transformers import SentenceTransformer
import faiss

# Thêm thư mục gốc vào path để import dạng 'from chatbot.xyz'
chatbot_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(chatbot_dir)
sys.path.append(workspace_dir)

# Đảm bảo in UTF-8
sys.stdout.reconfigure(encoding='utf-8')

from chatbot.config import PROFILE_INDEX_PATH
from chatbot.data_loader import load_data

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

def main():
    print("🚀 Đang tải dữ liệu phim...")
    df = load_data()
    print(f"✅ Đã tải xong {len(df):,} phim.")
    
    df_filtered = df.reset_index(drop=True)
    print(f"✅ Số lượng phim đưa vào profile index: {len(df_filtered):,}")
    
    print("🧹 Chuẩn bị văn bản profile cho mỗi phim...")
    profiles = df_filtered.apply(make_profile, axis=1).tolist()
    
    # Chọn thiết bị chạy (GPU nếu có)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Thiết bị chạy embedding: {device.upper()}")
    
    print("🧠 Đang tải mô hình sentence-transformers 'paraphrase-multilingual-MiniLM-L12-v2'...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
    
    print("⏳ Bắt đầu tính toán profile embeddings...")
    start_time = time.time()
    
    # Tính toán embedding
    embeddings = model.encode(
        profiles, 
        batch_size=128, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    
    elapsed = time.time() - start_time
    print(f"✅ Tính toán xong embeddings trong {elapsed:.2f} giây ({len(profiles)/elapsed:.1f} câu/giây).")
    
    # Tạo chỉ mục FAISS
    print("⚙️ Tạo chỉ mục FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    # Lưu chỉ mục
    print(f"💾 Đang lưu chỉ mục FAISS vào: {PROFILE_INDEX_PATH}")
    faiss.write_index(index, PROFILE_INDEX_PATH)
    print("🎉 Hoàn thành! Chỉ mục profile phim đã sẵn sàng.")

if __name__ == "__main__":
    main()
