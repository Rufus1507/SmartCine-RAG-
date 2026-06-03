import os
import sys
import pandas as pd
import numpy as np
import time
import torch
from sentence_transformers import SentenceTransformer
import faiss

# Thêm thư mục chatbot vào path để có cấu hình đồng nhất
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Đảm bảo in UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMDB_DATA_PATH = os.path.join(BASE_DIR, "data", "imdb_movies_all_years.csv")
ADVANCED_DATA_PATH = os.path.join(BASE_DIR, "data", "advanced_movies_details_all_years.csv")
INDEX_PATH = os.path.join(BASE_DIR, "chatbot", "description_embeddings.index")

# Bản sao hàm load_data của app.py để đảm bảo tính nhất quán
def load_data():
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
        
    df = pd.merge(df1, df2, left_on="Movie Link", right_on="link", how="inner")
    return df

def main():
    print("🚀 Đang tải dữ liệu phim...")
    df = load_data()
    print(f"✅ Đã tải xong {len(df):,} phim.")
    
    # Chuẩn bị cột description
    print("🧹 Chuẩn bị văn bản mô tả...")
    descriptions = df['description'].fillna('').astype(str).tolist()
    
    # Chọn thiết bị chạy (GPU nếu có)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Thiết bị chạy embedding: {device.upper()}")
    
    print("🧠 Đang tải mô hình sentence-transformers 'paraphrase-multilingual-MiniLM-L12-v2'...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
    
    print("⏳ Bắt đầu tính toán embeddings (sẽ mất một chút thời gian)...")
    start_time = time.time()
    
    # Tính toán embedding (chia batch và bật progress bar)
    embeddings = model.encode(
        descriptions, 
        batch_size=128, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    
    elapsed = time.time() - start_time
    print(f"✅ Tính toán xong embeddings trong {elapsed:.2f} giây ({len(descriptions)/elapsed:.1f} câu/giây).")
    
    # Tạo chỉ mục FAISS
    print("⚙️ Tạo chỉ mục FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    # Lưu chỉ mục
    print(f"💾 Đang lưu chỉ mục FAISS vào: {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)
    print("🎉 Hoàn thành! Hệ thống đã sẵn sàng sử dụng Semantic Search.")

if __name__ == "__main__":
    main()
