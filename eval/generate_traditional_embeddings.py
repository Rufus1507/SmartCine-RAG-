import os
import sys
import pandas as pd
import numpy as np
import time
import torch
from sentence_transformers import SentenceTransformer
import faiss

# Thêm thư mục gốc vào path để import
eval_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(eval_dir)
sys.path.append(workspace_dir)

# Đảm bảo in UTF-8
sys.stdout.reconfigure(encoding='utf-8')

from chatbot.data_loader import load_data

def main():
    print("🚀 [Traditional RAG] Đang tải dữ liệu phim...")
    df = load_data()
    print(f"✅ Đã tải xong {len(df):,} phim.")
    
    # Tạo văn bản ngữ cảnh tinh gọn (embedding_context) phục vụ biểu diễn vector ngữ nghĩa
    print("🧹 Chuẩn bị văn bản ngữ cảnh tinh gọn (embedding_context)...")
    # Nối Title, description, genres để có context ngữ nghĩa rõ ràng, không bị tràn giới hạn token của MiniLM
    embedding_contexts = []
    for _, row in df.iterrows():
        parts = []
        title = row.get('Title')
        if pd.notna(title) and str(title).strip():
            parts.append(f"Title: {str(title).strip()}")
        desc = row.get('description')
        if pd.notna(desc) and str(desc).strip():
            parts.append(f"Description: {str(desc).strip()}")
        genres = row.get('genres')
        if pd.notna(genres) and str(genres).strip():
            parts.append(f"Genres: {str(genres).strip()}")
        embedding_contexts.append(". ".join(parts) if parts else "")
    
    # Chọn thiết bị chạy (GPU nếu có)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Thiết bị chạy embedding: {device.upper()}")
    
    print("🧠 Đang tải mô hình sentence-transformers 'paraphrase-multilingual-MiniLM-L12-v2'...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
    
    print("⏳ Bắt đầu tính toán embeddings cho RAG truyền thống...")
    start_time = time.time()
    
    # Tính embedding với batch_size lớn + progress bar (SentenceTransformers mới tích hợp đa luồng)
    embeddings = model.encode(
        embedding_contexts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    elapsed = time.time() - start_time
    print(f"✅ Tính toán xong embeddings trong {elapsed:.2f} giây ({len(embedding_contexts)/elapsed:.1f} câu/giây).")
    
    # Tạo chỉ mục FAISS sử dụng Inner Product (đối với vector chuẩn hóa L2, IP tương đương Cosine Similarity)
    print("⚙️ Tạo chỉ mục FAISS với Cosine Similarity...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings_normalized = (embeddings / norms).astype('float32')
    
    dimension = embeddings_normalized.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_normalized)
    
    # Lưu chỉ mục
    traditional_index_path = os.path.join(workspace_dir, "data", "traditional_context.index")
    print(f"💾 Đang lưu chỉ mục FAISS vào: {traditional_index_path}")
    faiss.write_index(index, traditional_index_path)
    print("🎉 Hoàn thành tạo chỉ mục RAG truyền thống!")

if __name__ == "__main__":
    main()
