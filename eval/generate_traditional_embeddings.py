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
    
    # Từ điển ánh xạ thể loại & quốc gia Anh - Việt phục vụ vector matching ngữ nghĩa song ngữ
    GENRE_MAP = {
        "Action": "Hành động", "Adventure": "Phiêu lưu", "Animation": "Hoạt hình",
        "Comedy": "Hài, Hài hước", "Crime": "Tội phạm", "Documentary": "Tài liệu",
        "Drama": "Tâm lý, Chính kịch", "Family": "Gia đình", "Fantasy": "Kỳ ảo",
        "History": "Lịch sử", "Horror": "Kinh dị, Ma quỷ", "Music": "Âm nhạc",
        "Mystery": "Bí ẩn, Thám tử", "Romance": "Tình cảm, Lãng mạn", "Sci-Fi": "Khoa học viễn tưởng",
        "Sport": "Thể thao", "Thriller": "Giật gân", "War": "Chiến tranh", "Western": "Viễn tây"
    }

    COUNTRY_MAP = {
        "United States": "Mỹ, USA", "USA": "Mỹ, USA", "United Kingdom": "Anh, UK", "UK": "Anh, UK",
        "South Korea": "Hàn Quốc, Korea", "Korea": "Hàn Quốc, Korea", "Japan": "Nhật Bản",
        "France": "Pháp", "Germany": "Đức", "China": "Trung Quốc", "India": "Ấn Độ",
        "Italy": "Ý", "Spain": "Tây Ban Nha", "Canada": "Canada", "Australia": "Úc", "Vietnam": "Việt Nam"
    }

    print("🧹 Chuẩn bị văn bản ngữ cảnh đa thuộc tính song ngữ (embedding_context)...")
    embedding_contexts = []
    for _, row in df.iterrows():
        parts = []
        
        # Tiêu đề & Năm
        title = row.get('Title')
        year = row.get('Year')
        if pd.notna(title) and str(title).strip():
            year_str = f" ({int(year)})" if pd.notna(year) and str(year).strip() != "" else ""
            parts.append(f"Tên phim (Title): {str(title).strip()}{year_str}")

        # Thể loại (Anh - Việt)
        genres = row.get('genres')
        if pd.notna(genres) and str(genres).strip():
            g_raw = str(genres).strip()
            vi_genres = []
            for g in g_raw.split(','):
                g_clean = g.strip()
                if g_clean in GENRE_MAP:
                    vi_genres.append(GENRE_MAP[g_clean])
            vi_genre_str = f" ({', '.join(vi_genres)})" if vi_genres else ""
            parts.append(f"Thể loại (Genres): {g_raw}{vi_genre_str}")

        # Điểm IMDb & Thời lượng
        rating = row.get('Rating')
        if pd.notna(rating):
            parts.append(f"Điểm IMDb (Rating): {rating}/10")

        duration = row.get('duration_min')
        if pd.notna(duration):
            try:
                parts.append(f"Thời lượng (Duration): {int(float(duration))} phút (minutes)")
            except Exception:
                pass

        # Quốc gia (Anh - Việt)
        country = row.get('countries_origin')
        if pd.notna(country) and str(country).strip():
            c_raw = str(country).strip()
            vi_countries = []
            for c in c_raw.split(','):
                c_clean = c.strip()
                if c_clean in COUNTRY_MAP:
                    vi_countries.append(COUNTRY_MAP[c_clean])
            vi_country_str = f" ({', '.join(vi_countries)})" if vi_countries else ""
            parts.append(f"Quốc gia (Country): {c_raw}{vi_country_str}")

        # Giải thưởng (Oscar)
        has_oscar = row.get('has_oscar')
        awards = row.get('awards_content')
        award_parts = []
        if pd.notna(has_oscar) and (str(has_oscar) == "1" or has_oscar is True or has_oscar == 1):
            award_parts.append("Đoạt giải Oscar (Oscar Winner)")
        if pd.notna(awards) and str(awards).strip():
            award_parts.append(str(awards).strip())
        if award_parts:
            parts.append(f"Giải thưởng (Awards): {', '.join(award_parts)}")

        # Đạo diễn & Diễn viên
        directors = row.get('directors')
        if pd.notna(directors) and str(directors).strip():
            parts.append(f"Đạo diễn (Directors): {str(directors).strip()}")

        stars = row.get('stars')
        if pd.notna(stars) and str(stars).strip():
            parts.append(f"Diễn viên (Stars): {str(stars).strip()}")

        # Mô tả cốt truyện
        desc = row.get('description')
        if pd.notna(desc) and str(desc).strip():
            parts.append(f"Mô tả nội dung (Overview): {str(desc).strip()}")

        embedding_contexts.append(". ".join(parts) if parts else "")
    
    # Chọn thiết bị chạy (GPU nếu có)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Thiết bị chạy embedding: {device.upper()}")
    
    if device == 'cpu':
        cpu_cores = min(os.cpu_count() or 4, 8)
        torch.set_num_threads(cpu_cores)
        print(f"⚡ Đã cấu hình PyTorch CPU threads: {cpu_cores}")
    
    print("🧠 Đang tải mô hình sentence-transformers 'paraphrase-multilingual-MiniLM-L12-v2'...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
    
    print("⏳ Bắt đầu tính toán embeddings cho RAG truyền thống...")
    start_time = time.time()
    
    # Tính embedding chuẩn hóa bằng model.encode
    embeddings = model.encode(
        embedding_contexts,
        batch_size=512,
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
