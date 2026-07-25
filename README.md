# 🎬 SmartCine-RAG: CineBotV3 & Traditional RAG Benchmark System

Dự án nghiên cứu & xây dựng hệ thống Chatbot tìm kiếm và tư vấn phim thông minh, tích hợp cả 2 kiến trúc RAG để đánh giá đối chiếu:
1. **CineBotV3**: Hệ thống RAG nâng cao kết hợp Hybrid Search (Vector + BM25), trích xuất thực thể, Pandas Attribute Hard-Filtering và Graph RAG.
2. **Traditional RAG (Naive Vector RAG)**: Giao diện RAG truyền thống dựa trên truy vấn FAISS Cosine Similarity thuần túy và LLM Prompting.

---

## 🚀 Khởi Chạy Nhanh

### 1. Chạy Độc Lập Traditional RAG UI (Port 8502)
```bash
uv run streamlit run chatbot/app_traditional.py --server.port 8502
```
Hoặc nhấp đúp file: `run_traditional.bat`

### 2. Chạy Độc Lập CineBot V3 UI (Port 8501)
```bash
uv run streamlit run chatbot/app.py --server.port 8501
```

### 3. Chạy Song Song Cả 2 Hệ Thống Để So Sánh
Nhấp đúp file: `run_both_apps.bat`
- 🟦 **CineBot V3**: http://localhost:8501
- 🟧 **Traditional RAG**: http://localhost:8502

---

## 📖 Hướng Dẫn Chi Tiết
Xem tài liệu hướng dẫn đầy đủ tại: [HUONG_DAN_SU_DUNG_TRADITIONAL_RAG.md](file:///h:/PythonProject/smartcinev3/SmartCine-RAG-/HUONG_DAN_SU_DUNG_TRADITIONAL_RAG.md)
