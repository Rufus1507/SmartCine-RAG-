# 📖 Hướng Dẫn Chạy & Sử Dụng Giao Diện RAG Truyền Thống (Traditional RAG) & So Sánh Song Song Với CineBotV3

Tài liệu này hướng dẫn chi tiết cách khởi chạy, cấu hình và sử dụng giao diện UI/UX của **RAG Truyền Thống (Naive Vector RAG)**, cũng như cách bật song song với hệ thống **CineBotV3** để đối chiếu và so sánh hiệu năng.

---

## 🚀 1. Tổng Quan Kiến Trúc

| Tiêu chí | 🟧 Traditional RAG (Naive Vector RAG) | 🟦 CineBotV3 (Hybrid Graph-Pandas RAG) |
| :--- | :--- | :--- |
| **Cơ chế Retrieval** | Vector Search thuần túy (FAISS Cosine Similarity trên `final_context`) | Hybrid Search (FAISS + BM25 + Subgraph Expansion + Reranking) |
| **Lọc thuộc tính (Filters)** | **Không có** (phụ thuộc vào khả năng tự lọc của LLM trong prompt) | **Pandas Hard-Filtering** (Lọc chuẩn xác năm, điểm IMDb, quốc gia, diễn viên) |
| **Ràng buộc phủ định** | Thường bị nhầm lẫn từ khóa (overfitting vector match) | **Pandas Negative Filtering** (Lọc loại trừ triệt me) |
| **Suy luận Graph / Multi-hop** | Không hỗ trợ | **Graph RAG** (Truy vấn quan hệ đồ thị diễn viên - đạo diễn - phim) |
| **Cổng mặc định (Port)** | `8502` | `8501` |

---

## 🛠️ 2. Chuẩn Bị Môi Trường & Cấu Hình

### Bước 2.1: Cài Đặt Thư Viện
Đảm bảo bạn đã cài đặt các thư viện cần thiết bằng pip:
```bash
pip install -r chatbot/requirements.txt
```

### Bước 2.2: Kiểm Tra File Dữ Liệu & Chỉ Mục FAISS
Hệ thống sử dụng các file dữ liệu được tối ưu sẵn tại thư mục `data/`:
- `data/cinebot_movies.parquet` (Dữ liệu bộ phim)
- `data/traditional_context.index` (Chỉ mục FAISS truyền thống)

### Bước 2.3: Lấy Gemini API Key (Miễn phí)
1. Truy cập [Google AI Studio](https://aistudio.google.com/app/apikey) và tạo API Key.
2. Bạn có thể dán API Key trực tiếp trên giao diện Sidebar của ứng dụng hoặc thêm vào file `.env`:
```env
GEMINI_API_KEY=AIzaSy...
```

---

## 💻 3. Các Phương Thức Khởi Chạy

Bạn có thể chạy hệ thống theo 3 cách linh hoạt dưới đây:

### Cách 1: Khởi Chạy Tự Động Song Song 2 Hệ Thống (Khuyên Dùng Trên Windows)
Nhấp đúp chuột vào file batch:
```cmd
run_both_apps.bat
```
File này sẽ tự động kiểm tra môi trường và mở 2 cửa sổ tiến trình:
- 🟦 **CineBotV3**: [http://localhost:8501](http://localhost:8501)
- 🟧 **Traditional RAG**: [http://localhost:8502](http://localhost:8502)

---

### Cách 2: Chạy Độc Lập Traditional RAG Từ Dòng Lệnh Terminal
Gõ lệnh terminal tiêu chuẩn:
```bash
streamlit run chatbot/app_traditional.py --server.port 8502
```
Hoặc dùng python module:
```bash
python -m streamlit run chatbot/app_traditional.py --server.port 8502
```
Trình duyệt sẽ tự động mở giao diện Traditional RAG tại địa chỉ: `http://localhost:8502`.

---

### Cách 3: Chạy CineBotV3 Từ Dòng Lệnh Terminal
```bash
streamlit run chatbot/app.py --server.port 8501
```
Hoặc dùng python module:
```bash
python -m streamlit run chatbot/app.py --server.port 8501
```

---

## 🎯 4. Hướng Dẫn Sử Dụng Giao Diện UI/UX Traditional RAG (`app_traditional.py`)

Giao diện `app_traditional.py` tích hợp 2 chế độ linh hoạt trên thanh Sidebar:

### 1. Chế độ Standalone (RAG Truyền thống độc lập)
- **Khung Chat**: Nhập câu hỏi và nhận câu trả lời từ Naive RAG.
- **Movie Cards**: Hiển thị danh sách top phim vector matched kèm điểm tương đồng Cosine Similarity (`similarity`).
- **Timing Metrics**: Đo chi tiết thời gian truy xuất vector (Retrieval Latency) và thời gian sinh câu trả lời LLM (LLM Latency).
- **Context Inspector**: Nhấn vào expander `🔍 Context Inspector` để xem exact context thô và prompt template được đưa vào LLM.

### 2. Chế độ ⚔️ So Sánh Song Song (Dual Side-by-Side Mode)
- Đổi công tắc Sidebar sang **"⚔️ So sánh song song (Side-by-Side)"**.
- Nhập **1 câu hỏi duy nhất** ở ô chat dưới cùng.
- Màn hình sẽ chia làm **2 cột trực quan**:
  - **Cột Trái (🟧 RAG Truyền thống)**: Hiển thị kết quả Naive Vector Search + LLM.
  - **Cột Phải (🟦 CineBotV3)**: Hiển thị kết quả Hybrid Search (Graph + Pandas Filter) + Intent Detected + Filters Applied.

---

## 🧪 5. Bộ Câu Hỏi Kịch Bản Test So Sánh Tiêu Biểu

- **Lọc thuộc tính (Attribute Filter)**: *"Tìm các phim hành động phát hành sau năm 2010 có điểm IMDb trên 8.5"*
- **Ràng buộc phủ định (Negative Constraint)**: *"Gợi ý các phim viễn tưởng hại não giống Interstellar nhưng không phải của đạo diễn Christopher Nolan"*
- **Suy luận đồ thị (Graph Multi-hop)**: *"Đạo diễn của phim Alien: Romulus đã từng hợp tác với những diễn viên nào?"*
