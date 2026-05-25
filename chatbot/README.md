# 🎬 CineBot — Chatbot Tìm Phim

## Cấu trúc file
```
chatbot/
├── app.py            ← code chính
├── requirements.txt  ← thư viện cần cài
├── movies.csv        ← file dữ liệu của nhóm (đặt vào đây)
└── README.md
```

## Bước 1 — Cài thư viện
```bash
pip install -r requirements.txt
```

## Bước 2 — Đặt file CSV
Đặt file CSV 64k phim vào cùng thư mục, đặt tên là `movies.csv`

## Bước 3 — Đổi tên cột trong app.py
Mở `app.py`, tìm phần "MAPPING TÊN CỘT" và đổi lại cho đúng với CSV của nhóm:
```python
COL_TITLE    = "Series_Title"   # ← đổi thành tên cột tên phim trong CSV của nhóm
COL_GENRE    = "Genre"          # ← đổi thành tên cột thể loại
COL_DIRECTOR = "Director"       # ← đổi thành tên cột đạo diễn
COL_STAR1    = "Star1"          # ← đổi thành tên cột diễn viên 1
COL_STAR2    = "Star2"          # ← đổi thành tên cột diễn viên 2
COL_YEAR     = "Released_Year"  # ← đổi thành tên cột năm phát hành
COL_RATING   = "IMDB_Rating"    # ← đổi thành tên cột điểm IMDB
COL_OVERVIEW = "Overview"       # ← đổi thành tên cột mô tả phim
```

Để xem tên cột trong CSV:
```python
import pandas as pd
df = pd.read_csv("movies.csv")
print(df.columns.tolist())
print(df.head(2))
```

## Bước 4 — Lấy Gemini API Key (miễn phí)
1. Vào https://aistudio.google.com/app/apikey
2. Đăng nhập Google
3. Nhấn "Create API Key"
4. Copy key

## Bước 5 — Chạy app
```bash
streamlit run app.py
```
Trình duyệt tự mở tại http://localhost:8501
Dán API Key vào sidebar là dùng được.

## Kiến trúc hoạt động
```
User gõ câu hỏi
    ↓
Gemini (Tầng 1): phân tích → JSON {intent, filters}
    ↓
Pandas: lọc DataFrame 64k dòng theo filters
    ↓
Gemini (Tầng 2): nhận top 5 kết quả → sinh câu trả lời tự nhiên
    ↓
Streamlit hiển thị câu trả lời + card phim
```

## Xử lý lỗi thường gặp

**Lỗi "column not found"**: Tên cột trong `app.py` chưa khớp với CSV → kiểm tra lại Bước 3

**Lỗi "API key invalid"**: Key sai hoặc chưa kích hoạt → kiểm tra lại Bước 4

**Bot không hiểu câu tiếng Việt có dấu**: Bình thường — Gemini Flash hiểu tiếng Việt tốt

**Chạy chậm**: Mỗi tin nhắn gọi Gemini 2 lần (~1-2 giây) — đây là bình thường
