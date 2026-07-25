# Walkthrough - Khắc Phục Lỗi & Hướng Dẫn Chạy Song Song Traditional RAG vs CineBotV3

Tài liệu này tổng hợp việc khắc phục các lỗi phát sinh (bao gồm thiếu `torchvision` và lệnh `uv`) và hướng dẫn chi tiết từng bước để chạy song song 2 hệ thống **Traditional RAG (RAG Truyền Thống)** và **CineBotV3** bằng Python/Streamlit chuẩn.

---

## 🛠️ 1. Nguyên Nhân Lỗi & Cách Khắc Phục

### ❌ Lỗi 1: `uv : The term 'uv' is not recognized`
- **Nguyên nhân**: Máy tính của bạn đang dùng môi trường Python chuẩn (`pip` / `python`) và chưa cài đặt công cụ quản lý `uv`.
- **Cách khắc phục**: Không cần gõ chữ `uv run`. Chỉ cần dùng câu lệnh `streamlit run` hoặc `python -m streamlit run` trực tiếp.

### ❌ Lỗi 2: `ModuleNotFoundError: No module named 'torchvision'`
- **Nguyên nhân**: Streamlit tự động quét các thư viện hình ảnh phụ của `transformers`.
- **Cách khắc phục**: Đã cài đặt bổ sung `torchvision` bằng lệnh: `pip install torchvision`.

---

## 🚀 2. Hướng Dẫn Chạy Song Song 2 Hệ Thống Để So Sánh

Bạn có 2 cách chạy song song cực kỳ thuận tiện:

### 🌟 Cách 1: Sử Dụng Giao Diện So Sánh 2 Cột Trực Tiếp (Dual Side-by-Side Mode)

Đây là cách tốt nhất vì bạn **chỉ cần nhập 1 câu hỏi duy nhất**, ứng dụng sẽ tự động xử lý qua 2 pipeline và hiển thị 2 cột kết quả trực quan trên cùng 1 màn hình.

1. Khởi chạy ứng dụng bằng 1 trong các lệnh sau trong Terminal:
   ```bash
   streamlit run chatbot/app_traditional.py --server.port 8502
   ```
   *Hoặc dùng lệnh python trực tiếp:*
   ```bash
   python -m streamlit run chatbot/app_traditional.py --server.port 8502
   ```
   *(Hoặc nhấp đúp vào file [run_traditional.bat](file:///h:/PythonProject/smartcinev3/SmartCine-RAG-/run_traditional.bat))*

2. Trên thanh **Sidebar (Cột bên trái)**:
   - Chọn Nhà cung cấp LLM: **Gemini API** (hoặc Local LLM).
   - Nhập **Gemini API Key** (nếu chưa có trong file `.env`).
   - Tại mục **"Chọn chế độ hoạt động"**, chọn công tắc: **`⚔️ So sánh song song (Side-by-Side)`**.

3. **Trải nghiệm**:
   - Nhập câu hỏi ở ô chat bên dưới (ví dụ: *"Tìm các phim hành động phát hành sau năm 2010 có điểm IMDb > 8.5"*).
   - **Cột Trái (🟧 RAG Truyền Thống)**: Hiển thị kết quả Naive Vector Search + LLM.
   - **Cột Phải (🟦 CineBotV3)**: Hiển thị kết quả Hybrid Graph-Pandas Search + Intent Detected + Filters Applied.

---

### 🖥️ Cách 2: Bật Song Song 2 Tiến Trình Độc Lập Trên 2 Cổng (Port 8501 & 8502)

Nếu bạn muốn mở 2 cửa sổ trình duyệt độc lập để test riêng từng hệ thống:

#### Cách A: Dùng file Batch tự động (Windows - Tự động nhận diện không dùng uv)
Nhấp đúp chuột vào file:
👉 [run_both_apps.bat](file:///h:/PythonProject/smartcinev3/SmartCine-RAG-/run_both_apps.bat)

File này sẽ mở 2 cửa sổ terminal khởi chạy đồng thời:
- 🟦 **CineBot V3 UI**: [http://localhost:8501](http://localhost:8501)
- 🟧 **Traditional RAG UI**: [http://localhost:8502](http://localhost:8502)

#### Cách B: Chạy thủ công từ 2 cửa sổ Terminal
- **Terminal 1** (CineBotV3):
  ```bash
  streamlit run chatbot/app.py --server.port 8501
  ```
- **Terminal 2** (Traditional RAG):
  ```bash
  streamlit run chatbot/app_traditional.py --server.port 8502
  ```

---

## 🧪 3. Các Câu Hỏi Mẫu Để Thử Nghiệm So Sánh

Hãy nhập các câu hỏi dưới đây để thấy rõ sự khác biệt giữa 2 kiến trúc:

| Loại Kịch Bản | Câu Hỏi Mẫu | Kết Quả RAG Truyền Thống (Naive) | Kết Quả CineBotV3 (Hybrid) |
| :--- | :--- | :--- | :--- |
| **Attribute Filter** (Năm/Rating) | *"Tìm phim hành động sau năm 2010 điểm IMDb trên 8.5"* | Thường lẫn phim năm 2005 hoặc IMDb < 8.5 do chỉ search vector | Lọc chính xác 100% nhờ Pandas Hard-Filter (`Year > 2010`, `Rating > 8.5`) |
| **Negative Constraint** (Loại trừ) | *"Phim giống Interstellar nhưng không phải của Christopher Nolan"* | Bị kéo các phim của Nolan lên đầu do dính từ khóa "Nolan" | Lọc loại trừ triệt để Nolan nhờ Pandas Negative Filter |
| **Graph Multi-hop** (Đồ thị) | *"Đạo diễn của Alien: Romulus đã từng hợp tác với những ai?"* | Không trả lời được do không kết nối được thông tin qua đồ thị | Đưa ra danh sách diễn viên nhờ Graph RAG network |
