# 🔵 Sơ đồ Pipeline Chi tiết Traditional RAG (Kiến trúc Naive RAG Truyền thống)

Tài liệu này mô tả chi tiết quy trình xử lý thông tin của hệ thống RAG truyền thống (Traditional RAG). Quy trình hoạt động theo mô hình Naive RAG tuyến tính 3 bước cơ bản: Embedding -> Vector Search -> LLM Generation.

---

## 🗺️ Tổng quan Luồng Dữ liệu (System Architecture)

```
[Người dùng nhập câu hỏi]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ TẦNG 1: BI-ENCODER EMBEDDING                           │
│ - Chuẩn hóa câu hỏi dạng văn bản phẳng                  │
│ - Tạo vector nhúng 384 chiều và chuẩn hóa L2           │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼ (Query Vector)
┌────────────────────────────────────────────────────────┐
│ TẦNG 2: TRUY XUẤT FLAT VECTOR (FAISS FlatIP Search)     │
│ - Tìm kiếm K lân cận gần nhất (KNN) trên chỉ mục index │
│ - Đầu ra: Top-K phim có Cosine Similarity cao nhất     │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼ (Top-K Phim thô)
┌────────────────────────────────────────────────────────┐
│ TẦNG 3: SINH CÂU TRẢ LỜI (Context Generation)          │
│ - Nối final_context và Link IMDb thành văn bản phẳng   │
│ - LLM sinh câu trả lời tự nhiên dựa trên prompt        │
└────────────────────────────────────────────────────────┘
```

---

## 📌 Tầng 1: Bi-Encoder Embedding

Giai đoạn đầu tiên thực hiện chuyển đổi câu hỏi tự nhiên của người dùng thành biểu diễn toán học dạng vector số thực.

1. **Chuẩn hóa chuỗi:** Nhận câu hỏi gốc của người dùng, giữ nguyên các ký tự và đại từ chỉ định mà không bóc tách thực thể hay loại bỏ từ khóa gây nhiễu.
2. **Tính toán Vector nhúng:** Sử dụng mô hình Bi-Encoder `paraphrase-multilingual-MiniLM-L12-v2` để mã hóa toàn bộ câu hỏi thành một vector **384 chiều**.
3. **L2 Normalization (Chuẩn hóa độ dài):**
   * Tính toán độ dài vector (L2 norm).
   * Chia vector gốc cho độ dài của nó để đưa về độ dài bằng `1.0`.
   * *Mục đích:* Chuẩn hóa vector phục vụ cho phép tính tích vô hướng (Inner Product) ở tầng sau tương đương với phép tính độ tương đồng Cosine (Cosine Similarity).

---

## 📌 Tầng 2: Truy xuất Flat Vector (FAISS FlatIP Search)

Giai đoạn tìm kiếm các tài liệu có nội dung tương đồng ngữ nghĩa nhất trong cơ sở dữ liệu.

1. **Chỉ mục Vector (Vector Index):**
   * Sử dụng chỉ mục **FAISS FlatIP** (Flat Inner Product) được tạo sẵn từ văn bản gộp (`Title` + `Description` + `Genres`) của toàn bộ cơ sở dữ liệu phim.
2. **Tìm kiếm K lân cận gần nhất (KNN):**
   * Sử dụng vector câu hỏi đã chuẩn hóa L2 thực hiện phép tính tích vô hướng trên FAISS Index để tìm ra các tọa độ phim có khoảng cách góc nhỏ nhất.
   * Lấy ra đúng **Top-K** chỉ số (mặc định là **Top 5** phim).
3. **Trích xuất thông tin:**
   * So khớp các chỉ số thu được với cơ sở dữ liệu dạng bảng (DataFrame) để lấy thông tin chi tiết của phim (Tiêu đề, Siêu dữ liệu, Mô tả).
   * Gán giá trị điểm tương đồng (Similarity Score) thu được từ FAISS trực tiếp vào bảng kết quả.
   * *Hạn chế:* Không có bước hậu xử lý, lọc Metadata (năm, điểm số) hay loại bỏ thực thể loại trừ. Phim được lấy ra thuần túy dựa vào khoảng cách vector phẳng.

---

## 📌 Tầng 3: Sinh Câu Trả Lời (Context Generation)

Giai đoạn tổng hợp ngữ cảnh và gửi yêu cầu cho LLM để tạo phản hồi cho người dùng.

1. **Đóng gói Context (Văn bản phẳng):**
   * Ghép nối thông tin `final_context` (chứa Tiêu đề, Đạo diễn, Diễn viên, Thể loại, Tóm tắt cốt truyện có sẵn trong database) của Top-5 phim thành một đoạn văn bản dài.
   * Gắn kèm đường dẫn IMDb (`Link IMDb`) bên dưới mỗi bộ phim nếu có.
   * Không hiển thị lý do chấm điểm hay đường đi liên kết đồ thị.
2. **Tạo Prompt:**
   * Đúc chuỗi ngữ cảnh vừa ghép nối vào tham số `{movies_info}` của RAG Prompt Template cơ bản.
   * Đưa câu hỏi gốc của người dùng vào tham số `{input}`.
3. **LLM Generation:**
   * Gọi mô hình Chat LLM để đọc prompt và tự động suy luận câu trả lời.
   * *Hạn chế:* Do ngữ cảnh đầu vào không được lọc cứng theo điều kiện số học hay phủ định, LLM phải tự sàng lọc từ ngữ cảnh bị nhiễu. Điều này dẫn đến nguy cơ cao sinh ra câu trả lời chứa thông tin không chính xác hoặc vi phạm các ràng buộc mà người dùng đưa ra.
