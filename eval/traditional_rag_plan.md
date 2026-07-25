# Kế hoạch Triển khai RAG Truyền thống & So sánh Đối chiếu với CineBot V3

Tài liệu này trình bày chi tiết về phương án thiết kế hệ thống RAG truyền thống (Naive RAG) và phương pháp đánh giá so sánh với chatbot hiện tại của bạn.

---

## 1. Thiết kế Hệ thống RAG Truyền thống (Simple/Naive RAG)
Để so sánh một cách khách quan nhất, hệ thống RAG truyền thống sẽ giữ nguyên mô hình Embedding và LLM, nhưng loại bỏ toàn bộ các bước lọc, trích xuất và tối ưu hóa của CineBot V3:

*   **Dữ liệu sử dụng**: `cinebot_movies.parquet`.
*   **Văn bản biểu diễn**: Sử dụng cột `final_context` (chứa toàn bộ Title, Description, Genres, Directors, Stars, v.v. được nối thành một chuỗi duy nhất).
*   **Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2`.
*   **Chỉ mục Vector**: Chỉ mục FAISS FlatL2 riêng biệt (`data/traditional_context.index`).
*   **Luồng Retrieval (Naïve RAG)**:
    1. Câu hỏi của người dùng được vector hóa trực tiếp bằng mô hình Embedding.
    2. Truy vấn FAISS để lấy **Top 5** phim gần nhất theo khoảng cách L2.
    3. *Không* lọc Pandas, *không* dùng BM25, *không* dùng Graph RAG, *không* dùng Cross-Encoder reranking.
*   **Luồng Generation**:
    1. Đưa thông tin của 5 phim tìm được vào một prompt cơ bản làm Context.
    2. Gửi Prompt đến LLM cấu hình mặc định để sinh câu trả lời.

---

## 2. Kế hoạch Đánh giá & So sánh
Hệ thống sẽ chạy thử nghiệm trên 10 câu hỏi của `eval/hq_questions.json`.
Sau khi chạy xong, kết quả của RAG truyền thống sẽ được ghi nhận tại `eval/traditional_results_raw.json` để so sánh với kết quả CineBot V3 trong `eval/hq_results_raw.json`.

### Các khía cạnh so sánh (Strengths vs Weaknesses):
*   **Khả năng lọc thuộc tính (Attribute Filtering)** (Ví dụ: `hq1` - phim hành động > 8.5 điểm, năm > 2010): CineBot V3 sử dụng Pandas Filters cứng, trong khi Naive RAG chỉ dùng tương đồng vector và dễ đưa vào các phim vi phạm điều kiện.
*   **Khả năng phủ định (Negative Constraints)** (Ví dụ: `hq3` - phim giống Interstellar nhưng không phải của Nolan): Naive RAG sẽ bị lệch kết quả do từ khóa Nolan kéo các phim của Nolan lên đầu. CineBot V3 lọc bỏ Nolan bằng code Pandas.
*   **Suy luận nhiều bước (Multi-hop Reasoning / Graph RAG)** (Ví dụ: `hq6`, `hq9`, `hq10` - diễn viên hợp tác nhiều nhất, đạo diễn Alien: Romulus hợp tác với ai): Naive RAG không có khả năng liên kết dữ liệu giữa các bộ phim khác nhau. CineBot V3 truy vấn cấu trúc đồ thị (Graph RAG) để tìm liên kết chính xác.
*   **Từ khóa hiếm & Trùng tên phim (Decoy Titles / Keyword matching)**: CineBot V3 kết hợp BM25 và cấu trúc Split Vector giúp tránh overfitting từ khóa.

---

## 3. Các File Đầu ra Dự kiến
1. `eval/generate_traditional_embeddings.py`: Code tạo chỉ mục vector truyền thống.
2. `eval/traditional_rag.py`: Triển khai luồng RAG truyền thống.
3. `eval/run_traditional_harness.py`: Code tự động chạy 10 câu hỏi.
4. `eval/compare_results.py`: Script so sánh kết quả và tự động tạo báo cáo `traditional_vs_cinebot_report.md`.
