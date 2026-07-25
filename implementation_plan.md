# Kế hoạch Triển khai RAG Truyền thống & So sánh Đối chiếu với CineBot V3

Bản kế hoạch này mô tả các bước để xây dựng một hệ thống RAG truyền thống (Traditional/Naive RAG) từ cơ sở dữ liệu phim hiện tại, chạy đánh giá trên bộ câu hỏi khó `eval/hq_questions.json`, và so sánh trực tiếp để làm rõ những điểm cải tiến vượt trội của hệ thống hiện tại (**CineBot V3 - Hybrid RAG + Multi-stage Hybrid Retrieval + Feature Engineering + Graph RAG**).

---

## 1. Thiết kế Hệ thống RAG Truyền thống (Simple/Naive RAG)

Hệ thống RAG truyền thống sẽ được xây dựng tối giản để mô phỏng chính xác cách tiếp cận Naive RAG cơ bản:
1. **Dữ liệu nguồn**: Sử dụng cùng một tập dữ liệu phim (từ `cinebot_movies.parquet`).
2. **Text Chunking**: Mỗi bộ phim được biểu diễn bằng một văn bản duy nhất kết hợp tất cả các trường thông tin (sử dụng cột `final_context` có sẵn trong `data_loader.py` để đảm bảo hệ thống truyền thống cũng có đủ thông tin thuộc tính như tên phim, thể loại, đạo diễn, diễn viên, điểm số, v.v.).
3. **Embedding**: Sử dụng cùng mô hình embedding dense `paraphrase-multilingual-MiniLM-L12-v2` để sinh vector nhúng cho `final_context` của toàn bộ phim.
4. **Vector Database / Indexing**: Tạo một chỉ mục FAISS FlatL2 riêng biệt cho các vector `final_context` này (`data/traditional_context.index`).
5. **Retrieval**: Khi nhận câu hỏi từ người dùng:
   - Vector hóa câu hỏi trực tiếp bằng mô hình embedding.
   - Truy vấn chỉ mục FAISS lấy **Top 5** phim có khoảng cách L2 gần nhất.
   - **KHÔNG** sử dụng tìm kiếm từ khóa BM25.
   - **KHÔNG** sử dụng trích xuất thực thể hay bộ lọc metadata cứng (Pandas Filters).
   - **KHÔNG** sử dụng công thức tính tương đồng weighted hay cấu trúc Graph RAG.
   - **KHÔNG** sử dụng mô hình Cross-Encoder để rerank.
6. **Prompt Generation & Generation**:
   - Đưa trực tiếp 5 kết quả tìm được vào một Prompt mẫu đơn giản:
     ```
     Dựa vào thông tin các bộ phim sau:
     [Thông tin phim 1]
     [Thông tin phim 2]
     ...
     Hãy trả lời câu hỏi của người dùng: [Câu hỏi]
     ```
   - Gửi prompt này tới mô hình LLM để tạo câu trả lời cuối cùng (sử dụng cùng Client LLM hiện tại để đảm bảo tính khách quan).

---

## 2. Câu hỏi Cần Làm rõ & Thiết kế So sánh (Evaluation & Comparison Setup)

Chúng ta sẽ chạy 10 câu hỏi trong [hq_questions.json](file:///h:/PythonProject/smartcinev3/SmartCine-RAG-/eval/hq_questions.json) qua cả 2 hệ thống và đối chiếu các khía cạnh sau:

### Phân loại các nhóm thử thách trong câu hỏi:
1. **Lọc thuộc tính phức tạp (Complex Metadata Filter)** - `hq1`: Lọc phim theo điểm số, thời lượng, năm phát hành, thể loại.
   - *Traditional RAG*: Thường thất bại vì không thể ép bộ lọc cứng từ vector embedding. Các ứng viên trả về có thể vi phạm điều kiện (ví dụ: phát hành trước năm 2010 hoặc thời lượng > 150 phút).
   - *CineBot V3*: Thành công tuyệt đối nhờ lớp trích xuất thực thể kết hợp bộ lọc Pandas cứng trước khi tính điểm tương đồng.
2. **Ràng buộc phủ định / Loại trừ (Negative Constraints)** - `hq3`, `hq8`: Tìm phim giống *Interstellar* nhưng *không phải* của Christopher Nolan.
   - *Traditional RAG*: Gặp lỗi nghiêm trọng vì vector của *Christopher Nolan* và *Interstellar* sẽ kéo các bộ phim của Nolan lên hàng đầu (do độ tương đồng ngữ nghĩa cao).
   - *CineBot V3*: Giải quyết bằng cách trích xuất điều kiện loại trừ và áp dụng bộ lọc động `df['directors'] != 'Christopher Nolan'`.
3. **Suy luận đồ thị đa bước (Graph RAG / Multi-hop Reasoning)** - `hq2`, `hq6`, `hq9`, `hq10`: Các câu hỏi liên quan đến mối quan hệ hợp tác giữa diễn viên, đạo diễn (ví dụ: "Diễn viên hợp tác với Christopher Nolan nhiều nhất...").
   - *Traditional RAG*: Thất bại hoàn toàn vì các mối quan hệ mạng lưới không thể suy luận được từ việc đọc độc lập vài chunk mô tả phim.
   - *CineBot V3*: Giải quyết xuất sắc nhờ tích hợp Graph RAG chạy trên đồ thị NetworkX liên kết phim-nhân sự để tìm đường đi ngắn nhất hoặc đếm số lần collab.

---

## 3. Các Bước Thực hiện (Proposed Changes)

Để triển khai và thực hiện so sánh, chúng ta sẽ tạo các file mới trong thư mục `eval/` và chạy quy trình đánh giá.

### Component: Traditional RAG Implementation

#### [NEW] [generate_traditional_embeddings.py](file:///h:/PythonProject/smartcinev3/SmartCine-RAG-/eval/generate_traditional_embeddings.py)
- Script chạy một lần để tạo vector nhúng cho cột `final_context` của tất cả các phim trong `cinebot_movies.parquet` bằng mô hình `paraphrase-multilingual-MiniLM-L12-v2`.
- Lưu chỉ mục FAISS mới tại `data/traditional_context.index`.

#### [NEW] [traditional_rag.py](file:///h:/PythonProject/smartcinev3/SmartCine-RAG-/eval/traditional_rag.py)
- Triển khai pipeline RAG truyền thống:
  - Hàm `retrieve_traditional(query, index, model, df, top_k=5)` thực hiện truy vấn FAISS L2 thuần túy.
  - Hàm `run_traditional_rag_pipeline(query, llm, df, index, model)` lắp ghép ngữ cảnh và gọi LLM.

#### [NEW] [run_traditional_harness.py](file:///h:/PythonProject/smartcinev3/SmartCine-RAG-/eval/run_traditional_harness.py)
- Script đánh giá tương tự như `run_hq_harness.py` nhưng sử dụng pipeline RAG truyền thống.
- Chạy qua 10 câu hỏi trong `hq_questions.json`.
- Xuất kết quả chi tiết ra file `eval/traditional_results_raw.json`.

#### [NEW] [compare_results.py](file:///h:/PythonProject/smartcinev3/SmartCine-RAG-/eval/compare_results.py)
- Script tự động so sánh hai file kết quả `eval/hq_results_raw.json` (CineBot V3) và `eval/traditional_results_raw.json` (RAG truyền thống).
- Tổng hợp và tự động xuất ra báo cáo đối chiếu định dạng Markdown: `eval/traditional_vs_cinebot_report.md`.

---

## 4. Kế hoạch Kiểm thử & Xác minh (Verification Plan)

### Kiểm thử Tự động
1. Chạy tạo chỉ mục:
   ```powershell
   python eval/generate_traditional_embeddings.py
   ```
   *Kiểm tra:* File `data/traditional_context.index` được tạo thành công với kích thước khoảng ~290MB.
2. Chạy đánh giá RAG truyền thống:
   ```powershell
   python eval/run_traditional_harness.py
   ```
   *Kiểm tra:* File `eval/traditional_results_raw.json` được ghi lại đầy đủ 10 câu trả lời.
3. Chạy báo cáo so sánh:
   ```powershell
   python eval/compare_results.py
   ```
   *Kiểm tra:* Sinh ra báo cáo `eval/traditional_vs_cinebot_report.md` chi tiết.

### Xác minh Thủ công
- Đọc báo cáo so sánh `eval/traditional_vs_cinebot_report.md` để đánh giá định tính chất lượng câu trả lời của 2 hệ thống đối với từng câu hỏi cụ thể (kiểm tra xem RAG truyền thống có bị lỗi metadata filter, lỗi phủ định hay không thể trả lời câu hỏi multi-hop hay không).
