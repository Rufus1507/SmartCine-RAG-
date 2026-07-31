# 📊 Báo cáo Đánh giá Nghiêm ngặt: Traditional RAG vs CineBot V3
> **Benchmark 100 câu hỏi (L1 - L10)** | Tiêu chuẩn đánh giá: Loại trừ câu trả lời lỗi/bịa đặt/không đúng yêu cầu logic.

## 1. Tóm tắt kết quả kiểm thử (Strict Accuracy)

Trong đánh giá nghiêm ngặt này, một câu trả lời chỉ được tính là **ĐẠT (Passed)** khi nó đáp ứng chính xác 100% các điều kiện logic của câu hỏi (như lọc năm, điểm số, quốc gia, loại trừ, đếm số lượng, suy luận đồ thị quan hệ). 

Các câu trả lời của **Traditional RAG** mà chỉ trả về kết quả tìm kiếm ngữ nghĩa phẳng (FAISS) chứa các phim không đáp ứng đúng điều kiện lọc cứng hoặc trả lời chung chung mang tính chất "bịa đặt" (hallucination) từ LLM/fallback sẽ bị tính là **KHÔNG ĐẠT (Failed)**.

| Chỉ số đánh giá | Traditional RAG | CineBot V3 | Nhận xét sự chênh lệch |
|:---|:---:|:---:|:---|
| **Tổng số câu hỏi** | 100 | 100 | — |
| **Số câu trả lời ĐẠT (Strict Pass)** | **26 / 100** | **98 / 100** | CineBot V3 vượt trội **+72 câu** chính xác |
| **Tỷ lệ chính xác toàn bộ** | **26.0%** | **98.0%** | CineBot V3 đạt độ tin cậy thực tế vượt trội |
| **Số câu trả lời lỗi/bịa đặt** | **74 / 100** | **2 / 100** | RAG truyền thống bị lỗi logic do không có metadata/graph |
| **Thời gian phản hồi trung bình** | **3.57s** | **18.14s** | Traditional RAG nhanh hơn nhưng thiếu chính xác |

---

## 2. Bảng so sánh chi tiết theo 10 Cấp độ (Levels)

| Cấp độ | Nhóm câu hỏi / Category | Số câu | Traditional RAG | CineBot V3 | Phân tích lý do lỗi của Traditional RAG |
|:---:|:---|:---:|:---:|:---:|:---|
| **L1** | Semantic Retrieval / Recommendation (Rất dễ) | 10 | **10 / 10** | **10 / 10** | Đạt yêu cầu nhờ tìm kiếm ngữ nghĩa phẳng trên mô tả phim hoạt động tốt. |
| **L2** | Semantic Retrieval / Recommendation (Dễ) | 10 | **10 / 10** | **10 / 10** | Đạt yêu cầu. Các câu hỏi vẫn nằm trong vùng xử lý tốt của Vector Search phẳng. |
| **L3** | Metadata Filter - Lọc cơ bản (Dễ-Vừa) | 10 | **0 / 10** | **10 / 10** | **Thất bại hoàn toàn**: Không thể lọc cứng năm phát hành (ví dụ: phim sau năm 2018), điểm IMDb (ví dụ: > 8.0). Phim trả về bị lệch điều kiện. |
| **L4** | Semantic Reasoning - Suy luận ngữ nghĩa (Vừa) | 10 | **6 / 10** | **10 / 10** | **Thất bại 4 câu**: Các câu hỏi yêu cầu liên kết nhiều ngữ cảnh hoặc ý nghĩa sâu sắc hơn từ mô tả bị bỏ sót. |
| **L5** | Metadata Filter - Lọc kết hợp (Vừa) | 10 | **0 / 10** | **10 / 10** | **Thất bại hoàn toàn**: Gặp nhiều ràng buộc số học cùng lúc (năm và điểm số). RAG truyền thống chỉ lấy độ tương đồng vector nên kết quả vi phạm điều kiện. |
| **L6** | Metadata Filter - Lọc nâng cao (Vừa-Khó) | 10 | **0 / 10** | **10 / 10** | **Thất bại hoàn toàn**: Yêu cầu lọc theo thuộc tính phức tạp như Quốc gia phát hành, giải thưởng (Oscar), thời lượng phim. RAG truyền thống không phân tách được metadata này. |
| **L7** | Negative Constraint - Ràng buộc phủ định (Khó) | 10 | **0 / 10** | **10 / 10** | **Thất bại hoàn toàn**: Khi người dùng yêu cầu loại trừ một thể loại (ví dụ: "không phải phim kinh dị"), Vector Search lại lấy độ tương đồng cao với từ "kinh dị", dẫn đến trả về chính xác phim kinh dị. |
| **L8** | Aggregation - Gom nhóm & Thống kê (Rất khó) | 10 | **0 / 10** | **9 / 10** | **Thất bại hoàn toàn**: Không thể đếm số lượng phim, tính điểm trung bình của một đạo diễn, hay tìm phim có thời lượng ngắn nhất/dài nhất trong tập kết quả. |
| **L9** | Graph Reasoning - Suy luận đồ thị (Chuyên gia) | 10 | **0 / 10** | **10 / 10** | **Thất bại hoàn toàn**: Câu hỏi yêu cầu tìm mối quan hệ (ví dụ: đạo diễn X làm việc với diễn viên Y trên 2 lần). RAG truyền thống hoàn toàn không có cấu trúc liên kết đồ thị giữa các thực thể. |
| **L10** | Multi-hop Reasoning - Suy luận đa chặng (Chuyên gia+) | 10 | **0 / 10** | **9 / 10** | **Thất bại hoàn toàn**: Kết hợp giữa đồ thị quan hệ, điều kiện thời gian, và lọc metadata. RAG truyền thống chỉ đưa ra kết quả ngẫu nhiên không liên quan. |
| **Tổng** | **Toàn bộ bộ câu hỏi** | **100** | **26 / 100** | **98 / 100** | **CineBot V3 chiến thắng vượt trội về độ chính xác thực tế.** |

---

## 3. Phân tích nguyên nhân và ví dụ cụ thể

### 3.1 Thất bại của RAG truyền thống ở Metadata Filtering (L3, L5, L6)
* **Ví dụ Câu `q21`:** *"Tìm phim hài phát hành sau năm 2018."*
  * **Traditional RAG (0 điểm):** Chỉ thực hiện tìm kiếm vector cho cụm từ "phim hài sau năm 2018". Kết quả trả về các phim hài được sản xuất vào các năm 2010, 2012, 2015 do các phim đó có mô tả rất gần với từ khóa tìm kiếm. Điều kiện cứng "sau năm 2018" bị vi phạm hoàn toàn.
  * **CineBot V3 (1 điểm):** Sử dụng bộ phân tích ý định LLM (Intent Parsing) để trích xuất bộ lọc: `{"genres": "Comedy", "Year": {">": 2018}}`. Sau đó thực thi lọc trực tiếp trên DataFrame để đảm bảo 100% kết quả trả về chính xác.

### 3.2 Thất bại của RAG truyền thống ở Negative Constraint (L7)
* **Ví dụ Câu `q61`:** *"Gợi ý phim hành động nhưng không phải phim viễn tưởng."*
  * **Traditional RAG (0 điểm):** Do cơ chế nhúng văn bản (Embedding Vector), từ khóa "viễn tưởng" xuất hiện trong câu truy vấn khiến RAG truyền thống ưu tiên kéo các phim hành động viễn tưởng lên đầu tập kết quả.
  * **CineBot V3 (1 điểm):** LLM trích xuất điều kiện loại trừ: `{"genres": "Action", "exclude_genres": ["Sci-Fi"]}`. Bộ lọc Pandas xử lý loại bỏ hoàn toàn các phim chứa thể loại Sci-Fi trước khi Reranking.

### 3.3 Thất bại ở Aggregation & Graph Reasoning (L8, L9, L10)
* **Ví dụ Câu `q71`:** *"Đạo diễn Christopher Nolan đã đạo diễn bao nhiêu bộ phim có điểm IMDb trên 8.0 trong hệ thống?"*
  * **Traditional RAG (0 điểm):** Không thể thực hiện phép tính đếm. Hệ thống chỉ lấy ngẫu nhiên các đoạn mô tả chứa chữ "Christopher Nolan" và "IMDb trên 8.0" rồi liệt kê danh sách phim, không đưa ra được con số thống kê chính xác.
  * **CineBot V3 (1 điểm):** Phân tích được yêu cầu đếm và lọc: `df[(df['directors'] == 'Christopher Nolan') & (df['Rating'] > 8.0)].shape[0]`. Trả về con số chính xác là `5` phim cùng danh sách cụ thể.

---

## 4. Kết luận đánh giá

1. **Vùng an toàn của Traditional RAG:** Chỉ dừng lại ở các câu hỏi tìm kiếm nội dung chung chung (Semantic Search) hoặc gợi ý phim dựa trên sở thích (Recommendation) ở mức cơ bản (L1, L2, L4).
2. **Điểm yếu cốt tử:** Hoàn toàn bất lực trước dữ liệu số học, điều kiện thời gian, logic loại trừ, thống kê tập dữ liệu và các mối quan hệ liên kết (Graph). Nếu đưa vào hệ thống thực tế cho người dùng, RAG truyền thống sẽ liên tục trả về kết quả sai lệch nhưng được LLM diễn đạt trôi chảy (Hallucination nguy hiểm).
3. **Sức mạnh của CineBot V3:** Việc tích hợp cấu trúc Graph (635K nodes, 3.2M edges) kết hợp với công cụ lọc Pandas động và Hybrid Search (BM25 + FAISS + RRF) giúp giải quyết triệt để các bài toán khó từ L3 đến L10, mang lại độ chính xác thực tế đạt **98.0%**.
