# 🎬 Sơ đồ Pipeline Chi tiết CineBot V3 (Kiến trúc Đa tầng Hybrid Graph-RAG)

Tài liệu này mô tả chi tiết quy trình xử lý thông tin của hệ thống CineBot V3 từ lúc nhận câu hỏi của người dùng cho đến khi sinh câu trả lời cuối cùng. Quy trình được thiết kế theo dạng đường ống đa tầng (Multi-stage Pipeline) độc lập để tối ưu hóa độ chính xác và khả năng suy luận ngữ nghĩa.

---

## 🗺️ Tổng quan Luồng Dữ liệu (System Architecture)

```
[Người dùng nhập câu hỏi]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ TẦNG 0: TIỀN XỬ LÝ & TRÍCH XUẤT THỰC THỂ (Entity Extractor) │
│ - Chuẩn hóa văn bản, bóc tách n-grams                   │
│ - Tra cứu từ điển thực thể (Exact + Fuzzy Match)       │
└────────────────────────┬───────────────────────────────┘
                         │
                         ├───────────────────────────────┐
                         ▼ (Thực thể gợi ý)              │
┌────────────────────────────────────────────────────────┐│
│ TẦNG 1: PHÂN TÍCH Ý ĐỊNH & BỘ LỌC CỨNG (Intent Parser) ││
│ - LLM dịch ngôn ngữ tự nhiên thành JSON Filters        ││
│ - Tự động sửa lỗi bộ lọc & Chuẩn hóa định dạng        ││
└────────────────────────┬───────────────────────────────┘│
                         │                               │
                         ▼ (Intent & Filters JSON)       ▼ (Thực thể đã quét)
┌────────────────────────────────────────────────────────────────────────────────┐
│ TẦNG 2: ĐỊNH TUYẾN TRUY XUẤT (Retrieval Router)                                │
│ - Phân tích logic câu hỏi (Truy vấn tương tự? Thống kê? Tìm kiếm thường?)      │
│ - Quyết định luồng và nạp dữ liệu đồ thị nếu cần                               │
└────────────────────────┬───────────────────────────────────────────────────────┘
                         │
                         ▼ (Định tuyến + Dữ liệu Đồ thị)
┌────────────────────────────────────────────────────────────────────────────────┐
│ TẦNG 3: TRUY XUẤT ĐA TẦNG (Multi-stage Retriever)                              │
│ ├─ Stage 0: Duyệt Đồ thị BFS (Graph BFS Candidates)                            │
│ ├─ Stage 1: Sinh Ứng viên Lai (BM25 + FAISS Vector + Metadata)                 │
│ ├─ Stage 2: Bộ lọc Thuộc tính Cứng (Pandas Metadata Filtering)                 │
│ ├─ Stage 3: Chấm điểm Tương đồng Đa chiều (Weighted Similarity Engine)         │
│ ├─ Stage 4: Xếp hạng lại bằng Mạng Nơ-ron (Cross-Encoder Neural Reranking)      │
│ └─ Stage 5: Tái áp đặt Ràng buộc Cứng & Loại bỏ trùng lặp (Dedup)              │
└────────────────────────┬───────────────────────────────────────────────────────┘
                         │
                         ▼ (Top-K Phim sạch & Chuẩn nhất)
┌────────────────────────────────────────────────────────────────────────────────┐
│ TẦNG 4: SINH CÂU TRẢ LỜI (Answer Generator)                                    │
│ - Đóng gói ngữ cảnh có cấu trúc + Lý do tương đồng & liên kết đồ thị            │
│ - LLM sinh câu trả lời tự nhiên (Hỗ trợ chế độ Đồng bộ hoặc Streaming)         │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📌 Tầng 0: Tiền xử lý & Trích xuất Thực thể (Entity Extractor)

Tầng này chạy hoàn toàn bằng các thuật toán so khớp chuỗi tốc độ cao để nhận diện sớm các thực thể điện ảnh (Tên phim, Đạo diễn, Diễn viên, Thể loại) có sẵn trong cơ sở dữ liệu.

### 1. Chuẩn hóa & Sinh N-gram
* **Chuẩn hóa:** Loại bỏ dấu tiếng Việt (để dự phòng đối chiếu không dấu), chuyển về viết thường và loại bỏ các ký tự đặc biệt.
* **Tách N-gram:** Tách câu hỏi thành các cụm từ liên tiếp (N-gram) có độ dài từ 1 đến 5 từ.
* **Sắp xếp:** Sắp xếp danh sách N-gram theo độ dài giảm dần để ưu tiên khớp các cụm từ dài nhất trước (tránh hiện tượng trùng lặp từ đơn lẻ nằm trong tên riêng).

### 2. Tra cứu thực thể (3 mức ưu tiên)
* **Mức 1 — Khớp Bí danh (Aliases Lookup):** Tra cứu từ điển từ đồng nghĩa/tên viết tắt (O(1)). Ví dụ: `"nolan"` được giải mã ngay thành thực thể `"Christopher Nolan"`.
* **Mức 2 — Khớp Chính xác (Exact Lookup):** Kiểm tra xem N-gram có khớp 100% với tên phim/nhân sự/thể loại trong từ điển hệ thống hay không (O(1)).
* **Mức 3 — Khớp Mờ (Fuzzy Match):** Nếu khớp chính xác thất bại, áp dụng thuật toán so khớp tỷ lệ khoảng cách chuỗi (QRatio):
  * **Điều kiện áp dụng:** Cụm từ phải dài từ 5 ký tự trở lên và không nằm trong danh sách các từ dừng phổ biến (như "phim", "tìm", "đạo diễn", "thể loại",...).
  * **Ngưỡng tin cậy:** Đạt tối thiểu **90%** đối với tên Đạo diễn/Diễn viên/Biên kịch và **85%** đối với Thể loại phim.

### 3. Phán đoán Câu tiếp nối (Refinement Check)
* Quét câu hỏi xem có chứa các từ nối ngữ cảnh (như *"nhưng"*, *"chỉ"*, *"thêm"*, *"nữa"*, *"khác"*, *"họ"*,...) hay không để kích hoạt bộ nhớ lưu trữ bộ lọc từ câu trước.

---

## 📌 Tầng 1: Phân tích Ý định & Bộ lọc Cứng (Intent Parser)

Tầng này sử dụng một mô hình ngôn ngữ lớn (LLM) đóng vai trò làm bộ phân tích cú pháp để ánh xạ ngôn ngữ tự nhiên thành cấu trúc điều kiện tìm kiếm.

### 1. Nạp ngữ cảnh hỗ trợ
* **Entity Hints:** Đưa danh sách thực thể đã quét được ở Tầng 0 vào Prompt để định hướng LLM nhận diện đúng tên riêng, giảm thiểu ảo giác.
* **Conversation Memory:** Nạp lịch sử trò chuyện của 6 lượt chat gần nhất để LLM hiểu được các từ thay thế (ví dụ: *"ông ấy"*, *"phim đó"*).

### 2. Trích xuất Bộ lọc Điều kiện (JSON Schema)
LLM phân tích câu hỏi và điền thông tin vào các trường lọc:
* **Thuộc tính văn bản:** Tiêu đề phim (`title`), Thể loại (`genre`), Đạo diễn (`director`), Diễn viên (`star`), Quốc gia (`country`).
* **Thuộc tính số học:** Năm phát hành tối thiểu/tối đa (`year_min`/`year_max`), Điểm số IMDb tối thiểu (`rating_min`), Thời lượng tối thiểu/tối đa (`runtime_min`/`runtime_max`).
* **Thuộc tính phủ định (Exclude):** Đạo diễn cần loại bỏ (`director_exclude`), Diễn viên cần loại bỏ (`star_exclude`).
* **Thuộc tính giải thưởng:** Phim đạt giải Oscar (`has_oscar`), Phim có giải thưởng nói chung (`has_awards`).
* **Sắp xếp:** Trường tiêu chuẩn để xếp hạng (`sort_by`) và chiều sắp xếp (`sort_order`).

### 3. Tự động sửa lỗi & Chuẩn hóa dữ liệu
* **Sửa lỗi nhầm lẫn cột:** Nếu người dùng hỏi tên đạo diễn nhưng LLM nhận diện nhầm vào trường `title`, hệ thống so sánh độ tương đồng chuỗi để tự động chuyển giá trị sang trường `director`.
* **Phân tích Logic Thể loại:** Nếu câu hỏi chứa liên từ biểu thị quan hệ đồng thời (như *"và"*, *"and"*, *"&"*), chế độ lọc thể loại được chuyển thành `AND` (bắt buộc có đủ các thể loại). Ngược lại, mặc định là chế độ `OR` (chỉ cần có một thể loại).
* **Chuẩn hóa Quốc gia:** Chuyển đổi các cách gọi quốc gia thông thường về tên chuẩn trong cơ sở dữ liệu (ví dụ: *"Mỹ"* hoặc *"US"* thành *"United States"*, *"Hàn Quốc"* thành *"South Korea"*).

---

## 📌 Tầng 2: Định tuyến Truy xuất (Retrieval Router)

Nhận đầu ra từ Tầng 1 và quyết định luồng đi tối ưu nhất cho dữ liệu để giảm thiểu thời gian xử lý và tối đa hóa độ liên quan.

### 1. Nhánh Aggregation (Thống kê & Quan hệ)
* **Kích hoạt:** Khi câu hỏi yêu cầu thống kê hoặc tìm người hợp tác (ví dụ: *"ai hợp tác nhiều nhất với..."*).
* **Hành vi:** Kích hoạt mô hình đồ thị tri thức để tìm kiếm mối quan hệ trực tiếp của nhân sự. Nếu người dùng hỏi về phim nhưng không có tên nhân sự trực tiếp, hệ thống tìm phim trước, bóc tách đạo diễn của phim đó rồi chạy truy vấn đồ thị.

### 2. Nhánh Phim Tương Tự (Similar Movie Request)
* **Kích hoạt:** Khi câu hỏi chứa các mẫu câu yêu cầu gợi ý tác phẩm tương tự (ví dụ: *"giống phim X"*, *"tương tự phim Y"*).
* **Hành vi:**
  1. Trích xuất phim gốc làm mốc so sánh.
  2. Truy cập đồ thị tri thức để tìm kiếm tất cả các phim liên kết gián tiếp (qua đạo diễn, diễn viên, thể loại chung) trong phạm vi 3 bước (3 hops) để tạo ra tập ứng viên ban đầu (`graph_candidates`).

### 3. Nhánh Phân tích Cứng (Exact Filter Shortcut)
* **Kích hoạt:** Khi câu hỏi chỉ chứa điều kiện lọc nhân sự đơn giản (Đạo diễn/Diễn viên cụ thể) mà không đi kèm mô tả nội dung phức tạp.
* **Hành vi:** Bỏ qua hoàn toàn các bước tìm kiếm vector và BM25, truy vấn trực tiếp trên DataFrame để lấy kết quả nhằm rút ngắn thời gian phản hồi xuống mức tối thiểu.

### 4. Nhánh Lai ghép Đa tầng (Multistage Hybrid)
* **Kích hoạt:** Các câu hỏi tìm kiếm ngữ nghĩa, lọc kết hợp hoặc tìm kiếm tự do.
* **Hành vi:** Chuyển tiếp yêu cầu sang quy trình truy xuất lai ghép đầy đủ ở Tầng 3.

---

## 📌 Tầng 3: Truy xuất Đa tầng (Multi-stage Retriever)

Đây là tầng xử lý cốt lõi của CineBot V3, kết hợp cả dữ liệu cấu trúc, phi cấu trúc và thông tin liên kết mạng lưới qua 5 giai đoạn liên tục.

### Giai đoạn 0: Tạo ứng viên từ Đồ thị (Graph BFS Candidates)
* **Phạm vi áp dụng:** Chỉ kích hoạt khi định tuyến xác định đây là truy vấn phim tương tự.
* **Thuật toán:** Duyệt theo chiều rộng (BFS) trên đồ thị NetworkX chứa **635.072 nút** và **3.291.584 cạnh**.
* **Tham số cấu hình:** Chiều sâu tối đa (`max_hops`) = **3**; Giới hạn số lượng láng giềng duyệt tại mỗi nút (`max_neighbors_per_hop`) = **20**.
* **Đầu ra:** Tối đa **300 ứng viên** có liên kết chặt chẽ nhất kèm thông tin đường đi liên kết.

### Giai đoạn 1: Sinh Ứng viên Lai (Candidate Generation)
Chạy song song ba nguồn tìm kiếm để tạo ra tập ứng viên thô ban đầu:
1. **Truy xuất Từ khóa (BM25 Search):** Quét tiêu đề, thể loại và tên nhân sự để lấy ra **100 ứng viên** khớp từ khóa tốt nhất.
2. **Truy xuất Ngữ nghĩa (FAISS Dense Search):** Dùng mô hình embedding mã hóa câu hỏi thành vector, thực hiện tìm kiếm trên cơ sở dữ liệu vector phẳng để lấy ra **150 ứng viên** có khoảng cách ngữ nghĩa gần nhất.
3. **Truy xuất Thuộc tính (Metadata Candidates):** Lấy **500 ứng viên** thỏa mãn nhiều điều kiện lọc cứng nhất từ cơ sở dữ liệu.
* **Hợp nhất & Loại trùng (Rank Fusion & Deduplication):** Gộp toàn bộ kết quả từ các nguồn trên (bao gồm cả ứng viên đồ thị ở Giai đoạn 0 nếu có). Loại bỏ các bản ghi trùng lặp dựa trên thuộc tính liên kết phim độc nhất (`Movie Link`), giữ lại tối đa **500 ứng viên** tốt nhất.

### Giai đoạn 2: Lọc Thuộc tính Cứng (Pandas Filtering)
* Áp dụng trực tiếp các phép toán logic và so sánh số học (>, <, ==, loại trừ) lên DataFrame của 500 ứng viên.
* **Đầu ra:** Loại bỏ toàn bộ các phim vi phạm ràng buộc về năm, điểm số, thời lượng, quốc gia hoặc đạo diễn/diễn viên bị cấm. Giữ lại tối đa **200 ứng viên** hợp lệ để đi vào tầng chấm điểm chi tiết.
* *Lưu ý:* Nếu bộ lọc quá khắt khe khiến danh sách ứng viên trống, hệ thống tự động chạy lại bộ lọc trên toàn bộ cơ sở dữ liệu gốc để tránh trả về kết quả rỗng cho người dùng.

### Giai đoạn 3: Chấm điểm Tương đồng Đa chiều (Weighted Similarity Engine)
So sánh từng ứng viên còn lại với yêu cầu của người dùng (hoặc với phim gốc) trên 8 khía cạnh đặc trưng độc lập:
1. **Nội dung (Trọng số 0.35):** Tính toán độ tương đồng cosine giữa vector embedding của mô tả phim.
2. **Thể loại (Trọng số 0.25):** Tính chỉ số trùng lặp Jaccard giữa hai tập hợp thể loại phim.
3. **Diễn viên (Trọng số 0.15):** Tính tỷ lệ trùng khớp diễn viên trong dàn diễn viên chính.
4. **Đạo diễn (Trọng số 0.10):** Tính tỷ lệ trùng khớp đạo diễn.
5. **Quốc gia (Trọng số 0.05):** Đánh giá mức độ trùng lặp quốc gia sản xuất.
6. **Kết nối Đồ thị (Trọng số 0.05):** Gán điểm tuyệt đối nếu ứng viên nằm trong luồng kết nối đồ thị BFS ở Giai đoạn 0.
7. **Thập kỷ phát hành (Trọng số 0.03):** Điểm số giảm dần theo khoảng cách thế hệ thời gian giữa hai phim.
8. **Giải thưởng (Trọng số 0.02):** Tính độ tương đồng cosine của vector trạng thái giải thưởng (Oscar, đề cử, giải thưởng khác).

* **Cơ chế Tái phân phối Trọng số (Weight Redistribution):** Nếu câu hỏi không yêu cầu một thuộc tính nào đó (ví dụ không yêu cầu về quốc gia hoặc giải thưởng), trọng số của thuộc tính đó sẽ được chuyển và chia đều cho các thuộc tính đang hoạt động để tránh làm giảm điểm tổng của ứng viên.
* **Đầu ra:** Sắp xếp danh sách ứng viên theo điểm tương đồng tổng hợp giảm dần và chọn ra **Top 100** phim đi tiếp.

### Giai đoạn 4: Xếp hạng lại bằng Mạng Nơ-ron (Cross-Encoder Reranking)
* Nạp câu hỏi và văn bản hồ sơ đầy đủ của 100 phim ứng viên vào mô hình Cross-Encoder chuyên biệt (`ms-marco-MiniLM-L-6-v2`).
* Mô hình Cross-Encoder thực hiện tính toán sự tương tác hai chiều trực tiếp giữa câu hỏi và văn bản mô tả phim để chấm điểm độ tương quan thực tế (Relevance Score), khắc phục nhược điểm chỉ khớp thuộc tính đơn lẻ của các tầng trước.
* **Đầu ra:** Chọn ra **Top 20** phim có điểm liên quan ngữ nghĩa cao nhất.

### Giai đoạn 5: Tái áp đặt Ràng buộc Cứng & Dọn dẹp cuối
* **Kiểm tra lại bộ lọc:** Áp dụng lại các bộ lọc điểm số tối thiểu (`rating_min`) và thời lượng (`runtime_min`/`runtime_max`) lên Top 20 để loại bỏ triệt để các phim bị lọt lưới do điểm tương đồng ngữ nghĩa của Cross-Encoder quá cao.
* **Loại bỏ trùng lặp:** Loại bỏ các phim thiếu thông tin quốc gia và tiến hành gom nhóm theo mã định danh phim (`imdb_id`), ưu tiên giữ lại bản ghi có thông tin thể loại đầy đủ nhất.
* **Đầu ra cuối cùng:** Trích xuất **Top-K** phim xuất sắc nhất (mặc định là **Top 5** phim) làm ngữ cảnh nạp cho LLM sinh câu trả lời.

---

## 📌 Tầng 4: Sinh Câu Trả Lời (Answer Generator)

Nhận danh sách phim sạch từ Tầng 3 và tiến hành biên dịch thành câu trả lời tự nhiên.

### 1. Đóng gói ngữ cảnh có cấu trúc
Hệ thống chuyển đổi dữ liệu của Top-5 phim thành các khối thông tin rõ ràng cấp cho LLM:
* Các thông số cơ bản (Tiêu đề, Đạo diễn, Diễn viên chính, Năm, Thời lượng, Điểm số, Quốc gia).
* Tóm tắt nội dung chính của phim.
* Lý do hệ thống lựa chọn bộ phim này (được sinh từ điểm số vượt trội của Weighted Similarity).
* Giải thích đường đi quan hệ trên đồ thị (nếu có).

### 2. Sinh câu trả lời tự nhiên bằng LLM
* **Chế độ Đồng bộ (Sync):** Chờ LLM hoàn thành toàn bộ câu trả lời rồi hiển thị một lần.
* **Chế độ Dòng chảy (Streaming):** Trả về từng token ngay khi LLM vừa xử lý xong, giúp người dùng nhìn thấy câu trả lời xuất hiện ngay lập tức, tối ưu hóa trải nghiệm tương tác.
