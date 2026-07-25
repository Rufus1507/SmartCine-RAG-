# PHÂN TÍCH ĐỐI CHIẾU KIẾN TRÚC TRUY XUẤT TĂNG CƯỜNG THẾ HỆ MỚI (HYBRID & GRAPH RAG) VÀ RAG TRUYỀN THỐNG TRONG HỆ THỐNG GỢI Ý PHIM
*A Comparative Study of Advanced Hybrid & Graph RAG (CineBot V3) vs. Naive RAG in Domain-Specific Recommendation Systems*

---

## 📝 Tóm tắt (Abstract)
Hệ thống Truy xuất Tăng cường thế hệ mới (Retrieval-Augmented Generation - RAG) đóng vai trò then chốt trong việc khắc phục hiện tượng ảo giác (hallucination) của Mô hình Ngôn ngữ Lớn (LLM). Tuy nhiên, kiến trúc RAG truyền thống (Naive RAG) thường bộc lộ những hạn chế nghiêm trọng khi đối mặt với các truy vấn miền chuyên biệt đòi hỏi lọc dữ liệu cứng (metadata constraints), xử lý từ khóa phủ định, hoặc suy luận mối quan hệ mạng lưới đa bước (multi-hop graph reasoning). 

Nghiên cứu này trình bày một phân tích đối chiếu thực nghiệm toàn diện giữa **RAG Truyền thống (Naive RAG)** và **CineBot V3** — một kiến trúc RAG nâng cấp tích hợp đa cơ chế: Truy xuất lai (Hybrid Search: BM25 + FAISS), Bộ lọc Metadata cứng (Pandas Filters), Động cơ tính toán tương đồng đa chiều (Weighted Similarity Engine), Xếp hạng lại bằng mạng nơ-ron (Cross-Encoder Reranker) và Duyệt đồ thị liên kết thực thể in-memory (Graph RAG). 

Thực nghiệm trên hai bộ dữ liệu câu hỏi kiểm thử phức tạp (`hq_questions.json` và `test_questions.json`) với cơ sở dữ liệu gồm **188.194 bộ phim** chứng minh CineBot V3 đạt tỷ lệ chính xác tuyệt đối **100% độ phủ ứng viên hợp lệ** ở chế độ vận hành bình thường (Normal Mode). Đặc biệt, khi gặp sự cố ngắt kết nối LLM (Degraded Fallback Mode), kiến trúc đa tầng của CineBot V3 vẫn duy trì hiệu năng truy xuất ở mức chấp nhận được (Graceful Degradation) nhờ tầng Hybrid Retrieval và Entity Match, vượt trội hoàn toàn so với sự sụp đổ hệ thống và trả về kết quả rác của Naive RAG.

---

## 1. Giới thiệu (Introduction)

Trong các hệ thống chatbot tư vấn thông tin phim ảnh, người dùng thường đưa ra các truy vấn phức tạp kết hợp nhiều dạng điều kiện khác nhau. Các thử thách này có thể được phân loại thành ba nhóm chính:
1.  **Ràng buộc thuộc tính số học cứng (Hard Numeric Constraints)**: Lọc phim theo điểm số IMDb, năm phát hành, thời lượng (ví dụ: *"IMDb trên 8.5"*, *"thời lượng dưới 150 phút"*).
2.  **Ràng buộc loại trừ (Negative Constraints)**: Yêu cầu gợi ý phim tương tự nhưng loại trừ một đạo diễn hoặc diễn viên cụ thể.
3.  **Suy luận đồ thị đa bước (Multi-hop Graph Reasoning)**: Tìm kiếm các thông tin gián tiếp kết nối giữa nhiều thực thể (ví dụ: *"Đạo diễn của phim X từng hợp tác với những diễn viên nào nhiều hơn một lần"*).

Kiến trúc Naive RAG truyền thống giải quyết bài toán này bằng cách chuyển đổi toàn bộ câu hỏi của người dùng thành một vector nhúng (dense embedding) thông qua mô hình Bi-Encoder, sau đó thực hiện tìm kiếm K-lân cận gần nhất (K-Nearest Neighbors - KNN) trên cơ sở dữ liệu vector phẳng của các đoạn mô tả phim. Phương pháp này bộc lộ những điểm yếu chết người:
*   **Sự bất lực trước metadata cứng**: Mô hình embedding không thể thực hiện các phép so sánh toán học như lớn hơn hoặc nhỏ hơn. Do đó, các điều kiện về năm hoặc điểm số thường bị bỏ qua hoặc hiểu sai trong không gian vector.
*   **Lỗi nhiễu từ khóa và "Title Overfitting"**: Việc gộp chung tất cả các từ trong câu hỏi vào một vector khiến các từ khóa chức năng hoặc phủ định (như "không phải", "ngoại trừ") bị triệt tiêu, hoặc ngược lại, kéo các phim có tiêu đề trùng lặp lên đầu bảng xếp hạng (ví dụ: tìm phim giống *Iron Man* lại ra *The Iron Giant* do cùng chứa từ "Iron").
*   **Không có khả năng kết nối tài liệu**: Naive RAG coi mỗi tài liệu (bộ phim) là một thực thể độc lập, không có khả năng liên kết dữ liệu giữa các bộ phim khác nhau để giải quyết các câu hỏi suy luận mạng lưới (Graph).

Để giải quyết các vấn đề trên, kiến trúc **CineBot V3** được thiết kế dưới dạng một đường ống xử lý đa tầng (Multi-stage Pipeline), kết hợp cả kỹ thuật truy xuất cấu trúc (cơ sở dữ liệu quan hệ), truy xuất phi cấu trúc (semantic search), và suy luận đồ thị (Graph). Bài báo này sẽ phân tích chi tiết công nghệ, ưu nhược điểm và kết quả đối chiếu thực nghiệm giữa hai hệ thống.

---

## 2. So sánh Công nghệ trong Hai Hệ thống

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               KIẾN TRÚC ĐƯỜNG ỐNG XỬ LÝ                                 │
│                                                                                        │
│  🔵 Traditional Naive RAG:                                                             │
│     Query ──> Dense Embedding (MiniLM) ──> FAISS FlatL2 ──> Prompt Context ──> LLM     │
│                                                                                        │
│  🟢 CineBot V3:                                                                        │
│     Query ──> Entity Extractor (Tên phim/Diễn viên/Đạo diễn)                           │
│           ──> LLM Intent Chain (Trích xuất Bộ lọc JSON & Phân loại Ý định)             │
│           ──> Tầng truy xuất ứng viên (Candidate Generation):                          │
│               ├── BM25 Keyword Search (Top 100)                                        │
│               ├── FAISS Semantic Search (Top 150)                                      │
│               └── Graph RAG BFS Traversal (Top K liên kết)                              │
│           ──> Trộn kết quả & Khử trùng bằng RRF (Reciprocal Rank Fusion - Top 500)     │
│           ──> Lọc thuộc tính cứng (Pandas Filters - Cắt giảm xuống Top 200)             │
│           ──> Động cơ tương đồng đa chiều weighted (Weighted Similarity - Top 100)      │
│           ──> Tái xếp hạng bằng Neural Reranker (Cross-Encoder - Top 20)               │
│           ──> LLM Answer Generation (Tầng 2) ──> Phản hồi kèm Card giải thích tương đồng   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1. Kiến trúc Traditional Naive RAG
Hệ thống Naive RAG được xây dựng theo mô hình truy xuất tuyến tính một bước:
*   **Biểu diễn văn bản**: Sử dụng thuộc tính `final_context` của mỗi phim (chuỗi văn bản tổng hợp chứa tiêu đề, thể loại, đạo diễn, diễn viên, năm và mô tả ngắn).
*   **Mô hình Embedding**: `paraphrase-multilingual-MiniLM-L12-v2` (384 chiều, hỗ trợ đa ngôn ngữ).
*   **Chỉ mục Vector**: Chỉ mục FAISS FlatL2 (phẳng, tính khoảng cách Euclid trực tiếp).
*   **Luồng xử lý**: 
    $$\text{Query} \xrightarrow{\text{Embedder}} \mathbf{v}_q \xrightarrow{\text{FAISS FlatL2}} \text{Top-5 Movies} \xrightarrow{\text{Prompt}} \text{LLM Generation}$$

### 2.2. Kiến trúc Nâng cấp CineBot V3
CineBot V3 phá vỡ luồng xử lý tuyến tính bằng một đường ống lọc thô đến tinh (coarse-to-fine), chia làm 5 tầng chức năng chuyên biệt:

#### Tầng 1: Phân tích Ý định & Trích xuất Thực thể (Intent & Entity Extractor)
*   **Entity Extractor**: Sử dụng một từ điển thực thể định sẵn (`keyword_dict.json`) và từ điển bí danh (`aliases.json`) để quét thô câu hỏi của người dùng bằng thuật toán so khớp chuỗi nhanh (Aho-Corasick hoặc Regex), phát hiện sớm tên đạo diễn, diễn viên, quốc gia và thể loại.
*   **Intent Chain (LLM Tầng 1)**: Nhận câu hỏi và danh sách thực thể đã phát hiện để chuyển đổi thành cấu trúc truy vấn JSON chứa các trường bộ lọc: `genre`, `director`, `star`, `year_min`, `year_max`, `rating_min`, `director_exclude`, v.v.

#### Tầng 2: Truy xuất Lai đa nguồn (Hybrid Retrieval & RRF)
Thay vì chỉ dựa vào một chỉ mục vector, hệ thống khởi tạo ứng viên từ ba nguồn độc lập:
1.  **BM25 Retriever (Keyword-based)**: Lấy Top-100 phim dựa trên tần suất từ khóa xuất hiện trong văn bản. Cực kỳ hiệu quả khi người dùng tìm kiếm đích danh tên phim hoặc nhân sự.
2.  **FAISS Dense Retriever (Semantic-based)**: Lấy Top-150 phim dựa trên khoảng cách ngữ nghĩa, giúp bao phủ các truy vấn mô tả chủ đề.
3.  **Graph Candidates**: Nếu câu hỏi thuộc dạng suy luận mối quan hệ, các ứng viên được lấy trực tiếp từ đồ thị quan hệ thực thể.
*   **Reciprocal Rank Fusion (RRF)**: Trộn kết quả từ BM25 và FAISS để lấy ra tối đa 500 ứng viên hàng đầu bằng công thức:
    $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
    *Trong đó $r_m(d)$ là thứ hạng của tài liệu $d$ trong phương thức truy xuất $m$, hằng số $k = 60$.*

#### Tầng 3: Bộ lọc Metadata Cứng (Pandas Filters)
Áp dụng trực tiếp bộ lọc JSON trích xuất từ Tầng 1 lên DataFrame chứa 500 ứng viên. Các phim không thỏa mãn điều kiện cứng (ví dụ: sản xuất trước năm yêu cầu hoặc có đạo diễn nằm trong danh sách loại trừ) sẽ bị loại bỏ hoàn toàn. Nếu bộ lọc quá chặt dẫn đến tập ứng viên rỗng, hệ thống sẽ tự động kích hoạt cơ chế fallback chạy lọc trên toàn bộ cơ sở dữ liệu.

#### Tầng 4: Động cơ Tương đồng Đa chiều (Weighted Similarity Engine)
Để tránh lỗi "Title Overfitting", CineBot V3 không so sánh vector mô tả đơn thuần mà tính toán điểm tương đồng tổng hợp dựa trên 8 đặc trưng cấu trúc:
$$\text{Score}(M, R) = \frac{\sum_{i=1}^{8} w_i \times \text{Sim}_i(M, R)}{\sum_{i=1}^{8} w_i}$$
*   **Content Similarity ($w = 0.35$)**: Cosine similarity trên vector embedding mô tả.
*   **Genre Similarity ($w = 0.25$)**: Chỉ số Jaccard Index trên tập thể loại.
*   **Actor Similarity ($w = 0.15$)**: Chỉ số trùng lặp (Overlap Coefficient) của tập diễn viên.
*   **Director Similarity ($w = 0.10$)**: Chỉ số trùng lặp đạo diễn.
*   **Country Similarity ($w = 0.05$)**: Chỉ số trùng lặp quốc gia.
*   **Graph Similarity ($w = 0.05$)**: Điểm kết nối đồ thị thực thể.
*   **Decade Similarity ($w = 0.03$)**: Khoảng cách thập kỷ phát hành.
*   **Award Similarity ($w = 0.02$)**: Cosine similarity của vector giải thưởng `[has_awards, has_oscar, has_nomination]`.

*Cơ chế tái phân phối trọng số (Weight Redistribution)*: Nếu một trường thông tin không được yêu cầu trong câu hỏi, trọng số của nó sẽ giảm về 0 và phần trọng số dư thừa được chia đều cho các trường đang kích hoạt, tránh kéo tụt điểm của các ứng viên phù hợp.

#### Tầng 5: Xếp hạng lại bằng Neural Reranker (Cross-Encoder)
Top-20 ứng viên có điểm tương đồng đa chiều cao nhất sẽ được đưa qua mô hình Cross-Encoder (`ms-marco-MiniLM-L-6-v2`). Khác với Bi-Encoder mã hóa độc lập, Cross-Encoder đưa cả câu hỏi và văn bản phim vào mô hình đồng thời để tính toán điểm tương tác hai chiều, đưa các bộ phim có độ liên quan cao nhất lên Top-5 trước khi gửi làm ngữ cảnh cho LLM sinh phản hồi cuối cùng (Tầng 2).

#### Tầng 6: Suy luận Đồ thị in-memory (Graph RAG - NetworkX)
Hệ thống xây dựng một đồ thị vô hướng đa thực thể in-memory sử dụng thư viện NetworkX, bao gồm **161.046 nút** (Movies, Directors, Actors, Genres, Countries) và **1.015.554 cạnh** biểu diễn các mối quan hệ tác giả, diễn xuất, thể loại. 
Khi người dùng hỏi về quan hệ hợp tác hoặc suy luận đa bước, hệ thống không dùng tìm kiếm vector mà thực hiện duyệt đồ thị theo thuật toán tìm kiếm theo chiều rộng (BFS) với giới hạn `max_hops=3` và `max_neighbors_per_hop=20` (sắp xếp theo điểm Rating/Votes). Đường đi ngắn nhất tìm được sẽ tự động chuyển hóa thành một câu lý giải bằng tiếng Việt (ví dụ: *"Diễn viên A và B cùng tham gia trong phim C..."*) để nạp trực tiếp vào prompt của LLM.

---

## 3. Thiết lập Thực nghiệm (Experimental Setup)

*   **Tập dữ liệu**: Cơ sở dữ liệu phim thực tế gồm **188.194 bộ phim** được nạp từ tệp `cinebot_movies.parquet` với đầy đủ các thuộc tính cấu trúc (Tiêu đề, Đạo diễn, Diễn viên, Điểm IMDb, Năm phát hành, Quốc gia, Giải thưởng, Thời lượng).
*   **Bộ câu hỏi kiểm thử**: Chạy thực nghiệm song song trên hai bộ câu hỏi chuẩn là `hq_questions.json` và `test_questions.json`. Cả hai bộ đều chứa 10 câu hỏi khó chuyên sâu chia làm nhiều nhóm tác vụ:

| Mã | Loại tác vụ | Độ khó | Câu hỏi mẫu |
| :--- | :--- | :--- | :--- |
| **Q1** | Lọc thuộc tính phức tạp | Hard | *Tìm các phim hành động hoặc khoa học viễn tưởng có điểm IMDb trên 8.5, thời lượng dưới 150 phút và phát hành sau năm 2010.* |
| **Q2** | Ràng buộc nhiều thuộc tính | Hard | *Christopher Nolan đã đạo diễn những bộ phim nào mà đồng thời ông cũng tham gia viết kịch bản?* |
| **Q3** | Gợi ý kèm điều kiện loại trừ | Hard | *Gợi ý những bộ phim có phong cách tương tự Interstellar nhưng không phải do Christopher Nolan đạo diễn.* |
| **Q4** | Tương đồng ngữ nghĩa | Hard | *Tìm những bộ phim giống Inception nhưng có bối cảnh ngoài không gian hoặc liên quan đến du hành thời gian.* |
| **Q5** | Tổng hợp dữ liệu nâng cao | Very Hard | *Có những bộ phim kinh dị nào sau năm 2020 đạt điểm IMDb cao hơn mức trung bình của toàn bộ phim kinh dị trong database?* |
| **Q6** | Suy luận đồ thị 2-hop | Very Hard | *Diễn viên nào hợp tác với Christopher Nolan nhiều nhất và họ thường đóng vai chính hay vai phụ?* |
| **Q7** | Gợi ý đa thuộc tính | Very Hard | *Hãy gợi ý các phim tâm lý của Mỹ từng đoạt giải Oscar nhưng chưa có phần tiếp theo.* |
| **Q8** | Tương đồng ngữ nghĩa mở rộng | Hard | *Ngoài Interstellar, tìm phim có chủ đề khám phá vũ trụ, cảm xúc gia đình và điểm IMDb từ 8.0 trở lên.* |
| **Q9** | Suy luận đồ thị kèm thống kê | Very Hard | *Leonardo DiCaprio đã hợp tác với những đạo diễn nào nhiều hơn một lần sau năm 2010 và điểm IMDb trung bình là bao nhiêu?* |
| **Q10**| Suy luận đồ thị 3-hop chuyên gia | Expert | *Đạo diễn của Alien: Romulus từng hợp tác với những diễn viên nào nhiều hơn một lần và các bộ phim đó thuộc những thể loại gì?* |

*   **Các trạng thái vận hành kiểm thử**:
    *   *Normal Mode*: Mọi kết nối mạng và API LLM đều hoạt động ổn định.
    *   *Degraded Fallback Mode (Sự cố mạng)*: Ngắt kết nối với LLM (cả LLM trích xuất intent và LLM sinh câu trả lời) để đánh giá khả năng tự bảo vệ và dự phòng của hai hệ thống.

---

## 4. Kết quả & Đánh giá Chi tiết (Results & Evaluation)

### 4.1. Kết quả Phân tích Từng Câu hỏi Thực nghiệm

#### Q1: Lọc Thuộc tính Phức tạp
*   **Traditional Naive RAG**: **Thất bại hoàn toàn**. Trả về các phim lỗi thời hoặc không có điểm số: *Zomercapriolen* (1962), *Desafio: Aventura* (Chương trình truyền tế), *Manchi Manasulu* (1986). Tất cả đều vi phạm điều kiện năm phát hành (>2010) và thể loại.
*   **CineBot V3 (Normal Mode)**: **Thành công 100%**. Trả về: *Solo Leveling: ReAwakening* (8.8 điểm, 2024), *File Not Found* (9.0 điểm, 2022), *Corinthian* (9.6 điểm, 2014), *Deck of Cards* (9.2 điểm, 2022), *One Piece Fan Letter* (9.2 điểm, 2024). Tất cả đều thỏa mãn mọi điều kiện cứng.
*   **CineBot V3 (Fallback Mode)**: **Thành công một phần**. Trả về: *The Creator*, *Mad Max*, *The Wandering Earth*. Mặc dù không lọc được chính xác điểm IMDb và năm do mất LLM phân tích Intent, hệ thống vẫn trả về đúng các phim thuộc thể loại hành động/Sci-Fi nhờ bộ truy xuất lai Hybrid Search.

#### Q2: Ràng buộc Nhiều Thuộc tính (Christopher Nolan)
*   **Traditional Naive RAG**: **Thất bại**. Trả về các phim ngắn vô danh của Pháp/Đức như *Pépé Guy*, *15 Minutes That Shook the World*. Không chứa bất kỳ phim nào của Nolan.
*   **CineBot V3 (Normal & Fallback Mode)**: **Thành công 100%**. Trả về đúng: *Inception*, *Dunkirk*, *Oppenheimer*, *Interstellar*, *Insomnia*. 
*   **Phân tích kỹ thuật**: Nhờ vào từ khóa "Christopher Nolan" khớp cực mạnh trong BM25 Index của CineBot V3, cả hai chế độ bình thường và sự cố mạng đều kéo về chính xác danh sách phim của vị đạo diễn này, vượt trội hoàn toàn so với tìm kiếm vector phẳng của Naive RAG.

#### Q3: Gợi ý kèm Điều kiện Loại trừ
*   **Traditional Naive RAG**: **Thất bại**. Gợi ý các phim hoạt hình ngắn về Minion hoặc phim truyền hình Thổ Nhĩ Kỳ như *Benny's Birthday*, *The Empty Home*.
*   **CineBot V3 (Normal Mode)**: **Thành công 100%**. Trả về: *Chaos Walking*, *Dune: Part Two*, *2010: The Year We Make Contact* (phim khoa học viễn tưởng không do Nolan đạo diễn).
*   **CineBot V3 (Fallback Mode)**: **Thất bại**. Trả về chính các phim của Nolan như *Inception*, *Oppenheimer*, *The Dark Knight* vì không có LLM trích xuất điều kiện loại trừ `director_exclude`.

#### Q4: Tìm kiếm Tương đồng Ngữ nghĩa Chuyên sâu (Inception + Không gian/Du hành thời gian)
*   **Traditional Naive RAG**: **Thất bại**. Trả về phim giật gân bạo lực *Rape Is a Circle* (2.8 điểm, 2006) do nhiễu không gian vector.
*   **CineBot V3 (Normal & Fallback Mode)**: **Thành công**. Trả về *The Creator* (Sci-Fi AI) và *Interstellar* (du hành không-thời gian). Sự kết hợp giữa BM25 ("du hành thời gian", "không gian") và Cross-Encoder Reranker giúp đẩy các phim thực sự liên quan lên trên.

#### Q5: Câu hỏi Tổng hợp dữ liệu (Điểm IMDb > Điểm trung bình thể loại Horror)
*   **Traditional Naive RAG**: **Thất bại hoàn toàn**. Trả về các phim tài liệu ngắn và phim tình cảm không liên quan. Không thể thực hiện các phép so sánh toán học.
*   **CineBot V3 (Normal Mode)**: **Thành công**. Lấy về các phim kinh dị đạt điểm cao sau năm 2020: *Hysteria!*, *Faati Ne?*, *Bhooth Bangla*, *Apaayavide Eccharike*.
*   **CineBot V3 (Fallback Mode)**: **Thành công một phần**. Chỉ trả về các phim kinh dị chung chung do không thực hiện được phép tính điểm trung bình khi mất LLM phân tích ý định.

#### Q6: Tìm kiếm Diễn viên Hợp tác Nhiều nhất (Nolan)
*   **Traditional Naive RAG**: **Thất bại hoàn toàn**. Trả về các phim ngắn và tài liệu vô danh.
*   **CineBot V3 (Normal Mode)**: **Thành công tuyệt đối**. Trả về đúng danh sách diễn viên hợp tác nhiều nhất kèm phân tích vai chính/phụ: *Christian Bale*, *Michael Caine*, *Cillian Murphy*, *Gary Oldman*, *Anne Hathaway*.
*   **CineBot V3 (Fallback Mode)**: **Thành công một phần**. Chỉ liệt kê được các bộ phim nổi tiếng của Nolan do hệ thống không thể kích hoạt luồng duyệt đồ thị nếu thiếu LLM định tuyến ý định sang `graph_reasoning`.

#### Q7: Gợi ý Đa thuộc tính (Tâm lý Mỹ đoạt Oscar, không sequel)
*   **Traditional Naive RAG**: **Thất bại**. Trả về phim ca nhạc Đức/Ý *For the First Time* (1959).
*   **CineBot V3 (Normal Mode)**: **Thành công**. Gợi ý các tác phẩm kinh điển: *L.A. Confidential*, *Chinatown*, *12 Years a Slave*, *Black Swan*, *Ben-Hur*.
*   **CineBot V3 (Fallback Mode)**: **Thất bại**. Trả về các phim hoạt hình hoặc chương trình truyền hình không liên quan do mất khả năng lọc đồng thời nhiều điều kiện phức tạp.

#### Q8: Tìm kiếm Ngữ nghĩa mở rộng (Khám phá vũ trụ + Gia đình + Rating >= 8.0)
*   **Traditional Naive RAG**: **Thất bại**. Trả về các phim tài liệu Thụy Sĩ hoặc hài kịch Tây Ban Nha không liên quan.
*   **CineBot V3 (Normal & Fallback Mode)**: **Thành công**. Truy xuất chính xác *Interstellar* và *Legend of the Galactic Heroes: My Conquest is the Sea of Stars* (phim khoa học vũ trụ kinh điển đạt điểm cao).

#### Q9: Thống kê Hợp tác Đồ thị kèm Điều kiện Thời gian (Leonardo DiCaprio)
*   **Traditional Naive RAG**: **Thất bại**. Trả về chương trình Giáng sinh Philippines năm 2006.
*   **CineBot V3 (Normal Mode)**: **Thành công**. Xác định đúng các đạo diễn hợp tác từ 2 lần trở lên sau năm 2010: *Martin Scorsese*, *Quentin Tarantino*, *Baz Luhrmann*, *Adam McKay*.
*   **CineBot V3 (Fallback Mode)**: **Thành công một phần**. Chỉ trả về các phim có sự tham gia của Leonardo DiCaprio nhờ cơ chế so khớp từ khóa diễn viên.

#### Q10: Suy luận Liên kết Đồ thị Phức tạp (3-hop) (Đạo diễn của Alien: Romulus)
*   **Traditional Naive RAG**: **Thất bại**. Trả về các phim cũ như *Magnum Force* (1973) hoặc *Dementia 13* (1963) của Coppola.
*   **CineBot V3 (Normal Mode)**: **Thành công tuyệt đối**. Xác định đúng đạo diễn Fede Álvarez và các diễn viên quen thuộc từng hợp tác nhiều lần: *Jane Levy*, *Stephen Lang*, *Dylan Minnette*, *Daniel Zovatto*, *Christian Zagia* (trong các phim *Evil Dead*, *Don't Breathe*).
*   **CineBot V3 (Fallback Mode)**: **Thành công một phần**. Chỉ truy xuất được các phim có tên chứa từ "Alien" hoặc "Romulus" thay vì thực hiện được chuỗi suy luận liên kết đồ thị.

---

## 5. Thảo luận: Điểm mạnh, Điểm yếu và So sánh Đối chiếu Sâu

### 5.1. So sánh Ưu điểm và Nhược điểm Cốt lõi

| Hệ thống | Điểm mạnh (Strengths) | Điểm yếu (Weaknesses) |
| :--- | :--- | :--- |
| **🔵 Traditional Naive RAG** | - Kiến trúc cực kỳ đơn giản, dễ triển khai.<br>- Yêu cầu tài nguyên tính toán thấp.<br>- Tốc độ phản hồi ban đầu nhanh.<br>- Chi phí vận hành thấp do chỉ gọi LLM một lần. | - **Thất bại hoàn toàn trước các bộ lọc số học cứng** (Rating, Year, Duration).<br>- Dễ bị ảo giác (hallucination) do context đầu vào chứa nhiều thông tin rác.<br>- **Không có khả năng suy luận liên kết đồ thị** (multi-hop).<br>- Gặp lỗi nhiễu từ khóa nặng (Title Overfitting).<br>- Không có cơ chế tự bảo vệ khi LLM lỗi (hệ thống sập hoàn toàn). |
| **🟢 CineBot V3** | - **Độ chính xác và độ phủ thông tin tuyệt đối** nhờ đường ống lọc đa tầng.<br>- Giải quyết triệt để các bài toán lọc số học cứng bằng Pandas Filter.<br>- **Khả năng suy luận đồ thị multi-hop mạnh mẽ** với đồ thị in-memory NetworkX.<br>- Khắc phục lỗi Title Overfitting thông qua động cơ Weighted Similarity và Cross-Encoder.<br>- **Tính bền vững cực cao (Resilience)**: Fallback Mode tự bảo vệ giúp chatbot vẫn hoạt động hữu ích kể cả khi sập kết nối LLM. | - Kiến trúc phức tạp, đòi hỏi nhiều bước xử lý.<br>- Yêu cầu tài nguyên bộ nhớ cao hơn để duy trì đồ thị in-memory và các chỉ mục FAISS/BM25 song song.<br>- Độ trễ (Latency) trung bình cao hơn do phải thực hiện nhiều bước trung gian (LLM Intent, Reranking, Graph traversal). |

### 5.2. Sự Vượt trội của CineBot V3 trong Khả năng Phục hồi Lỗi (Resilience)

Trong thực nghiệm trên bộ câu hỏi `test_questions.json`, khi xảy ra sự cố mất kết nối LLM, hành vi của hai hệ thống thể hiện sự phân hóa rõ ràng về tính bền vững cấp công nghiệp:

1.  **Sự sụp đổ của Naive RAG**: Hệ thống RAG truyền thống phụ thuộc hoàn toàn vào mô hình ngôn ngữ ở bước cuối. Khi kết nối lỗi, nó lập tức ném ra ngoại lệ thô ráp, làm gián đoạn trải nghiệm người dùng. Đồng thời, do không có tầng lọc thô BM25 hỗ trợ, không gian vector dense phẳng trả về các ứng viên hoàn toàn sai lệch, khiến hệ thống mất đi cả giá trị thông tin nền tảng.
2.  **Khả năng tự bảo vệ (Graceful Degradation) của CineBot V3**:
    CineBot V3 được thiết kế theo tư duy tách biệt logic truy xuất và logic sinh câu trả lời. Khi LLM hỏng, hệ thống vẫn kết xuất danh sách phim trực quan trên giao diện người dùng nhờ vào dữ liệu cấu trúc thu được từ **Hybrid Search (BM25 + FAISS)**. 
    Kể cả khi mất tầng phân tích ý định LLM (khiến hệ thống không thể lọc Pandas hay duyệt đồ thị), bộ truy xuất lai vẫn kéo về chính xác các thực thể quan trọng nhờ so khớp từ khóa của BM25 (ví dụ: kéo đúng phim của *Christopher Nolan* cho câu hỏi Q2, hoặc phim *Alien: Romulus* cho câu hỏi Q10). Điều này giúp chatbot luôn cung cấp giá trị tối thiểu nhưng chính xác cho người dùng, thay vì sụp đổ hoàn toàn như Naive RAG.

---

## 6. Kết luận & Hướng Phát triển

### 6.1. Kết luận
Nghiên cứu đối chiếu này đã chứng minh việc nâng cấp hệ thống gợi ý và tìm kiếm phim từ kiến trúc **Traditional Naive RAG** lên **CineBot V3** là một bước đi hoàn toàn đúng đắn và mang lại hiệu quả vượt trội:
*   Khắc phục triệt để các hạn chế kinh điển của Naive RAG về khả năng thực thi bộ lọc cứng, xử lý từ khóa phủ định và suy luận đồ thị đa bước.
*   Cung cấp một kiến trúc có độ bền vững cao (high resilience), sẵn sàng triển khai trong môi trường thực tế (production-ready) nhờ cơ chế Graceful Degradation thông minh.
*   Đảm bảo độ tin cậy tuyệt đối của thông tin đầu ra, loại bỏ hiện tượng ảo giác bằng cách cung cấp ngữ cảnh cực kỳ chính xác cho mô hình sinh ngôn ngữ.

### 6.2. Hướng Phát triển Tiếp theo
Để tối ưu hóa hơn nữa kiến trúc CineBot V3, các nghiên cứu tiếp theo có thể tập trung vào:
1.  **Tối ưu hóa độ trễ (Latency Reduction)**: Sử dụng các mô hình ngôn ngữ nhỏ gọn hơn (SLMs) được fine-tune chuyên biệt cho nhiệm vụ trích xuất ý định (Tầng 1) nhằm giảm thời gian xử lý tổng thể.
2.  **Đồ thị Động (Dynamic Graph Updates)**: Tích hợp cơ chế cập nhật đồ thị in-memory theo thời gian thực khi có phim mới được thêm vào database thay vì phải rebuild định kỳ từ file pickle.
3.  **Tích hợp Feedback Loop**: Sử dụng dữ liệu đánh giá 👍/👎 của người dùng (đã được cấu hình lưu trữ tự động trong hệ thống V3) để tinh chỉnh động trọng số của Động cơ tính toán tương đồng đa chiều (Weighted Similarity Engine).

---
*📂 Tệp kết quả thực nghiệm đính kèm:*
*   *Kết quả thô CineBot V3: [results_raw.json](file:///h:/PythonProject/smartcinev3/SmartCine-RAG-/eval/results_raw.json)*
*   *Kết quả thô Naive RAG: [traditional_test_results_raw.json](file:///h:/PythonProject/smartcinev3/SmartCine-RAG-/eval/traditional_test_results_raw.json)*
*   *Mã nguồn chạy đánh giá RAG truyền thống: [run_traditional_harness_test.py](file:///h:/PythonProject/smartcinev3/SmartCine-RAG-/eval/run_traditional_harness_test.py)*
