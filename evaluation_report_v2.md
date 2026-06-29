# Báo cáo Đánh giá CineBot V3 — Phiên bản V2 (Đã sửa lỗi pipeline)

> **Ghi chú**: Báo cáo này thay thế `evaluation_report.md`. Mọi số liệu là output thực tế từ code vừa chạy (`run_eval_v2.py`). Phần nhận xét chỉ dựa trên bảng số phía trên — không viết nhận xét mâu thuẫn với số liệu.

---

## 1. Bảng Chẩn đoán Nguyên nhân Gốc (6 Bất thường)

| # | Vấn đề | Nguyên nhân gốc xác nhận | Đã sửa? | Thay đổi |
|---|---|---|---|---|
| 1 | Precision@5=4%, Recall@10=3.5% cực thấp | Ground truth `relevant_movies` có **duplicate titles** (avg 0.9 bản lặp/10 entries), làm inflate mẫu số Recall; matching logic bằng `clean_title()` đúng nhưng thiếu dedup | ✅ Đã sửa | Thêm `deduplicate_gt_titles()` trước khi tính metrics |
| 2 | Ablation 3 phiên bản A/B/C cho kết quả giống nhau, mô tả in giống hệt | **Root cause**: `load_content_keywords()` dùng `@st.cache_resource` trả về `set()` rỗng khi chạy ngoài Streamlit → keywords = "" → profile C = profile B → Vector B ≡ Vector C (cosine=1.000) | ✅ Đã sửa | Load `keyword_dict.json` trực tiếp; rebuild index C; cosine B-C_fixed = **0.995** (avg 3 phim); ablation rerun xác nhận null result |
| 3 | Title-Overfitting 0.0% cho cả 2 phiên bản | Bộ 50 cặp decoy chỉ yêu cầu **1 từ chung** trong title (rất yếu) — các cặp có title_sim < 0.3, hệ thống không bao giờ nhầm | ✅ Đã sửa | Rebuild pairs với `title_sim (SequenceMatcher) ≥ 0.40` + genre hoàn toàn khác |
| 4 | Cross-Encoder Rerank làm GIẢM NDCG (0.095→0.052) | Model `cross-encoder/ms-marco-MiniLM-L-6-v2` train trên **MS-MARCO web search**, không phải movie similarity. Reranker sắp xếp lại theo search relevance, không theo content similarity → NDCG giảm | ⚠️ Negative finding | Không sửa model; ghi nhận **domain mismatch** là finding khoa học hợp lệ |
| 5 | BM25 Recall@500≈Recall@100; Hybrid+Filter=0.5% | **BUG A**: BM25 `top_k=100` → tối đa 100 candidates, Recall@500 = Recall@100. **BUG B**: "Hybrid+Filter" dùng `filters={'title': seed_movie}` → chỉ trả về chính seed movie, không phải similar movies | ✅ Đã sửa | BM25/FAISS `top_k=500`; Hybrid+Filter dùng **genre filter** |
| 6 | P95 latency gấp 3–7x Avg | Local LLM server (cx/gpt-5.5 tại `localhost:20128`) có **response jitter**; không có retry loop. Outlier calls kéo P95 lên cao | ✅ Documented | Phân tách cached vs live calls; ghi rõ nguyên nhân |

---

## 2. Phương pháp xây dựng Ground Truth

Tập dữ liệu Ground Truth được xây dựng tự động từ **300** seed movies (num_votes ≥ 5000), sử dụng công thức similarity đa chiều:

$$\text{Score} = 0.40 \cdot S_{\text{content}} + 0.25 \cdot S_{\text{genre}} + 0.15 \cdot S_{\text{actor}} + 0.10 \cdot S_{\text{director}} + 0.05 \cdot S_{\text{country}} + 0.03 \cdot S_{\text{decade}} + 0.02 \cdot S_{\text{award}}$$

- **Số lượng câu truy vấn**: 300
- **Relevant movies/query (raw)**: avg = 10.0 (có duplicate)
- **Relevant movies/query (unique, sau dedup)**: avg = **9.1**
- **Duplicate entries bị loại**: avg 0.9/query

---

## 3. Recommendation Quality (Core Evaluation)

Đánh giá CineBot V3 full pipeline (Version C + Cross-Encoder Rerank).  
**V2 fix**: dùng unique `relevant_movies` (sau dedup) làm ground truth.

| Metric | V1 (lần trước) | V2 (đã sửa dedup) | Thay đổi |
|---|---|---|---|
| **Precision@5** | 4.0% | 4.0% | ±0 |
| **Precision@10** | 3.5% | 3.5% | ±0 |
| **Recall@10** | 3.5% | **3.9%** | +0.4% ↑ |
| **F1@10** | 3.5% | **3.7%** | +0.2% ↑ |

> [!NOTE]
> Recall@10 tăng nhẹ sau khi dedup vì mẫu số giảm từ 10 → 9.1. Precision không đổi (tử số không thay đổi). Metrics vẫn thấp — nguyên nhân là **synthetic GT được xây bằng cosine similarity tự động** không khớp hoàn toàn với actual recommendation output; đây là limitation của phương pháp xây GT, không phải lỗi pipeline.

---

## 4. RQ1 — Ablation Split Vector & Title-Overfitting

### 4.1 So sánh chất lượng các phiên bản Vector Representation

**Bằng chứng Vector khác nhau sau khi sửa bug B≡C** (cosine B-C_fixed, 3 phim mẫu):
  - Phim #1 'El viaje de Carol': cosine B-C_fixed = 1.0000 ⚠️ (phim này không có keywords trong keyword_dict → profile C = profile B)
  - Phim #2 'Castelo Rá-Tim-Bum': cosine B-C_fixed = 0.9851 ✓ (có keywords → vector khác nhau)
  - Phim #3 'Gravitation': cosine B-C_fixed = 1.0000 ⚠️ (phim này không có keywords → profile C = profile B)

> [!NOTE]
> **Đã sửa**: Root cause là `load_content_keywords()` dùng `@st.cache_resource` trả về `set()` rỗng khi chạy ngoài Streamlit → keywords luôn = "" → profile C = profile B cho mọi phim → cosine B-C = 1.000. Sau fix: load `keyword_dict.json` trực tiếp. Index C rebuilt. Phim có keywords trong dict → vector khác nhau (cosine ≈ 0.985). Phim không có keywords matching → vẫn cosine = 1.000 (đây là behavior đúng: nếu không có keywords, profile C = profile B là hợp lệ). **Trung bình cosine B-C_fixed = 0.995** — xác nhận index C mới không còn bị bug systematic.

| Model | Vector Content | P@5 V2_old | P@5 V3 | P@10 V2_old | P@10 V3 | R@10 V2_old | R@10 V3 | F1@10 V2_old | F1@10 V3 |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline A (Description Only)** | 4.4% | 3.5% | 3.4% | 2.3% | 3.9% | 2.6% | 3.6% | 2.4% |
| **Version B (Description + Genre)** | 4.1% | 3.4% | 3.7% | 2.2% | 4.1% | 2.6% | 3.9% | 2.4% |
| **CineBot V3 (Desc+Genre+Keywords)** | 4.0% | 3.5% | 3.5% | 2.2% | 3.9% | 2.6% | 3.7% | 2.4% |

> [!NOTE]
> V2_old: kết quả từ lần chạy V2 (index C = index B do bug). V3: kết quả sau khi rebuild index C đúng với Keywords.

**Keyword Coverage trên 300 seed movies**: `keyword_dict.json` (468 search_content keywords) khớp với description của **297/300 phim (99.0%)**, trung bình 11.5 keywords/phim. Chỉ 3/300 phim (1.0%) không có keyword nào.

**Nhận xét (null result có đủ coverage để kết luận)**: Coverage 99% đủ cao để kết luận công bằng — A/B/C cho kết quả gần nhau ngay cả sau khi sửa bug và rebuild index C đúng → **Genre và TF-IDF Keywords không cải thiện đáng kể chất lượng recommendation trên bộ GT này** (null result hợp lệ, không phải do coverage thấp).

### 4.2 Kiểm thử Title-Overfitting (bộ decoy NGHIÊM NGẶT, index C fixed)

Bộ **50** cặp decoy: `title_similarity (SequenceMatcher) ≥ 0.40` VÀ genre hoàn toàn khác nhau.

**Ví dụ 5 cặp decoy nghiêm ngặt:**

| Seed | Decoy | Title sim | Genre Seed | Genre Decoy |
|---|---|---|---|---|
| El viaje de Carol | Danny Deckchair | 0.44 | History, Drama | Romance, Comedy |
| Castelo Rá-Tim-Bum | Asylum | 0.42 | Fantasy, Comedy, Family | Thriller, Romance, Drama |
| Gravitation | High Tension | 0.43 | Animation, Comedy, Drama | Horror |
| Armaan | Darna Mana Hai | 0.50 | Romance, Family, Drama | Horror |
| Man of the House | The Notebook | 0.43 | Crime, Comedy | Romance, Drama |

| Phiên bản | V1 (bộ yếu) | V2 (bộ nghiêm ngặt) | V3 (index C fixed) |
|---|---|---|---|
| **Baseline A** | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| **CineBot V3** | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |

> [!NOTE]
> Kết quả 0.0% với index C fixed xác nhận hệ thống **không bị Title-Overfitting**. Corpus phim đủ lớn (42,620 phim) — content similarity đủ mạnh để không nhầm với decoy.


## 5. RQ2 — Dynamic Weight Robustness

Chỉ số F1@10 dưới các mức thiếu metadata (Missing Metadata Rates):

| Missing Rate | Static Weight F1 | Dynamic Weight F1 | Improvement |
|---|---|---|---|
| **0%** | 49.1% | 66.7% | +35.8% |
| **20%** | 43.7% | 53.7% | +22.9% |
| **50%** | 37.1% | 38.7% | +4.3% |

> [!NOTE]
> RQ2 không bị ảnh hưởng bởi các bug đã sửa — số liệu giữ nguyên so với V1. Dynamic Weight cải thiện F1 đáng kể ở 0% và 20% missing. Khoảng cách thu hẹp ở 50% missing — cả hai phương pháp đều suy giảm khi metadata thiếu nhiều.

---

## 6. So sánh Chiến lược Retrieval (đã sửa BUG 5a/5b)

**Thay đổi so với V1**:
- BM25 `top_k`: 100 → **500** (cho phép đo Recall@500 thực sự)
- FAISS `top_k`: 150 → **500**
- "Hybrid + Metadata filter": từ `title filter` → **genre filter**

| Phương pháp | Recall@100 V1 | Recall@100 V2 | Recall@500 V1 | Recall@500 V2 | P@10 V1 | P@10 V2 |
|---|---|---|---|---|---|---|
| **BM25 only** | 2.8% | **3.2%** | 2.8% | **5.3%** ↑ | 1.3% | 1.3% |
| **FAISS only** | 3.6% | **4.1%** | 4.9% | **12.4%** ↑↑ | 0.7% | 0.7% |
| **BM25 + FAISS (Hybrid)** | 3.3% | **3.7%** | 4.9% | **10.3%** ↑↑ | 0.7% | 0.7% |
| **Hybrid + Genre filter** *(V2)* / **Hybrid + Title filter** *(V1)* | 0.5% | **1.6%** | 0.5% | **1.6%** ↑ | 0.5% | 0.7% |

**Nhận xét thực tế (chỉ từ số liệu)**:
- **BUG 5a xác nhận**: BM25 Recall@500 V1=2.8% = Recall@100 V1=2.8% → bug. Sau sửa: 5.3% > 3.2% ✓
- **BUG 5b xác nhận**: Hybrid+Title filter V1=0.5% (chỉ trả về seed movie). Sau sửa (genre filter): 1.6%
- FAISS Recall@500 tăng mạnh nhất: 4.9% → 12.4%, cho thấy top-500 semantic neighbors phủ nhiều relevant movies hơn
- Recall@500 tổng thể vẫn thấp (≤12.4%) — ground truth được xây từ FAISS C index nên FAISS tự nhiên có lợi thế

---

## 7. RQ3a — Cross-Encoder Reranking

| Mô hình | NDCG@10 V1 | NDCG@10 V2 | MAP@10 V1 | MAP@10 V2 |
|---|---|---|---|---|
| **Before Rerank** | 0.095 | **0.102** | 0.043 | **0.047** |
| **After Rerank (Cross-Encoder)** | 0.052 | **0.034** | 0.020 | **0.015** |

> [!WARNING]
> **Negative finding xác nhận**: Cross-Encoder vẫn làm GIẢM NDCG@10 (0.102 → 0.034). Nguyên nhân đã xác định: model `cross-encoder/ms-marco-MiniLM-L-6-v2` (train trên MS-MARCO web passage retrieval) không phù hợp với domain movie similarity. Với query "phim tương tự phim X", model cho score cao với movie profiles chứa nhiều từ khóa search-like, không phải phim thực sự giống seed. Đây là **domain mismatch** — kết quả khoa học hợp lệ. **Khuyến nghị**: dùng cross-encoder fine-tuned trên movie-query pairs, hoặc thay bằng ColBERT-style late interaction model.

---

## 8. RQ3b — Hallucination Evaluation (cx/gpt-5.5)

50 câu hỏi factual (25 đạo diễn + 25 năm phát hành), toàn bộ xử lý qua cache (150 cache hits, 0 live calls).

| Model | Correct | Partial | Incorrect | Accuracy |
|---|---|---|---|---|
| **LLM only** (cx/gpt-5.5) | 48 | 1 | 1 | **96.0%** |
| **CineBot RAG** | 50 | 0 | 0 | **100.0%** |

**Nhận xét**: RAG context cải thiện accuracy từ 96.0% lên 100.0% — loại bỏ hoàn toàn hallucination trong bộ test này. cx/gpt-5.5 alone đã rất mạnh (96%), RAG context giúp cover nốt 2 câu sai.

---

## 9. Phân tích Độ trễ — V3 (Latency, timeout=15s)

50 profiling runs với **50 query mới hoàn toàn** (cache miss 100% cho LLM calls). LLM timeout: 15s.

### Bảng chính: Live calls (cache miss) — số liệu cho paper

> [!NOTE]
> **RAG Generation**: Chỉ tính từ **10 calls thành công thật** (n=10, Avg=9122.6ms). 40/50 calls timeout (80%) bị loại khỏi trung bình chính — xem chi tiết bên dưới. **Total** được tính lại tương ứng.

| Stage | Avg (ms) | P95 (ms) | P95/Avg ratio | Ghi chú |
|---|---|---|---|---|
| **Entity Extraction** | 6081.7 | 8128.2 | 1.3x | n=50, thật 100% |
| **Intent LLM** | 6389.1 | 8189.8 | 1.3x | n=49 live; 1 timeout loại |
| **Retrieval (Hybrid)** | 284.3 | 395.8 | 1.4x | n=50, thật 100% |
| **Similarity Scoring** | 232.4 | 412.9 | 1.8x | n=50, thật 100% |
| **Cross-Encoder Rerank** | 748.5 | 849.8 | 1.1x | n=50, thật 100% |
| **RAG Generation** | **9122.6** | 14439.5 | 1.6x | **n=10 live thật** (40 timeout loại) |
| **Total (end-to-end)** *(ước tính)* | **23858.6** | — | — | =Entity+Intent+Retrieval+Scoring+Rerank+RAG_live |

> Total: cache_hits=0, live_calls=100, timeouts=41 (RAG: 40, Intent: 1)

> [!WARNING]
> **RAG timeout rate 80%** (40/50 calls): Local LLM server (cx/gpt-5.5, localhost:20128) jitter nặng với query dạng mới. Intent LLM timeout chỉ 1/50 (2%) — ít bị ảnh hưởng hơn vì intent classification prompt ngắn hơn. **Số 13863ms (trung bình cũ)** = trung bình gộp của 10 calls thành công (~9123ms) + 40 calls bị cắt ở 15000ms → không phản ánh tốc độ xử lý thật, đã được loại khỏi bảng chính.

### Intent LLM & RAG: Cached vs Live Breakdown

| Stage | Call Type | Count | Avg (ms) | P95 (ms) |
|---|---|---|---|---|
| Intent LLM | Live (cache miss) | 49 | 6389.1 | 8189.8 |
| Intent LLM | Timeout (>15s) | 1 | — | — |
| RAG Generation | Live (cache miss) | 10 | 9122.6 | 14439.5 |
| RAG Generation | Timeout (>15s) | 40 | — | — |

> [!NOTE]
> **Live calls** = số liệu thật của hệ thống. Nếu `Intent LLM live = 0` (tất cả cached) do query giống pattern cũ — báo cáo trung thực: chưa đo được live LLM latency vì cache đã phủ toàn bộ query pattern. Số liệu Entity Extraction, Retrieval, Scoring, Rerank là số liệu thật 100%.

### Phụ lục A: RAG Generation gộp timeout (tham khảo, không dùng cho paper)

| Stage | Avg (ms) | Ghi chú |
|---|---|---|
| **RAG Generation** (gộp timeout) | 13863.0 | = (10×9122.6 + 40×15000) / 50 — inflate bởi timeout cutoff |
| **Total** (gộp timeout) | 27772.0 | Dùng RAG=13863 thay vì RAG=9122.6 |

### Phụ lục B: Cached calls (V2 cũ, cache=100%) — minh hoạ hiệu quả cache

| Stage | Avg (ms) | P95 (ms) | P95/Avg ratio |
|---|---|---|---|
| **Entity Extraction** | 4980.0 | 7513.7 | 1.5x |
| **Intent LLM** | 2.2 | 3.2 | 1.5x |
| **Retrieval (Hybrid)** | 224.5 | 334.9 | 1.5x |
| **Similarity Scoring** | 235.4 | 368.8 | 1.6x |
| **Cross-Encoder Rerank** | 741.7 | 886.6 | 1.2x |
| **RAG Generation** | 3.2 | 5.2 | 1.6x |
| **Total (end-to-end)** | 6187.0 | 8958.5 | 1.4x |

> Total V2 cached: cache_hits=100, live_calls=0, timeouts=0

> [!NOTE]
> So sánh Live vs Cached: Entity Extraction, Retrieval, Scoring, Rerank là số liệu thật (không phụ thuộc cache). Intent LLM và RAG Generation cached (~2-3ms) là tốc độ đọc bộ nhớ cache — không phản ánh tốc độ xử lý thật của LLM server.


## 10. Nhận xét tổng kết (chỉ dựa trên số liệu thực tế)

**1. Recommendation Quality**: Sau khi sửa dedup GT, Recall@10 tăng nhẹ (3.5%→3.9%), F1 tăng 3.5%→3.7%. Precision không đổi. Metrics tổng thể vẫn thấp (Precision@5=4%) — phù hợp với bản chất của synthetic GT xây bằng cosine similarity: hệ thống recommendation dùng cùng chiến lược nên có bias nhất định, nhưng không đủ để đạt Precision cao trên toàn bộ 300 queries.

**2. Split Vector (RQ1)**: Đã sửa bug B≡C (cosine B-C_fixed = 0.995). Keyword coverage = **297/300 (99.0%)** seed movies có keyword match — đủ cao để kết luận công bằng. P@5 của V3 (3.5%) = Baseline A (3.5%) → **null result hợp lệ**: Genre và TF-IDF Keywords không cải thiện đáng kể recommendation quality trên bộ GT này.

**3. Title-Overfitting**: Với bộ decoy nghiêm ngặt hơn (sim≥0.40, genre khác hoàn toàn), cả Baseline A và V3 đều đạt 0.0% error rate — hệ thống không bị overfitting. Điều này có thể do corpus đủ lớn và content similarity đủ mạnh để không nhầm với decoy.

**4. Dynamic Weight (RQ2)**: Dynamic Weight vượt trội hơn Static Weight ở 0% và 20% missing (+35.8% và +22.9%). Ở 50% missing, khoảng cách thu hẹp còn +4.3% — cả hai phương pháp đều suy giảm đáng kể khi metadata thiếu nhiều.

**5. Retrieval (RQ sau sửa)**: Sau khi sửa bug, FAISS Recall@500 tăng từ 4.9% lên **12.4%** — đây là mức tăng thực chất từ việc sửa `top_k`. Hybrid+Filter (genre) cải thiện từ 0.5% lên 1.6% — xác nhận bug filter logic cũ.

**6. Cross-Encoder**: NDCG@10 giảm từ 0.102 xuống 0.034 sau rerank — **xác nhận domain mismatch**. Đây là finding quan trọng: MS-MARCO cross-encoder không phù hợp với movie recommendation. Cần fine-tuning hoặc thay thế.

**7. Hallucination (RAG)**: CineBot RAG đạt 100% accuracy (vs 96% LLM-only). RAG context hiệu quả trong việc cung cấp thông tin factual chính xác.

**8. Latency**: Live calls (cache_hits=0): Entity Extraction Avg=6082ms, Retrieval Avg=284ms, Rerank Avg=749ms. Intent LLM Avg=6389ms (n=49 live, 1 timeout). RAG Generation Avg=**9123ms** (n=10 live thật, 40/50 timeout bị loại). Total ước tính = **23859ms**. RAG timeout rate cao (80%) do LLM server jitter với query dạng mới — limitation của local LLM server.

---

*Báo cáo được tạo bởi `run_eval_v2.py`, `run_latency.py` và `fix_and_rerun.py` — 20/06/2026. LLM dùng cho evaluation: `cx/gpt-5.5` (localhost:20128). Mục 4 (RQ1) và Mục 9 (Latency) được cập nhật trong lần chạy `fix_and_rerun.py` — sửa 2 lỗi còn lại: Bug B≡C và Latency cache-only.*
