# CINEBOT V3 — AUDIT REPORT (Non-LLM Modules)

> **Pham vi**: Chi cac module thuan thuat toan/rule-based, khong ton LLM API call.
> **Rang buoc**: Khong sua DEFAULT_WEIGHTS, giu nguyen signature public function.

---

## Tom tat thay doi da thuc hien

| # | File | Loai | Mo ta |
|---|------|------|--------|
| 1 | retrieval/multistage_retriever.py | BUG FIX | Doi thu tu merge candidate: metadata_candidates merge truoc graph/faiss/bm25 |
| 2 | feedback_logger.py | BUG FIX | Them threading.Lock bao ve ghi JSONL, tranh race condition multi-session |


---

## 1. retrieval/multistage_retriever.py

### Bug P0 (DA FIX)

| Van de | Muc do | De xuat fix | Can LLM |
|--------|--------|-------------|----------|
| metadata_candidates merge cuoi -> bi loai khi graph+faiss+bm25 > cap 500 | Cao | DA FIX: doi thu tu merge | Khong |

**Test verify**: chatbot/tests/test_multistage_retriever_merge_order.py - 6/6 PASS.

### Danh gia Hieu nang: Candidate Embedding Cache

embedder_model.encode() tai Stage 3 ma hoa lai toan bo candidate moi lan goi.
Khuyen nghi: Implement lazy LRU cache maxsize=5000 theo key (movie_link, version).

---

## 2. retrieval/retrieval_router.py

| Van de | Muc do | De xuat fix | Can LLM |
|--------|--------|-------------|----------|
| trace['stage0_graph'] KeyError tiem an | Trung | Them setdefault | Khong |
| filters dict bi mutate truc tiep (side effect) | Trung | Dung filters.copy() | Khong |

---

## 3. entity_extractor.py

| Van de | Muc do | De xuat fix | Can LLM |
|--------|--------|-------------|----------|
| get_fuzzy_candidates() co @st.cache_data | Thap | Khong can fix | Khong |
| is_refine_query() de false positive | Trung | Them context check | Khong |

---

## 4. retrieval/bm25_retriever.py

| Van de | Muc do | De xuat fix | Can LLM |
|--------|--------|-------------|----------|
| TRANSLATION_MAP co entry trung key (sieu anh hung x2, vu tru x2) | Trung | Xoa bo entry trung | Khong |

---

## 5. similarity/weighted_similarity.py

Kiem tra call sites doc lap:
- run_eval.py va run_eval_v2.py goi truc tiep cac ham similarity nhung co guard rieng
- KHONG co sai lech score trong evaluation scripts.

| Van de | Muc do | De xuat fix | Can LLM |
|--------|--------|-------------|----------|
| compute_genre_similarity([],[]) tra 1.0 - unreachable trong prod nhung confusing | Thap | Them docstring | Khong |

---

## 6. graph/graph_query.py

| Van de | Muc do | De xuat fix | Can LLM |
|--------|--------|-------------|----------|
| NaN rating: float('nan') or 0.0 tra nan -> sort khong on dinh | Trung | them check: 0.0 if (r!=r) else r | Khong |

---

## 7. tools.py

| Van de | Muc do | De xuat fix | Can LLM |
|--------|--------|-------------|----------|
| country_aliases load moi lan goi | Trung | Dung lru_cache | Khong |
| semantic_search_tool() tra df.copy() khi fail | Trung | Doi thanh DataFrame() | Khong |

---

## 8. feedback_logger.py (DA FIX)

| Van de | Muc do | De xuat fix | Can LLM |
|--------|--------|-------------|----------|
| open(a) khong atomic tren Windows - race condition | Cao | DA FIX: threading.Lock | Khong |
| Multi-process: threading.Lock khong du | Trung | Dung filelock khi scale | Khong |

---

## Tong hop

### Da fix (2):
1. multistage_retriever.py - Candidate merge order
2. feedback_logger.py - Thread-safety

### Nen fix sprint toi (6):
3. retrieval_router.py - trace KeyError tiem an
4. retrieval_router.py - filters dict mutation
5. bm25_retriever.py - Duplicate TRANSLATION_MAP entries
6. graph_query.py - NaN rating sort instability
7. tools.py - semantic_search_tool inconsistent return
8. tools.py - country_aliases cache missing

### Thap/document only (2):
9. weighted_similarity.py - 1.0 vs 0.0 empty convention
10. entity_extractor.py - is_refine_query false positive
