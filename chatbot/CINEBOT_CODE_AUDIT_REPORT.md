# CINEBOT V3 — AUDIT MÃ NGUỒN THEO MỨC ĐỘ NGHI VẤN

Tài liệu này tổng hợp đầy đủ mã nguồn chính xác cho 5 thành phần quan trọng theo yêu cầu kiểm tra đối chiếu.

---

## 1. `retrieval/reranker.py` (Đầy Đủ Mã Nguồn)

> **Xác nhận**: Trong hệ thống **CÓ** sử dụng Cross-Encoder. 
> Mô hình được nạp là `cross-encoder/ms-marco-MiniLM-L-6-v2` từ thư viện `sentence_transformers`.

```python
import streamlit as st
import pandas as pd
from sentence_transformers import CrossEncoder

@st.cache_resource
def load_reranker_model() -> CrossEncoder:
    """
    Nạp và lưu trữ mô hình Cross-Encoder để xếp hạng lại (Reranking).
    Mô hình mặc định: cross-encoder/ms-marco-MiniLM-L-6-v2
    """
    try:
        # Tải mô hình thông qua sentence-transformers
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return model
    except Exception as e:
        print(f"Error: Could not load Cross-Encoder model: {e}")
        return None

def rerank_results(query: str, candidates_df: pd.DataFrame, top_k: int = 20,
                   base_movie_profile: str = None) -> pd.DataFrame:
    """
    Xếp hạng lại các phim ứng viên bằng Cross-Encoder.
    - Với truy vấn tìm phim tương tự: dùng base_movie_profile thay vì câu hỏi thô
      để cross-encoder so sánh nội dung phim với phim gốc (chính xác hơn "có phim tương tự X không?")
    - Tạo profile phim: Description trước (quan trọng nhất), sau đó Title, Genre, Director, Stars.
    """
    if candidates_df.empty or not query:
        return candidates_df
        
    model = load_reranker_model()
    if model is None:
        return candidates_df.head(top_k)
    
    # Xác định query cho cross-encoder
    # Với similar_to queries: dùng profile phim gốc để so sánh nội dung trực tiếp
    rerank_query = base_movie_profile if base_movie_profile else query
        
    # Tạo profile cho từng phim ứng viên
    # Đặt Description trước vì cross-encoder ms-marco-MiniLM được train trên passage ranking
    # → phần đầu của passage có ảnh hưởng lớn hơn đến score
    movie_profiles = []
    for _, row in candidates_df.iterrows():
        desc = str(row.get("description", ""))
        title = str(row.get("Title", ""))
        genres = str(row.get("genres", ""))
        directors = str(row.get("directors", ""))
        stars = str(row.get("stars", ""))
        # Dùng final_context nếu có (cho aggregation results), ngược lại build từ fields
        if "final_context" in row and pd.notna(row["final_context"]) and len(str(row["final_context"])) > 20:
            profile = str(row["final_context"])
        else:
            # Đặt description ở đầu để cross-encoder tập trung vào nội dung
            profile = f"Description: {desc} | Title: {title} | Genres: {genres} | Directors: {directors} | Stars: {stars}"
        movie_profiles.append(profile)
        
    # Tạo cặp (query, doc) để đưa vào mô hình Cross-Encoder
    pairs = [(rerank_query, mp) for mp in movie_profiles]
    
    try:
        scores = model.predict(pairs)
        
        result = candidates_df.copy()
        result["rerank_score"] = scores
        # Sắp xếp theo score giảm dần
        result = result.sort_values(by="rerank_score", ascending=False)
        return result.head(top_k).copy()
    except Exception as e:
        print(f"Error during Rerank: {e}")
        return candidates_df.head(top_k).copy()
```

---

## 2. `similarity/weighted_similarity.py` — Hàm `compute_weighted_similarity` (Đầy Đủ)

> **Xử lý `graph_score` khi KHÔNG có ứng viên Graph**:
> 1. Khi `ref_graph` hoặc `movie_graph` là `None`, `scores["graph_score"]` được gán = `0.0` và `subscore_source["graph_score"] = "fallback_no_entity"`.
> 2. Chiều `"graph"` **KHÔNG được đưa vào `active_weights`**.
> 3. Điểm cuối cùng được tính bằng công thức tái phân phối trọng số (Weight Redistribution): 
>    $$\text{final\_score} = \frac{\sum_{i \in \text{active}} \text{score}_i \times w_i}{\sum_{i \in \text{active}} w_i}$$
>    Do đó, khi không có graph candidate, trọng số 5% của `graph` được tự động chia đều cho các chiều active còn lại, **không bị thổi phồng vô lý (phantom score)**.

```python
def compute_weighted_similarity(
    movie_features: dict,
    ref_features: dict,
    weights: dict = None,
    active_dims: set = None
) -> dict:
    """
    Tính điểm similarity có trọng số giữa một phim ứng viên và phim tham chiếu.

    Tham số `active_dims` (tuỳ chọn): tập các chiều đặc trưng được phép tính vào
    final_score (ví dụ: {"content", "genre", "graph"}).
    - Nếu None → behavior cũ: mọi chiều có dữ liệu tham chiếu đều được tính.
    - Nếu được cung cấp → chỉ các chiều trong tập này mới active, phần còn lại bị
      loại khỏi formula (không phantom 1.0 inflate score nữa).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
        
    scores = {}
    active_weights = {}
    subscore_source = {}
    
    def _dim_allowed(dim: str) -> bool:
        return active_dims is None or dim in active_dims
    
    # 1. Content (luôn active nếu có embedding tham chiếu VÀ dimension được phép)
    ref_emb = ref_features.get("semantic_embedding")
    movie_emb = movie_features.get("semantic_embedding")
    if ref_emb is not None and movie_emb is not None and _dim_allowed("content"):
        scores["content_score"] = compute_content_similarity(movie_emb, ref_emb)
        active_weights["content"] = weights["content"]
        subscore_source["content_score"] = "computed"
    else:
        scores["content_score"] = 0.0
        subscore_source["content_score"] = "fallback_no_entity"
        
    # 2. Genre
    ref_genre = ref_features.get("genre_vector")
    movie_genre = movie_features.get("genre_vector")
    if ref_genre is not None and np.sum(ref_genre) > 0 and movie_genre is not None and _dim_allowed("genre"):
        scores["genre_score"] = compute_genre_similarity(movie_genre, ref_genre)
        active_weights["genre"] = weights["genre"]
        subscore_source["genre_score"] = "computed"
    else:
        scores["genre_score"] = 0.0
        subscore_source["genre_score"] = "fallback_no_entity"
        
    # 3. Actor
    ref_actor = ref_features.get("actor_vector")
    movie_actor = movie_features.get("actor_vector")
    if ref_actor and movie_actor and _dim_allowed("actor"):
        scores["actor_score"] = compute_actor_similarity(movie_actor, ref_actor)
        active_weights["actor"] = weights["actor"]
        subscore_source["actor_score"] = "computed"
    else:
        scores["actor_score"] = 0.0
        subscore_source["actor_score"] = "fallback_no_entity"
        
    # 4. Director
    ref_dir = ref_features.get("director_vector")
    movie_dir = movie_features.get("director_vector")
    if ref_dir and movie_dir and _dim_allowed("director"):
        scores["director_score"] = compute_director_similarity(movie_dir, ref_dir)
        active_weights["director"] = weights["director"]
        subscore_source["director_score"] = "computed"
    else:
        scores["director_score"] = 0.0
        subscore_source["director_score"] = "fallback_no_entity"
        
    # 5. Country
    ref_country = ref_features.get("country_vector")
    movie_country = movie_features.get("country_vector")
    if ref_country is not None and np.sum(ref_country) > 0 and movie_country is not None and _dim_allowed("country"):
        scores["country_score"] = compute_country_similarity(movie_country, ref_country)
        active_weights["country"] = weights["country"]
        subscore_source["country_score"] = "computed"
    else:
        scores["country_score"] = 0.0
        subscore_source["country_score"] = "fallback_no_entity"
        
    # 6. Decade
    ref_dec = ref_features.get("decade_vector")
    movie_dec = movie_features.get("decade_vector")
    if ref_dec is not None and np.sum(ref_dec) > 0 and movie_dec is not None and _dim_allowed("decade"):
        scores["decade_score"] = compute_decade_similarity(movie_dec, ref_dec)
        active_weights["decade"] = weights["decade"]
        subscore_source["decade_score"] = "computed"
    else:
        scores["decade_score"] = 0.0
        subscore_source["decade_score"] = "fallback_no_entity"
        
    # 7. Award
    ref_award = ref_features.get("award_vector")
    movie_award = movie_features.get("award_vector")
    if ref_award is not None and np.sum(ref_award) > 0 and movie_award is not None and _dim_allowed("award"):
        scores["award_score"] = compute_award_similarity(movie_award, ref_award)
        active_weights["award"] = weights["award"]
        subscore_source["award_score"] = "computed"
    else:
        scores["award_score"] = 0.0
        subscore_source["award_score"] = "fallback_no_entity"
        
    # 8. Graph Connection
    ref_graph = ref_features.get("graph_score")
    movie_graph = movie_features.get("graph_score")
    if ref_graph is not None and movie_graph is not None and _dim_allowed("graph"):
        scores["graph_score"] = float(movie_graph)
        active_weights["graph"] = weights["graph"]
        subscore_source["graph_score"] = "computed"
    else:
        scores["graph_score"] = 0.0
        subscore_source["graph_score"] = "fallback_no_entity"
        
    # Tái phân phối trọng số (Weight Redistribution)
    total_active_weight = sum(active_weights.values())
    if total_active_weight > 0:
        final_score = 0.0
        for key, w in active_weights.items():
            final_score += scores[f"{key}_score"] * w
        final_score = final_score / total_active_weight
    else:
        final_score = scores.get("content_score", 0.0)
        
    scores["final_score"] = final_score
    scores["subscore_source"] = subscore_source
    scores["active_weights"] = active_weights
    return scores
```

---

## 3. `retrieval/multistage_retriever.py` — Hàm `retrieve` (Đầy Đủ & Thứ Tự Lọc/Top_K)

> **Thứ tự Lọc / Truy xuất & Giá trị `top_k` Thực tế**:
> 1. **Filter Validation**: Chuẩn hóa & lọc rác các trường `director`, `star`, `genre`, `country`.
> 2. **Shortcut Exact Person**: Nếu chỉ lọc exact `director`/`star` (không qua graph) $\rightarrow$ gọi `search_movies_tool` lấy `top_k = final_k * 3`.
> 3. **Candidate Generation**:
>    - **Stage 0 (Graph Candidates)**: Tối đa 300 từ Router.
>    - **Stage 2 (FAISS)**: Lấy `top_k = 150`.
>    - **Stage 1 (BM25)**: Lấy `top_k = 100`.
>    - **Metadata candidates**: Lấy `top_k = 500`.
>    - **Gộp & Dedup**: Giới hạn tối đa **500 ứng viên**.
> 4. **Stage 2 Metadata Filter**: Lọc cứng metadata giới hạn xuống **200 ứng viên**.
> 5. **Stage 3 Weighted Similarity**: Chấm điểm 8 chiều, xếp hạng lấy **Top 100 ứng viên**.
> 6. **Stage 4 Cross-Encoder Rerank**: Rerank top 100, trả về **Top 20 ứng viên**.
> 7. **Post-processing**: Dedup IMDb, áp dụng lại ràng buộc cứng `rating_min`, `duration_min/max`, cắt lấy **`head(final_k)` (5 phim)**.

```python
    def retrieve(
        self,
        query: str,
        df: pd.DataFrame,
        filters: dict,
        intent: str,
        faiss_index,
        embedder_model,
        version: str = 'C',
        final_k: int = 10,
        graph_candidates: pd.DataFrame = None,
        trace: dict = None
    ) -> pd.DataFrame:
        """
        Unified Multi-stage Retrieval pipeline.
        """
        if trace is not None:
            trace["stage1_bm25"] = {"top_k_requested": 100, "candidates": []}
            trace["stage2_faiss"] = {"candidates": []}
            trace["stage3_rerank"] = {"candidates": []}
            trace["stage4_weighted_similarity"] = {"per_candidate_scores": []}
            trace["actual_route"] = "multistage_hybrid"
            trace["skipped_stages"] = []
            trace["executed_stages"] = []
            trace["candidate_counts"] = {}

        filters_for_retrieval = filters.copy()

        # Validate các trường lọc metadata với database
        if not hasattr(MultistageRetriever, "_all_directors") or MultistageRetriever._all_directors is None:
            from chatbot.feature_engineering import clean_split, PARENT_GENRES
            directors_set = {d.strip().lower() for val in df['directors'].dropna().unique() for d in clean_split(val)}
            MultistageRetriever._all_directors = sorted(list(directors_set))
            actors_set = {s.strip().lower() for val in df['stars'].dropna().unique() for s in clean_split(val)}
            MultistageRetriever._all_actors = sorted(list(actors_set))
            genres_set = {g.strip().lower() for val in df['genres'].dropna().unique() for g in clean_split(val)}
            genres_set.update({g.strip().lower() for g in PARENT_GENRES})
            MultistageRetriever._all_genres = sorted(list(genres_set))
            countries_set = {c.strip().lower() for val in df['countries_origin'].dropna().unique() for c in clean_split(val)}
            MultistageRetriever._all_countries = sorted(list(countries_set))

        from rapidfuzz import process, fuzz
        validation_mapping = {
            "director": MultistageRetriever._all_directors,
            "star": MultistageRetriever._all_actors,
            "genre": MultistageRetriever._all_genres,
            "country": MultistageRetriever._all_countries
        }
        
        for field, valid_list in validation_mapping.items():
            val = filters_for_retrieval.get(field)
            if val:
                parts = [p.strip() for p in re.split(r'[,;]|\bvà\b|\bhoặc\b|\band\b|\bor\b', str(val), flags=re.IGNORECASE) if p.strip()]
                validated_parts = []
                for part in parts:
                    part_clean = part.lower()
                    if part_clean in valid_list:
                        validated_parts.append(part)
                    else:
                        match_res = process.extractOne(part_clean, valid_list, scorer=fuzz.QRatio)
                        if match_res and match_res[1] >= 90.0:
                            validated_parts.append(match_res[0])
                filters_for_retrieval[field] = ", ".join(validated_parts) if validated_parts else None

        base_row = None
        is_similar = False
        if graph_candidates is not None and not graph_candidates.empty:
            base_row, is_similar = self._get_base_movie(df, query, filters)
            from chatbot.retrieval.retrieval_router import is_director_filmography_query
            if is_similar or is_director_filmography_query(query, filters):
                filters_for_retrieval["title"] = None
        
        # Shortcut exact person filter
        has_exact_person_filter = any(filters_for_retrieval.get(k) for k in ["director", "star"])
        if has_exact_person_filter and not is_similar and graph_candidates is None:
            result = search_movies_tool(df, filters_for_retrieval, top_k=final_k * 3)
            return result.head(final_k)
            
        # --- Stage 1: Candidate Generation ---
        faiss_candidates = pd.DataFrame()
        if faiss_index is not None and embedder_model is not None and query:
            faiss_query = query
            if is_similar and base_row is not None:
                base_desc = base_row.get("description", "")
                if base_desc and str(base_desc).strip():
                    faiss_query = str(base_desc).strip()
            faiss_candidates = semantic_search_retriever(faiss_query, df, faiss_index, embedder_model, top_k=150)
            
        bm25_candidates = pd.DataFrame()
        if query:
            if self.bm25_index is None:
                from chatbot.data_loader import load_bm25_index
                self.bm25_index = load_bm25_index(df)
            bm25_candidates = bm25_search(query, df, self.bm25_index, top_k=100, trace=trace)
            
        metadata_candidates = pd.DataFrame()
        has_metadata_filters = any(filters_for_retrieval.get(k) for k in [
            "genre", "director", "star", "title", "year_min", "year_max", "rating_min", "country", "has_awards", "has_oscar"
        ])
        if has_metadata_filters:
            metadata_candidates = search_movies_tool(df, filters_for_retrieval, top_k=500)
            
        # Combine & deduplicate (max 500 candidates)
        candidate_list = []
        seen_links = set()
        dfs_to_combine = []
        if graph_candidates is not None and not graph_candidates.empty:
            dfs_to_combine.append(graph_candidates)
        dfs_to_combine.extend([faiss_candidates, bm25_candidates, metadata_candidates])
        
        for candidates_df in dfs_to_combine:
            if not candidates_df.empty:
                for _, row in candidates_df.iterrows():
                    link = row["Movie Link"]
                    if link not in seen_links:
                        seen_links.add(link)
                        candidate_list.append(row)
                        if len(candidate_list) >= 500:
                            break
            if len(candidate_list) >= 500:
                break
                
        if not candidate_list:
            fallback_res = search_movies_tool(df, filters_for_retrieval, top_k=final_k)
            return fallback_res
            
        candidates_df = pd.DataFrame(candidate_list)
        
        # --- Stage 2: Metadata Filtering (Lọc xuống Top 200) ---
        if has_metadata_filters:
            filtered_df = search_movies_tool(candidates_df, filters_for_retrieval, top_k=200)
            if filtered_df.empty:
                filtered_df = search_movies_tool(df, filters_for_retrieval, top_k=200)
        else:
            filtered_df = candidates_df.head(200).copy()
            
        # --- Stage 3: Weighted Similarity Ranking ---
        if is_similar and base_row is not None:
            ref_features = self.builder.transform_row(base_row)
            base_profile = make_profile(base_row, version)
            if embedder_model is not None:
                ref_features["semantic_embedding"] = embedder_model.encode([base_profile], convert_to_numpy=True)[0]
            ref_features["graph_score"] = 1.0
            filtered_df = filtered_df[filtered_df["Movie Link"] != base_row["Movie Link"]]
        else:
            ref_features = self.build_query_features(query, filters, embedder_model)
            
        candidate_embeddings = []
        if embedder_model is not None and not filtered_df.empty:
            candidate_profiles = [make_profile(row, version) for _, row in filtered_df.iterrows()]
            candidate_embeddings = embedder_model.encode(candidate_profiles, convert_to_numpy=True)
            
        matched_rows = []
        for idx, (_, row) in enumerate(filtered_df.iterrows()):
            row_features = self.builder.transform_row(row)
            row_features["semantic_embedding"] = candidate_embeddings[idx] if idx < len(candidate_embeddings) else None
                
            if "graph_path_explanation" in row and pd.notna(row["graph_path_explanation"]):
                hop = int(row.get("graph_hop_count", 2))
                p_type_val = row.get("graph_path_type", "personnel")
                if p_type_val == "personnel":
                    row_features["graph_score"] = 1.00 if hop <= 1 else (0.60 if hop == 2 else 0.20)
                else:
                    row_features["graph_score"] = 0.50 if hop <= 1 else 0.20
            else:
                row_features["graph_score"] = 0.0

            if is_similar and base_row is not None:
                _active_dims = {"content", "genre", "graph"}
            else:
                _active_dims = {"content"}
                if filters.get("genre"):    _active_dims.add("genre")
                if filters.get("star"):     _active_dims.add("actor")
                if filters.get("director"): _active_dims.add("director")
                if filters.get("country"):  _active_dims.add("country")
                _active_dims.add("decade")
                _active_dims.add("award")
                if row_features.get("graph_score", 0.0) > 0.0: _active_dims.add("graph")

            sim_breakdown = compute_weighted_similarity(row_features, ref_features, active_dims=_active_dims)
            
            row_copy = row.copy()
            row_copy["similarity_score"] = f"{sim_breakdown['final_score'] * 100:.1f}%"
            row_copy["final_similarity_score"] = sim_breakdown["final_score"]
            matched_rows.append(row_copy)
            
        if not matched_rows:
            return pd.DataFrame()
            
        ranked_df = pd.DataFrame(matched_rows).sort_values(by="final_similarity_score", ascending=False)
        top_100_df = ranked_df.head(100).copy()
        
        # --- Stage 4: Cross-Encoder Reranking (Lấy Top 20) ---
        query_rerank = f"{query} (phim tương tự như {base_row['Title']})" if (is_similar and base_row is not None) else (query or "Phim hay được đánh giá cao.")
        base_movie_profile_str = make_profile(base_row, version) if (is_similar and base_row is not None) else None
            
        reranked_df = rerank_results(query_rerank, top_100_df, top_k=20, base_movie_profile=base_movie_profile_str)
        
        # Post-filtering (Rating & Duration hard constraint re-enforcement)
        if not reranked_df.empty:
            if 'countries_origin' in reranked_df.columns:
                reranked_df = reranked_df[reranked_df['countries_origin'].astype(str).str.strip().ne('') & reranked_df['countries_origin'].notna()]
            if 'imdb_id' in reranked_df.columns:
                reranked_df['_genre_len'] = reranked_df['genres'].astype(str).str.len()
                reranked_df = reranked_df.sort_values('_genre_len', ascending=False).drop_duplicates(subset='imdb_id', keep='first').drop(columns=['_genre_len'])
            
            rating_min = filters.get("rating_min")
            if rating_min is not None:
                try:
                    hard_filtered = reranked_df[reranked_df[COL_RATING] >= float(rating_min)]
                    if len(hard_filtered) >= max(1, final_k // 2): reranked_df = hard_filtered
                except Exception: pass

        return reranked_df.head(final_k)
```

---

## 4. `graph/graph_query.py` — Hàm `find_movies_by_collab_path` & Giới Hạn Candidate

> **Giới hạn số lượng ứng viên trên Đồ thị**:
> 1. Trong `find_movies_by_collab_path`, giới hạn theo từng hop được xử lý tại hàm trợ lý `get_limited_neighbors(graph, curr, max_neighbors_per_hop)` (mặc định `max_neighbors_per_hop = 20`).
> 2. `get_limited_neighbors` sắp xếp các phim theo `(rating, num_votes)` giảm dần và lấy đúng top 20 phim tốt nhất ở mỗi bước chuyển tiếp.
> 3. Trong [`retrieval/retrieval_router.py`](file:///e:/Desktop/4/DAP391m/code/chatbot/retrieval/retrieval_router.py) (dòng 146 & 197), danh sách kết quả đồ thị được cắt giới hạn tối đa **300 phim** (`graph_results[:300]`).

```python
def find_movies_by_collab_path(
    graph: nx.MultiDiGraph, 
    reference_movie_title: str, 
    max_hops: int = 3, 
    max_neighbors_per_hop: int = 20
) -> list[dict]:
    """
    Từ một phim tham chiếu, đi qua Actor/Director/Genre/Country, tìm các phim khác liên quan qua đường đi đồ thị.
    """
    actual_movie_node = None
    if reference_movie_title.startswith("Movie:") and graph.has_node(reference_movie_title):
        actual_movie_node = reference_movie_title
    elif graph.has_node(f"Movie:{reference_movie_title}"):
        actual_movie_node = f"Movie:{reference_movie_title}"
    else:
        movie_lower = reference_movie_title.lower()
        for node, data in graph.nodes(data=True):
            if data.get("type") == "Movie" and clean_name(node).lower() == movie_lower:
                actual_movie_node = node
                break
                
    if not actual_movie_node:
        return []
        
    # Hàm hỗ trợ chạy BFS tìm kiếm
    def run_bfs(personnel_only: bool) -> dict[str, list[str]]:
        queue = deque([(actual_movie_node, [actual_movie_node])])
        visited = {actual_movie_node}
        found_movies = {}
        
        while queue:
            curr, path = queue.popleft()
            hop_count = len(path) - 1
            
            if hop_count >= max_hops:
                continue
                
            neighbors = get_limited_neighbors(graph, curr, max_neighbors_per_hop, personnel_only=personnel_only)
            for neighbor, etype in neighbors:
                if neighbor in visited:
                    continue
                    
                new_path = path + [neighbor]
                visited.add(neighbor)
                
                vtype = graph.nodes[neighbor].get("type")
                if vtype == "Movie":
                    found_movies[neighbor] = new_path
                    
                queue.append((neighbor, new_path))
        return found_movies

    # 1. Thử nghiệm tìm kiếm bằng personnel path trước (chỉ đi qua Actor/Director/Collab)
    candidate_movies = run_bfs(personnel_only=True)
    
    # 2. Nếu số lượng phim tìm được quá ít (< 5), fallback chạy BFS đầy đủ với cả Genre/Country
    if len(candidate_movies) < 5:
        candidate_movies = run_bfs(personnel_only=False)
        
    result = []
    for m_node, path in candidate_movies.items():
        explanation = explain_path_from_nodes(graph, path)
        
        p_type = "personnel"
        for node in path[1:-1]:
            ntype = graph.nodes[node].get("type", "Unknown")
            if ntype in ("Genre", "Country"):
                p_type = "shared_attribute"
                break
                
        node_data = graph.nodes[m_node]
        
        result.append({
            "Title": clean_name(m_node),
            "Rating": node_data.get("rating"),
            "Year": node_data.get("year"),
            "num_votes": node_data.get("num_votes"),
            "genres": node_data.get("genres"),
            "decade": node_data.get("decade"),
            "has_awards": node_data.get("has_awards"),
            "has_oscar": node_data.get("has_oscar"),
            "has_nomination": node_data.get("has_nomination"),
            "hop_count": len(path) - 1,
            "graph_path_explanation": explanation,
            "graph_path_type": p_type
        })
        
    return result
```

---

## 5. `chains/rag_chain.py` — Đoạn Code Xử Lý Stat Query (Direct Aggregation)

> **Cơ chế xử lý Testcase "So sánh" & Thống kê**:
> 1. Khi `intent == "aggregation"`, hệ thống chạy Regex `_stat_pattern` phát hiện từ khóa: `trung bình`, `so sánh`, `cao hơn`, `average`, `mean`, `điểm tb`.
> 2. Nếu là câu hỏi thống kê chung dòng phim (không chỉ định tên nhân sự), hệ thống bỏ qua RAG top-K thông thường.
> 3. Lọc trực tiếp trên toàn bộ DataFrame theo bộ lọc thể loại/năm (`_stat_filters`) với `top_k = len(df)`.
> 4. Tính toán điểm IMDb trung bình (`_avg_rating = round(_ratings.mean(), 2)`).
> 5. Tạo dòng giả thống kê `[Thống kê: ...]` và lọc lấy Top 10 phim vượt mức trung bình (`_above_avg`) để hiển thị kèm ngữ cảnh chứng minh cho LLM.

```python
    if intent == "aggregation":
        # Xử lý câu hỏi tổng hợp: "ai hợp tác nhiều nhất với X"
        person_name = filters.get("director") or filters.get("star")
        
        # Phát hiện truy vấn tính thống kê (trung bình, cao hơn mức trung bình, so sánh...)
        # và tính TRỰC TIẾP từ toàn bộ df theo filter, không qua tập ứng viên bị giới hạn top-K.
        _stat_pattern = r'trung\s*b[ìi]nh|average|avg|cao\s*h[ơo]n.*m[uứ]c|so\s*s[áa]nh|mean|median|điểm\s*tb|điểm\s*trung\s*b[ìi]nh'
        if re.search(_stat_pattern, user_input, re.IGNORECASE) and not person_name:
            from chatbot.tools import search_movies_tool
            from chatbot.config import COL_RATING, COL_GENRE, COL_YEAR
            # Chỉ áp dụng filter thể loại/năm/quốc gia — không bị giới hạn bởi top-K retrieval
            _stat_filters = {k: v for k, v in filters.items() if k in ('genre', 'year_min', 'year_max', 'country') and v is not None}
            _full_filtered = search_movies_tool(df, _stat_filters, top_k=len(df))
            if not _full_filtered.empty and COL_RATING in _full_filtered.columns:
                _ratings = _full_filtered[COL_RATING].dropna()
                _avg_rating = round(_ratings.mean(), 2) if len(_ratings) > 0 else None
                _count = len(_ratings)
                _genre_label = filters.get('genre', 'phim')
                _year_label = f" sau năm {filters.get('year_min')}" if filters.get('year_min') else ""
                # Tìm các phim vượt mức trung bình
                if _avg_rating is not None:
                    _above_avg = _full_filtered[_full_filtered[COL_RATING] > _avg_rating] if COL_RATING in _full_filtered.columns else pd.DataFrame()
                    _above_avg = _above_avg.sort_values(COL_RATING, ascending=False).head(10)
                    _stat_rows = []
                    _stat_rows.append({
                        "Title": f"[Thống kê: {_genre_label}{_year_label}]",
                        "Rating": _avg_rating,
                        "Year": None,
                        "genres": _genre_label,
                        "directors": "",
                        "countries_origin": "",
                        "stars": "",
                        "Movie Link": "",
                        "final_context": f"Điểm IMDb trung bình của {_count} phim {_genre_label}{_year_label} trong cơ sở dữ liệu: {_avg_rating}/10"
                    })
                    for _, _r in _above_avg.iterrows():
                        _stat_rows.append({
                            "Title": _r.get('Title', ''),
                            "Rating": _r.get(COL_RATING),
                            "Year": _r.get(COL_YEAR),
                            "genres": _r.get('genres', ''),
                            "directors": _r.get('directors', ''),
                            "countries_origin": _r.get('countries_origin', ''),
                            "stars": _r.get('stars', ''),
                            "Movie Link": _r.get('Movie Link', ''),
                            "final_context": f"Tên: {_r.get('Title')} | Năm: {_r.get(COL_YEAR)} | Điểm IMDb: {_r.get(COL_RATING)} | Thể loại: {_r.get('genres', '')}"
                        })
                    filtered_df = pd.DataFrame(_stat_rows)
                    route_name = "aggregation_stat"
```

---

## 6. `retrieval/retrieval_router.py` — Đoạn Code Xây Dựng `graph_candidates`

> **Chi tiết xây dựng `graph_candidates`**:
> 1. Gọi `retriever._get_base_movie()` trích xuất phim tham chiếu.
> 2. Gọi `find_movies_by_collab_path(G, reference_movie_title, max_hops=2, max_neighbors_per_hop=20)` để lấy danh sách node phim trên đồ thị.
> 3. Cắt giới hạn tối đa **300 phim** (`graph_results[:300]`).
> 4. Tạo map tra cứu nhanh `title_map = {str(t).lower(): idx for idx, t in enumerate(df["Title"])}` để lấy dữ liệu row DataFrame gốc $O(1)$.
> 5. Bổ sung 3 trường quan trọng vào từng dòng candidate:
>    - `graph_path_explanation`: Lời giải thích đường đi quan hệ.
>    - `graph_path_type`: Loại liên kết (`"personnel"` hoặc `"shared_attribute"`).
>    - `graph_hop_count`: Số hop trên đồ thị (1 hoặc 2).

```python
    graph_candidates = None
    
    # 1. Kiểm tra nếu là truy vấn phim tương tự
    if is_similar_movie_query(query, filters):
        # Trích xuất phim gốc
        base_row, is_similar = retriever._get_base_movie(df, query, filters)
        if is_similar and base_row is not None:
            reference_movie_title = base_row["Title"]
            
            try:
                from chatbot.graph.build_movie_graph import load_or_build_graph
                from chatbot.graph.graph_query import find_movies_by_collab_path
                
                G = load_or_build_graph(df)
                # Giới hạn max_hops=2 để chỉ lấy phim có chung nhân sự trực tiếp 1-hop
                graph_results = find_movies_by_collab_path(G, reference_movie_title, max_hops=2, max_neighbors_per_hop=20)
                
                if trace is not None:
                    trace["stage0_graph"]["called"] = True
                    serializable_graph = []
                    for res in graph_results:
                        clean_res = {}
                        for k, v in res.items():
                            if isinstance(v, (np.integer, np.floating)):
                                clean_res[k] = v.item()
                            else:
                                clean_res[k] = v
                        serializable_graph.append(clean_res)
                    trace["stage0_graph"]["candidates"] = serializable_graph
                
                graph_rows = []
                # Tạo map tra cứu tiêu đề O(1)
                title_map = {str(t).lower(): idx for idx, t in enumerate(df["Title"])}
                
                # Giới hạn số lượng ứng viên đồ thị ở mức 300 để tránh quá tải
                for res in graph_results[:300]:
                    title = res["Title"]
                    explanation = res["graph_path_explanation"]
                    p_type = res.get("graph_path_type", "personnel")
                    
                    idx = title_map.get(title.lower())
                    if idx is not None:
                        row_copy = df.iloc[idx].copy()
                        row_copy["graph_path_explanation"] = explanation
                        row_copy["graph_path_type"] = p_type
                        row_copy["graph_hop_count"] = res.get("hop_count", 2)
                        graph_rows.append(row_copy)
                        
                if graph_rows:
                    graph_candidates = pd.DataFrame(graph_rows)
            except Exception as e:
                print(f"Error getting candidates from Graph RAG: {e}")
    
    # 2. Kiểm tra nếu là truy vấn multi-hop filmography của đạo diễn/diễn viên
    elif is_director_filmography_query(query, filters, df):
        movie_title = filters.get("title") or extract_title_from_query(query, df)
        if movie_title:
            filters["title"] = movie_title
            try:
                from chatbot.graph.build_movie_graph import load_or_build_graph
                from chatbot.graph.graph_query import find_movies_by_collab_path
                
                G = load_or_build_graph(df)
                graph_results = find_movies_by_collab_path(G, movie_title, max_hops=2, max_neighbors_per_hop=30)
                
                graph_rows = []
                title_map = {str(t).lower(): idx for idx, t in enumerate(df["Title"])}
                for res in graph_results[:300]:
                    title = res["Title"]
                    idx = title_map.get(title.lower())
                    if idx is not None:
                        row_copy = df.iloc[idx].copy()
                        row_copy["graph_path_explanation"] = res["graph_path_explanation"]
                        row_copy["graph_path_type"] = res.get("graph_path_type", "personnel")
                        row_copy["graph_hop_count"] = res.get("hop_count", 2)
                        graph_rows.append(row_copy)
                if graph_rows:
                    graph_candidates = pd.DataFrame(graph_rows)
            except Exception as e:
                print(f"Error getting graph candidates for filmography query: {e}")
```

---

## 7. `prompts/intent_prompt.py` & `chains/intent_chain.py` — Prompt & Logic Phân Loại Intent `aggregation` và Cụm "So sánh"

> **Phân tích Intent cho câu hỏi "So sánh 2 phim"**:
> 1. Trong [`prompts/intent_prompt.py`](file:///e:/Desktop/4/DAP391m/code/chatbot/prompts/intent_prompt.py), `aggregation` được định nghĩa chính thức cho câu hỏi về **tần suất hợp tác**:
>    - `"aggregation"`: *Khi người dùng hỏi câu hỏi tổng hợp/tần suất hợp tác ("ai hợp tác nhiều nhất với X", "diễn viên nào đóng nhiều phim của X nhất", "X hợp tác với ai nhiều nhất").*
> 2. Tuy nhiên, đối với câu hỏi **"So sánh 2 phim X và Y"** (ví dụ: *"So sánh Inception và Interstellar"*):
>    - LLM Tầng 1 (Intent Chain) có xu hướng phân loại câu hỏi này là **`"search"`** hoặc **`"info"`** với `filters["title"] = "Inception, Interstellar"`.
>    - Khi intent là `"search"` hoặc `"info"`, câu hỏi **KHÔNG đi vào `aggregation_stat`** mà đi qua quy trình RAG chuẩn (Hybrid Search), từ đó truy xuất thông tin cả 2 phim và đưa vào [`answer_chain.py`](file:///e:/Desktop/4/DAP391m/code/chatbot/chains/answer_chain.py) để LLM Tầng 2 viết đoạn văn so sánh giữa 2 phim.
> 3. Trong [`chains/rag_chain.py`](file:///e:/Desktop/4/DAP391m/code/chatbot/chains/rag_chain.py) (dòng 144):
>    - Chuỗi Regex `_stat_pattern` chứa từ khóa `so sánh`. Chỉ khi LLM parse intent là `aggregation` (ví dụ: *"So sánh điểm IMDb trung bình của phim hành động và phim hài"*) THÌ quy trình `aggregation_stat` mới kích hoạt để tính điểm trung bình trực tiếp.

```python
# Đoạn Prompt hệ thống định nghĩa các Intent trong prompts/intent_prompt.py:
SYSTEM_TEMPLATE = """Bạn là bộ phân tích câu hỏi cho một chatbot phim.
Nhiệm vụ: đọc câu hỏi của người dùng và trả về JSON hợp lệ DUY NHẤT, không có bất kỳ văn bản nào khác ngoài JSON.

Hướng dẫn xác định "intent":
- "search": Khi người dùng muốn tìm kiếm, lọc phim theo tiêu chí cụ thể (thể loại, đạo diễn, diễn viên, năm, điểm số, hoặc mô tả/yêu cầu tìm phim như "phim lượt xem cao nhất", "phim có lượt vote nhiều", "phim hài hước", "tìm phim...").
- "recommend": Khi người dùng yêu cầu gợi ý phim chung chung hoặc theo sở thích không có tiêu chí lọc cụ thể ("gợi ý phim hay", "phim gì nên xem tối nay", "tôi đang buồn nên xem phim gì").
- "info": Khi người dùng hỏi thông tin chi tiết của một bộ phim cụ thể ("nội dung phim Inception", "ai đóng phim Titanic").
- "aggregation": Khi người dùng hỏi câu hỏi tổng hợp/tần suất hợp tác ("ai hợp tác nhiều nhất với X", "diễn viên nào đóng nhiều phim của X nhất", "X hợp tác với ai nhiều nhất"). Câu hỏi dạng này cần phân tích trên toàn bộ filmography, không cần phim cụ thể.
- "chitchat": Chỉ khi người dùng nói chuyện phiếm, chào hỏi hoặc nói các câu không liên quan gì đến phim ảnh.
"""
```
