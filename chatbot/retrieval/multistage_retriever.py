import re
import numpy as np
import pandas as pd
from chatbot.config import FINAL_TOP_K
from chatbot.retrieval.bm25_retriever import bm25_search, build_bm25_index
from chatbot.retrieval.retriever import semantic_search_retriever
from chatbot.tools import search_movies_tool, get_movie_detail_tool
from chatbot.retrieval.reranker import rerank_results
from chatbot.feature_engineering import MovieFeatureBuilder, clean_split, PARENT_GENRES, DECADES
from chatbot.similarity import compute_weighted_similarity
from chatbot.representation.semantic_representation import make_profile

class MultistageRetriever:
    def __init__(self):
        self.builder = MovieFeatureBuilder()
        self.bm25_index = None
        
    def _get_base_movie(self, df: pd.DataFrame, query: str, filters: dict) -> tuple[pd.Series, bool]:
        """
        Helper to extract the base movie row for similar movie queries.
        """
        candidate_title = None
        
        # Ưu tiên sử dụng filters["title"] nếu có sẵn để đảm bảo chính xác tuyệt đối
        if filters and filters.get("title"):
            candidate_title = filters["title"]
            
        if not candidate_title:
            similar_patterns = [
                r'(?:phim\s+)?(giống|tương\s+tự|tựa\s+như|tựa\s+với|như)\s+(?:phim\s+)?([^,.?]+)',
                r'(?:tương\s+tự|tựa)\s+với\s+(?:phim\s+)?([^,.?]+)',
                r'(?:phim\s+)?tựa\s+(?:bộ\s+|phim\s+)?([^,.?]+)',
                r'similar\s+to\s+([^,.?]+)',
                r'like\s+([^,.?]+)'
            ]
            
            for pat in similar_patterns:
                match = re.search(pat, query, re.IGNORECASE)
                if match:
                    candidate_title = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    candidate_title = candidate_title.strip()
                    break
                    
            words_in_msg = set(re.findall(r'\b\w+\b', query.lower()))
            if not candidate_title and filters.get("title") and not words_in_msg.isdisjoint({"giống", "giong", "tương tự", "tuong tu", "như", "nhu", "tựa", "tua"}):
                candidate_title = filters["title"]
            
        if not candidate_title:
            return None, False
            
        # Loại bỏ phần phủ định/loại trừ nếu regex lỡ khớp cả cụm phía sau
        candidate_title = re.split(r'\b(nhưng|không\s+phải|trừ|ngoại\s+trừ|except|but\s+not)\b', candidate_title, flags=re.IGNORECASE)[0].strip()
        candidate_title = re.sub(r'^(bộ\s+phim|phim|bộ|cái|con|những|các|tựa|tựa\s+phim|như|nhu)\s+', '', candidate_title, flags=re.IGNORECASE).strip()

        
        # Get details
        base_movie = get_movie_detail_tool(df, candidate_title)
        if base_movie.empty:
            title_matches = df[df['Title'].astype(str).str.contains(candidate_title, case=False, na=False)]
            if not title_matches.empty:
                base_movie = title_matches.iloc[[0]]
            else:
                return None, False
                
        return base_movie.iloc[0], True

    def build_query_features(self, query: str, filters: dict, embedder_model) -> dict:
        """
        Builds a target feature dictionary from the query's filters.
        """
        # 1. Genre Vector (multi-hot)
        genre_vec = np.zeros(len(PARENT_GENRES), dtype=np.float32)
        genre_val = filters.get("genre")
        if genre_val:
            from chatbot.feature_engineering.movie_feature_builder import GENRE_HIERARCHY
            for g in clean_split(genre_val):
                mapped = GENRE_HIERARCHY.get(g, [g] if g in PARENT_GENRES else [])
                for mg in mapped:
                    if mg in self.builder.genre_to_idx:
                        genre_vec[self.builder.genre_to_idx[mg]] = 1.0
                        
        # 2. Actor Vector (indices list)
        actor_vec = []
        star_val = filters.get("star")
        if star_val:
            for a in clean_split(star_val):
                if a in self.builder.actor_to_idx:
                    actor_vec.append(self.builder.actor_to_idx[a])
                    
        # 3. Director Vector (indices list)
        director_vec = []
        dir_val = filters.get("director")
        if dir_val:
            for d in clean_split(dir_val):
                if d in self.builder.director_to_idx:
                    director_vec.append(self.builder.director_to_idx[d])
                    
        # 4. Country Vector (multi-hot)
        country_vec = np.zeros(len(self.builder.vocabularies["countries"]), dtype=np.float32)
        country_val = filters.get("country")
        if country_val:
            for c in clean_split(country_val):
                if c in self.builder.country_to_idx:
                    country_vec[self.builder.country_to_idx[c]] = 1.0
                    
        # 5. Decade Vector (one-hot)
        decade_vec = np.zeros(len(DECADES), dtype=np.float32)
        # Check year filters to derive decade
        year_val = filters.get("year_min") or filters.get("year_max")
        if year_val:
            try:
                dec_val = (int(float(year_val)) // 10) * 10
                if dec_val in self.builder.decade_to_idx:
                    decade_vec[self.builder.decade_to_idx[dec_val]] = 1.0
            except Exception:
                pass
                
        # 6. Award Vector
        award_vec = np.zeros(3, dtype=np.float32)
        if filters.get("has_awards") == 1 or filters.get("awards") == 1:
            award_vec[0] = 1.0
        if filters.get("has_oscar") == 1:
            award_vec[1] = 1.0
        if filters.get("has_nomination") == 1:
            award_vec[2] = 1.0
            
        # 7. Semantic Embedding
        semantic_emb = None
        if query and embedder_model is not None:
            semantic_emb = embedder_model.encode([query], convert_to_numpy=True)[0]
            
        return {
            "genre_vector": genre_vec.tolist(),
            "actor_vector": actor_vec,
            "director_vector": director_vec,
            "country_vector": country_vec.tolist(),
            "decade_vector": decade_vec.tolist(),
            "award_vector": award_vec.tolist(),
            "semantic_embedding": semantic_emb
        }

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
        # Initialize trace fields if trace is present
        if trace is not None:
            trace["stage1_bm25"] = {
                "top_k_requested": 100,
                "candidates": []
            }
            trace["stage2_faiss"] = {
                "candidates": []
            }
            trace["stage3_rerank"] = {
                "candidates": []
            }
            trace["stage4_weighted_similarity"] = {
                "per_candidate_scores": []
            }
            # New observability trace fields
            trace["actual_route"] = "multistage_hybrid"
            trace["skipped_stages"] = []
            trace["executed_stages"] = []
            trace["candidate_counts"] = {}

        # Sao chép và hiệu chỉnh bộ lọc cho các bước truy xuất
        filters_for_retrieval = filters.copy()
        
        # Chỉ xác định phim gốc khi có graph_candidates (tức là truy vấn tương tự đã được xác nhận từ router)
        base_row = None
        is_similar = False
        if graph_candidates is not None and not graph_candidates.empty:
            base_row, is_similar = self._get_base_movie(df, query, filters)
            if is_similar:
                filters_for_retrieval["title"] = None
        
        # --- Shortcut: Exact filter theo director/star → query trực tiếp trên toàn bộ DataFrame ---
        has_exact_person_filter = any(filters_for_retrieval.get(k) for k in ["director", "star"])
        if has_exact_person_filter and not is_similar:
            result = search_movies_tool(df, filters_for_retrieval, top_k=final_k * 3)
            # Dedup theo imdb_id + country filter
            if not result.empty:
                if 'countries_origin' in result.columns:
                    result = result[
                        result['countries_origin'].astype(str).str.strip().ne('') &
                        result['countries_origin'].notna()
                    ]
                if 'imdb_id' in result.columns:
                    result['_genre_len'] = result['genres'].astype(str).str.len()
                    result = result.sort_values('_genre_len', ascending=False)
                    result = result.drop_duplicates(subset='imdb_id', keep='first')
                    result = result.drop(columns=['_genre_len'])
            
            # Record shortcut trace before early return
            if trace is not None:
                trace["actual_route"] = "exact_filter_shortcut"
                trace["skipped_stages"] = ["candidate_generation", "bm25", "faiss", "weighted_similarity", "cross_encoder"]
                trace["executed_stages"] = ["metadata_filter"]
                trace["candidate_counts"] = {
                    "candidate_generation": 0,
                    "bm25": 0,
                    "faiss": 0,
                    "weighted_similarity": 0,
                    "cross_encoder": 0,
                    "metadata_filter": len(result) if not result.empty else 0
                }
            return result.head(final_k)
            
        # --- Stage 1: Candidate Generation ---
        # Get up to 150 semantic search candidates
        faiss_candidates = pd.DataFrame()
        if faiss_index is not None and embedder_model is not None and query:
            faiss_candidates = semantic_search_retriever(query, df, faiss_index, embedder_model, top_k=150)
            
        if trace is not None and not faiss_candidates.empty:
            faiss_trace = []
            for _, row in faiss_candidates.iterrows():
                cand = {
                    "title": row.get("Title"),
                    "imdb_id": row.get("imdb_id")
                }
                # Check for any similarity score column
                for col in ["similarity_score", "score", "faiss_score", "distance"]:
                    if col in row:
                        val = row[col]
                        cand[col] = val.item() if isinstance(val, (np.integer, np.floating)) else val
                faiss_trace.append(cand)
            trace["stage2_faiss"]["candidates"] = faiss_trace
            
        # Get up to 100 keyword search candidates
        bm25_candidates = pd.DataFrame()
        if query:
            if self.bm25_index is None:
                from chatbot.data_loader import load_bm25_index
                self.bm25_index = load_bm25_index(df)
            bm25_candidates = bm25_search(query, df, self.bm25_index, top_k=100, trace=trace)
            
        # Tracing for bm25 is now handled directly inside bm25_search to capture detailed token-level preprocessing.
            
        # Get up to 500 metadata filtering candidates directly
        metadata_candidates = pd.DataFrame()
        has_metadata_filters = any(filters_for_retrieval.get(k) for k in [
            "genre", "director", "star", "title", "year_min", "year_max", "rating_min", "country", "has_awards", "has_oscar", "has_nomination"
        ])
        if has_metadata_filters:
            metadata_candidates = search_movies_tool(df, filters_for_retrieval, top_k=500)
            
        # Combine and deduplicate candidates
        candidate_list = []
        seen_links = set()
        
        # Priority order: graph candidates (if any), FAISS candidates, BM25 candidates, then metadata candidates
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
            # Fallback to general filtered list if no query and no matches
            fallback_res = search_movies_tool(df, filters_for_retrieval, top_k=final_k)
            if trace is not None:
                trace["actual_route"] = "multistage_hybrid"
                trace["skipped_stages"] = ["weighted_similarity", "cross_encoder"]
                trace["executed_stages"] = ["candidate_generation", "bm25", "faiss", "metadata_filter"]
                trace["candidate_counts"] = {
                    "candidate_generation": 0,
                    "bm25": 0,
                    "faiss": 0,
                    "metadata_filter": len(fallback_res) if not fallback_res.empty else 0,
                    "weighted_similarity": 0,
                    "cross_encoder": 0
                }
            return fallback_res
            
        candidates_df = pd.DataFrame(candidate_list)
        
        # --- Stage 2: Metadata Filtering ---
        filtered_df = pd.DataFrame()
        if has_metadata_filters:
            filtered_df = search_movies_tool(candidates_df, filters_for_retrieval, top_k=200)
            # Fallback to whole DB if candidates got completely filtered out
            if filtered_df.empty:
                filtered_df = search_movies_tool(df, filters_for_retrieval, top_k=200)
        else:
            filtered_df = candidates_df.head(200).copy()
            
        # --- Stage 3: Weighted Similarity Ranking ---
        # 1. Identify reference profile
        if is_similar and base_row is not None:
            # Similar movie mode: reference is the base movie
            ref_features = self.builder.transform_row(base_row)
            base_profile = make_profile(base_row, version)
            if embedder_model is not None:
                ref_features["semantic_embedding"] = embedder_model.encode([base_profile], convert_to_numpy=True)[0]
                
            # Kích hoạt tính năng tính điểm graph
            ref_features["graph_score"] = 1.0
            
            # Exclude the base movie itself from candidates
            filtered_df = filtered_df[filtered_df["Movie Link"] != base_row["Movie Link"]]
        else:
            # General query mode: reference is the query profile
            ref_features = self.build_query_features(query, filters, embedder_model)
            
        # 2. Compute similarity for each candidate
        candidate_embeddings = []
        if embedder_model is not None and not filtered_df.empty:
            candidate_profiles = [make_profile(row, version) for _, row in filtered_df.iterrows()]
            candidate_embeddings = embedder_model.encode(candidate_profiles, convert_to_numpy=True)
            
        matched_rows = []
        trace_similarity = []
        for idx, (_, row) in enumerate(filtered_df.iterrows()):
            row_features = self.builder.transform_row(row)
            
            # Get candidate embedding
            if embedder_model is not None and idx < len(candidate_embeddings):
                row_features["semantic_embedding"] = candidate_embeddings[idx]
            else:
                row_features["semantic_embedding"] = None
                
            # Đính kèm graph_score cho ứng viên nếu nó đến từ Graph RAG
            # Cả hai loại (personnel và content) đều được gán graph_score=1.0 để tăng độ ưu tiên
            if "graph_path_explanation" in row and pd.notna(row["graph_path_explanation"]):
                row_features["graph_score"] = 1.0
            else:
                row_features["graph_score"] = 0.0

                
            sim_breakdown = compute_weighted_similarity(row_features, ref_features)
            
            if trace is not None:
                clean_sim = {}
                for k, v in sim_breakdown.items():
                    if isinstance(v, (np.integer, np.floating)):
                        clean_sim[k] = v.item()
                    else:
                        clean_sim[k] = v
                trace_similarity.append(clean_sim)
            
            # Attach score breakdown to movie row
            row_copy = row.copy()
            row_copy["similarity_score"] = f"{sim_breakdown['final_score'] * 100:.1f}%"
            row_copy["final_similarity_score"] = sim_breakdown["final_score"]
            row_copy["genre_similarity"] = sim_breakdown["genre_score"]
            row_copy["actor_similarity"] = sim_breakdown["actor_score"]
            row_copy["director_similarity"] = sim_breakdown["director_score"]
            row_copy["country_similarity"] = sim_breakdown["country_score"]
            row_copy["decade_similarity"] = sim_breakdown["decade_score"]
            row_copy["award_similarity"] = sim_breakdown["award_score"]
            row_copy["content_similarity"] = sim_breakdown["content_score"]
            
            # Generate explaining reason
            reasons = []
            if sim_breakdown["genre_score"] > 0.7:
                reasons.append("cùng thể loại")
            if sim_breakdown["actor_score"] > 0:
                reasons.append("diễn viên tương đồng")
            if sim_breakdown["director_score"] > 0:
                reasons.append("cùng đạo diễn")
            if sim_breakdown["content_score"] > 0.6:
                reasons.append("nội dung/chủ đề tương đồng")
            if sim_breakdown["country_score"] > 0.7:
                reasons.append("quốc gia sản xuất")
            if "graph_path_explanation" in row_copy and pd.notna(row_copy["graph_path_explanation"]):
                if row_copy.get("graph_path_type") == "personnel":
                    reasons.append("quan hệ hợp tác gián tiếp qua graph")
                else:
                    reasons.append("liên kết thuộc tính chung qua đồ thị")
                
            if not reasons:
                reasons.append("phong cách nghệ thuật tương đồng")
            row_copy["similarity_reason"] = "Phim " + ", ".join(reasons) + "."
            matched_rows.append(row_copy)
            
        if trace is not None:
            trace["stage4_weighted_similarity"]["per_candidate_scores"] = trace_similarity
            
        if not matched_rows:
            if trace is not None:
                executed_stages = ["candidate_generation"]
                skipped_stages = ["cross_encoder"]
                if query:
                    executed_stages.append("bm25")
                else:
                    skipped_stages.append("bm25")
                if faiss_index is not None and embedder_model is not None and query:
                    executed_stages.append("faiss")
                else:
                    skipped_stages.append("faiss")
                executed_stages.extend(["metadata_filter", "weighted_similarity"])
                
                trace["actual_route"] = "multistage_hybrid"
                trace["executed_stages"] = executed_stages
                trace["skipped_stages"] = skipped_stages
                trace["candidate_counts"] = {
                    "candidate_generation": len(candidates_df) if not candidates_df.empty else 0,
                    "bm25": len(bm25_candidates) if not bm25_candidates.empty else 0,
                    "faiss": len(faiss_candidates) if not faiss_candidates.empty else 0,
                    "metadata_filter": len(filtered_df) if not filtered_df.empty else 0,
                    "weighted_similarity": 0,
                    "cross_encoder": 0
                }
            return pd.DataFrame()
            
        ranked_df = pd.DataFrame(matched_rows)
        # Sort by weighted score descending
        ranked_df = ranked_df.sort_values(by="final_similarity_score", ascending=False)
        top_100_df = ranked_df.head(100).copy()
        
        # --- Stage 4: Cross-Encoder Reranking ---
        # Sử dụng câu truy vấn gốc của user để rerank chính xác hơn về nội dung/chủ đề
        if is_similar and base_row is not None:
            # Kết hợp cả tên phim gốc và nội dung câu hỏi để reranker nắm bắt được yêu cầu chủ đề
            query_rerank = f"{query} (phim tương tự như {base_row['Title']})"
        else:
            query_rerank = query or "Phim hay được đánh giá cao."
            
        reranked_df = rerank_results(query_rerank, top_100_df, top_k=20)
        
        if trace is not None and not reranked_df.empty:
            rerank_trace = []
            for _, row in reranked_df.iterrows():
                val = row.get("rerank_score")
                rerank_trace.append({
                    "title": row.get("Title"),
                    "imdb_id": row.get("imdb_id"),
                    "rerank_score": val.item() if isinstance(val, (np.integer, np.floating)) else val
                })
            trace["stage3_rerank"]["candidates"] = rerank_trace
        
        # --- Stage 5: Final Results ---
        if not reranked_df.empty:
            # 5a. Loại phim thiếu country
            if 'countries_origin' in reranked_df.columns:
                reranked_df = reranked_df[
                    reranked_df['countries_origin'].astype(str).str.strip().ne('') &
                    reranked_df['countries_origin'].notna()
                ]
            
            # 5b. Dedup theo imdb_id: giữ dòng có genres dài nhất
            if 'imdb_id' in reranked_df.columns:
                reranked_df['_genre_len'] = reranked_df['genres'].astype(str).str.len()
                reranked_df = reranked_df.sort_values('_genre_len', ascending=False)
                reranked_df = reranked_df.drop_duplicates(subset='imdb_id', keep='first')
                reranked_df = reranked_df.drop(columns=['_genre_len'])
            
            # 5c. PRIORITY 4 FIX: Áp dụng lại các ràng buộc cứng sau khi rerank
            # (rating_min, runtime_max, runtime_min) để đảm bảo hard constraints không bị
            # override bởi soft semantic score trong Cross-Encoder
            from chatbot.config import COL_RATING, COL_DURATION
            rating_min = filters.get("rating_min")
            if rating_min is not None:
                try:
                    hard_filtered = reranked_df[reranked_df[COL_RATING] >= float(rating_min)]
                    # Chỉ áp dụng nếu vẫn còn đủ kết quả (tránh trả về rỗng)
                    if len(hard_filtered) >= max(1, final_k // 2):
                        reranked_df = hard_filtered
                except Exception:
                    pass
            
            dur_min = filters.get("duration_min") or filters.get("runtime_min")
            if dur_min is not None:
                try:
                    hard_filtered = reranked_df[reranked_df[COL_DURATION] >= float(dur_min)]
                    if len(hard_filtered) >= max(1, final_k // 2):
                        reranked_df = hard_filtered
                except Exception:
                    pass
            
            dur_max = filters.get("duration_max") or filters.get("runtime_max")
            if dur_max is not None:
                try:
                    hard_filtered = reranked_df[reranked_df[COL_DURATION] <= float(dur_max)]
                    if len(hard_filtered) >= max(1, final_k // 2):
                        reranked_df = hard_filtered
                except Exception:
                    pass
        
        if trace is not None:
            executed_stages = ["candidate_generation"]
            skipped_stages = []
            if query:
                executed_stages.append("bm25")
            else:
                skipped_stages.append("bm25")
            if faiss_index is not None and embedder_model is not None and query:
                executed_stages.append("faiss")
            else:
                skipped_stages.append("faiss")
            executed_stages.extend(["metadata_filter", "weighted_similarity", "cross_encoder"])
            
            trace["actual_route"] = "multistage_hybrid"
            trace["executed_stages"] = executed_stages
            trace["skipped_stages"] = skipped_stages
            trace["candidate_counts"] = {
                "candidate_generation": len(candidates_df) if not candidates_df.empty else 0,
                "bm25": len(bm25_candidates) if not bm25_candidates.empty else 0,
                "faiss": len(faiss_candidates) if not faiss_candidates.empty else 0,
                "metadata_filter": len(filtered_df) if not filtered_df.empty else 0,
                "weighted_similarity": len(matched_rows),
                "cross_encoder": len(reranked_df) if not reranked_df.empty else 0
            }

        return reranked_df.head(final_k)
