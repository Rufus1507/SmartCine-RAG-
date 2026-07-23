import re
import unicodedata
import pandas as pd
from langchain_core.language_models import BaseChatModel
from chatbot.entity_extractor import detect_entities, is_refine_query, extract_content_keywords_fallback
from chatbot.chains.intent_chain import run_intent_chain
from chatbot.chains.answer_chain import run_answer_chain
from chatbot.retrieval.hybrid_search import hybrid_search
from chatbot.tools import find_similar_movies
from rapidfuzz import fuzz

def _strip_diacritics(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def run_rag_pipeline(
    llm: BaseChatModel, 
    user_input: str, 
    df: pd.DataFrame, 
    keyword_dict: dict, 
    aliases_dict: dict, 
    faiss_index, 
    embedder_model, 
    chat_history: list,
    last_filters: dict,
    stream: bool = False,
    debug: bool = False
) -> tuple:
    """
    Điều phối luồng xử lý RAG (Orchestrator Chain):
    1. Trích xuất thực thể (entity extraction) từ câu hỏi.
    2. Chạy intent_chain phân tích ý định & bộ lọc thuộc tính.
    3. Hợp nhất bộ lọc với lịch sử chat (tránh lỗi logic với intent='info').
    4. Sửa lỗi bộ lọc tự động (Auto-correction).
    5. Thực hiện truy xuất: Tìm phim tương đồng (Similar) hoặc Hybrid Search (FAISS + Pandas).
    6. Chạy answer_chain sinh câu trả lời tự nhiên (đồng bộ hoặc stream).
    
    Trả về: (answer_result, filtered_df, intent, filters, detected) hoặc thêm trace nếu debug=True
    """
    import numpy as np

    # Khởi tạo trace nếu debug=True
    trace = None
    if debug:
        trace = {
            "entity_detection": {},
            "intent": {},
            "stage0_graph": {
                "called": False,
                "candidates": []
            },
            "stage1_bm25": {
                "top_k_requested": 100,
                "candidates": []
            },
            "stage2_faiss": {
                "candidates": []
            },
            "stage3_rerank": {
                "candidates": []
            },
            "stage4_weighted_similarity": {
                "per_candidate_scores": []
            },
            "final_filters": {},
            "final_route": "none"
        }

    # 1. Phát hiện thực thể
    detected = detect_entities(user_input, keyword_dict, aliases_dict)
    # Fallback content_keywords: khi query không chứa entity rõ ràng (genres/directors/stars rỗng),
    # tự động trích 2-4 danh từ/cụm từ mô tả nội dung để content dimension có ngữ cảnh truy xuất
    if not any(detected.get(k) for k in ("genres", "directors", "stars")) and not detected.get("content_keywords"):
        detected["content_keywords"] = extract_content_keywords_fallback(user_input, max_keywords=4)
    if debug and trace is not None:
        trace["entity_detection"] = detected.copy()
    
    # 2. Phân tích intent (Tầng 1 LLM)
    parsed = run_intent_chain(llm, user_input, detected, chat_history)
    intent = parsed.get("intent", "chitchat")
    filters = parsed.get("filters", {})
    if debug and trace is not None:
        trace["intent"] = parsed.copy()
    
    # 3. Quản lý ngữ cảnh và hợp nhất bộ lọc (Conversation Memory)
    # Khắc phục lỗi: Chỉ hợp nhất bộ lọc cho search/recommend nối tiếp, không áp dụng cho info
    if intent in ("search", "recommend") and is_refine_query(user_input):
        new_filters = {k: v for k, v in filters.items() if v is not None}
        filters = {**last_filters, **new_filters}
    elif intent == "info":
        # Khi hỏi chi tiết phim, loại bỏ hoàn toàn các filter cũ
        filters = {"title": filters.get("title")}

    # 4. Tự động sửa lỗi bộ lọc (Auto-correction)
    if filters.get("title"):
        title_lower = filters["title"].lower()
        # Nếu title khớp với đạo diễn đã phát hiện
        for d in detected.get("directors", []):
            if d.lower() == title_lower or fuzz.QRatio(d.lower(), title_lower) >= 90:
                filters["director"] = d
                filters["title"] = None
        # Nếu title khớp với diễn viên đã phát hiện
        for s in detected.get("stars", []):
            if s.lower() == title_lower or fuzz.QRatio(s.lower(), title_lower) >= 90:
                filters["star"] = s
                filters["title"] = None

    # Chuẩn hoá thể loại và quốc gia
    if filters.get("genre"):
        # Xác định chế độ logic cho nhiều thể loại
        # CHỈ dùng AND khi chuỗi genre chứa "và/and/&" giữa các tên thể loại
        # (không dùng toàn bộ user_input để tránh match nhầm "và" trong mệnh đề khác)
        genre_str = str(filters["genre"])
        if re.search(r'\bvà\b|\band\b|&', genre_str, re.IGNORECASE):
            filters["genre_mode"] = "AND"
        else:
            filters["genre_mode"] = "OR"
            
        from chatbot.tools import normalize_genre
        filters["genre"] = normalize_genre(filters["genre"])
    if filters.get("country"):
        from chatbot.data_loader import load_country_aliases
        country_aliases = load_country_aliases()
        country_query = str(filters["country"]).strip().lower()
        resolved = country_aliases.get(country_query)
        if not resolved:
            country_query_stripped = _strip_diacritics(country_query)
            for k, v in country_aliases.items():
                if _strip_diacritics(k) == country_query_stripped:
                    resolved = v
                    break
        if resolved:
            filters["country"] = resolved

    # 5. Truy xuất phim (Retrieval Layer)
    filtered_df = pd.DataFrame()
    route_name = "none"
    
    if intent == "aggregation":
        # Xử lý câu hỏi tổng hợp: "ai hợp tác nhiều nhất với X"
        person_name = filters.get("director") or filters.get("star")
        
        # PRIORITY 4 FIX: Phát hiện truy vấn tính thống kê (trung bình, cao hơn mức trung bình...)
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
        
        # PRIORITY 5 FIX: Nếu không có director/star nhưng có title,
        # thực hiện tra cứu multi-hop: title → tìm đạo diễn trong dataset → tìm collaborators.
        # Bỏ qua nếu P4 đã xử lý query này dưới dạng thống kê (aggregation_stat).
        if not person_name and filters.get("title") and route_name != "aggregation_stat":
            movie_title = filters["title"]
            # Ưu tiên exact match (case-insensitive) trước để tránh khớp nhầm tên phim gần giống
            exact_match = df[df['Title'].astype(str).str.lower() == movie_title.lower()]
            if not exact_match.empty:
                title_match = exact_match
            else:
                # Fallback: tìm kiếm mờ (fuzzy contains) nếu không có kết quả exact
                title_match = df[df['Title'].astype(str).str.contains(re.escape(movie_title), case=False, na=False)]
                # Sắp xếp ưu tiên kết quả có tên phim ngắn hơn (khớp gần hơn với tên tìm kiếm)
                if not title_match.empty:
                    title_match = title_match.assign(
                        _title_len=title_match['Title'].str.len()
                    ).sort_values('_title_len').drop(columns=['_title_len'])
            
            if not title_match.empty:
                row = title_match.iloc[0]
                director_val = row.get("directors", None) or row.get("Director", None)
                if director_val:
                    # Lấy tên đạo diễn đầu tiên nếu có nhiều người
                    person_name = str(director_val).split(",")[0].strip()
                    filters["director"] = person_name
            else:
                # Movie không có trong dataset - trả về thông báo data limitation
                not_found_row = [{
                    "Title": f"[Không tìm thấy: {movie_title}]",
                    "Rating": None,
                    "Year": None,
                    "genres": "",
                    "directors": "",
                    "countries_origin": "N/A",
                    "stars": "",
                    "Movie Link": "",
                    "final_context": f"Phim '{movie_title}' không có trong cơ sở dữ liệu hiện tại. Đây có thể là phim quá mới hoặc chưa được cập nhật vào hệ thống. Vui lòng thử với tên phim khác."
                }]
                filtered_df = pd.DataFrame(not_found_row)
                route_name = "aggregation_not_found"
                
        # Thực hiện Graph RAG tìm cộng tác viên hoặc phim hợp tác chung (Co-collaboration)
        if (filters.get("director") or filters.get("star")) and route_name != "aggregation_stat":
            try:
                from chatbot.graph.build_movie_graph import load_or_build_graph
                G = load_or_build_graph(df)
                
                # Tách danh sách director và star để kiểm tra co-collaboration
                directors_list = []
                if filters.get("director"):
                    directors_list = [d.strip() for d in re.split(r'[,;]|\bvà\b|\bhoặc\b|\band\b|\bor\b', str(filters["director"]), flags=re.IGNORECASE) if d.strip()]
                stars_list = []
                if filters.get("star"):
                    stars_list = [s.strip() for s in re.split(r'[,;]|\bvà\b|\bhoặc\b|\band\b|\bor\b', str(filters["star"]), flags=re.IGNORECASE) if s.strip()]
                
                if len(directors_list) + len(stars_list) >= 2:
                    # Truy vấn phim chung của nhiều thực thể (Co-collaboration)
                    from chatbot.graph.graph_query import find_common_movies_of_entities
                    common_movies = find_common_movies_of_entities(G, directors_list, stars_list)
                    
                    if debug and trace is not None:
                        trace["stage0_graph"]["called"] = True
                        trace["stage0_graph"]["candidates"] = [{"Title": m, "type": "Movie"} for m in common_movies]
                        
                    if common_movies:
                        # Lấy thông tin chi tiết các phim chung từ DataFrame
                        common_df = df[df["Title"].str.lower().isin([t.lower() for t in common_movies])]
                        movie_rows = []
                        for _, row in common_df.iterrows():
                            # Lấy các trường cơ bản, bổ sung final_context đầy đủ để RAG trả lời đúng
                            movie_rows.append({
                                "Title": row.get("Title"),
                                "Rating": row.get("Rating"),
                                "Year": row.get("Year"),
                                "genres": row.get("genres", ""),
                                "directors": row.get("directors", ""),
                                "countries_origin": row.get("countries_origin", ""),
                                "stars": row.get("stars", ""),
                                "Movie Link": row.get("Movie Link", ""),
                                "final_context": f"Tên phim: {row.get('Title')} | Năm phát hành: {int(row.get('Year')) if pd.notna(row.get('Year')) else 'N/A'} | Điểm IMDb: {row.get('Rating')} | Đạo diễn: {row.get('directors')} | Diễn viên: {row.get('stars')} | Thể loại: {row.get('genres')}"
                            })
                        filtered_df = pd.DataFrame(movie_rows)
                        route_name = "aggregation_graph"
                    else:
                        filtered_df = pd.DataFrame()
                        route_name = "aggregation_no_results"
                else:
                    # Fallback tìm kiếm cộng tác viên đơn lẻ như cũ
                    person_name = directors_list[0] if directors_list else (stars_list[0] if stars_list else None)
                    from chatbot.graph.graph_query import find_top_collaborator
                    top_collabs = find_top_collaborator(G, person_name, top_k=5)
                    
                    if debug and trace is not None:
                        trace["stage0_graph"]["called"] = True
                        serializable_graph = []
                        for c in top_collabs:
                            clean_c = {}
                            for k, v in c.items():
                                if isinstance(v, (np.integer, np.floating)):
                                    clean_c[k] = v.item()
                                else:
                                    clean_c[k] = v
                            serializable_graph.append(clean_c)
                        trace["stage0_graph"]["candidates"] = serializable_graph
                    
                    if top_collabs:
                        collab_rows = []
                        for c in top_collabs:
                            collab_name = c['name']
                            _shared_movies_info = ""
                            _avg_rating_info = ""
                            _role_info = ""
                            try:
                                from chatbot.config import COL_RATING, COL_STARS
                                _shared = df[
                                    df.get('directors', pd.Series(dtype=str)).astype(str).str.contains(re.escape(person_name), case=False, na=False) &
                                    df.get('stars', pd.Series(dtype=str)).astype(str).str.contains(re.escape(collab_name), case=False, na=False)
                                ] if 'directors' in df.columns and 'stars' in df.columns else pd.DataFrame()
                                if _shared.empty:
                                    _shared = df[
                                        df['directors'].astype(str).str.contains(re.escape(collab_name), case=False, na=False) &
                                        df['stars'].astype(str).str.contains(re.escape(person_name), case=False, na=False)
                                    ] if 'directors' in df.columns and 'stars' in df.columns else pd.DataFrame()
                                if not _shared.empty:
                                    _movie_titles = _shared['Title'].tolist()[:5]
                                    _shared_movies_info = f" | Phim chung: {', '.join(_movie_titles)}"
                                    _avg_r = _shared[COL_RATING].dropna().mean() if COL_RATING in _shared.columns else None
                                    if _avg_r is not None:
                                        _avg_rating_info = f" | Điểm IMDb trung bình: {round(_avg_r, 2)}"
                                    _lead_count = 0
                                    _support_count = 0
                                    for _, _sr in _shared.iterrows():
                                        _stars_list = str(_sr.get('stars', '')).split(',')
                                        _stars_clean = [s.strip() for s in _stars_list]
                                        if _stars_clean and _stars_clean[0].lower().startswith(collab_name.lower()[:8]):
                                            _lead_count += 1
                                        else:
                                            _support_count += 1
                                    _role_info = f" | Vai chính: {_lead_count} phim, Vai phụ: {_support_count} phim"
                            except Exception:
                                pass
                            collab_rows.append({
                                "Title": f"{collab_name} ({c['type']})",
                                "Rating": c["weight"],
                                "Year": None,
                                "genres": "",
                                "directors": person_name,
                                "countries_origin": "",
                                "stars": collab_name,
                                "Movie Link": "",
                                "final_context": f"Tên: {collab_name} | Vai trò: {c['type']} | Số lần hợp tác: {c['weight']}{_shared_movies_info}{_avg_rating_info}{_role_info}"
                            })
                        filtered_df = pd.DataFrame(collab_rows)
                        route_name = "aggregation_graph"
            except Exception as e:
                print(f"Aggregation error: {e}")
        route_name = route_name or "aggregation_no_person"
    elif intent in ("search", "recommend", "info"):
        has_metadata_filters = any(filters.get(k) for k in [
            "genre", "director", "star", "title", "year_min", "year_max", "rating_min", "country"
        ])
        
        if intent == "info" and not has_metadata_filters:
            # Trả về df rỗng để chatbot hỏi làm rõ
            filtered_df = pd.DataFrame()
            route_name = "info_clarify"
        else:
            from chatbot.retrieval.retrieval_router import route_retrieval
            from chatbot.retrieval.retrieval_router import is_similar_movie_query
            
            # Nếu là truy vấn tìm phim tương đương, đổi intent thành search
            if is_similar_movie_query(user_input, filters):
                intent = "search"
                
            filtered_df, route_name = route_retrieval(
                query=user_input,
                df=df,
                filters=filters,
                intent=intent,
                faiss_index=faiss_index,
                embedder_model=embedder_model,
                trace=trace
            )
    else:
        filtered_df = pd.DataFrame()

    # 6. Sinh câu trả lời (Tầng 2 LLM)
    answer_result = run_answer_chain(llm, user_input, filtered_df, intent, stream=stream, trace=trace)
    
    # Đóng gói route_name vào detected để trả về mà không làm gãy signature
    detected["route_name"] = route_name
    
    if debug and trace is not None:
        trace["final_filters"] = filters.copy() if filters else {}
        trace["final_route"] = route_name
        return answer_result, filtered_df, intent, filters, detected, trace
        
    return answer_result, filtered_df, intent, filters, detected
