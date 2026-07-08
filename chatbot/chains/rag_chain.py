import unicodedata
import pandas as pd
from langchain_core.language_models import BaseChatModel
from chatbot.entity_extractor import detect_entities, is_refine_query
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
        if person_name:
            try:
                from chatbot.graph.build_movie_graph import load_or_build_graph
                from chatbot.graph.graph_query import find_top_collaborator
                G = load_or_build_graph(df)
                top_collabs = find_top_collaborator(G, person_name, top_k=5)
                
                # Trace cho Graph RAG của aggregation
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
                        collab_rows.append({
                            "Title": f"{c['name']} ({c['type']})",
                            "Rating": c["weight"],
                            "Year": None,
                            "genres": "",
                            "directors": person_name,
                            "countries_origin": "",
                            "stars": c["name"],
                            "Movie Link": "",
                            "final_context": f"Tên: {c['name']} | Vai trò: {c['type']} | Số lần hợp tác: {c['weight']}"
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
    answer_result = run_answer_chain(llm, user_input, filtered_df, intent, stream=stream)
    
    # Đóng gói route_name vào detected để trả về mà không làm gãy signature
    detected["route_name"] = route_name
    
    if debug and trace is not None:
        trace["final_filters"] = filters.copy() if filters else {}
        trace["final_route"] = route_name
        return answer_result, filtered_df, intent, filters, detected, trace
        
    return answer_result, filtered_df, intent, filters, detected
