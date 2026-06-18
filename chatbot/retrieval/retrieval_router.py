import re
import pandas as pd
from chatbot.config import FINAL_TOP_K
from chatbot.tools import search_movies_tool
from chatbot.retrieval.bm25_retriever import bm25_search
from chatbot.data_loader import load_bm25_index
from chatbot.retrieval.hybrid_search import hybrid_search
from chatbot.retrieval.reranker import rerank_results

def is_similar_movie_query(query: str, filters: dict) -> bool:
    """
    Xác định xem truy vấn có phải là yêu cầu tìm phim tương tự hay không.
    """
    similar_patterns = [
        r'(?:phim\s+)?(giống|tương\s+tự|tựa\s+như|tựa\s+với|như)\s+(?:phim\s+)?',
        r'similar\s+to',
        r'like\s+'
    ]
    query_lower = query.lower()
    for pat in similar_patterns:
        if re.search(pat, query_lower):
            return True
            
    # Kiểm tra xem từ khóa có chứa từ chỉ định tương đồng không
    words_in_msg = set(re.findall(r'\b\w+\b', query_lower))
    if filters.get("title") and not words_in_msg.isdisjoint({"giống", "giong", "tương tự", "tuong tu", "như", "nhu", "tựa", "tua"}):
        return True
        
    return False

def route_retrieval(
    query: str,
    df: pd.DataFrame,
    filters: dict,
    intent: str,
    faiss_index,
    embedder_model,
    final_k: int = FINAL_TOP_K
) -> tuple[pd.DataFrame, str]:
    """
    Định tuyến truy vấn (Retrieval Router) sang hệ thống Multi-stage Hybrid Retrieval V3.
    Trả về tuple: (DataFrame kết quả, tên luồng định tuyến)
    """
    from chatbot.retrieval.multistage_retriever import MultistageRetriever
    retriever = MultistageRetriever()
    result = retriever.retrieve(
        query=query,
        df=df,
        filters=filters,
        intent=intent,
        faiss_index=faiss_index,
        embedder_model=embedder_model,
        final_k=final_k
    )
    return result, "multistage_hybrid"
