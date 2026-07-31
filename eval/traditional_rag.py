import os
import sys
import re
import pandas as pd
import numpy as np
import faiss
from langchain_core.language_models import BaseChatModel

# Thêm thư mục gốc vào path
eval_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(eval_dir)
sys.path.append(workspace_dir)

from chatbot.prompts.answer_prompt import get_rag_prompt
from chatbot.config import COL_TITLE, COL_YEAR, COL_RATING, COL_LINK
from chatbot.retrieval.bm25_retriever import bm25_search, build_bm25_index

# Lazy static cache cho BM25 index (không dùng Streamlit cache để chạy ngoài Streamlit)
_TRADITIONAL_BM25_INDEX = None

def get_traditional_bm25_index(df: pd.DataFrame):
    global _TRADITIONAL_BM25_INDEX
    if _TRADITIONAL_BM25_INDEX is None:
        print("⚙️  [Traditional RAG] Đang xây dựng BM25 index...")
        _TRADITIONAL_BM25_INDEX = build_bm25_index(df)
        print("✅ [Traditional RAG] BM25 index sẵn sàng.")
    return _TRADITIONAL_BM25_INDEX

def extract_metadata_filters(query: str) -> dict:
    """
    Tự động trích xuất các điều kiện ràng buộc thuộc tính (Metadata Filters) từ câu hỏi tự nhiên.
    Áp dụng quy tắc tổng quát (General Auto-Query Filtering) chuẩn trong Metadata-Aware RAG.
    """
    filters = {}
    q_lower = query.lower()
    
    # 1. Ràng buộc Năm phát hành (Year)
    m_year_exact = re.search(r'(?:năm|ra mắt năm|phát hành năm)\s*(\d{4})', q_lower)
    if m_year_exact:
        filters['year_exact'] = int(m_year_exact.group(1))
    else:
        m_after = re.search(r'(?:sau năm|từ năm)\s*(\d{4})', q_lower)
        if m_after:
            filters['year_min'] = int(m_after.group(1)) + 1
        m_before = re.search(r'(?:trước năm)\s*(\d{4})', q_lower)
        if m_before:
            filters['year_max'] = int(m_before.group(1)) - 1
            
    # 2. Ràng buộc Điểm IMDb (Rating)
    m_rating = re.search(r'(?:imdb|điểm)\s*(?:trên|từ|>=|>)\s*(\d+(?:\.\d+)?)', q_lower)
    if m_rating:
        filters['rating_min'] = float(m_rating.group(1))
        
    # 3. Ràng buộc Thời lượng (Duration min)
    m_dur_under = re.search(r'(?:dưới|nhỏ hơn|<)\s*(\d+)\s*phút', q_lower)
    if m_dur_under:
        filters['duration_max'] = int(m_dur_under.group(1))
    m_dur_over = re.search(r'(?:trên|lớn hơn|>)\s*(\d+)\s*phút', q_lower)
    if m_dur_over:
        filters['duration_min'] = int(m_dur_over.group(1))
        
    # 4. Ràng buộc Quốc gia (Country)
    if 'hàn quốc' in q_lower or 'korea' in q_lower:
        filters['country'] = 'South Korea'
    elif 'anh' in q_lower or 'nước anh' in q_lower or 'uk' in q_lower:
        filters['country'] = 'United Kingdom'
    elif 'mỹ' in q_lower or 'hoa kỳ' in q_lower or 'us' in q_lower:
        filters['country'] = 'United States'
        
    # 5. Ràng buộc Giải thưởng (Oscar)
    if 'oscar' in q_lower:
        filters['has_oscar'] = 1
        
    return filters

def retrieve_traditional(
    query: str,
    index: faiss.Index,
    model,
    df: pd.DataFrame,
    top_k: int = 20,
    use_hybrid: bool = True
) -> pd.DataFrame:
    """
    Truy xuất Top K bộ phim tương đồng nhất dùng Filtered Hybrid Search:
    Dense Vector Search (FAISS FlatIP) + Sparse Keyword Search (BM25Okapi)
    kết hợp RRF và tự động lọc Metadata constraints (Năm, Điểm, Thời lượng, Oscar...).
    """
    if not query or df.empty:
        return pd.DataFrame()
        
    fetch_depth = max(top_k * 10, 200)
    
    # 1. Dense Vector Search (FAISS)
    faiss_indices = []
    if index is not None and model is not None:
        query_vector = model.encode([query], convert_to_numpy=True).astype('float32')
        query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_norm = np.where(query_norm == 0, 1.0, query_norm)
        query_vector = query_vector / query_norm
        distances, indices = index.search(query_vector, fetch_depth)
        faiss_indices = [idx for idx in indices[0] if 0 <= idx < len(df)]

    # 2. Sparse Keyword Search (BM25)
    bm25_df = pd.DataFrame()
    if use_hybrid:
        try:
            bm25_idx = get_traditional_bm25_index(df)
            bm25_df = bm25_search(query, df, bm25_idx, top_k=fetch_depth)
        except Exception:
            bm25_df = pd.DataFrame()

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    k_const = 60
    
    # Tính RRF cho Dense Vector
    for rank, idx in enumerate(faiss_indices):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_const + rank + 1))
        
    # Tính RRF cho BM25
    if not bm25_df.empty:
        for rank, (idx, _) in enumerate(bm25_df.iterrows()):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_const + rank + 1))
            
    if not rrf_scores:
        return pd.DataFrame()
        
    sorted_all_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    # 4. Tự động áp dụng Metadata Auto-Filtering
    filters = extract_metadata_filters(query)
    
    filtered_indices = []
    unfiltered_indices = []
    
    for idx in sorted_all_indices:
        row = df.iloc[idx]
        pass_filter = True
        
        if 'year_exact' in filters and row.get('Year') != filters['year_exact']:
            pass_filter = False
        if 'year_min' in filters and (pd.isna(row.get('Year')) or row.get('Year') < filters['year_min']):
            pass_filter = False
        if 'year_max' in filters and (pd.isna(row.get('Year')) or row.get('Year') > filters['year_max']):
            pass_filter = False
        if 'rating_min' in filters and (pd.isna(row.get('Rating')) or row.get('Rating') < filters['rating_min']):
            pass_filter = False
        if 'duration_min' in filters and (pd.isna(row.get('duration_min')) or row.get('duration_min') < filters['duration_min']):
            pass_filter = False
        if 'duration_max' in filters and (pd.isna(row.get('duration_min')) or row.get('duration_min') > filters['duration_max']):
            pass_filter = False
        if 'country' in filters and (pd.isna(row.get('countries_origin')) or filters['country'].lower() not in str(row.get('countries_origin')).lower()):
            pass_filter = False
        if 'has_oscar' in filters and row.get('has_oscar') != 1:
            pass_filter = False
            
        if pass_filter:
            filtered_indices.append(idx)
        else:
            unfiltered_indices.append(idx)
            
    # Kết hợp ưu tiên các ứng viên thỏa mãn filter, sau đó bổ sung nếu chưa đủ top_k
    final_selected = filtered_indices[:top_k]
    if len(final_selected) < top_k:
        needed = top_k - len(final_selected)
        final_selected.extend(unfiltered_indices[:needed])
        
    retrieved_df = df.iloc[final_selected].copy()
    retrieved_df['rrf_score'] = [rrf_scores[idx] for idx in final_selected]
    
    return retrieved_df

def run_traditional_rag_pipeline(
    query: str,
    llm: BaseChatModel,
    df: pd.DataFrame,
    index: faiss.Index,
    model,
    top_k: int = 20
) -> tuple[str, pd.DataFrame]:
    """
    Chạy luồng RAG truyền thống (Standard Hybrid RAG): Retrieval (FAISS ∪ BM25) -> Generation.
    """
    # 1. Retrieval
    retrieved_df = retrieve_traditional(query, index, model, df, top_k=top_k)
    
    if retrieved_df.empty:
        return "Xin lỗi, tôi không tìm thấy bộ phim nào phù hợp trong cơ sở dữ liệu.", retrieved_df
        
    # 2. Định dạng ngữ cảnh (Context) chi tiết cho prompt
    movies_info_list = []
    for idx, (_, row) in enumerate(retrieved_df.iterrows(), 1):
        context_text = row.get('final_context')
        if not context_text or str(context_text).strip() == "":
            parts = []
            title = row.get('Title')
            year = row.get('Year')
            if pd.notna(title):
                parts.append(f"Title: {title}")
            if pd.notna(year):
                parts.append(f"Year: {year}")
            if pd.notna(row.get('genres')):
                parts.append(f"Genres: {row.get('genres')}")
            if pd.notna(row.get('Rating')):
                parts.append(f"Rating: {row.get('Rating')}")
            if pd.notna(row.get('duration_min')):
                parts.append(f"Duration: {row.get('duration_min')} minutes")
            if pd.notna(row.get('countries_origin')):
                parts.append(f"Country: {row.get('countries_origin')}")
            if pd.notna(row.get('directors')):
                parts.append(f"Directors: {row.get('directors')}")
            if pd.notna(row.get('stars')):
                parts.append(f"Stars: {row.get('stars')}")
            if pd.notna(row.get('description')):
                parts.append(f"Overview: {row.get('description')}")
            context_text = ". ".join(parts)
            
        movie_str = f"[{idx}] {context_text}\n"
        if COL_LINK in row and pd.notna(row[COL_LINK]):
            movie_str += f"  Link IMDb: {row[COL_LINK]}\n"
        movies_info_list.append(movie_str)
        
    movies_info = "\n".join(movies_info_list)
    
    # 3. Tạo Prompt & gọi LLM
    prompt_template = get_rag_prompt()
    chain = prompt_template | llm
    
    try:
        response = chain.invoke({
            "input": query,
            "movies_info": movies_info
        })
    except Exception as e:
        print(f"⚠️  [Traditional RAG] LLM call failed ({e}). Falling back to structured candidate synthesis.")
        fallback_lines = [
            f"Dựa trên dữ liệu tìm kiếm, dưới đây là các bộ phim phù hợp nhất cho câu hỏi '{query}':\n"
        ]
        for idx, (_, row) in enumerate(retrieved_df.head(5).iterrows(), 1):
            t = row.get("Title", "N/A")
            y = row.get("Year", "N/A")
            g = row.get("genres", "N/A")
            r = row.get("Rating", "N/A")
            d = row.get("description", row.get("final_context", "N/A"))
            l = row.get("Movie Link", "")
            
            line = f"{idx}. **{t}** ({y})"
            if pd.notna(g) and g != "N/A":
                line += f" - Thể loại: {g}"
            if pd.notna(r) and r != "N/A":
                line += f" | IMDb: {r}"
            if pd.notna(d) and d != "N/A":
                short_desc = str(d)[:200] + "..." if len(str(d)) > 200 else str(d)
                line += f"\n   - Mô tả: {short_desc}"
            if l:
                line += f"\n   - Chi tiết: {l}"
            fallback_lines.append(line)
            
        answer_result = "\n".join(fallback_lines)
        
    return answer_result, retrieved_df

