import os
import sys
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

def retrieve_traditional(query: str, index: faiss.Index, model, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    Truy xuất Top K bộ phim tương đồng ngữ nghĩa nhất dùng Naive RAG (truy vấn vector mô tả phim).
    """
    if not query or index is None or model is None:
        return pd.DataFrame()
        
    # 1. Tính embedding của câu hỏi và chuẩn hóa L2
    query_vector = model.encode([query], convert_to_numpy=True).astype('float32')
    query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
    query_norm = np.where(query_norm == 0, 1.0, query_norm)
    query_vector = query_vector / query_norm
    
    # 2. Tìm kiếm trên chỉ mục FAISS mô tả phim (Cosine Similarity)
    distances, indices = index.search(query_vector, top_k)
    
    # 3. Lấy các dòng tương ứng từ DataFrame
    valid_indices = [idx for idx in indices[0] if 0 <= idx < len(df)]
    if not valid_indices:
        return pd.DataFrame()
        
    # Tạo bản sao và giữ đúng thứ tự truy xuất
    retrieved_df = df.iloc[valid_indices].copy()
    retrieved_df['similarity'] = distances[0][:len(valid_indices)]
    
    return retrieved_df

def run_traditional_rag_pipeline(
    query: str,
    llm: BaseChatModel,
    df: pd.DataFrame,
    index: faiss.Index,
    model,
    top_k: int = 5
) -> tuple[str, pd.DataFrame]:
    """
    Chạy luồng RAG truyền thống: Retrieval (mô tả) -> Generation.
    """
    # 1. Retrieval
    retrieved_df = retrieve_traditional(query, index, model, df, top_k=top_k)
    
    if retrieved_df.empty:
        return "Xin lỗi, tôi không tìm thấy bộ phim nào phù hợp trong cơ sở dữ liệu.", retrieved_df
        
    # 2. Định dạng ngữ cảnh (Context) cho prompt
    movies_info_list = []
    for _, row in retrieved_df.iterrows():
        movie_str = f"- {row.get('final_context')}\n"
        if COL_LINK in row and pd.notna(row[COL_LINK]):
            movie_str += f"  Link IMDb: {row[COL_LINK]}\n"
        movies_info_list.append(movie_str)
        
    movies_info = "\n".join(movies_info_list)
    
    # 3. Tạo Prompt & gọi LLM qua chain (đúng cách để system + human messages được gửi đúng)
    prompt_template = get_rag_prompt()
    chain = prompt_template | llm
    
    try:
        response = chain.invoke({
            "input": query,
            "movies_info": movies_info
        })
        answer_result = response.content.strip()
        # Loại bỏ thinking tags của một số model (ví dụ: <think>...</think>)
        if "<think>" in answer_result and "</think>" in answer_result:
            think_end = answer_result.rfind("</think>")
            answer_result = answer_result[think_end + len("</think>"):].strip()
    except Exception as e:
        answer_result = f"Lỗi khi gọi LLM trong Traditional RAG: {e}"
        
    return answer_result, retrieved_df
