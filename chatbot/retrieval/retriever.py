import pandas as pd
from chatbot.config import SEMANTIC_TOP_K

def semantic_search_retriever(query: str, df: pd.DataFrame, index, model, top_k: int = SEMANTIC_TOP_K) -> pd.DataFrame:
    """
    Thực hiện tìm kiếm ngữ nghĩa trên trường mô tả phim (description) sử dụng FAISS index.
    Đầu ra trả về DataFrame chứa danh sách các phim phù hợp nhất với câu hỏi.
    """
    try:
        if df.empty or index is None or model is None or not query:
            return df.copy()
            
        # Sinh vector nhúng cho câu hỏi
        q_emb = model.encode([query]).astype('float32')
        
        # Tìm kiếm trong FAISS
        _, indices = index.search(q_emb, top_k)
        
        # Lọc lấy các chỉ số hợp lệ trong DataFrame
        valid_indices = [idx for idx in indices[0] if 0 <= idx < len(df)]
        return df.iloc[valid_indices].copy()
    except Exception:
        return df.copy()
