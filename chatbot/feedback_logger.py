import os
import json
import pandas as pd
from datetime import datetime
from chatbot.config import FEEDBACK_LOG_PATH

def log_feedback(
    session_id: str,
    turn_index: int,
    user_query: str,
    intent: str,
    filters: dict,
    route: str,
    bot_answer_preview: str,
    movie_titles_returned: list,
    rating: str,
    comment: str
):
    """
    Ghi một bản ghi phản hồi của người dùng vào tệp JSONL ở chế độ append.
    Tự động tạo thư mục chứa nếu chưa tồn tại.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "turn_index": turn_index,
        "user_query": user_query,
        "intent": intent,
        "filters": filters,
        "route": route,
        "bot_answer_preview": bot_answer_preview,
        "movie_titles_returned": movie_titles_returned,
        "rating": rating,
        "comment": comment
    }
    
    dir_path = os.path.dirname(FEEDBACK_LOG_PATH)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
        
    with open(FEEDBACK_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def get_feedback_csv_bytes() -> bytes | None:
    """
    Đọc tệp log JSONL, chuyển đổi thành DataFrame pandas và trả về dữ liệu dạng CSV bytes.
    Trả về None nếu tệp không tồn tại hoặc rỗng.
    """
    if not os.path.exists(FEEDBACK_LOG_PATH) or os.path.getsize(FEEDBACK_LOG_PATH) == 0:
        return None
        
    entries = []
    with open(FEEDBACK_LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
                    
    if not entries:
        return None
        
    df = pd.DataFrame(entries)
    return df.to_csv(index=False).encode('utf-8')
