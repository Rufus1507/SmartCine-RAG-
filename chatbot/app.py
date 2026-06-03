import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import re
import os
import ast
from rapidfuzz import process, fuzz
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import faiss
import torch
from sentence_transformers import SentenceTransformer

# ============================================================
# CẤU HÌNH MẶC ĐỊNH
# ============================================================
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:20128/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY",  "any")   # local server không cần key thật
LLM_MODEL    = os.getenv("LLM_MODEL",    "cx/gpt-5.5")

GEMINI_DEFAULT_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"

# Xác định đường dẫn file dựa trên thư mục chứa file app.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMDB_DATA_PATH = os.path.join(BASE_DIR, "data", "imdb_movies_all_years.csv")
ADVANCED_DATA_PATH = os.path.join(BASE_DIR, "data", "advanced_movies_details_all_years.csv")
KEYWORD_DICT_PATH = os.path.join(BASE_DIR, "chatbot", "keyword_dict.json")
ALIASES_PATH = os.path.join(BASE_DIR, "chatbot", "aliases.json")
INDEX_PATH = os.path.join(BASE_DIR, "chatbot", "description_embeddings.index")

# Mapping tên cột trong CSV của nhóm
COL_TITLE    = "Title"               # tên phim
COL_GENRE    = "genres"              # thể loại
COL_DIRECTOR = "directors"           # đạo diễn
COL_STARS    = "stars"               # danh sách diễn viên
COL_YEAR     = "Year"                # năm phát hành
COL_RATING   = "Rating"              # điểm IMDB
COL_OVERVIEW = "description"         # mô tả phim
COL_LINK     = "Movie Link"          # link phim (IMDB Link)

# ============================================================
# LOAD DỮ LIỆU
# ============================================================
@st.cache_data
def load_data():
    # Load dữ liệu từ 2 file CSV sử dụng encoding latin-1 để tránh lỗi giải mã ký tự đặc biệt
    try:
        df1 = pd.read_csv(IMDB_DATA_PATH, encoding='latin-1', low_memory=False)
        df1.columns = df1.columns.str.replace(r'^\xef\xbb\xbf', '', regex=True)
    except Exception:
        df1 = pd.read_csv(IMDB_DATA_PATH, low_memory=False)
        
    try:
        df2 = pd.read_csv(ADVANCED_DATA_PATH, encoding='latin-1', low_memory=False)
        df2.columns = df2.columns.str.replace(r'^\xef\xbb\xbf', '', regex=True)
    except Exception:
        df2 = pd.read_csv(ADVANCED_DATA_PATH, low_memory=False)
        
    # Merge 2 bảng lại thông qua link phim
    df = pd.merge(df1, df2, left_on="Movie Link", right_on="link", how="inner")
    
    # Làm sạch các cột chứa dữ liệu dạng list string (genres, directors, stars)
    def clean_list_column(val):
        if pd.isna(val):
            return ""
        val_str = str(val).strip()
        if not val_str:
            return ""
        if val_str.startswith("[") and val_str.endswith("]"):
            try:
                lst = ast.literal_eval(val_str)
                lst = [item for item in lst if item and item != "None"]
                return ", ".join(lst)
            except Exception:
                pass
        return val_str
        
    df[COL_GENRE] = df[COL_GENRE].apply(clean_list_column)
    df[COL_DIRECTOR] = df[COL_DIRECTOR].apply(clean_list_column)
    df[COL_STARS] = df[COL_STARS].apply(clean_list_column)
    
    # Chuẩn hoá kiểu dữ liệu
    df[COL_RATING] = pd.to_numeric(df[COL_RATING], errors="coerce")
    df[COL_YEAR]   = pd.to_numeric(df[COL_YEAR],   errors="coerce")
    
    # Chuẩn hoá số lượng votes (xử lý các ký tự 'K', 'M' và dấu phẩy)
    def clean_votes(val):
        if pd.isna(val):
            return 0
        val_str = str(val).strip().upper()
        if not val_str:
            return 0
        try:
            if val_str.endswith('K'):
                return int(float(val_str[:-1]) * 1000)
            elif val_str.endswith('M'):
                return int(float(val_str[:-1]) * 1000000)
            val_str = val_str.replace(',', '')
            return int(float(val_str))
        except Exception:
            return 0
            
    df['num_votes'] = df['Votes'].apply(clean_votes)
    return df

@st.cache_data
def load_keyword_dict():
    with open(KEYWORD_DICT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_aliases():
    try:
        with open(ALIASES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@st.cache_resource
def load_faiss_index():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Chỉ mục FAISS chưa được tạo.")
    return faiss.read_index(INDEX_PATH)

@st.cache_resource
def load_embedder_model():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Chỉ mục FAISS chưa được tạo.")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

def semantic_search(query: str, df: pd.DataFrame, model, index, k=100) -> pd.DataFrame:
    """
    Tìm kiếm ngữ nghĩa (semantic search) trên cột description của DataFrame.
    Trả về DataFrame con chứa top K kết quả phù hợp nhất.
    """
    try:
        q_emb = model.encode([query]).astype('float32')
        _, indices = index.search(q_emb, k)
        # Lọc ra các chỉ mục hợp lệ nằm trong phạm vi dòng của DataFrame
        valid_indices = [idx for idx in indices[0] if 0 <= idx < len(df)]
        return df.iloc[valid_indices].copy()
    except Exception:
        return df.copy()

def detect_entities(user_message: str, keyword_dict: dict, aliases_dict: dict) -> dict:
    """
    Quét qua câu hỏi của user để phát hiện các thực thể có trong keyword_dict.json.
    Sử dụng aliases_dict để chuyển đổi tên viết tắt/bí danh.
    Sử dụng rapidfuzz để fuzzy match các từ viết sai chính tả.
    """
    user_msg_lower = user_message.lower()
    detected = {
        "genres": [],
        "directors": [],
        "stars": [],
        "writers": [],
        "content_keywords": []
    }
    
    # Danh sách các từ phổ biến cần bỏ qua khi fuzzy match để tránh khớp nhầm
    IGNORE_FUZZY = {
        # Unaccented
        "phim", "tim", "kiem", "cho", "xem", "dao", "dien", "vien", "the", "loai", 
        "nha", "nam", "diem", "tren", "duoi", "tuyen", "hay", "nhat", "co", "ve", 
        "chieu", "rap", "bom", "tan", "le", "bo", "my", "han", "trung", "viet", "nhat",
        "cua", "cac", "nhung", "la", "va", "hoac", "trong", "ngoai", "cao", "thap",
        
        # Accented (Tiếng Việt có dấu)
        "tìm", "kiếm", "đạo", "diễn", "viên", "thể", "loại", "nhà", "năm", "điểm",
        "trên", "dưới", "tuyển", "có", "về", "chiếu", "rạp", "bộ", "mỹ", "hàn", 
        "của", "các", "những", "là", "và", "hoặc", "viễn", "tưởng", "khoa", "học"
    }
    
    # Làm sạch câu hỏi, giữ lại chữ và số để tách từ
    words = re.findall(r'\b\w+\b', user_msg_lower)
    n = len(words)
    
    # Tạo tất cả các n-gram từ độ dài 1 đến 5
    candidates = []
    for length in range(1, min(6, n + 1)):
        for i in range(n - length + 1):
            ngram = " ".join(words[i:i+length])
            candidates.append(ngram)
            
    # Lọc trùng và sắp xếp n-gram theo độ dài giảm dần
    candidates = sorted(list(set(candidates)), key=len, reverse=True)
    
    # Lưu lại danh sách các key của từ điển để phục vụ fuzzy lookup
    keyword_keys_list = list(keyword_dict.keys())
    
    matched_keys = set()
    for candidate in candidates:
        if any(candidate in already_matched for already_matched in matched_keys):
            continue
            
        target_key = None
        intent = None
        
        # 1. Kiểm tra trong aliases_dict
        if candidate in aliases_dict:
            resolved_key = aliases_dict[candidate]
            if resolved_key in keyword_dict:
                target_key = resolved_key
                intent = keyword_dict[resolved_key]
                
        # 2. Kiểm tra khớp chính xác trong keyword_dict
        elif candidate in keyword_dict:
            target_key = candidate
            intent = keyword_dict[candidate]
            
        # 3. Fuzzy match bằng rapidfuzz (cho các cụm từ có độ dài >= 5 và không chứa stopwords)
        elif len(candidate) >= 5:
            candidate_words = set(candidate.split())
            if candidate_words.isdisjoint(IGNORE_FUZZY):
                # Dùng QRatio để chạy cực nhanh và chính xác cho lỗi gõ sai
                match, score, _ = process.extractOne(
                    candidate, keyword_keys_list,
                    scorer=fuzz.QRatio
                )
                if score >= 80:  # Ngưỡng tin cậy 80%
                    target_key = match
                    intent = keyword_dict[match]
                
        # Phân loại intent tìm được vào bộ lọc tương ứng
        if target_key and intent:
            matched_keys.add(candidate)
            matched_keys.add(target_key)
            # Tránh khớp lại các từ đơn lẻ nằm trong cụm từ này
            for word in candidate.split():
                matched_keys.add(word)
            for word in target_key.split():
                matched_keys.add(word)
            
            if intent == "search_genre":
                detected["genres"].append(target_key)
            elif intent == "search_director":
                detected["directors"].append(target_key)
            elif intent == "search_star":
                detected["stars"].append(target_key)
            elif intent == "search_writer":
                # Do trong dữ liệu nhiều đạo diễn/diễn viên cũng là tác giả biên kịch,
                # ta ánh xạ vào cả director và star để LLM/Bộ lọc so khớp chính xác nhất.
                detected["directors"].append(target_key)
                detected["stars"].append(target_key)
            elif intent == "search_content":
                detected["content_keywords"].append(target_key)
                
    return detected

def is_refine_query(user_input: str) -> bool:
    """
    Xác định xem câu hỏi của người dùng có phải là câu nối tiếp/tinh chỉnh (refinement) hay không.
    Kiểm tra các từ nối, từ chỉ thời gian hoặc đại từ chỉ ngữ cảnh trong Tiếng Việt.
    """
    refine_keywords = {
        "nhưng", "nhung", "chỉ", "chi", "thêm", "them", "nữa", "nua", "còn", "con", 
        "khác", "khac", "đó", "do", "này", "nay", "ông ấy", "ong ay", "bà ấy", "ba ay",
        "họ", "ho", "sau", "trước", "truoc", "hơn", "hon", "dưới", "duoi", "trên", "tren"
    }
    words = set(re.findall(r'\b\w+\b', user_input.lower()))
    return not words.isdisjoint(refine_keywords)

# ============================================================
# PROMPT — TẦNG 1 (PARSE INTENT)
# ============================================================
SYSTEM_PROMPT = f"""
Bạn là bộ phân tích câu hỏi cho một chatbot phim.
Nhiệm vụ: đọc câu hỏi của người dùng và trả về JSON hợp lệ DUY NHẤT,
không có bất kỳ văn bản nào khác ngoài JSON.

Schema JSON:
{{
  "intent": "search" | "recommend" | "info" | "chitchat",
  "filters": {{
    "title":    <string hoặc null>,
    "genre":    <string hoặc null>,
    "director": <string hoặc null>,
    "star":     <string hoặc null>,
    "year_min": <int hoặc null>,
    "year_max": <int hoặc null>,
    "rating_min": <float hoặc null>
  }},
  "free_text": <câu hỏi gốc của user, dùng cho chitchat>
}}

Ví dụ:
User: "Tìm phim hành động của Christopher Nolan trên 8 điểm"
JSON: {{"intent":"search","filters":{{"title":null,"genre":"Action","director":"Christopher Nolan","star":null,"year_min":null,"year_max":null,"rating_min":8.0}},"free_text":"Tìm phim hành động của Christopher Nolan trên 8 điểm"}}

User: "Phim nào hay nhất năm 2020?"
JSON: {{"intent":"search","filters":{{"title":null,"genre":null,"director":null,"star":null,"year_min":2020,"year_max":2020,"rating_min":null}},"free_text":"Phim nào hay nhất năm 2020?"}}

User: "Xin chào"
JSON: {{"intent":"chitchat","filters":{{"title":null,"genre":null,"director":null,"star":null,"year_min":null,"year_max":null,"rating_min":null}},"free_text":"Xin chào"}}
"""

# ============================================================
# PYDANTIC SCHEMA CHO INTENT PARSING
# ============================================================
class Filters(BaseModel):
    title: Optional[str] = None
    genre: Optional[str] = None
    director: Optional[str] = None
    star: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    rating_min: Optional[float] = None

    @field_validator('year_min', 'year_max', mode='before')
    @classmethod
    def coerce_year(cls, v):
        if v is None or v == "":
            return None
        try:
            # Ví dụ: "năm 2020" -> 2020, "sau năm 2010" -> 2010
            clean_val = str(v).lower().replace("năm", "").strip()
            match = re.search(r'\d+', clean_val)
            if match:
                return int(match.group(0))
            return int(clean_val)
        except Exception:
            return None

    @field_validator('rating_min', mode='before')
    @classmethod
    def coerce_rating(cls, v):
        if v is None or v == "":
            return None
        try:
            return float(v)
        except Exception:
            return None

class ParsedIntent(BaseModel):
    intent: str = "chitchat"
    filters: Filters = Field(default_factory=Filters)
    free_text: str = ""

# ============================================================
# UNIFIED LLM CALLER
# ============================================================
def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """
    Gọi LLM dựa trên cấu hình nhà cung cấp được chọn trong Session State.
    Hỗ trợ Local LLM và Gemini API.
    """
    provider = st.session_state.get("llm_provider", "Local LLM")
    
    if provider == "Gemini API":
        from google import genai
        from google.genai import types
        
        api_key = st.session_state.get("gemini_api_key", "")
        model_name = st.session_state.get("gemini_model", "gemini-2.5-flash")
        
        if not api_key:
            raise ValueError("Vui lòng cấu hình Gemini API Key trong Sidebar.")
            
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        return response.text.strip()
    else:
        # Local LLM fallback
        base_url = st.session_state.get("local_base_url", LLM_BASE_URL)
        api_key = st.session_state.get("local_api_key", LLM_API_KEY)
        model_name = st.session_state.get("local_model", LLM_MODEL)
        
        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

# ============================================================
# LLM — TẦNG 1: PARSE INTENT → JSON
# ============================================================
def parse_intent(user_message: str, detected_entities: dict = None, chat_history: list = None) -> dict:
    # Bổ sung thông tin gợi ý thực thể vào prompt nếu có
    hints = ""
    if detected_entities:
        hints = "\nGỢI Ý THỰC THỂ TÌM THẤY TRONG CƠ SỞ DỮ LIỆU:\n"
        if detected_entities.get("genres"):
            hints += f"- Thể loại gợi ý: {', '.join(detected_entities['genres'])}\n"
        if detected_entities.get("directors"):
            hints += f"- Đạo diễn gợi ý: {', '.join(detected_entities['directors'])}\n"
        if detected_entities.get("stars"):
            hints += f"- Diễn viên gợi ý: {', '.join(detected_entities['stars'])}\n"
            
    # Bổ sung lịch sử chat vào prompt để giải quyết liên kết ngữ cảnh (co-reference resolution)
    history_str = ""
    if chat_history:
        # Lấy lịch sử trước tin nhắn hiện tại (loại bỏ tin nhắn cuối cùng vì đó chính là tin nhắn hiện tại)
        # Giới hạn lấy tối đa 6 tin nhắn gần nhất (tương đương 3 lượt hỏi - đáp)
        relevant_history = chat_history[:-1][-6:]
        history_lines = []
        for msg in relevant_history:
            role_label = "User" if msg["role"] == "user" else "Bot"
            content = msg["content"]
            if len(content) > 150:
                content = content[:150] + "..."
            history_lines.append(f"{role_label}: {content}")
        
        if history_lines:
            history_str = "\nLỊCH SỬ TRÒ CHUYỆN GẦN ĐÂY:\n" + "\n".join(history_lines) + "\n"

    system_content = SYSTEM_PROMPT + hints + history_str

    parsed = None
    try:
        raw = call_llm(system_content, user_message, temperature=0.1, max_tokens=300)
        # Tách JSON ra khỏi markdown code block nếu có
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        parsed_dict = json.loads(raw)
        # Validate và ép kiểu chặt chẽ bằng Pydantic
        validated = ParsedIntent(**parsed_dict)
        parsed = validated.model_dump()
    except Exception as e:
        st.error(f"Lỗi gọi/phân tích LLM (Tầng 1 - Parse Intent): {e}")
        parsed = {
            "intent": "chitchat",
            "filters": {},
            "free_text": user_message
        }
        
    # Hậu xử lý: Phục hồi bộ lọc từ keyword_dict nếu LLM bỏ sót hoặc lỗi kết nối
    intent = parsed.get("intent", "chitchat")
    
    # Nếu phát hiện thực thể liên quan tới phim nhưng LLM phân loại nhầm là chitchat, tự chuyển sang search
    has_entities = detected_entities and any(detected_entities.get(k) for k in [
        "genres", "directors", "stars", "content_keywords"
    ])
    if has_entities and intent == "chitchat":
        parsed["intent"] = "search"
        intent = "search"
        
    if detected_entities and intent in ("search", "recommend", "info"):
        filters = parsed.setdefault("filters", {})
        if detected_entities.get("genres") and not filters.get("genre"):
            filters["genre"] = detected_entities["genres"][0]
        if detected_entities.get("directors") and not filters.get("director"):
            filters["director"] = detected_entities["directors"][0]
        if detected_entities.get("stars") and not filters.get("star"):
            filters["star"] = detected_entities["stars"][0]
            
    return parsed

# ============================================================
# PANDAS FILTER — áp dụng bộ lọc từ JSON
# ============================================================
def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    result = df.copy()

    # Loại bỏ các phim có quá ít lượt vote để tránh kết quả rác (phim vô danh điểm 10)
    # Không áp dụng nếu người dùng đang tìm đích danh tên phim (title)
    if 'num_votes' in result.columns and not filters.get("title"):
        result = result[result['num_votes'] >= 1000]

    if filters.get("genre"):
        result = result[result[COL_GENRE].str.contains(
            filters["genre"], case=False, na=False
        )]
    if filters.get("director"):
        result = result[result[COL_DIRECTOR].str.contains(
            filters["director"], case=False, na=False
        )]
    if filters.get("star"):
        result = result[result[COL_STARS].str.contains(
            filters["star"], case=False, na=False
        )]
    if filters.get("title"):
        result = result[result[COL_TITLE].str.contains(
            filters["title"], case=False, na=False
        )]
    if filters.get("year_min"):
        result = result[result[COL_YEAR] >= filters["year_min"]]
    if filters.get("year_max"):
        result = result[result[COL_YEAR] <= filters["year_max"]]
    if filters.get("rating_min"):
        result = result[result[COL_RATING] >= filters["rating_min"]]

    # Sắp xếp theo rating giảm dần, lấy top 5
    result = result.sort_values(COL_RATING, ascending=False).head(5)
    return result

# ============================================================
# LLM — TẦNG 2: GÓI KẾT QUẢ THÀNH CÂU TRẢ LỜI TỰ NHIÊN
# ============================================================
def generate_answer(user_message: str, movies_df: pd.DataFrame, intent: str) -> str:
    if intent == "chitchat":
        system_msg = "Bạn là trợ lý phim thân thiện. Trả lời bằng tiếng Việt, ngắn gọn, tự nhiên."
        user_msg   = user_message
    elif movies_df.empty:
        system_msg = "Bạn là trợ lý phim thân thiện. Trả lời bằng tiếng Việt."
        if intent == "info":
            user_msg = (
                f"Người dùng hỏi: \"{user_message}\"\n"
                "Không xác định được bộ phim cụ thể nào trong cơ sở dữ liệu để cung cấp thông tin. "
                "Hãy trả lời thân thiện và lịch sự hỏi người dùng muốn biết thông tin của bộ phim nào."
            )
        else:
            user_msg = (
                f"Người dùng hỏi: \"{user_message}\"\n"
                "Không tìm thấy phim nào phù hợp trong cơ sở dữ liệu. "
                "Hãy trả lời thân thiện, gợi ý họ thử tìm kiếm với tiêu chí khác."
            )
    else:
        movies_info = movies_df[[COL_TITLE, COL_GENRE, COL_DIRECTOR, COL_STARS, COL_YEAR, COL_RATING]].to_string(index=False)
        system_msg = "Bạn là trợ lý phim thân thiện. Trả lời bằng tiếng Việt, thân thiện và tự nhiên. Không bịa thêm thông tin."
        user_msg   = (
            f"Người dùng hỏi: \"{user_message}\"\n"
            f"Danh sách phim tìm được:\n{movies_info}\n\n"
            "Hãy giới thiệu các phim này, đề cập tên phim, thể loại, đạo diễn, diễn viên và điểm IMDB."
        )

    try:
        return call_llm(system_msg, user_msg, temperature=0.7, max_tokens=1024)
    except Exception as e:
        # Fallback response listing the movies in case of LLM failure
        if not movies_df.empty:
            movie_list = []
            for _, row in movies_df.head(5).iterrows():
                movie_list.append(f"- {row[COL_TITLE]} ({int(row[COL_YEAR]) if pd.notna(row[COL_YEAR]) else 'N/A'}) - ⭐ {row[COL_RATING]}")
            movies_str = "\n".join(movie_list)
            return (
                f"Chào bạn! Kết nối với trí tuệ nhân tạo đang gặp sự cố nhỏ ({e}), "
                f"nhưng tôi đã tìm trực tiếp trong cơ sở dữ liệu và thấy các phim phù hợp này:\n\n{movies_str}"
            )
        return f"Lỗi gọi LLM (Tầng 2 - Sinh câu trả lời): {e}"

# ============================================================
# HIỂN THỊ KẾT QUẢ PHIM
# ============================================================
def render_movie_cards(df: pd.DataFrame):
    if df.empty:
        return
    cols = st.columns(min(len(df), 5))
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i]:
            st.markdown(f"**{row[COL_TITLE]}**")
            st.caption(f"⭐ {row[COL_RATING]}  •  {int(row[COL_YEAR]) if pd.notna(row[COL_YEAR]) else 'N/A'}")
            st.caption(f"🎬 Đạo diễn: {row[COL_DIRECTOR]}")
            st.caption(f"🎭 Thể loại: {row[COL_GENRE]}")
            if COL_STARS in row and row[COL_STARS]:
                # Hiển thị tối đa 3 diễn viên đầu để card gọn gàng
                stars_list = [s.strip() for s in row[COL_STARS].split(",")]
                st.caption(f"👥 Diễn viên: {', '.join(stars_list[:3])}")

# ============================================================
# STREAMLIT APP
# ============================================================
st.set_page_config(
    page_title="🎬 CineBot",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 CineBot — Chatbot Tìm Phim Thông Minh")
st.caption("Hỏi bất kỳ điều gì về phim: thể loại, đạo diễn, diễn viên, năm, điểm IMDB...")

# --- Sidebar: cấu hình ---
with st.sidebar:
    st.header("⚙️ Cấu hình")

    # Chọn nhà cung cấp LLM
    llm_provider = st.selectbox(
        "Nhà cung cấp LLM",
        ["Local LLM", "Gemini API"],
        index=0,
        key="llm_provider"
    )

    if llm_provider == "Gemini API":
        st.info("🔌 Sử dụng Google Gemini API")
        gemini_api_key = st.text_input(
            "Gemini API Key",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            key="gemini_api_key"
        )
        gemini_model = st.selectbox(
            "Chọn Model Gemini",
            ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash-001"],
            index=0,
            key="gemini_model"
        )
    else:
        st.info(f"🔌 Kết nối Local Endpoint")
        local_base_url = st.text_input(
            "Endpoint URL",
            value=LLM_BASE_URL,
            key="local_base_url"
        )
        local_api_key = st.text_input(
            "API Key (nếu có)",
            value=LLM_API_KEY,
            type="password",
            key="local_api_key"
        )
        model_options = ["cx/gpt-5.5", "cx/gpt-5.4", "cx/gpt-5.3-codex", "cx/gpt-5.3-codex-high"]
        local_model = st.selectbox(
            "Chọn Model Local",
            model_options,
            index=0,
            key="local_model"
        )

    st.divider()
    st.markdown("**Ví dụ câu hỏi:**")
    examples = [
        "Phim hành động điểm trên 8",
        "Phim của Christopher Nolan",
        "Phim kinh dị sau năm 2010",
        "Phim có Leonardo DiCaprio",
        "Phim hay nhất năm 2019",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["pending_input"] = ex

    st.divider()
    st.markdown("**🔍 Debugging:**")
    if "last_parsed" in st.session_state:
        st.json(st.session_state.last_parsed)
    else:
        st.caption("Chưa có truy vấn nào được thực hiện.")

# --- Load data & keyword dict ---
try:
    df = load_data()
    st.sidebar.success(f"✅ Đã load {len(df):,} phim")
except FileNotFoundError:
    st.error(f"❌ Không tìm thấy các file dữ liệu CSV tại đường dẫn:\n"
             f"- `{IMDB_DATA_PATH}`\n"
             f"- `{ADVANCED_DATA_PATH}`\n"
             f"Vui lòng kiểm tra lại thư mục `merged_output`.")
    st.stop()

try:
    keyword_dict = load_keyword_dict()
    st.sidebar.success(f"✅ Đã load {len(keyword_dict):,} từ khóa")
except FileNotFoundError:
    st.error(f"❌ Không tìm thấy file `keyword_dict.json` tại đường dẫn:\n"
             f"- `{KEYWORD_DICT_PATH}`")
    st.stop()

try:
    aliases_dict = load_aliases()
    st.sidebar.success(f"✅ Đã load {len(aliases_dict):,} biệt danh")
except Exception:
    aliases_dict = {}

# --- Load FAISS index & SentenceTransformer model ---
try:
    faiss_index = load_faiss_index()
    embedder_model = load_embedder_model()
except Exception:
    faiss_index = None
    embedder_model = None

if faiss_index is not None and embedder_model is not None:
    st.sidebar.success("🔮 Đã kích hoạt Semantic Search (FAISS)")
else:
    st.sidebar.warning("⚠️ Chưa có file chỉ mục. Chạy `python .\\chatbot\\generate_embeddings.py` để kích hoạt Semantic Search!")

# --- Chat history & Memory ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Xin chào! Tôi là CineBot 🎬 Bạn muốn tìm phim gì hôm nay? Hãy hỏi tôi về thể loại, đạo diễn, diễn viên hoặc bất kỳ điều gì về phim nhé!",
        "movies": None
    })

if "last_filters" not in st.session_state:
    st.session_state.last_filters = {}

# --- Hiển thị lịch sử chat ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("movies") is not None and not msg["movies"].empty:
            render_movie_cards(msg["movies"])

# --- Input từ ví dụ (sidebar button) ---
pending = st.session_state.pop("pending_input", None)

# --- Chat input ---
user_input = st.chat_input("Nhập câu hỏi của bạn...") or pending

if user_input:
    # Hiển thị tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": user_input, "movies": None})
    with st.chat_message("user"):
        st.write(user_input)

    # Xử lý
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm..."):

            # Phát hiện thực thể từ câu hỏi bằng keyword_dict và aliases_dict
            detected = detect_entities(user_input, keyword_dict, aliases_dict)

            # Tầng 1: parse intent (có bổ trợ thực thể từ từ điển và lịch sử cuộc trò chuyện)
            parsed    = parse_intent(user_input, detected, st.session_state.messages)
            intent    = parsed.get("intent", "chitchat")
            
            # Xử lý tinh chỉnh bộ lọc (Conversation Memory)
            if (intent in ("search", "recommend") and is_refine_query(user_input)) or intent == "info":
                last_filters = st.session_state.get("last_filters", {})
                # Chỉ lấy các bộ lọc mới khác None để tránh xóa thông tin cũ
                new_filters = {k: v for k, v in parsed.get("filters", {}).items() if v is not None}
                # Hợp nhất bộ lọc cũ và mới
                merged_filters = {**last_filters, **new_filters}
                parsed["filters"] = merged_filters
                
            filters = parsed.get("filters", {})
            
            # Tự động sửa lỗi bộ lọc (Auto-correction)
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
                        
            # Lưu lại bộ lọc của lượt này làm tiền đề cho lượt tiếp theo
            st.session_state.last_filters = filters

            # Lưu thông tin debug để hiển thị ở Sidebar
            st.session_state.last_parsed = {
                "user_input": user_input,
                "intent": parsed.get("intent"),
                "filters": filters,
                "detected": detected
            }

            # Lọc phim (Kết hợp Semantic Search và Metadata Filters)
            if intent in ("search", "recommend", "info"):
                has_metadata_filters = any(filters.get(k) for k in [
                    "genre", "director", "star", "title", "year_min", "year_max", "rating_min"
                ])

                if intent == "info" and not has_metadata_filters:
                    # Nếu intent là info nhưng không có bộ lọc thuộc tính nào, trả về DataFrame rỗng để chatbot hỏi làm rõ
                    filtered_df = pd.DataFrame()
                elif has_metadata_filters:
                    # Nếu user hỏi rõ đạo diễn / diễn viên / thể loại / năm / điểm
                    # thì lọc trực tiếp trên toàn bộ dataset
                    filtered_df = apply_filters(df, filters)

                elif faiss_index is not None and embedder_model is not None:
                    # Chỉ dùng semantic search khi user hỏi kiểu mô tả nội dung
                    filtered_df = semantic_search(user_input, df, embedder_model, faiss_index, k=100)
                    filtered_df = apply_filters(filtered_df, filters)

                else:
                    filtered_df = apply_filters(df, filters)
            else:
                filtered_df = pd.DataFrame()

            # Tầng 2: sinh câu trả lời
            answer = generate_answer(user_input, filtered_df, intent)

        st.write(answer)
        if not filtered_df.empty:
            render_movie_cards(filtered_df)

    # Lưu vào history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "movies": filtered_df if not filtered_df.empty else None
    })
