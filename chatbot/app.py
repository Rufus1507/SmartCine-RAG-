import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import json
import re
import os
import ast

# ============================================================
# CẤU HÌNH
# ============================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")   # Nhập API key mới tại đây hoặc qua sidebar
GEMINI_MODEL   = "gemini-1.5-flash"                  # Model mặc định

# Xác định đường dẫn file dựa trên thư mục chứa file app.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMDB_DATA_PATH = os.path.join(BASE_DIR, "data", "imdb_movies_all_years.csv")
ADVANCED_DATA_PATH = os.path.join(BASE_DIR, "data", "advanced_movies_details_all_years.csv")

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
        df1 = pd.read_csv(IMDB_DATA_PATH, encoding='latin-1')
        df1.columns = df1.columns.str.replace(r'^\xef\xbb\xbf', '', regex=True)
    except Exception:
        df1 = pd.read_csv(IMDB_DATA_PATH)
        
    try:
        df2 = pd.read_csv(ADVANCED_DATA_PATH, encoding='latin-1')
        df2.columns = df2.columns.str.replace(r'^\xef\xbb\xbf', '', regex=True)
    except Exception:
        df2 = pd.read_csv(ADVANCED_DATA_PATH)
        
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
    return df

# ============================================================
# PROMPT GỬI CHO GEMINI — TẦNG 1 (PARSE INTENT)
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
# GEMINI — TẦNG 1: PARSE INTENT → JSON
# ============================================================
def parse_intent(user_message: str) -> dict:
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
        )
    )
    raw = response.text.strip()

    # Tách JSON ra khỏi markdown code block nếu có
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "intent": "chitchat",
            "filters": {},
            "free_text": user_message
        }

# ============================================================
# PANDAS FILTER — áp dụng bộ lọc từ JSON
# ============================================================
def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    result = df.copy()

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
# GEMINI — TẦNG 2: GÓI KẾT QUẢ THÀNH CÂU TRẢ LỜI TỰ NHIÊN
# ============================================================
def generate_answer(user_message: str, movies_df: pd.DataFrame, intent: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)

    if intent == "chitchat":
        prompt = f"""
Bạn là trợ lý phim thân thiện. Trả lời câu sau bằng tiếng Việt, ngắn gọn, tự nhiên.
Câu: {user_message}
"""
    elif movies_df.empty:
        prompt = f"""
Người dùng hỏi: "{user_message}"
Không tìm thấy phim nào phù hợp trong cơ sở dữ liệu.
Hãy trả lời thân thiện bằng tiếng Việt, gợi ý họ thử tìm kiếm với tiêu chí khác.
"""
    else:
        movies_info = movies_df[[COL_TITLE, COL_GENRE, COL_DIRECTOR, COL_STARS, COL_YEAR, COL_RATING]].to_string(index=False)
        prompt = f"""
Người dùng hỏi: "{user_message}"
Dưới đây là danh sách phim tìm được từ cơ sở dữ liệu:

{movies_info}

Hãy giới thiệu các phim này bằng tiếng Việt, thân thiện và tự nhiên.
Đề cập tên phim, thể loại, đạo diễn, dàn diễn viên chính (stars) và điểm IMDB. Không bịa thêm thông tin.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return response.text.strip()

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
    api_key_input = st.text_input("Gemini API Key", type="password", value=GEMINI_API_KEY,
                                   placeholder="Nhập API key tại https://aistudio.google.com/apikey")
    if api_key_input:
        GEMINI_API_KEY = api_key_input

    model_options = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.5-flash-preview-05-20"]
    GEMINI_MODEL = st.selectbox("Model Gemini", model_options, index=0)
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

# --- Load data ---
try:
    df = load_data()
    st.sidebar.success(f"✅ Đã load {len(df):,} phim")
except FileNotFoundError:
    st.error(f"❌ Không tìm thấy các file dữ liệu CSV tại đường dẫn:\n"
             f"- `{IMDB_DATA_PATH}`\n"
             f"- `{ADVANCED_DATA_PATH}`\n"
             f"Vui lòng kiểm tra lại thư mục `merged_output`.")
    st.stop()

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Xin chào! Tôi là CineBot 🎬 Bạn muốn tìm phim gì hôm nay? Hãy hỏi tôi về thể loại, đạo diễn, diễn viên hoặc bất kỳ điều gì về phim nhé!",
        "movies": None
    })

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
            if not GEMINI_API_KEY:
                st.error("⚠️ Chưa có Gemini API Key. Vui lòng nhập vào sidebar.")
                st.stop()

            # Tầng 1: parse intent
            parsed    = parse_intent(user_input)
            intent    = parsed.get("intent", "chitchat")
            filters   = parsed.get("filters", {})

            # Lọc phim
            if intent in ("search", "recommend"):
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
