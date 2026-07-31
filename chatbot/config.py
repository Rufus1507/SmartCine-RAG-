import os

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN FILE
# ============================================================
CHATBOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CHATBOT_DIR)

MOVIE_DATA_PATH = os.path.join(BASE_DIR, "data", "cinebot_movies.parquet")
FEEDBACK_LOG_PATH = os.path.join(BASE_DIR, "data", "feedback", "feedback_log.jsonl")
KEYWORD_DICT_PATH = os.path.join(CHATBOT_DIR, "keyword_dict.json")
ALIASES_PATH = os.path.join(CHATBOT_DIR, "aliases.json")
INDEX_PATH = os.path.join(CHATBOT_DIR, "description_embeddings.index")
PROFILE_INDEX_PATH = os.path.join(CHATBOT_DIR, "movie_profile_embeddings.index")

# ============================================================
# MAPPING CỘT DỮ LIỆU
# ============================================================
COL_TITLE    = "Title"               # Tên phim
COL_GENRE    = "genres"              # Thể loại
COL_DIRECTOR = "directors"           # Đạo diễn
COL_STARS    = "stars"               # Danh sách diễn viên
COL_YEAR     = "Year"                # Năm phát hành
COL_RATING   = "Rating"              # Điểm IMDB
COL_OVERVIEW = "description"         # Mô tả phim
COL_LINK     = "Movie Link"          # Link phim (IMDB Link)

# Các cột thuộc tính mới từ movie_master.csv
COL_OSCAR      = "has_oscar"         # Đạt giải Oscar (1/0)
COL_AWARDS     = "has_awards"        # Đạt giải thưởng bất kỳ (1/0)
COL_NOMINATION = "has_nomination"    # Có đề cử (1/0)
COL_DURATION   = "duration_min"      # Thời lượng phim (phút)
COL_METASCORE  = "meta_score"        # Điểm Metacritic

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    load_dotenv()
except ImportError:
    pass

# ============================================================
# CẤU HÌNH LLM (LOCAL, OLLAMA & GEMINI)
# ============================================================
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:20128/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY",  "any")
LLM_MODEL    = os.getenv("LLM_MODEL",    "cx/gpt-5.3-codex")

# Cấu hình bổ sung cho Ollama Server
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY  = os.getenv("OLLAMA_API_KEY",  "any")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "qwen3.5:4b-q4_K_M")

GEMINI_DEFAULT_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_RATE_LIMIT_RPM = int(os.getenv("GEMINI_RATE_LIMIT_RPM", "12"))  # Dư 3 so với 15 RPM free tier

def update_env_variable(key: str, value: str):
    """
    Cập nhật biến môi trường trong os.environ và ghi đồng bộ vào file .env
    để toàn bộ hệ thống và các script khác sử dụng API Key/Cấu hình mới.
    """
    if value is None:
        value = ""
    os.environ[key] = value
    env_path = os.path.join(BASE_DIR, ".env")
    
    lines = []
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            lines = []
            
    found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}\n")
            found = True
        else:
            new_lines.append(line)
            
    if not found:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        new_lines.append(f"{key}={value}\n")
        
    try:
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    except Exception:
        pass


# ============================================================
# THAM SỐ TÌM KIẾM (RAG)
# ============================================================
SEMANTIC_TOP_K = 150       # Số lượng kết quả lấy từ FAISS trước khi lọc Pandas
BM25_TOP_K = 100           # Số lượng kết quả lấy từ BM25
FINAL_TOP_K = 5            # Số lượng phim tối đa hiển thị cho người dùng
MIN_VOTES_THRESHOLD = 0    # Không loại phim theo vote; giữ hằng số để tương thích ngược
