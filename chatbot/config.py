import os

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN FILE
# ============================================================
CHATBOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CHATBOT_DIR)

MOVIE_DATA_PATH = os.path.join(BASE_DIR, "movie_master", "movie_master.csv")
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

# ============================================================
# CẤU HÌNH LLM (LOCAL & GEMINI)
# ============================================================
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:20128/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY",  "any")
LLM_MODEL    = os.getenv("LLM_MODEL",    "cx/gpt-5.5")

GEMINI_DEFAULT_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"

# ============================================================
# THAM SỐ TÌM KIẾM (RAG)
# ============================================================
SEMANTIC_TOP_K = 150       # Số lượng kết quả lấy từ FAISS trước khi lọc Pandas
BM25_TOP_K = 100           # Số lượng kết quả lấy từ BM25
FINAL_TOP_K = 5            # Số lượng phim tối đa hiển thị cho người dùng
MIN_VOTES_THRESHOLD = 1000 # Số lượt vote tối thiểu để đảm bảo chất lượng phim lọc
