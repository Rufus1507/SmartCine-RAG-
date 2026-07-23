import re
import unicodedata
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi

# Bản đồ dịch từ Việt sang Anh (Dùng cho cả Query và Corpus)
TRANSLATION_MAP = {
    "khoa học viễn tưởng": "sci-fi",
    "viễn tưởng": "sci-fi",
    "science fiction": "sci-fi",
    "hành động": "action",
    "hài hước": "comedy",
    "ca nhạc": "music",
    "âm nhạc": "music",
    "tình cảm": "romance",
    "lãng mạn": "romance",
    "hoạt hình": "animation",
    "phiêu lưu": "adventure",
    "tội phạm": "crime",
    "hình sự": "crime",
    "giật gân": "thriller",
    "thần thoại": "fantasy",
    "tài liệu": "documentary",
    "gia đình": "family",
    "chiến tranh": "war",
    "lịch sử": "history",
    "miền tây": "western",
    "nhật bản": "japan",
    "hàn quốc": "south korea",
    "united kingdom": "united kingdom",
    "nước anh": "united kingdom",
    "nước pháp": "france",
    "trung quốc": "china",
    "hồng kông": "hong kong",
    "hong kong": "hong kong",
    "đài loan": "taiwan",
    "ấn độ": "india",
    "nước đức": "germany",
    "nước ý": "italy",
    "tây ban nha": "spain",
    "thái lan": "thailand",
    "nước úc": "australia",
    "hoa kỳ": "united states",
    "hài": "comedy",
    "mỹ": "united states",
    "hàn": "south korea",
    "nhật": "japan",
    "anh": "united kingdom",
    "pháp": "france",
    "trung": "china",
    "đức": "germany",
    "ý": "italy",
    "úc": "australia",
    "thái": "thailand",
    "ma": "horror",
    "kinh dị": "horror",
    "kịch tính": "drama",
    "chính kịch": "drama",
    "tâm lý": "drama",
    "nhạc": "music",
    "nga": "russia",
    "gợi ý": "suggest",
    "giống như": "like",
    "giống": "like",
    "tương tự": "like",
    # Sửa lỗi mixed-language: dịch thêm các cụm từ tiếng Việt phổ biến
    # thường xuất hiện sau khi các từ genre/country đã được dịch
    "dành cho trẻ em": "for children",
    "trẻ em": "children",
    "dành cho": "for",
    "tình bạn": "friendship",
    "con người": "human",
    "người máy": "robot",
    "vũ trụ": "space",
    "siêu anh hùng": "superhero",
    "xuyên không": "time travel",
    "du hành thời gian": "time travel",
    "sống sót": "survival",
    "báo thù": "revenge",
    "phục thù": "revenge",
    "thanh thiếu niên": "teen",
    "lớn lên": "coming of age",
    "trưởng thành": "coming of age",
    "huyền bí": "mystery",
    "bí ẩn": "mystery",
    "thể thao": "sport",
    "lãng mạng": "romance",
    "tuổi thơ": "childhood",
    # Glossary bổ sung cho P1
    "kho báu bị chôn giấu": "buried treasure",
    "kho báu": "treasure",
    "bị chôn giấu": "buried",
    "chôn giấu": "buried",
    "hoang dã châu phi": "african wildlife",
    "động vật hoang dã": "wildlife",
    "hoang dã": "wildlife",
    "động vật": "animal",
    "châu phi": "africa",
    "ngoài hành tinh": "alien",
    "người ngoài hành tinh": "alien",
    "quái vật": "monster",
    "tận thế": "apocalypse",
    "hậu tận thế": "post-apocalypse",
    "khủng long": "dinosaur",
    "thảm họa": "disaster"
}

# Tiền biên dịch (pre-compile) regex cho dịch thuật để tối ưu hóa hiệu năng
_TRANSLATION_REGEXES = []
for key in sorted(TRANSLATION_MAP.keys(), key=len, reverse=True):
    val = TRANSLATION_MAP[key]
    pattern = re.compile(rf'(?<!\w){re.escape(key)}(?!\w)', re.IGNORECASE)
    _TRANSLATION_REGEXES.append((pattern, val))

def replace_vietnamese_terms(text: str) -> str:
    """
    Dịch các từ/cụm từ thể loại và quốc gia tiếng Việt sang tiếng Anh 
    để khớp với dữ liệu tiếng Anh trong cơ sở dữ liệu.
    """
    if not text:
        return text
    text = unicodedata.normalize('NFC', text)
    
    # Duyệt qua các regex đã tiền biên dịch để thay thế cực nhanh
    for pattern, val in _TRANSLATION_REGEXES:
        text = pattern.sub(val, text)
        
    return text

def clean_tokenize(text: str) -> list[str]:
    """
    Làm sạch chuỗi văn bản và tách thành danh sách các từ (tokens) cơ bản.
    Giữ lại hàm này để tương thích ngược.
    """
    if not text or not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()

def clean_tokenize_corpus(text: str) -> list[str]:
    """
    Tiền xử lý và tách từ cho corpus:
    - Chuẩn hóa Unicode NFC
    - Chuyển thành chữ thường
    - Thay thế các từ tiếng Việt đặc trưng sang tiếng Anh
    - Loại bỏ ký tự đặc biệt
    - Loại bỏ các từ chỉ chứa số (numeric-only)
    - Loại bỏ các từ quá ngắn (length <= 1, trừ 'ý')
    """
    if not text or not isinstance(text, str):
        return []
    
    # 1. Chuyển chữ thường & Chuẩn hóa Unicode NFC
    text = unicodedata.normalize('NFC', text.lower())
    
    # 2. Thay thế các ký tự đặc biệt bằng khoảng trắng (Bỏ qua replace_vietnamese_terms vì corpus gốc đã thuần Anh, giúp tối ưu hiệu năng vượt trội)
    text = re.sub(r"[^\w\s]", " ", text)
    
    # 4. Tách từ & lọc
    tokens = text.split()
    cleaned_tokens = []
    for token in tokens:
        # Loại bỏ token chỉ chứa số
        if token.isdigit():
            continue
        # Loại bỏ token quá ngắn trừ 'ý'
        if len(token) <= 1 and token != "ý":
            continue
        cleaned_tokens.append(token)
        
    return cleaned_tokens

def preprocess_bm25_query(query: str) -> tuple[list[str], list[str]]:
    """
    Tiền xử lý câu truy vấn BM25:
    - Chuẩn hóa Unicode NFC & Chuyển thành chữ thường
    - Trích xuất và loại bỏ rating-like tokens (ví dụ: "8.0", "7.5")
    - Thay thế từ Việt sang Anh (genres/countries)
    - Thay thế ký tự đặc biệt
    - Tách từ
    - Loại bỏ generic tokens: "imdb", "movie", "film", "phim", "điểm", "rating"
    - Loại bỏ standalone numeric tokens (chỉ chứa số)
    - Loại bỏ token quá ngắn trừ 'ý'
    """
    if not query or not isinstance(query, str):
        return [], []
        
    removed_tokens = []
    
    # 1. Chữ thường & NFC
    query_lower = unicodedata.normalize('NFC', query.lower())
    
    # 2. Loại bỏ rating-like tokens
    rating_pattern = re.compile(r'\b\d+\.\d+\b')
    ratings = rating_pattern.findall(query_lower)
    for r in ratings:
        removed_tokens.append(r)
    query_processed = rating_pattern.sub(" ", query_lower)
    
    # 3. Thay thế từ tiếng Việt
    query_processed = replace_vietnamese_terms(query_processed)
    
    # 4. Thay thế ký tự đặc biệt
    query_processed = re.sub(r"[^\w\s]", " ", query_processed)
    
    # 5. Tách từ
    tokens = query_processed.split()
    
    generic_tokens = {
        "imdb", "movie", "film", "phim", "điểm", "diem", "rating",
        "tìm", "tim", "kiếm", "kiem", "có", "co", "trên", "tren", "dưới", "duoi",
        "năm", "nam", "trong", "ngoại", "ngoai", "trừ", "tru", "nhưng", "nhung",
        "không", "khong", "phải", "phai", "và", "va", "hoặc", "hoac", "những", "nhung",
        "các", "cac", "cho", "của", "cua", "tại", "tai", "ở", "o", "được", "duoc",
        "bộ", "bo", "một", "mot", "như", "nhu", "với", "voi", "từ", "tu", "ra",
        "đã", "da", "đang", "dang", "sẽ", "se", "từng", "tung", "làm", "lam",
        "đạo", "diễn", "dao", "dien", "viên", "vien", "vai", "chính", "phụ", "chinh", "phu",
        "thể", "loại", "the", "loai", "nào", "nao", "sản", "xuất", "san", "xuat",
        "đề", "de", "ai", "là", "la", "này", "nay", "đó", "do", "bản", "ban", "nước", "nuoc",
        # Các từ tiếng Việt phổ biến còn sót sau khi replace_vietnamese_terms
        # (chưa có entry dịch sang tiếng Anh) — loại ra để cleaned_query thuần Anh
        "giữa", "giua", "về", "ve", "hay", "nói", "noi", "bạn", "ban",
        "tôi", "toi", "mình", "minh", "gì", "gi", "kể", "ke", "câu", "cau",
        "chuyện", "chuyen", "lấy", "lay", "ở", "o", "khác", "khac", "theo",
        "cùng", "cung", "liên", "lien", "quan", "giới", "gioi", "thiệu", "thieu",
        "muốn", "muon", "xem", "gặp", "gap", "tìm", "nên", "nen", "thích", "thich"
    }
    
    tokenized_query = []
    for token in tokens:
        # Loại bỏ các token chỉ có số (standalone numeric tokens)
        if token.isdigit():
            removed_tokens.append(token)
            continue
            
        # Loại bỏ các từ khóa chung chung và stop-words
        if token in generic_tokens:
            removed_tokens.append(token)
            continue
            
        # Loại bỏ các từ khóa quá ngắn trừ 'ý'
        if len(token) <= 1 and token != "ý":
            removed_tokens.append(token)
            continue
            
        tokenized_query.append(token)
        
    # 7. Query Expansion (limited synonym expansion for P2)
    SYNONYM_EXPANSIONS = {
        "robot": ["android"],
        "friendship": ["friend"],
        "wildlife": ["animal"],
        "space": ["universe"],
        "dinosaur": ["jurassic"],
        "superhero": ["hero"],
        "treasure": ["gold"],
        "alien": ["extraterrestrial"]
    }
    
    expanded_query = []
    for token in tokenized_query:
        expanded_query.append(token)
        if token in SYNONYM_EXPANSIONS:
            for syn in SYNONYM_EXPANSIONS[token]:
                if syn not in tokenized_query:
                    expanded_query.append(syn)
        
    return expanded_query, removed_tokens

def build_bm25_index(df: pd.DataFrame) -> BM25Okapi:
    """
    Xây dựng chỉ mục BM25Okapi với trọng số trường (field weighting):
    - Title weight: repeat 2x
    - Genre weight: repeat 4x
    - Director weight: repeat 3x
    - Star weight: repeat 3x
    - Country weight: repeat 2x
    - Description/tfidf_text weight: repeat 1x
    """
    corpus = []
    
    # Kiểm tra tfidf_text trước tiên nếu có
    desc_col = "tfidf_text" if "tfidf_text" in df.columns else "description"
    
    # Sử dụng itertuples để duyệt qua DataFrame nhanh gấp 10-20 lần so với iterrows
    for row in df.itertuples(index=False):
        title = str(getattr(row, "Title", ""))
        genres = str(getattr(row, "genres", ""))
        directors = str(getattr(row, "directors", ""))
        stars = str(getattr(row, "stars", ""))
        countries = str(getattr(row, "countries_origin", ""))
        description = str(getattr(row, desc_col, ""))
        
        # Trọng số trường BM25 (P6 tuned):
        # - title ×3: tăng từ ×2 để khớp tốt hơn cho truy vấn exact-title
        # - genres ×3: giảm từ ×4 để giảm dominance thể loại, cho phép khớp cross-genre
        # - directors ×3: giữ nguyên — quan trọng cho truy vấn theo đạo diễn
        # - stars ×3: giữ nguyên — quan trọng cho truy vấn theo diễn viên
        # - countries ×2: giữ nguyên
        # - description ×2: tăng từ ×1 để cải thiện khớp chủ đề/nội dung cho similar_to queries
        title_tokens = clean_tokenize_corpus(title) * 3
        genres_tokens = clean_tokenize_corpus(genres) * 3
        directors_tokens = clean_tokenize_corpus(directors) * 3
        stars_tokens = clean_tokenize_corpus(stars) * 3
        countries_tokens = clean_tokenize_corpus(countries) * 2
        desc_tokens = clean_tokenize_corpus(description) * 2
        
        doc_tokens = title_tokens + genres_tokens + directors_tokens + stars_tokens + countries_tokens + desc_tokens
        corpus.append(doc_tokens)
        
    return BM25Okapi(corpus)


def bm25_search(query: str, df: pd.DataFrame, bm25_index: BM25Okapi, top_k: int = 100, trace: dict = None) -> pd.DataFrame:
    """
    Tìm kiếm từ khóa bằng BM25, trả về DataFrame các phim có điểm số cao nhất.
    Hỗ trợ bỏ qua truy vấn rỗng sau tiền xử lý và lưu vết chi tiết (trace).
    """
    if df.empty or bm25_index is None or not query:
        return pd.DataFrame()
        
    tokenized_query, removed_tokens = preprocess_bm25_query(query)
    
    # Safe skip nếu cleaned_query trống rỗng
    if not tokenized_query:
        if trace is not None:
            if "stage1_bm25" not in trace:
                trace["stage1_bm25"] = {}
            trace["stage1_bm25"]["raw_query"] = query
            trace["stage1_bm25"]["cleaned_query"] = ""
            trace["stage1_bm25"]["removed_tokens"] = list(removed_tokens)
            trace["stage1_bm25"]["top_k_requested"] = top_k
            trace["stage1_bm25"]["candidate_count"] = 0
            trace["stage1_bm25"]["candidates"] = []
            trace["stage1_bm25"]["top_candidates"] = []
        return pd.DataFrame()
        
    scores = bm25_index.get_scores(tokenized_query)
    
    # Lấy các index có score cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    result = df.iloc[top_indices].copy()
    result["bm25_score"] = scores[top_indices]
    
    # Chỉ giữ lại các phim có điểm tương hợp > 0
    result = result[result["bm25_score"] > 0]
    
    # Ghi log trace chi tiết
    if trace is not None:
        if "stage1_bm25" not in trace:
            trace["stage1_bm25"] = {}
        
        trace["stage1_bm25"]["raw_query"] = query
        trace["stage1_bm25"]["cleaned_query"] = " ".join(tokenized_query)
        trace["stage1_bm25"]["removed_tokens"] = list(removed_tokens)
        trace["stage1_bm25"]["top_k_requested"] = top_k
        trace["stage1_bm25"]["candidate_count"] = len(result)
        
        candidates = []
        for _, row in result.iterrows():
            val = row.get("bm25_score")
            candidates.append({
                "title": row.get("Title"),
                "imdb_id": row.get("imdb_id"),
                "bm25_score": float(val) if isinstance(val, (np.integer, np.floating)) else val
            })
        trace["stage1_bm25"]["candidates"] = candidates
        trace["stage1_bm25"]["top_candidates"] = candidates[:10]
        
    return result

