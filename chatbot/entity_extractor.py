import re
import streamlit as st
from rapidfuzz import process, fuzz

# Danh sách các từ tiếng Việt và tiếng Anh phổ biến cần bỏ qua khi fuzzy match để tránh khớp nhầm
IGNORE_FUZZY = {
    # Unaccented
    "phim", "tim", "kiem", "cho", "xem", "dao", "dien", "vien", "the", "loai", 
    "nha", "nam", "diem", "tren", "duoi", "tuyen", "hay", "nhat", "co", "ve", 
    "chieu", "rap", "bom", "tan", "le", "bo", "my", "han", "trung", "viet", "nhat",
    "cua", "cac", "nhung", "la", "va", "hoac", "trong", "ngoai", "cao", "thap",
    "hoang", "da", "hoang da", "chau", "phi", "chau phi",
    
    # Accented
    "tìm", "kiếm", "đạo", "diễn", "viên", "thể", "loại", "nhà", "năm", "điểm",
    "trên", "dưới", "tuyển", "có", "về", "chiếu", "rạp", "bộ", "mỹ", "hàn", 
    "của", "các", "những", "là", "và", "hoặc", "viễn", "tưởng", "khoa", "học",
    "dã", "hoang dã", "châu", "phi", "châu phi"
}

@st.cache_data
def get_fuzzy_candidates(keyword_dict: dict) -> list:
    """
    Trích xuất danh sách các thực thể (thể loại, đạo diễn, diễn viên) dùng cho fuzzy match.
    Bỏ qua các từ khóa nội dung mô tả (content_keywords) để giảm tải dữ liệu.
    """
    candidates = []
    for key, intent in keyword_dict.items():
        if intent in ("search_genre", "search_director", "search_star"):
            # Tránh các cụm từ quá ngắn để fuzzy match không bị nhiễu
            if len(key) >= 5:
                candidates.append(key)
    return list(set(candidates))

def detect_entities(user_message: str, keyword_dict: dict, aliases_dict: dict) -> dict:
    """
    Quét qua câu hỏi của user để phát hiện các thực thể có trong keyword_dict.
    1. Ưu tiên khớp chính xác (Exact match) với tốc độ O(1).
    2. Chỉ chạy Fuzzy match trên danh sách thực thể rút gọn khi khớp chính xác thất bại.
    """
    user_msg_lower = user_message.lower()
    detected = {
        "genres": [],
        "directors": [],
        "stars": [],
        "writers": [],
        "content_keywords": []
    }
    
    # Làm sạch câu hỏi, tách thành các từ đơn giản
    words = re.findall(r'\b\w+\b', user_msg_lower)
    n = len(words)
    
    # Tạo các n-gram từ độ dài 1 đến 5
    candidates = []
    for length in range(1, min(6, n + 1)):
        for i in range(n - length + 1):
            ngram = " ".join(words[i:i+length])
            candidates.append(ngram)
            
    # Lọc trùng và sắp xếp theo độ dài giảm dần (để ưu tiên khớp cụm từ dài trước)
    candidates = sorted(list(set(candidates)), key=len, reverse=True)
    
    # Lấy danh sách thực thể phục vụ fuzzy lookup rút gọn
    fuzzy_keys_list = get_fuzzy_candidates(keyword_dict)
    
    matched_keys = set()
    for candidate in candidates:
        # Bỏ qua nếu từ khóa đã được khớp bởi cụm từ dài hơn
        if any(candidate in already_matched for already_matched in matched_keys):
            continue
            
        target_key = None
        intent = None
        
        # 1. Kiểm tra khớp chính xác trong aliases_dict
        if candidate in aliases_dict:
            resolved_key = aliases_dict[candidate]
            if resolved_key in keyword_dict:
                target_key = resolved_key
                intent = keyword_dict[resolved_key]
                
        # 2. Kiểm tra khớp chính xác trong keyword_dict
        elif candidate in keyword_dict:
            target_key = candidate
            intent = keyword_dict[candidate]
            
        # 3. Fuzzy match bằng rapidfuzz (chỉ áp dụng cho cụm từ dài >= 5 và không chứa stopwords)
        elif len(candidate) >= 5:
            candidate_words = set(candidate.split())
            if candidate_words.isdisjoint(IGNORE_FUZZY):
                # Sử dụng danh sách rút gọn fuzzy_keys_list để chạy cực nhanh
                match_res = process.extractOne(
                    candidate, fuzzy_keys_list,
                    scorer=fuzz.QRatio
                )
                if match_res:
                    match, score, _ = match_res
                    intent_type = keyword_dict.get(match)
                    # Nâng ngưỡng tin cậy để tránh các trùng lặp sai (như Iron Man khớp với diễn viên ion manu)
                    min_score = 90 if intent_type in ("search_star", "search_director", "search_writer") else 85
                    if score >= min_score:
                        target_key = match
                        intent = intent_type
                
        # Phân loại thực thể tìm được
        if target_key and intent:
            matched_keys.add(candidate)
            matched_keys.add(target_key)
            # Tránh khớp lại các từ đơn lẻ nằm trong thực thể này
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
                detected["directors"].append(target_key)
                detected["stars"].append(target_key)
            elif intent == "search_content":
                detected["content_keywords"].append(target_key)
                
    return detected


# Danh sách các cụm từ ghép tiếng Việt phổ biến trong lĩnh vực phim ảnh (P1 Noun-Phrase)
COMPOUND_PHRASES = [
    "động vật hoang dã", "hoang dã", "trí tuệ nhân tạo", "du hành vũ trụ", "du hành thời gian",
    "ngoài hành tinh", "người ngoài hành tinh", "thế giới tương lai", "tận thế", "hậu tận thế",
    "quái vật", "sát nhân", "pháp thuật", "ma thuật", "phù thủy", "tình bạn", "kho báu",
    "rượt đuổi xe", "rượt đuổi xe hơi", "xe hơi", "đám cưới", "khủng long", "thảm họa",
    "hoạt hình", "khoa học viễn tưởng", "viễn tưởng", "hành động", "hài hước", "tình cảm",
    "lãng mạn", "phiêu lưu", "tội phạm", "hình sự", "giật gân", "thần thoại", "tài liệu",
    "gia đình", "chiến tranh", "lịch sử", "miền tây", "kinh dị", "kịch tính", "chính kịch",
    "tâm lý", "nhạc kịch", "ca nhạc", "âm nhạc", "động vật", "châu phi", "châu mỹ", "châu á",
    "châu âu", "phim tài liệu", "siêu anh hùng", "bị chôn giấu"
]

# Danh sách stopword tiếng Việt dùng cho fallback content_keywords extraction
# Khi query không có entity nào rõ ràng, giữ lại danh từ/cụm từ mô tả nội dung chính
_VI_STOPWORDS = {
    "tìm", "tim", "cho", "tôi", "toi", "mình", "minh", "một", "mot", "bộ", "bo",
    "phim", "film", "movie", "có", "co", "nào", "nao", "không", "khong", "khi",
    "nói", "noi", "về", "ve", "giữa", "giua", "và", "va", "với", "voi", "hay",
    "được", "duoc", "của", "cua", "là", "la", "trong", "này", "nay", "đó", "do",
    "theo", "ở", "o", "nên", "nen", "muốn", "muon", "xem", "gì", "gi",
    "câu", "cau", "chuyện", "chuyen", "kể", "ke", "giới", "gioi", "thiệu", "thieu",
    "gợi", "goi", "ý", "y", "xuất", "xuat", "sắc", "sac", "đặc", "dac",
    "như", "nhu", "thế", "the", "nào", "nao", "thể", "the", "tìm", "kiếm",
    "nhất", "nhat", "cũng", "cung", "nhiều", "nhieu", "ít", "it", "hơn", "hon",
    "liên", "lien", "quan", "các", "cac", "những", "nhung", "đã", "da", "đang",
    "sẽ", "se", "từng", "tung", "làm", "lam", "hoặc", "hoac", "cùng",
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "do", "does",
    "did", "have", "has", "had", "of", "in", "on", "at", "to", "for", "with",
    "about", "that", "this", "which", "who", "some", "any", "or", "and",
}

def extract_content_keywords_fallback(user_message: str, max_keywords: int = 4) -> list:
    """
    Trích từ khóa mô tả nội dung khi entity_detection không tìm được entity rõ ràng.
    Lọc stopword tiếng Việt/Anh, ưu tiên giữ lại các cụm từ ghép có nghĩa trong COMPOUND_PHRASES.
    Dùng làm fallback khi genres/directors/stars đều rỗng — đảm bảo content_keywords
    không rỗng với câu hỏi ngữ nghĩa thuần (ví dụ: "tình bạn giữa con người và robot").
    """
    user_msg_lower = user_message.lower()
    extracted = []
    
    # 1. Quét các cụm từ ghép có nghĩa trong danh sách trước
    sorted_phrases = sorted(COMPOUND_PHRASES, key=len, reverse=True)
    temp_msg = user_msg_lower
    for phrase in sorted_phrases:
        pattern = re.compile(rf'\b{re.escape(phrase)}\b', re.IGNORECASE)
        if pattern.search(temp_msg):
            extracted.append(phrase)
            temp_msg = pattern.sub(' ', temp_msg)
            
    # 2. Tách các từ đơn còn lại sau khi đã loại bỏ cụm từ ghép
    words = re.findall(r'\b\w+\b', temp_msg)
    content_words = [
        w for w in words
        if w not in _VI_STOPWORDS and len(w) >= 3
    ]
    for w in content_words:
        if w not in extracted and len(extracted) < max_keywords:
            extracted.append(w)
            
    return extracted[:max_keywords]


def is_refine_query(user_input: str) -> bool:
    """
    Xác định xem câu hỏi có phải là câu nối tiếp/tinh chỉnh ngữ cảnh hay không.
    """
    refine_keywords = {
        "nhưng", "nhung", "chỉ", "chi", "thêm", "them", "nữa", "nua", "còn", "con", 
        "khác", "khac", "đó", "do", "này", "nay", "ông ấy", "ong ay", "bà ấy", "ba ay",
        "họ", "ho", "sau", "trước", "truoc", "hơn", "hon", "dưới", "duoi", "trên", "tren"
    }
    words = set(re.findall(r'\b\w+\b', user_input.lower()))
    return not words.isdisjoint(refine_keywords)
