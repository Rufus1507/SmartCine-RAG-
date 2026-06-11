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
    
    # Accented
    "tìm", "kiếm", "đạo", "diễn", "viên", "thể", "loại", "nhà", "năm", "điểm",
    "trên", "dưới", "tuyển", "có", "về", "chiếu", "rạp", "bộ", "mỹ", "hàn", 
    "của", "các", "những", "là", "và", "hoặc", "viễn", "tưởng", "khoa", "học"
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
