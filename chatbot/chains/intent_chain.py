import re
import json
import time
import logging
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from langchain_core.language_models import BaseChatModel
from chatbot.prompts.intent_prompt import get_intent_prompt
from chatbot.rate_limiter import gemini_rate_limiter
from rapidfuzz import fuzz
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception

logger = logging.getLogger(__name__)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Trả về True nếu lỗi là 429 RESOURCE_EXHAUSTED."""
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "rate limit" in msg


@retry(
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(4),
    retry=retry_if_exception(_is_rate_limit_error),
)
def _invoke_llm_intent(llm, prompt: str):
    """Gọi LLM với rate limiter (chỉ khi dùng Gemini) + retry-on-429 cho intent chain."""
    provider = getattr(llm, "provider", None)
    model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
    openai_api_base = str(getattr(llm, "openai_api_base", ""))
    
    logger.debug("[intent_chain] LLM invocation start: provider=%s, model=%s", provider, model_name)
    
    # Task B3: Chỉ wait khi provider là Gemini API
    is_gemini = (provider == "Gemini API") or ("generativelanguage.googleapis.com" in openai_api_base)
    if is_gemini:
        gemini_rate_limiter.wait()

    t0 = time.monotonic()
    # Task B2: Bind max_tokens=300 cho intent chain (JSON ngắn)
    llm_call = llm.bind(max_tokens=300) if hasattr(llm, "bind") else llm
    result = llm_call.invoke(prompt)
    logger.debug("[intent_chain] llm.invoke took %.2fs", time.monotonic() - t0)
    return result


# ============================================================
# PYDANTIC SCHEMA CHO INTENT PARSING
# ============================================================
class Filters(BaseModel):
    title: Optional[str] = None
    genre: Optional[str] = None
    director: Optional[str] = None
    star: Optional[str] = None
    country: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    rating_min: Optional[float] = None
    director_exclude: Optional[str] = None
    star_exclude: Optional[str] = None
    runtime_min: Optional[int] = None
    runtime_max: Optional[int] = None
    duration_min: Optional[int] = None
    duration_max: Optional[int] = None
    has_oscar: Optional[bool] = None
    has_awards: Optional[bool] = None
    meta_score_min: Optional[int] = None
    sort_by: Optional[str] = None
    sort_order: Optional[str] = None


    @field_validator('year_min', 'year_max', mode='before')
    @classmethod
    def coerce_year(cls, v):
        if v is None or v == "":
            return None
        try:
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

    @field_validator('runtime_min', 'runtime_max', 'duration_min', 'duration_max', 'meta_score_min', mode='before')
    @classmethod
    def coerce_int(cls, v):
        if v is None or v == "":
            return None
        try:
            clean_val = str(v).lower().replace("phút", "").replace("phut", "").strip()
            match = re.search(r'\d+', clean_val)
            if match:
                return int(match.group(0))
            return int(clean_val)
        except Exception:
            return None

    @field_validator('has_oscar', 'has_awards', mode='before')
    @classmethod
    def coerce_bool(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, bool):
            return v
        val_str = str(v).lower().strip()
        if val_str in ("true", "1", "yes", "đúng", "dung", "có", "co"):
            return True
        if val_str in ("false", "0", "no", "không", "khong"):
            return False
        return None

class ParsedIntent(BaseModel):
    intent: str = "chitchat"
    filters: Filters = Field(default_factory=Filters)
    free_text: str = ""

def run_intent_chain(llm: BaseChatModel, user_message: str, detected_entities: dict = None, chat_history: list = None) -> dict:
    """
    Thực thi Chain phân tích ý định người dùng (Tầng 1):
    1. Chuẩn bị gợi ý thực thể và lịch sử hội thoại.
    2. Gọi LLM bằng PromptTemplate.
    3. Trích xuất JSON, validate bằng Pydantic và chạy luật khôi phục lỗi.
    """
    # 1. Bổ sung thông tin gợi ý thực thể
    hints_str = ""
    if detected_entities:
        hints_list = []
        if detected_entities.get("genres"):
            hints_list.append(f"- Thể loại gợi ý: {', '.join(detected_entities['genres'])}")
        if detected_entities.get("directors"):
            hints_list.append(f"- Đạo diễn gợi ý: {', '.join(detected_entities['directors'])}")
        if detected_entities.get("stars"):
            hints_list.append(f"- Diễn viên gợi ý: {', '.join(detected_entities['stars'])}")
        if hints_list:
            hints_str = "\nGỢI Ý THỰC THỂ TÌM THẤY TRONG CƠ SỞ DỮ LIỆU:\n" + "\n".join(hints_list) + "\n"

    # 2. Bổ sung lịch sử chat để giải quyết ngữ cảnh liên kết (Co-reference resolution)
    history_str = ""
    if chat_history:
        # Lấy tối đa 6 tin nhắn gần nhất trong lịch sử hội thoại (bỏ câu hỏi hiện tại)
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

    # 3. Tạo Prompt và gọi LLM
    prompt_template = get_intent_prompt(hints_str, history_str)
    
    parsed = None
    try:
        # Gọi LLM qua rate limiter + retry-on-429
        response = _invoke_llm_intent(llm, prompt_template.format(input=user_message))
        raw = response.content.strip()
        
        # Regex trích xuất JSON nếu LLM trả văn bản bao quanh
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        parsed_dict = json.loads(raw)
        validated = ParsedIntent(**parsed_dict)
        parsed = validated.model_dump()
    except Exception:
        # Fallback an toàn nếu LLM lỗi hoặc trả định dạng sai
        parsed = {
            "intent": "chitchat",
            "filters": {},
            "free_text": user_message
        }

    # 4. HẬU XỬ LÝ & KHÔI PHỤC LỖI (Intent Recovery)
    intent = parsed.get("intent", "chitchat")
    
    # Ép intent sang search nếu có thực thể tìm thấy nhưng LLM nhận diện sai
    has_entities = detected_entities and any(detected_entities.get(k) for k in [
        "genres", "directors", "stars", "content_keywords"
    ])
    if has_entities and intent == "chitchat":
        parsed["intent"] = "search"
        intent = "search"

    # Khôi phục nếu chứa từ khóa tìm kiếm tiếng Việt đặc trưng
    search_indicators = {
        "tìm", "tim", "kiếm", "kiem", "lọc", "loc", "gợi ý", "goi y", "đề xuất", "de xuat",
        "lượt xem", "luot xem", "điểm số", "diem so", "bộ phim", "bo phim", "phim nào", "phim co", "phim có",
        "phim", "tựa", "tua", "giống", "giong", "tương tự", "tuong tu", "giới thiệu", "gioi thieu", "như", "nhu"
    }
    words_in_msg = set(re.findall(r'\b\w+\b', user_message.lower()))
    if intent == "chitchat" and not words_in_msg.isdisjoint(search_indicators):
        parsed["intent"] = "search"
        intent = "search"

    # Khôi phục khi có mẫu câu hỏi phim tương tự
    similar_patterns = [
        r'(?:phim\s+)?(?:giống|tương\s+tự|tựa\s+như|tựa\s+với|như)\s+(?:phim\s+)?([^,.?]+)',
        r'(?:tương\s+tự|tựa)\s+với\s+(?:phim\s+)?([^,.?]+)',
        r'(?:phim\s+)?tựa\s+(?:bộ\s+|phim\s+)?([^,.?]+)',
        r'similar\s+to\s+([^,.?]+)',
        r'like\s+([^,.?]+)'
    ]
    for pat in similar_patterns:
        if re.search(pat, user_message, re.IGNORECASE):
            parsed["intent"] = "search"
            intent = "search"
            break

    # Cập nhật thực thể bị LLM trích xuất thiếu từ kết quả của Extractor
    if detected_entities and intent in ("search", "recommend", "info"):
        filters = parsed.setdefault("filters", {})
        if detected_entities.get("genres") and not filters.get("genre"):
            filters["genre"] = detected_entities["genres"][0]
        if detected_entities.get("directors") and not filters.get("director"):
            det_dir = detected_entities["directors"][0]
            exclude_dir = filters.get("director_exclude")
            if not exclude_dir or (det_dir.lower() != exclude_dir.lower() and fuzz.QRatio(det_dir.lower(), exclude_dir.lower()) < 90):
                filters["director"] = det_dir
        if detected_entities.get("stars") and not filters.get("star"):
            det_star = detected_entities["stars"][0]
            exclude_star = filters.get("star_exclude")
            if not exclude_star or (det_star.lower() != exclude_star.lower() and fuzz.QRatio(det_star.lower(), exclude_star.lower()) < 90):
                filters["star"] = det_star

    return parsed
