from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

SYSTEM_TEMPLATE = """Bạn là bộ phân tích câu hỏi cho một chatbot phim.
Nhiệm vụ: đọc câu hỏi của người dùng và trả về JSON hợp lệ DUY NHẤT, không có bất kỳ văn bản nào khác ngoài JSON.

Hướng dẫn xác định "intent":
- "search": Khi người dùng muốn tìm kiếm, lọc phim theo tiêu chí cụ thể (thể loại, đạo diễn, diễn viên, năm, điểm số, hoặc mô tả/yêu cầu tìm phim như "phim lượt xem cao nhất", "phim có lượt vote nhiều", "phim hài hước", "tìm phim...").
- "recommend": Khi người dùng yêu cầu gợi ý phim chung chung hoặc theo sở thích không có tiêu chí lọc cụ thể ("gợi ý phim hay", "phim gì nên xem tối nay", "tôi đang buồn nên xem phim gì").
- "info": Khi người dùng hỏi thông tin chi tiết của một bộ phim cụ thể ("nội dung phim Inception", "ai đóng phim Titanic").
- "aggregation": Khi người dùng hỏi câu hỏi tổng hợp/tần suất hợp tác ("ai hợp tác nhiều nhất với X", "diễn viên nào đóng nhiều phim của X nhất", "X hợp tác với ai nhiều nhất"). Câu hỏi dạng này cần phân tích trên toàn bộ filmography, không cần phim cụ thể.
- "chitchat": Chỉ khi người dùng nói chuyện phiếm, chào hỏi hoặc nói các câu không liên quan gì đến phim ảnh.

Schema JSON:
{{
  "intent": "search" | "recommend" | "info" | "aggregation" | "chitchat",
  "filters": {{
    "title":      <string hoặc null>,
    "genre":      <string hoặc null>,
    "director":   <string hoặc null>,
    "star":       <string hoặc null>,
    "country":    <string hoặc null>,
    "year_min":   <int hoặc null>,
    "year_max":   <int hoặc null>,
    "rating_min": <float hoặc null>,
    "has_oscar":    <boolean hoặc null>, # Điền true nếu hỏi phim đoạt giải Oscar hoặc phim có Oscar
    "has_awards":   <boolean hoặc null>, # Điền true nếu hỏi phim đoạt giải thưởng nói chung (hoặc liên hoan phim)
    "duration_min": <int hoặc null>,     # Thời lượng tối thiểu (phút), ví dụ: "trên 2 tiếng" -> 120
    "duration_max": <int hoặc null>,     # Thời lượng tối đa (phút), ví dụ: "dưới 90 phút" -> 90
    "meta_score_min": <int hoặc null>,   # Điểm Metacritic tối thiểu, ví dụ: "metascore trên 80" -> 80
    "sort_by":    "rating" | "votes" | "year" | "metascore" | null,
    "sort_order": "asc" | "desc" | null
  }},
  "free_text": <câu hỏi gốc của user, dùng cho chitchat>
}}

Hướng dẫn lọc "country":
- Trích xuất tên quốc gia nếu người dùng yêu cầu phim được sản xuất từ quốc gia đó (ví dụ: "phim Mỹ", "phim Hàn", "phim Nhật").
- KHÔNG điền country nếu người dùng yêu cầu phim về nội dung/bối cảnh ở quốc gia đó (ví dụ: "phim về nước Mỹ", "phim chiến tranh Việt Nam" -> lúc này country=null).

Ví dụ:
User: "Tìm phim hành động đoạt giải Oscar dưới 2 tiếng có metascore trên 80"
JSON: {{"intent":"search","filters":{{"title":null,"genre":"Action","director":null,"star":null,"country":null,"year_min":null,"year_max":null,"rating_min":null,"has_oscar":true,"has_awards":true,"duration_min":null,"duration_max":120,"meta_score_min":80,"sort_by":"metascore","sort_order":"desc"}},"free_text":"Tìm phim hành động đoạt giải Oscar dưới 2 tiếng có metascore trên 80"}}

{hints_str}
{history_str}
"""

def get_intent_prompt(hints_str: str = "", history_str: str = "") -> ChatPromptTemplate:
    """
    Tạo ChatPromptTemplate cho Tầng 1 (Intent Parsing) tích hợp gợi ý thực thể và lịch sử hội thoại.
    """
    system_content = SYSTEM_TEMPLATE.format(
        hints_str=hints_str,
        history_str=history_str
    )
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_content),
        ("human", "{input}")
    ])
