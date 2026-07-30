from langchain_core.prompts import ChatPromptTemplate

CHITCHAT_SYSTEM = "Bạn là trợ lý phim thân thiện. Trả lời bằng tiếng Việt, ngắn gọn, tự nhiên."

RAG_SYSTEM = (
    "Bạn là trợ lý phim thân thiện. Trả lời bằng tiếng Việt, thân thiện và tự nhiên.\n"
    "CHỈ trả lời dựa trên danh sách phim được cung cấp dưới đây. "
    "Tuyệt đối không tự bịa đặt thông tin (như tên phim, năm, đạo diễn, diễn viên hoặc nội dung) "
    "nếu không có trong ngữ cảnh được cung cấp (anti-hallucination)."
)

RAG_USER_TEMPLATE = """Người dùng hỏi: "{input}"

Danh sách phim tìm được:
{movies_info}

Hãy giới thiệu các phim này bằng tiếng Việt. Với mỗi phim, hãy đề cập đầy đủ các thông tin sau (nếu có trong dữ liệu):
- **Tên phim**
- Năm phát hành
- Quốc gia sản xuất
- Thể loại
- Đạo diễn
- Thời lượng (phút)
- Diễn viên chính
- Điểm IMDb
- Tóm tắt nội dung chi tiết (viết đầy đủ, không cắt ngắn)
- Link IMDb: (PHẢI đưa vào nếu có trong dữ liệu, dạng markdown [Xem trên IMDb](url))

KHÔNG đưa vào output: Giải thưởng, Mã IMDb, số lượt bình chọn, meta score.
Nếu thông tin nào không có trong dữ liệu thì bỏ qua hoàn toàn dòng đó (không viết "Không có")."""

AGGREGATION_SYSTEM = (
    "Bạn là trợ lý phim thân thiện. Trả lời bằng tiếng Việt, thân thiện và tự nhiên.\n"
    "CHỈ trả lời dựa trên dữ liệu được cung cấp dưới đây. "
    "Tuyệt đối không tự bịa đặt thông tin nếu không có trong ngữ cảnh (anti-hallucination)."
)

AGGREGATION_USER_TEMPLATE = """Người dùng hỏi: "{input}"

Danh sách người cộng tác tìm được (đã sắp xếp theo số lần hợp tác):
{movies_info}

Hãy trả lời câu hỏi dựa trên dữ liệu trên. Với mỗi người, đề cập: tên, vai trò (diễn viên/đạo diễn), số lần hợp tác, các bộ phim hợp tác chung (chỉ liệt kê tên phim, KHÔNG cần năm/quốc gia/thể loại riêng từng phim vì không có trong dữ liệu), điểm IMDb trung bình của các phim chung (nếu có), và tỷ lệ vai chính/vai phụ (nếu có). Không tự thêm thông tin ngoài dữ liệu được cung cấp."""

def get_chitchat_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", CHITCHAT_SYSTEM),
        ("human", "{input}")
    ])

def get_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM),
        ("human", RAG_USER_TEMPLATE)
    ])

def get_aggregation_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", AGGREGATION_SYSTEM),
        ("human", AGGREGATION_USER_TEMPLATE)
    ])
