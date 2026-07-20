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

Hãy giới thiệu các phim này. Với mỗi phim, hãy đề cập đầy đủ các thông tin sau: tên phim, năm phát hành, quốc gia sản xuất, thể loại, đạo diễn, diễn viên, điểm IMDb, tóm tắt nội dung cực kỳ ngắn gọn (chỉ từ 1 đến 2 câu ngắn), và kèm theo link IMDb (nếu có) để người dùng click. Tuyệt đối không viết tóm tắt quá dài dòng."""

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
