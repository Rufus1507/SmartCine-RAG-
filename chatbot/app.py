import os
import sys

# Thêm thư mục gốc vào sys.path để có thể import dạng 'from chatbot.xyz'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

# Nạp cấu hình đường dẫn và hằng số cột
from chatbot.config import (
    KEYWORD_DICT_PATH, ALIASES_PATH,
    COL_TITLE, COL_GENRE, COL_DIRECTOR, COL_STARS, COL_YEAR, COL_RATING, COL_OVERVIEW, COL_LINK,
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
)
# Nạp dữ liệu qua bộ loader dùng chung
from chatbot.data_loader import (
    load_data, load_keyword_dict, load_aliases, load_faiss_index, load_embedder_model
)
# Nạp client LLM và luồng RAG điều phối chính
from chatbot.llm_client import get_llm_client
from chatbot.chains.rag_chain import run_rag_pipeline

# ============================================================
# HIỂN THỊ CARD PHIM (STREAMLIT UI LAYER)
# ============================================================
def render_movie_cards(df: pd.DataFrame):
    """
    Vẽ giao diện hiển thị danh sách các bộ phim tìm kiếm được dạng Card cột.
    Tích hợp hiển thị điểm tương đồng và phân rã giải thích độ tương đồng.
    """
    if df.empty:
        return
    cols = st.columns(min(len(df), 5))
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i]:
            st.markdown(f"**{row[COL_TITLE]}**")
            
            # Hiển thị điểm tương đồng nếu có
            if "similarity_score" in row:
                st.markdown(f"🎯 **Độ tương đồng: {row['similarity_score']}**")
                
            st.caption(f"⭐ {row[COL_RATING]}  •  {int(row[COL_YEAR]) if pd.notna(row[COL_YEAR]) else 'N/A'}")
            st.caption(f"🎬 Đạo diễn: {row[COL_DIRECTOR]}")
            st.caption(f"🎭 Thể loại: {row[COL_GENRE]}")
            if COL_STARS in row and row[COL_STARS]:
                # Hiển thị tối đa 3 diễn viên đầu để card gọn gàng
                stars_list = [s.strip() for s in row[COL_STARS].split(",")]
                st.caption(f"👥 Diễn viên: {', '.join(stars_list[:3])}")
            if COL_OVERVIEW in row and pd.notna(row[COL_OVERVIEW]):
                desc = str(row[COL_OVERVIEW]).strip()
                if len(desc) > 120:
                    desc = desc[:120] + "..."
                st.caption(f"📝 Tóm tắt: {desc}")
                
            # Hiển thị giải thích độ tương đồng (Similarity Breakdown & Reason)
            if "similarity_reason" in row:
                st.caption(f"💡 {row['similarity_reason']}")
                # Expander giải thích điểm số chi tiết
                with st.expander("📊 Phân rã độ khớp", expanded=False):
                    st.caption(f"📖 Nội dung: {row.get('content_similarity', 0.0)*100:.0f}%")
                    st.caption(f"🎭 Thể loại: {row.get('genre_similarity', 0.0)*100:.0f}%")
                    st.caption(f"👥 Diễn viên: {row.get('actor_similarity', 0.0)*100:.0f}%")
                    st.caption(f"🎬 Đạo diễn: {row.get('director_similarity', 0.0)*100:.0f}%")
                    st.caption(f"🌍 Quốc gia: {row.get('country_similarity', 0.0)*100:.0f}%")
                    st.caption(f"📅 Thập kỷ: {row.get('decade_similarity', 0.0)*100:.0f}%")
                    st.caption(f"🏆 Giải thưởng: {row.get('award_similarity', 0.0)*100:.0f}%")
                    
            if COL_LINK in row and pd.notna(row[COL_LINK]):
                st.markdown(f"[🔗 Xem trên IMDb]({row[COL_LINK]})")

# ============================================================
# CẤU HÌNH TRANG STREAMLIT
# ============================================================
st.set_page_config(
    page_title="🎬 CineBot",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 CineBot — Chatbot Tìm Phim Thông Minh")
st.caption("Hỏi bất kỳ điều gì về phim: thể loại, đạo diễn, diễn viên, năm, điểm IMDB...")

# --- Sidebar: Cấu hình tài nguyên và API ---
with st.sidebar:
    st.header("⚙️ Cấu hình")

    # Chọn nhà cung cấp LLM
    llm_provider = st.selectbox(
        "Nhà cung cấp LLM",
        ["Local LLM", "Gemini API"],
        index=0,
        key="llm_provider"
    )

    if llm_provider == "Gemini API":
        st.info("🔌 Sử dụng Google Gemini API")
        gemini_api_key = st.text_input(
            "Gemini API Key",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            key="gemini_api_key"
        )
        gemini_model = st.selectbox(
            "Chọn Model Gemini",
            ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash-001"],
            index=0,
            key="gemini_model"
        )
    else:
        st.info(f"🔌 Kết nối Local Endpoint")
        local_base_url = st.text_input(
            "Endpoint URL",
            value=LLM_BASE_URL,
            key="local_base_url"
        )
        local_api_key = st.text_input(
            "API Key (nếu có)",
            value=LLM_API_KEY,
            type="password",
            key="local_api_key"
        )
        model_options = ["cx/gpt-5.5", "cx/gpt-5.4", "cx/gpt-5.3-codex", "cx/gpt-5.3-codex-high"]
        local_model = st.selectbox(
            "Chọn Model Local",
            model_options,
            index=0,
            key="local_model"
        )

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

    st.divider()
    st.markdown("**🔍 Debugging:**")
    if "last_parsed" in st.session_state:
        st.json(st.session_state.last_parsed)
    else:
        st.caption("Chưa có truy vấn nào được thực hiện.")

# --- Load data & keyword dict ---
try:
    df = load_data()
    st.sidebar.success(f"✅ Đã load {len(df):,} phim")
except FileNotFoundError:
    st.error(f"❌ Không tìm thấy các file dữ liệu CSV tại đường dẫn. Vui lòng kiểm tra thư mục dữ liệu.")
    st.stop()

try:
    keyword_dict = load_keyword_dict()
    st.sidebar.success(f"✅ Đã load {len(keyword_dict):,} từ khóa")
except FileNotFoundError:
    st.error(f"❌ Không tìm thấy file `keyword_dict.json`.")
    st.stop()

try:
    aliases_dict = load_aliases()
    st.sidebar.success(f"✅ Đã load {len(aliases_dict):,} biệt danh")
except Exception:
    aliases_dict = {}

# --- Nạp FAISS và embedder model ---
try:
    faiss_index = load_faiss_index()
    embedder_model = load_embedder_model()
except Exception:
    faiss_index = None
    embedder_model = None

if faiss_index is not None and embedder_model is not None:
    st.sidebar.success("🔮 Đã kích hoạt Semantic Search (FAISS)")
else:
    st.sidebar.warning("⚠️ Chưa có file chỉ mục. Chạy generate_embeddings.py trước!")

# --- Lịch sử chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Xin chào! Tôi là CineBot 🎬 Bạn muốn tìm phim gì hôm nay? Hãy hỏi tôi về thể loại, đạo diễn, diễn viên hoặc mô tả phim nhé!",
        "movies": None
    })

if "last_filters" not in st.session_state:
    st.session_state.last_filters = {}

# --- Hiển thị lịch sử hội thoại cũ ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("movies") is not None and not msg["movies"].empty:
            render_movie_cards(msg["movies"])

# --- Xử lý sự kiện click ví dụ ---
pending = st.session_state.pop("pending_input", None)
user_input = st.chat_input("Nhập câu hỏi của bạn...") or pending

if user_input:
    # Hiển thị tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": user_input, "movies": None})
    with st.chat_message("user"):
        st.write(user_input)

    # Phản hồi từ trợ lý
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm..."):
            
            # Cấu hình tham số gọi client LLM từ Sidebar
            provider = st.session_state.get("llm_provider", "Local LLM")
            if provider == "Gemini API":
                api_key = st.session_state.get("gemini_api_key", "")
                model_name = st.session_state.get("gemini_model", "gemini-2.5-flash")
                base_url = None
            else:
                provider = "Local LLM"
                base_url = st.session_state.get("local_base_url", LLM_BASE_URL)
                api_key = st.session_state.get("local_api_key", LLM_API_KEY)
                model_name = st.session_state.get("local_model", LLM_MODEL)
                
            try:
                # 1. Khởi tạo đối tượng ChatOpenAI qua cache
                llm = get_llm_client(provider, api_key, model_name, base_url)
                
                # 2. Gửi yêu cầu qua luồng RAG điều phối chính (Kích hoạt streaming=True)
                answer_generator, filtered_df, intent, filters, detected = run_rag_pipeline(
                    llm=llm,
                    user_input=user_input,
                    df=df,
                    keyword_dict=keyword_dict,
                    aliases_dict=aliases_dict,
                    faiss_index=faiss_index,
                    embedder_model=embedder_model,
                    chat_history=st.session_state.messages,
                    last_filters=st.session_state.get("last_filters", {}),
                    stream=True
                )
                
                # Lưu thông tin bộ lọc làm ngữ cảnh cho lượt kế tiếp
                st.session_state.last_filters = filters
                st.session_state.last_parsed = {
                    "user_input": user_input,
                    "intent": intent,
                    "filters": filters,
                    "detected": detected
                }
                
                # 3. Stream phản hồi lên Streamlit UI
                full_response = st.write_stream(answer_generator)
                
                # 4. Hiển thị card phim
                if not filtered_df.empty:
                    render_movie_cards(filtered_df)
                    
            except Exception as e:
                full_response = f"Lỗi hệ thống trong quá trình xử lý: {e}"
                st.write(full_response)
                filtered_df = pd.DataFrame()

    # Lưu phản hồi vào lịch sử cuộc trò chuyện
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "movies": filtered_df if not filtered_df.empty else None
    })
