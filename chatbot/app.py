import os
import sys

import uuid

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
                
            rating_display = row[COL_RATING] if pd.notna(row[COL_RATING]) else "N/A"
            year_display = int(row[COL_YEAR]) if pd.notna(row[COL_YEAR]) else "N/A"
            st.caption(f"⭐ {rating_display}  •  {year_display}")
            st.caption(f"🎬 Đạo diễn: {row[COL_DIRECTOR]}")
            st.caption(f"🎭 Thể loại: {row[COL_GENRE]}")
            if "countries_origin" in row and pd.notna(row["countries_origin"]) and str(row["countries_origin"]).strip():
                st.caption(f"🌍 Quốc gia: {row['countries_origin']}")
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

def render_feedback_ui(idx: int, msg: dict):
    """
    Vẽ giao diện 👍/👎 và lý do đóng góp cho từng tin nhắn của trợ lý.
    Sử dụng turn_index để tránh xung đột key và hỗ trợ re-render.
    """
    turn_index = idx // 2
    
    # Khởi tạo trạng thái cho lượt chat này
    rating_key = f"feedback_rating_{turn_index}"
    submitted_key = f"feedback_submitted_{turn_index}"
    
    if rating_key not in st.session_state:
        st.session_state[rating_key] = None
    if submitted_key not in st.session_state:
        st.session_state[submitted_key] = False
        
    if st.session_state[submitted_key]:
        st.caption("✅ Đã ghi nhận ý kiến đóng góp của bạn. Cảm ơn bạn!")
        return
        
    rating = st.session_state[rating_key]
    
    if rating is None:
        col1, col2, _ = st.columns([1, 1, 8])
        with col1:
            if st.button("👍", key=f"thumbs_up_{turn_index}"):
                st.session_state[rating_key] = "up"
                st.rerun()
        with col2:
            if st.button("👎", key=f"thumbs_down_{turn_index}"):
                st.session_state[rating_key] = "down"
                st.rerun()
    else:
        rating_emoji = "👍" if rating == "up" else "👎"
        
        col_desc, col_reset, _ = st.columns([2, 1.5, 6.5])
        col_desc.write(f"Đánh giá của bạn: **{rating_emoji}**")
        with col_reset:
            if st.button("Chọn lại", key=f"reset_rating_{turn_index}"):
                st.session_state[rating_key] = None
                st.rerun()
                
        comment = st.text_input(
            "Lý do ngắn hoặc đóng góp ý kiến (tùy chọn):",
            key=f"comment_input_{turn_index}"
        )
        
        if st.button("Gửi phản hồi", key=f"submit_btn_{turn_index}"):
            # Trích xuất thông tin lượt chat
            user_query = st.session_state.messages[idx - 1]["content"] if idx > 0 else ""
            
            # Lấy danh sách phim trả về
            movies_df = msg.get("movies")
            movie_titles = list(movies_df[COL_TITLE].values) if (movies_df is not None and not movies_df.empty) else []
            
            # Tóm tắt câu trả lời của bot
            answer_text = msg.get("content", "")
            bot_preview = answer_text[:200]
            
            # Log phản hồi
            from chatbot.feedback_logger import log_feedback
            log_feedback(
                session_id=st.session_state.get("session_id", "unknown"),
                turn_index=turn_index,
                user_query=user_query,
                intent=msg.get("intent", "none"),
                filters=msg.get("filters", {}),
                route=msg.get("route_name", "none"),
                bot_answer_preview=bot_preview,
                movie_titles_returned=movie_titles,
                rating=rating,
                comment=comment
            )
            
            st.session_state[submitted_key] = True
            st.rerun()

# ============================================================
# CẤU HÌNH TRANG STREAMLIT
# ============================================================
st.set_page_config(
    page_title="🎬 CineBot",
    page_icon="🎬",
    layout="wide"
)

# --- Khởi tạo các biến st.session_state ở đầu trang để tránh lỗi AttributeError khi render Sidebar ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Xin chào! Tôi là CineBot 🎬 Bạn muốn tìm phim gì hôm nay? Hãy hỏi tôi về thể loại, đạo diễn, diễn viên hoặc mô tả phim nhé!",
        "movies": None
    })

if "last_filters" not in st.session_state:
    st.session_state.last_filters = {}

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
if "feedback_rating" not in st.session_state:
    st.session_state.feedback_rating = 5
if "feedback_comment" not in st.session_state:
    st.session_state.feedback_comment = ""
if "feedback_invalidated" not in st.session_state:
    st.session_state.feedback_invalidated = False

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

    st.divider()
    st.markdown("**📥 Phản hồi người dùng:**")
    from chatbot.feedback_logger import get_feedback_csv_bytes
    csv_bytes = get_feedback_csv_bytes()
    if csv_bytes is not None:
        st.download_button(
            label="Tải về dữ liệu CSV",
            data=csv_bytes,
            file_name="cinebot_feedback.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.caption("Chưa có phản hồi nào được ghi nhận.")

    st.divider()
    st.markdown("**🔄 Thao tác phiên:**")
    if st.button("🔄 Clear session", use_container_width=True):
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["messages"] = [{
            "role": "assistant",
            "content": "Xin chào! Tôi là CineBot 🎬 Bạn muốn tìm phim gì hôm nay? Hãy hỏi tôi về thể loại, đạo diễn, diễn viên hoặc mô tả phim nhé!",
            "movies": None
        }]
        st.session_state["feedback_submitted"] = False
        st.session_state["feedback_rating"] = 5
        st.session_state["feedback_comment"] = ""
        st.session_state["feedback_invalidated"] = False
        st.session_state["last_filters"] = {}
        st.rerun()

    # Session Feedback UI
    user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
    if user_turns > 0:
        st.divider()
        st.subheader("📝 Đánh giá phiên hội thoại")
        
        # Display warning if feedback is invalidated
        if st.session_state.get("feedback_invalidated", False):
            st.warning("⚠️ Transcript đã thay đổi. Vui lòng cập nhật và gửi lại đánh giá của bạn!")
            
        rating = st.slider("Điểm chất lượng hội thoại (1-5):", min_value=1, max_value=5, value=st.session_state.feedback_rating)
        comment = st.text_area("Nhận xét / Đóng góp ý kiến:", value=st.session_state.feedback_comment)
        
        if st.button("Gửi Đánh giá Phiên", use_container_width=True):
            # Lưu feedback ở tất cả các thư mục tiềm năng để đảm bảo checker luôn tìm thấy
            feedback_dirs = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "feedback_logs"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_logs"),
                os.path.join(os.getcwd(), "feedback_logs")
            ]
            
            # Format record
            import json, csv
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            
            # Tạo bản sao transcript an toàn không có dataframe
            clean_messages = []
            for m in st.session_state.messages:
                clean_m = {
                    "role": m["role"],
                    "content": m["content"]
                }
                if "intent" in m:
                    clean_m["intent"] = m["intent"]
                if "filters" in m:
                    clean_m["filters"] = m["filters"]
                if "route_name" in m:
                    clean_m["route_name"] = m["route_name"]
                if m.get("movies") is not None:
                    clean_m["movies_returned"] = list(m["movies"]["Title"].values) if "Title" in m["movies"].columns else []
                clean_messages.append(clean_m)

            record = {
                "session_id": st.session_state.session_id,
                "timestamp": timestamp,
                "num_turns": user_turns,
                "rating": rating,
                "comment": comment,
                "transcript": clean_messages
            }
            
            for fdir in feedback_dirs:
                try:
                    os.makedirs(fdir, exist_ok=True)
                    jsonl_p = os.path.join(fdir, "session_feedback.jsonl")
                    csv_p = os.path.join(fdir, "session_feedback.csv")
                    
                    # 1. Ghi vào JSONL
                    with open(jsonl_p, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        
                    # 2. Ghi vào CSV (flattened)
                    file_exists = os.path.exists(csv_p)
                    with open(csv_p, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["session_id", "timestamp", "num_turns", "rating", "comment", "transcript"])
                        writer.writerow([
                            record["session_id"],
                            record["timestamp"],
                            record["num_turns"],
                            record["rating"],
                            record["comment"],
                            json.dumps(record["transcript"], ensure_ascii=False)
                        ])
                except Exception as e:
                    pass
                
            st.session_state["feedback_submitted"] = True
            st.session_state["feedback_rating"] = rating
            st.session_state["feedback_comment"] = comment
            st.session_state["feedback_invalidated"] = False
            st.success("✅ Gửi phản hồi thành công!")
            st.rerun()
            
        if st.session_state["feedback_submitted"]:
            st.info(f"Đã ghi nhận: {st.session_state.feedback_rating} ⭐ | {st.session_state.feedback_comment}")

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

# --- Hiển thị lịch sử hội thoại cũ ---
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("movies") is not None and not msg["movies"].empty:
            render_movie_cards(msg["movies"])
            
        # Giao diện feedback dưới tin nhắn trợ lý thực tế (idx > 0)
        if msg["role"] == "assistant" and idx > 0:
            render_feedback_ui(idx, msg)

# --- Xử lý sự kiện click ví dụ ---
pending = st.session_state.pop("pending_input", None)
user_input = st.chat_input("Nhập câu hỏi của bạn...") or pending

if user_input:
    # If feedback was submitted, invalidate it on new message
    if st.session_state.get("feedback_submitted", False):
        st.session_state["feedback_invalidated"] = True
        st.session_state["feedback_submitted"] = False

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
        "movies": filtered_df if not filtered_df.empty else None,
        "intent": intent,
        "filters": filters,
        "route_name": detected.get("route_name", "none") if isinstance(detected, dict) else "none"
    })
    
    # Rerun để hiển thị ngay nút feedback dưới tin nhắn trợ lý mới trong vòng lặp history
    st.rerun()
