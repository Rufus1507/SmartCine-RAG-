import os
import sys
import time
import uuid
import pandas as pd
import streamlit as st

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration & loaders
from chatbot.config import (
    KEYWORD_DICT_PATH, ALIASES_PATH,
    COL_TITLE, COL_GENRE, COL_DIRECTOR, COL_STARS, COL_YEAR, COL_RATING, COL_OVERVIEW, COL_LINK,
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    GEMINI_DEFAULT_KEY, GEMINI_DEFAULT_MODEL,
    update_env_variable
)
from chatbot.data_loader import (
    load_data, load_keyword_dict, load_aliases, load_faiss_index,
    load_traditional_faiss_index, load_embedder_model
)
from chatbot.llm_client import get_llm_client

# Import Traditional RAG pipeline
from eval.traditional_rag import run_traditional_rag_pipeline, retrieve_traditional

# Import CineBot V3 pipeline for Side-by-Side comparison
CINEBOT_IMPORT_ERROR = None
try:
    from chatbot.chains.rag_chain import run_rag_pipeline as run_cinebot_pipeline
    CINEBOT_AVAILABLE = True
except Exception as e:
    CINEBOT_AVAILABLE = False
    CINEBOT_IMPORT_ERROR = str(e)
    import traceback
    CINEBOT_IMPORT_ERROR = traceback.format_exc()

# ============================================================
# CẤU HÌNH TRANG STREAMLIT & CUSTOM STYLING
# ============================================================
st.set_page_config(
    page_title="Traditional RAG vs CineBot V3 Benchmark UI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại (Dark Theme Glassmorphism)
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 12px;
        backdrop-filter: blur(10px);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #9ea4b0;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #4facfe;
    }
    
    /* System Badge */
    .badge-trad {
        background: linear-gradient(135deg, #ff7e5f, #feb47b);
        color: #111;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-cinebot {
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        color: #fff;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* Movie Card Styling */
    .movie-card-box {
        background: #1a1d24;
        border-left: 4px solid #4facfe;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    
    /* Comparison Container */
    .comp-box-trad {
        border: 1px solid rgba(255, 126, 95, 0.3);
        border-radius: 12px;
        padding: 16px;
        background: rgba(255, 126, 95, 0.02);
    }
    .comp-box-cinebot {
        border: 1px solid rgba(0, 198, 255, 0.3);
        border-radius: 12px;
        padding: 16px;
        background: rgba(0, 198, 255, 0.02);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS DISPLAY
# ============================================================
def render_movie_cards_traditional(df: pd.DataFrame):
    """Hiển thị danh sách phim kết quả từ RAG truyền thống."""
    if df.empty:
        st.warning("⚠️ Không tìm thấy phim phù hợp trong cơ sở dữ liệu.")
        return
        
    cols = st.columns(min(len(df), 5))
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % len(cols)]:
            title = row.get(COL_TITLE, "Unassigned")
            rating = row.get(COL_RATING, "N/A")
            year = int(row.get(COL_YEAR)) if pd.notna(row.get(COL_YEAR)) else "N/A"
            sim = row.get("similarity", None)
            
            st.markdown(f"**🎬 {title}**")
            if sim is not None:
                st.caption(f"🎯 Similarity: `{float(sim):.4f}`")
            st.caption(f"⭐ {rating}  •  📅 {year}")
            
            if COL_DIRECTOR in row and pd.notna(row[COL_DIRECTOR]):
                st.caption(f"🎬 Đạo diễn: {row[COL_DIRECTOR]}")
            if COL_GENRE in row and pd.notna(row[COL_GENRE]):
                st.caption(f"🎭 Thể loại: {row[COL_GENRE]}")
            if COL_STARS in row and pd.notna(row[COL_STARS]):
                stars = [s.strip() for s in str(row[COL_STARS]).split(",") if s.strip()]
                st.caption(f"👥 Diễn viên: {', '.join(stars[:2])}")
                
            if COL_OVERVIEW in row and pd.notna(row[COL_OVERVIEW]):
                overview = str(row[COL_OVERVIEW]).strip()
                if len(overview) > 100:
                    overview = overview[:100] + "..."
                st.caption(f"📝 {overview}")
                
            if COL_LINK in row and pd.notna(row[COL_LINK]):
                st.markdown(f"[🔗 Xem IMDb]({row[COL_LINK]})")


# ============================================================
# MAIN APPLICATION LOGIC
# ============================================================
def main():
    # Session state initialization
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages_trad" not in st.session_state:
        st.session_state.messages_trad = []
    if "messages_compare" not in st.session_state:
        st.session_state.messages_compare = []

    # Header
    st.title("🎬 SmartCine Traditional RAG & Benchmark UI")
    st.caption("Giao diện kiểm thử RAG Truyền thống (Naive Vector Search) & So sánh đối chiếu với CineBotV3")

    # ============================================================
    # SIDEBAR CONFIGURATION
    # ============================================================
    st.sidebar.header("⚙️ Cấu hình Hệ thống")

    # Select Mode
    mode = st.sidebar.radio(
        "📌 Chọn chế độ hoạt động:",
        ["RAG Truyền thống (Standalone)", "⚔️ So sánh song song (Side-by-Side)"],
        index=0
    )

    st.sidebar.markdown("---")

    # LLM Settings
    st.sidebar.subheader("🤖 Cấu hình LLM")
    llm_provider = st.sidebar.selectbox(
        "Nhà cung cấp LLM:",
        ["Ollama Server", "Local LLM / Custom API", "Gemini API"],
        index=0
    )

    if llm_provider == "Ollama Server":
        current_ollama_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
        current_ollama_model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)

        ollama_url = st.sidebar.text_input("Ollama Base URL:", value=current_ollama_url)
        ollama_model = st.sidebar.text_input("Model Name:", value=current_ollama_model)

        if st.sidebar.button("Lưu cấu hình Ollama"):
            update_env_variable("OLLAMA_BASE_URL", ollama_url)
            update_env_variable("OLLAMA_MODEL", ollama_model)
            st.sidebar.success("Đã lưu cấu hình Ollama Server!")

    elif llm_provider == "Gemini API":
        current_gemini_key = os.getenv("GEMINI_API_KEY", GEMINI_DEFAULT_KEY)
        gemini_key = st.sidebar.text_input(
            "Gemini API Key:",
            value=current_gemini_key,
            type="password",
            help="Lấy key miễn phí tại Google AI Studio"
        )
        gemini_model = st.sidebar.selectbox(
            "Mô hình Gemini:",
            ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
            index=0
        )
        if gemini_key != current_gemini_key:
            update_env_variable("GEMINI_API_KEY", gemini_key)
            st.sidebar.success("Đã cập nhật Gemini API Key!")

    else:
        current_base_url = os.getenv("LLM_BASE_URL", LLM_BASE_URL)
        current_api_key = os.getenv("LLM_API_KEY", LLM_API_KEY)
        current_model_name = os.getenv("LLM_MODEL", LLM_MODEL)

        base_url = st.sidebar.text_input("Base URL:", value=current_base_url)
        api_key = st.sidebar.text_input("API Key:", value=current_api_key, type="password")
        model_name = st.sidebar.text_input("Model Name:", value=current_model_name)

        if st.sidebar.button("Lưu cấu hình Local LLM"):
            update_env_variable("LLM_BASE_URL", base_url)
            update_env_variable("LLM_API_KEY", api_key)
            update_env_variable("LLM_MODEL", model_name)
            st.sidebar.success("Đã lưu cấu hình Local LLM!")

    st.sidebar.markdown("---")

    # Retrieval parameters
    st.sidebar.subheader("🔍 Tham số Retrieval")
    top_k = st.sidebar.slider("Top K kết quả vector (Naive Retrieval):", min_value=1, max_value=10, value=5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Thông tin Kiến trúc")
    st.sidebar.info(
        "**Traditional Naive RAG**:\n"
        "- Vectorize câu hỏi trực tiếp bằng SentenceTransformer.\n"
        "- Cosine similarity trên FAISS index (`traditional_context.index`).\n"
        "- Đưa trực tiếp Top-K context thô vào Prompt LLM.\n"
        "- *Không* có Pandas Attribute Filtering, *Không* Graph RAG, *Không* BM25/Reranker."
    )

    # Initialize data & models
    with st.spinner("⏳ Đang tải dữ liệu phim & chỉ mục FAISS..."):
        try:
            df = load_data()
            embedder_model = load_embedder_model()
            traditional_index = load_traditional_faiss_index()
            
            if CINEBOT_AVAILABLE:
                cinebot_index = load_faiss_index()
                keyword_dict = load_keyword_dict()
                aliases_dict = load_aliases()
        except Exception as e:
            st.error(f"❌ Lỗi khi khởi tạo tài nguyên: {e}")
            st.stop()

    # Get LLM Client
    try:
        if llm_provider == "Ollama Server":
            active_url = ollama_url if 'ollama_url' in locals() and ollama_url else OLLAMA_BASE_URL
            active_model = ollama_model if 'ollama_model' in locals() and ollama_model else OLLAMA_MODEL
            llm = get_llm_client(provider="Ollama Server", base_url=active_url, api_key="any", model_name=active_model)
        elif llm_provider == "Gemini API":
            active_key = gemini_key if 'gemini_key' in locals() and gemini_key else GEMINI_DEFAULT_KEY
            active_model = gemini_model if 'gemini_model' in locals() else GEMINI_DEFAULT_MODEL
            llm = get_llm_client(provider="Gemini API", api_key=active_key, model_name=active_model)
        else:
            active_url = base_url if 'base_url' in locals() else LLM_BASE_URL
            active_key = api_key if 'api_key' in locals() else LLM_API_KEY
            active_model = model_name if 'model_name' in locals() else LLM_MODEL
            llm = get_llm_client(provider="Local LLM", base_url=active_url, api_key=active_key, model_name=active_model)
    except Exception as e:
        st.error(f"❌ Không thể kết nối với LLM: {e}")
        st.stop()


    # ============================================================
    # MODE 1: STANDALONE TRADITIONAL RAG
    # ============================================================
    if mode == "RAG Truyền thống (Standalone)":
        st.subheader("🟧 Giao diện Chatbot RAG Truyền Thống (Naive Vector Search)")

        # Render message history
        for msg in st.session_state.messages_trad:
            with st.chat_message(msg["role"]):
                # Hiển thị tiêu đề rõ ràng cho câu trả lời LLM trong lịch sử
                if msg["role"] == "assistant":
                    st.markdown(f"### 💬 Câu trả lời từ LLM (Traditional RAG):")
                    st.markdown(msg["content"])
                else:
                    st.markdown(msg["content"])
                if "metrics" in msg:
                    m = msg["metrics"]
                    st.caption(f"⏱️ Retrieval: `{m['retrieval_ms']:.1f}ms` | LLM: `{m['llm_ms']:.1f}ms` | Total: `{m['total_ms']:.1f}ms` | Top-K: `{m['count']}` phim")
                if "movies" in msg and not msg["movies"].empty:
                    with st.expander("🎬 Xem danh sách phim được trích xuất (Top Vector)", expanded=False):
                        render_movie_cards_traditional(msg["movies"])
                if "context_str" in msg:
                    with st.expander("🔍 Context Inspector (Prompt & Ngữ cảnh thô gửi cho LLM)", expanded=False):
                        st.code(msg["context_str"], language="text")

        # Chat Input
        query = st.chat_input("Nhập câu hỏi tìm phim của bạn (Ví dụ: Phim hành động hay nhất thập niên 90)...")
        if query:
            # Display user message
            st.session_state.messages_trad.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # Process Traditional RAG
            with st.chat_message("assistant"):
                with st.spinner("🧠 Đang truy xuất vector & sinh câu trả lời..."):
                    start_total = time.time()
                    
                    # Gọi pipeline 1 lần duy nhất (tránh double retrieval)
                    start_ret = time.time()
                    answer_text, retrieved_df = run_traditional_rag_pipeline(
                        query, llm, df, traditional_index, embedder_model, top_k=top_k
                    )
                    retrieval_ms = (time.time() - start_ret) * 1000
                    llm_ms = retrieval_ms  # pipeline đã bao gồm cả retrieval + generation
                    
                    total_ms = (time.time() - start_total) * 1000

                    # Prepare context string preview for inspector
                    context_lines = []
                    for idx, r in retrieved_df.iterrows():
                        context_lines.append(f"[{r.get(COL_TITLE)}] (Similarity: {r.get('similarity', 0):.4f})\n{r.get('final_context')}\n")
                    context_str = "\n".join(context_lines)

                    # Hiển thị câu trả lời từ LLM rõ ràng
                    st.markdown("### 💬 Câu trả lời từ LLM (Traditional RAG):")
                    st.markdown(answer_text)
                    st.caption(f"⏱️ Total: `{total_ms:.1f}ms` | Found: `{len(retrieved_df)}` phim")
                    
                    if not retrieved_df.empty:
                        with st.expander("🎬 Danh sách phim trích xuất (Naive Vector Match)", expanded=True):
                            render_movie_cards_traditional(retrieved_df)
                            
                    with st.expander("🔍 Context Inspector (Prompt & Context thô)", expanded=False):
                        st.code(context_str, language="text")

                    # Save to state
                    st.session_state.messages_trad.append({
                        "role": "assistant",
                        "content": answer_text,
                        "movies": retrieved_df,
                        "context_str": context_str,
                        "metrics": {
                            "retrieval_ms": retrieval_ms,
                            "llm_ms": llm_ms,
                            "total_ms": total_ms,
                            "count": len(retrieved_df)
                        }
                    })


    # ============================================================
    # MODE 2: DUAL SIDE-BY-SIDE COMPARISON
    # ============================================================
    else:
        st.subheader("⚔️ Màn Hình So Sánh Song Song: Traditional RAG vs CineBot V3")
        st.caption("Nhập câu hỏi 1 lần để hệ thống gửi đồng thời tới cả 2 pipeline và đối chiếu trực tiếp kết quả trả về")

        query_comp = st.chat_input("Nhập câu hỏi test so sánh (Ví dụ: Phim sci-fi IMDb > 8.5 sau năm 2010)...")
        
        # Display history of comparisons
        for item in st.session_state.messages_compare:
            st.markdown(f"#### ❓ Câu hỏi: **{item['query']}**")
            col_t, col_c = st.columns(2)
            
            with col_t:
                st.markdown("<span class='badge-trad'>🟧 Traditional Naive RAG</span>", unsafe_allow_html=True)
                st.markdown(f"**💬 Câu trả lời LLM:**\n\n{item['trad_ans']}")
                m_t = item["trad_metrics"]
                st.caption(f"⏱️ Total: `{m_t['total_ms']:.1f}ms` | Retrieval: `{m_t['retrieval_ms']:.1f}ms` | LLM: `{m_t['llm_ms']:.1f}ms` | Found: `{m_t['count']}`")
                if not item["trad_movies"].empty:
                    with st.expander("📜 Top Phim Vector Matched", expanded=False):
                        render_movie_cards_traditional(item["trad_movies"])
                        
            with col_c:
                st.markdown("<span class='badge-cinebot'>🟦 CineBot V3 (Hybrid Graph-Pandas)</span>", unsafe_allow_html=True)
                st.markdown(f"**💬 Câu trả lời LLM:**\n\n{item['cine_ans']}")
                m_c = item["cine_metrics"]
                st.caption(f"⏱️ Total: `{m_c['total_ms']:.1f}ms` | Intent: `{item['cine_intent']}` | Filters: `{item['cine_filters']}` | Found: `{m_c['count']}`")
                if not item["cine_movies"].empty:
                    with st.expander("📜 Top Phim CineBot Filtered & Reranked", expanded=False):
                        render_movie_cards_traditional(item["cine_movies"])

            st.markdown("---")

        # Handle new query submission
        if query_comp:
            st.markdown(f"#### ❓ Câu hỏi đang xử lý: **{query_comp}**")
            col_trad_active, col_cine_active = st.columns(2)

            # Execute Traditional RAG
            with col_trad_active:
                st.markdown("<span class='badge-trad'>🟧 Traditional Naive RAG</span>", unsafe_allow_html=True)
                with st.spinner("⏳ Đang xử lý Naive Vector RAG..."):
                    t0 = time.time()
                    t_ret0 = time.time()
                    trad_df = retrieve_traditional(query_comp, traditional_index, embedder_model, df, top_k=top_k)
                    t_ret_ms = (time.time() - t_ret0) * 1000
                    
                    t_llm0 = time.time()
                    trad_answer, _ = run_traditional_rag_pipeline(query_comp, llm, df, traditional_index, embedder_model, top_k=top_k)
                    t_llm_ms = (time.time() - t_llm0) * 1000
                    t_total_ms = (time.time() - t0) * 1000

                    st.markdown(f"**💬 Câu trả lời LLM:**\n\n{trad_answer}")
                    st.caption(f"⏱️ Total: `{t_total_ms:.1f}ms` | Retrieval: `{t_ret_ms:.1f}ms` | LLM: `{t_llm_ms:.1f}ms` | Found: `{len(trad_df)}`")
                    if not trad_df.empty:
                        with st.expander("📜 Top Phim Vector Matched", expanded=True):
                            render_movie_cards_traditional(trad_df)

            # Execute CineBot V3
            with col_cine_active:
                st.markdown("<span class='badge-cinebot'>🟦 CineBot V3 (Hybrid Graph-Pandas)</span>", unsafe_allow_html=True)
                if not CINEBOT_AVAILABLE:
                    st.error("❌ CineBot V3 pipeline không sẵn sàng — import lỗi.")
                    if CINEBOT_IMPORT_ERROR:
                        with st.expander("🔍 Chi tiết lỗi import CineBot", expanded=True):
                            st.code(CINEBOT_IMPORT_ERROR, language="text")
                    cine_answer = "Lỗi nạp CineBot V3 pipeline."
                    cine_df = pd.DataFrame()
                    cine_intent = "none"
                    cine_filters = {}
                    t_cine_ms = 0.0
                else:
                    with st.spinner("⏳ Đang xử lý CineBot V3 Hybrid Search..."):
                        tc0 = time.time()
                        try:
                            res = run_cinebot_pipeline(
                                llm=llm,
                                user_input=query_comp,
                                df=df,
                                keyword_dict=keyword_dict,
                                aliases_dict=aliases_dict,
                                faiss_index=cinebot_index,
                                embedder_model=embedder_model,
                                chat_history=[],
                                last_filters={},
                                stream=False
                            )
                            cine_answer = res[0]
                            cine_df = res[1]
                            cine_intent = res[2]
                            cine_filters = res[3]
                        except Exception as ex:
                            cine_answer = f"Lỗi CineBot V3: {ex}"
                            cine_df = pd.DataFrame()
                            cine_intent = "error"
                            cine_filters = {}
                        t_cine_ms = (time.time() - tc0) * 1000

                        st.markdown(cine_answer)
                        st.caption(f"⏱️ Total: `{t_cine_ms:.1f}ms` | Intent: `{cine_intent}` | Filters: `{cine_filters}` | Found: `{len(cine_df)}`")
                        if not cine_df.empty:
                            with st.expander("📜 Top Phim CineBot Filtered & Reranked", expanded=True):
                                render_movie_cards_traditional(cine_df)

            # Save comparison run to session state
            st.session_state.messages_compare.append({
                "query": query_comp,
                "trad_ans": trad_answer,
                "trad_movies": trad_df,
                "trad_metrics": {
                    "total_ms": t_total_ms,
                    "retrieval_ms": t_ret_ms,
                    "llm_ms": t_llm_ms,
                    "count": len(trad_df)
                },
                "cine_ans": cine_answer,
                "cine_movies": cine_df,
                "cine_intent": cine_intent,
                "cine_filters": cine_filters,
                "cine_metrics": {
                    "total_ms": t_cine_ms,
                    "count": len(cine_df)
                }
            })


if __name__ == "__main__":
    main()
