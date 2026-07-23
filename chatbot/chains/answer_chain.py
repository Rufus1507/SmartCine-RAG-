import time
import logging
import pandas as pd
from langchain_core.language_models import BaseChatModel
from chatbot.prompts.answer_prompt import get_chitchat_prompt, get_rag_prompt
from chatbot.config import COL_TITLE, COL_GENRE, COL_DIRECTOR, COL_STARS, COL_YEAR, COL_RATING, COL_OVERVIEW, COL_LINK
from chatbot.rate_limiter import gemini_rate_limiter
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
def _invoke_llm_answer(llm, prompt: str):
    """Gọi LLM (invoke) với rate limiter + retry-on-429 cho answer chain."""
    gemini_rate_limiter.wait()
    t0 = time.monotonic()
    result = llm.invoke(prompt)
    logger.debug("[answer_chain] llm.invoke took %.2fs", time.monotonic() - t0)
    return result


def _stream_llm_answer(llm, prompt: str):
    """Gọi LLM (stream) với rate limiter cho answer chain. Không wrap retry vì generator."""
    gemini_rate_limiter.wait()
    return llm.stream(prompt)

def safe_print(text: str):
    """
    In nội dung an toàn, tránh lỗi UnicodeEncodeError trên các terminal Windows không hỗ trợ UTF-8.
    """
    try:
        print(text)
    except Exception:
        try:
            # Fallback in dạng ASCII
            print(str(text).encode('ascii', errors='replace').decode('ascii'))
        except Exception:
            pass

def run_answer_chain(llm: BaseChatModel, user_message: str, movies_df: pd.DataFrame, intent: str, stream: bool = False, trace: dict = None):
    """
    Thực thi Chain sinh câu trả lời (Tầng 2):
    1. Nhận danh sách phim, định dạng thành chuỗi văn bản ngữ cảnh (Context).
    2. Chọn Prompt template phù hợp (Chitchat hoặc RAG).
    3. Thực hiện gọi LLM dưới dạng invoke (đồng bộ) hoặc stream (nếu cấu hình hiển thị stream).
    4. Có cơ chế fallback trả về danh sách text cứng nếu kết nối LLM lỗi.
    """
    from chatbot.config import COL_YEAR
    # Xử lý trường hợp trò chuyện phiếm (chitchat)
    if intent == "chitchat":
        prompt_template = get_chitchat_prompt()
        prompt_input = {"input": user_message}
        
    # Xử lý khi không tìm thấy phim nào trong cơ sở dữ liệu
    elif movies_df.empty:
        prompt_template = get_chitchat_prompt()
        if intent == "info":
            prompt_input = {
                "input": (
                    f"Người dùng hỏi: \"{user_message}\"\n"
                    "Không xác định được bộ phim cụ thể nào trong cơ sở dữ liệu để cung cấp thông tin. "
                    "Hãy trả lời thân thiện và lịch sự hỏi người dùng muốn biết thông tin của bộ phim nào."
                )
            }
        else:
            prompt_input = {
                "input": (
                    f"Người dùng hỏi: \"{user_message}\"\n"
                    "Không tìm thấy phim nào phù hợp trong cơ sở dữ liệu. "
                    "Hãy trả lời thân thiện, gợi ý họ thử tìm kiếm với tiêu chí khác."
                )
            }
            
    # Xử lý khi tìm thấy danh sách phim (RAG)
    else:
        movies_info_list = []
        for _, row in movies_df.iterrows():
            if "final_context" in row and pd.notna(row["final_context"]):
                movie_str = f"- {row['final_context']}\n"
                # P1: Đưa tường minh năm phát hành và quốc gia sản xuất vào ngữ cảnh cho LLM trích xuất
                if COL_YEAR in row and pd.notna(row[COL_YEAR]):
                    movie_str += f"  Năm phát hành: {int(row[COL_YEAR])}\n"
                if "countries_origin" in row and pd.notna(row["countries_origin"]) and str(row["countries_origin"]).strip():
                    movie_str += f"  Quốc gia sản xuất: {row['countries_origin']}\n"
                if "graph_path_explanation" in row and pd.notna(row["graph_path_explanation"]) and str(row["graph_path_explanation"]).strip():
                    movie_str += f"  Liên kết: {row['graph_path_explanation']}\n"
                if COL_LINK in row and pd.notna(row[COL_LINK]):
                    movie_str += f"  Link IMDb: {row[COL_LINK]}\n"
            else:
                movie_str = (
                    f"- Tên phim: {row[COL_TITLE]}\n"
                    f"  Năm phát hành: {int(row[COL_YEAR]) if pd.notna(row[COL_YEAR]) else 'N/A'}\n"
                    f"  Quốc gia sản xuất: {row['countries_origin'] if 'countries_origin' in row and pd.notna(row['countries_origin']) else 'N/A'}\n"
                    f"  Thể loại: {row[COL_GENRE]}\n"
                    f"  Đạo diễn: {row[COL_DIRECTOR]}\n"
                    f"  Diễn viên: {row[COL_STARS]}\n"
                    f"  Điểm: {row[COL_RATING]}\n"
                )
                if COL_OVERVIEW in row and pd.notna(row[COL_OVERVIEW]):
                    movie_str += f"  Tóm tắt: {row[COL_OVERVIEW]}\n"
                if "graph_path_explanation" in row and pd.notna(row["graph_path_explanation"]) and str(row["graph_path_explanation"]).strip():
                    movie_str += f"  Liên kết: {row['graph_path_explanation']}\n"
                if COL_LINK in row and pd.notna(row[COL_LINK]):
                    movie_str += f"  Link IMDb: {row[COL_LINK]}\n"
            movies_info_list.append(movie_str)
            
        movies_info = "\n".join(movies_info_list)
        prompt_template = get_rag_prompt()
        prompt_input = {
            "input": user_message,
            "movies_info": movies_info
        }

    try:
        formatted_prompt = prompt_template.format(**prompt_input)
        if stream:
            # P1: Wrapper generator để kiểm tra finish_reason và bắt lỗi kết nối khi streaming
            def chunk_inspector():
                try:
                    stream_obj = _stream_llm_answer(llm, formatted_prompt)
                    for chunk in stream_obj:
                        if chunk.response_metadata and chunk.response_metadata.get("finish_reason") == "length":
                            msg = "⚠️ CẢNH BÁO: Câu trả lời bị cắt ngắn do vượt quá giới hạn token (max_tokens)."
                            safe_print(msg)
                            if trace is not None:
                                trace["answer_truncation_warning"] = msg
                        yield chunk
                except Exception as e:
                    # Fallback an toàn nếu LLM gặp sự cố kết nối giữa chừng khi streaming
                    safe_print(f"⚠️ Lỗi kết nối khi streaming: {e}")
                    if not movies_df.empty:
                        movie_list = []
                        for _, row in movies_df.head(5).iterrows():
                            movie_item = f"- **{row[COL_TITLE]}** ({int(row[COL_YEAR]) if pd.notna(row[COL_YEAR]) else 'N/A'}) - ⭐ {row[COL_RATING]}"
                            if "countries_origin" in row and pd.notna(row["countries_origin"]) and str(row["countries_origin"]).strip():
                                movie_item += f"\n  *Quốc gia:* {row['countries_origin']}"
                            if COL_OVERVIEW in row and pd.notna(row[COL_OVERVIEW]):
                                movie_item += f"\n  *Tóm tắt:* {row[COL_OVERVIEW][:120]}..."
                            if COL_LINK in row and pd.notna(row[COL_LINK]):
                                movie_item += f"\n  *Link IMDb:* {row[COL_LINK]}"
                            movie_list.append(movie_item)
                        movies_str = "\n".join(movie_list)
                        fallback_msg = (
                            f"Chào bạn! Kết nối với AI gặp sự cố nhỏ ({e}), "
                            f"nhưng tôi đã tìm thấy các phim phù hợp trong cơ sở dữ liệu:\n\n{movies_str}"
                        )
                        yield fallback_msg
                    else:
                        yield f"Lỗi gọi LLM (Tầng 2 - Sinh câu trả lời): {e}"
            return chunk_inspector()
        else:
            response = _invoke_llm_answer(llm, formatted_prompt)
            # P1: Kiểm tra finish_reason cho trường hợp invoke đồng bộ
            if response.response_metadata and response.response_metadata.get("finish_reason") == "length":
                msg = "⚠️ CẢNH BÁO: Câu trả lời bị cắt ngắn do vượt quá giới hạn token (max_tokens)."
                safe_print(msg)
                if trace is not None:
                    trace["answer_truncation_warning"] = msg
            return response.content.strip()
    except Exception as e:
        # Fallback an toàn nếu LLM lỗi kết nối
        if not movies_df.empty:
            movie_list = []
            for _, row in movies_df.head(5).iterrows():
                movie_item = f"- **{row[COL_TITLE]}** ({int(row[COL_YEAR]) if pd.notna(row[COL_YEAR]) else 'N/A'}) - ⭐ {row[COL_RATING]}"
                if "countries_origin" in row and pd.notna(row["countries_origin"]) and str(row["countries_origin"]).strip():
                    movie_item += f"\n  *Quốc gia:* {row['countries_origin']}"
                if COL_OVERVIEW in row and pd.notna(row[COL_OVERVIEW]):
                    movie_item += f"\n  *Tóm tắt:* {row[COL_OVERVIEW][:120]}..."
                if COL_LINK in row and pd.notna(row[COL_LINK]):
                    movie_item += f"\n  *Link IMDb:* {row[COL_LINK]}"
                movie_list.append(movie_item)
            movies_str = "\n".join(movie_list)
            fallback_msg = (
                f"Chào bạn! Kết nối với AI gặp sự cố nhỏ ({e}), "
                f"nhưng tôi đã tìm thấy các phim phù hợp trong cơ sở dữ liệu:\n\n{movies_str}"
            )
            if stream:
                def fallback_generator():
                    yield fallback_msg
                return fallback_generator()
            return fallback_msg
        
        err_msg = f"Lỗi gọi LLM (Tầng 2 - Sinh câu trả lời): {e}"
        if stream:
            def fallback_err_generator():
                yield err_msg
            return fallback_err_generator()
        return err_msg
