import pandas as pd
from langchain_core.language_models import BaseChatModel
from chatbot.prompts.answer_prompt import get_chitchat_prompt, get_rag_prompt
from chatbot.config import COL_TITLE, COL_GENRE, COL_DIRECTOR, COL_STARS, COL_YEAR, COL_RATING, COL_OVERVIEW, COL_LINK

def run_answer_chain(llm: BaseChatModel, user_message: str, movies_df: pd.DataFrame, intent: str, stream: bool = False):
    """
    Thực thi Chain sinh câu trả lời (Tầng 2):
    1. Nhận danh sách phim, định dạng thành chuỗi văn bản ngữ cảnh (Context).
    2. Chọn Prompt template phù hợp (Chitchat hoặc RAG).
    3. Thực hiện gọi LLM dưới dạng invoke (đồng bộ) hoặc stream (nếu cấu hình hiển thị stream).
    4. Có cơ chế fallback trả về danh sách text cứng nếu kết nối LLM lỗi.
    """
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
                if "countries_origin" in row and pd.notna(row["countries_origin"]) and str(row["countries_origin"]).strip():
                    movie_str += f"  Quốc gia: {row['countries_origin']}\n"
                if "graph_path_explanation" in row and pd.notna(row["graph_path_explanation"]) and str(row["graph_path_explanation"]).strip():
                    movie_str += f"  Liên kết: {row['graph_path_explanation']}\n"
                if COL_LINK in row and pd.notna(row[COL_LINK]):
                    movie_str += f"  Link IMDb: {row[COL_LINK]}\n"
            else:
                movie_str = (
                    f"- Tên phim: {row[COL_TITLE]}\n"
                    f"  Thể loại: {row[COL_GENRE]}\n"
                    f"  Đạo diễn: {row[COL_DIRECTOR]}\n"
                    f"  Diễn viên: {row[COL_STARS]}\n"
                    f"  Năm: {row[COL_YEAR]}\n"
                    f"  Điểm: {row[COL_RATING]}\n"
                )
                if "countries_origin" in row and pd.notna(row["countries_origin"]) and str(row["countries_origin"]).strip():
                    movie_str += f"  Quốc gia: {row['countries_origin']}\n"
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
            # Trả về đối tượng generator để stream
            return llm.stream(formatted_prompt)
        else:
            response = llm.invoke(formatted_prompt)
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
                # Trả về generator giả lập để tương thích luồng stream
                def fallback_generator():
                    yield fallback_msg
                return fallback_generator()
            return fallback_msg
            
        err_msg = f"Lỗi gọi LLM (Tầng 2 - Sinh câu trả lời): {e}"
        if stream:
            def err_generator():
                yield err_msg
            return err_generator()
        return err_msg
