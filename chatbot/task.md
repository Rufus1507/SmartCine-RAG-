Bạn hãy refactor dự án CineBot hiện tại theo yêu cầu sau:

Mục tiêu của task này: Tách phần xử lý tìm kiếm/lọc phim thành module Tool riêng để chuẩn bị tích hợp Agentic RAG và Harness.

Bối cảnh:

* File chính hiện tại là `chatbot/app.py`.
* Trong `app.py` đang có các hàm như `apply_filters`, `semantic_search`, `generate_answer`, `parse_intent`, `detect_entities`.
* Hệ thống hiện tại dùng Streamlit, Pandas, FAISS, SentenceTransformer và LLM để gợi ý phim.

Yêu cầu thực hiện:

1. Tạo file mới:

`chatbot/tools.py`

2. Trong `tools.py`, tạo các tool sau:

* `search_movies_tool(df, filters, top_k=5)`
* `semantic_search_tool(query, df, index, model, top_k=100)`
* `get_movie_detail_tool(df, title)`
* `recommend_by_actor_tool(df, actor, top_k=5)`
* `recommend_by_director_tool(df, director, top_k=5)`
* `compare_movies_tool(df, movie_titles)`

3. Di chuyển logic lọc phim chính từ `apply_filters` trong `app.py` sang `search_movies_tool`.

4. Tool phải trả về dữ liệu dạng rõ ràng, dễ dùng:

* Nếu trả DataFrame thì giữ nguyên DataFrame.
* Nếu lỗi hoặc không có kết quả thì trả DataFrame rỗng, không crash app.
* Không để tool gọi LLM.
* Tool chỉ chịu trách nhiệm xử lý dữ liệu thật từ dataset.

5. Giữ tương thích với code cũ:

* Không làm mất chức năng hiện tại.
* `app.py` vẫn chạy Streamlit bình thường.
* Nếu cần, sửa `app.py` để gọi `search_movies_tool` thay cho `apply_filters`.
* Các hàm cũ nếu chưa xóa được thì có thể giữ wrapper tạm thời.

6. Thêm docstring ngắn cho mỗi tool, ví dụ:

```python
def search_movies_tool(...):
    """
    Filter movies from dataframe based on parsed filters.
    This tool does not call LLM.
    """
```

7. Đảm bảo xử lý an toàn các cột có thể bị thiếu:

* Title
* Genre
* Director
* Stars
* Year
* Rating
* Votes
* Description

8. Sau khi sửa xong, hãy cho tôi biết:

* Đã tạo file nào mới.
* Đã sửa file nào.
* Những hàm nào đã được chuyển sang tool.
* Cách `app.py` hiện tại gọi tool mới.
* Có lỗi tiềm năng nào còn cần kiểm tra không.

Lưu ý quan trọng:

* Không thay đổi giao diện Streamlit.
* Không thay đổi prompt LLM ở task này.
* Không thêm Harness ở task này.
* Không thêm Agent Router ở task này.
* Chỉ tập trung tách Tool Layer trước.
