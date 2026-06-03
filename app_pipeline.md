# 🎬 Pipeline của `app.py`

```mermaid
flowchart TD
    A(["👤 User nhập câu hỏi\nst.chat_input / sidebar button"]) 

    A --> B["load_data()\n@st.cache_data\nĐọc 2 CSV → pd.merge()\nclean list columns\nép kiểu Rating, Year"]

    B --> C["parse_intent(user_message)\n━━━━━━━━━━━━━━━━━\nLLM Tầng 1\ntemp=0.1 · max_tokens=300\nsystem = SYSTEM_PROMPT"]

    C --> D["JSON Parser\nre.search(r'\\{.*\\}', raw)\njson.loads(raw)\n↓ fallback nếu lỗi → chitchat"]

    D --> E{intent?}

    E -- "search / recommend" --> F["apply_filters(df, filters)\n━━━━━━━━━━━━━━━━━\nstr.contains: genre, director\nstar, title (case-insensitive)\n>= : year_min, year_max\n>= : rating_min\nsort Rating DESC → head(5)"]

    E -- "chitchat" --> G["filtered_df = DataFrame rỗng"]

    F --> H{Có kết quả?}
    G --> I

    H -- "Có phim" --> I["generate_answer(user_msg, movies_df, intent)\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nLLM Tầng 2 · max_tokens=1024\ncontext = movies_info string\nsystem = 'Trợ lý phim thân thiện'"]

    H -- "Rỗng" --> I

    I --> J["st.write(answer)\nrender_movie_cards(filtered_df)\n━━━━━━━━━━━━━━━━━━━━━━━\nst.columns(≤5)\nTitle · ⭐Rating · Year\nDirector · Genre · Actors(3)"]

    J --> K["Lưu vào session_state.messages\n{role, content, movies}"]

    K --> A
```

---

### Các hàm chính & vai trò

| Hàm | Dòng | Vai trò |
|-----|------|---------|
| `load_data()` | 35 | Đọc CSV, merge, làm sạch → DataFrame |
| `parse_intent()` | 114 | LLM Tầng 1: câu hỏi → JSON filters |
| `apply_filters()` | 144 | Pandas filter + sort → Top 5 |
| `generate_answer()` | 177 | LLM Tầng 2: dữ liệu → câu trả lời |
| `render_movie_cards()` | 212 | Hiển thị card phim dạng cột |
