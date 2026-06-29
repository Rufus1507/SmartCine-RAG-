import pandas as pd
import ast
import re
from collections import Counter
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load 2 bảng
df1 = pd.read_csv(r"C:\Users\Admin\Desktop\4\DAP391m\code\data\advanced_movies_details_all_years.csv", low_memory=False)   # có genres, directors, stars, writers
df2 = pd.read_csv(r"C:\Users\Admin\Desktop\4\DAP391m\code\data\imdb_movies_all_years.csv", low_memory=False)   # có title, description, rating, year

# Merge theo cột link (key chung giữa 2 bảng)
df = pd.merge(df2, df1, left_on="Movie Link", right_on="link", how="left")

# Parse list string → list thực sự
# Dữ liệu của nhóm có dạng "['Horror', 'Thriller']" → cần chuyển thành list
def safe_parse_list(val):
    if pd.isna(val) or val == "":
        return []
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except:
        # fallback: tách bằng dấu phẩy nếu literal_eval lỗi
        return [x.strip().strip("'\"") for x in str(val).strip("[]").split(",") if x.strip()]

df["genres_list"]    = df["genres"].apply(safe_parse_list)
df["directors_list"] = df["directors"].apply(safe_parse_list)
df["stars_list"]     = df["stars"].apply(safe_parse_list)
df["writers_list"]   = df["writers"].apply(safe_parse_list)
df["languages_list"] = df["Languages"].apply(safe_parse_list)
# ── 2A. GENRE KEYWORDS ──────────────────────────────────────
# Lấy tất cả genre xuất hiện trong toàn bộ dataset
all_genres = []
for g_list in df["genres_list"]:
    all_genres.extend([g.strip() for g in g_list if g and g != "None"])

genre_counter = Counter(all_genres)

# Từ điển genre: keyword → intent
# Chỉ giữ genre xuất hiện >= 10 lần để tránh noise
GENRE_KEYWORDS = {
    genre.lower(): "search_genre"
    for genre, count in genre_counter.items()
    if count >= 10
}

print(f"Số genre keywords: {len(GENRE_KEYWORDS)}")
print(dict(list(GENRE_KEYWORDS.items())[:10]))


# ── 2B. ENTITY KEYWORDS (người) ─────────────────────────────
PERSON_KEYWORDS = {}

# 1. Writers (Độ ưu tiên thấp nhất)
all_writers = []
for w_list in df["writers_list"]:
    all_writers.extend([w.strip() for w in w_list if w and w != "None"])

for name in set(all_writers):
    if name:
        PERSON_KEYWORDS[name.lower()] = "search_writer"

# 2. Stars / diễn viên (Độ ưu tiên trung bình - ghi đè writers)
all_stars = []
for s_list in df["stars_list"]:
    all_stars.extend([s.strip() for s in s_list if s and s != "None"])

star_counter = Counter(all_stars)
# Chỉ lấy diễn viên xuất hiện >= 2 phim (loại bỏ extra)
for name, count in star_counter.items():
    if name and count >= 2:
        PERSON_KEYWORDS[name.lower()] = "search_star"

# 3. Directors (Độ ưu tiên cao nhất - ghi đè stars và writers)
all_directors = []
for d_list in df["directors_list"]:
    all_directors.extend([d.strip() for d in d_list if d and d != "None"])

for name in set(all_directors):
    if name:
        PERSON_KEYWORDS[name.lower()] = "search_director"

print(f"Số person keywords: {len(PERSON_KEYWORDS)}")


# ── 2C. CONTENT KEYWORDS từ description (TF-IDF) ────────────
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Chỉ lấy các phim có description
desc_df = df[df["description"].notna() & (df["description"] != "")].copy()

vectorizer = TfidfVectorizer(
    max_features=500,       # top 500 từ đặc trưng nhất
    stop_words="english",
    ngram_range=(1, 2),     # unigram + bigram
    min_df=5                # từ phải xuất hiện trong >= 5 phim
)

tfidf_matrix = vectorizer.fit_transform(desc_df["description"])
content_keywords = vectorizer.get_feature_names_out()

CONTENT_KEYWORDS = {
    kw: "search_content"
    for kw in content_keywords
}

print(f"Số content keywords: {len(CONTENT_KEYWORDS)}")
print("Ví dụ:", list(content_keywords[:20]))
import json

# Gộp tất cả lại
KEYWORD_DICT = {}
KEYWORD_DICT.update(CONTENT_KEYWORDS)   # thêm trước (ưu tiên thấp)
KEYWORD_DICT.update(GENRE_KEYWORDS)     # ghi đè nếu trùng
KEYWORD_DICT.update(PERSON_KEYWORDS)    # ưu tiên cao nhất

print(f"\nTổng số keywords trong dictionary: {len(KEYWORD_DICT)}")

# Lưu ra các file để dùng lại
with open("keyword_dict.json", "w", encoding="utf-8") as f:
    json.dump(KEYWORD_DICT, f, ensure_ascii=False, indent=2)

import os
chatbot_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chatbot", "keyword_dict.json")
with open(chatbot_dir_path, "w", encoding="utf-8") as f:
    json.dump(KEYWORD_DICT, f, ensure_ascii=False, indent=2)

# Xem thống kê phân bổ intent
from collections import Counter
intent_dist = Counter(KEYWORD_DICT.values())
print("\nPhân bổ intent:")
for intent, count in intent_dist.most_common():
    print(f"  {intent}: {count} keywords")