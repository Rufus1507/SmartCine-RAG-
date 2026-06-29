import pandas as pd
import json
import os
import sys

# Đảm bảo in tiếng Việt không bị lỗi trong console Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN FILE
# ============================================================
CSV_FILE  = r"C:\Users\Admin\Desktop\4\DAP391m\code\merged_output\imdb_movies_all_years.csv"
JSON_FILE = r"C:\Users\Admin\Desktop\4\DAP391m\code\movie_ids_05_15_2026.json"

# File kết quả xuất ra (cùng thư mục với script này)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_IN_CSV_NOT_JSON = os.path.join(OUTPUT_DIR, "csv_titles_missing_from_json.csv")
OUT_IN_JSON_NOT_CSV = os.path.join(OUTPUT_DIR, "json_titles_not_in_csv.csv")

# ============================================================
# 1. ĐỌC FILE CSV  →  lấy cột "Title"
# ============================================================
print("Đang đọc CSV ...")
df_csv = pd.read_csv(CSV_FILE, usecols=["Title"], encoding="utf-8", low_memory=False)

# Chuẩn hoá: strip khoảng trắng, bỏ dòng rỗng, giữ thứ tự ban đầu và loại bỏ trùng lặp
csv_titles = []
csv_titles_set = set()
for t in df_csv["Title"].dropna().astype(str).str.strip():
    if t not in csv_titles_set:
        csv_titles_set.add(t)
        csv_titles.append(t)
print(f"  → Tổng số title duy nhất trong CSV : {len(csv_titles):,}")

# ============================================================
# 2. ĐỌC FILE JSON  →  lấy trường "original_title"
# ============================================================
print("Đang đọc JSON ...")
records = []
try:
    # Thử đọc dưới dạng JSON bình thường
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    # Hỗ trợ cả 2 cấu trúc phổ biến:
    #   - List trực tiếp:  [ {"original_title": "...", ...}, ... ]
    #   - Dict bọc ngoài:  {"results": [...]}  hoặc  {"movies": [...]}
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        # Thử tìm key chứa list
        list_key = next((k for k, v in raw.items() if isinstance(v, list)), None)
        records = raw[list_key] if list_key else []
except json.JSONDecodeError:
    # Nếu không phải JSON bình thường, thử JSONL (JSON Lines)
    print("  → Đang thử đọc dưới dạng JSONL (JSON Lines)...")
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

json_titles = []
json_titles_set = set()
for rec in records:
    if isinstance(rec, dict):
        val = rec.get("original_title")
        if val and str(val).strip():
            title = str(val).strip()
            if title not in json_titles_set:
                json_titles_set.add(title)
                json_titles.append(title)

print(f"  → Tổng số original_title duy nhất trong JSON: {len(json_titles):,}")

# ============================================================
# 3. SO SÁNH
# ============================================================
in_csv_not_json = [t for t in csv_titles if t not in json_titles_set]
in_json_not_csv = [t for t in json_titles if t not in csv_titles_set]

print(f"\nKết quả so sánh:")
print(f"  Có trong CSV  nhưng KHÔNG có trong JSON : {len(in_csv_not_json):,}")
print(f"  Có trong JSON nhưng KHÔNG có trong CSV  : {len(in_json_not_csv):,}")

# ============================================================
# 4. XUẤT RA FILE
# ============================================================
pd.DataFrame({"Title": in_csv_not_json}).to_csv(
    OUT_IN_CSV_NOT_JSON, index=False, encoding="utf-8-sig"
)

pd.DataFrame({"original_title": in_json_not_csv}).to_csv(
    OUT_IN_JSON_NOT_CSV, index=False, encoding="utf-8-sig"
)

print(f"\nĐã xuất file:")
print(f"  ✔ {OUT_IN_CSV_NOT_JSON}")
print(f"  ✔ {OUT_IN_JSON_NOT_CSV}")
print("\nHoàn thành!")