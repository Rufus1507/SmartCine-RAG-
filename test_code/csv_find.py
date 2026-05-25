import pandas as pd
import re
import sys

# Đảm bảo in tiếng Việt không bị lỗi trong console Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ============================================================
# Đường dẫn tới 2 file CSV
# ============================================================
FILE_1 = r"C:\Users\Admin\.cache\kagglehub\datasets\getaolga\moviepostersimdb\versions\1\MovieGenre.csv"
FILE_2 = r"C:\Users\Admin\.cache\kagglehub\datasets\ashpalsingh1525\imdb-movies-dataset\versions\1\imdb_movies.csv"

# ============================================================
# Hàm chuẩn hoá tên phim
# ============================================================
def normalize_title(title: str) -> str:
    """
    Chuẩn hoá tên phim để so sánh:
      1. Chuyển về chữ thường
      2. Xoá năm trong ngoặc ở cuối  vd: "Avatar (2009)" -> "avatar"
      3. Xoá mọi nội dung trong ngoặc ở cuối    vd: "Alien (Director's Cut)" -> "alien"
      4. Chuẩn hoá khoảng trắng thừa
    """
    t = title.lower().strip()
    # Xoá phần trong ngoặc đơn ở cuối chuỗi (có thể lặp nhiều lần)
    # vd: "title (2009) (extended)" -> "title"
    t = re.sub(r'\s*\(.*?\)\s*$', '', t).strip()
    # Chuẩn hoá nhiều khoảng trắng thành 1
    t = re.sub(r'\s+', ' ', t)
    return t

# ============================================================
# Đọc dữ liệu
# ============================================================
print("Đang đọc file 1...")
df1 = pd.read_csv(FILE_1, encoding="latin1", on_bad_lines="skip")

print("Đang đọc file 2...")
df2 = pd.read_csv(FILE_2, encoding="utf-8", on_bad_lines="skip")

# ============================================================
# Lấy cột tên phim gốc và tên đã chuẩn hoá
# ============================================================
titles1_raw = df1["Title"].dropna().str.strip()
titles2_raw = df2["names"].dropna().str.strip()

titles1_norm = titles1_raw.apply(normalize_title)
titles2_norm = titles2_raw.apply(normalize_title)

set1 = set(titles1_norm)
set2 = set(titles2_norm)

# Tìm tên trùng (dạng đã chuẩn hoá)
duplicates_norm = set1 & set2

print(f"\n✅ Tổng số tên trùng: {len(duplicates_norm)}")

# ============================================================
# Lấy lại tên gốc tương ứng từ cả 2 file
# ============================================================
# norm -> tên gốc đầu tiên tìm được (file 1)
norm_to_orig1 = {}
for raw, norm in zip(titles1_raw, titles1_norm):
    if norm not in norm_to_orig1:
        norm_to_orig1[norm] = raw

# norm -> tên gốc đầu tiên tìm được (file 2)
norm_to_orig2 = {}
for raw, norm in zip(titles2_raw, titles2_norm):
    if norm not in norm_to_orig2:
        norm_to_orig2[norm] = raw

# Sắp xếp theo tên chuẩn hoá
duplicates_sorted = sorted(duplicates_norm)

# ============================================================
# In ra màn hình
# ============================================================
print("\n--- Danh sách tên phim trùng ---")
print(f"{'STT':>4}  {'File 1 (Title)':<45}  {'File 2 (names)'}")
print("-" * 100)
for i, norm in enumerate(duplicates_sorted, 1):
    orig1 = norm_to_orig1.get(norm, "")
    orig2 = norm_to_orig2.get(norm, "")
    print(f"{i:>4}. {orig1:<45}  {orig2}")

# ============================================================
# Lưu kết quả ra file CSV
# ============================================================
output_path = "duplicate_titles.csv"
result_df = pd.DataFrame({
    "Title_File1":      [norm_to_orig1.get(n, "") for n in duplicates_sorted],
    "Title_File2":      [norm_to_orig2.get(n, "") for n in duplicates_sorted],
    "Title_normalized": duplicates_sorted,
})
result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n💾 Đã lưu kết quả vào: {output_path}")