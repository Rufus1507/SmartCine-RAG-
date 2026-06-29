"""
find_duplicates.py

Mục đích: Điều tra và xuất danh sách các phim bị trùng lặp trong movie_master.csv,
dựa trên phát hiện ở báo cáo EDA: 11,121 dòng trùng theo `imdb_id` dù cột
`is_duplicate` báo 100% = 0 (flag bị sai).

Cách dùng:
    python find_duplicates.py "C:\\Users\\Admin\\Desktop\\4\\DAP391m\\code\\movie_master\\movie_master.csv"

Hoặc sửa trực tiếp biến CSV_PATH ở dưới rồi chạy: python find_duplicates.py
"""

import sys
import pandas as pd

# ---- Cấu hình ----
CSV_PATH = r"C:\Users\Admin\Desktop\4\DAP391m\code\movie_master\movie_master.csv"
OUTPUT_DIR = r"C:\Users\Admin\Desktop\4\DAP391m\code\movie_master"

# Các cột dùng để xác định trùng lặp — chỉnh nếu schema thực tế khác
ID_COL = "imdb_id"
TITLE_COL = "title"
YEAR_COL = "year"

# Các cột muốn xem khi xuất báo cáo (để dễ so sánh bằng mắt)
PREVIEW_COLS = [
    "imdb_id", "title", "year", "duration_min", "rating", "votes",
    "genres", "directors", "description", "completeness_score",
]


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    print(f"Đang đọc file: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Tổng số dòng: {len(df):,} | Tổng số cột: {len(df.columns)}")

    preview_cols = [c for c in PREVIEW_COLS if c in df.columns]

    # ------------------------------------------------------------------
    # 1. Duplicate theo imdb_id (khóa định danh duy nhất — quan trọng nhất)
    # ------------------------------------------------------------------
    if ID_COL in df.columns:
        dup_mask_id = df.duplicated(subset=[ID_COL], keep=False) & df[ID_COL].notna()
        dup_by_id = df[dup_mask_id].sort_values(ID_COL)

        n_dup_rows_id = len(dup_by_id)
        n_dup_groups_id = df.loc[dup_mask_id, ID_COL].nunique()

        print("\n=== DUPLICATE THEO imdb_id ===")
        print(f"Số dòng liên quan đến trùng lặp: {n_dup_rows_id:,}")
        print(f"Số imdb_id bị trùng (số 'cụm' trùng): {n_dup_groups_id:,}")

        out_path_id = f"{OUTPUT_DIR}/duplicates_by_imdb_id.csv"
        dup_by_id[preview_cols].to_csv(out_path_id, index=False, encoding="utf-8-sig")
        print(f"Đã lưu danh sách đầy đủ -> {out_path_id}")

        # Bảng tóm tắt: mỗi cụm trùng có bao nhiêu bản, có khác nhau ở cột nào không
        summary_rows = []
        for imdb_id, group in dup_by_id.groupby(ID_COL):
            row = {
                "imdb_id": imdb_id,
                "so_ban_trung": len(group),
                "title_co_giong_nhau": group[TITLE_COL].nunique() == 1 if TITLE_COL in df.columns else None,
                "cac_title": " | ".join(group[TITLE_COL].astype(str).unique()) if TITLE_COL in df.columns else None,
                "rating_co_giong_nhau": group["rating"].nunique(dropna=False) == 1 if "rating" in df.columns else None,
                "votes_co_giong_nhau": group["votes"].nunique(dropna=False) == 1 if "votes" in df.columns else None,
                "completeness_score_max": group["completeness_score"].max() if "completeness_score" in df.columns else None,
                "completeness_score_min": group["completeness_score"].min() if "completeness_score" in df.columns else None,
            }
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows).sort_values("so_ban_trung", ascending=False)
        out_path_summary = f"{OUTPUT_DIR}/duplicates_summary_by_imdb_id.csv"
        summary_df.to_csv(out_path_summary, index=False, encoding="utf-8-sig")
        print(f"Đã lưu bảng tóm tắt theo từng imdb_id -> {out_path_summary}")

        # In nhanh vài ví dụ để xem ngay trên terminal
        print("\n--- 5 ví dụ đầu tiên (mỗi cụm trùng) ---")
        for imdb_id, group in list(dup_by_id.groupby(ID_COL))[:5]:
            print(f"\nimdb_id = {imdb_id} | số bản = {len(group)}")
            print(group[preview_cols].to_string(index=False))
    else:
        print(f"⚠️ Không tìm thấy cột '{ID_COL}' trong dataset. Bỏ qua kiểm tra theo imdb_id.")

    # ------------------------------------------------------------------
    # 2. Duplicate theo title + year (bắt các trường hợp có thể không
    #    cùng imdb_id nhưng thực chất là cùng 1 phim do lỗi khác nguồn)
    # ------------------------------------------------------------------
    if TITLE_COL in df.columns and YEAR_COL in df.columns:
        dup_mask_ty = df.duplicated(subset=[TITLE_COL, YEAR_COL], keep=False) & df[TITLE_COL].notna()
        dup_by_title_year = df[dup_mask_ty].sort_values([TITLE_COL, YEAR_COL])

        print("\n=== DUPLICATE THEO title + year ===")
        print(f"Số dòng liên quan: {len(dup_by_title_year):,}")

        out_path_ty = f"{OUTPUT_DIR}/duplicates_by_title_year.csv"
        dup_by_title_year[preview_cols].to_csv(out_path_ty, index=False, encoding="utf-8-sig")
        print(f"Đã lưu danh sách đầy đủ -> {out_path_ty}")

    # ------------------------------------------------------------------
    # 3. Kiểm tra chéo: cờ is_duplicate có bắt đúng các dòng trên không?
    # ------------------------------------------------------------------
    if "is_duplicate" in df.columns and ID_COL in df.columns:
        flagged_correctly = df.loc[dup_mask_id, "is_duplicate"].sum() if ID_COL in df.columns else 0
        print("\n=== KIỂM TRA CỜ is_duplicate ===")
        print(f"Trong số {n_dup_rows_id:,} dòng bị trùng imdb_id thật, "
              f"chỉ có {flagged_correctly:,} dòng được cờ is_duplicate=1 đánh dấu đúng.")
        if flagged_correctly == 0:
            print("=> Xác nhận: cờ is_duplicate KHÔNG phát hiện được các duplicate này (bug).")

    print("\nHoàn tất. Hãy mở các file CSV vừa tạo để xem danh sách phim bị trùng.")


if __name__ == "__main__":
    main()