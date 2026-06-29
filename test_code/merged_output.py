"""
Script merge dữ liệu IMDB từ các thư mục theo năm (1920-2025).
Sẽ tạo ra 3 file merged trong thư mục merged_output (cùng chỗ với script):
  - advanced_movies_details_all_years.csv   (không có STT)
  - imdb_movies_all_years.csv               (STT liên tục từ 1)
  - merged_movies_data_all_years.csv        (STT liên tục từ 1)
"""

import os
import re
import pandas as pd

# ============================================================
# CẤU HÌNH - Chỉnh sửa đường dẫn nếu cần
# ============================================================
BASE_DIR   = r"C:\Users\Admin\.cache\kagglehub\datasets\raedaddala\imdb-movies-from-1960-to-2023\versions\6\Data"
YEAR_START = 1920
YEAR_END   = 2025
# ============================================================

# Cấu hình từng loại file:
#   "reset_index" : True  => bỏ cột STT cũ, đánh lại STT liên tục
#                   False => giữ nguyên, không thêm STT
FILE_CONFIGS = {
    "advanced_movies_details": {"reset_index": False},
    "imdb_movies":             {"reset_index": True},
    "merged_movies_data":      {"reset_index": True},
}

# Thư mục output nằm cùng chỗ với file script này
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "merged_output")


def read_csv_safe(file_path: str):
    """Đọc CSV, thử utf-8 trước rồi fallback latin-1."""
    try:
        return pd.read_csv(file_path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin-1", low_memory=False)


def drop_old_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xóa cột STT/index cũ nếu có.
    Nhận diện: cột đầu tiên là kiểu số nguyên VÀ
    tên cột là 'Unnamed: 0', 'index', 'stt', hoặc chuỗi rỗng.
    """
    if df.empty:
        return df
    first_col = df.columns[0]
    auto_names = {"unnamed: 0", "index", "stt", ""}
    if str(first_col).strip().lower() in auto_names:
        df = df.drop(columns=[first_col])
    return df


def clean_title_numbering(df: pd.DataFrame) -> pd.DataFrame:
    """Xóa số thứ tự bị dính trong cột Title, ví dụ '1. Movie' -> 'Movie'."""
    if "Title" not in df.columns:
        return df

    df = df.copy()
    df["Title"] = df["Title"].apply(
        lambda value: re.sub(r"^\s*\d+\s*[\.)]\s*", "", value).strip()
        if isinstance(value, str)
        else value
    )
    return df


def merge_files_by_type(base_dir: str, output_dir: str, year_start: int, year_end: int):
    os.makedirs(output_dir, exist_ok=True)

    for file_type, config in FILE_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Đang xử lý: {file_type}")
        print(f"{'='*60}")

        frames        = []
        missing_years = []
        error_years   = []

        for year in range(year_start, year_end + 1):
            file_path = os.path.join(base_dir, str(year), f"{file_type}_{year}.csv")

            if not os.path.exists(file_path):
                missing_years.append(year)
                continue

            try:
                df = read_csv_safe(file_path)
            except Exception as e:
                error_years.append((year, str(e)))
                print(f"  ✗ {year}: Lỗi - {e}")
                continue

            # Nếu file có STT cũ thì bỏ đi trước khi gom
            if config["reset_index"]:
                df = drop_old_index(df)

            # Một số file không có cột STT riêng mà dính STT vào Title: "1. Tên phim".
            df = clean_title_numbering(df)

            frames.append(df)
            print(f"  ✓ {year}: {len(df):,} dòng")

        if missing_years:
            print(f"\n  Không tìm thấy file cho các năm: {missing_years}")
        if error_years:
            print(f"\n  Lỗi khi đọc file:")
            for yr, err in error_years:
                print(f"    Năm {yr}: {err}")

        if not frames:
            print(f"\n  Không có dữ liệu nào! Bỏ qua {file_type}.")
            continue

        merged_df = pd.concat(frames, ignore_index=True)

        # Đánh STT liên tục chỉ với file cần reset
        if config["reset_index"]:
            merged_df.insert(0, "stt", range(1, len(merged_df) + 1))

        output_file = os.path.join(output_dir, f"{file_type}_all_years.csv")
        merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"\n  => Đã lưu : {output_file}")
        print(f"     Tổng   : {len(merged_df):,} dòng  |  {len(merged_df.columns)} cột")
        print(f"     Từ     : {len(frames)} năm có dữ liệu")


def main():
    print("IMDB Data Merger")
    print(f"Thư mục gốc : {BASE_DIR}")
    print(f"Thư mục xuất: {OUTPUT_DIR}")
    print(f"Khoảng năm  : {YEAR_START} → {YEAR_END}")

    if not os.path.exists(BASE_DIR):
        print(f"\n[LỖI] Không tìm thấy thư mục: {BASE_DIR}")
        print("Vui lòng kiểm tra lại đường dẫn BASE_DIR trong script.")
        return

    merge_files_by_type(BASE_DIR, OUTPUT_DIR, YEAR_START, YEAR_END)

    print(f"\n{'='*60}")
    print("Hoàn thành! Kiểm tra kết quả tại:")
    print(f"  {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()