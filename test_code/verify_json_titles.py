import pandas as pd
import json
import sys
import os

# Đảm bảo in tiếng Việt không bị lỗi trong console Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

CSV_FILE = r"C:\Users\Admin\Desktop\4\DAP391m\code\json_titles_not_in_csv.csv"
JSON_FILE = r"C:\Users\Admin\Desktop\4\DAP391m\code\movie_ids_05_15_2026.json"

def main():
    print("--------------------------------------------------")
    print("Bắt đầu kiểm tra...")
    print("--------------------------------------------------")

    # 1. Đọc file JSON (dạng JSON Lines / JSONL)
    print("Đang đọc JSON...")
    json_titles = set()
    total_json_lines = 0
    parsed_json_lines = 0
    
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        for line in f:
            total_json_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    val = rec.get("original_title")
                    if val is not None:
                        json_titles.add(str(val).strip())
                        parsed_json_lines += 1
            except json.JSONDecodeError:
                pass
                
    print(f"Tổng số dòng trong file JSON: {total_json_lines:,}")
    print(f"Số dòng parse thành công: {parsed_json_lines:,}")
    print(f"Số original_title duy nhất trong JSON: {len(json_titles):,}")
    print("--------------------------------------------------")

    # 2. Đọc file CSV
    print("Đang đọc CSV...")
    if not os.path.exists(CSV_FILE):
        print(f"Lỗi: Không tìm thấy file CSV tại {CSV_FILE}")
        return

    # Sử dụng pandas để đọc
    df = pd.read_csv(CSV_FILE)
    if "original_title" not in df.columns:
        print(f"Lỗi: File CSV không có cột 'original_title'. Các cột hiện có: {list(df.columns)}")
        return

    # Lấy danh sách titles từ CSV
    csv_titles_series = df["original_title"].dropna().astype(str).str.strip()
    total_csv_titles = len(csv_titles_series)
    unique_csv_titles = csv_titles_series.unique()
    
    print(f"Tổng số titles trong CSV: {total_csv_titles:,}")
    print(f"Số titles duy nhất trong CSV: {len(unique_csv_titles):,}")
    print("--------------------------------------------------")

    # 3. So sánh đối chiếu
    print("Đang đối chiếu dữ liệu...")
    found_count = 0
    missing_titles = []

    for title in unique_csv_titles:
        if title in json_titles:
            found_count += 1
        else:
            missing_titles.append(title)

    # 4. Hiển thị kết quả
    print("\nKết quả đối chiếu:")
    print(f"  - Số titles trong CSV tìm thấy trong JSON: {found_count:,} ({found_count/len(unique_csv_titles)*100:.2f}%)")
    print(f"  - Số titles trong CSV KHÔNG tìm thấy trong JSON: {len(missing_titles):,} ({len(missing_titles)/len(unique_csv_titles)*100:.2f}%)")

    if missing_titles:
        print("\nMột số titles bị thiếu tiêu biểu (tối đa 20):")
        for i, title in enumerate(missing_titles[:20], 1):
            print(f"  {i}. {title}")
            
        # Xuất ra file
        missing_output_file = "missing_titles_report.csv"
        pd.DataFrame({"original_title": missing_titles}).to_csv(missing_output_file, index=False, encoding="utf-8-sig")
        print(f"\nDanh sách tất cả {len(missing_titles):,} titles bị thiếu đã được ghi vào file: {missing_output_file}")
    else:
        print("\n✔ Tuyệt vời! Tất cả các titles trong CSV đều được tìm thấy trong JSON.")

if __name__ == "__main__":
    main()
