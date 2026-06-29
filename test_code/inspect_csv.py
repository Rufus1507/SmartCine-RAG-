import pandas as pd
import json
import os

csv_path = r"c:\Users\Admin\Desktop\4\DAP391m\code\movie_master\movie_master.csv"

print(f"Checking if file exists: {os.path.exists(csv_path)}")
print(f"File size: {os.path.getsize(csv_path) / (1024*1024):.2f} MB")

# Read only first 5 rows to see structure
df_sample = pd.read_csv(csv_path, nrows=5)
print("\n--- Columns and Types (Sample) ---")
print(df_sample.dtypes)
print("\n--- First 2 rows ---")
print(df_sample.head(2).to_dict(orient="records"))

# Let's count rows
print("\nCounting lines...")
row_count = 0
with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        row_count += 1
print(f"Total lines in CSV (including header): {row_count}")

# Read columns info
df = pd.read_csv(csv_path, nrows=100)
print("\nColumn names list:")
print(df.columns.tolist())
