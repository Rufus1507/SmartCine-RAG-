import os
import sys
import pandas as pd

# Đảm bảo in UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào path để import dạng 'from chatbot.xyz'
chatbot_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(chatbot_dir)

from chatbot.data_loader import load_data, load_faiss_index, load_embedder_model
from chatbot.retrieval.retrieval_router import route_retrieval

def main():
    print("🎬 Tải dữ liệu phim và mô hình...")
    df = load_data()
    faiss_index = load_faiss_index()
    embedder_model = load_embedder_model()
    
    # 1. Truy vấn test: Phim giống Inception
    query = "Tìm phim giống như Inception"
    filters = {"title": "Inception"}
    intent = "search"
    
    print(f"\n🚀 Đang chạy thử nghiệm truy vấn: '{query}'")
    results, route_name = route_retrieval(
        query=query,
        df=df,
        filters=filters,
        intent=intent,
        faiss_index=faiss_index,
        embedder_model=embedder_model,
        final_k=5
    )
    
    print(f"✔️ Định tuyến: {route_name}")
    print(f"✔️ Kết quả trả về ({len(results)} phim):")
    
    for i, (_, row) in enumerate(results.iterrows()):
        year_val = row.get('Year')
        year_str = str(int(year_val)) if pd.notna(year_val) else "N/A"
        print(f"\nPhim {i+1}: {row['Title']} ({year_str})")
        print(f"  - Điểm tương đồng: {row.get('similarity_score')}")
        print(f"  - Lý do tương đồng: {row.get('similarity_reason')}")
        if "graph_path_explanation" in row and isinstance(row["graph_path_explanation"], str):
            print(f"  - Đường dẫn Graph: {row['graph_path_explanation']}")
        else:
            print("  - Đường dẫn Graph: [Không đi qua graph]")

if __name__ == "__main__":
    main()
