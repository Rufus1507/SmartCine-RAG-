import os
import sys
import time
import json
import pandas as pd

# Cấu hình đầu ra console UTF-8 để hiển thị ký tự tiếng Việt và emoji trên Windows
sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc của project vào sys.path
chatbot_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workspace_dir = os.path.dirname(chatbot_dir)
sys.path.append(workspace_dir)

from chatbot.data_loader import load_data, load_keyword_dict, load_aliases, load_faiss_index, load_embedder_model
from chatbot.llm_client import get_llm_client
from chatbot.chains.rag_chain import run_rag_pipeline

def run_harness():
    print("============================================================")
    print("CINEBOT EVALUATION HARNESS V2")
    print("============================================================")
    
    # 1. Nạp dữ liệu
    print("⏳ Đang nạp dữ liệu phim và từ điển...")
    try:
        df = load_data()
        keyword_dict = load_keyword_dict()
        aliases_dict = load_aliases()
        faiss_index = load_faiss_index()
        embedder_model = load_embedder_model()
        print(f"✅ Đã nạp thành công: {len(df):,} phim, {len(keyword_dict):,} từ khóa.")
    except Exception as e:
        print(f"❌ Lỗi nạp dữ liệu: {e}")
        return

    # 2. Khởi tạo LLM Client (Mặc định dùng Local LLM để test nội bộ)
    provider = os.getenv("LLM_PROVIDER", "Local LLM")
    api_key = os.getenv("LLM_API_KEY", "any")
    model_name = os.getenv("LLM_MODEL", "cx/gpt-5.5")
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:20128/v1")
    
    print(f"🔌 LLM Provider: {provider} | Model: {model_name}")
    try:
        llm = get_llm_client(provider, api_key, model_name, base_url)
    except Exception as e:
        print(f"❌ Không thể tạo LLM client: {e}")
        return

    # 3. Định nghĩa bộ test cases kiểm thử nâng cấp V2
    test_cases = [
        # Nhóm 1: Intent & Basic Filters
        {
            "group": "Basic Filters",
            "query": "Tìm phim hành động của Christopher Nolan trên 8 điểm",
            "expected_intent": "search",
            "expected_filters": {"genre": "Action", "director": "Christopher Nolan", "rating_min": 8.0}
        },
        {
            "group": "Basic Filters",
            "query": "Gợi ý phim khoa học viễn tưởng hay nhất năm 2019",
            "expected_intent": "search",
            "expected_filters": {"genre": "Sci-Fi", "year_min": 2019}
        },
        {
            "group": "Basic Filters",
            "query": "Thông tin tóm tắt nội dung phim Titanic",
            "expected_intent": "info",
            "expected_filters": {"title": "Titanic"}
        },
        {
            "group": "Chitchat",
            "query": "Thời tiết hôm nay ở Hà Nội thế nào?",
            "expected_intent": "chitchat",
            "expected_filters": {}
        },
        {
            "group": "Basic Filters",
            "query": "Phim hài của Leonardo DiCaprio sau năm 2010",
            "expected_intent": "search",
            "expected_filters": {"genre": "Comedy", "star": "Leonardo DiCaprio"}
        },
        # Nhóm 2: BM25 (Tên phim, Từ khóa đặc trưng)
        {
            "group": "BM25 Keyword",
            "query": "Iron Man",
            "expected_intent": ["search", "info"],
            "expected_filters": {"title": "Iron Man"},
            "check_retrieved_title": "Iron Man"
        },
        {
            "group": "BM25 Keyword",
            "query": "Titanic",
            "expected_intent": ["search", "info"],
            "expected_filters": {"title": "Titanic"},
            "check_retrieved_title": "Titanic"
        },
        {
            "group": "BM25 Keyword",
            "query": "Phim siêu anh hùng Marvel",
            "expected_intent": "search",
            "expected_min_movies": 1
        },
        # Nhóm 3: Diễn viên (Actor Search)
        {
            "group": "Actor Search",
            "query": "Tìm phim của diễn viên Leonardo DiCaprio",
            "expected_intent": "search",
            "expected_filters": {"star": "Leonardo DiCaprio"},
            "check_retrieved_star": "Leonardo DiCaprio"
        },
        {
            "group": "Actor Search",
            "query": "Phim của Tom Cruise",
            "expected_intent": "search",
            "expected_filters": {"star": "Tom Cruise"},
            "check_retrieved_star": "Tom Cruise"
        },
        # Nhóm 4: Đạo diễn (Director Search)
        {
            "group": "Director Search",
            "query": "Các phim do Christopher Nolan đạo diễn",
            "expected_intent": "search",
            "expected_filters": {"director": "Christopher Nolan"},
            "check_retrieved_director": "Christopher Nolan"
        },
        {
            "group": "Director Search",
            "query": "James Cameron director",
            "expected_intent": "search",
            "expected_filters": {"director": "James Cameron"},
            "check_retrieved_director": "James Cameron"
        },
        # Nhóm 5: Quốc gia (Country Search)
        {
            "group": "Country Search",
            "query": "Phim sản xuất tại Việt Nam",
            "expected_intent": "search",
            "expected_filters": {"country": "Vietnam"},
            "check_retrieved_country": "Vietnam"
        },
        {
            "group": "Country Search",
            "query": "Phim Hàn Quốc tuyển chọn",
            "expected_intent": "search",
            "expected_filters": {"country": "South Korea"},
            "check_retrieved_country": "South Korea"
        },
        {
            "group": "Country Search",
            "query": "Tìm phim Nhật Bản hay nhất",
            "expected_intent": "search",
            "expected_filters": {"country": "Japan"},
            "check_retrieved_country": "Japan"
        },
        # Nhóm 6: Phim tương tự (Similar Movies V2)
        {
            "group": "Similar Movie V2",
            "query": "Phim nào giống phim Inception",
            "expected_intent": "search",
            "check_similar_exclude": "Inception",
            "expected_min_movies": 1
        },
        {
            "group": "Similar Movie V2",
            "query": "Phim tương tự Titanic",
            "expected_intent": "search",
            "check_similar_exclude": "Titanic",
            "expected_min_movies": 1
        },
        {
            "group": "Similar Movie V2",
            "query": "Gợi ý phim giống phim Interstellar",
            "expected_intent": "search",
            "check_similar_exclude": "Interstellar",
            "expected_min_movies": 1
        }
    ]

    results = []
    
    # Các biến đếm tính metric
    correct_intents = 0
    correct_filters = 0
    correct_retrievals = 0
    country_matches = 0
    country_total = 0
    similar_matches = 0
    similar_total = 0
    
    total_latency = 0.0

    print(f"\n🚀 Bắt đầu đánh giá {len(test_cases)} trường hợp kiểm thử...")
    print("-" * 100)

    for idx, tc in enumerate(test_cases):
        query = tc["query"]
        group = tc["group"]
        start_time = time.time()
        
        chat_history = []
        last_filters = {}
        
        try:
            # Chạy pipeline RAG
            answer, filtered_df, intent, filters, detected = run_rag_pipeline(
                llm, query, df, keyword_dict, aliases_dict, faiss_index, embedder_model,
                chat_history, last_filters, stream=False
            )
            
            latency = time.time() - start_time
            total_latency += latency
            
            # 1. Đánh giá Intent
            expected_intent = tc["expected_intent"]
            if isinstance(expected_intent, list):
                is_intent_correct = (intent in expected_intent)
            else:
                is_intent_correct = (intent == expected_intent)
            if is_intent_correct:
                correct_intents += 1
                
            # 2. Đánh giá Filters
            is_filter_correct = True
            expected_f = tc.get("expected_filters", {})
            for k, v in expected_f.items():
                pred_v = filters.get(k)
                if not pred_v:
                    is_filter_correct = False
                    break
                if str(v).lower() not in str(pred_v).lower():
                    is_filter_correct = False
                    break
            if is_filter_correct:
                correct_filters += 1
                
            # 3. Đánh giá Retrieval Accuracy & Nội dung
            is_retrieval_correct = True
            
            # Kiểm tra số lượng tối thiểu
            expected_min = tc.get("expected_min_movies", 0)
            if expected_min > 0 and len(filtered_df) < expected_min:
                is_retrieval_correct = False
                
            # Kiểm tra khớp Title
            expected_title = tc.get("check_retrieved_title")
            if expected_title and not filtered_df.empty:
                title_found = any(expected_title.lower() in str(t).lower() for t in filtered_df["Title"].values)
                if not title_found:
                    is_retrieval_correct = False
                    
            # Kiểm tra khớp Diễn viên
            expected_star = tc.get("check_retrieved_star")
            if expected_star and not filtered_df.empty:
                star_found = any(expected_star.lower() in str(s).lower() for s in filtered_df["stars"].values)
                if not star_found:
                    is_retrieval_correct = False
                    
            # Kiểm tra khớp Đạo diễn
            expected_dir = tc.get("check_retrieved_director")
            if expected_dir and not filtered_df.empty:
                dir_found = any(expected_dir.lower() in str(d).lower() for d in filtered_df["directors"].values)
                if not dir_found:
                    is_retrieval_correct = False
                    
            # Đánh giá Quốc gia (Country Search Accuracy)
            expected_country = tc.get("check_retrieved_country")
            if expected_country:
                country_total += 1
                if not filtered_df.empty:
                    # Kiểm tra xem có cột countries_origin không và có chứa đúng quốc gia không
                    country_found = all(expected_country.lower() in str(c).lower() for c in filtered_df["countries_origin"].values if pd.notna(c))
                    if country_found:
                        country_matches += 1
                    else:
                        is_retrieval_correct = False
                else:
                    is_retrieval_correct = False
                    
            # Đánh giá Phim tương tự (Similar Movie Quality)
            exclude_title = tc.get("check_similar_exclude")
            if exclude_title:
                similar_total += 1
                if not filtered_df.empty:
                    # Đảm bảo không chứa chính bộ phim gốc
                    contains_original = any(exclude_title.lower() == str(t).lower() for t in filtered_df["Title"].values)
                    # Đảm bảo có lý do tương đồng và điểm tương đồng trong kết quả
                    has_reasons = all("similarity_reason" in filtered_df.columns for _ in [1])
                    if not contains_original and has_reasons:
                        similar_matches += 1
                    else:
                        is_retrieval_correct = False
                else:
                    is_retrieval_correct = False
                    
            if is_retrieval_correct and (expected_min > 0 or expected_title or expected_star or expected_dir or expected_country or exclude_title or not filtered_df.empty or intent == "chitchat"):
                correct_retrievals += 1
            else:
                is_retrieval_correct = False
                
            status = "PASS" if (is_intent_correct and is_filter_correct and is_retrieval_correct) else "FAIL"
            
            results.append({
                "case_id": idx + 1,
                "group": group,
                "query": query,
                "expected_intent": tc["expected_intent"],
                "predicted_intent": intent,
                "intent_correct": is_intent_correct,
                "expected_filters": expected_f,
                "filters_applied": filters,
                "filters_correct": is_filter_correct,
                "movies_retrieved": len(filtered_df),
                "retrieval_correct": is_retrieval_correct,
                "latency_seconds": round(latency, 3),
                "status": status
            })
            
            print(f"[{group}] Case #{idx+1}: '{query}'")
            print(f"  -> Intent: {intent} ({'✅' if is_intent_correct else '❌'}) | Filters: {filters} ({'✅' if is_filter_correct else '❌'})")
            print(f"  -> Phim tìm thấy: {len(filtered_df)} ({'✅' if is_retrieval_correct else '❌'}) | Trễ: {latency:.3f}s | Trạng thái: {status}")
            print("-" * 100)
            
        except Exception as ex:
            latency = time.time() - start_time
            results.append({
                "case_id": idx + 1,
                "group": group,
                "query": query,
                "expected_intent": tc["expected_intent"],
                "predicted_intent": "error",
                "intent_correct": False,
                "expected_filters": tc.get("expected_filters", {}),
                "filters_applied": {},
                "filters_correct": False,
                "movies_retrieved": 0,
                "retrieval_correct": False,
                "latency_seconds": round(latency, 3),
                "status": f"ERROR: {str(ex)}"
            })
            print(f"[{group}] Case #{idx+1}: '{query}' -> ❌ BỊ LỖI: {ex}")
            print("-" * 100)

    # 4. Tính toán số liệu thống kê tổng thể
    total_cases = len(test_cases)
    intent_acc = correct_intents / total_cases
    filter_acc = correct_filters / total_cases
    retrieval_acc = correct_retrievals / total_cases
    country_acc = (country_matches / country_total) if country_total > 0 else 1.0
    similar_quality = (similar_matches / similar_total) if similar_total > 0 else 1.0
    avg_latency = total_latency / total_cases
    overall_pass_rate = sum(1 for r in results if r["status"] == "PASS") / total_cases

    summary = {
        "total_tests": total_cases,
        "overall_pass_rate": overall_pass_rate,
        "intent_accuracy": intent_acc,
        "filter_accuracy": filter_acc,
        "retrieval_accuracy": retrieval_acc,
        "country_accuracy": country_acc,
        "similar_movie_quality": similar_quality,
        "average_latency_seconds": avg_latency,
        "details": results
    }

    # 5. Xuất báo cáo dạng JSON
    report_json_path = os.path.join(chatbot_dir, "evaluation", "evaluation_report.json")
    os.makedirs(os.path.dirname(report_json_path), exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 6. Xuất báo cáo dạng Markdown
    report_md_path = os.path.join(chatbot_dir, "evaluation", "evaluation_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# 📊 Báo cáo Đánh giá CineBot RAG V2 (Retrieval V2 Upgrade)\n\n")
        f.write(f"- **Thời gian chạy**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Tỉ lệ Pass toàn bộ**: **{overall_pass_rate*100:.1f}%** ({sum(1 for r in results if r['status'] == 'PASS')}/{total_cases})\n\n")
        
        f.write("### 📈 Chỉ số chất lượng chính (Key Metrics)\n")
        f.write(f"- **Độ chính xác Intent (Intent Accuracy)**: {intent_acc*100:.1f}%\n")
        f.write(f"- **Độ chính xác Lọc thuộc tính (Filter Accuracy)**: {filter_acc*100:.1f}%\n")
        f.write(f"- **Độ chính xác Truy xuất (Retrieval Accuracy)**: {retrieval_acc*100:.1f}%\n")
        f.write(f"- **Độ chính xác Quốc gia (Country Search Accuracy)**: {country_acc*100:.1f}%\n")
        f.write(f"- **Chất lượng gợi ý phim tương đồng (Similar Movie Quality)**: {similar_quality*100:.1f}%\n")
        f.write(f"- **Độ trễ trung bình (Average Latency)**: {avg_latency:.3f} giây\n\n")
        
        f.write("## Chi tiết kết quả kiểm thử các Test Case\n\n")
        f.write("| ID | Nhóm | Câu hỏi | Ý định kỳ vọng | Ý định dự đoán | Lọc thuộc tính | Số phim tìm thấy | Độ trễ (giây) | Trạng thái |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['case_id']} | {r['group']} | {r['query']} | {r['expected_intent']} | {r['predicted_intent']} | {r['filters_applied']} | {r['movies_retrieved']} | {r['latency_seconds']} | {r['status']} |\n")

    print("\n🎉 ĐÁNH GIÁ HOÀN TẤT!")
    print(f"  - Tỉ lệ Pass toàn bộ: {overall_pass_rate*100:.1f}%")
    print(f"  - Độ chính xác trích xuất Intent: {intent_acc*100:.1f}%")
    print(f"  - Độ chính xác trích xuất Filters: {filter_acc*100:.1f}%")
    print(f"  - Độ chính xác Truy xuất phim: {retrieval_acc*100:.1f}%")
    print(f"  - Độ chính xác Tìm phim quốc gia: {country_acc*100:.1f}%")
    print(f"  - Chất lượng phim tương đồng: {similar_quality*100:.1f}%")
    print(f"  - Độ trễ phản hồi trung bình: {avg_latency:.3f}s")
    print(f"  - Báo cáo JSON: {report_json_path}")
    print(f"  - Báo cáo Markdown: {report_md_path}")

if __name__ == "__main__":
    run_harness()
