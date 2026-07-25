"""
compare_50q.py
--------------
So sánh kết quả của RAG Truyền Thống vs CineBot V3 trên bộ 50 câu hỏi đại diện.
Sinh báo cáo Markdown đầy đủ ra file eval/50q_comparison_report.md.
"""
import os
import sys
import json
import time
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

sys.stdout.reconfigure(encoding='utf-8')

import builtins
def print(*args, **kwargs):
    builtins.print(*args, flush=True, **kwargs)

from langchain_openai import ChatOpenAI
from chatbot.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_stats(results):
    """Tính thống kê cơ bản theo category và difficulty."""
    by_cat = defaultdict(list)
    by_diff = defaultdict(list)
    latencies = []
    errors = 0
    for r in results:
        by_cat[r.get("category", "unknown")].append(r)
        by_diff[r.get("difficulty", "unknown")].append(r)
        latencies.append(r.get("latency_s", 0))
        if r.get("error"):
            errors += 1
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    return {
        "total": len(results),
        "errors": errors,
        "avg_latency_s": round(avg_lat, 2),
        "by_category": {k: len(v) for k, v in by_cat.items()},
        "by_difficulty": {k: len(v) for k, v in by_diff.items()},
    }


def main():
    trad_path   = os.path.join(ROOT_DIR, "eval", "traditional_50q_results.json")
    cinebot_path = os.path.join(ROOT_DIR, "eval", "cinebot_50q_results.json")
    report_path = os.path.join(ROOT_DIR, "eval", "50q_comparison_report.md")

    # Kiểm tra file đầu vào
    missing = [p for p in [trad_path, cinebot_path] if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"❌ Không tìm thấy file: {p}")
        sys.exit(1)

    print("📂 Đang tải kết quả...")
    trad_results    = load_json(trad_path)
    cinebot_results = load_json(cinebot_path)
    print(f"   Traditional RAG : {len(trad_results)} kết quả")
    print(f"   CineBot V3      : {len(cinebot_results)} kết quả")

    trad_stats    = build_stats(trad_results)
    cinebot_stats = build_stats(cinebot_results)

    # Map theo id để ghép đôi
    trad_map    = {r["id"]: r for r in trad_results}
    cinebot_map = {r["id"]: r for r in cinebot_results}

    comparison_data = []
    for q_id in sorted(trad_map.keys(), key=lambda x: int(x[1:])):
        t = trad_map.get(q_id, {})
        c = cinebot_map.get(q_id, {})
        comparison_data.append({
            "id":       q_id,
            "question": t.get("question", ""),
            "level":    t.get("level"),
            "difficulty": t.get("difficulty"),
            "category": t.get("category"),
            "traditional": {
                "movies":  [m.get("title") for m in t.get("movies", [])],
                "answer":  t.get("answer_result", "")[:600],   # Cắt ngắn để tiết kiệm token LLM
                "latency": t.get("latency_s"),
                "error":   t.get("error")
            },
            "cinebot_v3": {
                "movies":  [m.get("title") for m in c.get("movies", [])],
                "answer":  c.get("answer_result", "")[:600],
                "latency": c.get("latency_s"),
                "error":   c.get("error")
            }
        })

    print("\n🧠 Khởi tạo LLM để sinh báo cáo...")
    try:
        llm = ChatOpenAI(
            openai_api_key=LLM_API_KEY,
            openai_api_base=LLM_BASE_URL,
            model_name=LLM_MODEL,
            temperature=0.2,
            max_tokens=6000
        )
    except Exception as e:
        print(f"❌ Lỗi khởi tạo LLM: {e}")
        sys.exit(1)

    prompt = f"""Bạn là chuyên gia cao cấp về Retrieval-Augmented Generation (RAG) và AI Search.

Hãy viết một bản BÁO CÁO KHOA HỌC SO SÁNH bằng tiếng Việt, format Markdown đẹp, chi tiết và sâu sắc, đánh giá kết quả thực nghiệm trên bộ 50 câu hỏi đại diện (10 cấp độ, 7 loại category).

HỆ THỐNG SO SÁNH:
1. **Traditional RAG (Optimized)**: Cosine Similarity FlatIP + embedding tinh gọn (Title+Desc+Genres) + không có BM25/Pandas Filter/Graph RAG.
2. **CineBot V3**: Hybrid RAG (BM25+FAISS+Graph) + Pandas Metadata Filter + Cross-Encoder Reranker + Graph RAG.

THỐNG KÊ TỔNG QUAN:
- Traditional RAG: {json.dumps(trad_stats, ensure_ascii=False)}
- CineBot V3: {json.dumps(cinebot_stats, ensure_ascii=False)}

DỮ LIỆU SO SÁNH TỪNG CÂU HỎI (50 câu):
{json.dumps(comparison_data, ensure_ascii=False, indent=2)}

YÊU CẦU BÁO CÁO (format Markdown có heading, table, code block):
1. **Executive Summary**: Đánh giá tổng quan ngắn gọn, kết quả nổi bật nhất.
2. **Bảng so sánh tổng hợp theo Category**: So sánh điểm mạnh/yếu từng loại truy vấn.
3. **Phân tích chi tiết theo 6 nhóm Category**:
   - `semantic_retrieval` & `recommendation` (level 1-2): Cả 2 hệ thống xử lý ra sao?
   - `metadata_filter` (level 3-6): Traditional RAG có lọc được metadata không? CineBot V3 dùng cơ chế gì?
   - `semantic_reasoning` (level 4): Xử lý yêu cầu ngữ nghĩa phức tạp thế nào?
   - `negative_constraint` (level 7): Xử lý điều kiện loại trừ thế nào?
   - `aggregation` (level 8): Tính toán thống kê trên database?
   - `graph_reasoning` & `multi_hop_reasoning` (level 9-10): Suy luận đồ thị?
4. **So sánh Latency**: Phân tích tốc độ phản hồi trung bình theo category.
5. **Kết luận & Hướng Phát Triển**: Tóm tắt kỹ thuật cốt lõi tạo ra sự vượt trội.

Hãy viết chi tiết, khách quan, chỉ ra RÕ RÀNG cơ chế kỹ thuật nào dẫn đến từng kết quả.
"""

    print("⏳ Đang gửi yêu cầu phân tích tới LLM...")
    start = time.time()
    try:
        response = llm.invoke(prompt)
        report = response.content.strip()
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        elapsed = time.time() - start
        print(f"🎉 Báo cáo đã được lưu tại: {report_path}")
        print(f"⏱️  Thời gian tạo báo cáo: {elapsed:.1f}s")
    except Exception as e:
        print(f"❌ Lỗi sinh báo cáo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
