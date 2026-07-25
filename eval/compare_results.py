import os
import sys
import json
import time

# Thêm thư mục gốc vào path
eval_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(eval_dir)
sys.path.append(workspace_dir)

# Đảm bảo in UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Force flush on print
import builtins
def print(*args, **kwargs):
    builtins.print(*args, flush=True, **kwargs)

from langchain_openai import ChatOpenAI
from chatbot.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

def main():
    print("📊 [Result Comparison] Đang tải kết quả chạy thử nghiệm...")
    
    hq_results_path = os.path.join(workspace_dir, "eval", "hq_results_raw.json")
    traditional_results_path = os.path.join(workspace_dir, "eval", "traditional_results_raw.json")
    report_output_path = os.path.join(workspace_dir, "eval", "traditional_vs_cinebot_report.md")
    
    if not os.path.exists(hq_results_path):
        print(f"❌ Không tìm thấy kết quả của CineBot V3 tại {hq_results_path}")
        sys.exit(1)
        
    if not os.path.exists(traditional_results_path):
        print(f"❌ Không tìm thấy kết quả của RAG Truyền thống tại {traditional_results_path}")
        sys.exit(1)
        
    with open(hq_results_path, "r", encoding="utf-8") as f:
        hq_results = json.load(f)
        
    with open(traditional_results_path, "r", encoding="utf-8") as f:
        trad_results = json.load(f)
        
    print(f"✔️ Đã tải xong kết quả. CineBot V3: {len(hq_results)} câu, Traditional RAG: {len(trad_results)} câu.")
    
    # Chuẩn bị dữ liệu so sánh gửi cho LLM
    comparison_data = []
    
    # Map kết quả traditional theo ID
    trad_map = {item["id"]: item for item in trad_results}
    
    for hq_item in hq_results:
        q_id = hq_item["id"]
        question = hq_item["question"]
        category = hq_item.get("category", "N/A")
        
        # Lấy thông tin CineBot
        cb_movies = [m.get("title") for m in hq_item.get("movies", [])]
        cb_answer = hq_item.get("answer_result", "")
        
        # Trích xuất filters đã áp dụng trong trace (nếu có)
        cb_filters = {}
        if hq_item.get("trace") and hq_item["trace"].get("intent"):
            cb_filters = hq_item["trace"]["intent"].get("filters", {})
            
        # Lấy thông tin Traditional
        trad_item = trad_map.get(q_id, {})
        trad_movies = [m.get("title") for m in trad_item.get("movies", [])]
        trad_answer = trad_item.get("answer_result", "")
        
        comparison_data.append({
            "id": q_id,
            "question": question,
            "category": category,
            "cinebot_v3": {
                "retrieved_movies": cb_movies,
                "applied_filters": cb_filters,
                "answer": cb_answer
            },
            "traditional_rag": {
                "retrieved_movies": trad_movies,
                "answer": trad_answer
            }
        })
        
    print("🧠 Đang khởi tạo LLM để phân tích và lập báo cáo...")
    
    try:
        # Sử dụng OpenAI API client trực tiếp để tránh cache streamlit
        llm = ChatOpenAI(
            openai_api_key=LLM_API_KEY,
            openai_api_base=LLM_BASE_URL,
            model_name=LLM_MODEL,
            temperature=0.3,
            max_tokens=4000
        )
    except Exception as e:
        print(f"❌ Lỗi khởi tạo LLM: {e}")
        sys.exit(1)
        
    # Tạo prompt phân tích
    prompt = f"""Bạn là một chuyên gia cao cấp về kỹ thuật RAG (Retrieval-Augmented Generation) và AI Search.
Hãy viết một bản báo cáo so sánh, đánh giá khoa học, chi tiết bằng tiếng Việt để đối chiếu giữa hai hệ thống dựa trên kết quả chạy thử nghiệm dưới đây.

HỆ THỐNG ĐỐI CHIẾU:
1. **CineBot V3 (Hệ thống hiện tại)**: Hybrid RAG + Multi-stage Retrieval + Feature Engineering + Graph RAG + Pandas Filters.
2. **Traditional RAG (Hệ thống so sánh)**: Naive Semantic RAG, truy vấn FAISS phẳng trên chuỗi context nối, không lọc metadata cứng, không graph.

DỮ LIỆU CHẠY THỬ NGHIỆM:
{json.dumps(comparison_data, ensure_ascii=False, indent=2)}

YÊU CẦU BÁO CÁO (định dạng Markdown):
1. **Tiêu đề**: Rõ ràng, mang tính học thuật và thực tiễn chuyên sâu.
2. **Tóm tắt tổng quan (Executive Summary)**: Đánh giá ngắn gọn sự khác biệt cốt lõi và kết luận chung về hiệu năng.
3. **Bảng so sánh tổng hợp (Comparison Matrix)**: So sánh theo các tiêu chí: Khả năng lọc thuộc tính cứng (Metadata Filter), Khả năng loại bỏ nhiễu từ khóa (Title Overfitting), Suy luận đồ thị đa bước (Multi-hop Graph Reasoning), Độ chính xác thông tin (Anti-hallucination) và Tốc độ/Độ phức tạp.
4. **Phân tích chi tiết từng trường hợp câu hỏi (Q1 đến Q10)**:
   Với mỗi câu hỏi, chỉ rõ:
   - Thách thức của câu hỏi (Ví dụ: lọc nhiều thuộc tính, suy luận mối quan hệ, loại trừ thông tin).
   - Đánh giá kết quả của **Traditional RAG**: Đã lấy đúng phim chưa? Câu trả lời có đúng tiêu chí không? Gặp phải lỗi gì (lọt lưới metadata, hallucinate, lấy nhầm phim trùng tên...)?
   - Đánh giá kết quả của **CineBot V3**: Nhờ cơ chế nào (Pandas filter, Graph RAG, RRF hybrid...) mà khắc phục được lỗi của RAG truyền thống? Phân tích xem CineBot V3 đã thực sự giải quyết triệt để hay chưa.
   - So sánh định lượng và định tính cụ thể.
5. **Tổng kết những cải tiến cốt lõi**:
   Hệ thống CineBot V3 đã cải thiện cụ thể những vấn đề gì so với RAG truyền thống? Đúc kết thành các luận điểm kỹ thuật chính (ví dụ: Hybrid retrieval giúp chống vocab mismatch, Pandas filter giải quyết hard constraints, Graph RAG mở rộng khả năng suy luận phi tuyến tính).

Hãy viết báo cáo cực kỳ chi tiết, khách quan, sâu sắc, chỉ ra rõ cơ chế kỹ thuật đứng sau từng kết quả để người phát triển hiểu rõ giá trị của việc refactor hệ thống lên V3.
"""

    print("⏳ Đang gửi yêu cầu phân tích tới LLM (có thể mất 1-2 phút)...")
    start_time = time.time()
    try:
        response = llm.invoke(prompt)
        report_content = response.content.strip()
        
        # Ghi báo cáo ra file Markdown
        with open(report_output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        print(f"🎉 Đã tạo thành công báo cáo so sánh chi tiết tại: {report_output_path}")
        print(f"⏱️ Tổng thời gian thực hiện phân tích: {time.time() - start_time:.2f} giây.")
        
    except Exception as e:
        print(f"❌ Gặp lỗi khi gọi LLM sinh báo cáo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
