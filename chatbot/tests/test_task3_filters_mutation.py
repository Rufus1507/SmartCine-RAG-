"""
Test Task 3: retrieval_router.py — filters dict mutation leak in multi-turn conversation
=========================================================================================
Vấn đề: route_retrieval() mutate trực tiếp object `filters` được truyền vào,
dẫn đến filter tạm (vd: title) rò rỉ sang lượt hội thoại tiếp theo.

Fix: Thêm `filters = filters.copy()` ngay đầu route_retrieval().

Test này KHÔNG gọi LLM — dùng dict filters cố định để giả lập output intent_chain.
"""

import pandas as pd
from unittest.mock import MagicMock, patch


def make_sample_df(n: int = 20) -> pd.DataFrame:
    """Tạo DataFrame phim mẫu tối thiểu."""
    rows = []
    for i in range(n):
        rows.append({
            "Title": f"Movie {i}",
            "Rating": 7.0 + i * 0.1,
            "Movie Link": f"https://imdb.com/title/tt{i:07d}",
            "genres": "Action",
            "directors": f"Director {i}",
            "stars": f"Actor {i}",
            "countries_origin": "United States",
            "Year": 2010 + i,
            "imdb_id": f"tt{i:07d}",
            "description": f"Description for movie {i}",
            "num_votes": 10000 * (i + 1),
            "num_votes_scaled": 0.5,
        })
    return pd.DataFrame(rows)


def test_filters_not_mutated_after_route_retrieval():
    """
    Test chính: Lượt 1 gọi route_retrieval() với nhánh is_director_filmography_query.
    Sau đó kiểm tra dict filters gốc của caller KHÔNG bị sửa đổi.

    Không cần LLM: mock toàn bộ external dependency.
    """
    from chatbot.retrieval.retrieval_router import route_retrieval

    df = make_sample_df(20)

    # Lượt 1: filters gốc KHÔNG có "title"
    original_filters_turn1 = {"genre": "Action"}
    # Lưu reference tới dict gốc (không copy) để kiểm tra sau
    caller_filters_ref = original_filters_turn1

    # MultistageRetriever được import bên trong hàm route_retrieval nên patch module thật
    with patch("chatbot.retrieval.multistage_retriever.MultistageRetriever") as MockRetriever, \
         patch("chatbot.retrieval.retrieval_router.is_similar_movie_query", return_value=False), \
         patch("chatbot.retrieval.retrieval_router.is_director_filmography_query", return_value=True), \
         patch("chatbot.retrieval.retrieval_router.extract_title_from_query", return_value="Movie 0"):

        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve.return_value = pd.DataFrame()
        MockRetriever.return_value = mock_retriever_instance

        # Gọi route_retrieval với lượt 1
        route_retrieval(
            query="Đạo diễn của phim Movie 0 đã làm phim gì khác?",
            df=df,
            filters=original_filters_turn1,  # truyền dict gốc
            intent="director_filmography",
            faiss_index=None,
            embedder_model=None,
        )

    # KIỂM TRA: dict filters của caller KHÔNG bị mutate
    # Trước fix: filters["title"] = "Movie 0" sẽ được gán vào caller_filters_ref
    # Sau fix: filters = filters.copy() ngay đầu hàm → caller_filters_ref vẫn nguyên vẹn
    assert "title" not in caller_filters_ref, (
        f"BUG: filters của caller bị mutate! caller_filters_ref = {caller_filters_ref}"
    )
    assert caller_filters_ref == {"genre": "Action"}, (
        f"filters gốc đã bị thay đổi: {caller_filters_ref}"
    )


def test_multi_turn_no_title_leak():
    """
    Test mô phỏng scenario multi-turn đầy đủ:
    - Lượt 1: filters ban đầu = {}, route tự set filters["title"] = <movie>
    - Lấy filters sau lượt 1 làm last_filters cho lượt 2
    - Lượt 2: new_filters = {"genre": "Action"}
    - Merge: filters_turn2 = {**last_filters, **new_filters}
    - Assert: filters_turn2 KHÔNG chứa "title" sót lại từ lượt 1

    Sau fix (filters = filters.copy() trong route_retrieval), last_filters
    là dict gốc của caller và KHÔNG bị thay đổi → lượt 2 sạch.
    """
    from chatbot.retrieval.retrieval_router import route_retrieval

    df = make_sample_df(20)

    # === LƯỢT 1 ===
    filters_turn1 = {}  # filters gốc lượt 1 — caller giữ reference này

    with patch("chatbot.retrieval.multistage_retriever.MultistageRetriever") as MockRetriever, \
         patch("chatbot.retrieval.retrieval_router.is_similar_movie_query", return_value=False), \
         patch("chatbot.retrieval.retrieval_router.is_director_filmography_query", return_value=True), \
         patch("chatbot.retrieval.retrieval_router.extract_title_from_query", return_value="Inception"):

        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve.return_value = pd.DataFrame()
        MockRetriever.return_value = mock_retriever_instance

        route_retrieval(
            query="Đạo diễn của phim Inception đã làm gì khác?",
            df=df,
            filters=filters_turn1,  # dict gốc lượt 1
            intent="director_filmography",
            faiss_index=None,
            embedder_model=None,
        )

    # Sau lượt 1: caller dùng filters_turn1 làm last_filters
    last_filters = filters_turn1  # đây là dict gốc mà caller giữ

    # === LƯỢT 2 ===
    new_filters_turn2 = {"genre": "Action"}  # người dùng không nhắc phim nào ở lượt 2
    filters_turn2 = {**last_filters, **new_filters_turn2}  # merge như rag_chain.py

    # Assert: "title" từ lượt 1 KHÔNG được rò rỉ vào lượt 2
    assert "title" not in filters_turn2, (
        f"BUG: 'title' từ lượt 1 rò rỉ sang lượt 2! filters_turn2 = {filters_turn2}"
    )
    assert filters_turn2 == {"genre": "Action"}, (
        f"filters lượt 2 không đúng: {filters_turn2}"
    )


# ─── Runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_filters_not_mutated_after_route_retrieval,
        test_multi_turn_no_title_leak,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
