"""
Test: P0 Bug Fix — Candidate Merge Order in multistage_retriever.py
====================================================================
Bug: metadata_candidates được merge CUỐI cùng trong danh sách dfs_to_combine.
     Khi graph(300) + faiss(150) + bm25(100) = 550 đã vượt cap 500 trước khi
     metadata_candidates được thêm vào → các phim thỏa rating_min bị loại hoàn toàn.

Fix: metadata_candidates được merge ĐẦU TIÊN (trước graph, faiss, bm25).
     Logic đã được tách thành MultistageRetriever._merge_candidates().

Task 2 (REFACTORED): Các test nay gọi thẳng retriever._merge_candidates(...)
thay vì dùng bản sao logic simulate_old_merge/simulate_new_merge cũ.
Ghi chú: Mocked intent output — không gọi API thật.
"""

import pandas as pd
from chatbot.retrieval.multistage_retriever import MultistageRetriever


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_movie_row(title: str, rating: float, movie_link: str = None, **kwargs) -> dict:
    return {
        "Title": title,
        "Rating": rating,
        "Movie Link": movie_link or f"https://imdb.com/title/{title.replace(' ', '_')}",
        "genres": kwargs.get("genres", "Action"),
        "directors": kwargs.get("directors", "Director X"),
        "stars": kwargs.get("stars", "Actor Y"),
        "countries_origin": kwargs.get("countries_origin", "United States"),
        "Year": kwargs.get("year", 2020),
        "imdb_id": kwargs.get("imdb_id", title.replace(" ", "_")),
        "description": f"Movie about {title}",
        "num_votes": kwargs.get("num_votes", 100000),
    }


def make_large_df(n: int, base_rating: float = 5.0, link_prefix: str = "low") -> pd.DataFrame:
    rows = [make_movie_row(f"Low Movie {link_prefix}_{i}", base_rating,
                           movie_link=f"https://imdb.com/title/{link_prefix}_{i}",
                           imdb_id=f"tt{link_prefix}{i:05d}") for i in range(n)]
    return pd.DataFrame(rows)


def make_high_rated_df(n: int, rating: float = 9.0, prefix: str = "high") -> pd.DataFrame:
    rows = [make_movie_row(f"High Movie {prefix}_{i}", rating,
                           movie_link=f"https://imdb.com/title/{prefix}_{i}",
                           imdb_id=f"tt{prefix}{i:05d}") for i in range(n)]
    return pd.DataFrame(rows)


# ─── Test Cases — gọi thẳng MultistageRetriever._merge_candidates() ──────────

def test_metadata_excluded_when_no_filter():
    """
    Khi has_metadata_filters=False, metadata_candidates không được ưu tiên xếp đầu.
    Do đó khi pool graph(300) + faiss(150) + bm25(100) = 550 vượt cap 500,
    metadata_candidates bị loại hoàn toàn.

    Lưu ý: Bug gốc (thứ tự merge sai khi CÓ filter) được bảo vệ bởi
    test_fix_new_order_includes_metadata và test_rating_min_scenario_end_to_end, không phải test này.
    """
    graph_df = make_large_df(300, link_prefix="graph")
    faiss_df = make_large_df(150, link_prefix="faiss")
    bm25_df = make_large_df(100, link_prefix="bm25")
    meta_df = make_high_rated_df(10, rating=9.0)

    old_pool = MultistageRetriever._merge_candidates(
        graph_candidates=graph_df,
        faiss_candidates=faiss_df,
        bm25_candidates=bm25_df,
        metadata_candidates=meta_df,
        has_metadata_filters=False,
        cap=500
    )
    pool_links = {r["Movie Link"] for r in old_pool}
    meta_links = set(meta_df["Movie Link"].tolist())

    overlap = pool_links & meta_links
    assert len(overlap) == 0, f"Bug not reproduced: {len(overlap)} metadata movies appeared"
    assert len(old_pool) == 500


def test_fix_new_order_includes_metadata():
    """
    [Verify Fix] metadata merge trước (has_metadata_filters=True) →
    tất cả 10 phim rating cao có mặt trong pool.
    """
    graph_df = make_large_df(300, link_prefix="graph")
    faiss_df = make_large_df(150, link_prefix="faiss")
    bm25_df = make_large_df(100, link_prefix="bm25")
    meta_df = make_high_rated_df(10, rating=9.0)

    new_pool = MultistageRetriever._merge_candidates(
        graph_candidates=graph_df,
        faiss_candidates=faiss_df,
        bm25_candidates=bm25_df,
        metadata_candidates=meta_df,
        has_metadata_filters=True,  # FIX: meta được ưu tiên vào đầu
        cap=500
    )
    pool_links = {r["Movie Link"] for r in new_pool}
    meta_links = set(meta_df["Movie Link"].tolist())

    overlap = pool_links & meta_links
    assert len(overlap) == 10, f"Expected 10 high-rated in pool, got {len(overlap)}"
    assert len(new_pool) == 500


def test_dedup_shared_links_between_meta_and_graph():
    """
    Phim xuất hiện cả trong metadata VÀ graph → chỉ được tính 1 lần (dedup).
    """
    shared_links = [f"https://imdb.com/title/shared_{i}" for i in range(5)]

    meta_df = make_high_rated_df(5, rating=9.0, prefix="shared")
    meta_df["Movie Link"] = shared_links
    graph_df = make_high_rated_df(5, rating=7.0, prefix="shared")
    graph_df["Movie Link"] = shared_links

    pool = MultistageRetriever._merge_candidates(
        graph_candidates=graph_df,
        faiss_candidates=pd.DataFrame(),
        bm25_candidates=pd.DataFrame(),
        metadata_candidates=meta_df,
        has_metadata_filters=True,
        cap=500
    )
    pool_links = [r["Movie Link"] for r in pool]

    assert len(pool_links) == len(set(pool_links)), "Duplicate links in pool!"
    assert len(pool) == 5


def test_no_metadata_filter_pool_unchanged():
    """
    Khi has_metadata_filters=False, metadata không được ưu tiên — pool gồm graph+faiss+bm25.
    """
    graph_df = make_large_df(100, link_prefix="graph")
    faiss_df = make_large_df(80, link_prefix="faiss")
    bm25_df = make_large_df(60, link_prefix="bm25")

    pool = MultistageRetriever._merge_candidates(
        graph_candidates=graph_df,
        faiss_candidates=faiss_df,
        bm25_candidates=bm25_df,
        metadata_candidates=pd.DataFrame(),
        has_metadata_filters=False,
        cap=500
    )
    assert len(pool) == 240, f"Expected 240, got {len(pool)}"


def test_small_graph_metadata_still_prioritized():
    """
    Với graph nhỏ (50 phim), metadata vẫn phải đầy đủ trong pool.
    """
    graph_df = make_large_df(50, link_prefix="graph")
    faiss_df = make_large_df(150, link_prefix="faiss")
    bm25_df = make_large_df(100, link_prefix="bm25")
    meta_df = make_high_rated_df(20, rating=9.5)

    pool = MultistageRetriever._merge_candidates(
        graph_candidates=graph_df,
        faiss_candidates=faiss_df,
        bm25_candidates=bm25_df,
        metadata_candidates=meta_df,
        has_metadata_filters=True,
        cap=500
    )
    pool_links = {r["Movie Link"] for r in pool}
    meta_links = set(meta_df["Movie Link"].tolist())

    assert len(pool_links & meta_links) == 20, "All 20 high-rated movies must be in pool"


def test_rating_min_scenario_end_to_end():
    """
    Scenario hoàn chỉnh: similar-movie query + rating_min=8.0 (mocked filters).
    Mocked intent output (không gọi API thật):
      filters = {"title": "Inception", "rating_min": 8.0}  # giả lập output của intent_chain
    """
    # Giả lập: graph tìm 300 phim liên quan qua đồ thị (rating thấp)
    graph_df = make_large_df(300, base_rating=5.5, link_prefix="graph")

    # FAISS trả về 150 phim (rating thấp)
    faiss_df = make_large_df(150, base_rating=6.0, link_prefix="faiss")

    # BM25 trả về 100 phim (rating thấp)
    bm25_df = make_large_df(100, base_rating=5.0, link_prefix="bm25")

    # search_movies_tool với rating_min=8.0 trả về 25 phim rating cao
    meta_df = make_high_rated_df(25, rating=8.5, prefix="filtered_high")

    # === BUG: thứ tự cũ — has_metadata_filters=False ===
    old_pool = MultistageRetriever._merge_candidates(
        graph_candidates=graph_df,
        faiss_candidates=faiss_df,
        bm25_candidates=bm25_df,
        metadata_candidates=meta_df,
        has_metadata_filters=False,
        cap=500
    )
    old_meta_count = sum(1 for r in old_pool if r["Movie Link"].startswith("https://imdb.com/title/filtered_high"))
    assert old_meta_count == 0, f"Bug not present: old had {old_meta_count} high-rated"

    # === FIX: thứ tự mới — has_metadata_filters=True ===
    new_pool = MultistageRetriever._merge_candidates(
        graph_candidates=graph_df,
        faiss_candidates=faiss_df,
        bm25_candidates=bm25_df,
        metadata_candidates=meta_df,
        has_metadata_filters=True,
        cap=500
    )
    new_meta_count = sum(1 for r in new_pool if r["Movie Link"].startswith("https://imdb.com/title/filtered_high"))
    assert new_meta_count == 25, f"Fix failed: only {new_meta_count}/25 high-rated in pool"


# ─── Runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_metadata_excluded_when_no_filter,
        test_fix_new_order_includes_metadata,
        test_dedup_shared_links_between_meta_and_graph,
        test_no_metadata_filter_pool_unchanged,
        test_small_graph_metadata_still_prioritized,
        test_rating_min_scenario_end_to_end,
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
