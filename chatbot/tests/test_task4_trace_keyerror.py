"""
Test Task 4: retrieval_router.py — trace["stage0_graph"] KeyError
=================================================================
Xác nhận rằng khi gọi route_retrieval() với input khiến Stage 0 Graph
không được kích hoạt, trace["stage0_graph"] vẫn tồn tại và không raise KeyError.
"""

import pandas as pd
from unittest.mock import MagicMock, patch


def make_sample_df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "Title": [f"Movie {i}" for i in range(n)],
        "Rating": [7.0 + i * 0.1 for i in range(n)],
        "Movie Link": [f"https://imdb.com/title/tt{i:07d}" for i in range(n)],
        "genres": ["Action"] * n,
        "directors": [f"Director {i}" for i in range(n)],
        "stars": [f"Actor {i}" for i in range(n)],
        "countries_origin": ["United States"] * n,
        "Year": [2010 + i for i in range(n)],
        "num_votes": [100000] * n,
        "description": [f"desc {i}" for i in range(n)],
    })


def test_trace_stage0_graph_key_exists_when_graph_not_called():
    """
    Khi route không vào nhánh similar-movie hoặc filmography
    (Stage 0 Graph không được gọi), trace["stage0_graph"] vẫn phải tồn tại
    với giá trị default {"called": False, "candidates": []}.
    Không được raise KeyError.
    """
    from chatbot.retrieval.retrieval_router import route_retrieval

    df = make_sample_df(10)
    trace = {}

    with patch("chatbot.retrieval.multistage_retriever.MultistageRetriever") as MockRetriever, \
         patch("chatbot.retrieval.retrieval_router.is_similar_movie_query", return_value=False), \
         patch("chatbot.retrieval.retrieval_router.is_director_filmography_query", return_value=False):

        mock_instance = MagicMock()
        mock_instance.retrieve.return_value = pd.DataFrame()
        MockRetriever.return_value = mock_instance

        # Không được raise KeyError
        try:
            route_retrieval(
                query="Phim hài năm 2020",
                df=df,
                filters={"genre": "Comedy"},
                intent="search_movies",
                faiss_index=None,
                embedder_model=None,
                trace=trace,
            )
        except KeyError as e:
            assert False, f"KeyError raised: {e} — trace thiếu setdefault cho stage0_graph"

    # Xác nhận key tồn tại trong trace
    assert "stage0_graph" in trace, (
        "trace['stage0_graph'] không tồn tại sau khi route_retrieval hoàn thành!"
    )
    # Xác nhận giá trị default đúng
    assert trace["stage0_graph"].get("called") == False, (
        f"stage0_graph.called phải là False khi graph không được gọi, "
        f"got: {trace['stage0_graph'].get('called')}"
    )
    assert "candidates" in trace["stage0_graph"], (
        "stage0_graph phải có key 'candidates'"
    )
    print(f"  trace['stage0_graph'] = {trace['stage0_graph']}")


def test_trace_stage0_graph_no_keyerror_without_trace():
    """
    Khi trace=None, không có KeyError và function hoạt động bình thường.
    """
    from chatbot.retrieval.retrieval_router import route_retrieval

    df = make_sample_df(10)

    with patch("chatbot.retrieval.multistage_retriever.MultistageRetriever") as MockRetriever, \
         patch("chatbot.retrieval.retrieval_router.is_similar_movie_query", return_value=False), \
         patch("chatbot.retrieval.retrieval_router.is_director_filmography_query", return_value=False):

        mock_instance = MagicMock()
        mock_instance.retrieve.return_value = pd.DataFrame()
        MockRetriever.return_value = mock_instance

        try:
            result, route = route_retrieval(
                query="Phim kinh dị hay",
                df=df,
                filters={"genre": "Horror"},
                intent="search_movies",
                faiss_index=None,
                embedder_model=None,
                trace=None,  # trace=None
            )
        except Exception as e:
            assert False, f"Không được raise exception khi trace=None: {e}"

    print("  trace=None: no exception raised")


# ─── Runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_trace_stage0_graph_key_exists_when_graph_not_called,
        test_trace_stage0_graph_no_keyerror_without_trace,
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
