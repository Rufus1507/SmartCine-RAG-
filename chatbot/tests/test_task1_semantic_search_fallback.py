"""
Test Task 1: semantic_search_tool() silent fallback fix
========================================================
Xác nhận rằng khi FAISS search raise exception, hàm trả về DataFrame rỗng
(không phải toàn bộ df gốc chưa lọc).
"""

import pandas as pd
from unittest.mock import MagicMock


def make_sample_df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "Title": [f"Movie {i}" for i in range(n)],
        "Rating": [7.0 + i * 0.1 for i in range(n)],
        "Movie Link": [f"https://imdb.com/title/tt{i:07d}" for i in range(n)],
        "description": [f"Description of movie {i}" for i in range(n)],
    })


def test_semantic_search_tool_returns_empty_on_exception():
    """
    Khi model.encode() raise exception, hàm phải trả về DataFrame rỗng,
    KHÔNG trả về df.copy() (toàn bộ dataset chưa lọc).
    """
    from chatbot.tools import semantic_search_tool

    df = make_sample_df(10)

    # Mock index bình thường
    mock_index = MagicMock()

    # Mock model sẽ raise RuntimeError khi gọi encode()
    mock_model = MagicMock()
    mock_model.encode.side_effect = RuntimeError("FAISS encode error simulated")

    result = semantic_search_tool("any query", df, mock_index, mock_model, top_k=5)

    assert isinstance(result, pd.DataFrame), "Kết quả phải là DataFrame"
    assert len(result) == 0, (
        f"Khi exception, phải trả về DataFrame rỗng, "
        f"nhưng nhận được {len(result)} dòng (có thể là toàn bộ df gốc)"
    )


def test_semantic_search_tool_returns_empty_on_index_search_exception():
    """
    Khi index.search() raise exception, hàm phải trả về DataFrame rỗng.
    """
    from chatbot.tools import semantic_search_tool
    import numpy as np

    df = make_sample_df(10)

    mock_index = MagicMock()
    mock_index.search.side_effect = RuntimeError("FAISS search error simulated")

    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(1, 384).astype("float32")

    result = semantic_search_tool("some query", df, mock_index, mock_model, top_k=5)

    assert isinstance(result, pd.DataFrame), "Kết quả phải là DataFrame"
    assert len(result) == 0, (
        f"Khi index.search() raise exception, phải trả về DataFrame rỗng, "
        f"nhưng nhận được {len(result)} dòng"
    )


def test_semantic_search_tool_returns_empty_df_not_original_df():
    """
    Verify quan trọng: kết quả trả về khi lỗi phải là DataFrame rỗng HOÀN TOÀN MỚI,
    không phải bản copy của df gốc (10 dòng).
    """
    from chatbot.tools import semantic_search_tool

    df = make_sample_df(10)
    assert len(df) == 10, "Tiền điều kiện: df gốc phải có 10 dòng"

    mock_model = MagicMock()
    mock_model.encode.side_effect = Exception("simulated error")
    mock_index = MagicMock()

    result = semantic_search_tool("query", df, mock_index, mock_model, top_k=5)

    # Chắc chắn không phải toàn bộ df gốc (10 dòng)
    assert len(result) == 0, (
        f"BUG vẫn còn: trả về {len(result)} dòng thay vì 0 dòng khi exception"
    )


# ─── Runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_semantic_search_tool_returns_empty_on_exception,
        test_semantic_search_tool_returns_empty_on_index_search_exception,
        test_semantic_search_tool_returns_empty_df_not_original_df,
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
