"""
Test Task 7: tools.py — country_aliases load lai moi lan goi
=============================================================
Xac nhan rang load_country_aliases() chi doc file 1 lan qua nhieu lan goi
search_movies_tool() lien tiep (nho @lru_cache tren _load_country_aliases_cached).
"""

from unittest.mock import patch, MagicMock
import pandas as pd


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


def test_load_country_aliases_lru_cache_present():
    """
    Xac nhan _load_country_aliases_cached co @lru_cache (co cache_clear).
    """
    from chatbot.data_loader import _load_country_aliases_cached

    assert hasattr(_load_country_aliases_cached, "cache_clear"), (
        "_load_country_aliases_cached khong co @lru_cache! Thieu cache mechanism."
    )
    assert hasattr(_load_country_aliases_cached, "cache_info"), (
        "_load_country_aliases_cached khong co cache_info tu @lru_cache."
    )
    print("  @lru_cache confirmed on _load_country_aliases_cached")


def test_load_country_aliases_called_only_once_via_lru_cache():
    """
    Khi search_movies_tool() voi filter 'country' duoc goi 3 lan lien tiep,
    ham _load_country_aliases_cached() chi doc file dung 1 lan nho @lru_cache.
    """
    from chatbot.data_loader import _load_country_aliases_cached

    # Reset lru_cache truoc khi test
    _load_country_aliases_cached.cache_clear()

    call_count = 0
    original_open = open

    def counting_open(path, *args, **kwargs):
        nonlocal call_count
        if "country_aliases" in str(path):
            call_count += 1
        return original_open(path, *args, **kwargs)

    df = make_sample_df(10)
    filters_with_country = {"country": "US"}

    with patch("builtins.open", side_effect=counting_open):
        from chatbot.tools import search_movies_tool
        # Goi 3 lan lien tiep
        for _ in range(3):
            search_movies_tool(df, filters_with_country, top_k=5)

    # Voi lru_cache: chi 1 lan file duoc mo
    assert call_count <= 1, (
        f"BUG: _load_country_aliases_cached mo file {call_count} lan thay vi toi da 1 lan. "
        "Cache khong hoat dong!"
    )
    print(f"  country_aliases file read count: {call_count} (expected: <= 1)")


def test_load_country_aliases_returns_consistent_result():
    """
    Goi load_country_aliases() nhieu lan - ket qua phai giong nhau.
    """
    from chatbot.data_loader import load_country_aliases

    result1 = load_country_aliases()
    result2 = load_country_aliases()
    result3 = load_country_aliases()

    assert result1 == result2 == result3, (
        "load_country_aliases tra ket qua khong nhat quan!"
    )
    assert isinstance(result1, dict), "Ket qua phai la dict"
    print(f"  Consistent results: {len(result1)} country aliases loaded")


def test_cache_call_count_via_cache_info():
    """
    Dung cache_info() de xac nhan lru_cache hoat dong dung.
    Sau khi goi 3 lan, misses phai la 1 (lan dau), hits la 2.
    """
    from chatbot.data_loader import _load_country_aliases_cached

    _load_country_aliases_cached.cache_clear()

    # Goi 3 lan
    _load_country_aliases_cached()
    _load_country_aliases_cached()
    _load_country_aliases_cached()

    info = _load_country_aliases_cached.cache_info()
    assert info.misses == 1, f"Expected 1 miss (first call), got {info.misses}"
    assert info.hits == 2, f"Expected 2 hits (2nd and 3rd calls), got {info.hits}"
    print(f"  cache_info: hits={info.hits}, misses={info.misses} (expected: hits=2, misses=1)")


# Runner

if __name__ == "__main__":
    tests = [
        test_load_country_aliases_lru_cache_present,
        test_load_country_aliases_returns_consistent_result,
        test_cache_call_count_via_cache_info,
        test_load_country_aliases_called_only_once_via_lru_cache,
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
