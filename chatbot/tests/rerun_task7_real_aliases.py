"""
Task 7 re-run: verify with real country aliases (cache cleared before test).
"""
from chatbot.data_loader import _load_country_aliases_cached

# ─── Setup: reset cache để đảm bảo đọc file thật ──────────────────────────
_load_country_aliases_cached.cache_clear()

# TEST 1: lru_cache có mặt
assert hasattr(_load_country_aliases_cached, 'cache_clear')
assert hasattr(_load_country_aliases_cached, 'cache_info')
print("PASS  [1] @lru_cache confirmed on _load_country_aliases_cached")

# TEST 2: lần đọc đầu tiên → misses=1, hits=0
result1 = _load_country_aliases_cached()
info1 = _load_country_aliases_cached.cache_info()
assert info1.misses == 1, f"Expected 1 miss, got {info1.misses}"
assert info1.hits == 0, f"Expected 0 hits on 1st call, got {info1.hits}"
print(f"PASS  [2] 1st call: misses={info1.misses}, hits={info1.hits}")
print(f"         Loaded {len(result1)} real country aliases from file")

# In 8 aliases đầu bằng repr ASCII-safe
sample = list(result1.items())[:8]
print("         Sample aliases:")
for k, v in sample:
    k_safe = k.encode("ascii", "backslashreplace").decode()
    v_safe = v.encode("ascii", "backslashreplace").decode()
    print(f"           {k_safe!r} -> {v_safe!r}")

# TEST 3: gọi lần 2, 3 → hits tăng, misses không đổi
result2 = _load_country_aliases_cached()
result3 = _load_country_aliases_cached()
info3 = _load_country_aliases_cached.cache_info()
assert info3.misses == 1, f"Still 1 miss expected, got {info3.misses}"
assert info3.hits == 2, f"Expected 2 hits after 3 calls, got {info3.hits}"
print(f"PASS  [3] After 3 total calls: misses={info3.misses}, hits={info3.hits} (file read only ONCE)")

# TEST 4: kết quả nhất quán
assert result1 == result2 == result3
print(f"PASS  [4] All 3 calls return identical dict ({len(result1)} entries)")

# TEST 5: aliases thực sự chứa dữ liệu có nghĩa
assert len(result1) > 0, "Empty aliases dict!"
print(f"PASS  [5] Aliases non-empty: {len(result1)} entries")

print("")
print("5/5 tests passed.")
