import re
import os
import sys
from collections import Counter
from chatbot.retrieval.bm25_retriever import TRANSLATION_MAP

dict_len = len(TRANSLATION_MAP)
unique_len = len(set(TRANSLATION_MAP.keys()))

src_path = os.path.join("chatbot", "retrieval", "bm25_retriever.py")
with open(src_path, encoding="utf-8") as f:
    src = f.read()

key_pattern = re.compile(r'^\s+"([^"]+)":\s+"', re.MULTILINE)
all_raw_keys = key_pattern.findall(src)
counter = Counter(all_raw_keys)
duplicates = {k: v for k, v in counter.items() if v > 1}

# Output ASCII-safe for Windows terminal
print(f"Dict len: {dict_len}")
print(f"Unique keys in dict: {unique_len}")
print(f"Duplicates in source file: {len(duplicates)}")
for k, cnt in duplicates.items():
    kept_val = TRANSLATION_MAP[k]
    k_repr = k.encode("ascii", "backslashreplace").decode("ascii")
    val_repr = kept_val.encode("ascii", "backslashreplace").decode("ascii")
    print(f"  key={k_repr!r}  count={cnt}  value_kept={val_repr!r}")

if len(duplicates) > 0:
    print("ACTION NEEDED: Remove duplicate keys from bm25_retriever.py")
else:
    print("PASS: No duplicate keys in source file.")

# Assertion
assert dict_len == unique_len, f"Dict has {dict_len} but {unique_len} unique keys"
print(f"ASSERTION PASS: len(TRANSLATION_MAP) == len(set(TRANSLATION_MAP.keys())) == {dict_len}")
