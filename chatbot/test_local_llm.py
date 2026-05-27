import os
from openai import OpenAI

# ============================================================
# CAU HINH LOCAL LLM ENDPOINT
# ============================================================
BASE_URL = "http://localhost:20128/v1"
API_KEY  = "any"          # local server thuong khong can key that
MODEL    = "cx/gpt-5.5"

print("=" * 55)
print("TEST LOCAL LLM ENDPOINT (OpenAI-compatible)")
print(f"  Endpoint : {BASE_URL}")
print(f"  Model    : {MODEL}")
print("=" * 55)

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ============================================================
# TEST 1: Liet ke model kha dung
# ============================================================
print("\n[TEST 1] Liet ke model kha dung:")
print("-" * 40)
try:
    models = client.models.list()
    for m in models.data:
        print(f"  OK  {m.id}")
except Exception as e:
    print(f"  LOI: {e}")

# ============================================================
# TEST 2: Chat don gian
# ============================================================
print("\n[TEST 2] Chat don gian (1 + 1 = ?):")
print("-" * 40)
try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "1 + 1 = ? Answer briefly."}],
        max_tokens=50,
    )
    print(f"  OK - Phan hoi: {response.choices[0].message.content.strip()}")
except Exception as e:
    print(f"  LOI: {e}")

# ============================================================
# TEST 3: Chat voi system prompt (tieng Viet)
# ============================================================
print("\n[TEST 3] Chat voi system prompt:")
print("-" * 40)
try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Ban la tro ly phim than thien. Tra loi ngan gon."},
            {"role": "user",   "content": "Xin chao! Ban la ai?"},
        ],
        max_tokens=100,
    )
    print(f"  OK - Phan hoi: {response.choices[0].message.content.strip()}")
except Exception as e:
    print(f"  LOI: {e}")

# ============================================================
# TEST 4: Yeu cau tra ve JSON (dung cho parse intent)
# ============================================================
print("\n[TEST 4] Yeu cau tra ve JSON:")
print("-" * 40)
try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": 'Tra ve JSON dung voi schema: {"answer": <string>}. Chi tra JSON, khong text khac.'},
            {"role": "user",   "content": "Ten ban la gi?"},
        ],
        max_tokens=100,
    )
    print(f"  OK - Phan hoi: {response.choices[0].message.content.strip()}")
except Exception as e:
    print(f"  LOI: {e}")

print("\n" + "=" * 55)
print("Hoan tat kiem tra endpoint!")
print("=" * 55)
