import os
from google import genai
from google.genai import types

# ============================================================
# CAU HINH API KEY
# ============================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyALBqHfbrgyqiWFSe-8B9rl0Ajctxv88ik")

print("=" * 50)
print("TEST GEMINI API (google.genai)")
print("=" * 50)

# Khoi tao client
client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# TEST 1: Liet ke cac model kha dung
# ============================================================
print("\n[TEST 1] Cac model Gemini kha dung:")
print("-" * 40)
try:
    models = client.models.list()
    count = 0
    for m in models:
        if "generateContent" in (m.supported_actions or []):
            print(f"  OK  {m.name}")
            count += 1
    if count == 0:
        for m in models:
            print(f"  >>  {m.name}")
except Exception as e:
    print(f"  LOI: {e}")

# ============================================================
# TEST 2: Gui tin nhan voi gemini-2.5-flash (model moi nhat)
# ============================================================
print("\n[TEST 2] Gui tin nhan voi gemini-2.5-flash:")
print("-" * 40)
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Xin chao! Hay gioi thieu ban than bang mot cau ngan."
    )
    print(f"  OK - Phan hoi: {response.text.strip()}")
except Exception as e:
    print(f"  LOI: {e}")

# ============================================================
# TEST 3: Gui tin nhan voi gemini-2.0-flash-lite (nhe, it quota)
# ============================================================
print("\n[TEST 3] Gui tin nhan voi gemini-2.0-flash-lite:")
print("-" * 40)
try:
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents="Tra loi ngan gon: 1 + 1 = ?"
    )
    print(f"  OK - Phan hoi: {response.text.strip()}")
except Exception as e:
    print(f"  LOI: {e}")

# ============================================================
# TEST 4: Gui tin nhan voi gemini-2.0-flash-001
# ============================================================
print("\n[TEST 4] Gui tin nhan voi gemini-2.0-flash-001:")
print("-" * 40)
try:
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents="Tra loi ngan gon: 1 + 1 = ?"
    )
    print(f"  OK - Phan hoi: {response.text.strip()}")
except Exception as e:
    print(f"  LOI: {e}")

print("\n" + "=" * 50)
print("Hoan tat kiem tra API!")
print("=" * 50)
