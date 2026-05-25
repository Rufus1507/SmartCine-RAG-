import requests
import urllib.parse

# ============================================================
# CẤU HÌNH
# ============================================================
OMDB_API_KEY = "1e1f167"  # Đăng ký free tại: https://www.omdbapi.com/apikey.aspx

# ============================================================
# Hàm chuẩn hoá
# ============================================================
def normalize(text: str) -> str:
    return text.lower().strip()

# ============================================================
# Bước 1: Lấy IMDb ID từ suggestion API
# ============================================================
def get_best_match(query: str) -> dict | None:
    query_encoded = urllib.parse.quote(query.lower())
    first_letter  = query_encoded[0] if query_encoded else 'a'

    url     = f"https://v3.sg.media-imdb.com/suggestion/{first_letter}/{query_encoded}.json"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code != 200:
        return None

    data   = response.json()
    movies = [item for item in data.get('d', []) if item.get('id', '').startswith('tt')]
    if not movies:
        return None

    query_norm = normalize(query)

    # 1. Khớp chính xác
    for movie in movies:
        if normalize(movie.get('l', '')) == query_norm:
            return movie
    # 2. Bắt đầu bằng query
    for movie in movies:
        if normalize(movie.get('l', '')).startswith(query_norm):
            return movie
    # 3. Fallback: kết quả đầu tiên
    return movies[0]

# ============================================================
# Bước 2: Lấy chi tiết đầy đủ từ OMDb bằng IMDb ID
# ============================================================
def get_movie_details(imdb_id: str) -> dict | None:
    params   = {"i": imdb_id, "apikey": OMDB_API_KEY, "plot": "full"}
    url      = "http://www.omdbapi.com/?" + urllib.parse.urlencode(params)
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        print(f"  [LỖI HTTP {response.status_code}]")
        return None

    data = response.json()
    if data.get("Response") == "False":
        print(f"  [Không tìm thấy trên OMDb] {data.get('Error', '')}")
        return None

    # Map sang các field bạn yêu cầu
    return {
        "names":       data.get("Title"),
        "orig_title":  data.get("Title"),          # OMDb không tách riêng orig_title
        "date_x":      data.get("Released"),        # vd: "18 Dec 2009"
        "score":       data.get("imdbRating"),      # vd: "7.9"
        "genre":       data.get("Genre"),           # vd: "Action, Adventure, Fantasy"
        "overview":    data.get("Plot"),            # mô tả đầy đủ
        "crew":        {
            "director": data.get("Director"),       # vd: "James Cameron"
            "writer":   data.get("Writer"),
            "actors":   data.get("Actors"),         # vd: "Sam Worthington, Zoe Saldana"
        },
        "status":      "Released",                  # OMDb chỉ có phim đã phát hành
        "orig_lang":   data.get("Language"),        # vd: "English, Spanish"
        "budget_x":    "N/A",                       # OMDb không cung cấp
        "revenue":     "N/A",                       # OMDb không cung cấp
        "country":     data.get("Country"),         # vd: "United States, United Kingdom"
        # --- Thêm thông tin bonus ---
        "imdb_id":     data.get("imdbID"),
        "imdb_link":   f"https://www.imdb.com/title/{data.get('imdbID')}/",
        "poster_url":  data.get("Poster") if data.get("Poster") != "N/A" else None,
        "runtime":     data.get("Runtime"),
        "awards":      data.get("Awards"),
    }

# ============================================================
# Hàm tổng hợp: tên phim -> toàn bộ thông tin
# ============================================================
def search_full_movie_info(title: str) -> dict | None:
    print(f"\n🔍 Tìm kiếm: {title}")

    # Bước 1: Lấy IMDb ID
    match = get_best_match(title)
    if not match:
        print(f"  ❌ Không tìm thấy IMDb ID cho: {title}")
        return None

    imdb_id = match.get('id')
    print(f"  ✅ Tìm thấy ID: {imdb_id} ({match.get('l')} - {match.get('y', 'N/A')})")

    # Bước 2: Lấy chi tiết từ OMDb
    details = get_movie_details(imdb_id)
    if not details:
        return None

    print(f"  ✅ Lấy chi tiết thành công từ OMDb")
    return details

# ============================================================
# MAIN — Test
# ============================================================
if __name__ == "__main__":
    test_titles = "Invisible - Revealing Hidden Disabilities"

    for title in test_titles:
        info = search_full_movie_info(title)

        if info:
            print(f"""
┌─────────────────────────────────────────────
│ 🎬  {info['names']}
├─────────────────────────────────────────────
│ orig_title : {info['orig_title']}
│ date_x     : {info['date_x']}
│ score      : {info['score']}
│ genre      : {info['genre']}
│ overview   : {info['overview'][:120]}...
│ crew       : {info['crew']['director']} (dir) | {info['crew']['actors']}
│ status     : {info['status']}
│ orig_lang  : {info['orig_lang']}
│ budget_x   : {info['budget_x']}
│ revenue    : {info['revenue']}
│ country    : {info['country']}
│ imdb_id    : {info['imdb_id']}
│ imdb_link  : {info['imdb_link']}
│ poster_url : {info['poster_url']}
└─────────────────────────────────────────────""")