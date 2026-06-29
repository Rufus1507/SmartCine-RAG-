import os
import sys
import time
import torch
import faiss
from sentence_transformers import SentenceTransformer

# Thêm thư mục gốc vào path để import các modules
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

# Đảm bảo in UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from chatbot.config import MIN_VOTES_THRESHOLD, PROFILE_INDEX_PATH
from chatbot.data_loader import load_data
from chatbot.representation import (
    INDEX_A_PATH, INDEX_B_PATH, INDEX_C_PATH,
    generate_embeddings_for_version
)

def make_profile_for_similar_search(row):
    """Được copy từ generate_movie_profile_embeddings.py"""
    title = str(row.get('Title', '')).strip()
    genre = str(row.get('genres', '')).strip()
    director = str(row.get('directors', '')).strip()
    stars = str(row.get('stars', '')).strip()
    description = str(row.get('description', '')).strip()
    
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if genre:
        parts.append(f"Genre: {genre}")
    if director:
        parts.append(f"Director: {director}")
    if stars:
        parts.append(f"Stars: {stars}")
    if description:
        parts.append(f"Description: {description}")
        
    return "\n".join(parts)

def main():
    print("=============================================================")
    print("🛠️  REBUILD ALL EMBEDDING & REPRESENTATION INDICES")
    print("=============================================================")
    
    # 1. Load data
    print("🚀 Loading data...")
    df = load_data()
    df_filtered = df[df['num_votes'] >= MIN_VOTES_THRESHOLD].reset_index(drop=True)
    print(f"✅ Filtered movies (votes >= {MIN_VOTES_THRESHOLD}): {len(df_filtered):,}")
    
    # 2. Select Device and Load Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Running embeddings model on device: {device.upper()}")
    print("🧠 Loading model: 'paraphrase-multilingual-MiniLM-L12-v2'...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
    
    # 3. Rebuild representation_a, representation_b, representation_c
    print("\n⏳ Rebuilding Representation A (Description only)...")
    generate_embeddings_for_version(df_filtered, 'A', model, INDEX_A_PATH)
    
    print("\n⏳ Rebuilding Representation B (Genre + Description)...")
    generate_embeddings_for_version(df_filtered, 'B', model, INDEX_B_PATH)
    
    print("\n⏳ Rebuilding Representation C (Genre + Description + Keywords)...")
    generate_embeddings_for_version(df_filtered, 'C', model, INDEX_C_PATH)
    
    # 4. Rebuild movie_profile_embeddings.index
    print(f"\n⏳ Rebuilding Movie Profile Index (for similar movie retriever)...")
    profiles = df_filtered.apply(make_profile_for_similar_search, axis=1).tolist()
    embeddings = model.encode(
        profiles, 
        batch_size=128, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, PROFILE_INDEX_PATH)
    print(f"✅ Saved movie profile index to {PROFILE_INDEX_PATH}")
    
    print("\n🎉 Rebuild completed successfully! All index files are ready.")

if __name__ == "__main__":
    main()
