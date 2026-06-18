import numpy as np
from chatbot.feature_engineering import DECADES

DEFAULT_WEIGHTS = {
    "content": 0.40,
    "genre": 0.25,
    "actor": 0.15,
    "director": 0.10,
    "country": 0.05,
    "decade": 0.03,
    "award": 0.02
}

def compute_genre_similarity(g1, g2) -> float:
    """Jaccard Similarity for multi-hot genre vectors."""
    v1 = np.array(g1)
    v2 = np.array(g2)
    intersection = np.sum(np.minimum(v1, v2))
    union = np.sum(np.maximum(v1, v2))
    if union == 0:
        return 1.0
    return float(intersection / union)

def compute_actor_similarity(a1, a2) -> float:
    """Actor Overlap Score for list of actor indices."""
    s1 = set(a1)
    s2 = set(a2)
    intersection = len(s1 & s2)
    min_len = min(len(s1), len(s2))
    if min_len == 0:
        return 0.0
    return float(intersection / min_len)

def compute_director_similarity(d1, d2) -> float:
    """Director Overlap Score for list of director indices."""
    s1 = set(d1)
    s2 = set(d2)
    intersection = len(s1 & s2)
    min_len = min(len(s1), len(s2))
    if min_len == 0:
        return 0.0
    return float(intersection / min_len)

def compute_country_similarity(c1, c2) -> float:
    """Country Overlap Score for multi-hot country vectors."""
    v1 = np.array(c1)
    v2 = np.array(c2)
    intersection = np.sum(np.minimum(v1, v2))
    min_len = min(np.sum(v1), np.sum(v2))
    if min_len == 0:
        return 0.0
    return float(intersection / min_len)

def compute_decade_similarity(dec1_vec, dec2_vec) -> float:
    """Decade Distance Score from one-hot decade vectors."""
    v1 = np.array(dec1_vec)
    v2 = np.array(dec2_vec)
    if np.sum(v1) == 0 or np.sum(v2) == 0:
        return 1.0
    idx1 = np.argmax(v1)
    idx2 = np.argmax(v2)
    dec1 = DECADES[idx1]
    dec2 = DECADES[idx2]
    distance = abs(dec1 - dec2)
    return float(1.0 / (1.0 + distance / 10.0))

def compute_award_similarity(aw1, aw2) -> float:
    """Cosine Similarity of award vectors."""
    v1 = np.array(aw1)
    v2 = np.array(aw2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        if norm1 == 0 and norm2 == 0:
            return 1.0
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def compute_content_similarity(emb1, emb2) -> float:
    """Cosine Similarity of semantic embeddings."""
    v1 = np.array(emb1)
    v2 = np.array(emb2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def compute_weighted_similarity(movie_features: dict, ref_features: dict, weights: dict = None) -> dict:
    """
    Computes the weighted similarity scores between a movie's features and reference features.
    Automatically redistributes weights for attributes that are unspecified in reference features.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
        
    scores = {}
    active_weights = {}
    
    # 1. Content (always active if reference embedding is present)
    ref_emb = ref_features.get("semantic_embedding")
    movie_emb = movie_features.get("semantic_embedding")
    if ref_emb is not None and movie_emb is not None:
        scores["content_score"] = compute_content_similarity(movie_emb, ref_emb)
        active_weights["content"] = weights["content"]
    else:
        scores["content_score"] = 1.0
        
    # 2. Genre
    ref_genre = ref_features.get("genre_vector")
    movie_genre = movie_features.get("genre_vector")
    if ref_genre is not None and np.sum(ref_genre) > 0 and movie_genre is not None:
        scores["genre_score"] = compute_genre_similarity(movie_genre, ref_genre)
        active_weights["genre"] = weights["genre"]
    else:
        scores["genre_score"] = 1.0
        
    # 3. Actor
    ref_actor = ref_features.get("actor_vector")
    movie_actor = movie_features.get("actor_vector")
    if ref_actor and movie_actor:
        scores["actor_score"] = compute_actor_similarity(movie_actor, ref_actor)
        active_weights["actor"] = weights["actor"]
    else:
        scores["actor_score"] = 1.0
        
    # 4. Director
    ref_dir = ref_features.get("director_vector")
    movie_dir = movie_features.get("director_vector")
    if ref_dir and movie_dir:
        scores["director_score"] = compute_director_similarity(movie_dir, ref_dir)
        active_weights["director"] = weights["director"]
    else:
        scores["director_score"] = 1.0
        
    # 5. Country
    ref_country = ref_features.get("country_vector")
    movie_country = movie_features.get("country_vector")
    if ref_country is not None and np.sum(ref_country) > 0 and movie_country is not None:
        scores["country_score"] = compute_country_similarity(movie_country, ref_country)
        active_weights["country"] = weights["country"]
    else:
        scores["country_score"] = 1.0
        
    # 6. Decade
    ref_dec = ref_features.get("decade_vector")
    movie_dec = movie_features.get("decade_vector")
    if ref_dec is not None and np.sum(ref_dec) > 0 and movie_dec is not None:
        scores["decade_score"] = compute_decade_similarity(movie_dec, ref_dec)
        active_weights["decade"] = weights["decade"]
    else:
        scores["decade_score"] = 1.0
        
    # 7. Award
    ref_award = ref_features.get("award_vector")
    movie_award = movie_features.get("award_vector")
    if ref_award is not None and np.sum(ref_award) > 0 and movie_award is not None:
        scores["award_score"] = compute_award_similarity(movie_award, ref_award)
        active_weights["award"] = weights["award"]
    else:
        scores["award_score"] = 1.0
        
    # Calculate weighted final score based on active weights
    total_active_weight = sum(active_weights.values())
    if total_active_weight > 0:
        final_score = 0.0
        for key, w in active_weights.items():
            final_score += scores[f"{key}_score"] * w
        final_score = final_score / total_active_weight
    else:
        final_score = 1.0
        
    scores["final_score"] = final_score
    return scores
