import numpy as np
from dataclasses import dataclass

@dataclass
class UnifiedMovieRepresentation:
    movie_id: str
    title: str
    structured_features: dict
    # structured_features keys: genre_vector, actor_vector, director_vector, country_vector, decade_vector, award_vector
    semantic_embedding: np.ndarray = None
