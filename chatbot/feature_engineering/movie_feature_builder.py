import os
import re
import json
import ast
import numpy as np
import pandas as pd
from collections import Counter

FEATURE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(FEATURE_DIR, "vocabularies.json")
ACTOR_METADATA_PATH = os.path.join(FEATURE_DIR, "actor_metadata.json")
DIRECTOR_METADATA_PATH = os.path.join(FEATURE_DIR, "director_metadata.json")

# Define 22 parent genres
PARENT_GENRES = [
    "Drama", "Short", "Comedy", "Documentary", "Romance", "Thriller", "Crime", 
    "Action", "Horror", "Adventure", "Mystery", "Animation", "Fantasy", "Music", 
    "Family", "Sci-Fi", "Biography", "History", "Western", "War", "Musical", "Sport"
]

# Genre Hierarchy Mapping (subgenre -> parent genre(s))
GENRE_HIERARCHY = {
    "Dark Comedy": ["Comedy"],
    "Satire": ["Comedy"],
    "Period Drama": ["Drama"],
    "Slapstick": ["Comedy"],
    "Film Noir": ["Thriller", "Crime"],
    "Reality-TV": ["Documentary"],
    "Parody": ["Comedy"],
    "Romantic Comedy": ["Comedy", "Romance"],
    "Tragedy": ["Drama"],
    "Slasher Horror": ["Horror"],
    "Psychological Thriller": ["Thriller"],
    "Supernatural Horror": ["Horror"],
    "Psychological Drama": ["Drama"],
    "Talk-Show": ["Documentary"],
    "News": ["Documentary"],
    "Coming-of-Age": ["Drama"],
    "Superhero": ["Action", "Fantasy"],
    "Docudrama": ["Drama", "Documentary"],
    "True Crime": ["Crime", "Documentary"],
    "Anime": ["Animation"],
    "Classical Western": ["Western"],
    "Martial Arts": ["Action"],
    "Buddy Comedy": ["Comedy"],
    "Spaghetti Western": ["Western"],
    "Farce": ["Comedy"],
    "Whodunnit": ["Mystery"],
    "Computer Animation": ["Animation"],
    "Game-Show": ["Comedy"],
    "Screwball Comedy": ["Comedy"],
    "Epic": ["Drama"],
    "Space Sci-Fi": ["Sci-Fi"],
    "Quest": ["Adventure"],
    "Animal Adventure": ["Adventure"],
    "Jungle Adventure": ["Adventure"],
    "One-Person Army Action": ["Action"],
    "Dark Fantasy": ["Fantasy"],
    "Teen Drama": ["Drama"],
    "Supernatural Fantasy": ["Fantasy"],
    "Teen Comedy": ["Comedy"],
    "Political Drama": ["Drama"],
    "Psychological Horror": ["Horror"],
    "Caper": ["Crime"],
    "Spy": ["Action", "Thriller"],
    "Adult Animation": ["Animation"],
    "B-Horror": ["Horror"],
    "Tragic Romance": ["Romance", "Drama"],
    "Teen Horror": ["Horror"],
    "Feel-Good Romance": ["Romance"],
    "Erotic Thriller": ["Thriller"],
    "Swashbuckler": ["Adventure"],
    "Survival": ["Adventure", "Thriller"],
    "Hand-Drawn Animation": ["Animation"],
    "Road Trip": ["Adventure", "Comedy"],
    "Suspense Mystery": ["Mystery"],
    "Monster Horror": ["Horror"],
    "Quirky Comedy": ["Comedy"],
    "Dystopian Sci-Fi": ["Sci-Fi"],
    "Body Horror": ["Horror"],
    "Action Epic": ["Action"],
    "Raunchy Comedy": ["Comedy"],
    "Alien Invasion": ["Sci-Fi"],
    "Political Thriller": ["Thriller"],
    "Adventure Epic": ["Adventure"],
    "Costume Drama": ["Drama"],
    "Steamy Romance": ["Romance"],
    "Serial Killer": ["Thriller", "Crime"],
    "High-Concept Comedy": ["Comedy"],
    "Teen Romance": ["Romance"],
    "Workplace Drama": ["Drama"],
    "Conspiracy Thriller": ["Thriller"],
    "Gangster": ["Crime"],
    "Vampire Horror": ["Horror"],
    "Kaiju": ["Action", "Sci-Fi"],
    "Zombie Horror": ["Horror"],
    "Holiday": ["Family"],
    "Dark Romance": ["Romance"],
    "Urban Adventure": ["Adventure"],
    "Heist": ["Crime"],
    "Legal Drama": ["Drama"],
    "War Epic": ["War"],
    "Film-Noir": ["Crime", "Thriller"],
    "Fairy Tale": ["Fantasy"],
    "Sea Adventure": ["Adventure"],
    "Showbiz Drama": ["Drama"],
    "Sci-Fi Epic": ["Sci-Fi"],
    "Folk Horror": ["Horror"],
    "Sword & Sorcery": ["Fantasy"],
    "Splatter Horror": ["Horror"],
    "Prison Drama": ["Drama"],
    "Disaster": ["Action", "Thriller"],
    "Giallo": ["Thriller", "Horror"],
    "Globetrotting Adventure": ["Adventure"],
    "Time Travel": ["Sci-Fi"],
    "Desert Adventure": ["Adventure"],
    "Fantasy Epic": ["Fantasy"],
    "Historical Epic": ["History"],
    "Dinosaur Adventure": ["Adventure"],
    "Kung Fu": ["Action"],
    "B-Action": ["Action"],
    "Police Procedural": ["Crime"],
    "Boxing": ["Sport"],
    "Cyberpunk": ["Sci-Fi"],
    "Classic Musical": ["Musical"],
    "Pop Musical": ["Musical"],
    "Gun Fu": ["Action"],
    "Artificial Intelligence": ["Sci-Fi"],
    "Holiday Romance": ["Romance"],
    "Teen Adventure": ["Adventure"],
    "Holiday Comedy": ["Comedy"],
    "Cop Drama": ["Drama", "Crime"],
    "Buddy Cop": ["Comedy", "Crime"],
    "Found Footage Horror": ["Horror"],
    "Basketball": ["Sport"],
    "Contemporary Western": ["Western"],
    "Medical Drama": ["Drama"],
    "Stop Motion Animation": ["Animation"],
    "Romantic Epic": ["Romance"],
    "Car Action": ["Action"],
    "Sword & Sandal": ["Adventure"],
    "Samurai": ["Action"],
    "Teen Fantasy": ["Fantasy"],
    "Mountain Adventure": ["Adventure"],
    "Legal Thriller": ["Thriller"],
    "Werewolf Horror": ["Horror"],
    "Jukebox Musical": ["Musical"],
    "Music Documentary": ["Documentary", "Music"],
    "Football": ["Sport"],
    "Wuxia": ["Action"],
    "Cyber Thriller": ["Thriller"],
    "Sports Documentary": ["Documentary", "Sport"],
    "Steampunk": ["Sci-Fi"],
    "Baseball": ["Sport"],
    "Drug Crime": ["Crime"],
    "Concert": ["Music"],
    "Sketch Comedy": ["Comedy"],
    "Stoner Comedy": ["Comedy"],
    "Rock Musical": ["Musical"],
    "Body Swap Comedy": ["Comedy"],
    "Witch Horror": ["Horror"],
    "Holiday Family": ["Family"],
    "Motorsport": ["Sport"],
    "Western Epic": ["Western"],
    "Extreme Sport": ["Sport"],
    "Faith & Spirituality Documentary": ["Documentary"],
    "Bumbling Detective": ["Comedy"],
    "Hard-boiled Detective": ["Crime", "Thriller"],
    "Stand-Up": ["Comedy"],
    "Crime Documentary": ["Documentary", "Crime"],
    "Nature Documentary": ["Documentary"],
    "Political Documentary": ["Documentary"],
    "Military Documentary": ["Documentary"],
    "Water Sport": ["Sport"],
    "Adult": ["Drama"],
    "Cozy Mystery": ["Mystery"],
    "Mockumentary": ["Comedy", "Documentary"],
    "Financial Drama": ["Drama"],
    "History Documentary": ["Documentary", "History"],
    "Mecha": ["Sci-Fi", "Animation"],
    "Shōnen": ["Animation", "Action"],
    "Science & Technology Documentary": ["Documentary"],
    "Food Documentary": ["Documentary"],
    "Soccer": ["Sport"],
    "Seinen": ["Animation"],
    "Travel Documentary": ["Documentary"],
    "Shōjo": ["Animation"],
    "Slice of Life": ["Drama"],
    "Isekai": ["Animation", "Fantasy"],
    "Iyashikei": ["Drama"],
    "Holiday Animation": ["Animation"],
    "Soap Opera": ["Drama"],
    "Korean Drama": ["Drama"],
    "Sitcom": ["Comedy"],
    "Talk Show": ["Documentary"],
    "Reality TV": ["Documentary"],
    "Josei": ["Animation"],
    "Game Show": ["Comedy"]
}

# Standard list of decades
DECADES = [1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]

def clean_split(val):
    if pd.isna(val):
        return []
    val_str = str(val).strip()
    if not val_str:
        return []
    # Support pipe '|' and comma ',' split
    if "|" in val_str:
        return [item.strip() for item in val_str.split("|") if item.strip() and item.strip() != "None"]
    if "," in val_str:
        return [item.strip() for item in val_str.split(",") if item.strip() and item.strip() != "None"]
    return [val_str]

class MovieFeatureBuilder:
    def __init__(self):
        self.vocabularies = {}
        self.actor_metadata = {}
        self.director_metadata = {}
        
        # Load vocabularies if they exist
        if os.path.exists(VOCAB_PATH):
            with open(VOCAB_PATH, "r", encoding="utf-8") as f:
                self.vocabularies = json.load(f)
            if os.path.exists(ACTOR_METADATA_PATH):
                with open(ACTOR_METADATA_PATH, "r", encoding="utf-8") as f:
                    self.actor_metadata = json.load(f)
            if os.path.exists(DIRECTOR_METADATA_PATH):
                with open(DIRECTOR_METADATA_PATH, "r", encoding="utf-8") as f:
                    self.director_metadata = json.load(f)
            
            # Map items for quick search
            self.actor_to_idx = {name: idx for idx, name in enumerate(self.vocabularies["actors"])}
            self.director_to_idx = {name: idx for idx, name in enumerate(self.vocabularies["directors"])}
            self.country_to_idx = {name: idx for idx, name in enumerate(self.vocabularies["countries"])}
            self.genre_to_idx = {name: idx for idx, name in enumerate(PARENT_GENRES)}
            self.decade_to_idx = {dec: idx for idx, dec in enumerate(DECADES)}
            
    def fit(self, df: pd.DataFrame):
        """
        Builds vocabularies and tier classifications from the main DataFrame.
        """
        print("Fitting MovieFeatureBuilder...")
        actor_counts = Counter()
        director_counts = Counter()
        countries_set = set()
        
        for _, row in df.iterrows():
            actors = clean_split(row.get('stars'))
            directors = clean_split(row.get('directors'))
            countries = clean_split(row.get('countries_origin'))
            
            actor_counts.update(actors)
            director_counts.update(directors)
            countries_set.update(countries)
            
        # Classify actors into Tiers
        self.vocabularies["actors"] = []
        self.actor_metadata = {}
        for name, count in actor_counts.items():
            if count >= 100:
                tier = "Tier A"
            elif count >= 50:
                tier = "Tier B"
            elif count >= 20:
                tier = "Tier C"
            else:
                tier = "Tier D"
                
            self.actor_metadata[name] = {
                "actor_name": name,
                "movie_count": count,
                "actor_tier": tier
            }
            # Only keep Tier A, B, C in the active vector vocabulary
            if tier in ("Tier A", "Tier B", "Tier C"):
                self.vocabularies["actors"].append(name)
                
        # Sort actor vocabulary alphabetically
        self.vocabularies["actors"].sort()
        
        # Classify directors into Tiers
        self.vocabularies["directors"] = []
        self.director_metadata = {}
        for name, count in director_counts.items():
            if count >= 100:
                tier = "Tier A"
            elif count >= 50:
                tier = "Tier B"
            elif count >= 20:
                tier = "Tier C"
            else:
                tier = "Tier D"
                
            self.director_metadata[name] = {
                "director_name": name,
                "movie_count": count,
                "director_tier": tier
            }
            # Only keep Tier A, B, C in the active vector vocabulary
            if tier in ("Tier A", "Tier B", "Tier C"):
                self.vocabularies["directors"].append(name)
                
        # Sort director vocabulary alphabetically
        self.vocabularies["directors"].sort()
        
        # Sort countries alphabetically
        self.vocabularies["countries"] = sorted(list(countries_set))
        self.vocabularies["decades"] = DECADES
        
        # Save to disk
        os.makedirs(FEATURE_DIR, exist_ok=True)
        with open(VOCAB_PATH, "w", encoding="utf-8") as f:
            json.dump(self.vocabularies, f, ensure_ascii=False, indent=2)
        with open(ACTOR_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.actor_metadata, f, ensure_ascii=False, indent=2)
        with open(DIRECTOR_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.director_metadata, f, ensure_ascii=False, indent=2)
            
        # Map items for quick search
        self.actor_to_idx = {name: idx for idx, name in enumerate(self.vocabularies["actors"])}
        self.director_to_idx = {name: idx for idx, name in enumerate(self.vocabularies["directors"])}
        self.country_to_idx = {name: idx for idx, name in enumerate(self.vocabularies["countries"])}
        self.genre_to_idx = {name: idx for idx, name in enumerate(PARENT_GENRES)}
        self.decade_to_idx = {dec: idx for idx, dec in enumerate(DECADES)}
        
        print("Feature vocabularies built and saved successfully!")
        print(f"Active Actors (Tier A,B,C): {len(self.vocabularies['actors'])}")
        print(f"Active Directors (Tier A,B,C): {len(self.vocabularies['directors'])}")
        print(f"Active Countries: {len(self.vocabularies['countries'])}")

    def transform_row(self, row) -> dict:
        """
        Converts a movie row into structured vectors.
        Returns a dictionary containing the vectors.
        """
        # 1. Genre Vector (Multi-hot over 22 parent genres)
        genre_vec = np.zeros(len(PARENT_GENRES), dtype=np.float32)
        movie_genres = clean_split(row.get('genres'))
        for g in movie_genres:
            # Map through hierarchy if it is a subgenre
            mapped = GENRE_HIERARCHY.get(g, [g] if g in PARENT_GENRES else [])
            for mg in mapped:
                if mg in self.genre_to_idx:
                    genre_vec[self.genre_to_idx[mg]] = 1.0
                    
        # 2. Actor Vector (Sparse indices representation)
        actor_indices = []
        movie_actors = clean_split(row.get('stars'))
        for a in movie_actors:
            if a in self.actor_to_idx:
                actor_indices.append(self.actor_to_idx[a])
                
        # 3. Director Vector (Sparse indices representation)
        director_indices = []
        movie_dirs = clean_split(row.get('directors'))
        for d in movie_dirs:
            if d in self.director_to_idx:
                director_indices.append(self.director_to_idx[d])
                
        # 4. Country Vector (Multi-hot)
        country_vec = np.zeros(len(self.vocabularies["countries"]), dtype=np.float32)
        movie_countries = clean_split(row.get('countries_origin'))
        for c in movie_countries:
            if c in self.country_to_idx:
                country_vec[self.country_to_idx[c]] = 1.0
                
        # 5. Decade Vector (One-hot)
        decade_vec = np.zeros(len(DECADES), dtype=np.float32)
        movie_dec = row.get('decade')
        if pd.isna(movie_dec) and pd.notna(row.get('Year')):
            # Compute decade from year
            try:
                yr = int(float(row.get('Year')))
                movie_dec = float((yr // 10) * 10)
            except Exception:
                pass
        
        if pd.notna(movie_dec):
            dec_val = int(float(movie_dec))
            if dec_val in self.decade_to_idx:
                decade_vec[self.decade_to_idx[dec_val]] = 1.0
                
        # 6. Award Vector
        award_vec = np.zeros(3, dtype=np.float32)
        award_vec[0] = float(row.get('has_awards', 0) == 1)
        award_vec[1] = float(row.get('has_oscar', 0) == 1)
        award_vec[2] = float(row.get('has_nomination', 0) == 1)
        
        return {
            "genre_vector": genre_vec.tolist(),
            "actor_vector": actor_indices,
            "director_vector": director_indices,
            "country_vector": country_vec.tolist(),
            "decade_vector": decade_vec.tolist(),
            "award_vector": award_vec.tolist()
        }
