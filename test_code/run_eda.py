import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

csv_path = r"c:\Users\Admin\Desktop\4\DAP391m\code\movie_master\movie_master.csv"
output_dir = r"c:\Users\Admin\Desktop\4\DAP391m\code\movie_master"

print("Loading dataset...")
df = pd.read_csv(csv_path)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

results = {}

# 1. Schema & Overview
print("1. Schema & Overview...")
results["schema"] = {
    "num_rows": int(df.shape[0]),
    "num_cols": int(df.shape[1]),
    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
}

overview_table = []
for col in df.columns:
    non_null_count = int(df[col].notnull().sum())
    non_null_pct = float(non_null_count / len(df) * 100)
    unique_count = int(df[col].nunique())
    sample_val = df[col].dropna().head(2).tolist()
    overview_table.append({
        "column": col,
        "non_null_pct": non_null_pct,
        "unique_count": unique_count,
        "samples": sample_val
    })
results["overview_table"] = overview_table

# 2. Sanity Check
print("2. Sanity Check...")
dup_by_id = int(df.duplicated(subset=['imdb_id']).sum())
dup_by_title_year = int(df.duplicated(subset=['title', 'year']).sum())
cols_missing_gt_50 = [col for col in df.columns if df[col].isnull().sum() / len(df) > 0.5]

rating_neg = int((df['rating'] < 0).sum())
votes_neg = int((df['votes'] < 0).sum())
runtime_neg = int((df['duration_min'] < 0).sum())
meta_neg = int((df['meta_score'] < 0).sum())
year_neg = int((df['year'] < 0).sum())

year_out_of_range = int(((df['year'] < 1888) | (df['year'] > 2028)).sum())

results["sanity_check"] = {
    "duplicate_by_imdb_id": dup_by_id,
    "duplicate_by_title_year": dup_by_title_year,
    "cols_missing_gt_50": cols_missing_gt_50,
    "rating_negative": rating_neg,
    "votes_negative": votes_neg,
    "runtime_negative": runtime_neg,
    "meta_score_negative": meta_neg,
    "release_year_out_of_range": year_out_of_range
}

# 3. Numerical Variables
print("3. Numerical Variables...")
num_cols = ['rating', 'votes', 'meta_score', 'duration_min', 'year', 'release_year']
num_stats = {}
for col in num_cols:
    if col in df.columns:
        desc = df[col].describe()
        num_stats[col] = {
            "count": int(desc["count"]),
            "mean": float(desc["mean"]) if not np.isnan(desc["mean"]) else None,
            "std": float(desc["std"]) if not np.isnan(desc["std"]) else None,
            "min": float(desc["min"]) if not np.isnan(desc["min"]) else None,
            "25%": float(desc["25%"]) if not np.isnan(desc["25%"]) else None,
            "50%": float(desc["50%"]) if not np.isnan(desc["50%"]) else None,
            "75%": float(desc["75%"]) if not np.isnan(desc["75%"]) else None,
            "max": float(desc["max"]) if not np.isnan(desc["max"]) else None,
        }
results["numerical_stats"] = num_stats

# Plotting Rating and Votes Distribution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df['rating'].dropna(), bins=30, kde=True, color='skyblue')
plt.title('Movie Rating Distribution')
plt.xlabel('Rating')

plt.subplot(1, 2, 2)
# Log scale for votes since it is highly skewed
sns.histplot(df['votes'].dropna(), bins=30, kde=True, color='salmon', log_scale=True)
plt.title('Movie Votes Distribution (Log Scale)')
plt.xlabel('Votes')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "numerical_distributions.png"), dpi=150)
plt.close()

# Rating vs Votes Scatter Plot
plt.figure(figsize=(8, 6))
# Sampling to make scatter plot readable and fast
df_sample_scatter = df.dropna(subset=['rating', 'votes']).sample(min(20000, df.dropna(subset=['rating', 'votes']).shape[0]), random_state=42)
sns.scatterplot(data=df_sample_scatter, x='rating', y='votes', alpha=0.3, color='purple')
plt.yscale('log')
plt.title('Movie Rating vs. Vote Count (Log Scale for Votes, Sampled 20k)')
plt.xlabel('Rating')
plt.ylabel('Votes (Log Scale)')
plt.savefig(os.path.join(output_dir, "rating_vs_votes.png"), dpi=150)
plt.close()

# 4. Categorical Variables
print("4. Categorical Variables...")
def process_multi_value_col(col_name):
    # Splits values by | and counts occurrences
    if col_name not in df.columns:
        return []
    series = df[col_name].dropna()
    exploded = series.str.split('|').explode()
    counts = exploded.value_counts()
    total = len(df)
    
    cat_list = []
    for idx, val in counts.items():
        cat_list.append({
            "category": idx,
            "count": int(val),
            "percentage": float(val / total * 100)
        })
    return cat_list

results["categories"] = {
    "genres": process_multi_value_col("genres"),
    "countries_origin": process_multi_value_col("countries_origin"),
    "languages": process_multi_value_col("languages"),
    "directors": process_multi_value_col("directors")[:30], # Top 30
    "production_company": process_multi_value_col("production_company")[:30] # Top 30
}

# Plot Genre Distribution
genres_df = pd.DataFrame(results["categories"]["genres"][:15])
plt.figure(figsize=(10, 6))
sns.barplot(data=genres_df, y='category', x='count', palette='viridis')
plt.title('Top 15 Movie Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "genre_distribution.png"), dpi=150)
plt.close()

# 5. Time Analysis
print("5. Time Analysis...")
# Decades
df['decade_calc'] = (df['year'] // 10) * 10
decade_counts = df['decade_calc'].value_counts().sort_index()
results["decades"] = {
    "counts": {int(k): int(v) for k, v in decade_counts.items()},
    "avg_rating": {int(k): float(v) for k, v in df.groupby('decade_calc')['rating'].mean().dropna().items()}
}

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
decade_counts.plot(kind='line', marker='o', color='forestgreen')
plt.title('Number of Movies by Decade')
plt.xlabel('Decade')
plt.ylabel('Count')
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
df.groupby('decade_calc')['rating'].mean().plot(kind='line', marker='o', color='darkorange')
plt.title('Average Rating by Decade')
plt.xlabel('Decade')
plt.ylabel('Average Rating')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "temporal_trends.png"), dpi=150)
plt.close()

# 6. Text Analysis
print("6. Text Analysis...")
desc_col = 'description_clean' if 'description_clean' in df.columns else 'description'
desc_series = df[desc_col].dropna().astype(str)

char_lens = desc_series.apply(len)
word_lens = desc_series.apply(lambda x: len(x.split()))

empty_count = int(df[desc_col].isnull().sum() + (df[desc_col].astype(str).str.strip() == '').sum())
short_count = int((word_lens < 10).sum()) # count of non-null but short description (<10 words)
total_movies = len(df)

results["text_analysis"] = {
    "column_used": desc_col,
    "char_length": {
        "mean": float(char_lens.mean()),
        "min": int(char_lens.min()) if len(char_lens) > 0 else 0,
        "max": int(char_lens.max()) if len(char_lens) > 0 else 0,
        "median": float(char_lens.median())
    },
    "word_length": {
        "mean": float(word_lens.mean()),
        "min": int(word_lens.min()) if len(word_lens) > 0 else 0,
        "max": int(word_lens.max()) if len(word_lens) > 0 else 0,
        "median": float(word_lens.median())
    },
    "empty_or_whitespace_count": empty_count,
    "empty_percentage": float(empty_count / total_movies * 100),
    "short_description_lt_10_words": short_count,
    "short_description_percentage": float(short_count / total_movies * 100)
}

# 7. Correlation Matrix
print("7. Correlation Matrix...")
corr_cols = ['rating', 'votes', 'meta_score', 'duration_min', 'release_year']
corr_matrix = df[corr_cols].corr()
results["correlation"] = corr_matrix.to_dict()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150)
plt.close()

# 8. Recommendation System Readiness (Split Vector Architecture)
print("8. Recommendation Readiness...")
# Features required for Split Vector Architecture:
# - Description (text): description_clean or description
# - Genre: genres
# - Directors/Stars: directors, stars
# - Country: countries_origin
# - Decade: decade or release_year
# - Awards (sparse fields): has_awards or awards_content

features = {
    "description": df[desc_col].notnull(),
    "genres": df['genres'].notnull(),
    "directors": df['directors'].notnull(),
    "stars": df['stars'].notnull(),
    "countries_origin": df['countries_origin'].notnull(),
    "decade": df['decade'].notnull(),
    "languages": df['languages'].notnull(),
    "production_company": df['production_company'].notnull(),
    "awards_content": df['awards_content'].notnull(),
    "has_awards": df['has_awards'].notnull() # this has 0 or 1, so it is always present technically, but let's look at non-null percentage or value distribution
}

coverages = {}
for feat, mask in features.items():
    coverages[feat] = float(mask.sum() / total_movies * 100)

# Full completeness count (all primary features present)
# Primary: description, genres, directors, stars, countries_origin, decade
primary_mask = (
    df[desc_col].notnull() & 
    df['genres'].notnull() & 
    df['directors'].notnull() & 
    df['stars'].notnull() & 
    df['countries_origin'].notnull() & 
    df['decade'].notnull()
)
full_completeness_pct = float(primary_mask.sum() / total_movies * 100)

results["recommendation_readiness"] = {
    "coverages": coverages,
    "primary_features_completeness_pct": full_completeness_pct,
    "completeness_score_distribution": df['completeness_score'].value_counts().sort_index().to_dict()
}

# Write results json
results_json_path = r"c:\Users\Admin\Desktop\4\DAP391m\code\test_code\eda_results.json"
with open(results_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"EDA Finished. Saved results to {results_json_path}")
