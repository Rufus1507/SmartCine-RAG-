# Exploratory Data Analysis and Statistical Report: Movie Master Dataset

This report presents a comprehensive Exploratory Data Analysis (EDA) of the cleaned movie metadata dataset `movie_master.csv` used in the CineBot V3 recommendation framework. The analysis serves as descriptive statistics for the academic paper, outlining the characteristics, distribution, completeness, and quality of the underlying data.

---

## 1. Dataset Overview and Schema Analysis

The `movie_master.csv` dataset contains a total of **199,315 rows** and **35 columns**. The file size on disk is **425.03 MB**.

### 1.1 Column Classifications
* **Identifiers & Links**: `imdb_id`, `Movie_Link`, `title`, `title_clean`
* **Numerical Metrics**: `year`, `duration_min`, `rating`, `votes`, `meta_score`, `release_year`, `decade`
* **Categorical (Multi-valued & Single-valued)**: `mpa`, `genres`, `directors`, `writers`, `stars`, `languages`, `countries_origin`, `production_company`
* **Text Fields**: `description`, `description_clean`
* **Sparse Metadata & Indicators**: `release_date`, `release_date_clean`, `filming_locations`, `awards_content`, `has_awards`, `has_oscar`, `has_nomination`
* **System/Model Inputs**: `completeness_score`, `is_duplicate`, `rag_text`, `movie_context`, `final_context`, `tfidf_text`

### 1.2 Detail Schema and Completeness Matrix

The table below describes each column, its data type, number of unique values, and its non-null coverage percentage across the 199,315 records.

| Column Name | Pandas Dtype | Non-Null % | Unique Count | Sample Values |
| :--- | :--- | :--- | :--- | :--- |
| `imdb_id` | object | 100.00% | 188,194 | 'tt0331642', 'tt0331663' |
| `Movie_Link` | object | 100.00% | 188,194 | 'https://www.imdb.com/title/tt0331642/', ... |
| `title` | object | 100.00% | 181,284 | 'Rasigan', 'Sunshine Sally' |
| `title_clean` | object | 99.99% | 181,115 | 'rasigan', 'sunshine sally' |
| `year` | float64 | 95.68% | 138 | 1994.0, 1922.0 |
| `duration_raw` | object | 78.36% | 591 | '2h 20m', '55m' |
| `duration_min` | float64 | 77.29% | 446 | 140.0, 55.0 |
| `mpa` | object | 31.54% | 71 | 'PG-13', 'Not Rated' |
| `rating` | float64 | 53.17% | 91 | 6.9, 7.4 |
| `votes` | float64 | 58.50% | 1,797 | 953.0, 79.0 |
| `meta_score` | float64 | 6.28% | 98 | 60.0, 49.0 |
| `description` | object | 75.50% | 139,306 | 'Vijay and Anitha secretly love each other...' |
| `description_clean` | object | 75.50% | 139,306 | 'Vijay and Anitha secretly love each other...' |
| `release_date` | object | 84.05% | 46,539 | 'July 8, 1994', '16 Dec 1922' |
| `release_date_clean` | object | 84.05% | 32,479 | '1994-07-08', '1922-12-16' |
| `release_year` | float64 | 84.05% | 139 | 1994.0, 1922.0 |
| `decade` | float64 | 84.05% | 15 | 1990.0, 1920.0 |
| `genres` | object | 96.53% | 12,737 | 'Action\|Romance', 'Family\|Musical' |
| `directors` | object | 88.80% | 91,929 | 'S.A. Chandrashekhar', 'Lawson Harris' |
| `writers` | object | 77.64% | 123,633 | 'Shoba Chandrasekar\|S.A. Chandrashekhar' |
| `stars` | object | 88.61% | 174,269 | 'Marie Pavis\|Joy Revelle\|John Cosgrove' |
| `languages` | object | 91.14% | 5,348 | 'Tamil', 'English' |
| `countries_origin` | object | 97.91% | 6,332 | 'India', 'Australia' |
| `filming_locations` | object | 23.28% | 21,503 | 'High Street\|Moorpark\|California\|USA' |
| `production_company`| object | 30.94% | 36,272 | 'B. V. Combines', 'Warner Bros.' |
| `awards_content` | object | 27.45% | 7,401 | '2 wins & 6 nominations', '1 win total' |
| `has_awards` | int64 | 100.00% | 2 | 0, 1 |
| `has_oscar` | int64 | 100.00% | 2 | 0, 1 |
| `has_nomination` | int64 | 100.00% | 2 | 0, 1 |
| `completeness_score`| int64 | 100.00% | 7 | 6, 3 |
| `is_duplicate` | int64 | 100.00% | 1 | 0 |
| `rag_text` | object | 100.00% | 188,194 | 'rasigan. Description: Vijay and Anitha...' |
| `movie_context` | object | 100.00% | 195,701 | 'Genres: Action\|Romance. Director: S.A...' |
| `final_context` | object | 100.00% | 199,307 | 'Title: Rasigan. Description: Vijay...' |
| `tfidf_text` | object | 100.00% | 199,238 | 'rasigan Action\|Romance Vijay...' |

---

## 2. Sanity Check - Data Quality Verification

Although the dataset has gone through standard pipeline cleaning, a verification check is performed to quantify duplicates and potential anomalies. These numbers serve as empirical proof of dataset characteristics in the paper.

| Verification Check | Detected Count | Action Taken / Comment |
| :--- | :--- | :--- |
| Duplicate rows (based on exact `imdb_id`) | 11,121 | Retained as-is (validates baseline characteristics). |
| Duplicate rows (based on exact `title` + `year`) | 11,213 | Reflects duplicate mappings across different IDs. |
| Columns with >50% missing values | 5 columns | `mpa`, `meta_score`, `filming_locations`, `production_company`, `awards_content`. |
| Row count with negative numerical values | 0 | Checked `rating`, `votes`, `meta_score`, `duration_min`, `year`. |
| Out-of-bounds release years (<1888 or >2028) | 0 | Earliest year is 1888, latest year is 2028. All within bounds. |

---

## 3. Analysis of Numerical Features

Descriptive statistical analysis of key numerical features highlights the ranges, centers, and spread of the data.

### 3.1 Descriptives Table

| Metric | Movie Rating (`rating`) | Vote Count (`votes`) | Meta Score (`meta_score`) | Runtime (`duration_min`) | Release Year (`year`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Count** | 105,976 | 116,606 | 12,513 | 154,060 | 190,706 |
| **Mean** | 6.34 | 17,134.23 | 60.85 | 75.36 | 1,993.90 |
| **Std. Dev.**| 1.22 | 87,256.38 | 17.57 | 176.88 | 32.33 |
| **Minimum** | 1.00 | 5.00 | 1.00 | 1.00 | 1,888.00 |
| **25% (Q1)** | 5.60 | 45.00 | 49.00 | 45.00 | 1,972.00 |
| **50% (Q2)** | 6.40 | 351.00 | 62.00 | 85.00 | 2,007.00 |
| **75% (Q3)** | 7.10 | 3,000.00 | 74.00 | 99.00 | 2,021.00 |
| **Maximum** | 10.00 | 3,200,000.00 | 100.00 | 51,420.00 | 2,028.00 |

### 3.2 Key Distribution Findings
* **User Ratings**: Shows a relatively normal distribution centered around 6.4, with a slight left-skew toward lower values.
* **Vote Counts**: Exhibits an extreme positive skew (long-tail distribution). The median votes is only 351, while the maximum is 3.2 million. Standard algorithms must handle this tail to avoid popularity bias.
* **Runtimes**: The median runtime is 85 minutes. The maximum runtime is 51,420 minutes, representing multi-day experimental films (e.g., Logistics) or series packages present on IMDb.
* **Meta Score**: Strongly under-represented (only 6.28% coverage), but when present, behaves as a bell curve centered around 60.8.

### 3.3 Visualizations

#### Rating & Vote Distribution
A visualization is necessary to compare the normal shape of rating frequency against the highly skewed log-scale of user vote counts.
![Rating and Votes Distribution](movie_master/numerical_distributions.png)

#### Rating vs. Votes Scatter Plot
A scatter plot helps identify rating extremes. Movies with low vote counts show maximum rating variance (e.g., ratings of 1.0 or 10.0), whereas highly-voted movies converge on ratings between 6.0 and 9.0.
![Rating vs Votes Scatter Plot](movie_master/rating_vs_votes.png)

---

## 4. Analysis of Categorical Features

For multi-valued columns (e.g., `genres`, `countries_origin`, `languages`, `directors`, and `production_company`), records were exploded along the pipe delimiter (`|`) to calculate exact occurrences.

### 4.1 Genres (Top 15 Categories)
Drama remains the dominant genre, accounting for over 38% of all entries. Short films are the second largest segment (23.94%), indicating a heavy presence of historical or independent short films on IMDb.

| Genre | Film Count | % of Total Dataset |
| :--- | :---: | :---: |
| Drama | 76,386 | 38.32% |
| Short | 47,711 | 23.94% |
| Comedy | 44,962 | 22.56% |
| Documentary | 34,071 | 17.09% |
| Romance | 21,744 | 10.91% |
| Thriller | 17,010 | 8.53% |
| Crime | 16,237 | 8.15% |
| Action | 15,930 | 7.99% |
| Horror | 14,386 | 7.22% |
| Adventure | 12,453 | 6.25% |
| Mystery | 9,181 | 4.61% |
| Animation | 8,702 | 4.37% |
| Fantasy | 7,415 | 3.72% |
| Music | 7,373 | 3.70% |
| Family | 7,255 | 3.64% |

A bar plot highlights the exponential decrease in genre frequency, showcasing the dominance of the top four genres.
![Top 15 Genres](movie_master/genre_distribution.png)

### 4.2 Country of Origin (Top 10 Categories)
The United States and United Kingdom are the leading production locations, with France as the top non-English dominant producer.

| Country of Origin | Film Count | % of Total Dataset |
| :--- | :---: | :---: |
| United States | 73,869 | 37.06% |
| United Kingdom | 17,722 | 8.89% |
| France | 15,796 | 7.93% |
| USA (Alternative IMDb tag) | 10,622 | 5.33% |
| Italy | 9,390 | 4.71% |
| Canada | 8,626 | 4.33% |
| Germany | 8,314 | 4.17% |
| India | 7,382 | 3.70% |
| Spain | 6,551 | 3.29% |
| Japan | 5,121 | 2.57% |

### 4.3 Languages (Top 10 Categories)
English is by far the most represented language, accounting for over 42.33% of films, followed by European and Asian languages.

| Language | Film Count | % of Total Dataset |
| :--- | :---: | :---: |
| English | 84,365 | 42.33% |
| French | 19,163 | 9.61% |
| Spanish | 12,903 | 6.47% |
| German | 11,119 | 5.58% |
| Italian | 9,815 | 4.92% |
| Japanese | 6,516 | 3.27% |
| Hindi | 4,976 | 2.50% |
| Tamil | 4,220 | 2.12% |
| Russian | 4,181 | 2.10% |
| Portuguese | 3,959 | 1.99% |

### 4.4 Directors (Top 10 Most Prolific)
Historic and B-movie directors dominate the count, with William Beaudine leading the dataset with 165 films.

| Director | Film Count | % of Total Dataset |
| :--- | :---: | :---: |
| William Beaudine | 165 | 0.08% |
| Michael Curtiz | 157 | 0.08% |
| Richard Thorpe | 147 | 0.07% |
| Jesús Franco | 141 | 0.07% |
| Sam Newfield | 132 | 0.07% |
| John Ford | 121 | 0.06% |
| Lesley Selander | 120 | 0.06% |
| Raoul Walsh | 117 | 0.06% |
| Joseph Kane | 114 | 0.06% |
| Lew Landers | 111 | 0.06% |

### 4.5 Production Company (Top 10 Categories)
Legacy Hollywood studios are the most represented entities. Paramount Pictures, Warner Bros., and Universal Pictures occupy the top three spots.

| Production Company | Film Count | % of Total Dataset |
| :--- | :---: | :---: |
| Paramount Pictures | 1,786 | 0.90% |
| Warner Bros. | 1,607 | 0.81% |
| Universal Pictures | 1,595 | 0.80% |
| Metro-Goldwyn-Mayer (MGM)| 1,473 | 0.74% |
| Columbia Pictures | 1,472 | 0.74% |
| Twentieth Century Fox | 1,178 | 0.59% |
| Fox Film Corporation | 697 | 0.35% |
| RKO Radio Pictures | 659 | 0.33% |
| Republic Pictures (I) | 593 | 0.30% |
| Universal International Pictures (UI) | 331 | 0.17% |

---

## 5. Temporal Trend Analysis

Grouping dataset films by decade reveals a massive exponential growth in movie production over the last 140 years, with the majority of the content originating from the 2000s, 2010s, and 2020s.

### 5.1 Decadal Movie Counts and Average Ratings

| Decade | Film Count | Average Rating |
| :--- | :---: | :---: |
| 1880 | 2 | 6.15 |
| 1890 | 249 | 5.12 |
| 1900 | 673 | 5.39 |
| 1910 | 2,717 | 5.58 |
| 1920 | 7,569 | 6.15 |
| 1930 | 7,312 | 6.22 |
| 1940 | 7,605 | 6.33 |
| 1950 | 8,576 | 6.33 |
| 1960 | 9,939 | 6.22 |
| 1970 | 10,931 | 6.02 |
| 1980 | 11,816 | 6.06 |
| 1990 | 13,648 | 6.17 |
| 2000 | 20,730 | 6.43 |
| 2010 | 33,684 | 6.52 |
| 2020 | 55,255 | 6.46 |

### 5.2 Key Temporal Observations
* Movie volume has increased dramatically. More than **44%** of the movies in the dataset (88,939 films) were released from 2010 onwards.
* Ratings fluctuate slightly between decades but have remained relatively stable. There is a small rise in average ratings in the modern eras (2000s - 2020s), hovering around 6.45 - 6.52, compared to 6.02 in the 1970s.

A dual line plot shows the exponential increase in movie production volume and the stability of average movie ratings across decades.
![Temporal Trends](movie_master/temporal_trends.png)

---

## 6. Text Property Analysis (Description Fields)

Text properties of movie summaries (`description_clean`) are vital for indexing and semantic retrieval models (such as BERT or Sentence-Transformers embeddings).

* **Character Length**:
  * **Mean**: 282.51 characters
  * **Median**: 200.00 characters
  * **Minimum**: 1 character
  * **Maximum**: 15,558 characters
* **Word Count**:
  * **Mean**: 48.21 words
  * **Median**: 34.00 words
  * **Minimum**: 1 word
  * **Maximum**: 2,347 words
* **Sparsity Indicators**:
  * **Empty Description Count**: 48,838 films (**24.50%** of the dataset)
  * **Ultra-short Description (<10 words)**: 2,963 films (**1.49%** of the dataset)

---

## 7. Correlation Analysis

A correlation analysis was performed to verify dependencies between numerical fields.

| Variable | Rating | Votes | Meta Score | Runtime | Release Year |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Rating** | 1.000 | 0.150 | 0.705 | 0.003 | 0.108 |
| **Votes** | 0.150 | 1.000 | 0.180 | 0.025 | 0.093 |
| **Meta Score** | 0.705 | 0.180 | 1.000 | 0.195 | -0.160 |
| **Runtime** | 0.003 | 0.025 | 0.195 | 1.000 | -0.042 |
| **Release Year**| 0.108 | 0.093 | -0.160 | -0.042 | 1.000 |

### 7.1 Key Observations
* **Rating vs. Meta Score**: A strong positive correlation (**0.705**) exists. This is expected as both represent critical evaluations, though they are sourced differently (IMDb users vs. Metacritic reviews).
* **Rating vs. Release Year**: A small positive correlation (**0.108**) indicating minor upward rating inflation in newer movies.
* **Meta Score vs. Release Year**: A negative correlation (**-0.160**) indicating that critics are slightly more critical of modern films or that historical films in the database have survivor bias.
* **Runtime**: Shows almost no correlation with ratings (**0.003**) or votes (**0.025**).

A heatmap helps visualize the strong relationship between user ratings and critic meta scores, contrasted against the near-zero correlation of movie runtime.
![Correlation Heatmap](movie_master/correlation_heatmap.png)

---

## 8. Recommendation System Readiness Assessment (Split Vector Architecture)

The CineBot V3 utilizes a **Split Vector Architecture** that merges text embeddings (description) with metadata fields (genres, directors, stars, countries, decade, awards) to perform hybrid recommendations.

### 8.1 Individual Feature Coverage

The effectiveness of individual vectors relies on data availability. The table below lists the availability (completeness) of each attribute:

| Target Vector Feature | Dataset Coverage (%) | Sparsity Level | Role in Recommendation System |
| :--- | :---: | :--- | :--- |
| `countries_origin` | 97.91% | Very Low | Metadata filtering / regional bias |
| `genres` | 96.53% | Very Low | Content filtering / keyword index |
| `languages` | 91.14% | Low | Language preferences filtering |
| `directors` | 88.80% | Low | Director profile matching |
| `stars` | 88.61% | Low | Cast similarity indexing |
| `decade` | 84.05% | Low | Epoch categorization |
| `description` | 75.50% | Moderate | Text embedding (Semantic Search) |
| `production_company`| 30.94% | High | Studio bias matching |
| `awards_content` | 27.45% | High | Award descriptive mapping |
| `has_awards` | 100.00%* | Zero (Binary) | Prominence boosting (weighted scoring) |

*\*Note: `has_awards` is computed as an indicator based on `awards_content` and is fully populated with `0` or `1` for all rows.*

### 8.2 Primary Features Completeness Score

A movie profile is considered **fully complete** if it contains all 6 primary features required for the base Split Vector Model: `description`, `genres`, `directors`, `stars`, `countries_origin`, and `decade`.

* **Full Completeness Percentage**: **58.31%** (116,213 films)
* **Partial Completeness Percentage**: **41.69%** (83,102 films)

The distribution of the dataset completeness scores (ranging from 1 to 7 matching fields) is structured as follows:

| Completeness Score | Film Count | % of Dataset |
| :---: | :---: | :---: |
| 1 | 1,294 | 0.65% |
| 2 | 20,012 | 10.04% |
| 3 | 33,905 | 17.01% |
| 4 | 36,400 | 18.26% |
| 5 | 17,695 | 8.88% |
| 6 | 77,518 | 38.89% |
| 7 (Max) | 12,491 | 6.27% |

### 8.3 Empirical Justification for RQ2 (Dynamic Weight Redistribution)
These completeness metrics provide critical evidence for the paper:
1. Since **only 58.31%** of movies have complete primary features, a static vector similarity weight scheme (e.g., assigning fixed weights to description, genres, directors, stars, countries, decade) would result in significant information decay or false zeroes when calculating similarity for the other **41.69%** of the dataset.
2. High-sparsity features like `production_company` (30.94%) and `awards_content` (27.45%) represent severe data sparsity. If included in standard cosine calculations without adjustments, they distort the cosine distance.
3. This empirical distribution directly motivates the design of **Dynamic Weight Redistribution (RQ2)**, which dynamically adjusts weights at runtime based on the completeness profile of the queried seed movie, ensuring robust recommendations across all density ranges.

---

## 9. Generated Visualization Assets

All charts have been generated and saved locally in the `movie_master/` folder. They can be referenced in the paper using the following paths:

1. **Rating & Votes Distribution Plot**: `movie_master/numerical_distributions.png`
2. **Scatter Plot (Rating vs. Votes)**: `movie_master/rating_vs_votes.png`
3. **Genre Frequency Bar Chart**: `movie_master/genre_distribution.png`
4. **Decadal Temporal Trends Plot**: `movie_master/temporal_trends.png`
5. **Correlation Heatmap**: `movie_master/correlation_heatmap.png`
