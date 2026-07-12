"""
Hybrid Movie Recommender
=========================
Combines two approaches:

1. Content-based filtering — TF-IDF over each movie's genres, compared with
   cosine similarity. Good for "more like this" recommendations and for
   movies with few or no ratings (the "cold start" problem).

2. Collaborative filtering — item-based, using latent factors learned via
   Truncated SVD on the sparse user-item ratings matrix. Captures patterns
   in *how people actually rated things*, not just genre overlap.

The hybrid score is a weighted blend of both, so a recommendation has to be
both genre-relevant AND liked by similar users' rating patterns to rank highly.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


class HybridRecommender:
    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame, n_components: int = 50):
        self.movies = movies.reset_index(drop=True).copy()
        self.movies["genres_clean"] = self.movies["genres"].str.replace("|", " ", regex=False)

        self.movie_id_to_idx = {mid: i for i, mid in enumerate(self.movies["movie_id"])}
        self.idx_to_movie_id = {i: mid for mid, i in self.movie_id_to_idx.items()}

        self._fit_content_model()
        self._fit_collaborative_model(ratings, n_components=n_components)

    # ---------------- Content-based ----------------
    # Note: we store the (small) TF-IDF matrix itself, not the full n x n
    # pairwise similarity matrix — similarity for a single query movie is
    # computed on demand. This keeps the saved model under ~5MB instead of
    # ~250MB, since an n x n float matrix for ~3,900 movies is the expensive part.
    def _fit_content_model(self):
        tfidf = TfidfVectorizer(token_pattern=r"[^\s]+")
        self.tfidf_matrix = tfidf.fit_transform(self.movies["genres_clean"]).astype(np.float32)

    # ---------------- Collaborative (item-based via SVD) ----------------
    def _fit_collaborative_model(self, ratings: pd.DataFrame, n_components: int):
        ratings = ratings[ratings["movie_id"].isin(self.movie_id_to_idx)]

        user_ids = ratings["user_id"].unique()
        user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}

        rows = ratings["movie_id"].map(self.movie_id_to_idx).values
        cols = ratings["user_id"].map(user_id_to_idx).values
        vals = ratings["rating"].values

        n_movies = len(self.movie_id_to_idx)
        n_users = len(user_ids)
        item_user_matrix = csr_matrix((vals, (rows, cols)), shape=(n_movies, n_users))

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.item_factors = svd.fit_transform(item_user_matrix).astype(np.float32)

        self.avg_rating = ratings.groupby("movie_id")["rating"].mean()
        self.rating_count = ratings.groupby("movie_id")["rating"].count()

    # ---------------- Hybrid ----------------
    def recommend(self, title: str, top_n: int = 10, content_weight: float = 0.5):
        matches = self.movies[self.movies["title"].str.lower() == title.lower()]
        if matches.empty:
            matches = self.movies[self.movies["title"].str.lower().str.contains(title.lower())]
        if matches.empty:
            return None, f"No movie found matching '{title}'."

        idx = matches.index[0]
        matched_title = self.movies.loc[idx, "title"]

        content_sim_row = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).ravel()
        collab_sim_row = cosine_similarity(
            self.item_factors[idx].reshape(1, -1), self.item_factors
        ).ravel()

        collab_weight = 1 - content_weight
        hybrid_scores = content_weight * content_sim_row + collab_weight * collab_sim_row

        similar_idx = np.argsort(hybrid_scores)[::-1]
        similar_idx = [i for i in similar_idx if i != idx][:top_n]

        results = self.movies.loc[similar_idx, ["movie_id", "title", "genres"]].copy()
        results["hybrid_score"] = hybrid_scores[similar_idx]
        results["avg_rating"] = results["movie_id"].map(self.avg_rating).round(2)
        results["rating_count"] = results["movie_id"].map(self.rating_count)
        results = results.reset_index(drop=True)

        return matched_title, results

    def search_titles(self, query: str, limit: int = 15):
        matches = self.movies[self.movies["title"].str.lower().str.contains(query.lower())]
        return matches["title"].head(limit).tolist()
