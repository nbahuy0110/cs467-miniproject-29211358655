import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.nlp_pipeline import NLPPipeline


class GigRecommender:
    def __init__(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.items_df = items_df.reset_index(drop=True).copy()
        self.ratings_df = ratings_df.copy()
        self.item_id_to_idx = {
            iid: idx for idx, iid in enumerate(self.items_df["item_id"])
        }

        # Fit NLP Pipeline
        self.nlp = NLPPipeline()
        self.tfidf_matrix = self.nlp.fit_transform_items(self.items_df)
        self.content_sim_matrix = cosine_similarity(self.tfidf_matrix)

        # Build User-Item Matrix for Collaborative Filtering
        self.user_item_matrix = self.ratings_df.pivot_table(
            index="user_id", columns="item_id", values="rating"
        ).reindex(columns=self.items_df["item_id"]).fillna(0)

        # Item-Item Collaborative Filtering similarity matrix
        item_user_matrix = self.user_item_matrix.T.values
        self.cf_item_sim = cosine_similarity(item_user_matrix)

    def _get_rated_items(self, user_id: str):
        if user_id not in self.user_item_matrix.index:
            return set()
        return set(self.ratings_df[self.ratings_df["user_id"] == user_id]["item_id"])

    # 1. Popularity Baseline
    def recommend_popularity(self, user_id: str = None, top_n: int = 5, min_ratings: int = 2):
        stats = self.ratings_df.groupby("item_id")["rating"].agg(["mean", "count"]).reset_index()
        popular = stats[stats["count"] >= min_ratings].copy()
        # Weighted popularity score (Bayesian average)
        global_mean = self.ratings_df["rating"].mean()
        popular["score"] = (
            (popular["count"] * popular["mean"] + min_ratings * global_mean)
            / (popular["count"] + min_ratings)
        )

        if user_id:
            rated = self._get_rated_items(user_id)
            popular = popular[~popular["item_id"].isin(rated)]

        popular = popular.sort_values(by="score", ascending=False).head(top_n)
        merged = popular.merge(self.items_df, on="item_id")
        return [
            {
                "item_id": row["item_id"],
                "gig_title": row["gig_title"],
                "category": row["category"],
                "score": round(float(row["score"]), 4),
                "matched_keywords": ["popular", f"{int(row['count'])} ratings"]
            }
            for _, row in merged.iterrows()
        ]

    # 2a. NLP Content-Based (Free-text Query / Freelancer Bio)
    def recommend_by_query(self, query_text: str, top_n: int = 5):
        query_vec = self.nlp.transform_query(query_text)
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:top_n]

        results = []
        for idx in top_indices:
            item_row = self.items_df.iloc[idx]
            keywords = self.nlp.get_matching_keywords(query_vec, self.tfidf_matrix[idx], top_k=4)
            results.append({
                "item_id": item_row["item_id"],
                "gig_title": item_row["gig_title"],
                "category": item_row["category"],
                "score": round(float(sims[idx]), 4),
                "matched_keywords": keywords
            })
        return results

    # 2b. NLP Content-Based (User Profile from Past Ratings)
    def _build_user_profile_vec(self, user_id: str):
        user_ratings = self.ratings_df[
            (self.ratings_df["user_id"] == user_id) & (self.ratings_df["rating"] >= 4)
        ]
        if user_ratings.empty:
            user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if user_ratings.empty:
            return None

        weights = user_ratings["rating"].values
        indices = [self.item_id_to_idx[iid] for iid in user_ratings["item_id"] if iid in self.item_id_to_idx]
        if not indices:
            return None

        weighted_vecs = self.tfidf_matrix[indices].T.dot(weights) / np.sum(weights)
        return np.asarray(weighted_vecs).reshape(1, -1)

    def recommend_content_for_user(self, user_id: str, top_n: int = 5):
        profile_vec = self._build_user_profile_vec(user_id)
        if profile_vec is None:
            return self.recommend_popularity(user_id, top_n=top_n)

        sims = cosine_similarity(profile_vec, self.tfidf_matrix).flatten()
        rated = self._get_rated_items(user_id)

        scored_indices = [
            (idx, sims[idx])
            for idx in range(len(self.items_df))
            if self.items_df.iloc[idx]["item_id"] not in rated
        ]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        results = []
        from scipy.sparse import csr_matrix
        profile_sparse = csr_matrix(profile_vec)
        for idx, score in scored_indices[:top_n]:
            item_row = self.items_df.iloc[idx]
            keywords = self.nlp.get_matching_keywords(profile_sparse, self.tfidf_matrix[idx], top_k=4)
            results.append({
                "item_id": item_row["item_id"],
                "gig_title": item_row["gig_title"],
                "category": item_row["category"],
                "score": round(float(score), 4),
                "matched_keywords": keywords
            })
        return results

    # 3. Extension: Item-Based Collaborative Filtering + Hybrid Recommender
    def predict_cf_rating(self, user_id: str, item_id: str) -> float:
        if user_id not in self.user_item_matrix.index or item_id not in self.item_id_to_idx:
            return float(self.ratings_df["rating"].mean())

        target_idx = self.item_id_to_idx[item_id]
        user_row = self.user_item_matrix.loc[user_id].values
        rated_mask = user_row > 0
        if not np.any(rated_mask):
            return float(self.ratings_df["rating"].mean())

        sims = self.cf_item_sim[target_idx][rated_mask]
        ratings = user_row[rated_mask]
        denom = np.sum(np.abs(sims))
        if denom == 0:
            return float(np.mean(ratings))
        return float(np.dot(sims, ratings) / denom)

    def recommend_hybrid(self, user_id: str, top_n: int = 5, alpha: float = 0.6):
        """
        Blend NLP Content-Based similarity (alpha) with Item-Based CF score (1 - alpha).
        """
        profile_vec = self._build_user_profile_vec(user_id)
        if profile_vec is None:
            return self.recommend_popularity(user_id, top_n=top_n)

        cb_sims = cosine_similarity(profile_vec, self.tfidf_matrix).flatten()
        rated = self._get_rated_items(user_id)

        from scipy.sparse import csr_matrix
        profile_sparse = csr_matrix(profile_vec)

        candidates = []
        for idx in range(len(self.items_df)):
            iid = self.items_df.iloc[idx]["item_id"]
            if iid in rated:
                continue
            cb_score = float(cb_sims[idx])
            cf_pred = self.predict_cf_rating(user_id, iid)
            cf_norm = (cf_pred - 1.0) / 4.0  # Normalize [1, 5] -> [0, 1]
            hybrid_score = alpha * cb_score + (1.0 - alpha) * cf_norm
            candidates.append((idx, hybrid_score, cb_score, cf_pred))

        candidates.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, h_score, cb_s, cf_p in candidates[:top_n]:
            item_row = self.items_df.iloc[idx]
            keywords = self.nlp.get_matching_keywords(profile_sparse, self.tfidf_matrix[idx], top_k=4)
            results.append({
                "item_id": item_row["item_id"],
                "gig_title": item_row["gig_title"],
                "category": item_row["category"],
                "score": round(float(h_score), 4),
                "matched_keywords": keywords
            })
        return results