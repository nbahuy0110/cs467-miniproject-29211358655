import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.recommender import GigRecommender

os.makedirs("results", exist_ok=True)


def train_test_split_ratings(ratings_df: pd.DataFrame, test_per_user: int = 1, seed: int = 42):
    rng = np.random.default_rng(seed)
    test_indices = []
    for uid, group in ratings_df.groupby("user_id"):
        high_rated = group[group["rating"] >= 4]
        pool = high_rated if len(high_rated) >= test_per_user else group
        chosen = rng.choice(pool.index, size=min(test_per_user, len(pool)), replace=False)
        test_indices.extend(chosen)
    test_df = ratings_df.loc[test_indices].copy()
    train_df = ratings_df.drop(index=test_indices).copy()
    return train_df, test_df


def evaluate_models(items_df: pd.DataFrame, ratings_df: pd.DataFrame):
    train_df, test_df = train_test_split_ratings(ratings_df, test_per_user=2, seed=42)
    rec = GigRecommender(items_df, train_df)

    methods = {
        "Popularity Baseline": lambda u: rec.recommend_popularity(u, top_n=5),
        "NLP Content-Based": lambda u: rec.recommend_content_for_user(u, top_n=5),
        "Hybrid (NLP + Item-CF)": lambda u: rec.recommend_hybrid(u, top_n=5, alpha=0.6),
    }

    precision_scores = {}
    for name, fn in methods.items():
        precisions = []
        for uid, group in test_df.groupby("user_id"):
            relevant = set(group[group["rating"] >= 4]["item_id"])
            if not relevant:
                continue
            preds = [r["item_id"] for r in fn(uid)]
            hits = len(set(preds) & relevant)
            precisions.append(hits / 5.0)
        precision_scores[name] = round(float(np.mean(precisions)), 4)

    # RMSE & MAE for Item-CF rating predictions
    y_true = test_df["rating"].values
    y_pred = [rec.predict_cf_rating(row["user_id"], row["item_id"]) for _, row in test_df.iterrows()]
    rmse = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)
    mae = round(float(mean_absolute_error(y_true, y_pred)), 4)

    return precision_scores, rmse, mae


def generate_charts(items_df: pd.DataFrame, ratings_df: pd.DataFrame, precision_scores: dict):
    # Chart 1: Category Distribution
    plt.figure(figsize=(8, 4.5))
    cat_counts = items_df["category"].value_counts()
    cat_counts.plot(kind="bar", color="#4C72B0", edgecolor="black")
    plt.title("Distribution of Freelance Gigs by Category")
    plt.xlabel("Category")
    plt.ylabel("Number of Gigs")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("results/category_distribution.png", dpi=150)
    plt.close()

    # Chart 2: Evaluation Comparison (Precision@5)
    plt.figure(figsize=(7, 4.5))
    names = list(precision_scores.keys())
    vals = list(precision_scores.values())
    bars = plt.bar(names, vals, color=["#DD8452", "#55A868", "#4C72B0"], edgecolor="black")
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    plt.title("Offline Evaluation: Precision@5 Comparison")
    plt.ylabel("Precision@5")
    plt.ylim(0, max(vals) + 0.1)
    plt.tight_layout()
    plt.savefig("results/evaluation_comparison.png", dpi=150)
    plt.close()


def run_required_test_cases(items_df: pd.DataFrame, ratings_df: pd.DataFrame):
    rec = GigRecommender(items_df, ratings_df)
    lines = []

    # Test Case 1: Profile match
    bio = "React frontend, responsive UI, REST API integration"
    tc1 = rec.recommend_by_query(bio, top_n=5)
    lines.append("=== TEST CASE 1: Profile Match (Free-Text Bio) ===")
    lines.append(f"Input Bio: '{bio}'")
    for i, r in enumerate(tc1, 1):
        lines.append(f"  {i}. [{r['item_id']}] {r['gig_title']} ({r['category']}) | Score: {r['score']} | Keywords: {r['matched_keywords']}")

    # Test Case 2: History test (U01 rated Mobile UI Design 5/5)
    tc2 = rec.recommend_hybrid("U01", top_n=5)
    lines.append("\n=== TEST CASE 2: History Test (User U01 - Rated Mobile UI Design 5/5) ===")
    for i, r in enumerate(tc2, 1):
        lines.append(f"  {i}. [{r['item_id']}] {r['gig_title']} ({r['category']}) | Score: {r['score']} | Keywords: {r['matched_keywords']}")

    # Test Case 3: Cold-start test (Add 1 new gig with 0 ratings)
    new_gig = pd.DataFrame([{
        "item_id": "G41",
        "gig_title": "React SaaS Analytics Frontend",
        "category": "Web Development",
        "budget_tier": "High",
        "project_brief": "Build a responsive React frontend with interactive charts and REST API integration.",
        "required_skills": "react, frontend, responsive ui, rest api, javascript"
    }])
    items_cold = pd.concat([items_df, new_gig], ignore_index=True)
    rec_cold = GigRecommender(items_cold, ratings_df)
    tc3 = rec_cold.recommend_by_query(bio, top_n=5)
    lines.append("\n=== TEST CASE 3: Cold-Start Test (New Gig G41 with 0 Ratings) ===")
    lines.append(f"Query: '{bio}'")
    for i, r in enumerate(tc3, 1):
        tag = " <-- NEW COLD-START GIG" if r["item_id"] == "G41" else ""
        lines.append(f"  {i}. [{r['item_id']}] {r['gig_title']} | Score: {r['score']} | Keywords: {r['matched_keywords']}{tag}")

    return "\n".join(lines)


if __name__ == "__main__":
    items_df = pd.read_csv("data/items.csv")
    ratings_df = pd.read_csv("data/ratings.csv")

    precision_scores, rmse, mae = evaluate_models(items_df, ratings_df)
    generate_charts(items_df, ratings_df, precision_scores)
    tc_output = run_required_test_cases(items_df, ratings_df)

    report = [
        "=== OFFLINE EVALUATION RESULTS ===",
        f"Precision@5: {precision_scores}",
        f"Item-CF Rating Prediction -> RMSE: {rmse}, MAE: {mae}\n",
        tc_output
    ]
    full_text = "\n".join(report)
    print(full_text)

    with open("results/sample_outputs.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    print("\nSaved results/sample_outputs.txt and charts in results/.")