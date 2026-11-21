import pandas as pd
from src.recommender import GigRecommender


def print_recs(title: str, recs: list):
    print(f"\n--- {title} ---")
    for i, r in enumerate(recs, 1):
        print(f"{i}. [{r['item_id']}] {r['gig_title']} ({r['category']}) | Score: {r['score']} | Keywords: {r['matched_keywords']}")


def main():
    items_df = pd.read_csv("data/items.csv")
    ratings_df = pd.read_csv("data/ratings.csv")
    rec = GigRecommender(items_df, ratings_df)

    print("=== Freelance Project & Client Gig Recommender (MP-C05) ===")
    while True:
        print("\nOptions:\n1. Recommend by Freelancer Bio / Free-Text Query\n2. Recommend for Existing Freelancer (User ID)\n3. Exit")
        choice = input("Select (1/2/3): ").strip()
        if choice == "1":
            query = input("Enter freelancer bio or skills: ").strip()
            print_recs("Top-5 NLP Content-Based Matches", rec.recommend_by_query(query, top_n=5))
        elif choice == "2":
            uid = input("Enter User ID (e.g., U01 - U14): ").strip().upper()
            print_recs(f"Popularity Baseline for {uid}", rec.recommend_popularity(uid, top_n=5))
            print_recs(f"NLP Content-Based for {uid}", rec.recommend_content_for_user(uid, top_n=5))
            print_recs(f"Hybrid (NLP + Item-CF) for {uid}", rec.recommend_hybrid(uid, top_n=5))
        elif choice == "3":
            break
        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()