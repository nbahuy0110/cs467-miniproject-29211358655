import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

CUSTOM_STOP_WORDS = set(ENGLISH_STOP_WORDS).union({
    "project", "build", "create", "develop", "using", "need", "looking", "client"
})


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/numbers, collapse whitespace, remove stop words."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s+#]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in CUSTOM_STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


class NLPPipeline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
        self.tfidf_matrix = None
        self.feature_names = None

    def fit_transform_items(self, items_df):
        """Combine title, category, brief, and skills, clean text, and compute TF-IDF."""
        combined = (
            items_df["gig_title"].fillna("") + " " +
            items_df["category"].fillna("") + " " +
            items_df["project_brief"].fillna("") + " " +
            items_df["required_skills"].fillna("")
        )
        cleaned_docs = combined.apply(clean_text)
        self.tfidf_matrix = self.vectorizer.fit_transform(cleaned_docs)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        return self.tfidf_matrix

    def transform_query(self, query_text: str):
        """Transform any new free-text query or bio using the fitted vocabulary."""
        cleaned = clean_text(query_text)
        return self.vectorizer.transform([cleaned])

    def get_matching_keywords(self, query_vec, item_vec, top_k: int = 5):
        """Highlight top matching TF-IDF keywords/skills between query/profile and gig."""
        overlap = query_vec.multiply(item_vec).toarray().flatten()
        top_idx = np.argsort(overlap)[::-1]
        keywords = [
            self.feature_names[i]
            for i in top_idx[:top_k]
            if overlap[i] > 0
        ]
        return keywords