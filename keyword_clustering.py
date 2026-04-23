"""
models/keyword_clustering.py
Groups keywords by semantic intent using TF-IDF + KMeans.
Intent labels: Informational, Transactional, Navigational, Commercial.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from config import N_CLUSTERS, MODELS_DIR


# Words that signal each intent type
INTENT_SIGNALS = {
    "Transactional":  ["buy", "purchase", "order", "price", "cheap", "discount",
                       "deal", "hire", "get", "download", "subscribe"],
    "Navigational":   ["login", "sign in", "official", "website", "homepage",
                       "account", "portal", "dashboard"],
    "Commercial":     ["best", "top", "review", "compare", "vs", "alternative",
                       "recommended", "tool", "software", "service", "tool 2024"],
    "Informational":  ["how", "what", "why", "guide", "tutorial", "learn",
                       "example", "explain", "definition", "beginner"],
}


def assign_intent(keyword: str) -> str:
    """Rule-based intent classification using signal words."""
    kw = keyword.lower()
    for intent, signals in INTENT_SIGNALS.items():
        if any(sig in kw for sig in signals):
            return intent
    return "Informational"  # default


def cluster_keywords(gsc_df: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> pd.DataFrame:
    """
    Cluster keywords semantically using TF-IDF + KMeans.

    Parameters
    ----------
    gsc_df     : cleaned GSC DataFrame with 'query' column
    n_clusters : number of clusters

    Returns
    -------
    DataFrame with keyword, cluster_id, cluster_label, intent, metrics
    """
    print(f"[CLUSTERING] Running KMeans with {n_clusters} clusters …")

    # Aggregate to unique keywords with metrics
    kw_df = gsc_df.groupby("query").agg(
        impressions       = ("impressions", "sum"),
        clicks            = ("clicks", "sum"),
        avg_position      = ("avg_position", "mean"),
        opportunity_score = ("opportunity_score", "mean"),
    ).reset_index()

    keywords = kw_df["query"].tolist()

    # TF-IDF vectorization (character n-grams capture morphology)
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=3000,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(keywords)
    X_norm = normalize(X)

    # Find optimal k using silhouette if n_clusters not fixed
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=15, max_iter=300)
    labels = km.fit_predict(X_norm)

    sil = silhouette_score(X_norm, labels, sample_size=min(1000, len(keywords)))
    print(f"[CLUSTERING] Silhouette Score: {sil:.4f}")

    kw_df["cluster_id"]    = labels
    kw_df["intent"]        = kw_df["query"].apply(assign_intent)
    kw_df["cluster_label"] = kw_df["cluster_id"].apply(
        lambda cid: _get_cluster_label(kw_df[kw_df["cluster_id"] == cid]["query"].tolist())
    )

    # Identify content gaps: high impression clusters with low avg clicks
    kw_df["content_gap_flag"] = (
        (kw_df["impressions"] > kw_df["impressions"].median()) &
        (kw_df["clicks"] < kw_df["clicks"].quantile(0.25))
    )

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({"km": km, "vectorizer": vectorizer}, f"{MODELS_DIR}/keyword_clustering.pkl")
    print(f"[CLUSTERING] ✅ Model saved. {kw_df['content_gap_flag'].sum()} content gap keywords flagged.")

    return kw_df


def _get_cluster_label(keywords: list[str]) -> str:
    """Extract the most representative bigram/unigram for a cluster."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5)
    if len(keywords) < 2:
        return keywords[0] if keywords else "misc"
    try:
        X = vectorizer.fit_transform(keywords)
        top_term = vectorizer.get_feature_names_out()[X.sum(axis=0).argmax()]
        return top_term.title()
    except Exception:
        return "Mixed"


def get_cluster_summary(clustered_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cluster-level statistics for dashboard."""
    return clustered_df.groupby(["cluster_id", "cluster_label", "intent"]).agg(
        keyword_count     = ("query", "count"),
        total_impressions = ("impressions", "sum"),
        total_clicks      = ("clicks", "sum"),
        avg_position      = ("avg_position", "mean"),
        avg_opportunity   = ("opportunity_score", "mean"),
        content_gaps      = ("content_gap_flag", "sum"),
    ).reset_index().sort_values("total_impressions", ascending=False)