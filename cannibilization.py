
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import COSINE_SIM_THRESHOLD


def detect_cannibalization(gsc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify keywords where multiple pages compete.
    Returns a DataFrame of cannibalization pairs with severity scores.

    Parameters
    ----------
    gsc_df : cleaned GSC DataFrame with 'query' and 'page' columns

    Returns
    -------
    DataFrame of conflicting page pairs per keyword
    """
    print("[CANNIBALIZATION] Analysing keyword-page overlaps …")

    # Find queries appearing on more than one page
    query_page_counts = gsc_df.groupby("query")["page"].nunique()
    multi_page_queries = query_page_counts[query_page_counts > 1].index

    if len(multi_page_queries) == 0:
        print("[CANNIBALIZATION] No cannibalization detected.")
        return pd.DataFrame()

    conflict_records = []
    subset = gsc_df[gsc_df["query"].isin(multi_page_queries)]

    for query, group in subset.groupby("query"):
        pages = group.groupby("page").agg(
            impressions  = ("impressions", "sum"),
            clicks       = ("clicks", "sum"),
            avg_position = ("avg_position", "mean"),
        ).reset_index()

        if len(pages) < 2:
            continue

        # Vectorize page URLs for similarity
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
        try:
            X = vectorizer.fit_transform(pages["page"])
            sim_matrix = cosine_similarity(X)
        except Exception:
            sim_matrix = np.ones((len(pages), len(pages)))

        for i in range(len(pages)):
            for j in range(i + 1, len(pages)):
                sim = sim_matrix[i][j]
                p1  = pages.iloc[i]
                p2  = pages.iloc[j]

                # Severity: how close in position (both fighting for same spot)
                pos_diff  = abs(p1["avg_position"] - p2["avg_position"])
                severity  = _compute_severity(p1, p2, pos_diff)

                conflict_records.append({
                    "query":          query,
                    "page_1":         p1["page"],
                    "page_2":         p2["page"],
                    "page_1_pos":     round(p1["avg_position"], 1),
                    "page_2_pos":     round(p2["avg_position"], 1),
                    "page_1_clicks":  p1["clicks"],
                    "page_2_clicks":  p2["clicks"],
                    "url_similarity": round(sim, 3),
                    "pos_difference": round(pos_diff, 1),
                    "severity":       severity,
                    "recommendation": _recommend_action(p1, p2),
                })

    result = pd.DataFrame(conflict_records)
    if not result.empty:
        result.sort_values("severity", ascending=False, inplace=True)
        result.reset_index(drop=True, inplace=True)
        print(f"[CANNIBALIZATION] ✅ {len(result)} cannibalization conflicts found.")

    return result


def _compute_severity(p1: pd.Series, p2: pd.Series, pos_diff: float) -> str:
    """Classify conflict severity based on impression volume and position proximity."""
    high_imp = (p1["impressions"] + p2["impressions"]) > 500
    close    = pos_diff < 5

    if high_imp and close:
        return "🔴 High"
    elif high_imp or close:
        return "🟡 Medium"
    else:
        return "🟢 Low"


def _recommend_action(p1: pd.Series, p2: pd.Series) -> str:
    """Generate a consolidation recommendation."""
    winner = p1 if p1["clicks"] >= p2["clicks"] else p2
    loser  = p2 if winner["page"] == p1["page"] else p1

    return (
        f"Consolidate into '{winner['page']}'. "
        f"Add canonical tag or 301 redirect from '{loser['page']}'."
    )


def get_cannibalization_summary(conflicts_df: pd.DataFrame) -> dict:
    """High-level summary stats for dashboard."""
    if conflicts_df.empty:
        return {"total_conflicts": 0, "high": 0, "medium": 0, "low": 0}

    counts = conflicts_df["severity"].value_counts()
    return {
        "total_conflicts": len(conflicts_df),
        "high":   counts.get("🔴 High", 0),
        "medium": counts.get("🟡 Medium", 0),
        "low":    counts.get("🟢 Low", 0),
        "affected_queries": conflicts_df["query"].nunique(),
        "affected_pages":   pd.concat([
            conflicts_df["page_1"], conflicts_df["page_2"]
        ]).nunique(),
    }