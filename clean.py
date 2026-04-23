
import pandas as pd
import numpy as np
import re
from config import MIN_IMPRESSIONS


# ── GSC cleaning ─────────────────────────────────────────────────────────────

def clean_gsc(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate Google Search Console data."""
    print("[CLEAN] Cleaning GSC data …")
    df = df.copy()

    # Standardize column names
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Drop rows with null query or page
    df.dropna(subset=["query", "page"], inplace=True)

    # Normalize URL: strip domain, lowercase, ensure leading slash
    df["page"] = df["page"].apply(_normalize_url)

    # Lowercase queries, strip whitespace
    df["query"] = df["query"].str.lower().str.strip()

    # Remove queries with special characters only
    df = df[df["query"].str.match(r"[a-zA-Z0-9]")]

    # Filter low-impression keywords (noise)
    df = df[df["impressions"] >= MIN_IMPRESSIONS]

    # Clip position to valid range
    df["avg_position"] = df["avg_position"].clip(1.0, 100.0)

    # CTR sanity check
    df["ctr"] = df["ctr"].clip(0.0, 1.0)

    # Derived: opportunity score (high impressions, low position, low CTR)
    df["opportunity_score"] = _compute_opportunity_score(df)

    print(f"[CLEAN] GSC: {len(df)} rows after cleaning.")
    return df.reset_index(drop=True)


def clean_ga(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Google Analytics behavioural data."""
    print("[CLEAN] Cleaning GA data …")
    df = df.copy()
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    df["page"] = df["page"].apply(_normalize_url)
    df["bounce_rate"] = df["bounce_rate"].clip(0.0, 1.0)
    df.dropna(inplace=True)
    print(f"[CLEAN] GA: {len(df)} rows after cleaning.")
    return df.reset_index(drop=True)


def clean_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Clean page metadata."""
    print("[CLEAN] Cleaning page metadata …")
    df = df.copy()
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    df["page"] = df["page"].apply(_normalize_url)
    df["word_count"] = df["word_count"].clip(lower=0)
    df["backlinks"]  = df["backlinks"].clip(lower=0)
    df.fillna(0, inplace=True)
    print(f"[CLEAN] Meta: {len(df)} rows after cleaning.")
    return df.reset_index(drop=True)


# ── Merge ────────────────────────────────────────────────────────────────────

def merge_all(gsc: pd.DataFrame, ga: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate GSC to page-level, then left-join GA and meta features.
    Returns a unified page-level feature table.
    """
    print("[CLEAN] Merging datasets …")

    page_agg = gsc.groupby("page").agg(
        total_impressions=("impressions", "sum"),
        total_clicks=("clicks", "sum"),
        avg_ctr=("ctr", "mean"),
        avg_position=("avg_position", "mean"),
        keyword_count=("query", "nunique"),
        avg_opportunity_score=("opportunity_score", "mean"),
    ).reset_index()

    merged = page_agg.merge(ga, on="page", how="left")
    merged = merged.merge(meta, on="page", how="left")
    merged.fillna(0, inplace=True)

    print(f"[CLEAN] Merged dataset: {merged.shape}")
    return merged


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalize_url(url: str) -> str:
    url = str(url).strip().lower()
    # Remove protocol + domain if present
    url = re.sub(r"^https?://[^/]+", "", url)
    if not url.startswith("/"):
        url = "/" + url
    return url


def _compute_opportunity_score(df: pd.DataFrame) -> pd.Series:
    """
    Opportunity score = high impressions + low avg_position (close to top) + low CTR.
    Signals: "page is visible but not getting clicked — optimization opportunity."
    Normalized 0–100.
    """
    imp_norm = np.log1p(df["impressions"]) / np.log1p(df["impressions"].max())
    pos_inv  = 1 - (df["avg_position"].clip(1, 100) / 100)   # higher = closer to top
    ctr_gap  = 1 - df["ctr"].clip(0, 1)                      # higher = more room to grow
    score    = (imp_norm * 0.4 + pos_inv * 0.35 + ctr_gap * 0.25) * 100
    return score.round(2)