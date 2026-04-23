
import pandas as pd
import numpy as np
import os

np.random.seed(42)
os.makedirs("data/raw", exist_ok=True)

# ── Keyword pool ─────────────────────────────────────────────────────────────
keywords = [
    "machine learning tutorial", "python data analysis", "best seo tools 2024",
    "how to improve page speed", "xgboost classification example",
    "nlp text classification", "keyword research guide", "google search console setup",
    "data science projects for beginners", "buy seo software", "seo audit checklist",
    "page speed optimization", "backlink building strategies", "content gap analysis",
    "python pandas tutorial", "best data visualization tools", "machine learning pipeline",
    "seo competitor analysis", "technical seo guide", "google analytics 4 tutorial",
    "purchase seo tool", "hire data scientist", "machine learning for seo",
    "serp features guide", "core web vitals optimization", "buy backlinks online",
    "data analysis with python", "sklearn tutorial beginners", "seo content strategy 2024",
    "python machine learning projects", "website crawl analysis", "log file analysis seo",
    "ecommerce seo guide", "local seo optimization", "voice search optimization",
]

urls = [
    "/blog/machine-learning-tutorial/",
    "/blog/python-data-analysis/",
    "/blog/seo-tools-guide/",
    "/blog/page-speed-optimization/",
    "/blog/xgboost-guide/",
    "/blog/nlp-classification/",
    "/blog/keyword-research/",
    "/blog/google-search-console/",
    "/blog/data-science-projects/",
    "/services/seo-software/",
    "/blog/seo-audit-checklist/",
    "/blog/backlink-strategies/",
    "/blog/content-gap-analysis/",
    "/blog/pandas-tutorial/",
    "/blog/data-visualization/",
]

n = 1500
records = []
for _ in range(n):
    kw  = np.random.choice(keywords)
    url = np.random.choice(urls)
    pos = round(np.random.lognormal(mean=2.2, sigma=0.7), 1)
    pos = np.clip(pos, 1.0, 100.0)
    ctr = max(0, 0.35 * np.exp(-0.07 * pos) + np.random.normal(0, 0.02))
    imp = int(np.random.lognormal(mean=5.5, sigma=1.2))
    clicks = int(imp * ctr)
    records.append({
        "query":        kw,
        "page":         url,
        "impressions":  imp,
        "clicks":       clicks,
        "ctr":          round(ctr, 4),
        "avg_position": round(pos, 1),
        "date":         pd.Timestamp("2024-01-01") + pd.Timedelta(days=np.random.randint(0, 365)),
    })

gsc_df = pd.DataFrame(records)

# ── GA-style behavioural data ─────────────────────────────────────────────────
ga_records = []
for url in urls:
    sessions = np.random.randint(300, 8000)
    bounce   = round(np.random.uniform(0.30, 0.80), 2)
    duration = round(np.random.uniform(60, 360), 1)
    pages_ps = round(np.random.uniform(1.2, 4.5), 2)
    ga_records.append({
        "page":              url,
        "sessions":          sessions,
        "bounce_rate":       bounce,
        "avg_session_dur_s": duration,
        "pages_per_session": pages_ps,
    })

ga_df = pd.DataFrame(ga_records)

# ── Page metadata (features for ranking model) ────────────────────────────────
meta_records = []
for url in urls:
    meta_records.append({
        "page":            url,
        "word_count":      np.random.randint(400, 4000),
        "backlinks":       int(np.random.lognormal(mean=3, sigma=1.5)),
        "page_speed_score": np.random.randint(30, 100),
        "internal_links":  np.random.randint(1, 25),
        "h1_count":        np.random.randint(1, 3),
        "meta_desc_length": np.random.randint(50, 160),
        "schema_markup":   np.random.choice([0, 1], p=[0.4, 0.6]),
        "https":           1,
        "mobile_friendly": np.random.choice([0, 1], p=[0.1, 0.9]),
    })

meta_df = pd.DataFrame(meta_records)

gsc_df.to_csv("data/raw/gsc_data.csv", index=False)
ga_df.to_csv("data/raw/ga_data.csv",   index=False)
meta_df.to_csv("data/raw/page_meta.csv", index=False)

print(f"✅ GSC data:  {len(gsc_df)} rows  → data/raw/gsc_data.csv")
print(f"✅ GA data:   {len(ga_df)} rows   → data/raw/ga_data.csv")
print(f"✅ Meta data: {len(meta_df)} rows  → data/raw/page_meta.csv")