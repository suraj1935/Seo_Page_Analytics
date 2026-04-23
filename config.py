

import os
from dotenv import load_dotenv

load_dotenv()

# ── Google API ──────────────────────────────────────────────────────────────
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
GSC_SITE_URL            = os.getenv("GSC_SITE_URL", "https://example.com/")
GA_PROPERTY_ID          = os.getenv("GA_PROPERTY_ID", "properties/XXXXXXXXX")

# ── Date range ──────────────────────────────────────────────────────────────
DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE   = "2024-12-31"

# ── ETL ─────────────────────────────────────────────────────────────────────
RAW_DATA_DIR       = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR         = "models/saved"

# ── Keyword Clustering ───────────────────────────────────────────────────────
N_CLUSTERS         = 8          # Number of semantic keyword clusters
MIN_IMPRESSIONS    = 10         # Drop keywords below this threshold

# ── Ranking Predictor ────────────────────────────────────────────────────────
RANKING_TARGET_COL = "avg_position"
TEST_SIZE          = 0.2
RANDOM_STATE       = 42

# ── Cannibalization ──────────────────────────────────────────────────────────
COSINE_SIM_THRESHOLD = 0.75     # Pages above this are flagged as cannibalizing

# ── Dashboard ────────────────────────────────────────────────────────────────
DASHBOARD_TITLE = "SEO Performance Analytics Engine"