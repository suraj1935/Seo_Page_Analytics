

import os
import pandas as pd
from etl.ingest import load_from_csv
from etl.clean  import clean_gsc, clean_ga, clean_meta, merge_all
from config     import RAW_DATA_DIR, PROCESSED_DATA_DIR


def run_pipeline(mode: str = "csv") -> dict[str, pd.DataFrame]:
    """
    Run the full ETL pipeline.

    Parameters
    ----------
    mode : "csv"  → load from local CSVs (default/demo)
           "api"  → pull live from Google APIs (requires credentials)

    Returns
    -------
    dict with keys: gsc, ga, meta, merged
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # ── 1. Ingest ─────────────────────────────────────────────────────────────
    if mode == "csv":
        raw = load_from_csv(
            gsc_path  = f"{RAW_DATA_DIR}/gsc_data.csv",
            ga_path   = f"{RAW_DATA_DIR}/ga_data.csv",
            meta_path = f"{RAW_DATA_DIR}/page_meta.csv",
        )
    else:
        raise NotImplementedError("API mode: set credentials and use etl/ingest.py directly.")

    # ── 2. Clean ──────────────────────────────────────────────────────────────
    gsc_clean  = clean_gsc(raw["gsc"])
    ga_clean   = clean_ga(raw["ga"])
    meta_clean = clean_meta(raw["meta"])

    # ── 3. Merge ──────────────────────────────────────────────────────────────
    merged = merge_all(gsc_clean, ga_clean, meta_clean)

    # ── 4. Save processed data ────────────────────────────────────────────────
    gsc_clean.to_csv(f"{PROCESSED_DATA_DIR}/gsc_clean.csv",   index=False)
    ga_clean.to_csv(f"{PROCESSED_DATA_DIR}/ga_clean.csv",     index=False)
    meta_clean.to_csv(f"{PROCESSED_DATA_DIR}/meta_clean.csv", index=False)
    merged.to_csv(f"{PROCESSED_DATA_DIR}/merged.csv",         index=False)

    print(f"\n[PIPELINE] ✅ ETL complete. Processed files saved to '{PROCESSED_DATA_DIR}/'")

    return {"gsc": gsc_clean, "ga": ga_clean, "meta": meta_clean, "merged": merged}