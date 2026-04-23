"""
etl/ingest.py
Handles data ingestion from Google Search Console API or local CSV fallback.
"""

import os
import pandas as pd
from datetime import datetime


def load_from_csv(gsc_path: str, ga_path: str, meta_path: str) -> dict[str, pd.DataFrame]:
    """Load raw data from local CSVs (demo / offline mode)."""
    print("[INGEST] Loading from local CSVs …")
    return {
        "gsc":  pd.read_csv(gsc_path, parse_dates=["date"]),
        "ga":   pd.read_csv(ga_path),
        "meta": pd.read_csv(meta_path),
    }


def load_from_gsc_api(site_url: str, start_date: str, end_date: str,
                       credentials_path: str) -> pd.DataFrame:
    """
    Pull data directly from Google Search Console API.
    Requires a valid OAuth2 credentials.json from Google Cloud Console.
    """
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]
        flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
        creds = flow.run_local_server(port=0)
        service = build("webmasters", "v3", credentials=creds)

        payload = {
            "startDate": start_date,
            "endDate": end_date,
            "dimensions": ["query", "page"],
            "rowLimit": 25000,
        }
        response = service.searchanalytics().query(siteUrl=site_url, body=payload).execute()
        rows = response.get("rows", [])

        records = [{
            "query":        r["keys"][0],
            "page":         r["keys"][1],
            "clicks":       r["clicks"],
            "impressions":  r["impressions"],
            "ctr":          r["ctr"],
            "avg_position": r["position"],
        } for r in rows]

        df = pd.DataFrame(records)
        print(f"[INGEST] Fetched {len(df)} rows from GSC API.")
        return df

    except Exception as e:
        raise RuntimeError(f"GSC API error: {e}. Use CSV fallback instead.")