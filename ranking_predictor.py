"""
models/ranking_predictor.py
Predicts page avg_position using XGBoost regression on page-level features.
Outputs: predicted rank, feature importances, SHAP-style top drivers.
"""

import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from config import MODELS_DIR, RANDOM_STATE, TEST_SIZE


FEATURE_COLS = [
    "word_count", "backlinks", "page_speed_score",
    "internal_links", "h1_count", "meta_desc_length",
    "schema_markup", "https", "mobile_friendly",
    "bounce_rate", "avg_session_dur_s", "pages_per_session",
    "keyword_count",
]


def train_ranking_model(merged_df: pd.DataFrame) -> dict:
    """
    Train XGBoost regression model to predict avg_position.

    Parameters
    ----------
    merged_df : page-level merged DataFrame from ETL pipeline

    Returns
    -------
    dict: model, metrics, feature importances, predictions
    """
    print("[RANKING MODEL] Preparing features …")

    df = merged_df.copy()
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available_features)
    if missing:
        print(f"[RANKING MODEL] Warning: missing features {missing}. Proceeding with available.")

    df.dropna(subset=["avg_position"], inplace=True)
    X = df[available_features]
    y = df["avg_position"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    # ── Evaluation ─────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    cv_mae = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error").mean()

    metrics = {"MAE": round(mae, 3), "RMSE": round(rmse, 3),
               "R2": round(r2, 3), "CV_MAE": round(cv_mae, 3)}
    print(f"[RANKING MODEL] Metrics → MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")

    # ── Feature importances ────────────────────────────────────────────────────
    importances = pd.DataFrame({
        "feature":    available_features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # ── Predict on full dataset ────────────────────────────────────────────────
    df["predicted_position"] = model.predict(X).round(1)
    df["position_gap"]       = (df["avg_position"] - df["predicted_position"]).round(1)
    df["underperforming"]    = df["position_gap"] > 5   # actual >> predicted → fixable

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({"model": model, "features": available_features},
                f"{MODELS_DIR}/ranking_predictor.pkl")
    print(f"[RANKING MODEL] ✅ Model saved. {df['underperforming'].sum()} underperforming pages detected.")

    return {
        "model":            model,
        "metrics":          metrics,
        "importances":      importances,
        "predictions_df":   df[["page", "avg_position", "predicted_position",
                                  "position_gap", "underperforming"]],
        "features_used":    available_features,
    }


def predict_new_page(page_features: dict, model_path: str = None) -> float:
    """
    Predict ranking for a new page given its features.

    Parameters
    ----------
    page_features : dict of feature_name → value
    model_path    : path to saved model (default MODELS_DIR)

    Returns
    -------
    Predicted avg_position (float)
    """
    path = model_path or f"{MODELS_DIR}/ranking_predictor.pkl"
    saved = joblib.load(path)
    model, features = saved["model"], saved["features"]

    input_df = pd.DataFrame([{f: page_features.get(f, 0) for f in features}])
    pred = model.predict(input_df)[0]
    return round(float(pred), 1)