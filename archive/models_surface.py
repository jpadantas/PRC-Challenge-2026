"""
Dedicated Surface Layout and Congestion Model for Missing-AOBT departures.

When AOBT_3_flt is absent, flight plan timestamps cannot be trusted for taxi duration.
This model predicts taxi-out time purely from physical airport surface layout,
runway configurations, aircraft wake category, and real-time surface traffic congestion.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import pickle
import time
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb
import xgboost as xgb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

SURFACE_FEATURE_COLS = [
    # Spatial & Airport Geometry
    "adep", "runway", "stand", "stand_pier", "apt_rwy", "apt_pier_rwy",
    "aircraft_type", "wk_tbl", "flight_rule",
    # Target encodings
    "stat_med_apt", "stat_mean_apt", "stat_med_apt_rwy", "stat_mean_apt_rwy",
    "stat_med_apt_stand", "stat_mean_apt_stand", "stat_med_apt_pier_rwy", "stat_mean_apt_pier_rwy",
    "stat_med_apt_hour", "stat_med_apt_month", "stat_med_actype", "stat_med_wktbl",
    # Temporal
    "mvt_hour", "mvt_minute", "mvt_dow", "mvt_month", "is_weekend", "is_winter",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # Congestion & Traffic Pressure
    "cong_apt_dep_15m", "cong_apt_arr_15m", "cong_apt_tot_15m",
    "cong_apt_dep_30m", "cong_apt_arr_30m", "cong_apt_tot_30m",
    "cong_rwy_dep_15m", "cong_rwy_arr_15m", "cong_rwy_tot_15m",
    "cong_dep_ratio_15m",
    "log_cong_apt_dep_15m", "log_cong_apt_tot_15m",
    "log_cong_apt_dep_30m", "log_cong_apt_tot_30m",
    "log_cong_rwy_dep_15m", "log_cong_rwy_tot_15m",
]


def train_surface_models():
    """
    Train seasonal surface models using all departures with realistic operational taxi times (<= 5400s).
    Excludes non-physical database corruption artifacts.
    """
    print("=" * 70)
    print("🛫 TRAINING DEDICATED SURFACE REGRESSORS FOR MISSING AOBT")
    print("=" * 70)

    from src import features

    for season, months in [
        ("winter", ["2025-01", "2025-02", "2025-11", "2025-12"]),
        ("summer", ["2025-05", "2025-06", "2025-07", "2025-08"]),
    ]:
        print(f"\nTraining Surface Model for {season.upper()} ({months})...")
        t0 = time.time()
        df = features.load_training_data(RAW_DIR, months=months)
        df = features.compute_congestion_metrics(df)
        dep_df = df.filter(pl.col("PHASE_mvt") == "DEP")
        encs = features.compute_target_encodings(dep_df)

        X_pl, _, y = features.extract_features(df, target_encodings=encs)
        
        # Select available surface columns
        available_cols = [c for c in SURFACE_FEATURE_COLS if c in X_pl.columns]
        X = X_pl.select(available_cols).to_pandas()
        for c in X.select_dtypes(include="category").columns:
            X[c] = X[c].astype("category")

        # Clean training mask: realistic operational taxi times
        clean_mask = (y >= 60) & (y <= 5400)
        X_train = X.iloc[clean_mask].reset_index(drop=True)
        y_train = y[clean_mask]

        print(f"Dataset: {len(X_train):,} training rows, {len(available_cols)} surface features.")

        # Train LightGBM Surface Regressor
        model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.80,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        
        preds = np.clip(model.predict(X_train), 180, 5400)
        train_rmse = np.sqrt(np.mean((y_train - preds)**2))
        train_mae = np.mean(np.abs(y_train - preds))
        print(f"✅ Surface Model {season.upper()} fitted in {time.time() - t0:.1f}s | Train RMSE: {train_rmse:.2f}s | MAE: {train_mae:.2f}s")

        # Save artifacts
        with open(MODELS_DIR / f"surface_lgb_{season}.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(MODELS_DIR / f"surface_cols_{season}.pkl", "wb") as f:
            pickle.dump(available_cols, f)


def predict_surface_taxi(dep_df: pl.DataFrame, season: str, encs: dict) -> np.ndarray:
    """Predict taxi time for departures using the dedicated surface model."""
    from src import features
    with open(MODELS_DIR / f"surface_lgb_{season}.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / f"surface_cols_{season}.pkl", "rb") as f:
        cols = pickle.load(f)

    X_pl, _, _ = features.extract_features(dep_df, target_encodings=encs)
    X = X_pl.select(cols).to_pandas()
    for c in X.select_dtypes(include="category").columns:
        X[c] = X[c].astype("category")

    preds = model.predict(X)
    return np.clip(np.round(preds), 180, 4500)


if __name__ == "__main__":
    train_surface_models()
