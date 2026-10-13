"""
Train Annual Foundation Models (2.08M movements across all 12 months)
with Open-Meteo Weather and Runway Queue Sequencing dynamics.
Blends seamlessly with seasonal specialists for maximum generalization.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import time
import pickle
import numpy as np
import polars as pl
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ALL_12_MONTHS = [
    "2025-01", "2025-02", "2025-03", "2025-04",
    "2025-05", "2025-06", "2025-07", "2025-08",
    "2025-09", "2025-10", "2025-11", "2025-12"
]


def train_full_year_models():
    print("=" * 75)
    print("🌍 TRAINING ANNUAL FOUNDATION MODELS (2,085,000+ DEPARTURES)")
    print("=" * 75)

    t0 = time.time()
    # 1. Load all 12 months
    df = features.load_training_data(RAW_DIR, months=ALL_12_MONTHS)
    print(f"Loaded {len(df):,} total rows across all 12 months in {time.time() - t0:.1f}s")

    # 2. Compute congestion & runway sequencing
    df = features.compute_congestion_metrics(df)
    df = features.compute_runway_sequence_features(df)

    # 3. Target encodings
    dep_df = df.filter(pl.col("PHASE_mvt") == "DEP")
    encs = features.compute_target_encodings(dep_df)

    # 4. Feature extraction
    print("Extracting features (96 features including Weather, Runway Aerodynamics & ICAO Sequencing)...")
    X_pl, _, y = features.extract_features(df, target_encodings=encs, attach_weather=True)

    X = X_pl.to_pandas()
    for c in X.select_dtypes(include="category").columns:
        X[c] = X[c].astype("category")

    cols = list(X.columns)
    cat_dtypes = {c: X[c].dtype for c in X.select_dtypes(include="category").columns}

    valid_mask = (y >= 0) & (y <= 150000)
    X_train = X.iloc[valid_mask].reset_index(drop=True)
    y_train = y[valid_mask]
    print(f"Training set ready: {len(X_train):,} departure rows, {len(cols)} features.")

    # Save encodings and cols
    with open(MODELS_DIR / "encs_fullyear.pkl", "wb") as f:
        pickle.dump(encs, f)
    with open(MODELS_DIR / "cols_fullyear.pkl", "wb") as f:
        pickle.dump(cols, f)
    with open(MODELS_DIR / "cat_dtypes_fullyear.pkl", "wb") as f:
        pickle.dump(cat_dtypes, f)

    # --- Train LightGBM Foundation ---
    print("\n[1/2] Training LightGBM Full-Year Foundation Model...")
    t_lgb = time.time()
    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=850,
        learning_rate=0.035,
        num_leaves=110,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.80,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train)
    p_lgb = np.clip(lgb_model.predict(X_train), 60, 86400)
    rmse_lgb = np.sqrt(np.mean((y_train - p_lgb)**2))
    print(f"✅ LightGBM Full-Year fitted in {time.time() - t_lgb:.1f}s | Train RMSE: {rmse_lgb:.2f}s")

    with open(MODELS_DIR / "lgb_fullyear.pkl", "wb") as f:
        pickle.dump(lgb_model, f)

    # --- Train XGBoost Foundation ---
    print("\n[2/2] Training XGBoost Full-Year Foundation Model...")
    t_xgb = time.time()
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=800,
        learning_rate=0.04,
        max_depth=9,
        subsample=0.85,
        colsample_bytree=0.80,
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    p_xgb = np.clip(xgb_model.predict(X_train), 60, 86400)
    rmse_xgb = np.sqrt(np.mean((y_train - p_xgb)**2))
    print(f"✅ XGBoost Full-Year fitted in {time.time() - t_xgb:.1f}s | Train RMSE: {rmse_xgb:.2f}s")

    xgb_model.save_model(str(MODELS_DIR / "xgb_fullyear.json"))
    print(f"\n🎉 Annual Foundation Models saved successfully in {time.time() - t0:.1f}s total.")


if __name__ == "__main__":
    train_full_year_models()
