"""
Train XGBoost Full-Year Deep Model on all 12 Months of 2025 (2,085,047 Movements)
with all 135 features including Dynamic Runway Queue Sequencing (rwy_rolling_da_5).
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import time
import pickle
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ALL_12_MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]


def train_xgb_deep():
    print("=" * 80)
    print("🚀 TRAINING XGBOOST FULL-YEAR DEEP (2,085,000+ DEPARTURES, 135 FEATURES)")
    print("=" * 80)

    t0 = time.time()
    # 1. Load full year data
    print(f"[{time.strftime('%H:%M:%S')}] Loading all 12 months of 2025...")
    df = features.load_training_data(RAW_DIR, months=ALL_12_MONTHS)
    print(f"Loaded {len(df):,} total rows in {time.time() - t0:.1f}s")

    # 2. Compute Congestion, Instantaneous Surface Traffic & Runway Sequencing
    print(f"[{time.strftime('%H:%M:%S')}] Computing congestion and runway sequence features...")
    df = features.compute_congestion_metrics(df)
    df = features.compute_surface_accumulation_features(df)
    df = features.compute_apron_contention_features(df)
    df = features.compute_runway_sequence_features(df)

    # 3. Target encodings on departures
    dep_df = df.filter(pl.col("PHASE_mvt") == "DEP")
    print(f"[{time.strftime('%H:%M:%S')}] Computing target encodings across {len(dep_df):,} departures...")
    encs = features.compute_target_encodings(dep_df)

    # 4. Feature extraction
    print(f"[{time.strftime('%H:%M:%S')}] Extracting all 140+ features...")
    X_pl, _, y = features.extract_features(df, target_encodings=encs, attach_weather=True)

    X = X_pl.to_pandas()
    for c in X.select_dtypes(include=["object", "category"]).columns:
        X[c] = X[c].astype("category")

    cols = list(X.columns)
    cat_dtypes = {c: X[c].dtype for c in X.select_dtypes(include="category").columns}

    valid_mask = (y >= 60) & (y <= 150000)
    X_train = X.iloc[valid_mask].reset_index(drop=True)
    y_train = y[valid_mask]
    print(f"Training matrix ready: {len(X_train):,} departure rows, {len(cols)} features.")

    # 5. Train XGBoost Hist on GPU CUDA
    print(f"\n[{time.strftime('%H:%M:%S')}] Fitting XGBoost Deep on NVIDIA GPU CUDA (n_estimators=1300, max_depth=9, lr=0.035)...")
    t_xgb = time.time()
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1300,
        learning_rate=0.035,
        max_depth=9,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=3.0,
        enable_categorical=True,
        tree_method="hist",
        device="cuda",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = np.clip(model.predict(X_train), 60, 88500)
    train_rmse = np.sqrt(np.mean((y_train - preds) ** 2))
    print(f"\n✅ XGBoost Full-Year Deep fitted on GPU in {time.time() - t_xgb:.1f}s | Train RMSE: {train_rmse:.2f}s")

    # 6. Save Model and Metadata
    model.save_model(str(MODELS_DIR / "xgb_fullyear_deep.json"))
    with open(MODELS_DIR / "encs_xgb_fullyear_deep.pkl", "wb") as f:
        pickle.dump(encs, f)
    with open(MODELS_DIR / "cols_xgb_fullyear_deep.pkl", "wb") as f:
        pickle.dump(cols, f)
    with open(MODELS_DIR / "cat_dtypes_xgb_fullyear_deep.pkl", "wb") as f:
        pickle.dump(cat_dtypes, f)

    print(f"💾 Saved XGBoost Deep model to {MODELS_DIR / 'xgb_fullyear_deep.json'}")
    print(f"🎉 Total execution time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    train_xgb_deep()
