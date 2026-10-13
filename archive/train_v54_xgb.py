"""
Train Next-Generation XGBoost Full-Year Deep with:
1. METAR weather (visibility, severe LVP, snow, QNH, dewpoint depression)
2. Aerodynamic Runway Slope & Threshold Elevation
3. Aircraft Wing Area, Fuel Capacity, and Approach Speed
On NVIDIA RTX 4060 GPU CUDA.
Saves scratch/xgb_v54_test_preds.npy.
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
SCRATCH_DIR = PROJECT_ROOT / "scratch"

ALL_12_MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]


def main():
    print("=" * 80)
    print("🚀 TRAINING NEXT-GEN XGBOOST FULL-YEAR DEEP (WING AREA + RWY SLOPE + METAR)")
    print("=" * 80)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading all 12 months of 2025...")
    df = features.load_training_data(RAW_DIR, months=ALL_12_MONTHS)
    print(f"Loaded {len(df):,} total rows in {time.time() - t0:.1f}s")

    print(f"[{time.strftime('%H:%M:%S')}] Computing domain features...")
    df = features.compute_congestion_metrics(df)
    df = features.compute_surface_accumulation_features(df)
    df = features.compute_apron_contention_features(df)
    df = features.compute_runway_sequence_features(df)

    dep_df = df.filter(pl.col("PHASE_mvt") == "DEP")
    print(f"[{time.strftime('%H:%M:%S')}] Computing target encodings across {len(dep_df):,} departures...")
    encs = features.compute_target_encodings(dep_df)

    print(f"[{time.strftime('%H:%M:%S')}] Extracting all 170+ features...")
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

    print(f"\n[{time.strftime('%H:%M:%S')}] Fitting XGBoost on GPU CUDA (1500 estimators, max_depth=9, lr=0.035)...")
    t_xgb = time.time()
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1500,
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

    train_preds = np.clip(model.predict(X_train), 60, 88500)
    train_rmse = np.sqrt(np.mean((y_train - train_preds) ** 2))
    print(f"\n✅ XGBoost v54 fitted on GPU in {time.time() - t_xgb:.1f}s | Train RMSE: {train_rmse:.2f}s")

    # Save Model
    model.save_model(str(MODELS_DIR / "xgb_v54_deep.json"))
    with open(MODELS_DIR / "encs_xgb_v54.pkl", "wb") as f:
        pickle.dump(encs, f)
    with open(MODELS_DIR / "cols_xgb_v54.pkl", "wb") as f:
        pickle.dump(cols, f)
    with open(MODELS_DIR / "cat_dtypes_xgb_v54.pkl", "wb") as f:
        pickle.dump(cat_dtypes, f)

    # Predict test set
    print(f"\n[{time.strftime('%H:%M:%S')}] Generating test predictions on ranking.parquet...")
    rank_df = pl.read_parquet(RAW_DIR / "ranking.parquet")
    rank_df = features.compute_congestion_metrics(rank_df)
    rank_df = features.compute_surface_accumulation_features(rank_df)
    rank_df = features.compute_apron_contention_features(rank_df)
    rank_df = features.compute_runway_sequence_features(rank_df)

    dep_rank = rank_df.filter(pl.col("PHASE_mvt") == "DEP")
    X_test_pl, _, _ = features.extract_features(dep_rank, target_encodings=encs, attach_weather=True)
    X_test = X_test_pl.to_pandas()

    for c in cols:
        if c in cat_dtypes:
            X_test[c] = pd.Categorical(X_test[c].astype(str), categories=cat_dtypes[c].categories)
        elif c in X_test.columns and str(X_test[c].dtype) in ["object", "category"]:
            X_test[c] = X_test[c].astype("category")
        elif c not in X_test.columns:
            X_test[c] = 0.0

    test_preds = np.clip(model.predict(X_test[cols]), 60, 88500)
    out_npy = SCRATCH_DIR / "xgb_v54_test_preds.npy"
    np.save(out_npy, test_preds)
    print(f"💾 Saved v54 XGBoost test predictions: {out_npy}")
    print(f"   Stats: mean={test_preds.mean():.1f}s, min={test_preds.min():.1f}s, max={test_preds.max():.1f}s")
    print(f"🎉 Total execution time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
