"""
Train LightGBM Full-Year Deep on 2.08M rows with all 175 Physical + Aerodynamic + METAR features.
Saves models/lgb_v56_deep.pkl and scratch/lgb_v56_test_preds.npy.
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
import lightgbm as lgb

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
    print("⚡ TRAINING LIGHTGBM FULL-YEAR DEEP (175 PHYSICAL + METAR FEATURES)")
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

    print(f"[{time.strftime('%H:%M:%S')}] Extracting all 175 features...")
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

    print(f"\n[{time.strftime('%H:%M:%S')}] Fitting LightGBM (1100 trees, num_leaves=127, lr=0.03)...")
    t_lgb = time.time()
    lgb_train_data = lgb.Dataset(X_train, label=y_train)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 35,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }

    model = lgb.train(params, lgb_train_data, num_boost_round=1100)

    train_preds = np.clip(model.predict(X_train), 60, 88500)
    train_rmse = np.sqrt(np.mean((y_train - train_preds) ** 2))
    print(f"✅ LightGBM v56 fitted in {time.time() - t_lgb:.1f}s | Train RMSE: {train_rmse:.2f}s")

    # Save Model
    with open(MODELS_DIR / "lgb_v56_deep.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / "encs_lgb_v56.pkl", "wb") as f:
        pickle.dump(encs, f)
    with open(MODELS_DIR / "cols_lgb_v56.pkl", "wb") as f:
        pickle.dump(cols, f)
    with open(MODELS_DIR / "cat_dtypes_lgb_v56.pkl", "wb") as f:
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
    out_npy = SCRATCH_DIR / "lgb_v56_test_preds.npy"
    np.save(out_npy, test_preds)
    print(f"💾 Saved v56 LightGBM test predictions: {out_npy}")
    print(f"   Stats: mean={test_preds.mean():.1f}s, min={test_preds.min():.1f}s, max={test_preds.max():.1f}s")
    print(f"🎉 Total execution time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
