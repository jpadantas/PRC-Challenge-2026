"""
Train Full-Year LightGBM Deep v61 with all 182 physical, aerodynamic, METAR, and OSM taxiway network features.
Optimized for 16 CPU threads.
Saves models/lgb_v61_deep.pkl and scratch/lgb_v61_test_preds.npy.
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
    print("⚡ TRAINING LIGHTGBM FULL-YEAR DEEP V61 (182 PHYSICAL + OSM + METAR FEATURES)")
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

    print(f"[{time.strftime('%H:%M:%S')}] Extracting all 182 features...")
    X_pl, _, y = features.extract_features(df, target_encodings=encs, attach_weather=True)

    X = X_pl.to_pandas()
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].astype("category")

    valid_mask = (y >= 60) & (y <= 150000)
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"Training matrix ready: {len(X):,} departure rows, {X.shape[1]} features.")

    cat_cols_list = [c for c in X.select_dtypes(include=["category"]).columns]

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": 11,
        "min_child_samples": 50,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "n_jobs": 16,
        "random_state": 42,
        "verbose": -1,
    }

    n_trees = 1100
    print(f"\n[{time.strftime('%H:%M:%S')}] Fitting LightGBM ({n_trees} trees, num_leaves=127, lr=0.03)...")
    dtrain = lgb.Dataset(X, label=y, categorical_feature=cat_cols_list, free_raw_data=False)
    t_fit = time.time()
    model = lgb.train(params, dtrain, num_boost_round=n_trees)
    fit_time = time.time() - t_fit

    preds_train = model.predict(X)
    train_rmse = np.sqrt(np.mean((y - preds_train) ** 2))
    print(f"✅ LightGBM v61 fitted in {fit_time:.1f}s | Train RMSE: {train_rmse:.2f}s")

    model_path = MODELS_DIR / "lgb_v61_deep.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / "encs_lgb_v61.pkl", "wb") as f:
        pickle.dump(encs, f)
    print(f"💾 Model saved to {model_path}")

    # Generate Test Predictions on ranking.parquet
    print(f"\n[{time.strftime('%H:%M:%S')}] Generating test predictions on ranking.parquet...")
    rank_df = pl.read_parquet(RAW_DIR / "ranking.parquet")
    rank_df = features.compute_congestion_metrics(rank_df)
    rank_df = features.compute_surface_accumulation_features(rank_df)
    rank_df = features.compute_apron_contention_features(rank_df)
    rank_df = features.compute_runway_sequence_features(rank_df)

    dep_rank = rank_df.filter(pl.col("PHASE_mvt") == "DEP")
    X_test_pl, _, _ = features.extract_features(dep_rank, target_encodings=encs, attach_weather=True)
    X_test = X_test_pl.to_pandas()
    for c in X_test.select_dtypes(include=["object"]).columns:
        X_test[c] = X_test[c].astype("category")

    test_preds = model.predict(X_test)
    test_preds = np.clip(test_preds, 60, 88500)

    out_npy = SCRATCH_DIR / "lgb_v61_test_preds.npy"
    np.save(out_npy, test_preds)
    print(f"💾 Saved v61 LightGBM test predictions: {out_npy}")
    print(f"   Stats: mean={test_preds.mean():.1f}s, min={test_preds.min():.1f}s, max={test_preds.max():.1f}s")
    print(f"🎉 Total execution time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
