"""
Train Full-Year LightGBM Deep Offset Regressor (182 Features) on full 2025 dataset.
Predicts delta_AOBT = BLOCK_TIME_UTC_mvt - AOBT_3_flt for physical taxi reconstruction.
Saves models/lgb_offset_deep.pkl and scratch/lgb_offset_test_preds.npy.
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
    print("⚡ TRAINING LIGHTGBM FULL-YEAR OFFSET REGRESSOR (182 FEATURES)")
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
    X_pl, _, _ = features.extract_features(df, target_encodings=encs, attach_weather=True)

    X = X_pl.to_pandas()
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].astype("category")

    # Target: delta_offset = BLOCK_TIME - AOBT_3
    aobt_col = dep_df["AOBT_3_flt"]
    block_col = dep_df["BLOCK_TIME_UTC_mvt"]
    sched_col = dep_df["SCHED_TIME_UTC_mvt"]
    adep_col = dep_df["ADEP_mvt"].to_numpy()

    has_aobt = aobt_col.is_not_null().to_numpy()
    y_offset = (block_col - aobt_col).dt.total_seconds().to_numpy()

    # Filter out LIRF schedule-defaults (|BLOCK - SCHED| <= 15s)
    d_bs = (block_col - sched_col).dt.total_seconds().to_numpy()
    is_lirf_default = (adep_col == "LIRF") & (np.abs(d_bs) <= 15)

    # Valid mask for true physical offset
    valid_mask = has_aobt & (~is_lirf_default) & (y_offset >= -3600) & (y_offset <= 3600)

    X_train = X[valid_mask]
    y_train = y_offset[valid_mask]
    print(f"Offset Training Matrix ready: {len(X_train):,} rows ({np.mean(valid_mask):.2%} of departures), {X_train.shape[1]} features.")
    print(f"  Target stats: mean={np.mean(y_train):.2f}s, std={np.std(y_train):.2f}s, median={np.median(y_train):.2f}s")

    cat_cols_list = [c for c in X_train.select_dtypes(include=["category"]).columns]

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": 11,
        "min_child_samples": 40,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "n_jobs": 16,
        "random_state": 42,
        "verbose": -1,
    }

    n_trees = 1200
    print(f"\n[{time.strftime('%H:%M:%S')}] Fitting LightGBM ({n_trees} trees, num_leaves=127, lr=0.03)...")
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols_list, free_raw_data=False)
    t_fit = time.time()
    model = lgb.train(params, dtrain, num_boost_round=n_trees)
    fit_time = time.time() - t_fit

    preds_train = model.predict(X_train)
    train_rmse = np.sqrt(np.mean((y_train - preds_train) ** 2))
    print(f"✅ LightGBM Offset fitted in {fit_time:.1f}s | Train RMSE on delta: {train_rmse:.2f}s")

    model_path = MODELS_DIR / "lgb_offset_deep.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / "encs_lgb_offset.pkl", "wb") as f:
        pickle.dump(encs, f)
    print(f"💾 Model saved to {model_path}")

    # Generate Test Predictions on ranking.parquet
    print(f"\n[{time.strftime('%H:%M:%S')}] Generating test predictions on ranking.parquet...")
    rank_df = pl.read_parquet(RAW_DIR / "ranking.parquet")
    rank_df = features.compute_congestion_metrics(rank_df)
    rank_df = features.compute_surface_accumulation_features(rank_df)
    rank_df = features.compute_apron_contention_features(rank_df)
    rank_df = features.compute_runway_sequence_features(rank_df)

    X_test_pl, _, _ = features.extract_features(rank_df, target_encodings=encs, attach_weather=True)
    X_test = X_test_pl.to_pandas()
    for c in X_test.select_dtypes(include=["object"]).columns:
        X_test[c] = X_test[c].astype("category")

    print(f"Running inference on {len(X_test):,} test departure rows...")
    test_preds = model.predict(X_test)

    pred_path = SCRATCH_DIR / "lgb_offset_test_preds.npy"
    np.save(pred_path, test_preds)
    print(f"💾 Saved test offset predictions to {pred_path}")
    print(f"  Test offset stats: mean={np.mean(test_preds):.2f}s, std={np.std(test_preds):.2f}s, min={np.min(test_preds):.2f}s, max={np.max(test_preds):.2f}s")


if __name__ == "__main__":
    main()
