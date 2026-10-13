"""
Train Full-Year XGBoost Deep v61 with all 182 physical, aerodynamic, METAR, and OSM taxiway network features.
Optimized for NVIDIA GeForce RTX 4060 GPU with CUDA tree_method='hist'.
Saves models/xgb_v61_deep.json and scratch/xgb_v61_test_preds.npy.
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
    print("🚀 TRAINING NEXT-GEN XGBOOST FULL-YEAR DEEP V61 (182 PHYSICAL + OSM + METAR FEATURES)")
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
    for c in X.select_dtypes(include=["object", "category"]).columns:
        X[c] = X[c].astype("category")

    valid_mask = (y >= 60) & (y <= 150000)
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"Training matrix ready: {len(X):,} departure rows, {X.shape[1]} features.")

    params = {
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": 0.04,
        "max_depth": 10,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "min_child_weight": 25,
        "eval_metric": "rmse",
        "enable_categorical": True,
        "random_state": 42,
    }

    n_trees = 1500
    print(f"\n[{time.strftime('%H:%M:%S')}] Fitting XGBoost on RTX 4060 GPU ({n_trees} trees)...")
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
    t_fit = time.time()
    bst = xgb.train(params, dtrain, num_boost_round=n_trees)
    fit_time = time.time() - t_fit
    print(f"✅ XGBoost v61 fitted in {fit_time:.1f}s")

    preds_train = bst.predict(dtrain)
    train_rmse = np.sqrt(np.mean((y - preds_train) ** 2))
    print(f"Train RMSE: {train_rmse:.2f}s")

    model_path = MODELS_DIR / "xgb_v61_deep.json"
    bst.save_model(model_path)
    with open(MODELS_DIR / "encs_xgb_v61.pkl", "wb") as f:
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
    for c in X_test.select_dtypes(include=["object", "category"]).columns:
        X_test[c] = X_test[c].astype("category")

    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    test_preds = bst.predict(dtest)
    test_preds = np.clip(test_preds, 60, 88500)

    out_npy = SCRATCH_DIR / "xgb_v61_test_preds.npy"
    np.save(out_npy, test_preds)
    print(f"💾 Saved v61 XGBoost test predictions: {out_npy}")
    print(f"   Stats: mean={test_preds.mean():.1f}s, min={test_preds.min():.1f}s, max={test_preds.max():.1f}s")
    print(f"🎉 Total execution time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
