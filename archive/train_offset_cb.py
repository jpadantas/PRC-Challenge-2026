"""
Train Full-Year CatBoost Deep Offset Regressor (182 Features) on full 2025 dataset.
Accelerated on NVIDIA GeForce RTX 4060 GPU with task_type='GPU'.
Saves models/cb_offset_deep.cbm and scratch/cb_offset_test_preds.npy.
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
from catboost import CatBoostRegressor

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
    print("🐱 TRAINING FULL-YEAR CATBOOST OFFSET REGRESSOR (182 FEATURES, GPU)")
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
    cat_cols = [c for c in X.columns if str(X[c].dtype) in ["category", "object"]]
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("UNK")

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

    X_train = X.iloc[valid_mask].reset_index(drop=True)
    y_train = y_offset[valid_mask]
    print(f"Offset Training Matrix ready: {len(X_train):,} rows ({np.mean(valid_mask):.2%} of departures), {X_train.shape[1]} features.")
    print(f"  Target stats: mean={np.mean(y_train):.2f}s, std={np.std(y_train):.2f}s, median={np.median(y_train):.2f}s")
    print(f"Categorical features: {len(cat_cols)}")

    n_iter = 1200
    print(f"\n[{time.strftime('%H:%M:%S')}] Fitting CatBoost on RTX 4060 GPU ({n_iter} iterations, lr=0.04, depth=8)...")
    model = CatBoostRegressor(
        iterations=n_iter,
        learning_rate=0.04,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        task_type="GPU",
        verbose=100,
    )

    t_fit = time.time()
    model.fit(X_train, y_train, cat_features=cat_cols)
    fit_time = time.time() - t_fit
    print(f"✅ CatBoost Offset fitted in {fit_time:.1f}s")

    preds_train = model.predict(X_train)
    train_rmse = np.sqrt(np.mean((y_train - preds_train) ** 2))
    print(f"Train RMSE on delta: {train_rmse:.2f}s")

    model_path = MODELS_DIR / "cb_offset_deep.cbm"
    model.save_model(model_path)
    with open(MODELS_DIR / "encs_cb_offset.pkl", "wb") as f:
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
    for c in cat_cols:
        X_test[c] = X_test[c].astype(str).fillna("UNK")

    print(f"Running inference on {len(X_test):,} test departure rows...")
    test_preds = model.predict(X_test)

    pred_path = SCRATCH_DIR / "cb_offset_test_preds.npy"
    np.save(pred_path, test_preds)
    print(f"💾 Saved test offset predictions to {pred_path}")
    print(f"  Test offset stats: mean={np.mean(test_preds):.2f}s, std={np.std(test_preds):.2f}s, min={np.min(test_preds):.2f}s, max={np.max(test_preds):.2f}s")


if __name__ == "__main__":
    main()
