"""
Train Deep Foundation Models (CatBoost Deep + LightGBM Deep)
on All 12 Months of 2025 (2,085,047 Movements) with 135 Deep Domain & Runway Queue Features.
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
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features
from src.models_catboost import prepare_catboost_df

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ALL_12_MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]


def train_deep_foundations():
    print("=" * 80)
    print("🚀 TRAINING DEEP FOUNDATION MODELS ON 2.08 MILLION DEPARTURES (12 MONTHS)")
    print("=" * 80)

    t0 = time.time()
    # 1. Load full year data
    print(f"[{time.strftime('%H:%M:%S')}] Loading all 12 months of 2025...")
    df = features.load_training_data(RAW_DIR, months=ALL_12_MONTHS)
    print(f"Loaded {len(df):,} rows in {time.time() - t0:.1f}s")

    # 2. Compute Congestion & Dynamic Runway Sequencing
    print(f"[{time.strftime('%H:%M:%S')}] Computing congestion, surface traffic & runway sequence features...")
    df = features.compute_congestion_metrics(df)
    df = features.compute_surface_accumulation_features(df)
    df = features.compute_runway_sequence_features(df)

    # 3. Target encodings on departures
    dep_df = df.filter(pl.col("PHASE_mvt") == "DEP")
    print(f"[{time.strftime('%H:%M:%S')}] Computing target encodings across {len(dep_df):,} departures...")
    encs = features.compute_target_encodings(dep_df)

    # 4. Extract all 135 features
    print(f"[{time.strftime('%H:%M:%S')}] Extracting full feature matrix...")
    X_pl, _, y = features.extract_features(df, target_encodings=encs, attach_weather=True)

    # Valid mask for training
    valid_mask = (y >= 60) & (y <= 150000)
    y_train = y[valid_mask]
    print(f"Total valid training samples: {len(y_train):,}")

    # =========================================================================
    # MODEL 1: CatBoost Full-Year Deep
    # =========================================================================
    print("\n" + "=" * 70)
    print("🐱 [1/2] TRAINING CATBOOST FULL-YEAR DEEP (depth=8, iters=1200, lr=0.04)")
    print("=" * 70)
    t_cb = time.time()

    X_cb = prepare_catboost_df(X_pl)
    X_cb_train = X_cb.iloc[valid_mask].reset_index(drop=True)

    all_cat_cols = [c for c in X_cb_train.columns if str(X_cb_train[c].dtype) in ["category", "object"]]
    for c in all_cat_cols:
        X_cb_train[c] = X_cb_train[c].astype(str).fillna("UNK")
    cat_indices = [X_cb_train.columns.get_loc(c) for c in all_cat_cols]
    cols_cb = list(X_cb_train.columns)

    print(f"Training CatBoost on {len(X_cb_train):,} rows ({len(cols_cb)} features, {len(cat_indices)} categorical)...")
    cb_model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.04,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        cat_features=cat_indices,
        random_seed=42,
        thread_count=-1,
        verbose=150,
    )
    cb_model.fit(X_cb_train, y_train)

    cb_preds = np.clip(cb_model.predict(X_cb_train), 60, 88500)
    cb_rmse = np.sqrt(np.mean((y_train - cb_preds) ** 2))
    print(f"✅ CatBoost Full-Year Deep Train RMSE: {cb_rmse:.2f}s in {time.time() - t_cb:.1f}s")

    # Save CatBoost Deep
    cb_model.save_model(str(MODELS_DIR / "cb_fullyear_deep.cbm"))
    with open(MODELS_DIR / "encs_cb_fullyear_deep.pkl", "wb") as f:
        pickle.dump(encs, f)
    with open(MODELS_DIR / "cols_cb_fullyear_deep.pkl", "wb") as f:
        pickle.dump(cols_cb, f)
    print(f"💾 Saved CatBoost Deep to {MODELS_DIR / 'cb_fullyear_deep.cbm'}")

    # =========================================================================
    # MODEL 2: LightGBM Full-Year Deep
    # =========================================================================
    print("\n" + "=" * 70)
    print("⚡ [2/2] TRAINING LIGHTGBM FULL-YEAR DEEP (num_leaves=127, iters=1000, lr=0.03)")
    print("=" * 70)
    t_lgb = time.time()

    # Prepare DataFrame for LightGBM
    X_lgb = X_pl.to_pandas()
    for c in X_lgb.select_dtypes(include=["object", "category"]).columns:
        X_lgb[c] = X_lgb[c].astype("category")
    cols_lgb = list(X_lgb.columns)
    X_lgb_train = X_lgb.iloc[valid_mask].reset_index(drop=True)

    print(f"Training LightGBM on {len(X_lgb_train):,} rows ({len(cols_lgb)} features)...")
    lgb_train_data = lgb.Dataset(X_lgb_train, label=y_train)

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

    lgb_model = lgb.train(
        params,
        lgb_train_data,
        num_boost_round=1000,
    )

    lgb_preds = np.clip(lgb_model.predict(X_lgb_train), 60, 88500)
    lgb_rmse = np.sqrt(np.mean((y_train - lgb_preds) ** 2))
    print(f"✅ LightGBM Full-Year Deep Train RMSE: {lgb_rmse:.2f}s in {time.time() - t_lgb:.1f}s")

    # Save LightGBM Deep
    with open(MODELS_DIR / "lgb_fullyear_deep.pkl", "wb") as f:
        pickle.dump(lgb_model, f)
    with open(MODELS_DIR / "encs_lgb_fullyear_deep.pkl", "wb") as f:
        pickle.dump(encs, f)
    with open(MODELS_DIR / "cols_lgb_fullyear_deep.pkl", "wb") as f:
        pickle.dump(cols_lgb, f)
    print(f"💾 Saved LightGBM Deep to {MODELS_DIR / 'lgb_fullyear_deep.pkl'}")

    print("\n" + "=" * 80)
    print(f"🎉 ALL DEEP FOUNDATION MODELS TRAINED & SAVED IN {time.time() - t0:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    train_deep_foundations()
