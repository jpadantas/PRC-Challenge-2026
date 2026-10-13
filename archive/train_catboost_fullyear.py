"""
Train Full-Year CatBoost Model on all 12 months (2.08M movements).
Optimized for high-cardinality categorical airport layout and queuing dynamics.
"""

import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import time
import pickle
import numpy as np
import polars as pl
from catboost import CatBoostRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features
from src.models_catboost import prepare_catboost_df, CAT_COLS

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ALL_12_MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]


def train_catboost_fullyear(iterations: int = 700, learning_rate: float = 0.05, depth: int = 7):
    print("=" * 75)
    print("🐱 TRAINING FULL-YEAR CATBOOST MODEL (2,085,000+ DEPARTURES)")
    print("=" * 75)

    t0 = time.time()
    # 1. Load data
    df = features.load_training_data(RAW_DIR, months=ALL_12_MONTHS)
    print(f"Loaded {len(df):,} total rows across all 12 months in {time.time() - t0:.1f}s")

    df = features.compute_congestion_metrics(df)
    df = features.compute_runway_sequence_features(df)

    # 2. Target encodings
    dep_df = df.filter(pl.col("PHASE_mvt") == "DEP")
    encs = features.compute_target_encodings(dep_df)

    # 3. Features
    print("Extracting features for CatBoost...")
    X_pl, _, y = features.extract_features(df, target_encodings=encs)
    X = prepare_catboost_df(X_pl)

    valid_mask = (y >= 0) & (y <= 150000)
    X_train = X.iloc[valid_mask].reset_index(drop=True)
    y_train = y[valid_mask]

    # Automatically discover ALL categorical columns
    all_cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) in ["category", "object"]]
    for c in all_cat_cols:
        X_train[c] = X_train[c].astype(str).fillna("UNK")
    cat_indices = [X_train.columns.get_loc(c) for c in all_cat_cols]

    cols = list(X_train.columns)
    print(f"Training CatBoost on {len(X_train):,} rows ({len(cols)} features, {len(cat_indices)} categorical: {all_cat_cols})...")

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="RMSE",
        eval_metric="RMSE",
        cat_features=cat_indices,
        random_seed=42,
        thread_count=-1,
        verbose=100,
    )
    model.fit(X_train, y_train)

    train_preds = np.clip(model.predict(X_train), 60, 86400)
    train_rmse = np.sqrt(np.mean((y_train - train_preds) ** 2))
    print(f"✅ CatBoost Full-Year Train RMSE: {train_rmse:.2f}s in {time.time() - t0:.1f}s")

    # Save
    model.save_model(str(MODELS_DIR / "cb_fullyear.cbm"))
    with open(MODELS_DIR / "encs_cb_fullyear.pkl", "wb") as f:
        pickle.dump(encs, f)
    with open(MODELS_DIR / "cols_cb_fullyear.pkl", "wb") as f:
        pickle.dump(cols, f)

    print(f"🎉 CatBoost Full-Year saved successfully to {MODELS_DIR / 'cb_fullyear.cbm'}")


if __name__ == "__main__":
    train_catboost_fullyear(iterations=600, learning_rate=0.06, depth=7)
