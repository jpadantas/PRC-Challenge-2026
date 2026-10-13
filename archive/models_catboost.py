"""
CatBoost Training and Inference Pipeline for PRC Data Challenge 2026.

Trains seasonal CatBoost models leveraging native handling of high-cardinality
airport layout features (stands, runways, piers, airlines).
"""

import pickle
import time
from pathlib import Path
import numpy as np
import polars as pl
from catboost import CatBoostRegressor

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CAT_COLS = [
    "adep", "runway", "stand", "stand_pier", "apt_rwy",
    "apt_pier_rwy", "aircraft_type", "wk_tbl", "operator",
    "market_segment", "flight_rule"
]


def prepare_catboost_df(X_pl: pl.DataFrame, cat_cols: list[str] = CAT_COLS) -> tuple[np.ndarray, list[str]]:
    """Convert Polars feature DataFrame to pandas formatted for CatBoost."""
    X = X_pl.to_pandas()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("UNK")
    return X


def train_seasonal_catboost(
    season_name: str,
    months: list[str],
    iterations: int = 800,
    learning_rate: float = 0.06,
    depth: int = 7,
):
    print(f"\n{'=' * 60}")
    print(f"🐱 Training CatBoost {season_name.upper()} Model (Months: {months})")
    print(f"{'=' * 60}")

    t0 = time.time()
    # 1. Load data
    df = features.load_training_data(RAW_DIR, months=months)
    df = features.compute_congestion_metrics(df)
    df = features.compute_runway_sequence_features(df)

    # 2. Target encodings
    dep_df = df.filter(pl.col("PHASE_mvt") == "DEP")
    encs = features.compute_target_encodings(dep_df)

    # 3. Features
    X_pl, _, y = features.extract_features(df, target_encodings=encs)
    X = prepare_catboost_df(X_pl)

    # 4. Target formulation: Direct unclipped continuous TAXITIME_SEC_mvt
    valid_mask = (y >= 0) & (y <= 150000)
    X_train = X.iloc[valid_mask].reset_index(drop=True)
    y_train = y[valid_mask]

    print(f"Training CatBoost on {len(X_train):,} rows ({len(X.columns)} features)...")

    # 5. Fit CatBoost
    cat_indices = [X_train.columns.get_loc(c) for c in CAT_COLS if c in X_train.columns]

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
    train_mae = np.mean(np.abs(y_train - train_preds))
    print(f"✅ CatBoost {season_name.capitalize()} Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f} in {time.time() - t0:.1f}s")

    # 6. Save model & encodings
    model_path = MODELS_DIR / f"cb_{season_name}.cbm"
    encs_path = MODELS_DIR / f"encs_cb_{season_name}.pkl"
    cols_path = MODELS_DIR / f"cols_cb_{season_name}.pkl"

    model.save_model(str(model_path))
    with open(encs_path, "wb") as f:
        pickle.dump(encs, f)
    with open(cols_path, "wb") as f:
        pickle.dump(list(X.columns), f)

    print(f"Saved CatBoost model to {model_path}")
    return model, encs


if __name__ == "__main__":
    # Train winter and summer CatBoost models
    train_seasonal_catboost("winter", months=["2025-01", "2025-02", "2025-11", "2025-12"], iterations=800, learning_rate=0.06, depth=7)
    train_seasonal_catboost("summer", months=["2025-05", "2025-06", "2025-07", "2025-08"], iterations=800, learning_rate=0.06, depth=7)
