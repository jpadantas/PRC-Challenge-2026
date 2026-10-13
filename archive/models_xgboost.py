"""
XGBoost Training and Inference Pipeline for PRC Data Challenge 2026.

Trains seasonal XGBoost models using fast histogram-based tree splitting ('hist')
and native categorical support to provide high-diversity tree predictions for the ensemble.
"""

import pickle
import time
from pathlib import Path
import numpy as np
import polars as pl
import xgboost as xgb

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


def prepare_xgboost_df(X_pl: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Convert Polars DataFrame to pandas with category dtypes for XGBoost native categorical support."""
    X = X_pl.to_pandas()
    for c in X.select_dtypes(include="category").columns:
        X[c] = X[c].astype("category")
    return X


def train_seasonal_xgboost(
    season_name: str,
    months: list[str],
    n_estimators: int = 600,
    learning_rate: float = 0.05,
    max_depth: int = 9,
):
    print(f"\n{'=' * 60}")
    print(f"⚡ Training XGBoost {season_name.upper()} Model (Months: {months})")
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
    X = prepare_xgboost_df(X_pl)

    # 4. Target formulation: Direct unclipped continuous TAXITIME_SEC_mvt
    valid_mask = (y >= 0) & (y <= 150000)
    X_train = X.iloc[valid_mask].reset_index(drop=True)
    y_train = y[valid_mask]

    print(f"Training XGBoost on {len(X_train):,} rows ({len(X.columns)} features)...")

    # 5. Fit XGBoost with Optuna-tuned parameters by season
    if season_name == "winter":
        params = {
            "n_estimators": n_estimators,
            "learning_rate": 0.0457,
            "max_depth": 11,
            "subsample": 0.734,
            "colsample_bytree": 0.780,
            "reg_lambda": 1.915,
            "reg_alpha": 0.710,
            "min_child_weight": 16,
        }
    else:
        params = {
            "n_estimators": n_estimators,
            "learning_rate": 0.0599,
            "max_depth": 10,
            "subsample": 0.829,
            "colsample_bytree": 0.882,
            "reg_lambda": 2.208,
            "reg_alpha": 0.059,
            "min_child_weight": 20,
        }

    model = xgb.XGBRegressor(
        **params,
        tree_method="hist",
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)

    train_preds = np.clip(model.predict(X_train), 60, 86400)
    train_rmse = np.sqrt(np.mean((y_train - train_preds) ** 2))
    train_mae = np.mean(np.abs(y_train - train_preds))
    print(f"✅ XGBoost {season_name.capitalize()} Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f} in {time.time() - t0:.1f}s")

    # 6. Save model & encodings
    model_path = MODELS_DIR / f"xgb_{season_name}.json"
    encs_path = MODELS_DIR / f"encs_{season_name}.pkl"
    cols_path = MODELS_DIR / f"cols_{season_name}.pkl"
    cat_dtypes_path = MODELS_DIR / f"cat_dtypes_{season_name}.pkl"

    cat_dtypes = {c: X_train[c].dtype for c in X_train.select_dtypes(include="category").columns}
    with open(cat_dtypes_path, "wb") as f:
        pickle.dump(cat_dtypes, f)

    model.save_model(str(model_path))
    with open(encs_path, "wb") as f:
        pickle.dump(encs, f)
    with open(cols_path, "wb") as f:
        pickle.dump(list(X.columns), f)

    print(f"Saved tuned XGBoost model to {model_path} and categorical dtypes to {cat_dtypes_path}")
    return model, encs


if __name__ == "__main__":
    train_seasonal_xgboost("winter", months=["2025-01", "2025-02", "2025-11", "2025-12"], n_estimators=650)
    train_seasonal_xgboost("summer", months=["2025-05", "2025-06", "2025-07", "2025-08"], n_estimators=650)
