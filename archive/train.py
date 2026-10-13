"""
Training Pipeline for PRC Data Challenge 2026.

Trains two specialized seasonal models:
1. Winter Model (for January 2026 evaluation): trained on winter months (Nov, Dec 2025, Jan, Feb 2025)
2. Summer Model (for July 2026 evaluation): trained on summer months (May, Jun, Jul, Aug 2025)

Each model learns the seasonal taxi-out dynamics (winter de-icing queues vs summer peak runway traffic).
Saves models and target encodings to `models/`.
"""

import pickle
import time
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

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


def train_seasonal_model(
    season_name: str,
    months: list[str],
    n_estimators: int = 600,
    learning_rate: float = 0.05,
    num_leaves: int = 70,
):
    print(f"\n{'=' * 60}")
    print(f"🚀 Training {season_name.upper()} Model (Months: {months})")
    print(f"{'=' * 60}")

    t0 = time.time()
    # 1. Load data
    df = features.load_training_data(RAW_DIR, months=months)

    # 2. Congestion & Runway Sequence metrics
    df = features.compute_congestion_metrics(df)
    df = features.compute_runway_sequence_features(df)

    # 3. Target encodings
    dep_df = df.filter(pl.col("PHASE_mvt") == "DEP")
    encs = features.compute_target_encodings(dep_df)

    # 4. Feature extraction
    X_pl, base_pred, y = features.extract_features(df, target_encodings=encs)
    X = X_pl.to_pandas()

    for c in X.select_dtypes(include="category").columns:
        X[c] = X[c].astype("category")

    # 5. Target formulation: Direct prediction of unclipped continuous TAXITIME_SEC_mvt
    valid_mask = (y >= 0) & (y <= 150000)
    X_train = X.iloc[valid_mask].reset_index(drop=True)
    y_train = y[valid_mask]

    print(f"Training on {len(X_train):,} departure rows ({len(X.columns)} features)...")

    # 6. Fit LightGBM
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    train_preds = np.clip(model.predict(X_train), 60, 86400)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    train_mae = np.mean(np.abs(y_train - train_preds))
    print(f"✅ {season_name.capitalize()} Model Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f} in {time.time() - t0:.1f}s")

    # Feature importances
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:15]
    print(f"\nTop 15 features for {season_name}:")
    for idx in top_indices:
        print(f"  • {X.columns[idx]:25s}: {importances[idx]}")

    # 7. Save model and encodings
    model_path = MODELS_DIR / f"lgb_{season_name}.pkl"
    encs_path = MODELS_DIR / f"encs_{season_name}.pkl"
    cols_path = MODELS_DIR / f"cols_{season_name}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(encs_path, "wb") as f:
        pickle.dump(encs, f)
    with open(cols_path, "wb") as f:
        pickle.dump(list(X.columns), f)

    print(f"Saved artifacts to {MODELS_DIR}")
    return model, encs


def main():
    print("Starting PRC 2026 Model Training Pipeline...")
    # 1. Train Winter Model (Jan, Feb, Nov, Dec 2025)
    train_seasonal_model(
        season_name="winter",
        months=["2025-01", "2025-02", "2025-11", "2025-12"],
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=95,
    )

    # 2. Train Summer Model (May, Jun, Jul, Aug 2025)
    train_seasonal_model(
        season_name="summer",
        months=["2025-05", "2025-06", "2025-07", "2025-08"],
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=95,
    )

    print("\n🎉 Both models trained and saved successfully!")


if __name__ == "__main__":
    main()
