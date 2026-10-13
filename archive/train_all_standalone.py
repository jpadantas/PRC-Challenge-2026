"""
Retrain Standalone Models (LightGBM and XGBoost) with Enhanced Preprocessing.
Trains:
  1. LightGBM Winter & Summer
  2. XGBoost Winter & Summer
Saves models, smoothed target encodings, columns, and categorical dtypes into models/.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import time
import pickle
import numpy as np
import polars as pl
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

WINTER_MONTHS = ["2025-01", "2025-02", "2025-11", "2025-12"]
SUMMER_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08"]

def train_season(season_name: str, months: list[str]):
    print(f"\n{'=' * 70}")
    print(f"🚀 RETRAINING STANDALONE MODELS: {season_name.upper()} ({months})")
    print(f"{'=' * 70}")
    
    t0 = time.time()
    # 1. Load, congestion and runway sequence
    df = features.load_training_data(RAW_DIR, months=months)
    df = features.compute_congestion_metrics(df)
    df = features.compute_runway_sequence_features(df)
    
    # 2. Compute smoothed target encodings
    dep_df = df.filter(pl.col("PHASE_mvt") == "DEP")
    encs = features.compute_target_encodings(dep_df)
    
    # 3. Extract features
    print("Extracting enhanced preprocessed features...")
    X_pl, _, y = features.extract_features(df, target_encodings=encs)
    
    X = X_pl.to_pandas()
    for c in X.select_dtypes(include="category").columns:
        X[c] = X[c].astype("category")
        
    cols = list(X.columns)
    cat_dtypes = {c: X[c].dtype for c in X.select_dtypes(include="category").columns}
    
    valid_mask = (y >= 0) & (y <= 150000)
    X_train = X.iloc[valid_mask].reset_index(drop=True)
    y_train = y[valid_mask]
    
    print(f"Dataset ready: {len(X_train):,} rows, {len(cols)} features. ({time.time() - t0:.1f}s)")
    
    # 4. Save metadata
    with open(MODELS_DIR / f"encs_{season_name}.pkl", "wb") as f:
        pickle.dump(encs, f)
    with open(MODELS_DIR / f"cols_{season_name}.pkl", "wb") as f:
        pickle.dump(cols, f)
    with open(MODELS_DIR / f"cat_dtypes_{season_name}.pkl", "wb") as f:
        pickle.dump(cat_dtypes, f)
        
    # --- A. Train LightGBM ---
    print(f"\n[1/2] Training LightGBM {season_name.upper()}...")
    t_lgb = time.time()
    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=95,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.80,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train)
    p_lgb = np.clip(lgb_model.predict(X_train), 60, 86400)
    rmse_lgb = np.sqrt(np.mean((y_train - p_lgb)**2))
    print(f"✅ LightGBM {season_name.upper()} fitted in {time.time() - t_lgb:.1f}s | Train RMSE: {rmse_lgb:.2f}s")
    
    with open(MODELS_DIR / f"lgb_{season_name}.pkl", "wb") as f:
        pickle.dump(lgb_model, f)
        
    # --- B. Train XGBoost ---
    print(f"\n[2/2] Training XGBoost {season_name.upper()}...")
    t_xgb = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=650,
        learning_rate=0.045,
        max_depth=9,
        tree_method="hist",
        enable_categorical=True,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_lambda=1.5,
        reg_alpha=0.3,
        gamma=0.2,
        min_child_weight=25,
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    p_xgb = np.clip(xgb_model.predict(X_train), 60, 86400)
    rmse_xgb = np.sqrt(np.mean((y_train - p_xgb)**2))
    print(f"✅ XGBoost {season_name.upper()} fitted in {time.time() - t_xgb:.1f}s | Train RMSE: {rmse_xgb:.2f}s")
    
    xgb_model.save_model(str(MODELS_DIR / f"xgb_{season_name}.json"))
    print(f"🎉 All {season_name.upper()} models saved successfully in {time.time() - t0:.1f}s total.")


if __name__ == "__main__":
    train_season("winter", WINTER_MONTHS)
    train_season("summer", SUMMER_MONTHS)
