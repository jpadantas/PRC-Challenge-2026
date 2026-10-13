"""
QUANTILE REGRESSION XGBoost: Target the heavy tail (p75 quantile loss).
Flights with >2000s taxi time dominate the RMSE metric. A model trained
specifically with quantile loss at tau=0.75 will predict higher values for
congested flights and better approximate the right tail.

Blend strategy: 60% MSE model + 40% Quantile model 
= preserves accuracy in the bulk while lifting the tail predictions.
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

print("=" * 80)
print("QUANTILE REGRESSION XGBoost (tau=0.75) — TAIL MODELING")
print("=" * 80)
t0 = time.time()

# Load + feature engineering (reuse from global pipeline)
print(f"[{time.strftime('%H:%M:%S')}] Loading and processing training data...")
df = features.load_training_data(RAW_DIR, months=ALL_12_MONTHS)
df = features.compute_congestion_metrics(df)
df = features.compute_surface_accumulation_features(df)
df = features.compute_apron_contention_features(df)
df = features.compute_runway_sequence_features(df)

dep_df = df.filter(pl.col("PHASE_mvt") == "DEP")
encs = features.compute_target_encodings(dep_df)

X_pl, _, y = features.extract_features(df, target_encodings=encs, attach_weather=True)
X = X_pl.to_pandas()
for c in X.select_dtypes(include=["object", "category"]).columns:
    X[c] = X[c].astype("category")

valid_mask = (y >= 60) & (y <= 150000)
X_train = X.iloc[valid_mask].reset_index(drop=True)
y_train = y[valid_mask]
print(f"Training rows: {len(X_train):,} | Time: {time.time()-t0:.1f}s")

cols = list(X_train.columns)
cat_dtypes = {c: X_train[c].dtype for c in X_train.select_dtypes(include="category").columns}

# Train quantile model at tau=0.75 (better tail coverage)
print(f"\n[{time.strftime('%H:%M:%S')}] Training Quantile XGBoost (tau=0.75) on GPU CUDA...")
t_q = time.time()
model_q75 = xgb.XGBRegressor(
    objective="reg:quantileerror",
    quantile_alpha=0.75,
    n_estimators=1200,
    learning_rate=0.04,
    max_depth=9,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.5,
    reg_lambda=2.0,
    enable_categorical=True,
    tree_method="hist",
    device="cuda",
    random_state=42,
)
model_q75.fit(X_train, y_train)
preds_q75 = np.clip(model_q75.predict(X_train), 60, 88500)
q75_rmse = np.sqrt(np.mean((y_train - preds_q75)**2))
print(f"Quantile(0.75) Train RMSE: {q75_rmse:.2f}s | Time: {time.time()-t_q:.1f}s")
print(f"  mean={preds_q75.mean():.1f}s  std={preds_q75.std():.1f}s  p90={np.percentile(preds_q75,90):.1f}s")

# Also train at tau=0.65 for comparison
print(f"\n[{time.strftime('%H:%M:%S')}] Training Quantile XGBoost (tau=0.65) on GPU CUDA...")
t_q2 = time.time()
model_q65 = xgb.XGBRegressor(
    objective="reg:quantileerror",
    quantile_alpha=0.65,
    n_estimators=1200,
    learning_rate=0.04,
    max_depth=9,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.5,
    reg_lambda=2.0,
    enable_categorical=True,
    tree_method="hist",
    device="cuda",
    random_state=43,
)
model_q65.fit(X_train, y_train)
preds_q65 = np.clip(model_q65.predict(X_train), 60, 88500)
q65_rmse = np.sqrt(np.mean((y_train - preds_q65)**2))
print(f"Quantile(0.65) Train RMSE: {q65_rmse:.2f}s | Time: {time.time()-t_q2:.1f}s")

# Save models
model_q75.save_model(str(MODELS_DIR / "xgb_quantile075.json"))
model_q65.save_model(str(MODELS_DIR / "xgb_quantile065.json"))
with open(MODELS_DIR / "cols_quantile.pkl", "wb") as f:
    pickle.dump(cols, f)
with open(MODELS_DIR / "cat_dtypes_quantile.pkl", "wb") as f:
    pickle.dump(cat_dtypes, f)
print(f"Saved quantile models.")

# Run test inference
print(f"\n[{time.strftime('%H:%M:%S')}] Running test inference...")
from src.features import compute_congestion_metrics, compute_surface_accumulation_features
from src.features import compute_apron_contention_features, compute_runway_sequence_features

rank_df = pl.read_parquet(RAW_DIR / "ranking.parquet")
rank_df = compute_congestion_metrics(rank_df)
rank_df = compute_surface_accumulation_features(rank_df)
rank_df = compute_apron_contention_features(rank_df)
rank_df = compute_runway_sequence_features(rank_df)

dep_rank = rank_df.filter(pl.col("PHASE_mvt") == "DEP")
X_test_pl, _, _ = features.extract_features(dep_rank, target_encodings=encs, attach_weather=True)
X_test = X_test_pl.to_pandas()
for c in cols:
    if c in cat_dtypes:
        if c in X_test.columns:
            X_test[c] = pd.Categorical(X_test[c].astype(str), categories=cat_dtypes[c].categories)
        else:
            X_test[c] = pd.Categorical([None]*len(X_test), categories=cat_dtypes[c].categories)
    elif c not in X_test.columns:
        X_test[c] = 0.0

test_q75 = np.clip(model_q75.predict(X_test[cols]), 60, 88500)
test_q65 = np.clip(model_q65.predict(X_test[cols]), 60, 88500)

np.save(SCRATCH_DIR / "quantile075_test_preds.npy", test_q75)
np.save(SCRATCH_DIR / "quantile065_test_preds.npy", test_q65)

# Correlation with MSE model predictions
mse_preds = np.load(SCRATCH_DIR / "xgb_deep_test_preds.npy")
corr_075 = np.corrcoef(mse_preds, test_q75)[0,1]
corr_065 = np.corrcoef(mse_preds, test_q65)[0,1]
print(f"Q75 test: mean={test_q75.mean():.1f}s  std={test_q75.std():.1f}s  corr_with_MSE={corr_075:.4f}")
print(f"Q65 test: mean={test_q65.mean():.1f}s  std={test_q65.std():.1f}s  corr_with_MSE={corr_065:.4f}")
print(f"\nTotal time: {time.time()-t0:.1f}s")
print("Saved: quantile075_test_preds.npy, quantile065_test_preds.npy")
