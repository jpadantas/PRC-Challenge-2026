"""
Calibrate 3-Way Ensemble weights using SciPy optimization on Jan 2025.
"""

import pickle
from pathlib import Path
import numpy as np
import polars as pl
from scipy.optimize import minimize
from catboost import CatBoostRegressor
import xgboost as xgb

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features
from src.models_catboost import prepare_catboost_df

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"


def main():
    print("Loading 2025-01 data...")
    df = features.load_training_data(RAW_DIR, months=["2025-01"])
    df = features.compute_congestion_metrics(df)
    dep_df = df.filter(pl.col("PHASE_mvt") == "DEP")
    y = dep_df["TAXITIME_SEC_mvt"].to_numpy()

    # Clean operational mask
    clean = (y >= 60) & (y <= 7200)
    y_clean = y[clean]
    print(f"Total flights: {len(y):,}, Clean operational flights: {np.sum(clean):,}")

    # Load artifacts
    with open(MODELS_DIR / "encs_winter.pkl", "rb") as f:
        encs = pickle.load(f)
    with open(MODELS_DIR / "cols_winter.pkl", "rb") as f:
        cols = pickle.load(f)
    with open(MODELS_DIR / "lgb_winter.pkl", "rb") as f:
        lgb_model = pickle.load(f)

    cb_model = CatBoostRegressor()
    cb_model.load_model(str(MODELS_DIR / "cb_winter.cbm"))

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(str(MODELS_DIR / "xgb_winter.json"))

    print("Extracting features...")
    X_pl, _, _ = features.extract_features(dep_df, target_encodings=encs)

    print("Generating single model predictions...")
    # CatBoost
    X_cb = prepare_catboost_df(X_pl)
    p_cb = cb_model.predict(X_cb[cols])[clean]
    rmse_cb = np.sqrt(np.mean((y_clean - np.clip(p_cb, 60, 7200)) ** 2))
    print(f"  • CatBoost Clean RMSE: {rmse_cb:.4f}")

    # LightGBM
    X_lgb = X_pl.select(cols).to_pandas()
    for c in X_lgb.select_dtypes(include="category").columns:
        X_lgb[c] = X_lgb[c].astype("category")
    p_lgb = lgb_model.predict(X_lgb)[clean]
    rmse_lgb = np.sqrt(np.mean((y_clean - np.clip(p_lgb, 60, 7200)) ** 2))
    print(f"  • LightGBM Clean RMSE: {rmse_lgb:.4f}")

    # XGBoost
    p_xgb = xgb_model.predict(X_lgb)[clean]
    rmse_xgb = np.sqrt(np.mean((y_clean - np.clip(p_xgb, 60, 7200)) ** 2))
    print(f"  • XGBoost Clean RMSE:  {rmse_xgb:.4f}")

    # Stack predictions matrix
    P = np.column_stack([p_cb, p_xgb, p_lgb])

    def loss(weights):
        w = weights / np.sum(weights)
        pred = np.clip(P @ w, 60, 7200)
        return np.sqrt(np.mean((y_clean - pred) ** 2))

    init_w = np.array([0.5, 0.3, 0.2])
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    res = minimize(loss, init_w, bounds=bounds, method="Nelder-Mead")

    opt_w = res.x / np.sum(res.x)
    print(f"\n🏆 Optimal Weights Found (Nelder-Mead):")
    print(f"  • CatBoost: {opt_w[0]:.4f} ({opt_w[0]*100:.1f}%)")
    print(f"  • XGBoost:  {opt_w[1]:.4f} ({opt_w[1]*100:.1f}%)")
    print(f"  • LightGBM: {opt_w[2]:.4f} ({opt_w[2]*100:.1f}%)")
    print(f"  • Optimal Blend RMSE: {res.fun:.4f}")

    # Test some standard predefined weights
    grid = [
        ("Equal 3-Way", [0.333, 0.333, 0.334]),
        ("CatBoost Heavy (60/25/15)", [0.60, 0.25, 0.15]),
        ("CatBoost Heavy (50/35/15)", [0.50, 0.35, 0.15]),
        ("CatBoost Heavy (70/20/10)", [0.70, 0.20, 0.10]),
        ("User Paper Ratio (86.5% CB / 13.5% XGB)", [0.865, 0.135, 0.0]),
        ("XGB Heavy (30/50/20)", [0.30, 0.50, 0.20]),
    ]
    print("\nPredefined Grid Comparison:")
    for name, w in grid:
        l = loss(np.array(w))
        print(f"  • {name:40s}: RMSE = {l:.4f}")


if __name__ == "__main__":
    main()
