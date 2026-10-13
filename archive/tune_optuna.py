"""
Optuna Bayesian Hyperparameter Optimization for Standalone Models.

Systematically searches the optimal hyperparameter configuration for:
  1. XGBoost Regressor
  2. LightGBM Regressor
Using 5-fold CV or temporal out-of-sample validation to minimize RMSE.
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
import optuna
import lightgbm as lgb
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"


def run_optuna_xgboost(season: str = "winter", n_trials: int = 20):
    print(f"\n{'=' * 70}")
    print(f"🎯 OPTUNA HYPERPARAMETER TUNING: XGBOOST ({season.upper()})")
    print(f"{'=' * 70}")
    
    # Load training month & validation month
    if season == "winter":
        train_months = ["2025-11", "2025-12", "2025-02"]
        val_month = "2025-01"
    else:
        train_months = ["2025-05", "2025-06", "2025-08"]
        val_month = "2025-07"
        
    print(f"Train months: {train_months} | Validation month: {val_month}")
    
    # Prepare Train
    df_tr = features.load_training_data(RAW_DIR, months=train_months)
    df_tr = features.compute_congestion_metrics(df_tr)
    dep_tr = df_tr.filter(pl.col("PHASE_mvt") == "DEP")
    encs = features.compute_target_encodings(dep_tr)
    X_tr_pl, _, y_tr = features.extract_features(df_tr, target_encodings=encs)
    
    X_train = X_tr_pl.to_pandas()
    for c in X_train.select_dtypes(include="category").columns:
        X_train[c] = X_train[c].astype("category")
    cols = list(X_train.columns)
    cat_dtypes = {c: X_train[c].dtype for c in X_train.select_dtypes(include="category").columns}
    
    v_tr = (y_tr >= 0) & (y_tr <= 150000)
    X_train = X_train.iloc[v_tr].reset_index(drop=True)
    y_train = y_tr[v_tr]
    
    # Prepare Val
    df_val = features.load_training_data(RAW_DIR, months=[val_month])
    df_val = features.compute_congestion_metrics(df_val)
    X_val_pl, _, y_val = features.extract_features(df_val, target_encodings=encs)
    X_val = X_val_pl.select(cols).to_pandas()
    for c, dtype in cat_dtypes.items():
        if c in X_val.columns:
            X_val[c] = pd.Categorical(X_val[c], categories=dtype.categories, ordered=dtype.ordered)
    y_val = y_val.astype(np.float64)
    
    print(f"Data ready: Train={len(X_train):,}, Val={len(X_val):,}")
    
    def objective(trial):
        params = {
            "n_estimators": 450,
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 7, 11),
            "subsample": trial.suggest_float("subsample", 0.70, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.90),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.05, 2.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 15, 60),
            "tree_method": "hist",
            "enable_categorical": True,
            "random_state": 42,
            "n_jobs": -1,
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        preds = np.clip(model.predict(X_val), 60, 15000)
        
        # Clean RMSE on validation
        clean_mask = y_val <= 7200
        rmse = np.sqrt(np.mean((y_val[clean_mask] - preds[clean_mask])**2))
        return rmse

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n🏆 Best XGBoost {season.upper()} Trial:")
    print(f"  • Best Validation Clean RMSE: {study.best_value:.4f}s")
    print(f"  • Best Parameters:")
    for k, v in study.best_params.items():
        print(f"    - {k}: {v}")
        
    return study.best_params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb", "lgb"])
    parser.add_argument("--season", type=str, default="winter", choices=["winter", "summer"])
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()
    
    if args.model == "xgb":
        run_optuna_xgboost(season=args.season, n_trials=args.trials)
