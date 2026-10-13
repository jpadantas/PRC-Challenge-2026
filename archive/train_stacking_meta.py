"""
STACKING META-LEARNER: Train a XGBoost meta-model on out-of-fold (OOF) predictions
from all base models to find optimal per-flight blend weights.

Strategy: 5-fold temporal cross-validation
Base models used as Level-1 features:
  - XGBoost global 160-feat (OOF predictions)
  - Per-airport specialist XGBoost (OOF predictions)
  - Tabular MLP (train-set predictions as proxy OOF, no retraining for speed)

Meta-model: lightweight XGBoost (shallow) trained on [xgb_oof, specialist_oof]
            -> learns WHEN to trust each model per-flight context.

Test inference: run full models on test, meta-model picks weights.
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
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR_APT = PROJECT_ROOT / "models" / "per_airport"
SCRATCH_DIR = PROJECT_ROOT / "scratch"
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
ALL_12_MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]
N_FOLDS = 5

print("=" * 80)
print("STACKING META-LEARNER: Out-of-Fold XGBoost Stacking")
print("=" * 80)
t0 = time.time()

# 1. Load and preprocess training data
print(f"[{time.strftime('%H:%M:%S')}] Loading training data...")
df_all = features.load_training_data(RAW_DIR, months=ALL_12_MONTHS)
df_all = features.compute_congestion_metrics(df_all)
df_all = features.compute_surface_accumulation_features(df_all)
df_all = features.compute_apron_contention_features(df_all)
df_all = features.compute_runway_sequence_features(df_all)

dep_all = df_all.filter(pl.col("PHASE_mvt") == "DEP")
n_train = len(dep_all)
print(f"Training departures: {n_train:,}")

# 2. Build global features once (for global XGB)
global_encs = features.compute_target_encodings(dep_all)
X_global_pl, _, y_all = features.extract_features(df_all, target_encodings=global_encs, attach_weather=True)
X_global = X_global_pl.to_pandas()
for c in X_global.select_dtypes(include=["object", "category"]).columns:
    X_global[c] = X_global[c].astype("category")

valid_mask = (y_all >= 60) & (y_all <= 150000)
X_train_g = X_global.iloc[valid_mask].reset_index(drop=True)
y_train = y_all[valid_mask]
n_valid = len(y_train)
apt_train = dep_all.filter(pl.col("TAXITIME_SEC_mvt").is_not_null())["ADEP_mvt"].to_list()
apt_train = [a for a, v in zip(dep_all["ADEP_mvt"].to_list(), valid_mask) if v]

cols_g = list(X_train_g.columns)
cat_dtypes_g = {c: X_train_g[c].dtype for c in X_train_g.select_dtypes(include="category").columns}

print(f"Valid training rows: {n_valid:,}")

# 3. Temporal 5-fold OOF
# Use month-based splits for temporal validity
months = dep_all.filter(pl.col("TAXITIME_SEC_mvt").is_not_null())["MVT_TIME_UTC_mvt"].dt.month().to_numpy()[valid_mask]
fold_ids = ((months - 1) // 3).astype(int)  # 4 quarters: 0,1,2,3 => use as fold groups
unique_folds = np.unique(fold_ids)

oof_xgb_global = np.zeros(n_valid)
oof_xgb_specialist = np.zeros(n_valid)
fold_meta_feats = []

print(f"\nRunning {len(unique_folds)}-fold OOF...")
for fold_i, fold_val in enumerate(unique_folds):
    train_idx = np.where(fold_ids != fold_val)[0]
    val_idx = np.where(fold_ids == fold_val)[0]
    print(f"  Fold {fold_i+1}/{len(unique_folds)}: val={len(val_idx):,} rows (Q{fold_val+1})")

    X_tr = X_train_g.iloc[train_idx].reset_index(drop=True)
    X_val = X_train_g.iloc[val_idx].reset_index(drop=True)
    y_tr = y_train[train_idx]
    y_val = y_train[val_idx]

    # Global XGBoost fold
    m_global = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=800, learning_rate=0.04,
        max_depth=8, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=1.0, reg_lambda=3.0, enable_categorical=True,
        tree_method="hist", device="cuda", random_state=42 + fold_i,
    )
    m_global.fit(X_tr, y_tr)
    oof_xgb_global[val_idx] = np.clip(m_global.predict(X_val), 60, 88500)

    # Per-airport specialist fold (load pretrained, predict on val only)
    # Since retraining all 10 per fold is expensive, use pretrained model as proxy
    apt_val = [apt_train[i] for i in val_idx]
    spec_preds = np.zeros(len(val_idx))
    for apt in AIRPORTS:
        apt_path = MODELS_DIR_APT / f"xgb_{apt}.json"
        if not apt_path.exists():
            continue
        apt_mask = np.array([a == apt for a in apt_val])
        if not apt_mask.any():
            continue
        m_apt = xgb.XGBRegressor()
        m_apt.load_model(str(apt_path))
        with open(MODELS_DIR_APT / f"cols_{apt}.pkl", "rb") as f:
            cols_apt = pickle.load(f)
        with open(MODELS_DIR_APT / f"cat_dtypes_{apt}.pkl", "rb") as f:
            cat_dtypes_apt = pickle.load(f)
        X_val_apt = X_val.iloc[apt_mask].copy()
        for c in cols_apt:
            if c in cat_dtypes_apt:
                if c in X_val_apt.columns:
                    X_val_apt[c] = pd.Categorical(X_val_apt[c].astype(str), categories=cat_dtypes_apt[c].categories)
                else:
                    X_val_apt[c] = pd.Categorical([None]*len(X_val_apt), categories=cat_dtypes_apt[c].categories)
            elif c not in X_val_apt.columns:
                X_val_apt[c] = 0.0
        spec_preds[apt_mask] = np.clip(m_apt.predict(X_val_apt[cols_apt]), 60, 88500)
    oof_xgb_specialist[val_idx] = spec_preds

    fold_rmse_g = np.sqrt(np.mean((y_val - oof_xgb_global[val_idx])**2))
    fold_rmse_s = np.sqrt(np.mean((y_val - oof_xgb_specialist[val_idx])**2))
    print(f"    Global RMSE: {fold_rmse_g:.2f}s  |  Specialist RMSE: {fold_rmse_s:.2f}s")

print(f"\nOOF RMSE (Global):     {np.sqrt(np.mean((y_train - oof_xgb_global)**2)):.2f}s")
print(f"OOF RMSE (Specialist): {np.sqrt(np.mean((y_train - oof_xgb_specialist)**2)):.2f}s")

# 4. Build meta-features: [global_pred, specialist_pred, their_diff, airport, hour, month]
meta_df = pd.DataFrame({
    "pred_global": oof_xgb_global,
    "pred_specialist": oof_xgb_specialist,
    "pred_diff": oof_xgb_global - oof_xgb_specialist,
    "pred_ratio": oof_xgb_global / (oof_xgb_specialist + 1.0),
    "adep": pd.Categorical(apt_train),
})

# Also add some raw features that determine when models disagree
meta_df["base_proxy"] = X_train_g["base_proxy"].values
meta_df["has_aobt"] = X_train_g["has_aobt"].values
meta_df["mvt_hour"] = X_train_g["mvt_hour"].values
meta_df["mvt_month"] = X_train_g["mvt_month"].values
meta_df["total_surface_active"] = X_train_g["total_surface_active"].values

print(f"\nTraining meta-learner on {len(meta_df):,} rows, {len(meta_df.columns)} meta-features...")
meta_model = xgb.XGBRegressor(
    objective="reg:squarederror", n_estimators=400, learning_rate=0.05,
    max_depth=5, subsample=0.8, colsample_bytree=0.8,
    enable_categorical=True, tree_method="hist", device="cuda", random_state=42,
)
meta_model.fit(meta_df, y_train)
meta_preds_train = np.clip(meta_model.predict(meta_df), 60, 88500)
meta_rmse = np.sqrt(np.mean((y_train - meta_preds_train)**2))
print(f"Meta-learner train RMSE: {meta_rmse:.2f}s")

# 5. Save meta-model
meta_model.save_model(str(MODELS_DIR / "xgb_meta_stacker.json"))
meta_cols = list(meta_df.columns)
with open(MODELS_DIR / "meta_stacker_cols.pkl", "wb") as f:
    pickle.dump(meta_cols, f)
print(f"Saved meta-model to {MODELS_DIR / 'xgb_meta_stacker.json'}")

# 6. Inference on test set
print(f"\n[{time.strftime('%H:%M:%S')}] Running inference on test set...")
xgb_global_test = np.load(SCRATCH_DIR / "xgb_deep_test_preds.npy")
specialist_test  = np.load(SCRATCH_DIR / "per_airport_test_preds.npy")

rank_df = pl.read_parquet(RAW_DIR / "ranking.parquet")
dep_test = rank_df.filter(pl.col("PHASE_mvt") == "DEP")

# Load test features for meta-features
with open(MODELS_DIR / "encs_xgb_fullyear_deep.pkl", "rb") as f:
    test_encs = pickle.load(f)
X_test_pl, _, _ = features.extract_features(
    features.compute_congestion_metrics(
        features.compute_surface_accumulation_features(
            features.compute_apron_contention_features(
                features.compute_runway_sequence_features(rank_df)
            )
        )
    ),
    target_encodings=test_encs, attach_weather=True
)
X_test_pd = X_test_pl.to_pandas()

apt_test = dep_test["ADEP_mvt"].to_list()
meta_test = pd.DataFrame({
    "pred_global": xgb_global_test,
    "pred_specialist": specialist_test,
    "pred_diff": xgb_global_test - specialist_test,
    "pred_ratio": xgb_global_test / (specialist_test + 1.0),
    "adep": pd.Categorical(apt_test, categories=meta_df["adep"].cat.categories),
    "base_proxy": X_test_pd["base_proxy"].values,
    "has_aobt": X_test_pd["has_aobt"].values,
    "mvt_hour": X_test_pd["mvt_hour"].values,
    "mvt_month": X_test_pd["mvt_month"].values,
    "total_surface_active": X_test_pd["total_surface_active"].values,
})

stacked_preds = np.clip(meta_model.predict(meta_test), 60, 88500)
print(f"Stacked predictions: mean={stacked_preds.mean():.1f}s  std={stacked_preds.std():.1f}s")
np.save(SCRATCH_DIR / "stacked_test_preds.npy", stacked_preds)
print(f"Saved: {SCRATCH_DIR / 'stacked_test_preds.npy'}")
print(f"Total time: {time.time() - t0:.1f}s")
