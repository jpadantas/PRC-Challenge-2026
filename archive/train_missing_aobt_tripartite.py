"""
Tripartite Ensemble for Missing-AOBT Departures (LightGBM + XGBoost GPU + CatBoost GPU).
Trains on all 2025 missing AOBT departures with 5-fold cross validation.
Generates ultra-low variance predictions for the 6,200 missing AOBT flights in ranking.parquet.
Saves to scratch/missing_aobt_test_preds.npy.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import time
import pickle
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["POLARS_MAX_THREADS"] = "6"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
SCRATCH_DIR = PROJECT_ROOT / "scratch"
ALL_12_MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]


def main():
    t0 = time.time()
    print("=" * 80)
    print("🚀 TRIPARTITE ENSEMBLE: MISSING-AOBT SPECIALIST (LGB + XGB + CB)")
    print("=" * 80)

    utt_stats = pl.read_parquet(SCRATCH_DIR / "empirical_utt_lookup.parquet")

    # 1. Load full training data
    print(f"[{time.strftime('%H:%M:%S')}] Loading all 12 months of training data...")
    df_train = features.load_training_data(RAW_DIR, months=ALL_12_MONTHS)
    print(f"Loaded {len(df_train):,} total rows in {time.time() - t0:.1f}s")

    print(f"[{time.strftime('%H:%M:%S')}] Computing domain features...")
    df_train = features.compute_congestion_metrics(df_train)
    df_train = features.compute_surface_accumulation_features(df_train)
    df_train = features.compute_apron_contention_features(df_train)
    df_train = features.compute_runway_sequence_features(df_train)

    dep_train = df_train.filter(pl.col("PHASE_mvt") == "DEP")
    aobt_tr = dep_train["AOBT_3_flt"]
    mvt_tr = dep_train["MVT_TIME_UTC_mvt"]
    da_tr = (mvt_tr - aobt_tr).dt.total_seconds().to_numpy()
    is_miss_tr = np.isnan(da_tr) | (da_tr < 120) | (da_tr > 7200)

    y_tr = dep_train["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    valid_taxi = (y_tr >= 60) & (y_tr <= 7200)
    mask_miss_train = is_miss_tr & valid_taxi

    print(f"Training set: {np.sum(mask_miss_train):,} usable missing-AOBT rows.")

    encs = features.compute_target_encodings(dep_train)
    X_pl_tr, _, _ = features.extract_features(dep_train, target_encodings=encs, attach_weather=True)
    dep_ext_tr = dep_train.join(utt_stats, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")

    X_df_tr = X_pl_tr.to_pandas()
    X_df_tr["full_year_utt_p10"] = dep_ext_tr["utt_p10"].fill_null(600.0).to_numpy()
    X_df_tr["full_year_utt_p25"] = dep_ext_tr["utt_p25"].fill_null(750.0).to_numpy()
    X_df_tr["full_year_utt_p50"] = dep_ext_tr["utt_p50"].fill_null(950.0).to_numpy()

    drop_cols = [c for c in X_df_tr.columns if "aobt" in c.lower() or "d_ai" in c.lower() or "d_al" in c.lower()]
    X_tr_clean = X_df_tr.drop(columns=drop_cols)

    for c in X_tr_clean.select_dtypes(include=["object"]).columns:
        X_tr_clean[c] = X_tr_clean[c].astype("category")

    X_train = X_tr_clean[mask_miss_train].reset_index(drop=True)
    y_train = y_tr[mask_miss_train]

    # 2. 5-Fold Cross Validation for Exact Meta-Weighting
    print(f"\n[{time.strftime('%H:%M:%S')}] Running 5-fold cross validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(y_train))
    oof_xgb = np.zeros(len(y_train))
    oof_cb = np.zeros(len(y_train))

    cat_cols = list(X_train.select_dtypes(include=["category"]).columns)

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr_f, y_tr_f = X_train.iloc[trn_idx], y_train[trn_idx]
        X_va_f, y_va_f = X_train.iloc[val_idx], y_train[val_idx]

        # LGB
        m_lgb = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators=600,
            learning_rate=0.035,
            num_leaves=45,
            subsample=0.85,
            colsample_bytree=0.80,
            random_state=42 + fold,
            n_jobs=6,
            verbose=-1,
        )
        m_lgb.fit(X_tr_f, y_tr_f)
        oof_lgb[val_idx] = m_lgb.predict(X_va_f)

        # XGB
        m_xgb = xgb.XGBRegressor(
            tree_method="hist",
            device="cuda",
            n_estimators=600,
            learning_rate=0.035,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.80,
            enable_categorical=True,
            random_state=42 + fold,
        )
        m_xgb.fit(X_tr_f, y_tr_f, verbose=False)
        oof_xgb[val_idx] = m_xgb.predict(X_va_f)

        # CB
        X_tr_cb = X_tr_f.copy()
        X_va_cb = X_va_f.copy()
        for c in cat_cols:
            X_tr_cb[c] = X_tr_cb[c].astype(str).fillna("UNK")
            X_va_cb[c] = X_va_cb[c].astype(str).fillna("UNK")

        m_cb = CatBoostRegressor(
            iterations=600,
            learning_rate=0.04,
            depth=7,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42 + fold,
            task_type="GPU",
            verbose=False,
            cat_features=cat_cols,
        )
        m_cb.fit(X_tr_cb, y_tr_f)
        oof_cb[val_idx] = m_cb.predict(X_va_cb)

    rmse_lgb = np.sqrt(np.mean((oof_lgb - y_train)**2))
    rmse_xgb = np.sqrt(np.mean((oof_xgb - y_train)**2))
    rmse_cb = np.sqrt(np.mean((oof_cb - y_train)**2))

    print(f"5-Fold CV OOF Results:")
    print(f"  LightGBM OOF RMSE: {rmse_lgb:.2f}s")
    print(f"  XGBoost  OOF RMSE: {rmse_xgb:.2f}s")
    print(f"  CatBoost OOF RMSE: {rmse_cb:.2f}s")

    # Optimal weights via Ridge regression on OOF
    oof_matrix = np.vstack([oof_lgb, oof_xgb, oof_cb]).T
    from scipy.optimize import minimize

    def obj(w):
        return np.mean((oof_matrix @ w - y_train)**2)

    res = minimize(obj, [0.35, 0.45, 0.20], bounds=[(0, 1), (0, 1), (0, 1)], constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    w_opt = res.x
    rmse_ens = np.sqrt(res.fun)
    print(f"🎯 Optimal OOF Ensemble: {w_opt[0]:.3f} LGB + {w_opt[1]:.3f} XGB + {w_opt[2]:.3f} CB")
    print(f"   Ensemble OOF RMSE: {rmse_ens:.2f}s (gain vs best single: {min(rmse_lgb, rmse_xgb, rmse_cb) - rmse_ens:+.2f}s)")

    # 3. Fit final models on 100% data
    print(f"\n[{time.strftime('%H:%M:%S')}] Retraining final models on 100% missing data...")
    m_lgb_final = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=750,
        learning_rate=0.03,
        num_leaves=45,
        subsample=0.85,
        colsample_bytree=0.80,
        random_state=42,
        n_jobs=6,
        verbose=-1,
    )
    m_lgb_final.fit(X_train, y_train)

    m_xgb_final = xgb.XGBRegressor(
        tree_method="hist",
        device="cuda",
        n_estimators=750,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.80,
        enable_categorical=True,
        random_state=42,
    )
    m_xgb_final.fit(X_train, y_train, verbose=False)

    X_train_cb = X_train.copy()
    for c in cat_cols:
        X_train_cb[c] = X_train_cb[c].astype(str).fillna("UNK")

    m_cb_final = CatBoostRegressor(
        iterations=750,
        learning_rate=0.035,
        depth=7,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        task_type="GPU",
        verbose=False,
        cat_features=cat_cols,
    )
    m_cb_final.fit(X_train_cb, y_train)

    # 4. Predict on ranking test set
    print(f"\n[{time.strftime('%H:%M:%S')}] Predicting on test ranking flights...")
    df_test = pl.read_parquet(RAW_DIR / "ranking.parquet")
    df_test = features.compute_congestion_metrics(df_test)
    df_test = features.compute_surface_accumulation_features(df_test)
    df_test = features.compute_apron_contention_features(df_test)
    df_test = features.compute_runway_sequence_features(df_test)

    dep_test = df_test.filter(pl.col("PHASE_mvt") == "DEP")
    aobt_te = dep_test["AOBT_3_flt"]
    mvt_te = dep_test["MVT_TIME_UTC_mvt"]
    da_te = (mvt_te - aobt_te).dt.total_seconds().to_numpy()
    is_miss_te = np.isnan(da_te) | (da_te < 120) | (da_te > 7200)

    X_pl_te, _, _ = features.extract_features(dep_test, target_encodings=encs, attach_weather=True)
    dep_ext_te = dep_test.join(utt_stats, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")

    X_df_te = X_pl_te.to_pandas()
    X_df_te["full_year_utt_p10"] = dep_ext_te["utt_p10"].fill_null(600.0).to_numpy()
    X_df_te["full_year_utt_p25"] = dep_ext_te["utt_p25"].fill_null(750.0).to_numpy()
    X_df_te["full_year_utt_p50"] = dep_ext_te["utt_p50"].fill_null(950.0).to_numpy()
    X_te_clean = X_df_te.drop(columns=drop_cols)

    for c in cat_cols:
        cats = list(X_train[c].cat.categories)
        if c in X_te_clean.columns:
            X_te_clean[c] = pd.Categorical(X_te_clean[c], categories=cats)

    p_lgb_te = m_lgb_final.predict(X_te_clean)
    p_xgb_te = m_xgb_final.predict(X_te_clean)

    X_te_cb = X_te_clean.copy()
    for c in cat_cols:
        X_te_cb[c] = X_te_cb[c].astype(str).fillna("UNK")
    p_cb_te = m_cb_final.predict(X_te_cb)

    p_tripartite = w_opt[0] * p_lgb_te + w_opt[1] * p_xgb_te + w_opt[2] * p_cb_te
    p_tripartite = np.clip(np.round(p_tripartite), 180, 4500).astype(np.float64)

    out_file = SCRATCH_DIR / "missing_aobt_test_preds.npy"
    np.save(out_file, p_tripartite)
    print(f"💾 Updated {out_file} with tripartite missing-AOBT specialist predictions!")
    print(f"Stats on missing test flights: N={np.sum(is_miss_te):,}, Mean={np.mean(p_tripartite[is_miss_te]):.2f}s, Median={np.median(p_tripartite[is_miss_te]):.0f}s, Min={np.min(p_tripartite[is_miss_te])}, Max={np.max(p_tripartite[is_miss_te])}")
    print(f"🎉 Tripartite Missing-AOBT pipeline completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
