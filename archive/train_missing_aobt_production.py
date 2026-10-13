"""
Production Training Pipeline for Missing-AOBT Departure Specialist.
Trains dual GBDT (LightGBM + GPU XGBoost) exclusively on departures lacking radar pushback (AOBT).
Predicts for the 6,200 missing AOBT test flights in ranking.parquet.
Saves predictions to scratch/missing_aobt_test_preds.npy.
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

# Thermal safety
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
    print("🚀 PRODUCTION TRAINING: MISSING-AOBT DEPARTURE SPECIALIST")
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

    # Drop columns dependent on AOBT
    drop_cols = [c for c in X_df_tr.columns if "aobt" in c.lower() or "d_ai" in c.lower() or "d_al" in c.lower()]
    X_tr_clean = X_df_tr.drop(columns=drop_cols)

    for c in X_tr_clean.select_dtypes(include=["object"]).columns:
        X_tr_clean[c] = X_tr_clean[c].astype("category")

    X_train = X_tr_clean[mask_miss_train].reset_index(drop=True)
    y_train = y_tr[mask_miss_train]

    # 2. Train LightGBM Missing Specialist
    print(f"\n[{time.strftime('%H:%M:%S')}] 1/2 Training LightGBM Missing-AOBT Specialist...")
    lgb_miss = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=45,
        subsample=0.85,
        colsample_bytree=0.80,
        random_state=42,
        n_jobs=6,
        verbose=-1
    )
    lgb_miss.fit(X_train, y_train)
    with open(MODELS_DIR / "lgb_missing_aobt.pkl", "wb") as f:
        pickle.dump(lgb_miss, f)

    # 3. Train XGBoost GPU Missing Specialist
    print(f"\n[{time.strftime('%H:%M:%S')}] 2/2 Training XGBoost GPU Missing-AOBT Specialist...")
    xgb_miss = xgb.XGBRegressor(
        tree_method="hist",
        device="cuda",
        n_estimators=700,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.80,
        enable_categorical=True,
        random_state=42
    )
    xgb_miss.fit(X_train, y_train, verbose=False)
    xgb_miss.save_model(MODELS_DIR / "xgb_missing_aobt.json")

    # 4. Predict on ranking test set
    print(f"\n[{time.strftime('%H:%M:%S')}] Predicting on ranking test set for missing flights...")
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

    print(f"Ranking test missing departures: {np.sum(is_miss_te):,} flights.")

    X_pl_te, _, _ = features.extract_features(dep_test, target_encodings=encs, attach_weather=True)
    dep_ext_te = dep_test.join(utt_stats, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")

    X_df_te = X_pl_te.to_pandas()
    X_df_te["full_year_utt_p10"] = dep_ext_te["utt_p10"].fill_null(600.0).to_numpy()
    X_df_te["full_year_utt_p25"] = dep_ext_te["utt_p25"].fill_null(750.0).to_numpy()
    X_df_te["full_year_utt_p50"] = dep_ext_te["utt_p50"].fill_null(950.0).to_numpy()
    X_te_clean = X_df_te.drop(columns=drop_cols)

    for c in X_train.select_dtypes(include=["category"]).columns:
        cats = list(X_train[c].cat.categories)
        if c in X_te_clean.columns:
            X_te_clean[c] = pd.Categorical(X_te_clean[c], categories=cats)

    p_lgb = lgb_miss.predict(X_te_clean)
    p_xgb = xgb_miss.predict(X_te_clean)
    p_ensemble = 0.50 * p_lgb + 0.50 * p_xgb
    p_ensemble = np.clip(np.round(p_ensemble), 180, 4500).astype(np.float64)

    out_file = SCRATCH_DIR / "missing_aobt_test_preds.npy"
    np.save(out_file, p_ensemble)
    print(f"💾 Saved missing AOBT predictions: {out_file}")
    print(f"Stats on missing test flights: N={np.sum(is_miss_te):,}, Mean={np.mean(p_ensemble[is_miss_te]):.2f}s, Median={np.median(p_ensemble[is_miss_te]):.0f}s, Min={np.min(p_ensemble[is_miss_te])}, Max={np.max(p_ensemble[is_miss_te])}")
    print(f"🎉 Production Missing AOBT pipeline completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
