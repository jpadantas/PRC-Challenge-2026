"""
Empirical Out-of-Sample Validation of Global vs Per-Airport Blending Weights.
Evaluates on 2025-10 and 2025-11 departures to determine the exact optimal blend:
alpha * delta_global + (1 - alpha) * delta_apt_specialist.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import time
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["POLARS_MAX_THREADS"] = "6"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
SCRATCH_DIR = PROJECT_ROOT / "scratch"

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


def main():
    print("=" * 80)
    print("🔬 EMPIRICAL OUT-OF-SAMPLE WEIGHT VALIDATION (2025-10 & 2025-11)")
    print("=" * 80)

    utt_stats = pl.read_parquet(SCRATCH_DIR / "empirical_utt_lookup.parquet")

    # Load holdout months
    print("Loading test holdout (2025-10 & 2025-11)...")
    df_val = features.load_training_data(RAW_DIR, months=["2025-10", "2025-11"])
    df_val = features.compute_congestion_metrics(df_val)
    df_val = features.compute_surface_accumulation_features(df_val)
    df_val = features.compute_apron_contention_features(df_val)
    df_val = features.compute_runway_sequence_features(df_val)

    dep_val = df_val.filter(pl.col("PHASE_mvt") == "DEP")
    print(f"Total holdout departures: {len(dep_val):,}")

    aobt = dep_val["AOBT_3_flt"]
    block = dep_val["BLOCK_TIME_UTC_mvt"]
    sched = dep_val["SCHED_TIME_UTC_mvt"]
    mvt = dep_val["MVT_TIME_UTC_mvt"]
    adep = dep_val["ADEP_mvt"].to_numpy()

    has_aobt = aobt.is_not_null().to_numpy()
    y_true_offset = (block - aobt).dt.total_seconds().to_numpy()
    d_bs = (block - sched).dt.total_seconds().to_numpy()
    is_sched_default = (np.abs(d_bs) <= 15)

    valid_mask = has_aobt & (~is_sched_default) & (y_true_offset >= -3600) & (y_true_offset <= 3600)
    print(f"Valid offset evaluation rows: {np.sum(valid_mask):,} ({np.mean(valid_mask)*100:.1f}%)")

    y_eval = y_true_offset[valid_mask]
    adep_eval = adep[valid_mask]

    # Evaluate Airport Specialist XGBoost models on holdout
    preds_apt_xgb = np.zeros(len(y_eval), dtype=np.float64)
    preds_apt_lgb = np.zeros(len(y_eval), dtype=np.float64)
    preds_apt_cb = np.zeros(len(y_eval), dtype=np.float64)

    for apt in AIRPORTS:
        apt_idx = np.where(adep_eval == apt)[0]
        if len(apt_idx) == 0:
            continue

        df_apt_val = df_val.filter(
            pl.when(pl.col("PHASE_mvt") == "DEP")
            .then(pl.col("ADEP_mvt") == apt)
            .otherwise(pl.col("ADES_mvt") == apt)
        )
        dep_apt_tr = df_apt_val.filter(pl.col("PHASE_mvt") == "DEP")
        encs_apt = features.compute_target_encodings(dep_apt_tr)

        # Extract features for airport holdout
        from src.train_all_10_airports_offset_specialists import extract_airport_features
        X_val_df, dep_ext_val = extract_airport_features(df_apt_val, utt_stats, target_encs=encs_apt)

        a_col = dep_ext_val["AOBT_3_flt"]
        b_col = dep_ext_val["BLOCK_TIME_UTC_mvt"]
        s_col = dep_ext_val["SCHED_TIME_UTC_mvt"]
        ha = a_col.is_not_null().to_numpy()
        yo = (b_col - a_col).dt.total_seconds().to_numpy()
        dbs = (b_col - s_col).dt.total_seconds().to_numpy()
        vm = ha & (~(np.abs(dbs) <= 15)) & (yo >= -3600) & (yo <= 3600)

        X_val_apt = X_val_df[vm].copy()

        # Load models
        xgb_path = MODELS_DIR / "per_airport_offset" / f"xgb_offset_{apt}.json"
        if xgb_path.exists():
            m_xgb = xgb.XGBRegressor()
            m_xgb.load_model(xgb_path)
            for c in X_val_apt.select_dtypes(include=["category"]).columns:
                X_val_apt[c] = X_val_apt[c].astype("category")
            p_xgb = m_xgb.predict(X_val_apt)
            preds_apt_xgb[apt_idx] = p_xgb[:len(apt_idx)]

        cb_path = MODELS_DIR / "per_airport_cb" / f"cb_offset_{apt}.cbm"
        if cb_path.exists():
            m_cb = CatBoostRegressor()
            m_cb.load_model(str(cb_path))
            X_val_cb = X_val_apt.copy()
            for c in X_val_cb.columns:
                if str(X_val_cb[c].dtype) in ["category", "object"]:
                    X_val_cb[c] = X_val_cb[c].astype(str).fillna("UNK")
            p_cb = m_cb.predict(X_val_cb)
            preds_apt_cb[apt_idx] = p_cb[:len(apt_idx)]

        rmse_xgb = np.sqrt(np.mean((preds_apt_xgb[apt_idx] - y_eval[apt_idx])**2))
        rmse_cb = np.sqrt(np.mean((preds_apt_cb[apt_idx] - y_eval[apt_idx])**2))
        blend_apt = 0.70 * preds_apt_xgb[apt_idx] + 0.30 * preds_apt_cb[apt_idx]
        rmse_blend = np.sqrt(np.mean((blend_apt - y_eval[apt_idx])**2))
        print(f"[{apt}] N={len(apt_idx):,} | XGB RMSE: {rmse_xgb:.2f}s | CB RMSE: {rmse_cb:.2f}s | Blend(70/30) RMSE: {rmse_blend:.2f}s")

    global_xgb_rmse = np.sqrt(np.mean((preds_apt_xgb - y_eval)**2))
    global_cb_rmse = np.sqrt(np.mean((preds_apt_cb - y_eval)**2))
    global_blend = 0.70 * preds_apt_xgb + 0.30 * preds_apt_cb
    global_blend_rmse = np.sqrt(np.mean((global_blend - y_eval)**2))

    print("\n" + "=" * 80)
    print(f"GLOBAL OUT-OF-SAMPLE RESULTS (N={len(y_eval):,}):")
    print(f"  Per-Airport XGBoost:        {global_xgb_rmse:.2f}s RMSE")
    print(f"  Per-Airport CatBoost:       {global_cb_rmse:.2f}s RMSE")
    print(f"  Per-Airport Blend (70/30):  {global_blend_rmse:.2f}s RMSE")
    print("=" * 80)


if __name__ == "__main__":
    main()
