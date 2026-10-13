"""
Evaluate optimal blending weight between Global Winter Offset and Airport Specialist
on 500,714 out-of-sample flights from 2025-10 and 2025-11.
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
from scipy.optimize import minimize_scalar

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
    print("🔬 OPTIMAL GLOBAL VS PER-AIRPORT WEIGHT OPTIMIZATION (500,714 OOS FLIGHTS)")
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

    aobt = dep_val["AOBT_3_flt"]
    block = dep_val["BLOCK_TIME_UTC_mvt"]
    sched = dep_val["SCHED_TIME_UTC_mvt"]
    adep = dep_val["ADEP_mvt"].to_numpy()

    has_aobt = aobt.is_not_null().to_numpy()
    y_true_offset = (block - aobt).dt.total_seconds().to_numpy()
    d_bs = (block - sched).dt.total_seconds().to_numpy()
    is_sched_default = (np.abs(d_bs) <= 15)

    valid_mask = has_aobt & (~is_sched_default) & (y_true_offset >= -3600) & (y_true_offset <= 3600)
    y_eval = y_true_offset[valid_mask]
    adep_eval = adep[valid_mask]
    N = len(y_eval)
    print(f"Total evaluation rows: {N:,}")

    # 1. Evaluate Airport Specialists
    preds_apt = np.zeros(N, dtype=np.float64)
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
        xgb_path = MODELS_DIR / "per_airport_offset" / f"xgb_offset_{apt}.json"
        m_xgb = xgb.XGBRegressor()
        m_xgb.load_model(xgb_path)
        for c in X_val_apt.select_dtypes(include=["category"]).columns:
            X_val_apt[c] = X_val_apt[c].astype("category")
        preds_apt[apt_idx] = m_xgb.predict(X_val_apt)[:len(apt_idx)]

    rmse_apt = np.sqrt(np.mean((preds_apt - y_eval)**2))
    print(f"Per-Airport Specialists Baseline RMSE: {rmse_apt:.2f}s")

    # 2. Evaluate Global Winter Offset model
    print("\nEvaluating Global Winter Offset model...")
    from src.train_dual_season_offset import extract_advanced_matrix
    encs_global = features.compute_target_encodings(dep_val)
    X_global, _ = extract_advanced_matrix(df_val, utt_stats, target_encs=encs_global)
    for c in X_global.select_dtypes(include=["object", "category"]).columns:
        X_global[c] = X_global[c].astype("category")

    X_global_eval = X_global.iloc[valid_mask].copy()

    m_global = xgb.XGBRegressor()
    m_global.load_model(MODELS_DIR / "xgb_winter_offset.json")
    preds_global = m_global.predict(X_global_eval)

    rmse_global = np.sqrt(np.mean((preds_global - y_eval)**2))
    print(f"Global Winter Offset RMSE: {rmse_global:.2f}s")

    # 3. Find optimal alpha
    print("\n" + "=" * 80)
    print("CONVEX COMBINATION SEARCH: alpha * Global + (1 - alpha) * Airport_Specialist")
    print("=" * 80)

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        blend = alpha * preds_global + (1.0 - alpha) * preds_apt
        rmse = np.sqrt(np.mean((blend - y_eval)**2))
        print(f"  alpha={alpha:.1f} (Global {alpha*100:3.0f}% / Airport {(1-alpha)*100:3.0f}%): RMSE = {rmse:.2f}s")

    def loss(a):
        b = a * preds_global + (1.0 - a) * preds_apt
        return np.mean((b - y_eval)**2)

    res = minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded")
    alpha_star = res.x
    best_rmse = np.sqrt(res.fun)
    print(f"\n🎯 EXACT OPTIMAL BLEND: alpha* = {alpha_star:.4f} (Global {alpha_star*100:.1f}% / Airport {(1-alpha_star)*100:.1f}%)")
    print(f"   Optimal RMSE: {best_rmse:.2f}s (gain vs pure airport: {rmse_apt - best_rmse:+.2f}s, vs pure global: {rmse_global - best_rmse:+.2f}s)")
    print("=" * 80)


if __name__ == "__main__":
    main()
