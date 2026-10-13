"""
Train 10 Dedicated Airport-Specialist CatBoost GPU Offset Models for PRC Data Challenge 2026.
Airports: EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM.

Strategy:
Symmetric (oblivious) decision trees trained on GPU with dedicated airport capacity.
CatBoost's oblivious trees provide regularized, low-variance predictions that complement
XGBoost's depth-wise and LightGBM's leaf-wise trees.
Saves models to models/per_airport_cb/ and predictions to scratch/cb_per_airport_offset_test_preds.npy.
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
from catboost import CatBoostRegressor

# Thermal safety
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["POLARS_MAX_THREADS"] = "6"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models" / "per_airport_cb"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_DIR = PROJECT_ROOT / "scratch"

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
ALL_12_MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]


def extract_airport_features(df_apt: pl.DataFrame, utt_stats: pl.DataFrame, target_encs: dict | None = None) -> tuple[pd.DataFrame, pl.DataFrame]:
    X_pl, _, _ = features.extract_features(df_apt, target_encodings=target_encs, attach_weather=True)
    dep = df_apt.filter(pl.col("PHASE_mvt") == "DEP")
    dep = dep.join(utt_stats, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")

    aobt = dep["AOBT_3_flt"]
    sched = dep["SCHED_TIME_UTC_mvt"]
    iobt = dep["IOBT_flt"]
    lobt = dep["LOBT_flt"]
    eobt = dep["EOBT_1_flt"]

    d_ai = (aobt - iobt).dt.total_seconds().clip(-7200, 14400).fill_null(0.0).to_numpy()
    d_al = (aobt - lobt).dt.total_seconds().clip(-7200, 14400).fill_null(0.0).to_numpy()
    d_li = (lobt - iobt).dt.total_seconds().clip(-3600, 7200).fill_null(0.0).to_numpy()
    d_ei = (eobt - iobt).dt.total_seconds().clip(-3600, 7200).fill_null(0.0).to_numpy()
    d_si = (sched - iobt).dt.total_seconds().clip(-7200, 14400).fill_null(0.0).to_numpy()

    utt_p10 = dep["utt_p10"].fill_null(600.0).to_numpy()
    utt_p25 = dep["utt_p25"].fill_null(750.0).to_numpy()
    utt_p50 = dep["utt_p50"].fill_null(950.0).to_numpy()

    X_df = X_pl.to_pandas()
    X_df["tactical_d_ai"] = d_ai
    X_df["tactical_d_al"] = d_al
    X_df["tactical_d_li"] = d_li
    X_df["tactical_d_ei"] = d_ei
    X_df["tactical_d_si"] = d_si
    X_df["full_year_utt_p10"] = utt_p10
    X_df["full_year_utt_p25"] = utt_p25
    X_df["full_year_utt_p50"] = utt_p50
    X_df["mkt_segment"] = dep["MARKET_SEGMENT_flt"].to_pandas().astype(str).fillna("UNK")
    X_df["flight_type"] = dep["FLIGHT_TYPE_flt"].to_pandas().astype(str).fillna("UNK")
    X_df["wk_tbl_cat"] = dep["WK_TBL_CAT_flt"].to_pandas().astype(str).fillna("UNK")

    for c in X_df.columns:
        if str(X_df[c].dtype) in ["category", "object"]:
            X_df[c] = X_df[c].astype(str).fillna("UNK")

    return X_df, dep


def main():
    t_global = time.time()
    print("=" * 80)
    print("🚀 TRAINING 10 PER-AIRPORT SPECIALIZED CATBOOST GPU OFFSET MODELS")
    print("=" * 80)

    utt_stats = pl.read_parquet(SCRATCH_DIR / "empirical_utt_lookup.parquet")

    # 1. Load full ranking test set once
    print(f"[{time.strftime('%H:%M:%S')}] Loading ranking test set...")
    df_test = pl.read_parquet(RAW_DIR / "ranking.parquet")
    df_test = features.compute_congestion_metrics(df_test)
    df_test = features.compute_surface_accumulation_features(df_test)
    df_test = features.compute_apron_contention_features(df_test)
    df_test = features.compute_runway_sequence_features(df_test)

    rank_dep = df_test.filter(pl.col("PHASE_mvt") == "DEP")
    adep_test = rank_dep["ADEP_mvt"].to_numpy()
    n_test = len(rank_dep)
    p_specialist_full = np.zeros(n_test, dtype=np.float64)

    # 2. Load all 12 months of training data once
    print(f"[{time.strftime('%H:%M:%S')}] Loading all 12 months of 2025 training data...")
    df_all = features.load_training_data(RAW_DIR, months=ALL_12_MONTHS)
    print(f"Loaded {len(df_all):,} rows in {time.time() - t_global:.1f}s")

    print(f"[{time.strftime('%H:%M:%S')}] Precomputing global training domain features...")
    df_all = features.compute_congestion_metrics(df_all)
    df_all = features.compute_surface_accumulation_features(df_all)
    df_all = features.compute_apron_contention_features(df_all)
    df_all = features.compute_runway_sequence_features(df_all)

    # 3. Train each airport
    for apt in AIRPORTS:
        t_apt = time.time()
        print(f"\n{'='*70}")
        print(f"🐱 CATBOOST GPU AIRPORT SPECIALIST: {apt}")
        print(f"{'='*70}")

        # Filter training data for this airport
        df_apt_tr = df_all.filter(
            pl.when(pl.col("PHASE_mvt") == "DEP")
            .then(pl.col("ADEP_mvt") == apt)
            .otherwise(pl.col("ADES_mvt") == apt)
        )
        dep_tr = df_apt_tr.filter(pl.col("PHASE_mvt") == "DEP")

        # Airport-specific target encodings
        encs_apt = features.compute_target_encodings(dep_tr)

        # Extract features
        X_tr_df, dep_ext_tr = extract_airport_features(df_apt_tr, utt_stats, target_encs=encs_apt)

        aobt_col = dep_ext_tr["AOBT_3_flt"]
        block_col = dep_ext_tr["BLOCK_TIME_UTC_mvt"]
        sched_col = dep_ext_tr["SCHED_TIME_UTC_mvt"]

        has_aobt = aobt_col.is_not_null().to_numpy()
        y_offset = (block_col - aobt_col).dt.total_seconds().to_numpy()
        d_bs = (block_col - sched_col).dt.total_seconds().to_numpy()
        is_sched_default = (np.abs(d_bs) <= 15)

        valid_mask = has_aobt & (~is_sched_default) & (y_offset >= -3600) & (y_offset <= 3600)
        X_tr = X_tr_df[valid_mask].reset_index(drop=True)
        y_tr = y_offset[valid_mask]

        print(f"[{apt}] Train rows: {len(X_tr):,} | Target std: {np.std(y_tr):.2f}s | Mean: {np.mean(y_tr):.2f}s")

        # Filter test set for this airport
        test_mask_apt = (adep_test == apt)
        df_apt_te = df_test.filter(
            pl.when(pl.col("PHASE_mvt") == "DEP")
            .then(pl.col("ADEP_mvt") == apt)
            .otherwise(pl.col("ADES_mvt") == apt)
        )
        X_te_df, _ = extract_airport_features(df_apt_te, utt_stats, target_encs=encs_apt)
        X_te = X_te_df.copy().reset_index(drop=True)
        assert len(X_te) == np.sum(test_mask_apt), f"Mismatch: {len(X_te)} vs {np.sum(test_mask_apt)}"

        cat_cols = [c for c in X_tr.columns if str(X_tr[c].dtype) in ["category", "object"]]

        # Train CatBoost on GPU
        model = CatBoostRegressor(
            iterations=900,
            learning_rate=0.04,
            depth=7,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            task_type="GPU",
            verbose=False,
            cat_features=cat_cols,
        )
        model.fit(X_tr, y_tr)
        model.save_model(str(MODELS_DIR / f"cb_offset_{apt}.cbm"))

        preds_apt = model.predict(X_te)
        p_specialist_full[test_mask_apt] = preds_apt

        print(f"[{apt}] Trained in {time.time() - t_apt:.1f}s | Test predictions: N={len(preds_apt):,}, Mean={np.mean(preds_apt):.2f}s, Std={np.std(preds_apt):.2f}s")

    # Save full predictions array
    out_npy = SCRATCH_DIR / "cb_per_airport_offset_test_preds.npy"
    np.save(out_npy, p_specialist_full)
    print(f"\n💾 Saved all-airport CatBoost specialist test predictions to {out_npy}")
    print(f"Global specialist predictions: Mean={np.mean(p_specialist_full):.2f}s, Std={np.std(p_specialist_full):.2f}s")
    print(f"🎉 All 10 CatBoost Airport Specialists completed in {time.time() - t_global:.1f}s")


if __name__ == "__main__":
    main()
