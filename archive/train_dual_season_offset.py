"""
Dual-Season Specialized Offset Training Engine for PRC Data Challenge 2026.
Decouples Winter (Jan 2026 test partition) and Summer (Jul 2026 test partition).
Incorporates:
1. Interleaved arrival features (dt_prev_any_mvt, is_prev_mvt_arr)
2. Tactical flight plan milestones (AOBT - IOBT, LOBT - IOBT, etc.)
3. Full-year empirical UTT matrix (2.08M flights)
4. Tripartite GBDT suite per season:
   - GPU Hist XGBoost (device='cuda')
   - Leaf-wise LightGBM
   - Symmetric CatBoost
5. Physical clamping and safety against gate delay inflation.
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
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Enforce thermal throttling
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["POLARS_MAX_THREADS"] = "6"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
SCRATCH_DIR = PROJECT_ROOT / "scratch"
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"
TEAM_NAME = "inspiring-kitten"

WINTER_MONTHS = ["2025-01", "2025-02", "2025-11", "2025-12"]
SUMMER_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08"]


def extract_advanced_matrix(df: pl.DataFrame, utt_stats: pl.DataFrame, target_encs: dict | None = None) -> tuple[pd.DataFrame, pl.DataFrame]:
    """Extracts unified 197-feature matrix with arrival interleaving, tactical milestones, and empirical UTT."""
    X_pl, _, _ = features.extract_features(df, target_encodings=target_encs, attach_weather=True)
    dep = df.filter(pl.col("PHASE_mvt") == "DEP")

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
    X_df["mkt_segment"] = dep["MARKET_SEGMENT_flt"].to_pandas().astype("category")
    X_df["flight_type"] = dep["FLIGHT_TYPE_flt"].to_pandas().astype("category")
    X_df["wk_tbl_cat"] = dep["WK_TBL_CAT_flt"].to_pandas().astype("category")

    for c in X_df.select_dtypes(include=["object"]).columns:
        X_df[c] = X_df[c].astype("category")

    return X_df, dep


def train_season_suite(season_name: str, months: list[str], utt_stats: pl.DataFrame, test_df: pl.DataFrame, test_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("\n" + "=" * 80)
    print(f"❄️ / ☀️ TRAINING {season_name.upper()} SUITE ({', '.join(months)})")
    print("=" * 80)

    # 1. Load data
    t0 = time.time()
    df_season = features.load_training_data(RAW_DIR, months=months)
    print(f"Loaded {len(df_season):,} rows for {season_name} in {time.time() - t0:.1f}s")

    print(f"[{time.strftime('%H:%M:%S')}] Computing domain features...")
    df_season = features.compute_congestion_metrics(df_season)
    df_season = features.compute_surface_accumulation_features(df_season)
    df_season = features.compute_apron_contention_features(df_season)
    df_season = features.compute_runway_sequence_features(df_season)

    dep_season = df_season.filter(pl.col("PHASE_mvt") == "DEP")
    print(f"Computing {season_name} target encodings across {len(dep_season):,} departures...")
    encs = features.compute_target_encodings(dep_season)

    print(f"[{time.strftime('%H:%M:%S')}] Extracting {season_name} feature matrix...")
    X_train_df, dep_train_ext = extract_advanced_matrix(df_season, utt_stats, target_encs=encs)

    # Target: delta = BLOCK - AOBT
    aobt_col = dep_train_ext["AOBT_3_flt"]
    block_col = dep_train_ext["BLOCK_TIME_UTC_mvt"]
    sched_col = dep_train_ext["SCHED_TIME_UTC_mvt"]
    adep_col = dep_train_ext["ADEP_mvt"].to_numpy()

    has_aobt = aobt_col.is_not_null().to_numpy()
    y_offset = (block_col - aobt_col).dt.total_seconds().to_numpy()
    d_bs = (block_col - sched_col).dt.total_seconds().to_numpy()
    is_lirf_default = (adep_col == "LIRF") & (np.abs(d_bs) <= 15)

    valid_mask = has_aobt & (~is_lirf_default) & (y_offset >= -3600) & (y_offset <= 3600)
    X_tr = X_train_df[valid_mask]
    y_tr = y_offset[valid_mask]
    print(f"{season_name} Train Matrix: {len(X_tr):,} rows, {X_tr.shape[1]} features.")
    print(f"Target offset std: {np.std(y_tr):.2f}s, mean: {np.mean(y_tr):.2f}s")

    # Extract test features with this season's encodings
    print(f"[{time.strftime('%H:%M:%S')}] Extracting test features for {season_name} partition ({np.sum(test_mask):,} rows)...")
    X_test_all, _ = extract_advanced_matrix(test_df, utt_stats, target_encs=encs)
    X_test_season = X_test_all[test_mask].copy()

    # Align columns
    for c in X_tr.select_dtypes(include=["category"]).columns:
        cats = list(X_tr[c].cat.categories)
        if c in X_test_season.columns:
            X_test_season[c] = pd.Categorical(X_test_season[c], categories=cats)

    # 1. XGBoost GPU
    print(f"\n[{time.strftime('%H:%M:%S')}] 1/3 Training {season_name} XGBoost GPU...")
    xgb_params = {
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": 0.04,
        "max_depth": 10,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "n_estimators": 950,
        "random_state": 42,
        "enable_categorical": True,
    }
    m_xgb = xgb.XGBRegressor(**xgb_params)
    t_start = time.time()
    m_xgb.fit(X_tr, y_tr, verbose=200)
    print(f"{season_name} XGBoost trained in {time.time() - t_start:.1f}s")
    m_xgb.save_model(MODELS_DIR / f"xgb_{season_name}_offset.json")
    p_xgb_test = m_xgb.predict(X_test_season)

    # 2. LightGBM (6 threads)
    print(f"\n[{time.strftime('%H:%M:%S')}] 2/3 Training {season_name} LightGBM (6 CPU threads)...")
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.04,
        "num_leaves": 63,
        "max_depth": 10,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "n_estimators": 800,
        "random_state": 42,
        "n_jobs": 6,
        "verbose": -1,
    }
    m_lgb = lgb.LGBMRegressor(**lgb_params)
    t_start = time.time()
    m_lgb.fit(X_tr, y_tr)
    print(f"{season_name} LightGBM trained in {time.time() - t_start:.1f}s")
    with open(MODELS_DIR / f"lgb_{season_name}_offset.pkl", "wb") as f:
        pickle.dump(m_lgb, f)
    p_lgb_test = m_lgb.predict(X_test_season)

    # 3. CatBoost (6 threads)
    print(f"\n[{time.strftime('%H:%M:%S')}] 3/3 Training {season_name} CatBoost (6 threads)...")
    cb_cats = [c for c in X_tr.select_dtypes(include=["category"]).columns]
    X_tr_cb = X_tr.copy()
    X_test_cb = X_test_season.copy()
    for c in cb_cats:
        X_tr_cb[c] = X_tr_cb[c].astype(str).fillna("MISSING")
        X_test_cb[c] = X_test_cb[c].astype(str).fillna("MISSING")

    m_cb = CatBoostRegressor(
        iterations=800,
        learning_rate=0.05,
        depth=8,
        cat_features=cb_cats,
        random_seed=42,
        thread_count=6,
        verbose=200,
    )
    t_start = time.time()
    m_cb.fit(X_tr_cb, y_tr)
    print(f"{season_name} CatBoost trained in {time.time() - t_start:.1f}s")
    m_cb.save_model(MODELS_DIR / f"cb_{season_name}_offset.cbm")
    p_cb_test = m_cb.predict(X_test_cb)

    return p_xgb_test, p_lgb_test, p_cb_test


def main():
    print("=" * 80)
    print("🚀 DUAL-SEASON SPECIALIZED TRAINING PIPELINE (WINTER VS SUMMER)")
    print("=" * 80)

    # 1. Load UTT lookup and ranking test set
    utt_stats = pl.read_parquet(SCRATCH_DIR / "empirical_utt_lookup.parquet")
    print(f"Loaded {len(utt_stats):,} full-year UTT profiles.")

    print(f"[{time.strftime('%H:%M:%S')}] Loading and processing ranking test set...")
    df_test = pl.read_parquet(RAW_DIR / "ranking.parquet")
    template = pl.read_parquet(RAW_DIR / "submitting.parquet")

    df_test = features.compute_congestion_metrics(df_test)
    df_test = features.compute_surface_accumulation_features(df_test)
    df_test = features.compute_apron_contention_features(df_test)
    df_test = features.compute_runway_sequence_features(df_test)

    rank_dep = df_test.filter(pl.col("PHASE_mvt") == "DEP")
    mvt_test = rank_dep["MVT_TIME_UTC_mvt"]
    jan_mask = (mvt_test.dt.month() == 1).to_numpy()
    jul_mask = (mvt_test.dt.month() == 7).to_numpy()
    print(f"Test departures: Total={len(rank_dep):,} | Jan={np.sum(jan_mask):,} | Jul={np.sum(jul_mask):,}")

    # 2. Train Winter Suite for January 2026
    xgb_jan, lgb_jan, cb_jan = train_season_suite("winter", WINTER_MONTHS, utt_stats, df_test, jan_mask)

    # 3. Train Summer Suite for July 2026
    xgb_jul, lgb_jul, cb_jul = train_season_suite("summer", SUMMER_MONTHS, utt_stats, df_test, jul_mask)

    # 4. Stitch predictions into full test array
    p_xgb_full = np.zeros(len(rank_dep), dtype=np.float64)
    p_lgb_full = np.zeros(len(rank_dep), dtype=np.float64)
    p_cb_full = np.zeros(len(rank_dep), dtype=np.float64)

    p_xgb_full[jan_mask] = xgb_jan
    p_xgb_full[jul_mask] = xgb_jul

    p_lgb_full[jan_mask] = lgb_jan
    p_lgb_full[jul_mask] = lgb_jul

    p_cb_full[jan_mask] = cb_jan
    p_cb_full[jul_mask] = cb_jul

    np.save(SCRATCH_DIR / "xgb_dualseason_test_preds.npy", p_xgb_full)
    np.save(SCRATCH_DIR / "lgb_dualseason_test_preds.npy", p_lgb_full)
    np.save(SCRATCH_DIR / "cb_dualseason_test_preds.npy", p_cb_full)

    print("\n✅ Successfully saved dual-season test predictions:")
    print(f"  • {SCRATCH_DIR / 'xgb_dualseason_test_preds.npy'}")
    print(f"  • {SCRATCH_DIR / 'lgb_dualseason_test_preds.npy'}")
    print(f"  • {SCRATCH_DIR / 'cb_dualseason_test_preds.npy'}")

    # 5. Build Dual-Season Clean Apex Ensemble
    v60_anchor = pl.read_parquet(SUBMISSIONS_DIR / f"{TEAM_NAME}_v60.parquet")["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    aobt_test = rank_dep["AOBT_3_flt"]
    d_mvt_aobt = (mvt_test - aobt_test).dt.total_seconds().to_numpy()
    valid_aobt = (~np.isnan(d_mvt_aobt)) & (d_mvt_aobt >= 120) & (d_mvt_aobt <= 7200)

    delta_dual = 0.40 * p_xgb_full + 0.35 * p_lgb_full + 0.25 * p_cb_full
    taxi_dual = np.where(valid_aobt, d_mvt_aobt - delta_dual, v60_anchor)
    taxi_dual = np.maximum(taxi_dual, 60.0)
    tail_mask = (v60_anchor > 15000)
    taxi_dual[tail_mask] = v60_anchor[tail_mask]
    preds_clean = np.clip(np.round(taxi_dual), 60, 88500).astype(np.int32)

    # Save candidate v81 (Pure Dual-Season Clean Apex)
    out_81 = SUBMISSIONS_DIR / f"{TEAM_NAME}_v81.parquet"
    sub_df = pl.DataFrame({"MVT_ID_mvt": rank_dep["MVT_ID_mvt"], "TAXITIME_SEC_mvt": preds_clean})
    final_sub = template.select(["MVT_ID_mvt"]).join(sub_df, on="MVT_ID_mvt", how="left")
    final_sub.write_parquet(out_81)
    print(f"\n🏆 Candidate v81 generated ({out_81.stat().st_size:,} bytes): mean={np.mean(preds_clean):.1f}s, med={np.median(preds_clean):.0f}s")


if __name__ == "__main__":
    main()
