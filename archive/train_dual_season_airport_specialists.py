"""
Train Dual-Season Dedicated Airport-Specialist XGBoost GPU Offset Models.
Airports: EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM.
Seasons:
  - Winter Suite (Jan, Feb, Nov, Dec 2025) -> Predicts January 2026 Test Departures
  - Summer Suite (May, Jun, Jul, Aug 2025) -> Predicts July 2026 Test Departures

Strategy:
Isolates airport-specific winter de-icing / LVP taxiway dynamics from summer peak convective
congestion. Completely eliminates seasonal cross-contamination per hub.
Saves predictions to scratch/xgb_dualseason_per_airport_offset_test_preds.npy.
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

# Thermal safety
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["POLARS_MAX_THREADS"] = "6"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models" / "dualseason_per_airport"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_DIR = PROJECT_ROOT / "scratch"

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
WINTER_MONTHS = ["2025-01", "2025-02", "2025-11", "2025-12"]
SUMMER_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08"]


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
    X_df["mkt_segment"] = dep["MARKET_SEGMENT_flt"].to_pandas().astype("category")
    X_df["flight_type"] = dep["FLIGHT_TYPE_flt"].to_pandas().astype("category")
    X_df["wk_tbl_cat"] = dep["WK_TBL_CAT_flt"].to_pandas().astype("category")

    for c in X_df.select_dtypes(include=["object"]).columns:
        X_df[c] = X_df[c].astype("category")

    return X_df, dep


def train_season_airports(season_name: str, months: list[str], test_df: pl.DataFrame, test_season_mask: np.ndarray, utt_stats: pl.DataFrame) -> np.ndarray:
    t_season = time.time()
    print("\n" + "=" * 80)
    print(f"❄️ / ☀️ TRAINING {season_name.upper()} AIRPORT SPECIALISTS ({', '.join(months)})")
    print("=" * 80)

    # 1. Load season training data
    df_tr_season = features.load_training_data(RAW_DIR, months=months)
    print(f"[{season_name.upper()}] Loaded {len(df_tr_season):,} rows in {time.time() - t_season:.1f}s")

    print(f"[{season_name.upper()}] Precomputing domain features...")
    df_tr_season = features.compute_congestion_metrics(df_tr_season)
    df_tr_season = features.compute_surface_accumulation_features(df_tr_season)
    df_tr_season = features.compute_apron_contention_features(df_tr_season)
    df_tr_season = features.compute_runway_sequence_features(df_tr_season)

    rank_dep = test_df.filter(pl.col("PHASE_mvt") == "DEP")
    adep_test = rank_dep["ADEP_mvt"].to_numpy()

    p_season_out = np.zeros(np.sum(test_season_mask), dtype=np.float64)
    # Mapping indices within test_season_mask
    season_indices = np.where(test_season_mask)[0]

    for apt in AIRPORTS:
        t_apt = time.time()
        print(f"\n--- [{season_name.upper()}] {apt} ---")

        # Filter training data for this airport
        df_apt_tr = df_tr_season.filter(
            pl.when(pl.col("PHASE_mvt") == "DEP")
            .then(pl.col("ADEP_mvt") == apt)
            .otherwise(pl.col("ADES_mvt") == apt)
        )
        dep_tr = df_apt_tr.filter(pl.col("PHASE_mvt") == "DEP")
        encs_apt = features.compute_target_encodings(dep_tr)

        X_tr_df, dep_ext_tr = extract_airport_features(df_apt_tr, utt_stats, target_encs=encs_apt)

        aobt_col = dep_ext_tr["AOBT_3_flt"]
        block_col = dep_ext_tr["BLOCK_TIME_UTC_mvt"]
        sched_col = dep_ext_tr["SCHED_TIME_UTC_mvt"]

        has_aobt = aobt_col.is_not_null().to_numpy()
        y_offset = (block_col - aobt_col).dt.total_seconds().to_numpy()
        d_bs = (block_col - sched_col).dt.total_seconds().to_numpy()
        is_sched_default = (np.abs(d_bs) <= 15)

        valid_mask = has_aobt & (~is_sched_default) & (y_offset >= -3600) & (y_offset <= 3600)
        X_tr = X_tr_df[valid_mask]
        y_tr = y_offset[valid_mask]

        print(f"[{season_name.upper()} {apt}] Train rows: {len(X_tr):,} | Target std: {np.std(y_tr):.2f}s | Mean: {np.mean(y_tr):.2f}s")

        # Filter test set for this airport & season
        apt_test_mask = (adep_test == apt) & test_season_mask
        if np.sum(apt_test_mask) == 0:
            continue

        df_apt_te = test_df.filter(
            pl.when(pl.col("PHASE_mvt") == "DEP")
            .then(pl.col("ADEP_mvt") == apt)
            .otherwise(pl.col("ADES_mvt") == apt)
        )
        # Extract features for all test departures of this airport
        X_te_df, dep_ext_te = extract_airport_features(df_apt_te, utt_stats, target_encs=encs_apt)
        mvt_apt_te = dep_ext_te["MVT_TIME_UTC_mvt"]
        if season_name == "winter":
            season_sub_mask = (mvt_apt_te.dt.month() == 1).to_numpy()
        else:
            season_sub_mask = (mvt_apt_te.dt.month() == 7).to_numpy()

        X_te = X_te_df[season_sub_mask].copy()

        # Align categories
        for c in X_tr.select_dtypes(include=["category"]).columns:
            cats = list(X_tr[c].cat.categories)
            if c in X_te.columns:
                X_te[c] = pd.Categorical(X_te[c], categories=cats)

        # Train dedicated XGBoost GPU model
        m = xgb.XGBRegressor(
            tree_method="hist",
            device="cuda",
            learning_rate=0.035,
            max_depth=9,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=1.0,
            reg_lambda=3.0,
            n_estimators=900,
            enable_categorical=True,
            random_state=42,
        )
        m.fit(X_tr, y_tr, verbose=False)
        m.save_model(MODELS_DIR / f"xgb_{season_name}_{apt}.json")

        preds_apt = m.predict(X_te)

        # Assign predictions to season out array
        # Find positions of apt_test_mask within season_indices
        pos_in_season = np.where(adep_test[test_season_mask] == apt)[0]
        assert len(pos_in_season) == len(preds_apt), f"Count mismatch: {len(pos_in_season)} vs {len(preds_apt)}"
        p_season_out[pos_in_season] = preds_apt

        print(f"[{season_name.upper()} {apt}] Trained in {time.time() - t_apt:.1f}s | Test N={len(preds_apt):,}, Mean={np.mean(preds_apt):.2f}s, Std={np.std(preds_apt):.2f}s")

    print(f"\n🎉 {season_name.upper()} Suite completed in {time.time() - t_season:.1f}s")
    return p_season_out


def main():
    t_global = time.time()
    print("=" * 80)
    print("🚀 TRAINING DUAL-SEASON PER-AIRPORT XGBOOST GPU SPECIALISTS")
    print("=" * 80)

    utt_stats = pl.read_parquet(SCRATCH_DIR / "empirical_utt_lookup.parquet")

    # 1. Load ranking test set once
    print(f"[{time.strftime('%H:%M:%S')}] Loading ranking test set...")
    df_test = pl.read_parquet(RAW_DIR / "ranking.parquet")
    df_test = features.compute_congestion_metrics(df_test)
    df_test = features.compute_surface_accumulation_features(df_test)
    df_test = features.compute_apron_contention_features(df_test)
    df_test = features.compute_runway_sequence_features(df_test)

    rank_dep = df_test.filter(pl.col("PHASE_mvt") == "DEP")
    mvt_test = rank_dep["MVT_TIME_UTC_mvt"]
    jan_mask = (mvt_test.dt.month() == 1).to_numpy()
    jul_mask = (mvt_test.dt.month() == 7).to_numpy()
    n_test = len(rank_dep)

    print(f"Test departures: Total={n_test:,} | Jan (Winter)={np.sum(jan_mask):,} | Jul (Summer)={np.sum(jul_mask):,}")

    # 2. Train Winter Suite for January
    preds_jan = train_season_airports("winter", WINTER_MONTHS, df_test, jan_mask, utt_stats)

    # 3. Train Summer Suite for July
    preds_jul = train_season_airports("summer", SUMMER_MONTHS, df_test, jul_mask, utt_stats)

    # 4. Stitch full predictions
    p_full = np.zeros(n_test, dtype=np.float64)
    p_full[jan_mask] = preds_jan
    p_full[jul_mask] = preds_jul

    out_file = SCRATCH_DIR / "xgb_dualseason_per_airport_offset_test_preds.npy"
    np.save(out_file, p_full)
    print("\n" + "=" * 80)
    print(f"💾 Saved full Dual-Season Per-Airport predictions to {out_file}")
    print(f"Global stats: Mean={np.mean(p_full):.2f}s, Std={np.std(p_full):.2f}s")
    print(f"January (Winter) stats: Mean={np.mean(preds_jan):.2f}s, Std={np.std(preds_jan):.2f}s")
    print(f"July (Summer) stats:    Mean={np.mean(preds_jul):.2f}s, Std={np.std(preds_jul):.2f}s")
    print(f"🎉 All Dual-Season Airport Specialists completed in {time.time() - t_global:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
