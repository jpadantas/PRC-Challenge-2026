"""
Final production pipeline (paper, Sections 2.3-2.5).

Predicts the 670,790 departures of the Phase 2 set (January, February, June and July 2026):
1. Base features (src/features_fast.py) + physics feature engine (src/features_physics_engine.py).
2. Regime 1 (NM off-block time present, 120 <= da <= 7200 s): LightGBM, XGBoost (GPU) and
   CatBoost (GPU) trained on the offset da - TAXITIME, averaged with weights 0.40/0.35/0.25.
   Separate winter (Jan, Feb, Nov, Dec 2025) and summer (May-Aug 2025) models.
3. Regime 2A (no NM off-block time, or da < 120 s): flight-plan LightGBM, clipped to [60, 3600] s.
4. Regime 2B (LIRF, no NM off-block time, MVT - SCHED > 7200 s): MVT - SCHED.
5. Regime 3 (da > 7200 s): da minus the airport's median training offset, not clipped at 2 h.

Writes:
   - inspiring-kitten_final.parquet (Phase 2 file, all four months)
   - inspiring-kitten_v111.parquet  (Phase 1 subset, January and July)
Nothing is uploaded.
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
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

from src.features_fast import compute_target_encodings_clean, extract_features_seasonal
from src.features_physics_engine import compute_physics_features_fast

TEAM_NAME = "inspiring-kitten"
OUT_FINAL = SUBMISSIONS_DIR / f"{TEAM_NAME}_final.parquet"
OUT_V111 = SUBMISSIONS_DIR / f"{TEAM_NAME}_v111.parquet"

WINTER_MONTHS = ["2025-01", "2025-02", "2025-11", "2025-12"]
SUMMER_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08"]


def load_season_training_data(months: list[str]) -> pl.DataFrame:
    files = []
    for m in months:
        year, month = int(m[:4]), int(m[5:7])
        if month < 12:
            p = RAW_DIR / f"training_{m}-01_{year}-{month+1:02d}-01.parquet"
        else:
            p = RAW_DIR / f"training_{m}-01_{year+1}-01-01.parquet"
        assert p.exists(), f"Missing dataset: {p}"
        files.append(p)
    return pl.concat([pl.read_parquet(f) for f in files])


def align_categories(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    test_aligned = test_df.copy()
    for col in train_df.select_dtypes(include=["category"]).columns:
        cats = list(train_df[col].cat.categories)
        if col in test_aligned.columns:
            test_aligned[col] = pd.Categorical(test_aligned[col], categories=cats)
    return test_aligned


def train_tripartite_seasonal(
    season_name: str,
    months: list[str],
    df_test_all: pl.DataFrame,
    p_test_pd: pd.DataFrame,
    test_season_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    print("\n" + "=" * 80)
    print(f"✈️  TRAINING {season_name.upper()} TRIPARTITE GBDT WITH ADVANCED PHYSICS ENGINE")
    print(f"   Historical Base: {', '.join(months)} | Target Evaluation Partition")
    print("=" * 80)

    t0 = time.time()
    df_train = load_season_training_data(months)
    dep_train = df_train.filter(pl.col("PHASE_mvt") == "DEP")
    print(f"[{season_name.upper()}] Loaded {len(df_train):,} total movements ({len(dep_train):,} departures) in {time.time()-t0:.1f}s")

    # 1. Target encodings on clean departures
    encs = compute_target_encodings_clean(dep_train)

    # 2. Extract Base Features & Physics Features
    t_feat = time.time()
    X_train_df, dep_train_out = extract_features_seasonal(df_train, encs)
    p_train_pd = compute_physics_features_fast(df_train).drop("MVT_ID_mvt").to_pandas()
    
    # Drop overlap before merge
    overlap_tr = [c for c in p_train_pd.columns if c in X_train_df.columns]
    p_train_clean = p_train_pd.drop(columns=overlap_tr)
    X_train_full = pd.concat([X_train_df.reset_index(drop=True), p_train_clean.reset_index(drop=True)], axis=1)

    y_raw_tr = dep_train_out["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    da_tr = X_train_df["da"].to_numpy()
    has_aobt_tr = X_train_df["has_aobt"].to_numpy() == 1
    offset_tr = da_tr - y_raw_tr

    clean_mask_tr = (
        (y_raw_tr >= 60)
        & (y_raw_tr <= 7200)
        & has_aobt_tr
        & (da_tr >= 120)
        & (da_tr <= 7200)
        & (offset_tr >= -3600)
        & (offset_tr <= 3600)
    )

    X_tr = X_train_full[clean_mask_tr].reset_index(drop=True)
    y_offset_tr = offset_tr[clean_mask_tr]
    adep_tr = dep_train_out.filter(clean_mask_tr)["ADEP_mvt"].to_numpy()

    # Convert object columns to category
    for col in X_tr.select_dtypes(include=["object"]).columns:
        X_tr[col] = X_tr[col].astype("category")

    print(f"[{season_name.upper()}] Clean offset training set: {len(X_tr):,} rows ({X_tr.shape[1]} features) extracted in {time.time()-t_feat:.1f}s")
    print(f"[{season_name.upper()}] Target Offset Mean: {np.mean(y_offset_tr):.1f}s, Std: {np.std(y_offset_tr):.1f}s")

    # Extract test features for this season
    X_test_all, dep_test_out = extract_features_seasonal(df_test_all, encs)
    overlap_te = [c for c in p_test_pd.columns if c in X_test_all.columns]
    p_test_clean = p_test_pd.drop(columns=overlap_te)
    X_test_full = pd.concat([X_test_all.reset_index(drop=True), p_test_clean.reset_index(drop=True)], axis=1)

    X_test_season = X_test_full[test_season_mask].copy().reset_index(drop=True)
    X_test_season = align_categories(X_tr, X_test_season)
    da_test_season = X_test_season["da"].to_numpy()

    # Model 1: LightGBM (Leaf-wise trees)
    print(f"\n[{season_name.upper()}] [1/3] Training LightGBM...")
    t_m = time.time()
    m_lgb = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        learning_rate=0.035,
        num_leaves=127,
        max_depth=9,
        subsample=0.85,
        colsample_bytree=0.80,
        min_child_samples=40,
        n_estimators=1000,
        random_state=42,
        n_jobs=6,
        verbose=-1,
    )
    m_lgb.fit(X_tr, y_offset_tr)
    pred_off_lgb = m_lgb.predict(X_test_season)
    print(f"   • LightGBM fitted in {time.time()-t_m:.1f}s | Mean offset: {np.mean(pred_off_lgb):.1f}s")

    # Model 2: XGBoost GPU (Level-wise regularized hist)
    print(f"\n[{season_name.upper()}] [2/3] Training XGBoost on RTX 4060 GPU...")
    t_m = time.time()
    m_xgb = xgb.XGBRegressor(
        tree_method="hist",
        device="cuda",
        learning_rate=0.035,
        max_depth=9,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=1.0,
        reg_lambda=3.0,
        n_estimators=900,
        random_state=42,
        enable_categorical=True,
    )
    m_xgb.fit(X_tr, y_offset_tr, verbose=0)
    pred_off_xgb = m_xgb.predict(X_test_season)
    print(f"   • XGBoost GPU fitted in {time.time()-t_m:.1f}s | Mean offset: {np.mean(pred_off_xgb):.1f}s")

    # Model 3: CatBoost GPU (Symmetric oblivious trees)
    print(f"\n[{season_name.upper()}] [3/3] Training CatBoost on RTX 4060 GPU...")
    t_m = time.time()
    cat_cols = list(X_tr.select_dtypes(include=["category"]).columns)
    X_tr_cb = X_tr.copy()
    X_te_cb = X_test_season.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(str).fillna("MISSING")
        X_te_cb[c] = X_te_cb[c].astype(str).fillna("MISSING")

    m_cb = CatBoostRegressor(
        iterations=900,
        learning_rate=0.045,
        depth=8,
        loss_function="RMSE",
        task_type="GPU",
        random_seed=42,
        verbose=0,
    )
    m_cb.fit(X_tr_cb, y_offset_tr, cat_features=cat_cols)
    pred_off_cb = m_cb.predict(X_te_cb)
    print(f"   • CatBoost GPU fitted in {time.time()-t_m:.1f}s | Mean offset: {np.mean(pred_off_cb):.1f}s")

    # Tripartite Weighted Ensemble for Offset
    offset_ensemble = 0.40 * pred_off_lgb + 0.35 * pred_off_xgb + 0.25 * pred_off_cb
    pred_taxi_season = da_test_season - offset_ensemble

    # Compute airport median offsets for fallback
    apt_offsets = {}
    for apt in np.unique(adep_tr):
        m = adep_tr == apt
        apt_offsets[apt] = float(np.median(y_offset_tr[m]))

    diagnostics = {
        "lgb_mean": float(np.mean(pred_off_lgb)),
        "xgb_mean": float(np.mean(pred_off_xgb)),
        "cb_mean": float(np.mean(pred_off_cb)),
        "ensemble_mean": float(np.mean(offset_ensemble)),
        "apt_offsets": apt_offsets,
    }

    return pred_taxi_season, diagnostics


def train_missing_aobt_specialist(df_train_sample: pl.DataFrame, df_test: pl.DataFrame) -> np.ndarray:
    print("\n" + "=" * 80)
    print("TRAINING REGIME 2A MODEL (MISSING OR IMPLAUSIBLE NM OFF-BLOCK TIME)")
    print("=" * 80)

    t0 = time.time()
    dep_all = df_train_sample.filter(pl.col("PHASE_mvt") == "DEP")
    encs = compute_target_encodings_clean(dep_all)

    X_train_df, dep_train_out = extract_features_seasonal(df_train_sample, encs)
    y_raw = dep_train_out["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()

    has_aobt_train = X_train_df["has_aobt"].to_numpy() == 1
    adep_train = X_train_df["adep"].to_numpy()
    ds_train = X_train_df["ds"].to_numpy()
    is_lirf_extreme_train = (adep_train == "LIRF") & (ds_train > 7200)

    train_missing_mask = (~has_aobt_train) & (~is_lirf_extreme_train) & (y_raw >= 60) & (y_raw <= 7200)
    drop_cols = ["da", "aobt_minus_eobt", "aobt_minus_iobt", "aobt_minus_lobt", "has_aobt"]
    X_miss_tr = X_train_df[train_missing_mask].drop(columns=drop_cols).reset_index(drop=True)
    y_miss_tr = y_raw[train_missing_mask]

    for col in X_miss_tr.select_dtypes(include=["object"]).columns:
        X_miss_tr[col] = X_miss_tr[col].astype("category")

    print(f"Missing-AOBT regular training samples: {len(X_miss_tr):,}")

    X_test_all, _ = extract_features_seasonal(df_test, encs)
    X_test_miss = X_test_all.drop(columns=drop_cols).copy().reset_index(drop=True)
    X_test_miss = align_categories(X_miss_tr, X_test_miss)

    m_miss = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        learning_rate=0.035,
        num_leaves=63,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.80,
        n_estimators=800,
        random_state=42,
        n_jobs=6,
        verbose=-1,
    )
    m_miss.fit(X_miss_tr, y_miss_tr)
    preds_missing = m_miss.predict(X_test_miss)
    print(f"Missing-AOBT engine fitted in {time.time()-t0:.1f}s | Mean={np.mean(preds_missing):.1f}s, Median={np.median(preds_missing):.0f}s")
    return preds_missing


def main():
    t_start = time.time()
    print("=" * 80)
    print("🚀 PRC DATA CHALLENGE 2026 — MASTER PHYSICS PREDICTION PIPELINE")
    print(f"   Team: {TEAM_NAME} | Output Files: {OUT_FINAL.name} & {OUT_V111.name}")
    print("=" * 80)

    # 1. Load test data
    df_test_ranking = pl.read_parquet(RAW_DIR / "final_ranking.parquet")
    template_final = pl.read_parquet(RAW_DIR / "final_submitting.parquet")
    template_p1 = pl.read_parquet(RAW_DIR / "submitting.parquet")

    # Match departures to predict in exact template order
    test_dep = df_test_ranking.filter(pl.col("PHASE_mvt") == "DEP")
    print(f"Total Ranking Movements: {len(df_test_ranking):,} | Total Departures: {len(test_dep):,}")
    print(f"Template Departures to Predict: {len(template_final):,}")
    assert len(test_dep) == len(template_final), f"Departures count mismatch: {len(test_dep)} vs {len(template_final)}"

    mvt_test = test_dep["MVT_TIME_UTC_mvt"]
    aobt_test = test_dep["AOBT_3_flt"]
    sched_test = test_dep["SCHED_TIME_UTC_mvt"]
    adep_test = test_dep["ADEP_mvt"].to_numpy()

    has_aobt = aobt_test.is_not_null().to_numpy()
    da_test = np.where(has_aobt, (mvt_test - aobt_test).dt.total_seconds().to_numpy(), np.nan)
    d_ms_test = (mvt_test - sched_test).dt.total_seconds().to_numpy()

    # Compute test physics features once
    print("\nComputing Advanced Physics Features for Test Set (670k rows)...")
    t_p = time.time()
    p_test_pd = compute_physics_features_fast(df_test_ranking).drop("MVT_ID_mvt").to_pandas()
    print(f"Test physics features computed in {time.time()-t_p:.1f}s ({p_test_pd.shape[1]} features)")

    months = mvt_test.dt.month().to_numpy()
    winter_mask = (months == 1) | (months == 2)
    summer_mask = (months == 6) | (months == 7)

    print(f"\nSeasonal Distribution of Test Flights:")
    print(f"  • Winter Partition (Jan & Feb 2026): {np.sum(winter_mask):,} flights ({np.sum(winter_mask)/len(test_dep)*100:.2f}%)")
    print(f"  • Summer Partition (Jun & Jul 2026): {np.sum(summer_mask):,} flights ({np.sum(summer_mask)/len(test_dep)*100:.2f}%)")

    # 2. Train Winter Suite & Predict Jan/Feb 2026
    preds_winter, diag_winter = train_tripartite_seasonal(
        "winter", WINTER_MONTHS, df_test_ranking, p_test_pd, winter_mask
    )

    # 3. Train Summer Suite & Predict Jun/Jul 2026
    preds_summer, diag_summer = train_tripartite_seasonal(
        "summer", SUMMER_MONTHS, df_test_ranking, p_test_pd, summer_mask
    )

    # Combine regular predictions
    preds_regular = np.zeros(len(test_dep), dtype=np.float64)
    preds_regular[winter_mask] = preds_winter
    preds_regular[summer_mask] = preds_summer

    # 4. Train Missing AOBT Specialist
    print("\nLoading representative 2025 multi-season data for missing-AOBT engine...")
    sample_files = [
        RAW_DIR / "training_2025-01-01_2025-02-01.parquet",
        RAW_DIR / "training_2025-02-01_2025-03-01.parquet",
        RAW_DIR / "training_2025-06-01_2025-07-01.parquet",
        RAW_DIR / "training_2025-07-01_2025-08-01.parquet",
    ]
    df_sample_train = pl.concat([pl.read_parquet(f) for f in sample_files])
    preds_missing = train_missing_aobt_specialist(df_sample_train, df_test_ranking)

    # 5. Regime routing (paper, Section 2.4)
    print("\n" + "=" * 80)
    print("🧭 SURGICAL MULTI-REGIME OPERATIONAL ROUTING & TAIL PROTECTION")
    print("=" * 80)

    clean_aobt_mask = has_aobt & (da_test >= 120.0) & (da_test <= 7200.0)
    extreme_delay_mask = has_aobt & (da_test > 7200.0)
    corrupt_aobt_mask = has_aobt & (da_test < 120.0)
    missing_aobt_mask = (~has_aobt) | corrupt_aobt_mask
    lirf_gate_hold_mask = (adep_test == "LIRF") & (~has_aobt) & (d_ms_test > 7200.0)

    print(f"Routing Distribution:")
    print(f"  • Regime 1 (Clean AOBT, 120s-7200s):         {np.sum(clean_aobt_mask):,} flights ({np.sum(clean_aobt_mask)/len(test_dep)*100:.2f}%)")
    print(f"  • Regime 2A (Regular Missing/Corrupt AOBT):  {np.sum(missing_aobt_mask & ~lirf_gate_hold_mask):,} flights")
    print(f"  • Regime 2B (LIRF Scheduled Gate Holds):     {np.sum(lirf_gate_hold_mask):,} flights")
    print(f"  • Regime 3 (Extreme Apron Hold da > 7200s):  {np.sum(extreme_delay_mask):,} flights")

    final_predictions = np.zeros(len(test_dep), dtype=np.float64)

    # Regime 1: Regular Tripartite GBDT Offset predictions
    final_predictions[clean_aobt_mask] = np.clip(preds_regular[clean_aobt_mask], 60.0, 7200.0)

    # Regime 2A: Specialist Missing-AOBT GBDT predictions (bounded to [60, 3600])
    final_predictions[missing_aobt_mask] = np.clip(preds_missing[missing_aobt_mask], 60.0, 3600.0)

    # Regime 2B: LIRF Deterministic Scheduled Fallback (Pushback = Sched)
    final_predictions[lirf_gate_hold_mask] = d_ms_test[lirf_gate_hold_mask]

    # Regime 3: Physical Apron Hold Tail Protection (da - median_airport_offset)
    median_apt_offsets = {**diag_winter["apt_offsets"], **diag_summer["apt_offsets"]}
    for idx in np.where(extreme_delay_mask)[0]:
        apt_code = adep_test[idx]
        off = median_apt_offsets.get(apt_code, 54.0)
        final_predictions[idx] = max(60.0, da_test[idx] - off)

    # Discrete integer rounding within Eurocontrol physical bounds [60, 88500]
    final_taxitime_int = np.clip(np.round(final_predictions), 60, 88500).astype(np.int32)

    # 6. Rigorous 8-Point Sanity Check
    print("\n" + "=" * 80)
    print("🔍 PERFORMING 8-POINT RIGOROUS INTEGRITY AUDIT")
    print("=" * 80)

    sub_df = pl.DataFrame({
        "MVT_ID_mvt": test_dep["MVT_ID_mvt"],
        "TAXITIME_SEC_mvt": final_taxitime_int,
    })
    final_sub = template_final.select(["MVT_ID_mvt"]).join(sub_df, on="MVT_ID_mvt", how="left")

    assert len(final_sub) == 670790, f"Check 1 Failed: Expected 670,790 rows, got {len(final_sub)}"
    print("  [Pass 1/8] Total row count matches exactly 670,790.")
    assert final_sub["TAXITIME_SEC_mvt"].null_count() == 0, "Check 2 Failed: Output contains null values"
    print("  [Pass 2/8] Zero null or NaN values.")
    assert (final_sub["MVT_ID_mvt"] == template_final["MVT_ID_mvt"]).all(), "Check 3 Failed: ID alignment"
    print("  [Pass 3/8] Exact row-for-row alignment with template.")
    min_taxi = int(final_sub["TAXITIME_SEC_mvt"].min())
    assert min_taxi >= 60, f"Check 4 Failed: Min taxi is {min_taxi}"
    print(f"  [Pass 4/8] Physical lower bound respected: min = {min_taxi}s.")
    max_taxi = int(final_sub["TAXITIME_SEC_mvt"].max())
    assert max_taxi <= 88500, f"Check 5 Failed: Max taxi is {max_taxi}"
    print(f"  [Pass 5/8] Physical upper bound respected: max = {max_taxi}s.")
    mean_taxi = float(final_sub["TAXITIME_SEC_mvt"].mean())
    std_taxi = float(final_sub["TAXITIME_SEC_mvt"].std())
    assert 900 <= mean_taxi <= 1200, f"Check 6 Failed: Suspicious mean taxi {mean_taxi:.1f}s"
    print(f"  [Pass 6/8] Aviation operational reality check: mean = {mean_taxi:.1f}s, std = {std_taxi:.1f}s.")
    glitch_60_cnt = int((final_sub["TAXITIME_SEC_mvt"] == 60).sum())
    assert glitch_60_cnt < 20, f"Check 7 Failed: Too many 60s flights: {glitch_60_cnt}"
    print(f"  [Pass 7/8] Zero artificial 60s clipping glitch: exactly {glitch_60_cnt} flights at 60s.")
    assert final_sub["TAXITIME_SEC_mvt"].dtype == pl.Int32, "Check 8 Failed: Non-integer dtype"
    print("  [Pass 8/8] Data type is cleanly int32.")

    # 7. Save inspiring-kitten_final.parquet (Official 4-month file)
    final_sub.write_parquet(OUT_FINAL)
    print(f"\n💾 OFFICIAL 4-MONTH FILE SAVED: {OUT_FINAL} ({OUT_FINAL.stat().st_size:,} bytes)")

    # 8. Save inspiring-kitten_v111.parquet (Phase 1 344k file for live leaderboard)
    # Join on Phase 1 submitting template
    sub_v111 = template_p1.select(["MVT_ID_mvt"]).join(sub_df, on="MVT_ID_mvt", how="left")
    assert len(sub_v111) == 344841, f"Expected 344,841 rows for v111, got {len(sub_v111)}"
    assert sub_v111["TAXITIME_SEC_mvt"].null_count() == 0
    sub_v111.write_parquet(OUT_V111)
    print(f"💾 LIVE LEADERBOARD FILE SAVED: {OUT_V111} ({OUT_V111.stat().st_size:,} bytes)")

    print(f"\n🏁 Pipeline completed successfully in {time.time()-t_start:.1f}s.")


if __name__ == "__main__":
    main()
