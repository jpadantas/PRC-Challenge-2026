"""
Inference & Assembly Pipeline for Next-Gen Submissions (v71 to v75).
1. Extracts training categorical categories to align XGBoost input.
2. Predicts test ranking offsets with LightGBM, CatBoost, and XGBoost.
3. Assembles v71 through v75 with LIRF classifier & extreme tail shield.
4. Validates all constraints (344,841 rows, 0 nulls, [60, 88500]).
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
import lightgbm as lgb
from catboost import CatBoostRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
SCRATCH_DIR = PROJECT_ROOT / "scratch"
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"
TEAM_NAME = "inspiring-kitten"

ALL_12_MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]


def extract_advanced_matrix(df: pl.DataFrame, utt_stats: pl.DataFrame, target_encs: dict | None = None) -> tuple[pd.DataFrame, pl.DataFrame]:
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


def load_lirf_predictions(rank_dep: pl.DataFrame) -> np.ndarray:
    clf_path = MODELS_DIR / "lirf_stage1_classifier.pkl"
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)

    lirf_mask = (rank_dep["ADEP_mvt"] == "LIRF").to_numpy()
    sub_lirf = rank_dep.filter(pl.Series("m", lirf_mask))

    mvt = sub_lirf["MVT_TIME_UTC_mvt"]
    sched = sub_lirf["SCHED_TIME_UTC_mvt"]
    aobt = sub_lirf["AOBT_3_flt"]
    eobt = sub_lirf["EOBT_1_flt"]
    iobt = sub_lirf["IOBT_flt"]
    lobt = sub_lirf["LOBT_flt"]

    d_ms = (mvt - sched).dt.total_seconds().to_numpy()
    d_ma = (mvt - aobt).dt.total_seconds().fill_null(1000.0).to_numpy()
    d_as = (aobt - sched).dt.total_seconds().fill_null(0.0).to_numpy()
    d_ae = (aobt - eobt).dt.total_seconds().fill_null(0.0).to_numpy()
    d_es = (eobt - sched).dt.total_seconds().fill_null(0.0).to_numpy()
    d_is = (iobt - sched).dt.total_seconds().fill_null(0.0).to_numpy()
    d_ls = (lobt - sched).dt.total_seconds().fill_null(0.0).to_numpy()

    has_aobt = aobt.is_not_null().cast(pl.Int8).to_numpy()
    has_eobt = eobt.is_not_null().cast(pl.Int8).to_numpy()

    stand_str = sub_lirf["STAND_mvt"].cast(pl.Utf8).fill_null("UNK")
    pier = stand_str.str.slice(0, 1)
    hour = mvt.dt.hour().to_numpy()
    dow = mvt.dt.weekday().to_numpy()
    month = mvt.dt.month().to_numpy()

    feat_df = pd.DataFrame({
        "ades": sub_lirf["ADES_mvt"].to_pandas().astype("category"),
        "runway": sub_lirf["RUNWAY_mvt"].to_pandas().astype("category"),
        "stand": stand_str.to_pandas().astype("category"),
        "pier": pier.to_pandas().astype("category"),
        "actype": sub_lirf["AIRCRAFT_TYPE_mvt"].to_pandas().astype("category"),
        "operator": sub_lirf["AIRCRAFT_OPERATOR_flt"].to_pandas().astype("category"),
        "month": month,
        "hour": hour,
        "dow": dow,
        "d_ms": d_ms,
        "d_ma": d_ma,
        "d_as": d_as,
        "d_ae": d_ae,
        "d_es": d_es,
        "d_is": d_is,
        "d_ls": d_ls,
        "has_aobt": has_aobt,
        "has_eobt": has_eobt,
        "is_delay_gt_1h": (d_ms > 3600).astype(int),
        "is_delay_gt_3h": (d_ms > 10800).astype(int),
        "is_delay_gt_6h": (d_ms > 21600).astype(int),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
    })
    for c in feat_df.select_dtypes(include=["object", "category"]).columns:
        feat_df[c] = feat_df[c].astype("category")

    probs = clf.predict_proba(feat_df)[:, 1]
    full_probs = np.zeros(len(rank_dep), dtype=np.float32)
    full_probs[lirf_mask] = probs
    return full_probs


def assemble_predictions(rank_dep: pl.DataFrame, delta_pred: np.ndarray, lirf_probs: np.ndarray, v60_anchor: np.ndarray) -> np.ndarray:
    mvt = rank_dep["MVT_TIME_UTC_mvt"]
    sched = rank_dep["SCHED_TIME_UTC_mvt"]
    aobt = rank_dep["AOBT_3_flt"]
    adep = rank_dep["ADEP_mvt"].to_numpy()

    has_aobt = aobt.is_not_null().to_numpy()
    d_mvt_aobt = (mvt - aobt).dt.total_seconds().to_numpy()
    d_mvt_sched = (mvt - sched).dt.total_seconds().to_numpy()

    taxi_via_offset = d_mvt_aobt - delta_pred
    is_lirf = (adep == "LIRF")
    is_lirf_default = is_lirf & (lirf_probs >= 0.40)

    final_taxi = np.where(is_lirf_default, d_mvt_sched, np.where(has_aobt, taxi_via_offset, v60_anchor))
    tail_mask = (v60_anchor > 15000)
    final_taxi[tail_mask] = v60_anchor[tail_mask]
    return np.clip(np.round(final_taxi), 60, 88500).astype(np.int32)


def validate_and_save(version_num: int, preds: np.ndarray, rank_dep: pl.DataFrame, template: pl.DataFrame) -> Path:
    sub_df = pl.DataFrame({
        "MVT_ID_mvt": rank_dep["MVT_ID_mvt"],
        "TAXITIME_SEC_mvt": preds,
    })
    final_sub = template.select(["MVT_ID_mvt"]).join(sub_df, on="MVT_ID_mvt", how="left")
    assert len(final_sub) == len(template)
    assert final_sub["TAXITIME_SEC_mvt"].null_count() == 0
    assert (final_sub["MVT_ID_mvt"] == template["MVT_ID_mvt"]).all()
    assert (final_sub["TAXITIME_SEC_mvt"] >= 5).all() and (final_sub["TAXITIME_SEC_mvt"] <= 88500).all()

    out_path = SUBMISSIONS_DIR / f"{TEAM_NAME}_v{version_num}.parquet"
    final_sub.write_parquet(out_path)
    vals = final_sub["TAXITIME_SEC_mvt"].to_numpy()
    print(f"✅ Saved v{version_num} ({out_path.stat().st_size:,} bytes): mean={np.mean(vals):.2f}s, med={np.median(vals):.0f}s, >15k={np.sum(vals > 15000)}")
    return out_path


def main():
    print("=" * 80)
    print("🚀 INFERENCE & ASSEMBLY PIPELINE: NEXT-GEN SUITE (v71 to v75)")
    print("=" * 80)

    # 1. Load UTT and Encodings
    utt_stats = pl.read_parquet(SCRATCH_DIR / "empirical_utt_lookup.parquet")
    with open(MODELS_DIR / "encs_v75_nextgen.pkl", "rb") as f:
        encs = pickle.load(f)
    print("Loaded UTT profiles and Target Encodings.")

    # 2. Extract training categories to align XGBoost
    cat_dtypes_path = MODELS_DIR / "cat_dtypes_nextgen.pkl"
    if not cat_dtypes_path.exists():
        print("Extracting training categorical levels from full 2025 dataset...")
        df_train = features.load_training_data(RAW_DIR, months=ALL_12_MONTHS)
        X_train_df, dep_train_ext = extract_advanced_matrix(df_train, utt_stats, target_encs=encs)
        cat_dtypes = {c: X_train_df[c].dtype for c in X_train_df.select_dtypes(include="category").columns}
        with open(cat_dtypes_path, "wb") as f:
            pickle.dump(cat_dtypes, f)
        print(f"Saved {len(cat_dtypes)} categorical dtypes.")
        del df_train, X_train_df, dep_train_ext
    else:
        with open(cat_dtypes_path, "rb") as f:
            cat_dtypes = pickle.load(f)
        print(f"Loaded {len(cat_dtypes)} categorical dtypes.")

    # 3. Load ranking test set
    print("\nLoading and processing ranking test set...")
    df_test = features.load_ranking_data(RAW_DIR)
    template = features.load_submission_template(RAW_DIR)

    df_test = features.compute_congestion_metrics(df_test)
    df_test = features.compute_surface_accumulation_features(df_test)
    df_test = features.compute_apron_contention_features(df_test)
    df_test = features.compute_runway_sequence_features(df_test)

    X_test_df, dep_test_ext = extract_advanced_matrix(df_test, utt_stats, target_encs=encs)
    rank_dep = df_test.filter(pl.col("PHASE_mvt") == "DEP")
    print(f"Ranking test departures: {len(rank_dep):,}")

    # 4. Predict with LightGBM
    print("\n1/3 Predicting test offsets with LightGBM Next-Gen...")
    with open(MODELS_DIR / "lgb_offset_nextgen.pkl", "rb") as f:
        m_lgb = pickle.load(f)
    p_lgb = m_lgb.predict(X_test_df)
    np.save(SCRATCH_DIR / "lgb_offset_nextgen_test_preds.npy", p_lgb)
    print(f"  • LightGBM: mean={np.mean(p_lgb):.2f}s, std={np.std(p_lgb):.2f}s")

    # 5. Predict with CatBoost
    print("\n2/3 Predicting test offsets with CatBoost Next-Gen...")
    m_cb = CatBoostRegressor()
    m_cb.load_model(str(MODELS_DIR / "cb_offset_nextgen.cbm"))
    X_test_cb = X_test_df.copy()
    cb_cats = list(X_test_cb.select_dtypes(include=["category"]).columns)
    for c in cb_cats:
        X_test_cb[c] = X_test_cb[c].astype(str).fillna("MISSING")
    p_cb = m_cb.predict(X_test_cb)
    np.save(SCRATCH_DIR / "cb_offset_nextgen_test_preds.npy", p_cb)
    print(f"  • CatBoost: mean={np.mean(p_cb):.2f}s, std={np.std(p_cb):.2f}s")

    # 6. Predict with XGBoost (categories aligned)
    print("\n3/3 Predicting test offsets with XGBoost Next-Gen...")
    X_test_xgb = X_test_df.copy()
    for c, dtype in cat_dtypes.items():
        if c in X_test_xgb.columns:
            X_test_xgb[c] = pd.Categorical(X_test_xgb[c], categories=dtype.categories, ordered=dtype.ordered)

    booster = xgb.Booster()
    booster.load_model(str(MODELS_DIR / "xgb_offset_nextgen.json"))
    dtest = xgb.DMatrix(X_test_xgb, enable_categorical=True)
    p_xgb = booster.predict(dtest)
    np.save(SCRATCH_DIR / "xgb_offset_nextgen_test_preds.npy", p_xgb)
    print(f"  • XGBoost:  mean={np.mean(p_xgb):.2f}s, std={np.std(p_xgb):.2f}s")

    # 7. Load Anchors & Prior Generations
    v60_anchor = pl.read_parquet(SUBMISSIONS_DIR / f"{TEAM_NAME}_v60.parquet")["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    v68_anchor = pl.read_parquet(SUBMISSIONS_DIR / f"{TEAM_NAME}_v68.parquet")["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    lirf_probs = load_lirf_predictions(rank_dep)

    p_res = np.load(SCRATCH_DIR / "resnet_offset_test_preds.npy").astype(np.float64)

    # 8. Assemble Super-Suite v71 through v75
    print("\n" + "=" * 80)
    print("🎯 ASSEMBLING SUBMISSIONS (v71 to v75)")
    print("=" * 80)

    # v71: Next-Gen XGBoost GPU (195 features, tactical milestones + full-year UTT)
    delta_v71 = p_xgb
    preds_v71 = assemble_predictions(rank_dep, delta_v71, lirf_probs, v60_anchor)
    validate_and_save(71, preds_v71, rank_dep, template)

    # v72: Next-Gen Dual-Engine (50% XGB + 50% LGB)
    delta_v72 = 0.50 * p_xgb + 0.50 * p_lgb
    preds_v72 = assemble_predictions(rank_dep, delta_v72, lirf_probs, v60_anchor)
    validate_and_save(72, preds_v72, rank_dep, template)

    # v73: Next-Gen Tripartite GBDT (40% XGB + 35% LGB + 25% CB)
    delta_v73 = 0.40 * p_xgb + 0.35 * p_lgb + 0.25 * p_cb
    preds_v73 = assemble_predictions(rank_dep, delta_v73, lirf_probs, v60_anchor)
    validate_and_save(73, preds_v73, rank_dep, template)

    # v74: Next-Gen Quad-Stack with ResNet (35% XGB + 30% LGB + 20% CB + 15% ResNet)
    delta_v74 = 0.35 * p_xgb + 0.30 * p_lgb + 0.20 * p_cb + 0.15 * p_res
    preds_v74 = assemble_predictions(rank_dep, delta_v74, lirf_probs, v60_anchor)
    validate_and_save(74, preds_v74, rank_dep, template)

    # v75: Multi-Epoch Consensus Apex (50% Next-Gen v74 + 30% v68 Quad-Stack + 20% v60 Meta-Anchor)
    preds_v75_raw = 0.50 * preds_v74.astype(np.float64) + 0.30 * v68_anchor + 0.20 * v60_anchor
    preds_v75 = np.clip(np.round(preds_v75_raw), 60, 88500).astype(np.int32)
    tail_mask = (v60_anchor > 15000)
    preds_v75[tail_mask] = v60_anchor[tail_mask].astype(np.int32)
    validate_and_save(75, preds_v75, rank_dep, template)

    print("\n🎉 ALL 5 NEXT-GEN SUBMISSIONS (v71 to v75) GENERATED & RIGOROUSLY VALIDATED!")


if __name__ == "__main__":
    main()
