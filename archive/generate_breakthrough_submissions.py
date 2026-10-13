"""
Generate Breakthrough Submissions v66 through v70:
Unified Physics Identity Engine with Dual-Regime LIRF Schedule Classifier,
Offset Regressor Ensemble, and Extreme Horizon Tail Shield.
Targeting #1 on the Leaderboard (<235s).
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import pickle
from pathlib import Path
import polars as pl
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
SCRATCH_DIR = PROJECT_ROOT / "scratch"
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"
TEAM_NAME = "inspiring-kitten"


def load_lirf_predictions(rank_dep: pl.DataFrame) -> np.ndarray:
    """Predicts schedule-default probability for all LIRF test flights."""
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

    # Map back to full rank length
    full_probs = np.zeros(len(rank_dep), dtype=np.float32)
    full_probs[lirf_mask] = probs
    return full_probs


def assemble_predictions(
    rank_dep: pl.DataFrame,
    delta_pred: np.ndarray,
    lirf_probs: np.ndarray,
    v60_anchor: np.ndarray,
    lirf_threshold: float = 0.40,
) -> np.ndarray:
    """Assembles final physical taxi predictions from delta offset, LIRF classifier, and radar identities."""
    mvt = rank_dep["MVT_TIME_UTC_mvt"]
    sched = rank_dep["SCHED_TIME_UTC_mvt"]
    aobt = rank_dep["AOBT_3_flt"]
    adep = rank_dep["ADEP_mvt"].to_numpy()

    has_aobt = aobt.is_not_null().to_numpy()
    d_mvt_aobt = (mvt - aobt).dt.total_seconds().to_numpy()
    d_mvt_sched = (mvt - sched).dt.total_seconds().to_numpy()

    # 1. Physical Offset Identity: TAXITIME = (MVT - AOBT_3) - delta
    taxi_via_offset = d_mvt_aobt - delta_pred

    # 2. LIRF Schedule Default Regime: TAXITIME = MVT - SCHED
    is_lirf = (adep == "LIRF")
    is_lirf_default = is_lirf & (lirf_probs >= lirf_threshold)

    # 3. Assemble: default to v60 fallback for missing AOBT
    final_taxi = np.where(
        is_lirf_default,
        d_mvt_sched,
        np.where(
            has_aobt,
            taxi_via_offset,
            v60_anchor,
        ),
    )

    # 4. Tail Shield: preserve verified extreme horizon (>15,000s) from v18/v60
    tail_mask = (v60_anchor > 15000)
    final_taxi[tail_mask] = v60_anchor[tail_mask]

    # 5. Physical bounds clamp and round
    final_taxi = np.clip(np.round(final_taxi), 60, 88500).astype(np.int32)
    return final_taxi


def validate_and_save(version_num: int, preds: np.ndarray, rank_dep: pl.DataFrame, template: pl.DataFrame) -> Path:
    """Rigorous sanity checks and saving to data/submissions/."""
    sub_df = pl.DataFrame({
        "MVT_ID_mvt": rank_dep["MVT_ID_mvt"],
        "TAXITIME_SEC_mvt": preds,
    })

    # Join with template to guarantee exact ordering
    final_sub = template.select(["MVT_ID_mvt"]).join(sub_df, on="MVT_ID_mvt", how="left")

    # Assertions
    assert len(final_sub) == len(template), f"Length mismatch: {len(final_sub)} vs {len(template)}"
    assert final_sub["TAXITIME_SEC_mvt"].null_count() == 0, f"Found {final_sub['TAXITIME_SEC_mvt'].null_count()} nulls!"
    assert (final_sub["MVT_ID_mvt"] == template["MVT_ID_mvt"]).all(), "Ordering mismatch with template!"
    assert (final_sub["TAXITIME_SEC_mvt"] >= 5).all() and (final_sub["TAXITIME_SEC_mvt"] <= 88500).all(), "Values outside physical limits!"

    out_path = SUBMISSIONS_DIR / f"{TEAM_NAME}_v{version_num}.parquet"
    final_sub.write_parquet(out_path)

    vals = final_sub["TAXITIME_SEC_mvt"].to_numpy()
    print(f"✅ Submission v{version_num} saved to {out_path.name}:")
    print(f"   Size: {out_path.stat().st_size:,} bytes | Mean: {np.mean(vals):.2f}s | Median: {np.median(vals):.0f}s | Min: {np.min(vals)}s | Max: {np.max(vals)}s")
    print(f"   Tail > 7200s: {np.sum(vals > 7200):,} | > 15000s: {np.sum(vals > 15000):,} | > 60000s: {np.sum(vals > 60000)}")
    return out_path


def main():
    print("=" * 80)
    print("🚀 GENERATING BREAKTHROUGH SUBMISSIONS (v66 to v70) FOR LEADERBOARD #1")
    print("=" * 80)

    # 1. Load ranking test data and template
    rank_dep = pl.read_parquet(RAW_DIR / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
    template = pl.read_parquet(RAW_DIR / "submitting.parquet")
    print(f"Loaded {len(rank_dep):,} test departures.")

    # 2. Load v60 champion anchor
    v60_path = SUBMISSIONS_DIR / f"{TEAM_NAME}_v60.parquet"
    v60_anchor = pl.read_parquet(v60_path)["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    print(f"Loaded v60 champion anchor (score: 293.42s).")

    # 3. Compute LIRF schedule default probabilities
    print("\nComputing LIRF Stage-1 Schedule-Default Probabilities...")
    lirf_probs = load_lirf_predictions(rank_dep)
    n_lirf_defaults = np.sum((rank_dep["ADEP_mvt"].to_numpy() == "LIRF") & (lirf_probs >= 0.40))
    print(f"Identified {n_lirf_defaults:,} LIRF departures switching to exact MVT - SCHED identity.")

    # 4. Load Offset Model Predictions
    print("\nLoading Offset Model Predictions from scratch/...")
    p_lgb = np.load(SCRATCH_DIR / "lgb_offset_test_preds.npy").astype(np.float64)
    print(f"  • LightGBM Offset: loaded {len(p_lgb):,} predictions (mean={np.mean(p_lgb):.2f}s, std={np.std(p_lgb):.2f}s)")

    p_xgb_file = SCRATCH_DIR / "xgb_offset_test_preds.npy"
    if p_xgb_file.exists():
        p_xgb = np.load(p_xgb_file).astype(np.float64)
        print(f"  • XGBoost Offset:  loaded {len(p_xgb):,} predictions (mean={np.mean(p_xgb):.2f}s, std={np.std(p_xgb):.2f}s)")
    else:
        p_xgb = p_lgb.copy()
        print("  • XGBoost Offset: not yet available, aliased to LightGBM.")

    p_cb_file = SCRATCH_DIR / "cb_offset_test_preds.npy"
    if p_cb_file.exists():
        p_cb = np.load(p_cb_file).astype(np.float64)
        print(f"  • CatBoost Offset: loaded {len(p_cb):,} predictions (mean={np.mean(p_cb):.2f}s, std={np.std(p_cb):.2f}s)")
    else:
        p_cb = p_lgb.copy()
        print("  • CatBoost Offset: not yet available, aliased to LightGBM.")

    p_res_file = SCRATCH_DIR / "resnet_offset_test_preds.npy"
    if p_res_file.exists():
        p_resnet = np.load(p_res_file).astype(np.float64)
        print(f"  • PyTorch ResNet:  loaded {len(p_resnet):,} predictions (mean={np.mean(p_resnet):.2f}s, std={np.std(p_resnet):.2f}s)")
    else:
        p_resnet = p_lgb.copy()
        print("  • PyTorch ResNet: not yet available, aliased to LightGBM.")

    # 5. Build Submissions
    print("\n" + "=" * 80)
    print("ASSEMBLING SUBMISSION CANDIDATES")
    print("=" * 80)

    # v66: Dual-Engine Blend (50% LGB + 50% XGB) + LIRF Classifier
    print("\n--- Generating v66 (Dual-Engine: 50% LGB + 50% XGB GPU) ---")
    delta_v66 = 0.50 * p_lgb + 0.50 * p_xgb
    preds_v66 = assemble_predictions(rank_dep, delta_v66, lirf_probs, v60_anchor, lirf_threshold=0.40)
    validate_and_save(66, preds_v66, rank_dep, template)

    # v67: Tripartite GBDT (40% XGB + 35% LGB + 25% CatBoost) + LIRF Classifier
    print("\n--- Generating v67 (Tripartite GBDT: 40% XGB + 35% LGB + 25% CB) ---")
    delta_v67 = 0.40 * p_xgb + 0.35 * p_lgb + 0.25 * p_cb
    preds_v67 = assemble_predictions(rank_dep, delta_v67, lirf_probs, v60_anchor, lirf_threshold=0.40)
    validate_and_save(67, preds_v67, rank_dep, template)

    # v68: Quad-Paradigm Grandmaster (35% XGB + 30% LGB + 20% CB + 15% ResNet) + LIRF
    print("\n--- Generating v68 (Quad-Paradigm Grandmaster: 35% XGB + 30% LGB + 20% CB + 15% ResNet) ---")
    delta_v68 = 0.35 * p_xgb + 0.30 * p_lgb + 0.20 * p_cb + 0.15 * p_resnet
    preds_v68 = assemble_predictions(rank_dep, delta_v68, lirf_probs, v60_anchor, lirf_threshold=0.40)
    validate_and_save(68, preds_v68, rank_dep, template)

    # v69: Orthogonal Master Blend (75% Quad-Paradigm Offset + 25% v60 Champion Meta-Anchor)
    print("\n--- Generating v69 (Orthogonal Master Blend: 75% Offset + 25% v60 Anchor) ---")
    preds_v69_raw = 0.75 * preds_v68.astype(np.float64) + 0.25 * v60_anchor
    preds_v69 = np.clip(np.round(preds_v69_raw), 60, 88500).astype(np.int32)
    # Re-apply tail shield
    tail_mask = (v60_anchor > 15000)
    preds_v69[tail_mask] = v60_anchor[tail_mask].astype(np.int32)
    validate_and_save(69, preds_v69, rank_dep, template)

    # v70: Conservative Risk Hedge (50% Quad-Paradigm Offset + 50% v60 Anchor)
    print("\n--- Generating v70 (Conservative Hedge: 50% Offset + 50% v60 Anchor) ---")
    preds_v70_raw = 0.50 * preds_v68.astype(np.float64) + 0.50 * v60_anchor
    preds_v70 = np.clip(np.round(preds_v70_raw), 60, 88500).astype(np.int32)
    preds_v70[tail_mask] = v60_anchor[tail_mask].astype(np.int32)
    validate_and_save(70, preds_v70, rank_dep, template)

    print("\n🎉 ALL 5 BREAKTHROUGH SUBMISSIONS (v66 to v70) GENERATED & RIGOROUSLY VALIDATED!")


if __name__ == "__main__":
    main()
