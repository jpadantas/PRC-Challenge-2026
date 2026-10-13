"""
Master Candidate Generator & Validator (v81 to v85).
Assembles the 5 next-generation disruptive candidates for the 2026-09-26 submission window.
All candidates feature:
1. Strict physical gate delay clamping (no unconstrained SCHED substitution).
2. Dual-season specialized offset predictions (Winter Jan + Summer Jul).
3. Quad-stack architecture (GPU XGBoost + LightGBM + CatBoost + Tabular ResNet).
4. Physical bounds: clip [60, 88500], tail shield on v60 > 15000s.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import polars as pl
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SCRATCH_DIR = PROJECT_ROOT / "scratch"
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"
TEAM_NAME = "inspiring-kitten"


def validate_and_save(version_num: int, preds: np.ndarray, rank_dep: pl.DataFrame, template: pl.DataFrame) -> Path:
    sub_df = pl.DataFrame({
        "MVT_ID_mvt": rank_dep["MVT_ID_mvt"],
        "TAXITIME_SEC_mvt": preds,
    })
    final_sub = template.select(["MVT_ID_mvt"]).join(sub_df, on="MVT_ID_mvt", how="left")
    assert len(final_sub) == len(template), f"Length mismatch: {len(final_sub)} vs {len(template)}"
    assert final_sub["TAXITIME_SEC_mvt"].null_count() == 0, "Contains nulls"
    assert (final_sub["MVT_ID_mvt"] == template["MVT_ID_mvt"]).all(), "ID alignment mismatch"
    assert (final_sub["TAXITIME_SEC_mvt"] >= 5).all() and (final_sub["TAXITIME_SEC_mvt"] <= 88500).all(), "Values out of bounds"

    out_path = SUBMISSIONS_DIR / f"{TEAM_NAME}_v{version_num}.parquet"
    final_sub.write_parquet(out_path)
    vals = final_sub["TAXITIME_SEC_mvt"].to_numpy()
    print(f"✅ Saved v{version_num} ({out_path.stat().st_size:,} bytes): mean={np.mean(vals):.2f}s, med={np.median(vals):.0f}s, min={np.min(vals)}, max={np.max(vals)}, >15k={np.sum(vals > 15000)}")
    return out_path


def main():
    print("=" * 80)
    print("🏆 MASTER CANDIDATE GENERATOR & VALIDATOR (v81 to v85)")
    print("=" * 80)

    rank_dep = pl.read_parquet(RAW_DIR / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
    template = pl.read_parquet(RAW_DIR / "submitting.parquet")

    # Load Dual-Season GBDT predictions
    p_xgb = np.load(SCRATCH_DIR / "xgb_dualseason_test_preds.npy").astype(np.float64)
    p_lgb = np.load(SCRATCH_DIR / "lgb_dualseason_test_preds.npy").astype(np.float64)
    p_cb = np.load(SCRATCH_DIR / "cb_dualseason_test_preds.npy").astype(np.float64)
    p_res = np.load(SCRATCH_DIR / "resnet_offset_nextgen_test_preds.npy").astype(np.float64)

    # Load Dual-Season 10-Airport Specialist (XGBoost GPU) and Missing-AOBT Specialist predictions
    p_apt = np.load(SCRATCH_DIR / "xgb_dualseason_per_airport_offset_test_preds.npy").astype(np.float64)
    p_missing = np.load(SCRATCH_DIR / "missing_aobt_test_preds.npy").astype(np.float64)

    # Load historical anchors
    v60_anchor = pl.read_parquet(SUBMISSIONS_DIR / f"{TEAM_NAME}_v60.parquet")["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    v68_anchor = pl.read_parquet(SUBMISSIONS_DIR / f"{TEAM_NAME}_v68.parquet")["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    v52_anchor = pl.read_parquet(SUBMISSIONS_DIR / f"{TEAM_NAME}_v52.parquet")["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)

    # Physical valid AOBT mask
    mvt = rank_dep["MVT_TIME_UTC_mvt"]
    aobt = rank_dep["AOBT_3_flt"]
    da = (mvt - aobt).dt.total_seconds().to_numpy()
    valid_aobt = (~np.isnan(da)) & (da >= 120) & (da <= 7200)
    tail_mask = (v60_anchor > 15000)

    # 1. Candidate v81: Dual-Season Tripartite GBDT + Dual-Season Airport Specialist (16.3% Tripartite + 83.7% DS Airport Spec)
    print("\n--- Assembling v81 (Dual-Season Tripartite + Dual-Season Airport Specialist) ---")
    delta_81_global = 0.40 * p_xgb + 0.35 * p_lgb + 0.25 * p_cb
    delta_81 = 0.163 * delta_81_global + 0.837 * p_apt
    taxi_81 = np.where(valid_aobt, da - delta_81, p_missing)
    taxi_81 = np.maximum(taxi_81, 60.0)
    taxi_81[tail_mask] = v60_anchor[tail_mask]
    preds_81 = np.clip(np.round(taxi_81), 60, 88500).astype(np.int32)
    validate_and_save(81, preds_81, rank_dep, template)

    # 2. Candidate v82: Dual-Season Quad-Stack + Dual-Season Airport Specialist (16.3% Quad-Stack + 83.7% DS Airport Spec)
    print("\n--- Assembling v82 (Dual-Season Quad-Stack ResNet + Dual-Season Airport Specialist) ---")
    delta_82_global = 0.35 * p_xgb + 0.30 * p_lgb + 0.20 * p_cb + 0.15 * p_res
    delta_82 = 0.163 * delta_82_global + 0.837 * p_apt
    taxi_82 = np.where(valid_aobt, da - delta_82, p_missing)
    taxi_82 = np.maximum(taxi_82, 60.0)
    taxi_82[tail_mask] = v60_anchor[tail_mask]
    preds_82 = np.clip(np.round(taxi_82), 60, 88500).astype(np.int32)
    validate_and_save(82, preds_82, rank_dep, template)

    # 3. Candidate v83: Convex Blend of Twin Foundations (50% v81 + 50% v82)
    print("\n--- Assembling v83 (Foundation Consensus: 50% v81 + 50% v82) ---")
    preds_83_raw = 0.50 * preds_81.astype(np.float64) + 0.50 * preds_82.astype(np.float64)
    preds_83 = np.clip(np.round(preds_83_raw), 60, 88500).astype(np.int32)
    preds_83[tail_mask] = v60_anchor[tail_mask].astype(np.int32)
    validate_and_save(83, preds_83, rank_dep, template)

    # 4. Candidate v84: Analytical Golden Blend (60% v82 + 40% v60 Champion Anchor)
    print("\n--- Assembling v84 (Golden Ridge Blend: 60% v82 Quad + 40% v60 Anchor) ---")
    preds_84_raw = 0.60 * preds_82.astype(np.float64) + 0.40 * v60_anchor
    preds_84 = np.clip(np.round(preds_84_raw), 60, 88500).astype(np.int32)
    preds_84[tail_mask] = v60_anchor[tail_mask].astype(np.int32)
    validate_and_save(84, preds_84, rank_dep, template)

    # 5. Candidate v85: Hyper-Hedge Multi-Era Master Shield (45% v82 + 25% v81 + 20% v60 + 10% v68)
    print("\n--- Assembling v85 (Hyper-Hedge Multi-Era Consensus: 45% v82 + 25% v81 + 20% v60 + 10% v68) ---")
    preds_85_raw = 0.45 * preds_82.astype(np.float64) + 0.25 * preds_81.astype(np.float64) + 0.20 * v60_anchor + 0.10 * v68_anchor
    preds_85 = np.clip(np.round(preds_85_raw), 60, 88500).astype(np.int32)
    preds_85[tail_mask] = v60_anchor[tail_mask].astype(np.int32)
    validate_and_save(85, preds_85, rank_dep, template)

    print("\n🎉 ALL 5 CANDIDATES (v81 to v85) SUCCESSFULLY GENERATED & FULLY VALIDATED!")


if __name__ == "__main__":
    main()
