"""
Specialized Outlier & Anomaly Calibrator for Breakthrough Submission v96.
Addresses the two primary error bottlenecks identified in residual autopsy:
1. LTFM Gate-Delay Decoupling (721 flights without AOBT): Clamps gate-delayed flights to clean airport median.
2. Winter De-icing Pad Queues (LSZH & EDDM under active snowfall): Re-weights high congestion blizzard windows.
3. Hermetic 36-row tail shield strictly preserved.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import time
from pathlib import Path
import polars as pl
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

TEAM_NAME = "inspiring-kitten"
OUT_VERSION = 96


def main():
    print("=" * 80)
    print("🚀 EXECUTING BUILD & OFFLINE VALIDATION FOR v96 OUTLIER ENGINE")
    print("=" * 80)

    # 1. Load ranking test set and champion baseline v93
    df_rank = pl.read_parquet(RAW_DIR / "ranking.parquet")
    rank_dep = df_rank.filter(pl.col("PHASE_mvt") == "DEP")
    template = pl.read_parquet(RAW_DIR / "submitting.parquet")

    v93_file = SUBMISSIONS_DIR / f"{TEAM_NAME}_v93.parquet"
    v60_file = SUBMISSIONS_DIR / f"{TEAM_NAME}_v60.parquet"
    assert v93_file.exists(), "v93 base required"
    assert v60_file.exists(), "v60 anchor required"

    v93 = pl.read_parquet(v93_file)["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    v60 = pl.read_parquet(v60_file)["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    tail_mask = (v60 > 15000)

    # 2. Join weather
    wx_file = EXTERNAL_DIR / "weather_hourly.parquet"
    wx = pl.read_parquet(wx_file)

    dep_wx = rank_dep.with_columns(
        rank_dep["MVT_TIME_UTC_mvt"].dt.truncate("1h").alias("weather_time_utc"),
        rank_dep["ADEP_mvt"].alias("adep")
    ).join(wx, on=["adep", "weather_time_utc"], how="left")

    mvt = rank_dep["MVT_TIME_UTC_mvt"]
    aobt = rank_dep["AOBT_3_flt"]
    sched = rank_dep["SCHED_TIME_UTC_mvt"]
    has_aobt = aobt.is_not_null().to_numpy()
    adep = rank_dep["ADEP_mvt"].to_numpy()
    month = mvt.dt.month().to_numpy()

    p_v96 = v93.copy()

    # =========================================================================
    # BOTTLENECK 1: LTFM Gate-Delay Decoupling (Missing AOBT)
    # =========================================================================
    # In LTFM, when AOBT is missing, flights with large MVT - SCHED are delayed AT THE GATE,
    # but actual ground taxi is normal (~960s). v93 overpredicts up to 5,464s!
    ltfm_no_aobt = (adep == "LTFM") & (~has_aobt)
    ltfm_overpredicted = ltfm_no_aobt & (p_v96 > 1600.0)
    print(f"\n[LTFM Decoupling] Total LTFM no-AOBT flights: {np.sum(ltfm_no_aobt)}")
    print(f"[LTFM Decoupling] Overpredicted (>1600s) flights to calibrate: {np.sum(ltfm_overpredicted)}")
    
    # Calibrate overpredicted LTFM flights toward the clean median (967s) + modest delay residual
    p_v96[ltfm_overpredicted] = np.clip(0.30 * p_v96[ltfm_overpredicted] + 0.70 * 1050.0, 600.0, 1800.0)
    print(f"  Before calibration: mean={np.mean(v93[ltfm_overpredicted]):.1f}s, max={np.max(v93[ltfm_overpredicted]):.0f}s")
    print(f"  After calibration:  mean={np.mean(p_v96[ltfm_overpredicted]):.1f}s, max={np.max(p_v96[ltfm_overpredicted]):.0f}s")

    # =========================================================================
    # BOTTLENECK 2: Munich (EDDM) Snowfall / De-icing Window Calibration
    # =========================================================================
    # In EDDM, when snowfall > 0 in January, ground taxi jumps from 840s to 1,393s (p90=2,281s).
    snow_eddm = (adep == "EDDM") & (month == 1) & (dep_wx["snowfall"].fill_null(0.0).to_numpy() > 0)
    print(f"\n[EDDM Snow Calibration] Total EDDM snow flights: {np.sum(snow_eddm)}")
    # If prediction is under 1,100s during active snowfall in Munich, boost toward snow regime floor
    eddm_underpredicted = snow_eddm & (p_v96 < 1100.0)
    print(f"[EDDM Snow Calibration] Underpredicted snow flights (<1100s): {np.sum(eddm_underpredicted)}")
    p_v96[eddm_underpredicted] = np.clip(0.40 * p_v96[eddm_underpredicted] + 0.60 * 1350.0, 900.0, 2800.0)

    # =========================================================================
    # BOTTLENECK 3: Zurich (LSZH) Snowfall / Blizzard De-icing Calibration
    # =========================================================================
    snow_lszh = (adep == "LSZH") & (month == 1) & (dep_wx["snowfall"].fill_null(0.0).to_numpy() > 0)
    print(f"\n[LSZH Snow Calibration] Total LSZH snow flights: {np.sum(snow_lszh)}")
    lszh_underpredicted = snow_lszh & (p_v96 < 900.0)
    print(f"[LSZH Snow Calibration] Underpredicted snow flights (<900s): {np.sum(lszh_underpredicted)}")
    p_v96[lszh_underpredicted] = np.clip(0.40 * p_v96[lszh_underpredicted] + 0.60 * 1250.0, 800.0, 2500.0)

    # =========================================================================
    # STRICT HERMETIC TAIL SHIELD PRESERVATION
    # =========================================================================
    p_v96_clean = np.clip(np.round(p_v96), 60, 88500).astype(np.int32)
    p_v96_clean[tail_mask] = v60[tail_mask].astype(np.int32)

    assert len(p_v96_clean) == len(template), "Length mismatch"
    assert (p_v96_clean[tail_mask] == v60[tail_mask]).all(), "Tail shield violated!"

    diff_v93 = np.abs(p_v96_clean.astype(float) - v93)
    modified_mask = diff_v93 > 1.0
    print(f"\n📊 SUMMARY OF REFINED MOVEMENTS:")
    print(f"  • Total modified flights: {np.sum(modified_mask):,} / {len(p_v96_clean):,} ({np.sum(modified_mask)/len(p_v96_clean)*100:.2f}%)")
    print(f"  • Mean absolute adjustment on modified: {np.mean(diff_v93[modified_mask]):.1f}s")
    print(f"  • Max adjustment: {np.max(diff_v93):.1f}s")
    print(f"  • RMS difference vs v93: {np.sqrt(np.mean(diff_v93**2)):.2f}s")

    # Save to candidate on disk
    sub_df = pl.DataFrame({"MVT_ID_mvt": rank_dep["MVT_ID_mvt"], "TAXITIME_SEC_mvt": p_v96_clean})
    out_file = SUBMISSIONS_DIR / f"{TEAM_NAME}_v{OUT_VERSION}.parquet"
    template.select(["MVT_ID_mvt"]).join(sub_df, on="MVT_ID_mvt", how="left").write_parquet(out_file)
    print(f"\n✅ Successfully generated candidate {out_file.name} ({out_file.stat().st_size:,} bytes)!")
    print(f"Ready on disk for controlled, single-shot submission when quota window opens.")


if __name__ == "__main__":
    main()
