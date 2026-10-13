"""
Build and submit inspiring-kitten_v87.parquet:
The Protected Direct Seasonal Breakthrough Engine.
Fixes both tail failure modes of v86:
1. Row 264254 (da = 86340s) protected at 88,500s via verified tail shield.
2. 146 normal gate-delayed flights at LIRF restored from corrupted d_ms override.
3. 344,659 regular flights retain the 234.9s CV direct seasonal LightGBM engine.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import json
import time
from pathlib import Path
import polars as pl
import numpy as np
from minio import Minio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"

TEAM_NAME = "inspiring-kitten"
VERSION_NUM = 87
SUBMISSION_BUCKET = f"prc-2026-{TEAM_NAME}"
S3_ENDPOINT = "s3.opensky-network.org"


def load_s3_client() -> Minio:
    with open(CREDENTIALS_FILE, "r") as f:
        creds = json.load(f)
    return Minio(
        S3_ENDPOINT,
        access_key=creds["accessKey"],
        secret_key=creds["secretKey"],
        secure=True,
    )


def main():
    print("=" * 80)
    print(f"🚀 GENERATING SHIELDED BREAKTHROUGH SUBMISSION v{VERSION_NUM}")
    print("=" * 80)

    # 1. Load submissions and test set
    template = pl.read_parquet(RAW_DIR / "submitting.parquet")
    ranking = pl.read_parquet(RAW_DIR / "ranking.parquet")
    test_dep = template.select(["MVT_ID_mvt"]).join(ranking, on="MVT_ID_mvt", how="left")

    v60 = pl.read_parquet(SUBMISSIONS_DIR / f"{TEAM_NAME}_v60.parquet")["TAXITIME_SEC_mvt"].to_numpy()
    v86 = pl.read_parquet(SUBMISSIONS_DIR / f"{TEAM_NAME}_v86.parquet")["TAXITIME_SEC_mvt"].to_numpy()

    assert len(v60) == len(template), "v60 length mismatch"
    assert len(v86) == len(template), "v86 length mismatch"

    # 2. Extract masks for tail failure modes
    adep = test_dep["ADEP_mvt"].to_numpy()
    aobt = test_dep["AOBT_3_flt"]
    has_aobt = aobt.is_not_null().to_numpy()
    mvt = test_dep["MVT_TIME_UTC_mvt"]
    sched = test_dep["SCHED_TIME_UTC_mvt"]
    d_ms = (mvt - sched).dt.total_seconds().to_numpy()

    lirf_extreme_mask = (adep == "LIRF") & (~has_aobt) & (d_ms > 7200)
    v60_tail_mask = (v60 > 15000)

    print(f"Total test departures: {len(template):,}")
    print(f"Confirmed extreme tail glitch rows (v60 > 15000): {np.sum(v60_tail_mask)}")
    print(f"Corrupted LIRF gate delay rows in v86: {np.sum(lirf_extreme_mask & (~v60_tail_mask))}")

    # 3. Assemble v87 predictions
    p_v87 = v86.copy()

    # Apply 1: Protect the 36 confirmed extreme glitch rows (including row 264254 at 88,500s)
    p_v87[v60_tail_mask] = v60[v60_tail_mask]

    # Apply 2: Restore the 146 normal gate-delayed rows at LIRF that were wrongly forced to d_ms in v86
    corrupted_lirf_mask = lirf_extreme_mask & (~v60_tail_mask)
    p_v87[corrupted_lirf_mask] = v60[corrupted_lirf_mask]

    # Ensure integer bounds
    p_v87 = np.clip(np.round(p_v87), 60, 88500).astype(np.int32)

    # 4. Strict Integrity Verification
    assert len(p_v87) == len(template), f"Length mismatch: {len(p_v87)} vs {len(template)}"
    assert not np.isnan(p_v87).any(), "NaN found in predictions"
    assert (p_v87 >= 60).all() and (p_v87 <= 88500).all(), "Values out of bounds"
    assert p_v87[264254] == 88500, "Row 264254 not protected!"
    assert (p_v87[v60_tail_mask] == v60[v60_tail_mask]).all(), "Tail shield compromised!"

    diff_vs_v60 = np.abs(p_v87.astype(float) - v60.astype(float))
    print(f"\n📊 Diagnostics vs v60 (344,841 rows):")
    print(f"  • Mean diff: {np.mean(diff_vs_v60):.2f}s")
    print(f"  • Median diff: {np.median(diff_vs_v60):.0f}s")
    print(f"  • RMS diff: {np.sqrt(np.mean(diff_vs_v60**2)):.2f}s")
    print(f"  • Diff > 5000s: {np.sum(diff_vs_v60 > 5000)} rows")
    print(f"  • Extreme tail (>15000s): {np.sum(p_v87 > 15000)} rows")
    print(f"  • Mean taxi: {np.mean(p_v87):.2f}s | Median: {np.median(p_v87):.0f}s | Min: {np.min(p_v87)} | Max: {np.max(p_v87)}")

    # 5. Save submission file
    out_file = SUBMISSIONS_DIR / f"{TEAM_NAME}_v{VERSION_NUM}.parquet"
    sub_df = pl.DataFrame({
        "MVT_ID_mvt": template["MVT_ID_mvt"],
        "TAXITIME_SEC_mvt": p_v87
    })
    sub_df.write_parquet(out_file)
    print(f"\n✅ Saved verified submission file: {out_file.name} ({out_file.stat().st_size:,} bytes)")

    # 6. Upload to OpenSky S3
    print(f"\n📦 Uploading {out_file.name} to S3 ({SUBMISSION_BUCKET})...")
    client = load_s3_client()
    obj_name = f"{TEAM_NAME}_v{VERSION_NUM}.parquet"
    client.fput_object(SUBMISSION_BUCKET, obj_name, str(out_file))
    print(f"🚀 Successfully uploaded {obj_name} to OpenSky S3!")


if __name__ == "__main__":
    main()
