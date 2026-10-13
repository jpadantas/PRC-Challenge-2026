"""
Assemble and Submit v100: Quad-Paradigm Dual-Engine Master Ensemble.
Blends:
- 86% v99 (Current Champion 291.0341s)
- 8% 10-Airport Dual-Engine Specialists (LightGBM + CatBoost GPU, -3.04s holdout drop)
- 4% Deep Tabular ResNet (Entity Embeddings Neural Network)
- 2% Calibrated XGBoost GPU (59-Feature CUDA Hist Engine)
Strict 36-row hermetic tail shield strictly preserved.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import polars as pl
import numpy as np
from minio import Minio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"
SCRATCH_DIR = PROJECT_ROOT / "scratch"
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
LOG_FILE = PROJECT_ROOT / "SUBMISSIONS_LOG.md"

TEAM_NAME = "inspiring-kitten"
VERSION_NUM = 100
BUCKET = f"prc-2026-{TEAM_NAME}"
S3_ENDPOINT = "s3.opensky-network.org"


def main():
    print("=" * 80)
    print(f"🚀 ASSEMBLING & SUBMITTING v{VERSION_NUM}: QUAD-PARADIGM DUAL-ENGINE MASTER ENSEMBLE")
    print("=" * 80)

    template = pl.read_parquet(RAW_DIR / "submitting.parquet")
    rank = pl.read_parquet(RAW_DIR / "ranking.parquet")
    dep = rank.filter(pl.col("PHASE_mvt") == "DEP")

    v99 = pl.read_parquet(SUBMISSIONS_DIR / f"{TEAM_NAME}_v99.parquet")["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    v60 = pl.read_parquet(SUBMISSIONS_DIR / f"{TEAM_NAME}_v60.parquet")["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    p_dual_raw = np.load(SCRATCH_DIR / "dual_engine_specialists_test_preds.npy").astype(float)
    res_raw = np.load(SCRATCH_DIR / "resnet_offset_nextgen_test_preds.npy").astype(float)
    xgb_raw = np.load(SCRATCH_DIR / "xgboost_arrival_test_preds.npy").astype(float)

    mvt = dep["MVT_TIME_UTC_mvt"]
    aobt = dep["AOBT_3_flt"]
    has_aobt = aobt.is_not_null().to_numpy()
    da = np.where(has_aobt, (mvt - aobt).dt.total_seconds().to_numpy(), np.nan)
    clean_mask = has_aobt & (da >= 120.0) & (da <= 7200.0)
    tail_mask = (v60 > 15000)

    # Route irregulars to champion v99 and shield tail
    p_dual_clean = np.where(clean_mask, p_dual_raw, v99)
    p_dual_clean[tail_mask] = v60[tail_mask]

    taxi_res = np.where(clean_mask, da - res_raw, v99)
    taxi_res[tail_mask] = v60[tail_mask]

    taxi_xgb = np.where(clean_mask, xgb_raw, v99)
    taxi_xgb[tail_mask] = v60[tail_mask]

    # Quad-Paradigm Master Blend: 86% v99 + 8% Dual-Engine + 4% ResNet + 2% XGBoost
    w_99, w_dual, w_res, w_xgb = 0.86, 0.08, 0.04, 0.02
    p_v100 = w_99 * v99 + w_dual * p_dual_clean + w_res * taxi_res + w_xgb * taxi_xgb

    p_v100_clean = np.clip(np.round(p_v100), 60, 88500).astype(np.int32)
    p_v100_clean[tail_mask] = v60[tail_mask].astype(np.int32)

    assert len(p_v100_clean) == len(template), "Length mismatch"
    assert (p_v100_clean[tail_mask] == v60[tail_mask]).all(), "Tail shield violated!"

    diff_v99 = np.abs(p_v100_clean.astype(float) - v99)
    print(f"\n📊 Diagnostics v{VERSION_NUM} vs v99 (Champion 291.0341s):")
    print(f"  • Mean diff: {np.mean(diff_v99):.2f}s")
    print(f"  • Median diff: {np.median(diff_v99):.0f}s")
    print(f"  • RMS diff: {np.sqrt(np.mean(diff_v99**2)):.2f}s")
    print(f"  • Max diff: {np.max(diff_v99):.1f}s")
    print(f"  • Tail rows strictly preserved: {np.sum(tail_mask)}")

    out_file = SUBMISSIONS_DIR / f"{TEAM_NAME}_v{VERSION_NUM}.parquet"
    sub_df = pl.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"], "TAXITIME_SEC_mvt": p_v100_clean})
    template.select(["MVT_ID_mvt"]).join(sub_df, on="MVT_ID_mvt", how="left").write_parquet(out_file)
    print(f"\n✅ Successfully generated {out_file.name} ({out_file.stat().st_size:,} bytes)!")

    # Controlled Single-Shot Upload to OpenSky
    with open(CREDENTIALS_FILE, "r") as f:
        creds = json.load(f)

    client = Minio(
        S3_ENDPOINT,
        access_key=creds["accessKey"],
        secret_key=creds["secretKey"],
        secure=True,
    )

    filename = f"{TEAM_NAME}_v{VERSION_NUM}.parquet"
    print(f"\n📦 Uploading {filename} to OpenSky S3 ({BUCKET})...")
    client.fput_object(BUCKET, filename, str(out_file))
    print(f"Upload complete. Waiting for {filename}_result.json...")

    result_key = f"{filename}_result.json"
    t0 = time.time()
    score = None
    for attempt in range(120):
        time.sleep(5)
        try:
            resp = client.get_object(BUCKET, result_key)
            data = json.loads(resp.read().decode("utf-8"))
            score = float(data.get("score", 0.0))
            status = data.get("status", "UNKNOWN")
            print(f"\n" + "=" * 80)
            print(f"🎯 OFFICIAL LEADERBOARD SCORE v{VERSION_NUM}: {score:.4f}s RMSE (Status: {status})")
            print(f"   Evaluation took {time.time() - t0:.1f} seconds")
            print("=" * 80)

            # Log to SUBMISSIONS_LOG.md
            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            log_entry = (
                f"\n| `v{VERSION_NUM}` | `{now_utc}` | `{status}` | **`{score:.4f}s`** | "
                f"Quad-Paradigm Dual-Engine Master (86% v99 + 8% Dual-Engine + 4% ResNet + 2% XGB) |\n"
            )
            with open(LOG_FILE, "a", encoding="utf-8") as lf:
                lf.write(log_entry)
            print(f"✅ Logged score to {LOG_FILE.name}")
            return score
        except Exception:
            elapsed = time.time() - t0
            print(f"   Waiting for result... ({elapsed:.0f}s elapsed)", end="\r", flush=True)

    print(f"\n⚠️ Timed out waiting for evaluation result after {time.time() - t0:.1f}s")
    return None


if __name__ == "__main__":
    main()
