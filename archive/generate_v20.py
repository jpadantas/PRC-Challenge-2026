"""
Generate Submission v20:
Tripartite Foundation Ensemble (76% v18 + 16% CatBoost FY + 8% LightGBM FY)
with Empirical Missing Bias Calibration across Amsterdam, Zurich, Munich, and Rome.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
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
SUBMISSION_BUCKET = f"prc-2026-{TEAM_NAME}"
S3_ENDPOINT = "s3.opensky-network.org"

def generate_v20(upload: bool = True):
    print("=" * 75)
    print("🚀 GENERATING SUBMISSION v20: TRIPARTITE ENSEMBLE + EMPIRICAL CALIBRATION")
    print("=" * 75)

    # 1. Load ranking test data
    print("Loading test data (ranking.parquet)...")
    rank = pl.read_parquet(RAW_DIR / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
    template = pl.read_parquet(RAW_DIR / "submitting.parquet")
    n_test = len(rank)
    print(f"Total test departures: {n_test:,}")

    # 2. Load model predictions
    print("Loading component model predictions...")
    p18 = pl.read_parquet(SUBMISSIONS_DIR / f"{TEAM_NAME}_v18.parquet")["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    p_cb = np.load(PROJECT_ROOT / "scratch" / "cb_fullyear_test_preds.npy").astype(np.float64)
    p_lgb = np.load(PROJECT_ROOT / "scratch" / "lgb_fullyear_test_preds.npy").astype(np.float64)

    # 3. Tripartite Blend (Convex Variance Minimization)
    w18, w_cb, w_lgb = 0.76, 0.16, 0.08
    print(f"Blending weights: v18 = {w18:.2f}, CatBoost Full-Year = {w_cb:.2f}, LightGBM Full-Year = {w_lgb:.2f}")
    p_blend = w18 * p18 + w_cb * p_cb + w_lgb * p_lgb

    # 4. Empirical Missing Bias Calibration
    print("\nApplying Empirical Missing Bias Calibration...")
    adep = rank["ADEP_mvt"].to_numpy()
    null_aobt = rank["AOBT_3_flt"].is_null().to_numpy()
    d_aobt = (rank["MVT_TIME_UTC_mvt"] - rank["AOBT_3_flt"]).dt.total_seconds().to_numpy()
    p_cal = p_blend.copy()

    # A. Amsterdam (EHAM): 1,206 missing flights
    m_eham = null_aobt & (adep == "EHAM")
    med_eham_before = np.median(p_cal[m_eham])
    scale_eham = 549.0 / med_eham_before
    p_cal[m_eham] = np.clip(p_cal[m_eham] * scale_eham, 60, 3600)
    print(f"  • EHAM Missing ({np.sum(m_eham)} flts): median calibrated {med_eham_before:.0f}s -> {np.median(p_cal[m_eham]):.0f}s")

    # B. Zurich (LSZH): 659 missing flights
    m_lszh = null_aobt & (adep == "LSZH")
    med_lszh_before = np.median(p_cal[m_lszh])
    scale_lszh = 672.0 / med_lszh_before
    p_cal[m_lszh] = np.clip(p_cal[m_lszh] * scale_lszh, 60, 3600)
    print(f"  • LSZH Missing ({np.sum(m_lszh)} flts): median calibrated {med_lszh_before:.0f}s -> {np.median(p_cal[m_lszh]):.0f}s")

    # C. Munich (EDDM): 238 missing flights
    m_eddm = null_aobt & (adep == "EDDM")
    med_eddm_before = np.median(p_cal[m_eddm])
    scale_eddm = 896.0 / med_eddm_before
    p_cal[m_eddm] = np.clip(p_cal[m_eddm] * scale_eddm, 60, 4500)
    print(f"  • EDDM Missing ({np.sum(m_eddm)} flts): median calibrated {med_eddm_before:.0f}s -> {np.median(p_cal[m_eddm]):.0f}s")

    # D. Rome (LIRF): 383 missing flights -> cap at 60,000s (p99 of 12-month ground truth)
    m_lirf = null_aobt & (adep == "LIRF")
    count_capped = np.sum(p_cal[m_lirf] > 60000)
    p_cal[m_lirf] = np.clip(p_cal[m_lirf], 60, 60000)
    print(f"  • LIRF Missing ({np.sum(m_lirf)} flts): capped {count_capped} runaway flights at 60,000s ceiling")

    # 5. Round to integer and enforce physical bounds
    final_preds = np.clip(np.round(p_cal), 5, 88500).astype(np.int32)

    # 6. Construct final submission DataFrame aligned with template
    sub_df = pl.DataFrame({
        "MVT_ID_mvt": rank["MVT_ID_mvt"],
        "TAXITIME_SEC_mvt": final_preds,
    })

    # Left join to preserve exact template order
    final_sub = template.select(["MVT_ID_mvt"]).join(sub_df, on="MVT_ID_mvt", how="left")

    # Rigorous Sanity Checks
    assert len(final_sub) == len(template), f"Length mismatch: {len(final_sub)} vs {len(template)}"
    assert final_sub["TAXITIME_SEC_mvt"].null_count() == 0, f"Found {final_sub['TAXITIME_SEC_mvt'].null_count()} null predictions!"
    assert (final_sub["MVT_ID_mvt"] == template["MVT_ID_mvt"]).all(), "Order mismatch with submitting template!"
    assert (final_sub["TAXITIME_SEC_mvt"] >= 5).all() and (final_sub["TAXITIME_SEC_mvt"] <= 88500).all(), "Predictions outside [5, 88500]!"

    vals = final_sub["TAXITIME_SEC_mvt"].to_numpy()
    print("\n" + "=" * 75)
    print("📊 SUBMISSION v20 FINAL AUDIT METRICS:")
    print(f"  • Total Rows: {len(final_sub):,}")
    print(f"  • Min Taxi Time: {vals.min():.0f}s")
    print(f"  • Median Taxi Time: {np.median(vals):.0f}s")
    print(f"  • Mean Taxi Time: {vals.mean():.1f}s")
    print(f"  • Max Taxi Time: {vals.max():.0f}s")
    print(f"  • Count > 3,600s: {np.sum(vals > 3600):,}")
    print(f"  • Count > 7,200s: {np.sum(vals > 7200):,}")
    print(f"  • Count > 15,000s: {np.sum(vals > 15000):,}")
    print(f"  • Count > 60,000s: {np.sum(vals > 60000):,}")
    print("=" * 75)

    filename = f"{TEAM_NAME}_v20.parquet"
    out_path = SUBMISSIONS_DIR / filename
    final_sub.write_parquet(out_path)
    print(f"\n💾 Saved {filename} to {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    if upload:
        print(f"\n☁️ Uploading {filename} to S3 bucket '{SUBMISSION_BUCKET}'...")
        with open(CREDENTIALS_FILE, "r") as f:
            creds = json.load(f)
        client = Minio(
            S3_ENDPOINT,
            access_key=creds["accessKey"],
            secret_key=creds["secretKey"],
            secure=True,
        )
        client.fput_object(
            bucket_name=SUBMISSION_BUCKET,
            object_name=filename,
            file_path=str(out_path),
        )
        print(f"🎉 Upload successful! S3 URI: s3://{SUBMISSION_BUCKET}/{filename}")


if __name__ == "__main__":
    generate_v20(upload=True)
