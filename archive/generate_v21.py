"""
Generate Final Daily Submission v21 (Sept 18, 5/5):
Analytical Optimal Convex Blend (82% v18 + 18% CatBoost Full-Year)
with Tail-Preserving Horizon on Extreme Radar Delays (100% v18 above 15,000s).
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
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


def generate_v21(upload: bool = True):
    print("=" * 75)
    print("👑 GENERATING FINAL SUBMISSION v21: ANALYTICAL OPTIMAL BLEND (SHOT 5/5)")
    print("=" * 75)

    # 1. Load ranking test data & template
    print("Loading test data (ranking.parquet) & submission template...")
    rank = pl.read_parquet(RAW_DIR / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
    template = pl.read_parquet(RAW_DIR / "submitting.parquet")
    n_test = len(rank)
    print(f"Total test departures: {n_test:,}")

    # 2. Load component predictions
    print("Loading component model predictions...")
    p18 = pl.read_parquet(SUBMISSIONS_DIR / f"{TEAM_NAME}_v18.parquet")["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    p19 = pl.read_parquet(SUBMISSIONS_DIR / f"{TEAM_NAME}_v19.parquet")["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    p_cb = np.load(PROJECT_ROOT / "scratch" / "cb_fullyear_test_preds.npy").astype(np.float64)

    # 3. Identify extreme radar delay horizon (32 flights in LIRF with p18 > 15,000s)
    high_mask = p18 > 15000
    n_high = np.sum(high_mask)
    print(f"\nExtreme Radar Delay Horizon Analysis:")
    print(f"  • Found {n_high} extreme flights (> 15,000s, max {p18.max():.0f}s) in LIRF.")
    print(f"  • Preserving 100% v18 predictions on these {n_high} flights to prevent tail underprediction.")

    # 4. Analytical Optimal Convex Blend on all 344,809 normal flights
    w_cb = 0.18
    w_18 = 0.82
    print(f"\nConvex Risk-Minimization Blend on normal flights (<= 15,000s):")
    print(f"  • v18 weight:        {w_18:.2f}")
    print(f"  • CatBoost FY weight:{w_cb:.2f} (optimal analytical variance minimum)")

    final_preds_float = w_18 * p18 + w_cb * p_cb
    # Preserve 100% v18 on extreme radar delay horizon
    final_preds_float[high_mask] = p18[high_mask]

    # 5. Round to integer and enforce physical bounds [5, 88500]
    final_preds = np.clip(np.round(final_preds_float), 5, 88500).astype(np.int32)

    # 6. Construct final DataFrame aligned with template
    sub_df = pl.DataFrame({
        "MVT_ID_mvt": rank["MVT_ID_mvt"],
        "TAXITIME_SEC_mvt": final_preds,
    })

    final_sub = template.select(["MVT_ID_mvt"]).join(sub_df, on="MVT_ID_mvt", how="left")

    # 7. Rigorous Sanity Checks
    assert len(final_sub) == len(template), f"Length mismatch: {len(final_sub)} vs {len(template)}"
    assert final_sub["TAXITIME_SEC_mvt"].null_count() == 0, f"Found {final_sub['TAXITIME_SEC_mvt'].null_count()} null predictions!"
    assert (final_sub["MVT_ID_mvt"] == template["MVT_ID_mvt"]).all(), "Order mismatch with submitting template!"
    assert (final_sub["TAXITIME_SEC_mvt"] >= 5).all() and (final_sub["TAXITIME_SEC_mvt"] <= 88500).all(), "Predictions outside [5, 88500]!"

    vals = final_sub["TAXITIME_SEC_mvt"].to_numpy()
    print("\n" + "=" * 75)
    print("📊 SUBMISSION v21 FINAL AUDIT METRICS:")
    print(f"  • Total Rows: {len(final_sub):,}")
    print(f"  • Min Taxi Time: {vals.min():.0f}s")
    print(f"  • Median Taxi Time: {np.median(vals):.0f}s")
    print(f"  • Mean Taxi Time: {vals.mean():.1f}s")
    print(f"  • Max Taxi Time: {vals.max():.0f}s")
    print(f"  • Count > 3,600s: {np.sum(vals > 3600):,}")
    print(f"  • Count > 7,200s: {np.sum(vals > 7200):,}")
    print(f"  • Count > 15,000s: {np.sum(vals > 15000):,}")
    print(f"  • Count > 60,000s: {np.sum(vals > 60000):,}")
    print(f"  • Correlation with v19 (Record: 312.96s): {np.corrcoef(vals, p19)[0, 1]:.6f}")
    print(f"  • Mean Absolute Difference vs v19: {np.mean(np.abs(vals - p19)):.2f}s")
    print("=" * 75)

    filename = f"{TEAM_NAME}_v21.parquet"
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
    generate_v21(upload=True)
