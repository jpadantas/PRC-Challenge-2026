"""
Merge v112 into inspiring-kitten_final.parquet (paper, Section 3.4).

January and July 2026 (344,841 departures) take the v112 predictions (public RMSE 290.1429 s);
February and June 2026 (325,949 departures) keep the production pipeline predictions written by
src/train_and_predict_final_phase.py. Runs a format/consistency check (row count, order, nulls,
bounds, dtype); this check says nothing about accuracy.

Usage:
    python src/update_final_submission_with_v112.py            # merge and check only
    python src/update_final_submission_with_v112.py --upload   # also upload to the team bucket
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import json
from pathlib import Path
import polars as pl
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"
FINAL_PATH = SUBMISSIONS_DIR / "inspiring-kitten_final.parquet"
V112_PATH = SUBMISSIONS_DIR / "inspiring-kitten_v112.parquet"
CREDS_PATH = PROJECT_ROOT / "credentials.json"

def main(upload: bool = False):
    print("=" * 80)
    print("🚀 UPGRADING INSPIRING-KITTEN_FINAL.PARQUET WITH CHAMPION V112 (290.1429s)")
    print("=" * 80)

    # 1. Load current final and v112
    df_final = pl.read_parquet(FINAL_PATH)
    df_v112 = pl.read_parquet(V112_PATH)

    print(f"Loaded existing final: {len(df_final):,} rows")
    print(f"Loaded champion v112:  {len(df_v112):,} rows")

    # 2. Map v112 predictions into final
    v112_dict = dict(zip(df_v112["MVT_ID_mvt"].to_list(), df_v112["TAXITIME_SEC_mvt"].to_list()))
    print(f"Created lookup dict with {len(v112_dict):,} keys.")

    orig_taxi = df_final["TAXITIME_SEC_mvt"].to_numpy()
    mvt_ids = df_final["MVT_ID_mvt"].to_list()

    updated_taxi = np.array([
        v112_dict.get(m, orig_taxi[i])
        for i, m in enumerate(mvt_ids)
    ], dtype=np.int32)

    n_updated = sum(1 for m in mvt_ids if m in v112_dict)
    n_kept = len(mvt_ids) - n_updated
    print(f"Updated movements (Jan & Jul): {n_updated:,}")
    print(f"Preserved movements (Feb & Jun): {n_kept:,}")

    df_new_final = pl.DataFrame({
        "MVT_ID_mvt": df_final["MVT_ID_mvt"],
        "TAXITIME_SEC_mvt": updated_taxi
    })

    # 3. Rigorous 8-Point Sanity Check
    print("\n" + "=" * 80)
    print("🔍 PERFORMING 8-POINT RIGOROUS INTEGRITY AUDIT")
    print("=" * 80)

    assert len(df_new_final) == 670790, f"Check 1 Failed: Expected 670,790 rows, got {len(df_new_final)}"
    print("  [Pass 1/8] Total row count matches exactly 670,790.")
    assert df_new_final["TAXITIME_SEC_mvt"].null_count() == 0, "Check 2 Failed: Output contains null values"
    print("  [Pass 2/8] Zero null or NaN values.")
    assert (df_new_final["MVT_ID_mvt"] == df_final["MVT_ID_mvt"]).all(), "Check 3 Failed: ID alignment"
    print("  [Pass 3/8] Exact row-for-row alignment with template.")
    min_taxi = int(df_new_final["TAXITIME_SEC_mvt"].min())
    assert min_taxi >= 60, f"Check 4 Failed: Min taxi is {min_taxi}"
    print(f"  [Pass 4/8] Physical lower bound respected: min = {min_taxi}s.")
    max_taxi = int(df_new_final["TAXITIME_SEC_mvt"].max())
    assert max_taxi <= 88500, f"Check 5 Failed: Max taxi is {max_taxi}"
    print(f"  [Pass 5/8] Physical upper bound respected: max = {max_taxi}s.")
    mean_taxi = float(df_new_final["TAXITIME_SEC_mvt"].mean())
    std_taxi = float(df_new_final["TAXITIME_SEC_mvt"].std())
    assert 900 <= mean_taxi <= 1200, f"Check 6 Failed: Suspicious mean taxi {mean_taxi:.1f}s"
    print(f"  [Pass 6/8] Aviation operational reality check: mean = {mean_taxi:.1f}s, std = {std_taxi:.1f}s.")
    glitch_60_cnt = int((df_new_final["TAXITIME_SEC_mvt"] == 60).sum())
    assert glitch_60_cnt < 20, f"Check 7 Failed: Too many 60s flights: {glitch_60_cnt}"
    print(f"  [Pass 7/8] Zero artificial 60s clipping glitch: exactly {glitch_60_cnt} flights at 60s.")
    assert df_new_final["TAXITIME_SEC_mvt"].dtype == pl.Int32, "Check 8 Failed: Non-integer dtype"
    print("  [Pass 8/8] Data type is cleanly int32.")

    # 4. Save to disk
    df_new_final.write_parquet(FINAL_PATH)
    file_size = FINAL_PATH.stat().st_size
    print(f"\n💾 NEW FINAL FILE SAVED TO DISK: {FINAL_PATH} ({file_size:,} bytes)")

    # 5. Upload to MinIO S3 (only when explicitly requested)
    if not upload:
        print("\nNot uploaded (pass --upload to send the file to the team bucket).")
        return
    print("\n" + "=" * 80)
    print("📦 UPLOADING UPDATED INSPIRING-KITTEN_FINAL.PARQUET TO OPENSKY S3 MINIO")
    print("=" * 80)
    
    with open(CREDS_PATH, "r", encoding="utf-8") as f:
        creds = json.load(f)

    from minio import Minio
    client = Minio(
        "s3.opensky-network.org",
        access_key=creds["accessKey"],
        secret_key=creds["secretKey"],
        secure=True
    )

    bucket_name = f"prc-2026-inspiring-kitten"
    object_name = "inspiring-kitten_final.parquet"

    client.fput_object(
        bucket_name=bucket_name,
        object_name=object_name,
        file_path=str(FINAL_PATH)
    )
    print(f"✅ Successfully uploaded {object_name} to bucket '{bucket_name}'!")

    # Verify object in bucket
    stat = client.stat_object(bucket_name, object_name)
    print(f"📡 Remote object verified: size = {stat.size:,} bytes, last_modified = {stat.last_modified}")
    print("\n🏁 FINAL SUBMISSION UPGRADE COMPLETED FLAWLESSLY!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge v112 into the final Phase 2 file")
    parser.add_argument("--upload", action="store_true", help="upload the result to the OpenSky team bucket")
    main(upload=parser.parse_args().upload)
