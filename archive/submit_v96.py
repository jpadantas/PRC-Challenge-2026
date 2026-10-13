"""
Controlled Single-Shot Submission for v96 (Outlier Calibrator).
Uploads data/submissions/inspiring-kitten_v96.parquet to OpenSky S3.
Awaits official evaluation result and logs to SUBMISSIONS_LOG.md.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from minio import Minio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"
LOG_FILE = PROJECT_ROOT / "SUBMISSIONS_LOG.md"

TEAM_NAME = "inspiring-kitten"
VERSION_NUM = 96
BUCKET = f"prc-2026-{TEAM_NAME}"
S3_ENDPOINT = "s3.opensky-network.org"


def main():
    print("=" * 80)
    print(f"🚀 CONTROLLED SINGLE-SHOT SUBMISSION: {TEAM_NAME}_v{VERSION_NUM}.parquet")
    print("=" * 80)

    filename = f"{TEAM_NAME}_v{VERSION_NUM}.parquet"
    filepath = SUBMISSIONS_DIR / filename
    assert filepath.exists(), f"File not found: {filepath}"

    with open(CREDENTIALS_FILE, "r") as f:
        creds = json.load(f)

    client = Minio(
        S3_ENDPOINT,
        access_key=creds["accessKey"],
        secret_key=creds["secretKey"],
        secure=True,
    )

    print(f"📦 Uploading {filename} ({filepath.stat().st_size:,} bytes) to {BUCKET}...")
    client.fput_object(BUCKET, filename, str(filepath))
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
            try:
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"\n| `v{VERSION_NUM}` | `{now_str}` | `{status}` | **`{score:.4f}s`** | Controlled Single-Shot Outlier Engine (LTFM/EDDM/LSZH) |\n")
                print("✅ Logged score to SUBMISSIONS_LOG.md")
            except Exception as e_log:
                print(f"Could not log: {e_log}")
            break
        except Exception:
            elapsed = time.time() - t0
            print(f"   Waiting for result... ({elapsed:.0f}s elapsed)", end="\r", flush=True)

    if score is None:
        print("\n⚠️ Result json not returned within 600s.")


if __name__ == "__main__":
    main()
