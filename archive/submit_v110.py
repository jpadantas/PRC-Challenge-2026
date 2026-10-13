"""
Submit v110 to OpenSky S3 bucket and monitor live evaluation response.
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
VERSION_NUM = 110
BUCKET = f"prc-2026-{TEAM_NAME}"
S3_ENDPOINT = "s3.opensky-network.org"


def main():
    print("=" * 80)
    print(f"🚀 SUBMITTING TO OPENSKY S3: {TEAM_NAME}_v{VERSION_NUM}.parquet")
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
    result_data = None

    for attempt in range(60):
        time.sleep(5)
        try:
            resp = client.get_object(BUCKET, result_key)
            result_data = json.loads(resp.read().decode("utf-8"))
            print(f"\n" + "=" * 80)
            print(f"📡 SERVER RESPONSE for v{VERSION_NUM}:")
            print(json.dumps(result_data, indent=2))
            print("=" * 80)

            score = result_data.get("score")
            status = result_data.get("status", "UNKNOWN")
            error = result_data.get("error")

            if score is not None:
                print(f"\n🎯 OFFICIAL LEADERBOARD SCORE v{VERSION_NUM}: {score:.4f}s RMSE (Status: {status})")
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"\n| `v{VERSION_NUM}` | `{now_str}` | `{status}` | **`{score:.4f}s`** | Calibrated 50/50 GBDT-v104 Blend with Hermetic Tail Shield |\n")
                print("✅ Score logged to SUBMISSIONS_LOG.md")
            else:
                print(f"\n⚠️ Result returned without score: status={status}, error={error}")
            break
        except Exception:
            elapsed = time.time() - t0
            print(f"   Waiting for evaluation... ({elapsed:.0f}s elapsed)", end="\r", flush=True)

    if result_data is None:
        print("\n⏳ Result json not returned within 300s. Backend may be processing queue.")


if __name__ == "__main__":
    main()
