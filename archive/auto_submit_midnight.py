"""
Automated Dispatcher & Result Watcher for PRC Data Challenge 2026.

Waits for the daily submission quota reset at 00:00:15 UTC, uploads
inspiring-kitten_v{version}.parquet, monitors for official evaluation result,
and updates SUBMISSIONS_LOG.md with the live score and leaderboard rank.
"""

import json
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from minio import Minio

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"
LOG_FILE = PROJECT_ROOT / "SUBMISSIONS_LOG.md"

TEAM_NAME = "inspiring-kitten"
BUCKET = f"prc-2026-{TEAM_NAME}"
S3_ENDPOINT = "s3.opensky-network.org"
LEADERBOARD_URL = "https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard"


def get_client() -> Minio:
    with open(CREDENTIALS_FILE, "r") as f:
        creds = json.load(f)
    return Minio(
        S3_ENDPOINT,
        access_key=creds["accessKey"],
        secret_key=creds["secretKey"],
        secure=True,
    )


def wait_for_reset():
    now = datetime.now(timezone.utc)
    if now.hour == 0 and now.minute < 5:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Already past 00:00 UTC today. Proceeding immediately!")
        return

    # Compute next 00:00:15 UTC
    tomorrow = now.date() + timedelta(days=1)
    next_midnight = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 15, tzinfo=timezone.utc)

    seconds_to_wait = (next_midnight - now).total_seconds()
    print(f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target reset time: {next_midnight.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Waiting {seconds_to_wait/3600:.2f} hours ({seconds_to_wait/60:.1f} minutes) until 00:00:15 UTC for quota reset...")
    
    # Sleep in intervals with periodic heartbeat
    while True:
        now = datetime.now(timezone.utc)
        if now >= next_midnight:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Quota reset time reached (00:00 UTC)! Proceeding to upload.")
            break
        remaining = (next_midnight - now).total_seconds()
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        seconds = int(remaining % 60)
        print(f"  [Heartbeat {now.strftime('%H:%M:%S')}] {hours}h {minutes}m {seconds}s remaining until reset...", end="\r", flush=True)
        time.sleep(min(60, remaining))


def upload_submission(version: int = 12) -> str:
    filename = f"{TEAM_NAME}_v{version}.parquet"
    filepath = SUBMISSIONS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Submission file not found: {filepath}")

    print(f"\n🚀 Uploading {filename} ({filepath.stat().st_size / 1e6:.2f} MB) to s3://{BUCKET}/{filename}...")
    client = get_client()
    client.fput_object(
        bucket_name=BUCKET,
        object_name=filename,
        file_path=str(filepath),
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Upload successful! Waiting for evaluation server to process...")
    return filename


def watch_result(filename: str, max_wait_sec: int = 600) -> dict:
    result_key = f"{filename}_result.json"
    client = get_client()
    t0 = time.time()
    
    print(f"Polling for {result_key}...")
    while time.time() - t0 < max_wait_sec:
        try:
            resp = client.get_object(BUCKET, result_key)
            result_data = json.loads(resp.read().decode("utf-8"))
            print("\nEvaluation result received!")
            print(json.dumps(result_data, indent=2))
            return result_data
        except Exception:
            time.sleep(5)
            print(".", end="", flush=True)

    print(f"\nTimeout waiting for {result_key} after {max_wait_sec}s.")
    return {}


def fetch_leaderboard():
    cursor = None
    all_items = []
    page = 0
    while page < 10:
        page += 1
        url = LEADERBOARD_URL
        if cursor:
            url += f"?cursor={cursor}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            items = data.get("items", [])
            all_items.extend(items)
            cursor = data.get("nextCursor")
            if not cursor or not items:
                break
        except Exception as e:
            print(f"Leaderboard query note: {e}")
            break

    best_scores = {}
    for it in all_items:
        name = it.get("teamName")
        sc = it.get("score")
        if sc is not None:
            if name not in best_scores or sc < best_scores[name]["score"]:
                best_scores[name] = {"score": sc, "file": it.get("filename")}

    return sorted(best_scores.items(), key=lambda x: x[1]["score"])


def update_log_and_leaderboard(version: int, result: dict):
    score = result.get("score")
    status = result.get("status")
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\nSubmission v{version} Result: Status={status}, Score={score}")

    leaderboard = fetch_leaderboard()
    my_rank = None
    for rank, (team, info) in enumerate(leaderboard, 1):
        if team == TEAM_NAME:
            my_rank = rank
            break

    print("\nLive Leaderboard Standings:")
    for rank, (team, info) in enumerate(leaderboard[:10], 1):
        marker = " [YOU!]" if team == TEAM_NAME else ""
        print(f"  {rank:2d}. {team:25s}: {info['score']:.4f}{marker}")
    if my_rank and my_rank > 10:
        print(f"  ...\n  {my_rank:2d}. {TEAM_NAME:25s}: {score:.4f} [YOU!]")

    # Append to SUBMISSIONS_LOG.md
    if LOG_FILE.exists() and score is not None:
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                content = f.read()

            if version == 13:
                arch_title = "**Decoupled Two-Engine Mega-Blend**"
                target_form = "Engine 1 (Valid AOBT) + Engine 2 (Missing Foundation)"
                innov = "Unified Missing Foundation (22.4k rows, 12 months) + High-Precision Valid Blend"
                cv_score = "217.2s (clean/global Jan 25)"
                details = f"""- **Architecture:** Decoupled Two-Engine Mega-Blend
  - **Engine 1 (Valid AOBT, 98.3%):** High-Precision Mega-Blend (50% Seasonal XGBoost + 35% Full-Year Foundation XGBoost + 15% Seasonal LightGBM)
  - **Engine 2 (Missing AOBT, 1.7%):** Unified Missing-AOBT Foundation Model trained across all 12 months (22,470 departures) combining LightGBM + XGBoost with schedule delay physics
  - **Local Validation Breakthrough:** Out-of-sample RMSE on Jan 2025 dropped from 329.37s to 217.20s (-112.17s)"""
            else:
                arch_title = "**Two-Regime Physics Foundation**"
                target_form = "Unclipped Physical Horizon"
                innov = "Runway wind decomposition (head/cross) + ICAO Doc 4444 separation matrix"
                cv_score = "165.6s (clean)"
                details = f"""- **Architecture:** Two-Regime Aviation Physics Foundation
  - Runway Vector Wind Decomposition: Active magnetic heading vs 10m wind direction (headwind, crosswind, tailwind flags)
  - ICAO Doc 4444 Wake Separation Minima (J, H, M, L) and forced ground hold delay
  - Full-Year Foundation XGBoost + Seasonal XGBoost + Seasonal LightGBM mega-blend
  - Calibrated LIRF missing AOBT minimax ceiling at 60,000s"""

            new_table_row = f"| **`v{version}`** | {now_utc[:16]} | {arch_title} | {target_form} | {innov} | {cv_score} | **`{score:.4f}`** | Official Result |\n"
            
            # Find insertion point after v12 (or v11) table row
            prev_tag = f"| **`v{version-1}`**"
            if prev_tag in content:
                parts = content.split(prev_tag)
                line_end = parts[1].find("\n") + 1
                updated = parts[0] + prev_tag + parts[1][:line_end] + new_table_row + parts[1][line_end:]
            elif "| **`v11`**" in content:
                parts = content.split("| **`v11`**")
                line_end = parts[1].find("\n") + 1
                updated = parts[0] + "| **`v11`**" + parts[1][:line_end] + new_table_row + parts[1][line_end:]
            else:
                updated = content + f"\n{new_table_row}\n"

            # Add experiment log section at bottom
            experiment_section = f"""
---

### Submission `inspiring-kitten_v{version}.parquet` ({arch_title.replace('*', '')})
- **Timestamp:** {now_utc}
{details}
- **Official Evaluation Result:**
  - **Status:** {status}
  - **Score (RMSE):** **`{score:.4f}`**
  - **Leaderboard Rank:** #{my_rank}
"""
            updated += experiment_section
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write(updated)
            print(f"Updated {LOG_FILE.name} successfully with v{version} score: {score:.4f}!")
        except Exception as e:
            print(f"Could not update log file: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PRC Data Challenge 2026 Midnight Dispatcher")
    parser.add_argument("--now", action="store_true", help="Upload immediately without waiting for midnight")
    parser.add_argument("--version", type=int, default=22, help="Submission version number (default: 22)")
    parser.add_argument("--dry-run", action="store_true", help="Test connectivity without uploading")
    args = parser.parse_args()

    if args.dry_run:
        print("Testing MinIO and OpenSky connectivity...")
        client = get_client()
        objs = list(client.list_objects(BUCKET))
        print(f"Bucket '{BUCKET}' contains {len(objs)} objects.")
        print("Fetching leaderboard...")
        lb = fetch_leaderboard()
        print(f"Fetched {len(lb)} teams on leaderboard. Top team: {lb[0][0]} ({lb[0][1]['score']:.4f})")
        print("Dry run complete! Everything is operational.")
        return

    if not args.now:
        wait_for_reset()

    filename = upload_submission(version=args.version)
    res = watch_result(filename)
    update_log_and_leaderboard(args.version, res)


if __name__ == "__main__":
    main()
