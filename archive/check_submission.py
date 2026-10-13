"""
Check submission status and live leaderboard.
"""

import json
import urllib.request
from pathlib import Path
from minio import Minio

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
TEAM_NAME = "inspiring-kitten"
BUCKET = f"prc-2026-{TEAM_NAME}"
LEADERBOARD_URL = "https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard"


def check_bucket():
    print(f"\n📦 Checking S3 Bucket: {BUCKET}")
    with open(CREDENTIALS_FILE, "r") as f:
        creds = json.load(f)
    client = Minio(
        "s3.opensky-network.org",
        access_key=creds["accessKey"],
        secret_key=creds["secretKey"],
        secure=True,
    )
    objects = list(client.list_objects(BUCKET))
    print(f"Total objects in bucket: {len(objects)}")
    for obj in objects:
        print(f"  • {obj.object_name:35s} ({obj.size:,} bytes, last modified: {obj.last_modified})")


def check_leaderboard():
    print(f"\n🏆 Fetching Live Leaderboard...")
    all_items = []
    cursor = None
    while True:
        url = LEADERBOARD_URL + (f"?cursor={cursor}" if cursor else "")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        all_items.extend(data.get("items", []))
        cursor = data.get("nextCursor")
        if not cursor:
            break

    team_subs = [x for x in all_items if x.get("teamName") == TEAM_NAME]

    if team_subs:
        print(f"\n🎉 Submissions for {TEAM_NAME} (Total: {len(team_subs)}):")
        for s in sorted(team_subs, key=lambda x: x.get("score") if x.get("score") is not None else 999999):
            print(f"  • {s.get('filename')}: Score = {s.get('score')} (processed at {s.get('processedAt')})")
    else:
        print(f"\nℹ️ No processed submissions found on leaderboard for {TEAM_NAME} yet.")

    # Show top 5 and our rank
    best_scores = {}
    for item in all_items:
        name = item.get("teamName")
        score = item.get("score")
        if score is not None:
            if name not in best_scores or score < best_scores[name]:
                best_scores[name] = score

    sorted_teams = sorted(best_scores.items(), key=lambda x: x[1])
    print(f"\nLeaderboard Summary: {len(sorted_teams)} unique teams, {len(all_items)} total submissions.")
    print("Top 10 Teams:")
    for rank, (name, sc) in enumerate(sorted_teams[:10], 1):
        marker = " 👈 (YOU!)" if name == TEAM_NAME else ""
        print(f"  {rank:2d}. {name:25s}: {sc:.4f}{marker}")

    if TEAM_NAME in best_scores:
        my_rank = [i + 1 for i, (name, _) in enumerate(sorted_teams) if name == TEAM_NAME][0]
        print(f"\n⭐ {TEAM_NAME} Rank: #{my_rank} of {len(sorted_teams)} teams | Best Score: {best_scores[TEAM_NAME]:.4f}")


if __name__ == "__main__":
    check_bucket()
    check_leaderboard()
