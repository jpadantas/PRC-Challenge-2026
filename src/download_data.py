"""
Download all PRC Data Challenge 2026 datasets from OpenSky S3.

Usage:
    python src/download_data.py
"""

import json
import os
from pathlib import Path
from minio import Minio

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# S3 Configuration
S3_ENDPOINT = "s3.opensky-network.org"
COMPETITION_BUCKET = "competition-data"
TEAM_BUCKET = "prc-2026-inspiring-kitten"


def load_credentials() -> dict:
    """Load S3 credentials from credentials.json."""
    with open(CREDENTIALS_FILE, "r") as f:
        creds = json.load(f)
    return creds["accessKey"], creds["secretKey"]


def create_client(access_key: str, secret_key: str) -> Minio:
    """Create a MinIO client for OpenSky S3."""
    return Minio(
        S3_ENDPOINT,
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
    )


def list_buckets(client: Minio):
    """List all accessible buckets."""
    print("\n📦 Accessible Buckets:")
    print("-" * 50)
    buckets = client.list_buckets()
    for bucket in buckets:
        print(f"  • {bucket.name}  (created: {bucket.creation_date})")
    return [b.name for b in buckets]


def download_bucket_contents(client: Minio, bucket_name: str, target_dir: Path):
    """Download all files from a bucket to a local directory."""
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Downloading from bucket: {bucket_name}")
    print(f"   Target directory: {target_dir}")
    print("-" * 50)

    objects = list(client.list_objects(bucket_name, recursive=True))

    if not objects:
        print("   (empty bucket)")
        return

    for obj in objects:
        local_path = target_dir / obj.object_name
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already downloaded and same size
        if local_path.exists() and local_path.stat().st_size == obj.size:
            print(f"   ⏭️  {obj.object_name} ({obj.size / 1e6:.1f} MB) — already exists")
            continue

        size_mb = obj.size / 1e6
        print(f"   ⬇️  {obj.object_name} ({size_mb:.1f} MB) ...", end="", flush=True)
        client.fget_object(bucket_name, obj.object_name, str(local_path))
        print(" ✅")

    print(f"\n   Done! Files saved to: {target_dir}")


def main():
    print("=" * 60)
    print("  PRC Data Challenge 2026 — Data Downloader")
    print("  Team: inspiring-kitten")
    print("=" * 60)

    # Load credentials
    access_key, secret_key = load_credentials()
    print(f"\n🔑 Access Key: {access_key[:8]}...")

    # Create S3 client
    client = create_client(access_key, secret_key)

    # List buckets
    bucket_names = list_buckets(client)

    # Download competition data (training + ranking + submitting)
    if COMPETITION_BUCKET in bucket_names:
        download_bucket_contents(client, COMPETITION_BUCKET, RAW_DATA_DIR)
    else:
        print(f"\n⚠️  Bucket '{COMPETITION_BUCKET}' not found. Available: {bucket_names}")
        # Try to find the right bucket name
        for name in bucket_names:
            if "compet" in name.lower() or "data" in name.lower() or "challenge" in name.lower():
                print(f"   Trying: {name}")
                download_bucket_contents(client, name, RAW_DATA_DIR)
                break

    # List team bucket contents (for submissions/results)
    if TEAM_BUCKET in bucket_names:
        print(f"\n📋 Team Bucket ({TEAM_BUCKET}) contents:")
        print("-" * 50)
        objects = list(client.list_objects(TEAM_BUCKET, recursive=True))
        if objects:
            for obj in objects:
                print(f"   • {obj.object_name} ({obj.size / 1e6:.1f} MB)")
        else:
            print("   (empty — no submissions yet)")

    print("\n✅ All done!")


if __name__ == "__main__":
    main()
