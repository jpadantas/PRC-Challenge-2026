"""
Download real airport METAR/ASOS station observations from Iowa State Mesonet
for the 10 PRC Data Challenge airports across 2025 (training) and 2026 (testing).
"""

import io
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
import pandas as pd
import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

AIRPORTS = [
    "EDDF", "EDDM", "EGLL", "EHAM", "LEBL",
    "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"
]

# Month spans to fetch: (year1, month1, day1, year2, month2, day2)
MONTH_SPANS = [
    # 2025 Full Training Year
    (2025, 1, 1, 2025, 2, 1),
    (2025, 2, 1, 2025, 3, 1),
    (2025, 3, 1, 2025, 4, 1),
    (2025, 4, 1, 2025, 5, 1),
    (2025, 5, 1, 2025, 6, 1),
    (2025, 6, 1, 2025, 7, 1),
    (2025, 7, 1, 2025, 8, 1),
    (2025, 8, 1, 2025, 9, 1),
    (2025, 9, 1, 2025, 10, 1),
    (2025, 10, 1, 2025, 11, 1),
    (2025, 11, 1, 2025, 12, 1),
    (2025, 12, 1, 2026, 1, 1),
    # 2026 Official Test Evaluation Periods
    (2026, 1, 1, 2026, 2, 1),
    (2026, 7, 1, 2026, 8, 1),
]


def fetch_month_batch(y1, m1, d1, y2, m2, d2, max_retries=3):
    station_params = "&".join([f"station={s}" for s in AIRPORTS])
    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        f"{station_params}&data=all&year1={y1}&month1={m1}&day1={d1}"
        f"&year2={y2}&month2={m2}&day2={d2}&tz=Etc%2FUTC&format=onlycomma"
    )
    headers = {"User-Agent": "Mozilla/5.0 (Educational research; PRC-Data-Challenge-2026)"}
    
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as resp:
                data_str = resp.read().decode("utf-8")
            
            lines = [l for l in data_str.strip().split("\n") if l and not l.startswith("#")]
            if len(lines) <= 1:
                print(f"  [WARN] Warning: empty result for {y1}-{m1:02d}")
                return pd.DataFrame()
            
            df = pd.read_csv(io.StringIO("\n".join(lines)), na_values=["M", "T", ""])
            print(f"  [OK] Fetched {len(df):,} records for {y1}-{m1:02d} ({len(df['station'].unique())} stations)")
            return df
        except Exception as e:
            print(f"  Attempt {attempt}/{max_retries} failed for {y1}-{m1:02d}: {e}")
            if attempt < max_retries:
                time.sleep(5 * attempt)
            else:
                raise


def main():
    print(f"[*] Starting METAR ingestion for {len(AIRPORTS)} airports...")
    all_dfs = []
    
    for y1, m1, d1, y2, m2, d2 in MONTH_SPANS:
        print(f"Fetching {y1}-{m1:02d} to {y2}-{m2:02d}...")
        df_month = fetch_month_batch(y1, m1, d1, y2, m2, d2)
        if not df_month.empty:
            all_dfs.append(df_month)
        # Courteous pause to avoid rate limiting
        time.sleep(2.0)
    
    if not all_dfs:
        raise RuntimeError("No data downloaded!")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n[SUMMARY] Total records downloaded: {len(combined):,}")
    print(f"Stations present: {sorted(combined['station'].unique())}")
    
    out_file = EXTERNAL_DIR / "metar_raw.parquet"
    combined.to_parquet(out_file, index=False)
    print(f"[SAVED] Saved raw METAR archive to {out_file} ({out_file.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
