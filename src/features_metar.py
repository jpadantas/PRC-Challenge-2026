"""
METAR Feature Extraction & Processing for PRC Data Challenge 2026.

Parses raw Iowa State Mesonet METAR records into standardized hourly airport weather features:
- Visibility in meters (worst within the hour)
- Cloud ceiling in feet
- Low Visibility Procedures (LVP) Severe and Moderate indicators
- Active weather codes (Freezing fog, Snow/Ice, Fog, Thunderstorms)
- Peak reported wind gusts
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
RAW_FILE = EXTERNAL_DIR / "metar_raw.parquet"
PROCESSED_FILE = EXTERNAL_DIR / "metar_hourly.parquet"


def process_metar():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw METAR file not found: {RAW_FILE}. Run src/download_metar.py first.")

    print(f"[*] Loading raw METAR from {RAW_FILE}...")
    df = pl.read_parquet(RAW_FILE)
    print(f"  Loaded {len(df):,} raw records.")

    # Parse timestamps
    df = df.with_columns([
        pl.col("valid").str.strptime(pl.Datetime("us", "UTC"), "%Y-%m-%d %H:%M").alias("valid_dt"),
        pl.col("station").str.to_uppercase().alias("adep"),
    ])
    df = df.with_columns(pl.col("valid_dt").dt.truncate("1h").alias("weather_time_utc"))

    # Parse visibility (statute miles -> meters)
    vsby_clean = pl.col("vsby").cast(pl.Float64, strict=False)
    vsby_m = vsby_clean * 1609.34

    # Parse ceiling (skyl1 in feet)
    skyl1_clean = pl.col("skyl1").cast(pl.Float64, strict=False)

    # Weather flags
    wxcodes_upper = pl.col("wxcodes").fill_null("").str.to_uppercase()
    metar_upper = pl.col("metar").fill_null("").str.to_uppercase()

    is_fog = (
        wxcodes_upper.str.contains("FG|BR") |
        metar_upper.str.contains(r"\bFG\b|\bBR\b|\bFZFG\b")
    ).cast(pl.Int8)

    is_freezing_fog = (
        wxcodes_upper.str.contains("FZFG") |
        metar_upper.str.contains("FZFG")
    ).cast(pl.Int8)

    is_snow = (
        wxcodes_upper.str.contains("SN|SG|IC|PL|FZRA") |
        metar_upper.str.contains(r"\bSN\b|\+SN\b|\-SN\b|\bBLSN\b|\bFZRA\b")
    ).cast(pl.Int8)

    is_ts = (
        wxcodes_upper.str.contains("TS") |
        metar_upper.str.contains("TS")
    ).cast(pl.Int8)

    # LVP definitions:
    # Severe LVP (ICAO Cat II/III): Visibility < 600m OR Ceiling <= 200 ft
    # Moderate LVP (ICAO Cat I prep): Visibility < 1500m OR Ceiling <= 500 ft
    is_lvp_severe = (
        (vsby_m < 600.0) | (skyl1_clean <= 200.0)
    ).fill_null(False).cast(pl.Int8)

    is_lvp_moderate = (
        (vsby_m < 1500.0) | (skyl1_clean <= 500.0)
    ).fill_null(False).cast(pl.Int8)

    gust_clean = pl.col("gust").cast(pl.Float64, strict=False).fill_null(0.0)

    # Parse QNH (inHg -> hPa)
    alti_clean = pl.col("alti").cast(pl.Float64, strict=False)
    qnh_hpa_val = alti_clean * 33.8639
    qnh_hpa = pl.when((qnh_hpa_val >= 930.0) & (qnh_hpa_val <= 1070.0)).then(qnh_hpa_val).otherwise(1013.25)

    # Parse temperatures and dewpoint depression
    tmpf_clean = pl.col("tmpf").cast(pl.Float64, strict=False)
    dwpf_clean = pl.col("dwpf").cast(pl.Float64, strict=False)
    tmpc = (tmpf_clean - 32.0) * (5.0 / 9.0)
    dwpc = (dwpf_clean - 32.0) * (5.0 / 9.0)
    dp_dep_c = (tmpc - dwpc).clip(0.0, 50.0).fill_null(5.0)

    df = df.with_columns([
        vsby_m.alias("vsby_m"),
        skyl1_clean.alias("ceiling_ft"),
        is_lvp_severe.alias("is_lvp_severe"),
        is_lvp_moderate.alias("is_lvp_moderate"),
        is_fog.alias("is_fog"),
        is_freezing_fog.alias("is_freezing_fog"),
        is_snow.alias("is_snow"),
        is_ts.alias("is_ts"),
        gust_clean.alias("gust_kts"),
        qnh_hpa.alias("qnh_hpa"),
        dp_dep_c.alias("dp_dep_c"),
    ])

    # Group by (adep, weather_time_utc) to get worst-case hourly condition
    print("[*] Aggregating to hourly bins per airport...")
    hourly = df.group_by(["adep", "weather_time_utc"]).agg([
        pl.col("vsby_m").min().alias("metar_vsby_m"),
        pl.col("ceiling_ft").min().alias("metar_ceiling_ft"),
        pl.col("is_lvp_severe").max().alias("metar_is_lvp_severe"),
        pl.col("is_lvp_moderate").max().alias("metar_is_lvp_moderate"),
        pl.col("is_fog").max().alias("metar_is_fog"),
        pl.col("is_freezing_fog").max().alias("metar_is_freezing_fog"),
        pl.col("is_snow").max().alias("metar_is_snow"),
        pl.col("is_ts").max().alias("metar_is_ts"),
        pl.col("gust_kts").max().alias("metar_gust_kts"),
        pl.col("qnh_hpa").mean().alias("metar_qnh_hpa"),
        pl.col("dp_dep_c").min().alias("metar_dewpoint_dep_c"),
    ]).sort(["adep", "weather_time_utc"])

    # Fill nulls with standard clear-day defaults
    hourly = hourly.with_columns([
        pl.col("metar_vsby_m").fill_null(10000.0),
        pl.col("metar_ceiling_ft").fill_null(5000.0),
        pl.col("metar_is_lvp_severe").fill_null(0),
        pl.col("metar_is_lvp_moderate").fill_null(0),
        pl.col("metar_is_fog").fill_null(0),
        pl.col("metar_is_freezing_fog").fill_null(0),
        pl.col("metar_is_snow").fill_null(0),
        pl.col("metar_is_ts").fill_null(0),
        pl.col("metar_gust_kts").fill_null(0.0),
        pl.col("metar_qnh_hpa").fill_null(1013.25),
        pl.col("metar_dewpoint_dep_c").fill_null(5.0),
    ])

    print(f"  Processed {len(hourly):,} hourly records across {len(hourly['adep'].unique())} airports.")
    print("  Summary of active weather flags:")
    print(f"    LVP Severe hours:    {hourly['metar_is_lvp_severe'].sum():,}")
    print(f"    LVP Moderate hours:  {hourly['metar_is_lvp_moderate'].sum():,}")
    print(f"    Fog hours:           {hourly['metar_is_fog'].sum():,}")
    print(f"    Freezing Fog hours:  {hourly['metar_is_freezing_fog'].sum():,}")
    print(f"    Snow hours:          {hourly['metar_is_snow'].sum():,}")
    print(f"    Thunderstorm hours:  {hourly['metar_is_ts'].sum():,}")

    hourly.write_parquet(PROCESSED_FILE)
    print(f"[SAVED] Saved hourly METAR features to {PROCESSED_FILE} ({PROCESSED_FILE.stat().st_size / 1e3:.1f} KB)")
    return hourly


if __name__ == "__main__":
    process_metar()
