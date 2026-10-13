"""
Merge processed METAR hourly features into weather_hourly.parquet.
Replaces the 100% empty visibility/LVP columns with real station observations
and adds ceiling, freezing fog, snow, and thunderstorm flags.
"""

import shutil
import sys
from pathlib import Path
import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
WEATHER_FILE = EXTERNAL_DIR / "weather_hourly.parquet"
BACKUP_FILE = EXTERNAL_DIR / "weather_hourly_backup.parquet"
METAR_FILE = EXTERNAL_DIR / "metar_hourly.parquet"


def merge_metar_into_weather():
    if not METAR_FILE.exists():
        raise FileNotFoundError(f"METAR hourly file not found: {METAR_FILE}. Run src/features_metar.py first.")
    if not WEATHER_FILE.exists():
        raise FileNotFoundError(f"Weather hourly file not found: {WEATHER_FILE}.")

    # Backup original weather file if not already backed up
    if not BACKUP_FILE.exists():
        print(f"[*] Creating backup of original weather file: {BACKUP_FILE}...")
        shutil.copyfile(WEATHER_FILE, BACKUP_FILE)
    else:
        print(f"[*] Backup already exists at {BACKUP_FILE}.")

    print(f"[*] Reading weather ({WEATHER_FILE}) and METAR ({METAR_FILE})...")
    wx = pl.read_parquet(WEATHER_FILE)
    metar = pl.read_parquet(METAR_FILE)

    # Cast weather_time_utc to match
    wx = wx.with_columns(pl.col("weather_time_utc").cast(pl.Datetime("us", "UTC")))
    metar = metar.with_columns(pl.col("weather_time_utc").cast(pl.Datetime("us", "UTC")))

    # Drop old empty/unusable columns if present to avoid collisions
    cols_to_replace = [
        "visibility", "wx_visibility_m", "wx_is_fog_lvp",
        "wx_is_lvp_moderate", "wx_is_lvp_severe",
        "wx_ceiling_ft", "wx_is_freezing_fog", "wx_is_snow",
        "wx_is_thunderstorm", "wx_gust_kts", "wx_qnh_hpa", "wx_dewpoint_dep_c"
    ]
    wx_clean = wx.drop([c for c in cols_to_replace if c in wx.columns])

    # Left join METAR onto weather
    merged = wx_clean.join(metar, on=["adep", "weather_time_utc"], how="left")

    # Rename & standardize METAR columns to wx_* prefix
    merged = merged.with_columns([
        pl.col("metar_vsby_m").fill_null(10000.0).alias("wx_visibility_m"),
        pl.col("metar_ceiling_ft").fill_null(5000.0).alias("wx_ceiling_ft"),
        pl.col("metar_is_lvp_severe").fill_null(0).cast(pl.Int8).alias("wx_is_lvp_severe"),
        pl.col("metar_is_lvp_moderate").fill_null(0).cast(pl.Int8).alias("wx_is_lvp_moderate"),
        pl.col("metar_is_fog").fill_null(0).cast(pl.Int8).alias("wx_is_fog_lvp"),
        pl.col("metar_is_freezing_fog").fill_null(0).cast(pl.Int8).alias("wx_is_freezing_fog"),
        pl.col("metar_is_snow").fill_null(0).cast(pl.Int8).alias("wx_is_snow"),
        pl.col("metar_is_ts").fill_null(0).cast(pl.Int8).alias("wx_is_thunderstorm"),
        pl.col("metar_gust_kts").fill_null(0.0).alias("wx_gust_kts"),
        pl.col("metar_qnh_hpa").fill_null(1013.25).alias("wx_qnh_hpa"),
        pl.col("metar_dewpoint_dep_c").fill_null(5.0).alias("wx_dewpoint_dep_c"),
    ]).drop([c for c in metar.columns if c.startswith("metar_")])

    print(f"\n[SUMMARY] Merged Weather Dataset:")
    print(f"  Total rows: {len(merged):,}")
    print(f"  Non-null wx_visibility_m: {merged['wx_visibility_m'].drop_nulls().len():,} ({merged['wx_visibility_m'].mean():.1f}m mean)")
    print(f"  LVP Severe active hours:   {(merged['wx_is_lvp_severe'] > 0).sum():,}")
    print(f"  LVP Moderate active hours: {(merged['wx_is_lvp_moderate'] > 0).sum():,}")
    print(f"  Fog active hours:          {(merged['wx_is_fog_lvp'] > 0).sum():,}")
    print(f"  Snow active hours:         {(merged['wx_is_snow'] > 0).sum():,}")
    print(f"  Freezing fog active hours: {(merged['wx_is_freezing_fog'] > 0).sum():,}")
    print(f"  Thunderstorm active hours: {(merged['wx_is_thunderstorm'] > 0).sum():,}")

    merged.write_parquet(WEATHER_FILE)
    print(f"[SAVED] Successfully updated {WEATHER_FILE} ({WEATHER_FILE.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    merge_metar_into_weather()
