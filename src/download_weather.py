"""
Aviation Meteorological Data Ingestion for PRC Data Challenge 2026.

Fetches open historical meteorological data from Open-Meteo Historical Weather API
for the 10 challenge airports across all 2025 training months and 2026 test months.
Provides critical features for de-icing procedures (temperatures <= 3C + precipitation),
visibility conditions (LVP), and wind dynamics for runway alignments.
"""

import json
import time
import urllib.request
from pathlib import Path
import polars as pl
import pandas as pd

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

# 10 Official PRC Challenge Airports
AIRPORT_COORDS = {
    "EGLL": {"lat": 51.4700, "lon": -0.4543, "name": "London Heathrow"},
    "LFPG": {"lat": 49.0097, "lon": 2.5479,  "name": "Paris Charles de Gaulle"},
    "EDDF": {"lat": 50.0379, "lon": 8.5622,  "name": "Frankfurt"},
    "EHAM": {"lat": 52.3105, "lon": 4.7683,  "name": "Amsterdam Schiphol"},
    "LEMD": {"lat": 40.4839, "lon": -3.5680, "name": "Madrid Barajas"},
    "LEBL": {"lat": 41.2974, "lon": 2.0833,  "name": "Barcelona El Prat"},
    "LSZH": {"lat": 47.4582, "lon": 8.5555,  "name": "Zurich"},
    "EDDM": {"lat": 48.3537, "lon": 11.7750, "name": "Munich"},
    "LTFM": {"lat": 41.2753, "lon": 28.7519, "name": "Istanbul"},
    "LIRF": {"lat": 41.8003, "lon": 12.2389, "name": "Rome Fiumicino"},
}

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "snowfall",
    "weather_code",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "visibility",
    "cloud_cover_low",
]


def fetch_weather_range(icao: str, lat: float, lon: float, start_date: str, end_date: str) -> pl.DataFrame:
    """Fetch hourly weather from Open-Meteo for a given airport and date range."""
    vars_str = ",".join(HOURLY_VARS)
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"hourly={vars_str}&timezone=UTC"
    )

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (PRC-2026-Research)"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError(f"No hourly data returned for {icao}")

    df_pd = pd.DataFrame(hourly)
    df_pd["time"] = pd.to_datetime(df_pd["time"], utc=True)
    df_pd["adep"] = icao

    df_pl = pl.from_pandas(df_pd).rename({"time": "weather_time_utc"})
    return df_pl


def download_all_weather():
    """Download weather data for all 10 airports for 2025 and 2026 test periods."""
    print("=" * 70)
    print("🌦️ DOWNLOADING OPEN METEOROLOGICAL DATA FOR ALL 10 AIRPORTS")
    print("=" * 70)

    periods = [
        # 2025 Training Year
        ("2025-01-01", "2025-12-31"),
        # 2026 Test Periods (Jan & Jul)
        ("2026-01-01", "2026-01-31"),
        ("2026-07-01", "2026-07-31"),
    ]

    all_dfs = []
    for icao, meta in AIRPORT_COORDS.items():
        print(f"\nFetching weather for {icao} ({meta['name']})...")
        for start_d, end_d in periods:
            print(f"  • {start_d} to {end_d}...", end=" ", flush=True)
            try:
                df = fetch_weather_range(icao, meta["lat"], meta["lon"], start_d, end_d)
                all_dfs.append(df)
                print(f"OK ({len(df)} hourly observations)")
            except Exception as e:
                print(f"FAILED: {e}")
            time.sleep(0.5)  # respectful delay between API calls

    combined = pl.concat(all_dfs).unique(subset=["adep", "weather_time_utc"])

    # Compute key aviation physical features:
    # 1. De-icing risk: temp <= 3C + precipitation or high humidity
    temp = combined["temperature_2m"]
    snow = combined["snowfall"]
    precip = combined["precipitation"]
    rh = combined["relative_humidity_2m"]
    wcode = combined["weather_code"]

    is_freezing = (temp <= 0.0).cast(pl.Int8)
    is_deicing_risk = (
        (temp <= 3.0) & ((snow > 0) | (precip > 0.1) | (rh >= 90.0) | wcode.is_in([71, 73, 75, 77, 85, 86]))
    ).cast(pl.Int8)

    # 2. Low Visibility Procedures (LVP) risk: fog/mist weather codes (45, 48) or extreme humidity
    is_fog_lvp = (wcode.is_in([45, 48])).cast(pl.Int8)

    # 3. Graduated LVP severity from actual visibility (m), not just weather-code proxy.
    # ICAO/EASA LVP category thresholds: CAT II ~<1200m runway visual range triggers
    # reduced-throughput procedures, CAT III <550m triggers the most severe restrictions.
    visibility_m = combined["visibility"].fill_null(9999.0) if "visibility" in combined.columns else None
    if visibility_m is not None:
        is_lvp_moderate = (visibility_m < 1500.0).cast(pl.Int8)
        is_lvp_severe = (visibility_m < 600.0).cast(pl.Int8)
        extra = [
            visibility_m.alias("wx_visibility_m"),
            is_lvp_moderate.alias("wx_is_lvp_moderate"),
            is_lvp_severe.alias("wx_is_lvp_severe"),
        ]
        if "cloud_cover_low" in combined.columns:
            extra.append(combined["cloud_cover_low"].fill_null(0.0).alias("wx_cloud_cover_low"))
    else:
        extra = []

    combined = combined.with_columns([
        is_freezing.alias("wx_is_freezing"),
        is_deicing_risk.alias("wx_is_deicing_risk"),
        is_fog_lvp.alias("wx_is_fog_lvp"),
        *extra,
    ])

    out_file = EXTERNAL_DIR / "weather_hourly.parquet"
    combined.write_parquet(out_file)
    print("\n" + "=" * 70)
    print(f"✅ Weather dataset saved: {out_file}")
    print(f"  • Total rows: {len(combined):,}")
    print(f"  • De-icing risk hours identified: {combined['wx_is_deicing_risk'].sum():,}")
    print(f"  • Freezing hours identified: {combined['wx_is_freezing'].sum():,}")
    print(f"  • Fog/LVP hours identified: {combined['wx_is_fog_lvp'].sum():,}")
    print("=" * 70)


if __name__ == "__main__":
    download_all_weather()
