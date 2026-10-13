"""
Feature Engineering for PRC Data Challenge 2026 — Taxi-out Time Prediction.

Key components:
1. High-precision timestamp differences from actual movement time and flight plans (MVT_TIME, AOBT_3, EOBT_1, SCHED_TIME)
2. Airport congestion metrics (departure & arrival density in 15m/30m/1h windows, flow ratios)
3. Physical airport layout features (runway, stand, pier/terminal, EHAM Polderbaan, LFPG 2G, LEMD T4S, EGLL T5)
4. Historical target-encoded taxi statistics (medians/means by airport, runway, stand, aircraft type, operator)
5. Eurocontrol Unimpeded Taxi Time (UTT) baseline matrix (15th percentile unimpeded travel time)
6. RECAT-EU 6-category wake vortex separation minima & preceding departure wake deficit
7. Helicopter / VFR zero-taxi physics
8. External physical datasets:
   - OurAirports: flight distance (great circle), elevation diff, domestic/intercontinental, physical runway length/width/heading
   - ICAO Doc 8643: aircraft MTOW, wingspan, length, operating range, engine class
   - Open-Meteo: runway vector wind decomposition (head/cross/tailwind), deicing risk, freezing, fog/LVP
"""

import polars as pl
import numpy as np
from pathlib import Path
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# RECAT-EU Mapping for aircraft types
RECAT_EU_MAP = {
    # Cat A: Super Heavy
    "A388": "A",
    # Cat B: Upper Heavy
    "B77W": "B", "B772": "B", "B773": "B", "B77L": "B", "B77F": "B",
    "B744": "B", "B748": "B", "A359": "B", "A35K": "B", "A343": "B", "A346": "B", "MD11": "B",
    # Cat C: Lower Heavy
    "B788": "C", "B789": "C", "B78X": "C", "A332": "C", "A333": "C", "A339": "C", "B763": "C", "B764": "C",
    # Cat D: Upper Medium
    "B752": "D", "B753": "D",
    # Cat E: Lower Medium
    "A320": "E", "A321": "E", "A319": "E", "A20N": "E", "A21N": "E", "A19N": "E",
    "B738": "E", "B737": "E", "B739": "E", "B38M": "E", "B39M": "E", "B734": "E",
    "E190": "E", "E195": "E", "E290": "E", "E295": "E", "BCS1": "E", "BCS3": "E", "CRJ9": "E", "CRK": "E",
    # Cat F: Light / Regional
    "AT72": "F", "AT76": "F", "AT45": "F", "DH8D": "F", "E145": "F", "E170": "F", "E75L": "F", "CRJ2": "F",
    "PC12": "F", "BE20": "F", "C56X": "F", "C680": "F", "CL30": "F", "CL60": "F", "GLF5": "F", "GLF6": "F",
    # Helicopters (Helo)
    "EC20": "H", "A109": "H", "AS50": "H", "EC35": "H", "B06": "H", "R44": "H", "S76": "H", "A139": "H",
    "EC45": "H", "B429": "H", "AS32": "H", "AW18": "H",
}

# RECAT-EU Departure Radar Separation Minima Matrix (seconds)
RECAT_SEP_SEC = {
    ("A", "A"): 100.0, ("A", "B"): 120.0, ("A", "C"): 140.0, ("A", "D"): 160.0, ("A", "E"): 180.0, ("A", "F"): 200.0, ("A", "H"): 200.0,
    ("B", "A"): 60.0,  ("B", "B"): 90.0,  ("B", "C"): 100.0, ("B", "D"): 120.0, ("B", "E"): 140.0, ("B", "F"): 160.0, ("B", "H"): 160.0,
    ("C", "A"): 60.0,  ("C", "B"): 60.0,  ("C", "C"): 80.0,  ("C", "D"): 100.0, ("C", "E"): 120.0, ("C", "F"): 140.0, ("C", "H"): 140.0,
    ("D", "A"): 60.0,  ("D", "B"): 60.0,  ("D", "C"): 60.0,  ("D", "D"): 80.0,  ("D", "E"): 100.0, ("D", "F"): 120.0, ("D", "H"): 120.0,
    ("E", "A"): 60.0,  ("E", "B"): 60.0,  ("E", "C"): 60.0,  ("E", "D"): 60.0,  ("E", "E"): 80.0,  ("E", "F"): 100.0, ("E", "H"): 100.0,
    ("F", "A"): 60.0,  ("F", "B"): 60.0,  ("F", "C"): 60.0,  ("F", "D"): 60.0,  ("F", "E"): 60.0,  ("F", "F"): 80.0,  ("F", "H"): 80.0,
    ("H", "A"): 60.0,  ("H", "B"): 60.0,  ("H", "C"): 60.0,  ("H", "D"): 60.0,  ("H", "E"): 60.0,  ("H", "F"): 60.0,  ("H", "H"): 60.0,
}

HELICOPTER_TYPES = {
    "EC20", "A109", "AS50", "EC35", "B06", "R44", "S76", "A139", "EC45", "B429", "AS32", "AW18"
}

# Airline Alliances & Operational Segments
STAR_ALLIANCE = {
    "DLH", "THY", "SWR", "UAL", "DLA", "LHX", "AUA", "TAP", "EDW", "AEE", "LOT", "BEL", "ACA", "SXS", "TGZ", "SAS"
}
SKYTEAM_ALLIANCE = {
    "AFR", "KLM", "ITY", "DAL", "AEA", "VIR", "SVA", "TAR", "MEA", "AMX", "CES", "AZA"
}
ONEWORLD_ALLIANCE = {
    "BAW", "IBE", "VLG", "ANE", "IBS", "SHT", "AAL", "QTR", "EIN", "FIN", "JAL", "RAM", "RJA", "CPA", "MAS"
}
LCC_CARRIERS = {
    "RYR", "EJU", "EZY", "WMT", "WZZ", "TRA", "EWG", "PGT", "VOE", "EZS", "EXS", "TUI", "NOZ", "TOM", "JAF", "FPO"
}
CARGO_CARRIERS = {
    "GEC", "TAY", "BCS", "UPS", "FDX", "CLX", "BOX", "ABR", "SWN", "MNB", "CKS", "GTI"
}

HOME_HUBS = {
    "EDDF": {"DLH", "DLA", "LHX", "CFG", "GEC", "OCN"},
    "EDDM": {"DLH", "DLA", "LHX", "CFG", "OCN"},
    "EGLL": {"BAW", "SHT", "VIR"},
    "LFPG": {"AFR", "HOP"},
    "EHAM": {"KLM", "TRA", "MPH"},
    "LEMD": {"IBE", "IBS", "ANE", "AEA"},
    "LEBL": {"VLG", "IBE", "ANE"},
    "LTFM": {"THY"},
    "LSZH": {"SWR", "EDW"},
    "LIRF": {"ITY", "AZA"},
}

AIRPORT_COORDS = {
    "EDDF": (50.0379, 8.5622),
    "EDDM": (48.3538, 11.7861),
    "EGLL": (51.4700, -0.4543),
    "EHAM": (52.3086, 4.7639),
    "LEBL": (41.2971, 2.0785),
    "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479),
    "LIRF": (41.8003, 12.2389),
    "LSZH": (47.4647, 8.5492),
    "LTFM": (41.2753, 28.7519),
}

NATIONAL_HOLIDAYS = {
    "EDDF": {(1, 1), (1, 6)},
    "EDDM": {(1, 1), (1, 6)},
    "EGLL": {(1, 1)},
    "LFPG": {(1, 1), (7, 14)},
    "EHAM": {(1, 1)},
    "LEMD": {(1, 1), (1, 6), (7, 25)},
    "LEBL": {(1, 1), (1, 6)},
    "LTFM": {(1, 1), (7, 15)},
    "LSZH": {(1, 1), (1, 2)},
    "LIRF": {(1, 1), (1, 6)},
}

def compute_solar_elevation(dt_series: pl.Series, adep_series: pl.Series) -> np.ndarray:
    """Compute solar elevation angle in degrees for each flight timestamp and airport."""
    doy = dt_series.dt.ordinal_day().to_numpy().astype(np.float64)
    hour = dt_series.dt.hour().to_numpy().astype(np.float64) + dt_series.dt.minute().to_numpy().astype(np.float64) / 60.0
    adeps = adep_series.to_numpy()
    lats = np.array([AIRPORT_COORDS.get(a, (50.0, 10.0))[0] for a in adeps], dtype=np.float64)
    lons = np.array([AIRPORT_COORDS.get(a, (50.0, 10.0))[1] for a in adeps], dtype=np.float64)
    
    delta = 23.45 * np.sin(np.radians((360.0 / 365.0) * (doy - 81.0)))
    B = np.radians((360.0 / 365.0) * (doy - 81.0))
    eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    time_offset = eot + 4.0 * lons
    lst = hour + time_offset / 60.0
    H = np.radians(15.0 * (lst - 12.0))
    lat_rad = np.radians(lats)
    delta_rad = np.radians(delta)
    sin_elev = np.sin(lat_rad) * np.sin(delta_rad) + np.cos(lat_rad) * np.cos(delta_rad) * np.cos(H)
    return np.degrees(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))



def load_training_data(raw_dir: Path, months: list[str] | None = None) -> pl.DataFrame:
    """Load training parquet files."""
    files = sorted(raw_dir.glob("training_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No training files found in {raw_dir}")

    if months is not None:
        files = [f for f in files if any(m in f.name for m in months)]

    print(f"Loading {len(files)} training file(s)...")
    dfs = []
    for f in files:
        df = pl.read_parquet(f)
        dfs.append(df)
    combined = pl.concat(dfs)
    print(f"Loaded {len(combined):,} total rows ({len(combined.filter(pl.col('PHASE_mvt') == 'DEP')):,} DEP)")
    return combined


def load_ranking_data(raw_dir: Path) -> pl.DataFrame:
    """Load ranking (test) dataset."""
    path = raw_dir / "ranking.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Ranking file not found: {path}")
    df = pl.read_parquet(path)
    print(f"Loaded ranking data: {len(df):,} rows ({len(df.filter(pl.col('PHASE_mvt') == 'DEP')):,} DEP)")
    return df


def load_submission_template(raw_dir: Path) -> pl.DataFrame:
    """Load submission template (submitting.parquet)."""
    path = raw_dir / "submitting.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Submission template not found: {path}")
    return pl.read_parquet(path)


def compute_runway_sequence_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes chronological runway sequencing features using MVT_TIME_UTC_mvt.
    - Time delta to previous departure on same runway
    - Previous departure wake turbulence category (H, M, L, J)
    - Preceding heavy departure flag (wake turbulence separation minima)
    - RECAT-EU 6-category classification & wake deficit
    """
    print("Computing runway queue sequencing & wake separation features...")
    
    actypes = df["AIRCRAFT_TYPE_mvt"].cast(pl.Utf8).fill_null("UNK").to_list()
    recat_classes = [RECAT_EU_MAP.get(ac, "E") for ac in actypes]
    
    df = df.with_columns([
        pl.col("RUNWAY_mvt").fill_null("UNK").alias("runway_clean"),
        pl.col("WK_TBL_CAT_flt").fill_null("M").alias("wtc_clean"),
        pl.Series("recat_clean", recat_classes),
    ])
    
    # Airport where the movement occurs (ADEP for DEP, ADES for ARR)
    apt_expr = pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt"))
    is_heavy = pl.col("wtc_clean").is_in(["H", "J"]).cast(pl.Int32)

    rwy_df = df.with_columns([
        apt_expr.alias("apt_clean"),
        is_heavy.alias("is_heavy_mvt")
    ]).select([
        "MVT_ID_mvt", "apt_clean", "ADEP_mvt", "runway_clean", "PHASE_mvt", 
        "MVT_TIME_UTC_mvt", "wtc_clean", "recat_clean", "is_heavy_mvt"
    ]).sort(["apt_clean", "runway_clean", "MVT_TIME_UTC_mvt"])
    
    # Cross-traffic: check whether the immediately preceding movement on the runway was an ARRIVAL
    rwy_df = rwy_df.with_columns([
        (pl.col("MVT_TIME_UTC_mvt").diff().dt.total_seconds().over(["apt_clean", "runway_clean"])).alias("dt_prev_any_mvt"),
        pl.col("PHASE_mvt").shift(1).over(["apt_clean", "runway_clean"]).alias("prev_mvt_phase"),
    ])
    rwy_df = rwy_df.with_columns([
        (pl.col("prev_mvt_phase") == "ARR").cast(pl.Int32).fill_null(0).alias("is_prev_mvt_arr"),
        pl.col("dt_prev_any_mvt").clip(0, 1800).fill_null(1800),
    ])

    dep_mask = pl.col("PHASE_mvt") == "DEP"
    dep_rwy = rwy_df.filter(dep_mask).with_columns([
        (pl.col("MVT_TIME_UTC_mvt").diff().dt.total_seconds().over(["ADEP_mvt", "runway_clean"])).alias("dt_prev_dep_rwy"),
        pl.col("wtc_clean").shift(1).over(["ADEP_mvt", "runway_clean"]).alias("prev_dep_wtc"),
        pl.col("recat_clean").shift(1).over(["ADEP_mvt", "runway_clean"]).alias("prev_dep_recat"),
        pl.col("is_heavy_mvt").shift(1).over(["ADEP_mvt", "runway_clean"]).alias("prev_dep_is_heavy")
    ])
    
    dep_rwy = dep_rwy.with_columns([
        pl.col("dt_prev_dep_rwy").clip(0, 1800).fill_null(1800),
        pl.col("prev_dep_is_heavy").fill_null(0),
        pl.col("prev_dep_wtc").fill_null("M"),
        pl.col("prev_dep_recat").fill_null("E"),
    ])

    # ICAO Doc 4444 Wake Turbulence Radar Separation Minima (seconds)
    icao_sep = (
        pl.when((pl.col("prev_dep_wtc") == "J") & (pl.col("wtc_clean") == "L")).then(240.0)
        .when((pl.col("prev_dep_wtc") == "J") & (pl.col("wtc_clean") == "M")).then(180.0)
        .when((pl.col("prev_dep_wtc") == "J") & (pl.col("wtc_clean") == "H")).then(120.0)
        .when((pl.col("prev_dep_wtc") == "H") & (pl.col("wtc_clean") == "L")).then(180.0)
        .when((pl.col("prev_dep_wtc") == "H") & (pl.col("wtc_clean") == "M")).then(120.0)
        .when((pl.col("prev_dep_wtc") == "H") & (pl.col("wtc_clean") == "H")).then(90.0)
        .when((pl.col("prev_dep_wtc") == "M") & (pl.col("wtc_clean") == "L")).then(120.0)
        .otherwise(60.0)
    ).alias("icao_wake_sep_sec")

    dep_rwy = dep_rwy.with_columns(icao_sep)

    wake_deficit = (
        (pl.col("icao_wake_sep_sec") - pl.col("dt_prev_dep_rwy")).clip(0, 300)
    ).alias("wake_deficit_sec")

    dep_rwy = dep_rwy.with_columns(wake_deficit)
    
    # RECAT-EU separation calculation
    p_recat = dep_rwy["prev_dep_recat"].to_list()
    c_recat = dep_rwy["recat_clean"].to_list()
    dt_prev = dep_rwy["dt_prev_dep_rwy"].to_numpy()
    
    recat_sep = np.array([RECAT_SEP_SEC.get((p, c), 60.0) for p, c in zip(p_recat, c_recat)], dtype=np.float32)
    recat_deficit = np.clip(recat_sep - dt_prev, 0.0, 300.0)
    is_prev_heavy_recat = np.isin(p_recat, ["A", "B"]).astype(np.int32)
    
    dep_rwy = dep_rwy.with_columns([
        pl.Series("recat_sep_sec", recat_sep),
        pl.Series("recat_wake_deficit_sec", recat_deficit),
        pl.Series("prev_dep_is_recat_heavy", is_prev_heavy_recat),
    ])
    
    # Dynamic Runway Queue Rolling Features
    # Computes preceding aircraft queue delay on the exact runway
    mvt_r = dep_rwy["MVT_TIME_UTC_mvt"]
    aobt_r = dep_rwy["AOBT_3_flt"] if "AOBT_3_flt" in dep_rwy.columns else None
    sched_r = dep_rwy["SCHED_TIME_UTC_mvt"] if "SCHED_TIME_UTC_mvt" in dep_rwy.columns else None

    if aobt_r is not None and sched_r is not None:
        da_raw = (mvt_r - aobt_r).dt.total_seconds().to_numpy()
        ds_raw = (mvt_r - sched_r).dt.total_seconds().to_numpy()
        da_clean = np.where((~np.isnan(da_raw)) & (da_raw >= 60) & (da_raw <= 7200), da_raw, np.clip(ds_raw, 600.0, 1800.0))
        da_clean = np.nan_to_num(da_clean, nan=1000.0)
    else:
        da_clean = np.full(len(dep_rwy), 1000.0, dtype=np.float64)

    dep_rwy = dep_rwy.with_columns([
        pl.Series("da_proxy_rwy", da_clean)
    ])

    dep_rwy = dep_rwy.with_columns([
        pl.col("da_proxy_rwy").shift(1).over(["ADEP_mvt", "runway_clean"]).alias("prev_da_1"),
        pl.col("da_proxy_rwy").shift(2).over(["ADEP_mvt", "runway_clean"]).alias("prev_da_2"),
        pl.col("da_proxy_rwy").shift(3).over(["ADEP_mvt", "runway_clean"]).alias("prev_da_3"),
        pl.col("da_proxy_rwy").shift(4).over(["ADEP_mvt", "runway_clean"]).alias("prev_da_4"),
        pl.col("da_proxy_rwy").shift(5).over(["ADEP_mvt", "runway_clean"]).alias("prev_da_5"),
        pl.col("dt_prev_dep_rwy").shift(1).over(["ADEP_mvt", "runway_clean"]).alias("prev_dt_1"),
        pl.col("dt_prev_dep_rwy").shift(2).over(["ADEP_mvt", "runway_clean"]).alias("prev_dt_2"),
        pl.col("dt_prev_dep_rwy").shift(3).over(["ADEP_mvt", "runway_clean"]).alias("prev_dt_3"),
    ])

    dep_rwy = dep_rwy.with_columns([
        ((pl.col("prev_da_1") + pl.col("prev_da_2") + pl.col("prev_da_3")) / 3.0).fill_null(1000.0).alias("rwy_rolling_da_3"),
        ((pl.col("prev_da_1") + pl.col("prev_da_2") + pl.col("prev_da_3") + pl.col("prev_da_4") + pl.col("prev_da_5")) / 5.0).fill_null(1000.0).alias("rwy_rolling_da_5"),
        ((pl.col("dt_prev_dep_rwy") + pl.col("prev_dt_1") + pl.col("prev_dt_2") + pl.col("prev_dt_3")) / 4.0).fill_null(180.0).alias("rwy_rolling_dt_5"),
    ])

    seq_feats = dep_rwy.select([
        "MVT_ID_mvt", "dt_prev_dep_rwy", "prev_dep_wtc", "prev_dep_recat", "prev_dep_is_heavy",
        "icao_wake_sep_sec", "wake_deficit_sec", "recat_sep_sec", "recat_wake_deficit_sec",
        "prev_dep_is_recat_heavy", "rwy_rolling_da_3", "rwy_rolling_da_5", "rwy_rolling_dt_5",
        "dt_prev_any_mvt", "is_prev_mvt_arr"
    ])

    return df.join(seq_feats, on="MVT_ID_mvt", how="left")


def compute_congestion_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute airport-level and runway-level congestion metrics.
    Uses all movements (DEP and ARR) around MVT_TIME_UTC_mvt.
    Correctly maps arrivals to their destination hub (ADES_mvt).
    """
    print("Computing airport & runway congestion metrics (with true arrival flows)...")
    
    # Define hub airport: ADEP for DEP, ADES for ARR
    if "hub_airport" not in df.columns:
        df = df.with_columns([
            pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("hub_airport")
        ])

    # Create time bins on MVT_TIME_UTC_mvt
    df = df.with_columns([
        pl.col("MVT_TIME_UTC_mvt").dt.truncate("15m").alias("bin_15m"),
        pl.col("MVT_TIME_UTC_mvt").dt.truncate("30m").alias("bin_30m"),
        pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h").alias("bin_1h"),
    ])

    # Airport-level 15m counts
    apt_15 = (
        df.group_by(["hub_airport", "bin_15m", "PHASE_mvt"])
        .len()
        .pivot(index=["hub_airport", "bin_15m"], on="PHASE_mvt", values="len")
        .fill_null(0)
    )
    if "DEP" in apt_15.columns:
        apt_15 = apt_15.rename({"DEP": "cong_apt_dep_15m"})
    else:
        apt_15 = apt_15.with_columns(pl.lit(0).alias("cong_apt_dep_15m"))
    if "ARR" in apt_15.columns:
        apt_15 = apt_15.rename({"ARR": "cong_apt_arr_15m"})
    else:
        apt_15 = apt_15.with_columns(pl.lit(0).alias("cong_apt_arr_15m"))

    # Airport-level 30m counts
    apt_30 = (
        df.group_by(["hub_airport", "bin_30m", "PHASE_mvt"])
        .len()
        .pivot(index=["hub_airport", "bin_30m"], on="PHASE_mvt", values="len")
        .fill_null(0)
    )
    if "DEP" in apt_30.columns:
        apt_30 = apt_30.rename({"DEP": "cong_apt_dep_30m"})
    else:
        apt_30 = apt_30.with_columns(pl.lit(0).alias("cong_apt_dep_30m"))
    if "ARR" in apt_30.columns:
        apt_30 = apt_30.rename({"ARR": "cong_apt_arr_30m"})
    else:
        apt_30 = apt_30.with_columns(pl.lit(0).alias("cong_apt_arr_30m"))

    # Runway-level 15m counts (hub_airport + RUNWAY_mvt)
    rwy_15 = (
        df.filter(pl.col("RUNWAY_mvt").is_not_null())
        .group_by(["hub_airport", "RUNWAY_mvt", "bin_15m", "PHASE_mvt"])
        .len()
        .pivot(index=["hub_airport", "RUNWAY_mvt", "bin_15m"], on="PHASE_mvt", values="len")
        .fill_null(0)
    )
    if "DEP" in rwy_15.columns:
        rwy_15 = rwy_15.rename({"DEP": "cong_rwy_dep_15m"})
    else:
        rwy_15 = rwy_15.with_columns(pl.lit(0).alias("cong_rwy_dep_15m"))
    if "ARR" in rwy_15.columns:
        rwy_15 = rwy_15.rename({"ARR": "cong_rwy_arr_15m"})
    else:
        rwy_15 = rwy_15.with_columns(pl.lit(0).alias("cong_rwy_arr_15m"))

    # Join metrics back onto dataframe
    df = df.join(apt_15, on=["hub_airport", "bin_15m"], how="left")
    df = df.join(apt_30, on=["hub_airport", "bin_30m"], how="left")
    df = df.join(rwy_15, on=["hub_airport", "RUNWAY_mvt", "bin_15m"], how="left")

    df = df.with_columns([
        pl.col("cong_apt_dep_15m").fill_null(0),
        pl.col("cong_apt_arr_15m").fill_null(0),
        pl.col("cong_apt_dep_30m").fill_null(0),
        pl.col("cong_apt_arr_30m").fill_null(0),
        pl.col("cong_rwy_dep_15m").fill_null(0),
        pl.col("cong_rwy_arr_15m").fill_null(0),
        (pl.col("cong_apt_dep_15m").fill_null(0) + pl.col("cong_apt_arr_15m").fill_null(0)).alias("cong_apt_tot_15m"),
        (pl.col("cong_apt_dep_30m").fill_null(0) + pl.col("cong_apt_arr_30m").fill_null(0)).alias("cong_apt_tot_30m"),
        (pl.col("cong_rwy_dep_15m").fill_null(0) + pl.col("cong_rwy_arr_15m").fill_null(0)).alias("cong_rwy_tot_15m"),
        (pl.col("cong_apt_dep_15m").fill_null(0) / (pl.col("cong_apt_dep_15m").fill_null(0) + pl.col("cong_apt_arr_15m").fill_null(0) + 1.0)).alias("cong_dep_ratio_15m"),
        (pl.col("cong_apt_dep_15m").fill_null(0) / (pl.col("cong_apt_arr_15m").fill_null(0) + 1.0)).alias("dep_arr_flow_ratio"),
        (pl.col("cong_apt_arr_15m").fill_null(0) / (pl.col("cong_apt_dep_15m").fill_null(0) + 1.0)).alias("arr_dep_flow_ratio"),
        (pl.col("cong_rwy_dep_15m").fill_null(0) / (pl.col("cong_apt_dep_15m").fill_null(0) + 1.0)).alias("rwy_dep_pressure_ratio"),
        ((pl.col("cong_rwy_dep_15m").fill_null(0) + pl.col("cong_rwy_arr_15m").fill_null(0)) / (pl.col("cong_apt_dep_15m").fill_null(0) + pl.col("cong_apt_arr_15m").fill_null(0) + 1.0)).alias("rwy_tot_pressure_ratio"),
    ])

    df = df.drop(["bin_15m", "bin_30m", "bin_1h"], strict=False)
    return df


def compute_surface_accumulation_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes instantaneous surface traffic accumulation (N_active) via dual binary search (searchsorted).
    At departure pushback and takeoff:
    - Exactly how many arrival aircraft are currently taxiing between touchdown and gate?
    - Exactly how many departures are currently taxiing between pushback and takeoff?
    """
    print("Computing instantaneous surface traffic accumulation (N_surface) via binary search...")
    if "hub_airport" not in df.columns:
        df = df.with_columns([
            pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("hub_airport")
        ])

    arr_df = df.filter(pl.col("PHASE_mvt") == "ARR")
    dep_df = df.filter(pl.col("PHASE_mvt") == "DEP")
    airports = [a for a in dep_df["hub_airport"].unique().to_list() if a is not None]

    n_dep = len(dep_df)
    active_arr_at_mvt = np.zeros(n_dep, dtype=np.float32)
    active_arr_at_aobt = np.zeros(n_dep, dtype=np.float32)
    active_dep_at_mvt = np.zeros(n_dep, dtype=np.float32)

    dep_mvt_np = dep_df["MVT_TIME_UTC_mvt"].dt.epoch(time_unit="s").to_numpy()
    dep_aobt_np = dep_df["AOBT_3_flt"].dt.epoch(time_unit="s").to_numpy() if "AOBT_3_flt" in dep_df.columns else np.full(n_dep, np.nan)
    dep_apt_np = dep_df["hub_airport"].to_numpy()

    for apt in airports:
        mask_apt_dep = (dep_apt_np == apt)
        if not np.any(mask_apt_dep):
            continue

        sub_arr = arr_df.filter(pl.col("hub_airport") == apt)
        sub_dep_mvt = dep_mvt_np[mask_apt_dep]
        sub_dep_aobt = dep_aobt_np[mask_apt_dep]

        if len(sub_arr) > 0:
            arr_touch = np.sort(sub_arr["MVT_TIME_UTC_mvt"].dt.epoch(time_unit="s").drop_nulls().to_numpy())
            arr_block = np.sort(sub_arr["BLOCK_TIME_UTC_mvt"].dt.epoch(time_unit="s").drop_nulls().to_numpy()) if "BLOCK_TIME_UTC_mvt" in sub_arr.columns else arr_touch + 600

            # Active arrivals at departure takeoff time
            n_touch = np.searchsorted(arr_touch, sub_dep_mvt, side="right")
            n_block = np.searchsorted(arr_block, sub_dep_mvt, side="left")
            active_arr_at_mvt[mask_apt_dep] = np.clip(n_touch - n_block, 0, None)

            # Active arrivals at departure pushback time
            valid_aobt = ~np.isnan(sub_dep_aobt)
            n_touch_a = np.searchsorted(arr_touch, np.where(valid_aobt, sub_dep_aobt, sub_dep_mvt - 900), side="right")
            n_block_a = np.searchsorted(arr_block, np.where(valid_aobt, sub_dep_aobt, sub_dep_mvt - 900), side="left")
            active_arr_at_aobt[mask_apt_dep] = np.clip(n_touch_a - n_block_a, 0, None)

        # Active departures at takeoff time
        apt_dep_valid = dep_df.filter((pl.col("hub_airport") == apt) & pl.col("AOBT_3_flt").is_not_null())
        if len(apt_dep_valid) > 0:
            dep_push = np.sort(apt_dep_valid["AOBT_3_flt"].dt.epoch(time_unit="s").to_numpy())
            dep_take = np.sort(sub_dep_mvt)
            n_push = np.searchsorted(dep_push, sub_dep_mvt, side="right")
            n_take = np.searchsorted(dep_take, sub_dep_mvt, side="left")
            active_dep_at_mvt[mask_apt_dep] = np.clip(n_push - n_take, 0, None)

    total_surface = active_arr_at_mvt + active_dep_at_mvt
    surf_feats = pl.DataFrame({
        "MVT_ID_mvt": dep_df["MVT_ID_mvt"],
        "active_arr_at_takeoff": active_arr_at_mvt,
        "active_arr_at_pushback": active_arr_at_aobt,
        "active_dep_at_takeoff": active_dep_at_mvt,
        "total_surface_active": total_surface,
    })

    return df.join(surf_feats, on="MVT_ID_mvt", how="left")


def compute_apron_contention_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes instantaneous pier/apron pushback contention (alleyway gridlock) via binary search.
    For each departure:
    - Exactly how many other departures on the same pier pushed back within [-15m, +15m]?
    - Exactly how many pushed back in the preceding 15 minutes?
    """
    print("Computing apron pier contention metrics (alleyway gridlock)...")
    dep_df = df.filter(pl.col("PHASE_mvt") == "DEP")
    n_dep = len(dep_df)

    stand_str = dep_df["STAND_mvt"].cast(pl.Utf8).fill_null("UNK")
    stand_pier = stand_str.str.slice(0, 1)

    mvt_epoch = dep_df["MVT_TIME_UTC_mvt"].dt.epoch(time_unit="s").to_numpy()
    aobt_epoch = dep_df["AOBT_3_flt"].dt.epoch(time_unit="s").to_numpy() if "AOBT_3_flt" in dep_df.columns else np.full(n_dep, np.nan)
    valid_aobt = ~np.isnan(aobt_epoch)
    t_push = np.where(valid_aobt, aobt_epoch, mvt_epoch - 900)

    dep_temp = dep_df.with_columns([
        stand_pier.alias("stand_pier_temp"),
        pl.Series("t_push_temp", t_push)
    ])

    pier_contention_15m = np.zeros(n_dep, dtype=np.float32)
    pier_contention_30m = np.zeros(n_dep, dtype=np.float32)

    airports = [a for a in dep_df["ADEP_mvt"].unique().to_list() if a is not None]

    for apt in airports:
        sub_apt = dep_temp.filter(pl.col("ADEP_mvt") == apt)
        piers = [p for p in sub_apt["stand_pier_temp"].unique().to_list() if p is not None and p != "U"]
        for pier in piers:
            sub_pier = sub_apt.filter(pl.col("stand_pier_temp") == pier)
            if len(sub_pier) <= 1:
                continue

            t_sorted = np.sort(sub_pier["t_push_temp"].to_numpy())
            mask = (dep_temp["ADEP_mvt"] == apt) & (dep_temp["stand_pier_temp"] == pier)
            t_query = dep_temp.filter(mask)["t_push_temp"].to_numpy()

            idx_right_15 = np.searchsorted(t_sorted, t_query, side="right")
            idx_left_15 = np.searchsorted(t_sorted, t_query - 900, side="left")
            c_15m = np.clip(idx_right_15 - idx_left_15 - 1, 0, None)

            idx_right_30 = np.searchsorted(t_sorted, t_query + 900, side="right")
            c_30m = np.clip(idx_right_30 - idx_left_15 - 1, 0, None)

            pier_contention_15m[mask.to_numpy()] = c_15m
            pier_contention_30m[mask.to_numpy()] = c_30m

    apron_feats = pl.DataFrame({
        "MVT_ID_mvt": dep_df["MVT_ID_mvt"],
        "pier_contention_15m": pier_contention_15m,
        "pier_contention_30m": pier_contention_30m,
    })

    return df.join(apron_feats, on="MVT_ID_mvt", how="left")


def compute_target_encodings(train_dep_df: pl.DataFrame) -> dict:
    """
    Compute target taxi-out time statistics (mean, median, std) and Unimpeded Taxi Time (UTT).
    Filters out extreme non-physical outliers (> 2 hours or < 2 min).
    """
    print("Computing target encodings & Unimpeded Taxi Time (UTT) from training data...")
    clean = train_dep_df.filter(
        (pl.col("TAXITIME_SEC_mvt") >= 120) & (pl.col("TAXITIME_SEC_mvt") <= 7200)
    )

    clean = clean.with_columns([
        clean["STAND_mvt"].cast(pl.Utf8).fill_null("UNK").str.slice(0, 1).alias("stand_pier"),
        clean["MVT_TIME_UTC_mvt"].dt.hour().alias("mvt_hour"),
        clean["MVT_TIME_UTC_mvt"].dt.month().alias("mvt_month"),
        clean["AIRCRAFT_OPERATOR_flt"].cast(pl.Utf8).fill_null("UNK").alias("operator_clean"),
    ])

    stats = {}

    # 1. Airport baseline
    stats["apt"] = clean.group_by("ADEP_mvt").agg([
        pl.col("TAXITIME_SEC_mvt").median().alias("stat_med_apt"),
        pl.col("TAXITIME_SEC_mvt").mean().alias("stat_mean_apt"),
        pl.col("TAXITIME_SEC_mvt").std().alias("stat_std_apt"),
        pl.col("TAXITIME_SEC_mvt").quantile(0.15).alias("stat_utt_apt"),
    ])

    global_mean = clean["TAXITIME_SEC_mvt"].mean()

    # 2. Airport + Runway (with smoothing m=15)
    rwy_grp = clean.group_by(["ADEP_mvt", "RUNWAY_mvt"]).agg([
        pl.col("TAXITIME_SEC_mvt").median().alias("stat_med_apt_rwy"),
        pl.col("TAXITIME_SEC_mvt").mean().alias("raw_mean_rwy"),
        pl.col("TAXITIME_SEC_mvt").quantile(0.15).alias("stat_utt_rwy"),
        pl.len().alias("count_rwy"),
    ])
    rwy_grp = rwy_grp.with_columns(
        ((pl.col("count_rwy") * pl.col("raw_mean_rwy") + 15.0 * global_mean) / (pl.col("count_rwy") + 15.0)).alias("stat_mean_apt_rwy")
    ).drop(["raw_mean_rwy", "count_rwy"])
    stats["apt_rwy"] = rwy_grp

    # 3. Airport + Stand (with smoothing m=20 to prevent overfitting on rare stands)
    stand_grp = clean.group_by(["ADEP_mvt", "STAND_mvt"]).agg([
        pl.col("TAXITIME_SEC_mvt").median().alias("stat_med_apt_stand"),
        pl.col("TAXITIME_SEC_mvt").mean().alias("raw_mean_stand"),
        pl.len().alias("stat_count_apt_stand"),
    ])
    stand_grp = stand_grp.with_columns(
        ((pl.col("stat_count_apt_stand") * pl.col("raw_mean_stand") + 20.0 * global_mean) / (pl.col("stat_count_apt_stand") + 20.0)).alias("stat_mean_apt_stand")
    ).drop("raw_mean_stand")
    stats["apt_stand"] = stand_grp

    # 4. Airport + Pier + Runway (layout route proxy with smoothing m=15 and UTT)
    pier_grp = clean.group_by(["ADEP_mvt", "stand_pier", "RUNWAY_mvt"]).agg([
        pl.col("TAXITIME_SEC_mvt").median().alias("stat_med_apt_pier_rwy"),
        pl.col("TAXITIME_SEC_mvt").mean().alias("raw_mean_pier"),
        pl.col("TAXITIME_SEC_mvt").quantile(0.15).alias("stat_utt_pier_rwy"),
        pl.len().alias("count_pier"),
    ])
    pier_grp = pier_grp.with_columns(
        ((pl.col("count_pier") * pl.col("raw_mean_pier") + 15.0 * global_mean) / (pl.col("count_pier") + 15.0)).alias("stat_mean_apt_pier_rwy")
    ).drop(["raw_mean_pier", "count_pier"])
    stats["apt_pier_rwy"] = pier_grp

    # 5. Airport + Stand + Runway (Route Distance Level 1 — UTT + Median + P85 for variance capture)
    stand_rwy_grp = clean.group_by(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]).agg([
        pl.col("TAXITIME_SEC_mvt").quantile(0.15).alias("stat_utt_stand_rwy"),
        pl.col("TAXITIME_SEC_mvt").median().alias("stat_med_stand_rwy"),
        pl.col("TAXITIME_SEC_mvt").quantile(0.85).alias("stat_p85_stand_rwy"),
        pl.col("TAXITIME_SEC_mvt").mean().alias("raw_mean_stand_rwy"),
        pl.len().alias("count_stand_rwy"),
    ]).filter(pl.col("count_stand_rwy") >= 5)
    stand_rwy_grp = stand_rwy_grp.with_columns(
        ((pl.col("count_stand_rwy") * pl.col("raw_mean_stand_rwy") + 10.0 * global_mean) / (pl.col("count_stand_rwy") + 10.0)).alias("stat_mean_stand_rwy")
    ).drop(["raw_mean_stand_rwy", "count_stand_rwy"])
    stats["apt_stand_rwy"] = stand_rwy_grp

    # 5b. Airport + Stand + Runway + Aircraft Type (route + aircraft width/speed combo)
    stand_rwy_ac_grp = clean.group_by(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt"]).agg([
        pl.col("TAXITIME_SEC_mvt").median().alias("stat_med_stand_rwy_actype"),
        pl.len().alias("count_stand_rwy_ac"),
    ]).filter(pl.col("count_stand_rwy_ac") >= 8).drop("count_stand_rwy_ac")
    stats["apt_stand_rwy_actype"] = stand_rwy_ac_grp

    # 6. Airport + Operator
    op_grp = clean.group_by(["ADEP_mvt", "operator_clean"]).agg([
        pl.col("TAXITIME_SEC_mvt").median().alias("stat_med_operator"),
        pl.col("TAXITIME_SEC_mvt").mean().alias("raw_mean_op"),
        pl.len().alias("count_op"),
    ])
    op_grp = op_grp.with_columns(
        ((pl.col("count_op") * pl.col("raw_mean_op") + 25.0 * global_mean) / (pl.col("count_op") + 25.0)).alias("stat_mean_operator")
    ).drop(["raw_mean_op", "count_op"])
    stats["operator"] = op_grp

    # 7. Airport + Hour of day
    stats["apt_hour"] = clean.group_by(["ADEP_mvt", "mvt_hour"]).agg([
        pl.col("TAXITIME_SEC_mvt").median().alias("stat_med_apt_hour"),
        pl.col("TAXITIME_SEC_mvt").mean().alias("stat_mean_apt_hour"),
    ])

    # 8. Airport + Month
    stats["apt_month"] = clean.group_by(["ADEP_mvt", "mvt_month"]).agg([
        pl.col("TAXITIME_SEC_mvt").median().alias("stat_med_apt_month"),
        pl.col("TAXITIME_SEC_mvt").mean().alias("stat_mean_apt_month"),
    ])

    # 9. Aircraft Type
    stats["actype"] = clean.group_by(["ADEP_mvt", "AIRCRAFT_TYPE_mvt"]).agg([
        pl.col("TAXITIME_SEC_mvt").median().alias("stat_med_actype"),
    ])

    # 10. Wake turbulence category
    stats["wktbl"] = clean.group_by(["ADEP_mvt", "WK_TBL_CAT_flt"]).agg([
        pl.col("TAXITIME_SEC_mvt").median().alias("stat_med_wktbl"),
    ])

    return stats


def extract_features(
    df: pl.DataFrame,
    target_encodings: dict | None = None,
    attach_weather: bool = True,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray | None]:
    """
    Extract all features for departures.
    
    Returns:
        X_df: Polars DataFrame of all features
        base_pred: NumPy array of baseline prediction from AOBT_3 or fallback
        y: NumPy array of target (or None if test)
    """
    dep = df.filter(pl.col("PHASE_mvt") == "DEP")
    n = len(dep)

    mvt = dep["MVT_TIME_UTC_mvt"]
    sched = dep["SCHED_TIME_UTC_mvt"]
    aobt = dep["AOBT_3_flt"]
    eobt = dep["EOBT_1_flt"]
    iobt = dep["IOBT_flt"]
    lobt = dep["LOBT_flt"]

    # Calculate raw time differences in seconds
    diff_aobt = (mvt - aobt).dt.total_seconds()
    diff_eobt = (mvt - eobt).dt.total_seconds()
    diff_iobt = (mvt - iobt).dt.total_seconds()
    diff_lobt = (mvt - lobt).dt.total_seconds()
    diff_sched = (mvt - sched).dt.total_seconds()

    # Discrepancies between flight plan milestones
    aobt_minus_eobt = (aobt - eobt).dt.total_seconds().clip(-7200, 7200).fill_null(0)
    aobt_minus_sched = (aobt - sched).dt.total_seconds().clip(-7200, 14400).fill_null(0)
    eobt_minus_sched = (eobt - sched).dt.total_seconds().clip(-7200, 14400).fill_null(0)
    eobt_minus_iobt = (eobt - iobt).dt.total_seconds().clip(-7200, 7200).fill_null(0)

    # NEW: LOBT-based delay features — captures ATC slot/CTOT pressure (corr~0.27 with taxi)
    aobt_minus_lobt = (aobt - lobt).dt.total_seconds().clip(-3600, 14400).fill_null(0)
    iobt_minus_lobt = (iobt - lobt).dt.total_seconds().clip(-3600, 7200).fill_null(0)
    eobt_minus_lobt = (eobt - lobt).dt.total_seconds().clip(-3600, 7200).fill_null(0)
    # ATFM delay indicator: large IOBT-LOBT gap = slot holding
    is_atfm_delayed = (iobt_minus_lobt > 900).cast(pl.Int8).fill_null(0)
    atfm_delay_severity = iobt_minus_lobt.clip(0, 7200)  # positive portion only

    # Missing flags
    has_aobt = aobt.is_not_null().cast(pl.Int8)
    has_eobt = eobt.is_not_null().cast(pl.Int8)

    # Stand and pier parsing
    stand_str = dep["STAND_mvt"].cast(pl.Utf8).fill_null("UNK")
    stand_pier = stand_str.str.slice(0, 1)
    rwy_str = dep["RUNWAY_mvt"].cast(pl.Utf8).fill_null("UNK")
    apt_str = dep["ADEP_mvt"].cast(pl.Utf8)
    op_str = dep["AIRCRAFT_OPERATOR_flt"].cast(pl.Utf8).fill_null("UNK")
    actype_str = dep["AIRCRAFT_TYPE_mvt"].cast(pl.Utf8).fill_null("UNK")

    # Dates & temporal
    hour = mvt.dt.hour()
    minute = mvt.dt.minute()
    dow = mvt.dt.weekday()
    month = mvt.dt.month()
    doy = mvt.dt.ordinal_day()
    is_weekend = (dow >= 5).cast(pl.Int8)
    is_winter = month.is_in([1, 2, 3, 11, 12]).cast(pl.Int8)

    # Layout Indicators
    is_helicopter = dep["AIRCRAFT_TYPE_mvt"].is_in(list(HELICOPTER_TYPES)).cast(pl.Int8)
    is_vfr_flight = (dep["FLIGHT_RULE_mvt"].cast(pl.Utf8) == "V").cast(pl.Int8).fill_null(0)
    is_eham_polderbaan = ((apt_str == "EHAM") & (rwy_str.is_in(["18R", "36L"]))).cast(pl.Int8)
    is_lfpg_t2g = ((apt_str == "LFPG") & (stand_pier == "G")).cast(pl.Int8)
    is_lemd_t4s = ((apt_str == "LEMD") & (stand_str.str.slice(0, 1).is_in(["6", "7", "8", "R"]))).cast(pl.Int8)
    is_egll_t5 = ((apt_str == "EGLL") & (stand_str.str.slice(0, 1) == "5")).cast(pl.Int8)

    # Attach pier and hour for target encoding join
    dep_ext = dep.with_columns([
        stand_pier.alias("stand_pier"),
        hour.alias("mvt_hour"),
        month.alias("mvt_month"),
        op_str.alias("operator_clean"),
    ])

    if target_encodings is not None:
        dep_ext = dep_ext.join(target_encodings["apt"], on="ADEP_mvt", how="left")
        dep_ext = dep_ext.join(target_encodings["apt_rwy"], on=["ADEP_mvt", "RUNWAY_mvt"], how="left")
        dep_ext = dep_ext.join(target_encodings["apt_stand"], on=["ADEP_mvt", "STAND_mvt"], how="left")
        dep_ext = dep_ext.join(target_encodings["apt_pier_rwy"], on=["ADEP_mvt", "stand_pier", "RUNWAY_mvt"], how="left")
        dep_ext = dep_ext.join(target_encodings["apt_hour"], on=["ADEP_mvt", "mvt_hour"], how="left")
        dep_ext = dep_ext.join(target_encodings["apt_month"], on=["ADEP_mvt", "mvt_month"], how="left")
        dep_ext = dep_ext.join(target_encodings["actype"], on=["ADEP_mvt", "AIRCRAFT_TYPE_mvt"], how="left")
        dep_ext = dep_ext.join(target_encodings["wktbl"], on=["ADEP_mvt", "WK_TBL_CAT_flt"], how="left")

        if "apt_stand_rwy" in target_encodings:
            dep_ext = dep_ext.join(target_encodings["apt_stand_rwy"], on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")
        else:
            dep_ext = dep_ext.with_columns(pl.lit(None).alias("stat_utt_stand_rwy"))

        if "apt_stand_rwy_actype" in target_encodings:
            dep_ext = dep_ext.join(
                target_encodings["apt_stand_rwy_actype"],
                on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt"],
                how="left"
            )
        else:
            dep_ext = dep_ext.with_columns(pl.lit(None).cast(pl.Float64).alias("stat_med_stand_rwy_actype"))

        if "operator" in target_encodings:
            dep_ext = dep_ext.join(target_encodings["operator"], on=["ADEP_mvt", "operator_clean"], how="left")
        else:
            dep_ext = dep_ext.with_columns(pl.lit(None).alias("stat_mean_operator"))

        med_apt = dep_ext["stat_med_apt"].fill_null(950)
        mean_apt = dep_ext["stat_mean_apt"].fill_null(950)
        med_apt_rwy = dep_ext["stat_med_apt_rwy"].fill_null(med_apt)
        mean_apt_rwy = dep_ext["stat_mean_apt_rwy"].fill_null(mean_apt)
        med_apt_stand = dep_ext["stat_med_apt_stand"].fill_null(med_apt)
        mean_apt_stand = dep_ext["stat_mean_apt_stand"].fill_null(mean_apt)
        med_apt_pier_rwy = dep_ext["stat_med_apt_pier_rwy"].fill_null(med_apt_rwy)
        mean_apt_pier_rwy = dep_ext["stat_mean_apt_pier_rwy"].fill_null(mean_apt_rwy)
        med_apt_hour = dep_ext["stat_med_apt_hour"].fill_null(med_apt)
        med_apt_month = dep_ext["stat_med_apt_month"].fill_null(med_apt)
        med_actype = dep_ext["stat_med_actype"].fill_null(med_apt)
        med_wktbl = dep_ext["stat_med_wktbl"].fill_null(med_apt)
        
        # UTT Hierarchical fallback
        utt_apt_val = dep_ext["stat_utt_apt"].fill_null(600.0) if "stat_utt_apt" in dep_ext.columns else pl.Series("stat_utt_apt", [600.0] * n)
        utt_rwy_val = dep_ext["stat_utt_rwy"].fill_null(utt_apt_val) if "stat_utt_rwy" in dep_ext.columns else utt_apt_val
        utt_pier_val = dep_ext["stat_utt_pier_rwy"].fill_null(utt_rwy_val) if "stat_utt_pier_rwy" in dep_ext.columns else utt_rwy_val
        utt_stand_val = dep_ext["stat_utt_stand_rwy"].fill_null(utt_pier_val) if "stat_utt_stand_rwy" in dep_ext.columns else utt_pier_val
        stat_utt_hierarchical = utt_stand_val.fill_null(600.0)
        # NEW route stats
        stat_med_stand_rwy = dep_ext["stat_med_stand_rwy"].fill_null(med_apt_pier_rwy) if "stat_med_stand_rwy" in dep_ext.columns else med_apt_pier_rwy
        stat_p85_stand_rwy = dep_ext["stat_p85_stand_rwy"].fill_null(mean_apt_pier_rwy) if "stat_p85_stand_rwy" in dep_ext.columns else mean_apt_pier_rwy
        stat_mean_stand_rwy = dep_ext["stat_mean_stand_rwy"].fill_null(mean_apt_pier_rwy) if "stat_mean_stand_rwy" in dep_ext.columns else mean_apt_pier_rwy
        stat_med_stand_rwy_actype = dep_ext["stat_med_stand_rwy_actype"].fill_null(stat_med_stand_rwy) if "stat_med_stand_rwy_actype" in dep_ext.columns else stat_med_stand_rwy
        
        mean_operator = dep_ext["stat_mean_operator"].fill_null(mean_apt) if "stat_mean_operator" in dep_ext.columns else mean_apt
    else:
        med_apt = pl.Series("stat_med_apt", [950.0] * n)
        mean_apt = med_apt
        med_apt_rwy = med_apt
        mean_apt_rwy = med_apt
        med_apt_stand = med_apt
        mean_apt_stand = med_apt
        med_apt_pier_rwy = med_apt
        mean_apt_pier_rwy = med_apt
        med_apt_hour = med_apt
        med_apt_month = med_apt
        med_actype = med_apt
        med_wktbl = med_apt
        stat_utt_hierarchical = pl.Series("stat_utt_hierarchical", [600.0] * n)
        mean_operator = mean_apt
        stat_med_stand_rwy = med_apt
        stat_p85_stand_rwy = med_apt
        stat_mean_stand_rwy = med_apt
        stat_med_stand_rwy_actype = med_apt

    # Base proxy
    diff_aobt_np = diff_aobt.to_numpy()
    valid_aobt_mask = (~np.isnan(diff_aobt_np)) & (diff_aobt_np >= 120) & (diff_aobt_np <= 15000)
    fallback_np = med_apt_pier_rwy.to_numpy()
    base_pred = np.where(valid_aobt_mask, diff_aobt_np - 54.0, fallback_np)
    base_pred = np.clip(base_pred, 60.0, 15000.0)

    # Boolean anomaly and delay indicators
    is_delayed_pushback = ((aobt - eobt).dt.total_seconds() > 1800).fill_null(False).cast(pl.Int8)
    is_early_aobt = ((aobt - eobt).dt.total_seconds() < -600).fill_null(False).cast(pl.Int8)
    is_aobt_past_mvt = (diff_aobt < 0).fill_null(False).cast(pl.Int8)
    has_iobt = iobt.is_not_null().cast(pl.Int8)
    has_lobt = lobt.is_not_null().cast(pl.Int8)

    diff_sched_np = diff_sched.to_numpy()
    has_aobt_np = has_aobt.to_numpy() == 1
    apt_np = apt_str.to_numpy()
    
    # Specific radar anomaly indicator: Italian ANSP at LIRF when AOBT is missing
    is_lirf_no_aobt = ((apt_str == "LIRF") & (has_aobt == 0)).cast(pl.Int8)
    diff_sched_lirf_no_aobt = np.where((apt_np == "LIRF") & (~has_aobt_np), diff_sched_np, 0.0)

    aobt_minus_eobt_raw = (aobt - eobt).dt.total_seconds()
    aobt_minus_sched_raw = (aobt - sched).dt.total_seconds()
    eobt_minus_sched_raw = (eobt - sched).dt.total_seconds()
    iobt_minus_sched_raw = (iobt - sched).dt.total_seconds()
    lobt_minus_sched_raw = (lobt - sched).dt.total_seconds()

    # RECAT-EU class
    recat_classes = [RECAT_EU_MAP.get(ac, "E") for ac in actype_str.to_list()]

    # Airline Alliance & Home Hub priority
    callsigns_list = dep["CALLSIGN_flt"].cast(pl.Utf8).fill_null("UNK").to_list() if "CALLSIGN_flt" in dep.columns else ["UNK"] * n
    apt_list = apt_str.to_list()
    
    alliances = []
    home_hubs = []
    for cs, a in zip(callsigns_list, apt_list):
        pfx = cs[:3]
        if pfx in STAR_ALLIANCE:
            al = "STAR"
        elif pfx in SKYTEAM_ALLIANCE:
            al = "SKYTEAM"
        elif pfx in ONEWORLD_ALLIANCE:
            al = "ONEWORLD"
        elif pfx in LCC_CARRIERS:
            al = "LCC"
        elif pfx in CARGO_CARRIERS:
            al = "CARGO"
        else:
            al = "OTHER"
        alliances.append(al)
        is_home = 1 if pfx in HOME_HUBS.get(a, set()) else 0
        home_hubs.append(is_home)
        
    # Solar Elevation & Day/Night
    solar_elev = compute_solar_elevation(mvt, apt_str)
    is_night = (solar_elev < -6.0).astype(np.int8)
    
    # National Holidays
    m_np = month.to_numpy()
    d_np = dep["MVT_TIME_UTC_mvt"].dt.day().to_numpy()
    is_holiday = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if (m_np[i], d_np[i]) in NATIONAL_HOLIDAYS.get(apt_list[i], set()):
            is_holiday[i] = 1
            
    # Curfew window (night bans)
    h_np = hour.to_numpy()
    is_curfew = np.zeros(n, dtype=np.int8)
    for i in range(n):
        a = apt_list[i]
        h = h_np[i]
        if a == "EDDF" and (h >= 23 or h < 5):
            is_curfew[i] = 1
        elif a == "LSZH" and (h >= 23 or h < 6):
            is_curfew[i] = 1
        elif a == "EGLL" and (h >= 23 or h < 6):
            is_curfew[i] = 1

    # Assemble feature dataframe
    feature_dict = {
        # Airline Alliance & Operational Domain Features
        "airline_alliance": pl.Series("airline_alliance", alliances).cast(pl.Categorical),
        "is_home_hub": pl.Series("is_home_hub", home_hubs).cast(pl.Int8),
        "solar_elevation_deg": pl.Series("solar_elevation_deg", solar_elev.astype(np.float32)),
        "is_night": pl.Series("is_night", is_night).cast(pl.Int8),
        "is_national_holiday": pl.Series("is_national_holiday", is_holiday).cast(pl.Int8),
        "is_curfew_window": pl.Series("is_curfew_window", is_curfew).cast(pl.Int8),

        # Raw continuous timestamp differences
        "diff_aobt_raw": diff_aobt,
        "diff_eobt_raw": diff_eobt,
        "diff_iobt_raw": diff_iobt,
        "diff_lobt_raw": diff_lobt,
        "diff_sched_raw": diff_sched,
        "diff_sched_lirf_no_aobt": pl.Series("diff_sched_lirf_no_aobt", diff_sched_lirf_no_aobt),
        "aobt_minus_eobt_raw": aobt_minus_eobt_raw,
        "aobt_minus_sched_raw": aobt_minus_sched_raw,
        "eobt_minus_sched_raw": eobt_minus_sched_raw,
        "iobt_minus_sched_raw": iobt_minus_sched_raw,
        "lobt_minus_sched_raw": lobt_minus_sched_raw,

        # Clipped variations for stable linear splits
        "diff_aobt_clipped": diff_aobt.clip(60, 7200).fill_null(1000),
        "diff_eobt_clipped": diff_eobt.clip(60, 7200).fill_null(1000),
        "diff_sched_clipped": diff_sched.clip(-1800, 7200),

        # Indicators
        "has_aobt": has_aobt,
        "has_eobt": has_eobt,
        "has_iobt": has_iobt,
        "has_lobt": has_lobt,
        "is_delayed_pushback": is_delayed_pushback,
        "is_early_aobt": is_early_aobt,
        "is_aobt_past_mvt": is_aobt_past_mvt,
        "is_lirf_no_aobt": is_lirf_no_aobt,
        # NEW LOBT/ATFM delay features
        "aobt_minus_lobt": aobt_minus_lobt,
        "iobt_minus_lobt": iobt_minus_lobt,
        "eobt_minus_lobt": eobt_minus_lobt,
        "is_atfm_delayed": is_atfm_delayed,
        "atfm_delay_severity": atfm_delay_severity,
        "is_helicopter": is_helicopter,
        "is_vfr_flight": is_vfr_flight,

        # Airport Layout Indicators
        "is_eham_polderbaan": is_eham_polderbaan,
        "is_lfpg_t2g": is_lfpg_t2g,
        "is_lemd_t4s": is_lemd_t4s,
        "is_egll_t5": is_egll_t5,

        # Base proxy feature
        "base_proxy": pl.Series("base_proxy", base_pred),

        # Temporal
        "mvt_hour": hour,
        "mvt_minute": minute,
        "mvt_dow": dow,
        "mvt_month": month,
        "mvt_doy": doy,
        "is_weekend": is_weekend,
        "is_winter": is_winter,
        "hour_sin": (np.pi * 2 * hour / 24).sin(),
        "hour_cos": (np.pi * 2 * hour / 24).cos(),
        "dow_sin": (np.pi * 2 * dow / 7).sin(),
        "dow_cos": (np.pi * 2 * dow / 7).cos(),

        # Categoricals
        "adep": apt_str.fill_null("UNK").cast(pl.Categorical),
        "runway": rwy_str.fill_null("UNK").cast(pl.Categorical),
        "stand": stand_str.fill_null("UNK").cast(pl.Categorical),
        "stand_pier": stand_pier.fill_null("UNK").cast(pl.Categorical),
        "apt_rwy": (apt_str + "_" + rwy_str).cast(pl.Categorical),
        "apt_pier_rwy": (apt_str + "_" + stand_pier + "_" + rwy_str).cast(pl.Categorical),
        "aircraft_type": actype_str.cast(pl.Categorical),
        "wk_tbl": dep["WK_TBL_CAT_flt"].cast(pl.Utf8).fill_null("UNK").cast(pl.Categorical),
        "recat_eu": pl.Series("recat_eu", recat_classes).cast(pl.Categorical),
        "operator": op_str.cast(pl.Categorical),
        "market_segment": dep["MARKET_SEGMENT_flt"].cast(pl.Utf8).fill_null("UNK").cast(pl.Categorical),
        "flight_rule": dep["FLIGHT_RULE_mvt"].cast(pl.Utf8).fill_null("UNK").cast(pl.Categorical),

        # Target encodings & UTT
        "stat_med_apt": med_apt,
        "stat_mean_apt": mean_apt,
        "stat_med_apt_rwy": med_apt_rwy,
        "stat_mean_apt_rwy": mean_apt_rwy,
        "stat_med_apt_stand": med_apt_stand,
        "stat_mean_apt_stand": mean_apt_stand,
        "stat_med_apt_pier_rwy": med_apt_pier_rwy,
        "stat_mean_apt_pier_rwy": mean_apt_pier_rwy,
        "stat_med_apt_hour": med_apt_hour,
        "stat_med_apt_month": med_apt_month,
        "stat_med_actype": med_actype,
        "stat_med_wktbl": med_wktbl,
        "stat_utt_hierarchical": stat_utt_hierarchical,
        "stat_mean_operator": mean_operator,
        # NEW: Stand-Runway route distance encodings
        "stat_med_stand_rwy": stat_med_stand_rwy,
        "stat_p85_stand_rwy": stat_p85_stand_rwy,
        "stat_mean_stand_rwy": stat_mean_stand_rwy,
        "stat_med_stand_rwy_actype": stat_med_stand_rwy_actype,
    }

    # Add congestion features & log-transformed variations
    for cong_col in [
        "cong_apt_dep_15m", "cong_apt_arr_15m", "cong_apt_tot_15m",
        "cong_apt_dep_30m", "cong_apt_arr_30m", "cong_apt_tot_30m",
        "cong_rwy_dep_15m", "cong_rwy_arr_15m", "cong_rwy_tot_15m",
        "cong_dep_ratio_15m", "dep_arr_flow_ratio", "arr_dep_flow_ratio",
        "rwy_dep_pressure_ratio", "rwy_tot_pressure_ratio"
    ]:
        if cong_col in dep.columns:
            vals = dep[cong_col].fill_null(0)
            feature_dict[cong_col] = vals
            if "tot" in cong_col or "dep" in cong_col:
                feature_dict[f"log_{cong_col}"] = (vals.cast(pl.Float64) + 1.0).log()
        else:
            feature_dict[cong_col] = pl.Series(cong_col, [0] * n)
            if "tot" in cong_col or "dep" in cong_col:
                feature_dict[f"log_{cong_col}"] = pl.Series(f"log_{cong_col}", [0.0] * n)

    # Add runway queue sequencing & wake features
    seq_cols = [
        "seq_dt_prev_dep_rwy", "seq_prev_dep_is_heavy", "seq_prev_is_H", "seq_prev_is_J",
        "seq_icao_wake_sep_sec", "seq_wake_deficit_sec", "recat_sep_sec", "recat_wake_deficit_sec",
        "prev_dep_is_recat_heavy", "rwy_rolling_da_3", "rwy_rolling_da_5", "rwy_rolling_dt_5",
        "dt_prev_any_mvt", "is_prev_mvt_arr"
    ]
    for sc in seq_cols:
        if sc in dep.columns:
            feature_dict[sc] = dep[sc].fill_null(0)
        elif sc.replace("seq_", "") in dep.columns:
            feature_dict[sc] = dep[sc.replace("seq_", "")].fill_null(0)
        else:
            feature_dict[sc] = pl.Series(sc, [0.0] * n)

    # Add instantaneous surface traffic accumulation features (N_active)
    for surf_col in ["active_arr_at_takeoff", "active_arr_at_pushback", "active_dep_at_takeoff", "total_surface_active"]:
        if surf_col in dep.columns:
            vals = dep[surf_col].fill_null(0.0)
            feature_dict[surf_col] = vals
            feature_dict[f"log_{surf_col}"] = (vals.cast(pl.Float64) + 1.0).log()
        else:
            feature_dict[surf_col] = pl.Series(surf_col, [0.0] * n)
            feature_dict[f"log_{surf_col}"] = pl.Series(f"log_{surf_col}", [0.0] * n)

    # Add apron pier contention features (alleyway gridlock)
    for pier_col in ["pier_contention_15m", "pier_contention_30m"]:
        if pier_col in dep.columns:
            vals = dep[pier_col].fill_null(0.0)
            feature_dict[pier_col] = vals
            feature_dict[f"log_{pier_col}"] = (vals.cast(pl.Float64) + 1.0).log()
        else:
            feature_dict[pier_col] = pl.Series(pier_col, [0.0] * n)
            feature_dict[f"log_{pier_col}"] = pl.Series(f"log_{pier_col}", [0.0] * n)

    X_df = pl.DataFrame(feature_dict)

    if attach_weather:
        X_df = attach_weather_features(X_df, dep)
        
    X_df = attach_external_features(X_df, dep)

    y = None
    if "TAXITIME_SEC_mvt" in dep.columns and dep["TAXITIME_SEC_mvt"].null_count() == 0:
        y = dep["TAXITIME_SEC_mvt"].to_numpy()

    return X_df, base_pred, y


def attach_weather_features(X_df: pl.DataFrame, raw_dep: pl.DataFrame, weather_path: Path = None) -> pl.DataFrame:
    """
    Attach open meteorological and de-icing features from weather_hourly.parquet.
    """
    if weather_path is None:
        weather_path = PROJECT_ROOT / "data" / "external" / "weather_hourly.parquet"
    if not weather_path.exists():
        return X_df

    wx = pl.read_parquet(weather_path)
    wx = wx.with_columns(pl.col("weather_time_utc").cast(pl.Datetime("us", "UTC")))

    mvt_hour = raw_dep["MVT_TIME_UTC_mvt"].dt.truncate("1h")
    col_name = "adep_mvt" if "adep_mvt" in raw_dep.columns else "ADEP_mvt"
    dep_join = raw_dep.select([col_name]).with_columns([
        mvt_hour.alias("weather_time_utc")
    ]).rename({col_name: "adep"})

    dep_wx = dep_join.join(wx, on=["adep", "weather_time_utc"], how="left")
    wx_cols = [
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "snowfall", "wind_speed_10m", "wind_direction_10m",
        "wx_is_freezing", "wx_is_deicing_risk"
    ]
    for col in wx_cols:
        val = dep_wx[col].fill_null(0.0)
        col_out = col if col.startswith("wx_") else f"wx_{col}"
        X_df = X_df.with_columns(val.alias(col_out))

    # Additional meteorological & runway aerodynamics
    w_spd = dep_wx["wind_speed_10m"].fill_null(0.0)
    w_dir = dep_wx["wind_direction_10m"].fill_null(0.0)
    rwy_str = raw_dep["RUNWAY_mvt"].cast(pl.Utf8).fill_null("UNK")
    rwy_hdg = rwy_str.str.slice(0, 2).cast(pl.Float64, strict=False) * 10.0
    has_hdg = rwy_hdg.is_not_null()
    hdg_val = rwy_hdg.fill_null(0.0)

    delta_rad = (w_dir - hdg_val) * (np.pi / 180.0)
    headwind = pl.when(has_hdg).then(w_spd * delta_rad.cos()).otherwise(0.0)
    crosswind = pl.when(has_hdg).then((w_spd * delta_rad.sin()).abs()).otherwise(0.0)
    tailwind = pl.when(headwind < -3.0).then(headwind.abs()).otherwise(0.0)
    is_high_crosswind = (crosswind > 25.0).cast(pl.Int8)
    is_tailwind = (headwind < -5.0).cast(pl.Int8)

    extra_cols = [
        headwind.alias("wx_headwind"),
        crosswind.alias("wx_crosswind"),
        tailwind.alias("wx_tailwind"),
        is_high_crosswind.alias("wx_is_high_crosswind"),
        is_tailwind.alias("wx_is_tailwind"),
    ]
    if "dew_point_2m" in dep_wx.columns:
        extra_cols.append(dep_wx["dew_point_2m"].fill_null(0.0).alias("wx_dew_point_2m"))
    if "wind_gusts_10m" in dep_wx.columns:
        extra_cols.append(dep_wx["wind_gusts_10m"].fill_null(0.0).alias("wx_wind_gusts_10m"))
    if "wx_is_fog_lvp" in dep_wx.columns:
        extra_cols.append(dep_wx["wx_is_fog_lvp"].fill_null(0).alias("wx_is_fog_lvp"))
    if "wx_visibility_m" in dep_wx.columns:
        extra_cols.append(dep_wx["wx_visibility_m"].fill_null(9999.0).alias("wx_visibility_m"))
    if "wx_is_lvp_moderate" in dep_wx.columns:
        extra_cols.append(dep_wx["wx_is_lvp_moderate"].fill_null(0).alias("wx_is_lvp_moderate"))
    if "wx_is_lvp_severe" in dep_wx.columns:
        extra_cols.append(dep_wx["wx_is_lvp_severe"].fill_null(0).alias("wx_is_lvp_severe"))
    if "wx_ceiling_ft" in dep_wx.columns:
        extra_cols.append(dep_wx["wx_ceiling_ft"].fill_null(5000.0).alias("wx_ceiling_ft"))
    if "wx_is_freezing_fog" in dep_wx.columns:
        extra_cols.append(dep_wx["wx_is_freezing_fog"].fill_null(0).alias("wx_is_freezing_fog"))
    if "wx_is_snow" in dep_wx.columns:
        extra_cols.append(dep_wx["wx_is_snow"].fill_null(0).alias("wx_is_snow"))
    if "wx_is_thunderstorm" in dep_wx.columns:
        extra_cols.append(dep_wx["wx_is_thunderstorm"].fill_null(0).alias("wx_is_thunderstorm"))
    if "wx_cloud_cover_low" in dep_wx.columns:
        extra_cols.append(dep_wx["wx_cloud_cover_low"].fill_null(0.0).alias("wx_cloud_cover_low"))
    if "wx_qnh_hpa" in dep_wx.columns:
        extra_cols.append(dep_wx["wx_qnh_hpa"].fill_null(1013.25).alias("wx_qnh_hpa"))
    if "wx_dewpoint_dep_c" in dep_wx.columns:
        extra_cols.append(dep_wx["wx_dewpoint_dep_c"].fill_null(5.0).alias("wx_dewpoint_dep_c"))

    X_df = X_df.with_columns(extra_cols)
    return X_df


# Cached external datasets
_AIRPORTS_DF = None
_RUNWAYS_DF = None
_AIRCRAFT_DF = None
_OSM_DF = None
_OSM_MEDIAN_DF = None

ADEP_COORDS = {
    "EDDF": (50.0267, 8.5583, 364, "DE", "EU"),
    "EDDM": (48.3538, 11.7861, 1487, "DE", "EU"),
    "EGLL": (51.4707, -0.4599, 83, "GB", "EU"),
    "EHAM": (52.3086, 4.7639, -11, "NL", "EU"),
    "LEBL": (41.2971, 2.0785, 12, "ES", "EU"),
    "LEMD": (40.4934, -3.5722, 1998, "ES", "EU"),
    "LFPG": (49.0090, 2.5541, 392, "FR", "EU"),
    "LIRF": (41.8045, 12.2520, 13, "IT", "EU"),
    "LSZH": (47.4581, 8.5481, 1417, "CH", "EU"),
    "LTFM": (41.2749, 28.7321, 325, "TR", "AS"),
}


def attach_external_features(X_df: pl.DataFrame, raw_dep: pl.DataFrame) -> pl.DataFrame:
    """
    Attach deep physical features from external datasets:
    1. OurAirports airports.csv: Great circle flight distance to destination, domestic/intercontinental, elevation diff.
    2. OurAirports runways.csv: Physical runway length, width, true heading, displaced threshold, threshold elevation, slope.
    3. ICAO Doc 8643 aircraft_types.csv: MTOW (kg), wingspan (m), length (m), range (km), wing area, fuel capacity, approach speed, engines.
    4. OSM Airport Taxiway Networks: Exact stand-to-runway taxiway routing distance, straight line distance, tortuosity.
    """
    global _AIRPORTS_DF, _RUNWAYS_DF, _AIRCRAFT_DF, _OSM_DF, _OSM_MEDIAN_DF
    ext_dir = PROJECT_ROOT / "data" / "external"

    # 1. Airports
    if _AIRPORTS_DF is None:
        ap_file = ext_dir / "airports.csv"
        if ap_file.exists():
            ap = pl.read_csv(ap_file, infer_schema_length=1000)
            _AIRPORTS_DF = ap.select([
                pl.col("ident").alias("ades_ident"),
                pl.col("latitude_deg").alias("ades_lat"),
                pl.col("longitude_deg").alias("ades_lon"),
                pl.col("elevation_ft").alias("ades_elev_ft"),
                pl.col("iso_country").alias("ades_country"),
                pl.col("continent").alias("ades_continent"),
            ])

    # 2. Runways
    if _RUNWAYS_DF is None:
        rw_file = ext_dir / "runways.csv"
        if rw_file.exists():
            rw = pl.read_csv(rw_file, infer_schema_length=10000)
            rw_sub = rw.filter(pl.col("airport_ident").is_in(list(ADEP_COORDS.keys()))).select([
                pl.col("airport_ident").alias("rw_adep"),
                pl.col("le_ident").alias("rw_ident"),
                pl.col("length_ft").alias("rwy_length_ft"),
                pl.col("width_ft").alias("rwy_width_ft"),
                pl.col("le_heading_degT").alias("rwy_true_hdg"),
                pl.col("le_displaced_threshold_ft").fill_null(0).alias("rwy_disp_thresh_ft"),
                pl.col("le_elevation_ft").cast(pl.Float64, strict=False).fill_null(0.0).alias("rwy_threshold_elev_ft"),
                ((pl.col("he_elevation_ft").cast(pl.Float64, strict=False) - pl.col("le_elevation_ft").cast(pl.Float64, strict=False)) / pl.col("length_ft").cast(pl.Float64, strict=False) * 100.0).fill_null(0.0).alias("rwy_slope_pct"),
            ])
            rw_recip = rw.filter(pl.col("airport_ident").is_in(list(ADEP_COORDS.keys()))).filter(pl.col("he_ident").is_not_null()).select([
                pl.col("airport_ident").alias("rw_adep"),
                pl.col("he_ident").alias("rw_ident"),
                pl.col("length_ft").alias("rwy_length_ft"),
                pl.col("width_ft").alias("rwy_width_ft"),
                pl.col("he_heading_degT").alias("rwy_true_hdg"),
                pl.col("he_displaced_threshold_ft").fill_null(0).alias("rwy_disp_thresh_ft"),
                pl.col("he_elevation_ft").cast(pl.Float64, strict=False).fill_null(0.0).alias("rwy_threshold_elev_ft"),
                ((pl.col("le_elevation_ft").cast(pl.Float64, strict=False) - pl.col("he_elevation_ft").cast(pl.Float64, strict=False)) / pl.col("length_ft").cast(pl.Float64, strict=False) * 100.0).fill_null(0.0).alias("rwy_slope_pct"),
            ])
            _RUNWAYS_DF = pl.concat([rw_sub, rw_recip]).unique(subset=["rw_adep", "rw_ident"])

    # 3. Aircraft specs & Engine physics
    if _AIRCRAFT_DF is None:
        at_file = ext_dir / "aircraft_types.csv"
        if at_file.exists():
            at = pl.read_csv(at_file, infer_schema_length=10000)
            desc_str = at["description"].fill_null("L2J")
            num_eng = [int(s[1]) if len(s) >= 2 and s[1].isdigit() else 2 for s in desc_str.to_list()]
            is_jet = [1 if len(s) >= 3 and s[2] == "J" else 0 for s in desc_str.to_list()]
            is_prop = [1 if len(s) >= 3 and s[2] == "T" else 0 for s in desc_str.to_list()]

            _AIRCRAFT_DF = at.select([
                pl.col("icao_code").alias("ac_icao"),
                pl.col("mtow").alias("ac_mtow_kg"),
                pl.col("wingspan").alias("ac_wingspan_m"),
                pl.col("length").alias("ac_length_m"),
                pl.col("operating_range").alias("ac_range_km"),
                pl.col("wing_area").cast(pl.Float64, strict=False).fill_null(120.0).alias("ac_wing_area_m2"),
                pl.col("fuel_capacity").cast(pl.Float64, strict=False).fill_null(25000.0).alias("ac_fuel_capacity_l"),
                pl.col("approach_speed").cast(pl.Float64, strict=False).fill_null(140.0).alias("ac_approach_speed_kts"),
            ]).with_columns([
                pl.Series("ac_num_engines", num_eng).cast(pl.Int8),
                pl.Series("ac_is_jet", is_jet).cast(pl.Int8),
                pl.Series("ac_is_turboprop", is_prop).cast(pl.Int8),
                ((pl.col("ac_mtow_kg").fill_null(75000.0) > 140000.0) | (pl.Series("ac_num_engines", num_eng) > 2)).cast(pl.Int8).alias("ac_is_widebody"),
            ])

    # 4. OSM Taxiway Network Distance Lookups
    if _OSM_DF is None:
        osm_file = ext_dir / "osm_taxiway_distances.parquet"
        if osm_file.exists():
            _OSM_DF = pl.read_parquet(osm_file).select([
                pl.col("adep").alias("osm_adep"),
                pl.col("stand").alias("osm_stand"),
                pl.col("runway").alias("osm_runway"),
                pl.col("osm_dist_m"),
                pl.col("sl_dist_m"),
                pl.col("tortuosity").alias("osm_tortuosity"),
            ])
            _OSM_MEDIAN_DF = _OSM_DF.group_by(["osm_adep", "osm_runway"]).agg([
                pl.col("osm_dist_m").median().alias("rwy_median_osm_dist_m"),
                pl.col("sl_dist_m").median().alias("rwy_median_sl_dist_m"),
                pl.col("osm_tortuosity").median().alias("rwy_median_tortuosity"),
            ])

    dep_ext = raw_dep.select(["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"])
    if _AIRPORTS_DF is not None:
        dep_ext = dep_ext.join(_AIRPORTS_DF, left_on="ADES_mvt", right_on="ades_ident", how="left")
        adep_lat = [ADEP_COORDS.get(a, (50.0, 10.0, 300, "EU", "EU"))[0] for a in dep_ext["ADEP_mvt"].to_list()]
        adep_lon = [ADEP_COORDS.get(a, (50.0, 10.0, 300, "EU", "EU"))[1] for a in dep_ext["ADEP_mvt"].to_list()]
        adep_elev = [ADEP_COORDS.get(a, (50.0, 10.0, 300, "EU", "EU"))[2] for a in dep_ext["ADEP_mvt"].to_list()]
        adep_country = [ADEP_COORDS.get(a, (50.0, 10.0, 300, "EU", "EU"))[3] for a in dep_ext["ADEP_mvt"].to_list()]

        dep_ext = dep_ext.with_columns([
            pl.Series("adep_lat", adep_lat),
            pl.Series("adep_lon", adep_lon),
            pl.Series("adep_elev_ft", adep_elev),
            pl.Series("adep_country", adep_country),
        ])

        lat1 = dep_ext["adep_lat"] * (np.pi / 180.0)
        lon1 = dep_ext["adep_lon"] * (np.pi / 180.0)
        lat2 = dep_ext["ades_lat"].fill_null(dep_ext["adep_lat"]) * (np.pi / 180.0)
        lon2 = dep_ext["ades_lon"].fill_null(dep_ext["adep_lon"]) * (np.pi / 180.0)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (dlat / 2.0).sin()**2 + lat1.cos() * lat2.cos() * (dlon / 2.0).sin()**2
        c = 2.0 * a.sqrt().arcsin()
        dist_km = c * 6371.0

        X_df = X_df.with_columns([
            dist_km.alias("flight_dist_km"),
            (dep_ext["adep_country"] == dep_ext["ades_country"].fill_null("UNK")).cast(pl.Int8).alias("is_domestic"),
            (dep_ext["ades_continent"].fill_null("EU") != "EU").cast(pl.Int8).alias("is_intercontinental"),
            (dist_km > 3500.0).cast(pl.Int8).alias("is_long_haul"),
            (dist_km < 600.0).cast(pl.Int8).alias("is_short_haul"),
            (dep_ext["ades_elev_ft"].fill_null(0) - dep_ext["adep_elev_ft"]).alias("elev_diff_ft"),
        ])
    else:
        X_df = X_df.with_columns([
            pl.Series("flight_dist_km", [1000.0] * len(X_df)),
            pl.Series("is_domestic", [0] * len(X_df)),
            pl.Series("is_intercontinental", [0] * len(X_df)),
            pl.Series("is_long_haul", [0] * len(X_df)),
            pl.Series("is_short_haul", [0] * len(X_df)),
            pl.Series("elev_diff_ft", [0.0] * len(X_df)),
        ])

    if _RUNWAYS_DF is not None:
        r_join = dep_ext.join(_RUNWAYS_DF, left_on=["ADEP_mvt", "RUNWAY_mvt"], right_on=["rw_adep", "rw_ident"], how="left")
        X_df = X_df.with_columns([
            r_join["rwy_length_ft"].fill_null(10000.0).alias("rwy_length_ft"),
            r_join["rwy_width_ft"].fill_null(150.0).alias("rwy_width_ft"),
            r_join["rwy_true_hdg"].fill_null(90.0).alias("rwy_true_hdg"),
            r_join["rwy_disp_thresh_ft"].fill_null(0.0).alias("rwy_disp_thresh_ft"),
            r_join["rwy_threshold_elev_ft"].fill_null(300.0).alias("rwy_threshold_elev_ft"),
            r_join["rwy_slope_pct"].fill_null(0.0).alias("rwy_slope_pct"),
        ])
    else:
        X_df = X_df.with_columns([
            pl.Series("rwy_length_ft", [10000.0] * len(X_df)),
            pl.Series("rwy_width_ft", [150.0] * len(X_df)),
            pl.Series("rwy_true_hdg", [90.0] * len(X_df)),
            pl.Series("rwy_disp_thresh_ft", [0.0] * len(X_df)),
            pl.Series("rwy_threshold_elev_ft", [300.0] * len(X_df)),
            pl.Series("rwy_slope_pct", [0.0] * len(X_df)),
        ])

    if _AIRCRAFT_DF is not None:
        a_join = dep_ext.join(_AIRCRAFT_DF, left_on="AIRCRAFT_TYPE_mvt", right_on="ac_icao", how="left")
        X_df = X_df.with_columns([
            a_join["ac_mtow_kg"].fill_null(75000.0).alias("ac_mtow_kg"),
            a_join["ac_wingspan_m"].fill_null(35.0).alias("ac_wingspan_m"),
            a_join["ac_length_m"].fill_null(37.0).alias("ac_length_m"),
            a_join["ac_range_km"].fill_null(6000.0).alias("ac_range_km"),
            a_join["ac_wing_area_m2"].fill_null(120.0).alias("ac_wing_area_m2"),
            a_join["ac_fuel_capacity_l"].fill_null(25000.0).alias("ac_fuel_capacity_l"),
            a_join["ac_approach_speed_kts"].fill_null(140.0).alias("ac_approach_speed_kts"),
            a_join["ac_num_engines"].fill_null(2).alias("ac_num_engines"),
            a_join["ac_is_jet"].fill_null(1).alias("ac_is_jet"),
            a_join["ac_is_turboprop"].fill_null(0).alias("ac_is_turboprop"),
            a_join["ac_is_widebody"].fill_null(0).alias("ac_is_widebody"),
        ])
    else:
        X_df = X_df.with_columns([
            pl.Series("ac_mtow_kg", [75000.0] * len(X_df)),
            pl.Series("ac_wingspan_m", [35.0] * len(X_df)),
            pl.Series("ac_length_m", [37.0] * len(X_df)),
            pl.Series("ac_range_km", [6000.0] * len(X_df)),
            pl.Series("ac_wing_area_m2", [120.0] * len(X_df)),
            pl.Series("ac_fuel_capacity_l", [25000.0] * len(X_df)),
            pl.Series("ac_approach_speed_kts", [140.0] * len(X_df)),
            pl.Series("ac_num_engines", [2] * len(X_df)),
            pl.Series("ac_is_jet", [1] * len(X_df)),
            pl.Series("ac_is_turboprop", [0] * len(X_df)),
            pl.Series("ac_is_widebody", [0] * len(X_df)),
        ])

    # 4. Attach OSM Taxiway Network Distance & Tortuosity
    if _OSM_DF is not None:
        o_join = dep_ext.join(_OSM_DF, left_on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], right_on=["osm_adep", "osm_stand", "osm_runway"], how="left")
        if _OSM_MEDIAN_DF is not None:
            o_join = o_join.join(_OSM_MEDIAN_DF, left_on=["ADEP_mvt", "RUNWAY_mvt"], right_on=["osm_adep", "osm_runway"], how="left")
            dist = o_join["osm_dist_m"].fill_null(o_join["rwy_median_osm_dist_m"]).fill_null(2500.0)
            sl = o_join["sl_dist_m"].fill_null(o_join["rwy_median_sl_dist_m"]).fill_null(1800.0)
            tort = o_join["osm_tortuosity"].fill_null(o_join["rwy_median_tortuosity"]).fill_null(1.35)
        else:
            dist = o_join["osm_dist_m"].fill_null(2500.0)
            sl = o_join["sl_dist_m"].fill_null(1800.0)
            tort = o_join["osm_tortuosity"].fill_null(1.35)

        X_df = X_df.with_columns([
            dist.alias("osm_taxiway_dist_m"),
            sl.alias("osm_sl_dist_m"),
            tort.alias("osm_tortuosity"),
        ])
    else:
        X_df = X_df.with_columns([
            pl.Series("osm_taxiway_dist_m", [2500.0] * len(X_df)),
            pl.Series("osm_sl_dist_m", [1800.0] * len(X_df)),
            pl.Series("osm_tortuosity", [1.35] * len(X_df)),
        ])

    return X_df

