"""
High-Precision, Fast Feature Extraction for PRC Data Challenge 2026.
Pure vectorized Polars + NumPy implementation for seasonal training.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np

EXTERNAL_DIR = Path(__file__).resolve().parent.parent / "data" / "external"
_WEATHER_DF = None
_OSM_DF = None

def get_weather_df() -> pl.DataFrame:
    global _WEATHER_DF
    if _WEATHER_DF is None:
        p = EXTERNAL_DIR / "weather_hourly.parquet"
        if p.exists():
            _WEATHER_DF = pl.read_parquet(p)
        else:
            _WEATHER_DF = pl.DataFrame()
    return _WEATHER_DF

def get_osm_df() -> pl.DataFrame:
    global _OSM_DF
    if _OSM_DF is None:
        p = EXTERNAL_DIR / "osm_taxiway_distances.parquet"
        if p.exists():
            _OSM_DF = pl.read_parquet(p)
        else:
            _OSM_DF = pl.DataFrame()
    return _OSM_DF

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
}

RUNWAY_HEADINGS = {
    ("EDDF", "07C"): 69.6, ("EDDF", "25C"): 249.6,
    ("EDDF", "07L"): 69.6, ("EDDF", "25R"): 249.6,
    ("EDDF", "07R"): 69.6, ("EDDF", "25L"): 249.6,
    ("EDDF", "18"): 180.0,
    ("EDDM", "08L"): 83.4, ("EDDM", "26R"): 263.4,
    ("EDDM", "08R"): 83.4, ("EDDM", "26L"): 263.4,
    ("EGLL", "09L"): 90.0, ("EGLL", "27R"): 270.0,
    ("EGLL", "09R"): 90.0, ("EGLL", "27L"): 270.0,
    ("EHAM", "04"): 41.0, ("EHAM", "22"): 221.0,
    ("EHAM", "06"): 58.0, ("EHAM", "24"): 238.0,
    ("EHAM", "09"): 87.0, ("EHAM", "27"): 267.0,
    ("EHAM", "18C"): 183.0, ("EHAM", "36C"): 3.0,
    ("EHAM", "18L"): 183.0, ("EHAM", "36R"): 3.0,
    ("EHAM", "18R"): 183.0, ("EHAM", "36L"): 3.0,
    ("LEBL", "02"): 19.0, ("LEBL", "20"): 199.0,
    ("LEBL", "06L"): 66.0, ("LEBL", "24R"): 246.0,
    ("LEBL", "06R"): 66.0, ("LEBL", "24L"): 246.0,
    ("LEMD", "14L"): 144.2, ("LEMD", "32R"): 324.2,
    ("LEMD", "14R"): 144.2, ("LEMD", "32L"): 324.2,
    ("LEMD", "18L"): 181.0, ("LEMD", "36R"): 1.0,
    ("LEMD", "18R"): 181.0, ("LEMD", "36L"): 2.0,
    ("LFPG", "08L"): 85.0, ("LFPG", "26R"): 265.0,
    ("LFPG", "08R"): 85.1, ("LFPG", "26L"): 265.1,
    ("LFPG", "09L"): 85.3, ("LFPG", "27R"): 265.3,
    ("LFPG", "09R"): 86.0, ("LFPG", "27L"): 266.0,
    ("LIRF", "07"): 70.0, ("LIRF", "25"): 250.0,
    ("LIRF", "16C"): 163.0, ("LIRF", "34C"): 343.0,
    ("LIRF", "16L"): 163.0, ("LIRF", "34R"): 343.0,
    ("LIRF", "16R"): 163.0, ("LIRF", "34L"): 343.0,
    ("LSZH", "10"): 96.0, ("LSZH", "28"): 276.0,
    ("LSZH", "14"): 137.0, ("LSZH", "32"): 317.0,
    ("LSZH", "16"): 155.0, ("LSZH", "34"): 335.0,
    ("LTFM", "09"): 89.0, ("LTFM", "27"): 269.0,
    ("LTFM", "16L"): 174.0, ("LTFM", "34R"): 354.0,
    ("LTFM", "16R"): 174.0, ("LTFM", "34L"): 354.0,
    ("LTFM", "17L"): 174.0, ("LTFM", "35R"): 354.0,
    ("LTFM", "17R"): 174.0, ("LTFM", "35L"): 354.0,
    ("LTFM", "18"): 179.0, ("LTFM", "36"): 354.0,
}


def compute_target_encodings_clean(train_dep_df: pl.DataFrame) -> dict:
    """Computes robust historical target encodings on clean training departures (60s <= taxi <= 7200s)."""
    clean = train_dep_df.filter(
        (pl.col("TAXITIME_SEC_mvt") >= 60) & (pl.col("TAXITIME_SEC_mvt") <= 7200)
    )
    global_median = float(clean["TAXITIME_SEC_mvt"].median())
    global_mean = float(clean["TAXITIME_SEC_mvt"].mean())
    
    # 1. Airport + Stand + Runway
    sr = clean.group_by(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]).agg([
        pl.col("TAXITIME_SEC_mvt").median().alias("enc_med_stand_rwy"),
        pl.col("TAXITIME_SEC_mvt").mean().alias("raw_mean_sr"),
        pl.col("TAXITIME_SEC_mvt").quantile(0.15).alias("enc_utt_stand_rwy"),
        pl.len().alias("count_sr")
    ])
    sr = sr.with_columns(
        ((pl.col("count_sr") * pl.col("raw_mean_sr") + 15.0 * global_mean) / (pl.col("count_sr") + 15.0)).alias("enc_mean_stand_rwy")
    ).drop(["raw_mean_sr", "count_sr"])

    # 2. Airport + Runway
    rwy = clean.group_by(["ADEP_mvt", "RUNWAY_mvt"]).agg([
        pl.col("TAXITIME_SEC_mvt").median().alias("enc_med_rwy"),
        pl.col("TAXITIME_SEC_mvt").mean().alias("raw_mean_rwy"),
        pl.len().alias("count_rwy")
    ])
    rwy = rwy.with_columns(
        ((pl.col("count_rwy") * pl.col("raw_mean_rwy") + 15.0 * global_mean) / (pl.col("count_rwy") + 15.0)).alias("enc_mean_rwy")
    ).drop(["raw_mean_rwy", "count_rwy"])

    # 3. Airport + Stand
    stand = clean.group_by(["ADEP_mvt", "STAND_mvt"]).agg([
        pl.col("TAXITIME_SEC_mvt").median().alias("enc_med_stand"),
        pl.col("TAXITIME_SEC_mvt").mean().alias("raw_mean_st"),
        pl.len().alias("count_st")
    ])
    stand = stand.with_columns(
        ((pl.col("count_st") * pl.col("raw_mean_st") + 15.0 * global_mean) / (pl.col("count_st") + 15.0)).alias("enc_mean_stand")
    ).drop(["raw_mean_st", "count_st"])

    # 4. Airport + Operator
    clean_op = clean.with_columns(clean["AIRCRAFT_OPERATOR_flt"].cast(pl.Utf8).fill_null("UNK").alias("op_clean"))
    op = clean_op.group_by(["ADEP_mvt", "op_clean"]).agg([
        pl.col("TAXITIME_SEC_mvt").median().alias("enc_med_operator"),
    ])

    return {
        "global_median": global_median,
        "global_mean": global_mean,
        "stand_rwy": sr,
        "rwy": rwy,
        "stand": stand,
        "operator": op,
    }


def extract_features_seasonal(df: pl.DataFrame, encs: dict | None = None) -> tuple[pd.DataFrame, pl.DataFrame]:
    """
    Extracts features for departures from df (which contains both DEP and ARR).
    Uses full traffic (both DEP and ARR) to calculate airport-level and runway-level congestion queues.
    """
    # 1. Filter departures
    dep = df.filter(pl.col("PHASE_mvt") == "DEP")
    n_dep = len(dep)

    mvt = dep["MVT_TIME_UTC_mvt"]
    sched = dep["SCHED_TIME_UTC_mvt"]
    aobt = dep["AOBT_3_flt"]
    eobt = dep["EOBT_1_flt"]
    iobt = dep["IOBT_flt"]
    lobt = dep["LOBT_flt"]

    da = (mvt - aobt).dt.total_seconds().cast(pl.Float64)
    ds = (mvt - sched).dt.total_seconds().cast(pl.Float64)

    # 2. Runway & Airport Congestion Queues
    # Use all movements sorted by MVT_TIME_UTC_mvt
    mvt_all = df.sort("MVT_TIME_UTC_mvt")
    dep_apt_all = mvt_all["ADEP_mvt"].to_numpy()
    des_apt_all = mvt_all["ADES_mvt"].to_numpy()
    phase_all = mvt_all["PHASE_mvt"].to_numpy()
    rwy_all = mvt_all["RUNWAY_mvt"].to_numpy()
    t_all = mvt_all["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()

    # Crucial fix: Arrivals are at ADES_mvt, Departures are at ADEP_mvt!
    apt_all = np.where(phase_all == "DEP", dep_apt_all, des_apt_all)

    # Query times for departures
    dep_apt = dep["ADEP_mvt"].to_numpy()
    dep_rwy = dep["RUNWAY_mvt"].to_numpy()
    t_dep = mvt.dt.epoch("s").to_numpy()

    rwy_dep_q_15 = np.zeros(n_dep, dtype=np.float32)
    rwy_dep_q_30 = np.zeros(n_dep, dtype=np.float32)
    rwy_all_q_15 = np.zeros(n_dep, dtype=np.float32)
    apt_dep_q_15 = np.zeros(n_dep, dtype=np.float32)
    apt_dep_q_30 = np.zeros(n_dep, dtype=np.float32)
    apt_arr_q_15 = np.zeros(n_dep, dtype=np.float32)
    apt_arr_q_30 = np.zeros(n_dep, dtype=np.float32)
    arr_taxi_15m = np.zeros(n_dep, dtype=np.float32)
    arr_taxi_30m = np.zeros(n_dep, dtype=np.float32)
    arr_taxi_60m = np.zeros(n_dep, dtype=np.float32)
    arr_taxi_trend = np.zeros(n_dep, dtype=np.float32)
    dt_prev_dep_rwy = np.zeros(n_dep, dtype=np.float32)
    dt_prev_any_rwy = np.zeros(n_dep, dtype=np.float32)
    is_prev_arr = np.zeros(n_dep, dtype=np.float32)

    # Arrival taxi times for ground congestion tracking
    taxi_all = mvt_all["TAXITIME_SEC_mvt"].fill_null(0).to_numpy()

    unique_airports = np.unique(dep_apt)
    for apt in unique_airports:
        # Flights at this airport in all movements
        mask_apt_dep = (apt_all == apt) & (phase_all == "DEP")
        mask_apt_arr = (apt_all == apt) & (phase_all == "ARR")
        mask_apt_all = (apt_all == apt)
        
        t_apt_dep = t_all[mask_apt_dep]
        t_apt_arr = t_all[mask_apt_arr]
        t_apt_all = t_all[mask_apt_all]
        tx_apt_arr = taxi_all[mask_apt_arr]
        
        dep_mask_apt = (dep_apt == apt)
        t_queries = t_dep[dep_mask_apt]
        idx_dep_in_output = np.where(dep_mask_apt)[0]

        # Airport departures in last 15 and 30 mins
        i15_d = np.searchsorted(t_apt_dep, t_queries - 900, side="left")
        i_curr_d = np.searchsorted(t_apt_dep, t_queries, side="right")
        apt_dep_q_15[idx_dep_in_output] = np.maximum(0, i_curr_d - i15_d - 1)

        i30_d = np.searchsorted(t_apt_dep, t_queries - 1800, side="left")
        apt_dep_q_30[idx_dep_in_output] = np.maximum(0, i_curr_d - i30_d - 1)

        # Airport arrivals in last 15 and 30 mins (NOW ACCURATELY CAPTURING FULL ARRIVALS!)
        i15_a = np.searchsorted(t_apt_arr, t_queries - 900, side="left")
        i_curr_a = np.searchsorted(t_apt_arr, t_queries, side="right")
        apt_arr_q_15[idx_dep_in_output] = np.maximum(0, i_curr_a - i15_a)

        i30_a = np.searchsorted(t_apt_arr, t_queries - 1800, side="left")
        apt_arr_q_30[idx_dep_in_output] = np.maximum(0, i_curr_a - i30_a)

        # Real-time Arrival Taxi-In Congestion tracking
        if len(t_apt_arr) > 0:
            cum_tx = np.concatenate([[0.0], np.cumsum(tx_apt_arr)])
            mean_apt_tx = float(np.mean(tx_apt_arr)) if len(tx_apt_arr) > 0 else 540.0
            
            # 15m rolling mean
            cnt_15 = i_curr_a - i15_a
            sum_15 = cum_tx[i_curr_a] - cum_tx[i15_a]
            m15 = np.where(cnt_15 > 0, sum_15 / np.maximum(1, cnt_15), mean_apt_tx)
            arr_taxi_15m[idx_dep_in_output] = m15

            # 30m rolling mean
            cnt_30 = i_curr_a - i30_a
            sum_30 = cum_tx[i_curr_a] - cum_tx[i30_a]
            m30 = np.where(cnt_30 > 0, sum_30 / np.maximum(1, cnt_30), mean_apt_tx)
            arr_taxi_30m[idx_dep_in_output] = m30
            
            # 60m rolling mean
            i60_a = np.searchsorted(t_apt_arr, t_queries - 3600, side="left")
            cnt_60 = i_curr_a - i60_a
            sum_60 = cum_tx[i_curr_a] - cum_tx[i60_a]
            m60 = np.where(cnt_60 > 0, sum_60 / np.maximum(1, cnt_60), mean_apt_tx)
            arr_taxi_60m[idx_dep_in_output] = m60

            # Real-time Congestion Surge Trend (15m vs 60m velocity)
            arr_taxi_trend[idx_dep_in_output] = np.clip(m15 / np.maximum(50.0, m60), 0.2, 5.0)

        # Runway level queues & preceding movement separation
        rwys_apt_dep = rwy_all[mask_apt_dep]
        rwys_apt_all = rwy_all[mask_apt_all]
        phase_apt_all = phase_all[mask_apt_all]

        dep_rwys_query = dep_rwy[dep_mask_apt]
        for rwy in np.unique(dep_rwys_query):
            mask_rwy_dep = (rwys_apt_dep == rwy)
            t_rwy_dep = t_apt_dep[mask_rwy_dep]
            
            mask_rwy_all = (rwys_apt_all == rwy)
            t_rwy_all = t_apt_all[mask_rwy_all]
            ph_rwy_all = phase_apt_all[mask_rwy_all]

            mask_sub = (dep_rwys_query == rwy)
            t_q_rwy = t_queries[mask_sub]
            idx_out_rwy = idx_dep_in_output[mask_sub]

            if len(t_rwy_dep) > 0:
                ir_15 = np.searchsorted(t_rwy_dep, t_q_rwy - 900, side="left")
                ir_30 = np.searchsorted(t_rwy_dep, t_q_rwy - 1800, side="left")
                ir_curr = np.searchsorted(t_rwy_dep, t_q_rwy, side="right")

                rwy_dep_q_15[idx_out_rwy] = np.maximum(0, ir_curr - ir_15 - 1)
                rwy_dep_q_30[idx_out_rwy] = np.maximum(0, ir_curr - ir_30 - 1)

                # Preceding departure time
                prev_idx = np.maximum(0, ir_curr - 2)
                prev_t = t_rwy_dep[prev_idx]
                dt_prev = np.where(ir_curr >= 2, t_q_rwy - prev_t, 1800.0)
                dt_prev_dep_rwy[idx_out_rwy] = np.clip(dt_prev, 0, 3600)

            # Preceding ANY movement on this runway (DEP or ARR)
            if len(t_rwy_all) > 0:
                ir_all_15 = np.searchsorted(t_rwy_all, t_q_rwy - 900, side="left")
                ir_all_curr = np.searchsorted(t_rwy_all, t_q_rwy, side="right")
                rwy_all_q_15[idx_out_rwy] = np.maximum(0, ir_all_curr - ir_all_15 - 1)

                prev_all_idx = np.maximum(0, ir_all_curr - 2)
                prev_all_t = t_rwy_all[prev_all_idx]
                prev_all_ph = ph_rwy_all[prev_all_idx]
                dt_any = np.where(ir_all_curr >= 2, t_q_rwy - prev_all_t, 1800.0)
                dt_prev_any_rwy[idx_out_rwy] = np.clip(dt_any, 0, 3600)
                is_prev_arr[idx_out_rwy] = np.where((ir_all_curr >= 2) & (prev_all_ph == "ARR"), 1.0, 0.0)

    # Stand and pier parsing
    stand_str = dep["STAND_mvt"].cast(pl.Utf8).fill_null("UNK")
    stand_pier = stand_str.str.slice(0, 1)
    apt_str = dep["ADEP_mvt"].cast(pl.Utf8)
    rwy_str = dep["RUNWAY_mvt"].cast(pl.Utf8).fill_null("UNK")
    actype_str = dep["AIRCRAFT_TYPE_mvt"].cast(pl.Utf8).fill_null("UNK")

    recat_classes = [RECAT_EU_MAP.get(ac, "E") for ac in actype_str.to_list()]

    # Layout indicators
    is_eham_polderbaan = ((apt_str == "EHAM") & (rwy_str.is_in(["18R", "36L"]))).cast(pl.Int8).to_numpy()
    is_lfpg_t2g = ((apt_str == "LFPG") & (stand_pier == "G")).cast(pl.Int8).to_numpy()
    is_lemd_t4s = ((apt_str == "LEMD") & (stand_str.str.slice(0, 1).is_in(["6", "7", "8", "R"]))).cast(pl.Int8).to_numpy()
    is_egll_t5 = ((apt_str == "EGLL") & (stand_str.str.slice(0, 1) == "5")).cast(pl.Int8).to_numpy()

    # Milestone differences
    aobt_minus_eobt = (aobt - eobt).dt.total_seconds().clip(-7200, 7200).fill_null(0.0).to_numpy()
    aobt_minus_iobt = (aobt - iobt).dt.total_seconds().clip(-7200, 7200).fill_null(0.0).to_numpy()
    aobt_minus_lobt = (aobt - lobt).dt.total_seconds().clip(-7200, 7200).fill_null(0.0).to_numpy()
    eobt_minus_sched = (eobt - sched).dt.total_seconds().clip(-7200, 7200).fill_null(0.0).to_numpy()
    eobt_minus_iobt = (eobt - iobt).dt.total_seconds().clip(-7200, 7200).fill_null(0.0).to_numpy()
    lobt_minus_iobt = (lobt - iobt).dt.total_seconds().clip(-7200, 7200).fill_null(0.0).to_numpy()
    sched_minus_iobt = (sched - iobt).dt.total_seconds().clip(-7200, 7200).fill_null(0.0).to_numpy()

    hour = mvt.dt.hour().to_numpy()
    minute = mvt.dt.minute().to_numpy()
    dow = mvt.dt.weekday().to_numpy()
    has_aobt = aobt.is_not_null().cast(pl.Int8).to_numpy()

    feat_dict = {
        # Core continuous milestone differences
        "da": da.fill_null(1000.0).to_numpy(),
        "ds": ds.to_numpy(),
        "has_aobt": has_aobt,
        "aobt_minus_eobt": aobt_minus_eobt,
        "aobt_minus_iobt": aobt_minus_iobt,
        "aobt_minus_lobt": aobt_minus_lobt,
        "eobt_minus_sched": eobt_minus_sched,
        "eobt_minus_iobt": eobt_minus_iobt,
        "lobt_minus_iobt": lobt_minus_iobt,
        "sched_minus_iobt": sched_minus_iobt,

        # Surface and runway congestion
        "rwy_dep_q_15": rwy_dep_q_15,
        "rwy_dep_q_30": rwy_dep_q_30,
        "rwy_all_q_15": rwy_all_q_15,
        "apt_dep_q_15": apt_dep_q_15,
        "apt_dep_q_30": apt_dep_q_30,
        "apt_arr_q_15": apt_arr_q_15,
        "apt_arr_q_30": apt_arr_q_30,
        "arr_taxi_15m": arr_taxi_15m,
        "arr_taxi_30m": arr_taxi_30m,
        "arr_taxi_60m": arr_taxi_60m,
        "arr_taxi_trend": arr_taxi_trend,
        "dt_prev_dep_rwy": dt_prev_dep_rwy,
        "dt_prev_any_rwy": dt_prev_any_rwy,
        "is_prev_arr": is_prev_arr,

        # Temporal
        "hour": hour,
        "minute": minute,
        "dow": dow,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),

        # Layout flags
        "is_eham_polderbaan": is_eham_polderbaan,
        "is_lfpg_t2g": is_lfpg_t2g,
        "is_lemd_t4s": is_lemd_t4s,
        "is_egll_t5": is_egll_t5,

        # Categoricals
        "adep": apt_str.to_pandas().astype("category"),
        "ades": dep["ADES_mvt"].cast(pl.Utf8).fill_null("UNK").to_pandas().astype("category"),
        "runway": rwy_str.to_pandas().astype("category"),
        "stand": stand_str.to_pandas().astype("category"),
        "stand_pier": stand_pier.to_pandas().astype("category"),
        "actype": actype_str.to_pandas().astype("category"),
        "operator": dep["AIRCRAFT_OPERATOR_flt"].cast(pl.Utf8).fill_null("UNK").to_pandas().astype("category"),
        "wktbl": dep["WK_TBL_CAT_flt"].cast(pl.Utf8).fill_null("UNK").to_pandas().astype("category"),
        "mkt_segment": dep["MARKET_SEGMENT_flt"].cast(pl.Utf8).fill_null("UNK").to_pandas().astype("category"),
        "flight_type": dep["FLIGHT_TYPE_flt"].cast(pl.Utf8).fill_null("UNK").to_pandas().astype("category"),
        "recat_eu": pd.Series(recat_classes, dtype="category"),
    }

    # Add Target Encodings if provided
    if encs is not None:
        dep_enc = dep.select(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt"])
        dep_enc = dep_enc.with_columns(dep_enc["AIRCRAFT_OPERATOR_flt"].cast(pl.Utf8).fill_null("UNK").alias("op_clean"))
        
        dep_enc = dep_enc.join(encs["stand_rwy"], on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")
        dep_enc = dep_enc.join(encs["rwy"], on=["ADEP_mvt", "RUNWAY_mvt"], how="left")
        dep_enc = dep_enc.join(encs["stand"], on=["ADEP_mvt", "STAND_mvt"], how="left")
        dep_enc = dep_enc.join(encs["operator"], on=["ADEP_mvt", "op_clean"], how="left")

        med_global = encs["global_median"]
        mean_global = encs["global_mean"]

        enc_rwy = dep_enc["enc_med_rwy"].fill_null(med_global)
        enc_stand = dep_enc["enc_med_stand"].fill_null(med_global)
        enc_sr = dep_enc["enc_med_stand_rwy"].fill_null(enc_rwy)
        enc_mean_sr = dep_enc["enc_mean_stand_rwy"].fill_null(mean_global)
        enc_utt_sr = dep_enc["enc_utt_stand_rwy"].fill_null(enc_rwy * 0.8)
        enc_op = dep_enc["enc_med_operator"].fill_null(med_global)

        feat_dict["enc_med_stand_rwy"] = enc_sr.to_numpy()
        feat_dict["enc_mean_stand_rwy"] = enc_mean_sr.to_numpy()
        feat_dict["enc_utt_stand_rwy"] = enc_utt_sr.to_numpy()
        feat_dict["enc_med_rwy"] = enc_rwy.to_numpy()
        feat_dict["enc_med_stand"] = enc_stand.to_numpy()
        feat_dict["enc_med_operator"] = enc_op.to_numpy()

    # Add External Weather (METAR/De-icing) and OSM Physical Distances
    wx = get_weather_df()
    osm = get_osm_df()

    if len(wx) > 0 or len(osm) > 0:
        dep_ext = dep.with_columns(
            dep["MVT_TIME_UTC_mvt"].dt.truncate("1h").alias("weather_time_utc"),
            dep["ADEP_mvt"].alias("adep_ext"),
            dep["STAND_mvt"].alias("stand_ext"),
            dep["RUNWAY_mvt"].alias("runway_ext"),
        )
        if len(wx) > 0:
            dep_ext = dep_ext.join(wx, left_on=["adep_ext", "weather_time_utc"], right_on=["adep", "weather_time_utc"], how="left")
            for c in ["snowfall", "wx_is_deicing_risk", "wx_is_freezing_fog", "temperature_2m", "wind_speed_10m", "wx_visibility_m"]:
                if c in dep_ext.columns:
                    feat_dict[c] = dep_ext[c].fill_null(-1.0).to_numpy()
            
            # Compute aerodynamic runway wind decomposition
            if "wind_speed_10m" in dep_ext.columns and "wind_direction_10m" in dep_ext.columns:
                w_spd = dep_ext["wind_speed_10m"].fill_null(0.0).to_numpy()
                w_dir = dep_ext["wind_direction_10m"].fill_null(0.0).to_numpy()
                adep_v = dep["ADEP_mvt"].cast(pl.Utf8).to_numpy()
                rwy_v = dep["RUNWAY_mvt"].cast(pl.Utf8).to_numpy()
                
                rwy_hdg = np.array([RUNWAY_HEADINGS.get((adep_v[i], rwy_v[i]), np.nan) for i in range(len(adep_v))])
                has_hdg = ~np.isnan(rwy_hdg)
                delta_rad = np.radians(w_dir - rwy_hdg)
                
                feat_dict["headwind"] = np.where(has_hdg, w_spd * np.cos(delta_rad), 0.0)
                feat_dict["crosswind"] = np.where(has_hdg, np.abs(w_spd * np.sin(delta_rad)), 0.0)
                feat_dict["is_tailwind"] = np.where(has_hdg & (feat_dict["headwind"] < -2.0), 1.0, 0.0)
        if len(osm) > 0:
            dep_ext = dep_ext.join(osm, left_on=["adep_ext", "stand_ext", "runway_ext"], right_on=["adep", "stand", "runway"], how="left")
            for c in ["osm_dist_m", "sl_dist_m", "tortuosity"]:
                if c in dep_ext.columns:
                    feat_dict[c] = dep_ext[c].fill_null(-1.0).to_numpy()

    X_df = pd.DataFrame(feat_dict)
    return X_df, dep
