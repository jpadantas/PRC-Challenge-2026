"""
Physics feature engine (paper, Section 2.3). Adds nine features per departure:

1. rwy_queue_at_push, apt_queue_at_push: departures of the same runway / airport that are
   between off-block (NM AOBT, or MVT - 900 s if missing) and take-off when the flight pushes
   back. Computed with an O(N log N) event sweep.
2. prev_dep_wake_penalty, dt_to_prev_dep, is_behind_heavy: time-based separation from a
   RECAT-EU-like matrix (WAKE_SEPARATION_SECONDS) minus the observed interval to the preceding
   departure on the same runway; unmapped aircraft types default to category E.
3. utt_osm_seconds: OpenStreetMap stand-to-runway distance divided by 9 m/s.
4. pier_queue_15m: pushbacks in the preceding 15 min from stands sharing the same leading letter
   (stands that start with a digit are not grouped).
5. is_lirf_gate_hold, is_extreme_apron_hold: regime indicators (Regimes 2B and 3).
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

RECAT_EU_MAP = {
    "A388": "A",
    "B77W": "B", "B772": "B", "B773": "B", "B77L": "B", "B77F": "B",
    "B744": "B", "B748": "B", "A359": "B", "A35K": "B", "A343": "B", "A346": "B", "MD11": "B",
    "B788": "C", "B789": "C", "B78X": "C", "A332": "C", "A333": "C", "A339": "C", "B763": "C", "B764": "C",
    "B752": "D", "B753": "D",
    "A320": "E", "A321": "E", "A319": "E", "A20N": "E", "A21N": "E", "A19N": "E",
    "B738": "E", "B737": "E", "B739": "E", "B38M": "E", "B39M": "E", "B734": "E",
    "E190": "E", "E195": "E", "E290": "E", "E295": "E", "BCS1": "E", "BCS3": "E", "CRJ9": "E", "CRK": "E",
    "AT72": "F", "AT76": "F", "AT45": "F", "DH8D": "F", "E145": "F", "E170": "F", "E75L": "F", "CRJ2": "F",
}

WAKE_SEPARATION_SECONDS = {
    ("A", "A"): 100, ("A", "B"): 120, ("A", "C"): 140, ("A", "D"): 160, ("A", "E"): 180, ("A", "F"): 200,
    ("B", "A"): 90,  ("B", "B"): 100, ("B", "C"): 120, ("B", "D"): 140, ("B", "E"): 120, ("B", "F"): 180,
    ("C", "A"): 90,  ("C", "B"): 90,  ("C", "C"): 100, ("C", "D"): 120, ("C", "E"): 120, ("C", "F"): 180,
    ("D", "A"): 75,  ("D", "B"): 75,  ("D", "C"): 75,  ("D", "D"): 80,  ("D", "E"): 90,  ("D", "F"): 120,
    ("E", "A"): 60,  ("E", "B"): 60,  ("E", "C"): 60,  ("E", "D"): 60,  ("E", "E"): 70,  ("E", "F"): 120,
    ("F", "A"): 60,  ("F", "B"): 60,  ("F", "C"): 60,  ("F", "D"): 60,  ("F", "E"): 60,  ("F", "F"): 60,
}

_OSM_DF = None

def get_osm_df() -> pl.DataFrame:
    global _OSM_DF
    if _OSM_DF is None:
        p = EXTERNAL_DIR / "osm_taxiway_distances.parquet"
        if p.exists():
            _OSM_DF = pl.read_parquet(p)
        else:
            _OSM_DF = pl.DataFrame()
    return _OSM_DF


def compute_fast_queue_events(t_push: np.ndarray, t_takeoff: np.ndarray) -> np.ndarray:
    """
    O(N log N) event simulation counting how many aircraft are active (pushed back but not taken off)
    at the exact moment each aircraft pushes back.
    """
    n = len(t_push)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    # Events: +1 at t_push, -1 at t_takeoff
    events_time = np.concatenate([t_push, t_takeoff])
    events_delta = np.concatenate([np.ones(n, dtype=np.int32), -np.ones(n, dtype=np.int32)])

    order = np.argsort(events_time, kind="mergesort")
    sorted_time = events_time[order]
    cum_active = np.cumsum(events_delta[order])

    # For each aircraft query at t_push[i]
    idx = np.searchsorted(sorted_time, t_push, side="right") - 1
    # Subtract 1 to not count itself
    return np.maximum(0, cum_active[idx] - 1).astype(np.float32)


def compute_physics_features_fast(df: pl.DataFrame) -> pl.DataFrame:
    """
    Vectorized computation of all physics features.
    """
    dep = df.filter(pl.col("PHASE_mvt") == "DEP")
    n_dep = len(dep)

    mvt_epoch = dep["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
    has_aobt = dep["AOBT_3_flt"].is_not_null().to_numpy()
    aobt_epoch = np.where(has_aobt, dep["AOBT_3_flt"].dt.epoch("s").to_numpy(), mvt_epoch - 900)
    
    apt = dep["ADEP_mvt"].to_numpy()
    rwy = dep["RUNWAY_mvt"].cast(pl.Utf8).fill_null("UNK").to_numpy()
    stand = dep["STAND_mvt"].cast(pl.Utf8).fill_null("UNK").to_numpy()
    ac_type = dep["AIRCRAFT_TYPE_mvt"].cast(pl.Utf8).fill_null("UNK").to_numpy()

    recat = np.array([RECAT_EU_MAP.get(act, "E") for act in ac_type])

    # 1. Exact Runway and Airport Queues at Pushback (O(N log N))
    rwy_queue_at_push = np.zeros(n_dep, dtype=np.float32)
    apt_queue_at_push = np.zeros(n_dep, dtype=np.float32)

    for airport in np.unique(apt):
        apt_mask = (apt == airport)
        apt_idx = np.where(apt_mask)[0]
        t_p_apt = aobt_epoch[apt_mask]
        t_t_apt = mvt_epoch[apt_mask]
        rwy_apt = rwy[apt_mask]

        # Airport-level active queue at pushback
        apt_queue_at_push[apt_idx] = compute_fast_queue_events(t_p_apt, t_t_apt)

        # Runway-level active queue at pushback
        for runway in np.unique(rwy_apt):
            if runway == "UNK":
                continue
            r_mask = (rwy_apt == runway)
            r_idx_global = apt_idx[r_mask]
            rwy_queue_at_push[r_idx_global] = compute_fast_queue_events(t_p_apt[r_mask], t_t_apt[r_mask])

    # 2. Wake Vortex Penalty from Preceding Departure
    prev_dep_wake_penalty = np.zeros(n_dep, dtype=np.float32)
    dt_to_prev_dep = np.full(n_dep, 9999.0, dtype=np.float32)
    is_behind_heavy = np.zeros(n_dep, dtype=np.int8)

    for airport in np.unique(apt):
        for runway in np.unique(rwy[apt == airport]):
            if runway == "UNK":
                continue
            rwy_mask = (apt == airport) & (rwy == runway)
            rwy_idx = np.where(rwy_mask)[0]
            if len(rwy_idx) < 2:
                continue

            order = np.argsort(mvt_epoch[rwy_idx])
            sorted_idx = rwy_idx[order]
            sorted_mvt = mvt_epoch[sorted_idx]
            sorted_recat = recat[sorted_idx]

            # Vectorized consecutive differences
            dts = np.diff(sorted_mvt).astype(np.float32)
            dt_to_prev_dep[sorted_idx[1:]] = dts

            prev_recats = sorted_recat[:-1]
            curr_recats = sorted_recat[1:]

            for k in range(len(dts)):
                p_r = prev_recats[k]
                c_r = curr_recats[k]
                target_idx = sorted_idx[k + 1]

                if p_r in ("A", "B", "C"):
                    is_behind_heavy[target_idx] = 1

                req_sep = WAKE_SEPARATION_SECONDS.get((p_r, c_r), 70)
                prev_dep_wake_penalty[target_idx] = max(0.0, req_sep - dts[k])

    # 3. OSM Taxiway Network Distances & Unimpeded Taxi Time (UTT)
    osm_df = get_osm_df()
    if len(osm_df) > 0:
        dep_keys = pl.DataFrame({
            "MVT_ID_mvt": dep["MVT_ID_mvt"],
            "adep": apt,
            "stand": stand,
            "runway": rwy,
        })
        dep_osm = dep_keys.join(osm_df, on=["adep", "stand", "runway"], how="left")
        osm_dist = dep_osm["osm_dist_m"].fill_null(np.nan).to_numpy().astype(np.float32)
        sl_dist = dep_osm["sl_dist_m"].fill_null(np.nan).to_numpy().astype(np.float32)
        tortuosity = dep_osm["tortuosity"].fill_null(np.nan).to_numpy().astype(np.float32)
    else:
        osm_dist = np.full(n_dep, np.nan, dtype=np.float32)
        sl_dist = np.full(n_dep, np.nan, dtype=np.float32)
        tortuosity = np.full(n_dep, np.nan, dtype=np.float32)

    utt_osm_seconds = np.where(~np.isnan(osm_dist), osm_dist / 9.0, np.nan).astype(np.float32)

    # 4. Terminal Pier / Stand Contention
    pier = [s[:1] if len(s) > 1 and s[:1].isalpha() else "UNK" for s in stand]
    pier_arr = np.array(pier)
    pier_queue_15m = np.zeros(n_dep, dtype=np.float32)

    for airport in np.unique(apt):
        apt_mask = (apt == airport)
        apt_idx = np.where(apt_mask)[0]
        apt_pier = pier_arr[apt_mask]
        apt_push = aobt_epoch[apt_mask]

        for p_code in np.unique(apt_pier):
            if p_code == "UNK":
                continue
            p_mask = (apt_pier == p_code)
            p_idx = apt_idx[p_mask]
            p_push = apt_push[p_mask]

            if len(p_push) < 2:
                continue

            order = np.argsort(p_push)
            sorted_idx = p_idx[order]
            sorted_push = p_push[order]

            # searchsorted for 15m window
            i_15 = np.searchsorted(sorted_push, sorted_push - 900, side="left")
            n_recent = np.arange(len(sorted_push)) - i_15
            pier_queue_15m[sorted_idx] = n_recent.astype(np.float32)

    # 5. Regime indicators (Regimes 2B and 3)
    # d_sched = MVT - SCHED
    d_sched = (dep["MVT_TIME_UTC_mvt"] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds().cast(pl.Float64).to_numpy()
    is_lirf_gate_hold = ((apt == "LIRF") & (~has_aobt) & (d_sched > 7200.0)).astype(np.int8)
    is_extreme_apron_hold = (has_aobt & ((mvt_epoch - aobt_epoch) > 7200.0)).astype(np.int8)

    res_df = pl.DataFrame({
        "MVT_ID_mvt": dep["MVT_ID_mvt"],
        "rwy_queue_at_push": rwy_queue_at_push,
        "apt_queue_at_push": apt_queue_at_push,
        "prev_dep_wake_penalty": prev_dep_wake_penalty,
        "dt_to_prev_dep": dt_to_prev_dep,
        "is_behind_heavy": is_behind_heavy,
        "osm_dist_m": osm_dist,
        "sl_dist_m": sl_dist,
        "tortuosity": tortuosity,
        "utt_osm_seconds": utt_osm_seconds,
        "pier_queue_15m": pier_queue_15m,
        "is_lirf_gate_hold": is_lirf_gate_hold,
        "is_extreme_apron_hold": is_extreme_apron_hold,
    })
    return res_df


if __name__ == "__main__":
    print("Testing physics feature extraction on full month...")
    sample = pl.read_parquet("data/raw/training_2025-01-01_2025-02-01.parquet")
    import time
    t0 = time.time()
    feats = compute_physics_features_fast(sample)
    print(f"Features computed: {feats.shape} in {time.time()-t0:.2f}s!")
    print(feats.head(5))
