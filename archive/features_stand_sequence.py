"""
STAND SEQUENCE FEATURE: When 2+ aircraft depart from the same stand within 20min,
the alley is congested. The 2nd aircraft has longer taxi because it has to wait for
the previous aircraft to clear the alley before it can push back properly.

This feature is distinct from pier_contention (which counts ALL departures on the pier)
— this specifically captures the "tail-gating" effect within the same stand.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import polars as pl
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    import sys; sys.path.insert(0, str(PROJECT_ROOT))


def compute_stand_sequence_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes stand-level sequencing features for departures:
    - same_stand_prev_dt: seconds since previous departure from same stand (negative = no previous)
    - same_stand_2nd_aircraft: flag if another aircraft departed same stand in past 20min
    - same_stand_busy_15m: count of departures from same stand in past 15min
    - same_stand_n_rolling: rolling count of departures from same stand in past 30min
    """
    print("Computing stand sequence features (same-stand tailgating)...")
    dep_df = df.filter(pl.col("PHASE_mvt") == "DEP")
    n_dep = len(dep_df)

    stand_arr = dep_df["STAND_mvt"].cast(pl.Utf8).fill_null("UNK").to_numpy()
    apt_arr = dep_df["ADEP_mvt"].to_numpy()
    mvt_epoch = dep_df["MVT_TIME_UTC_mvt"].dt.epoch(time_unit="s").to_numpy()
    aobt_epoch = dep_df["AOBT_3_flt"].dt.epoch(time_unit="s").to_numpy() if "AOBT_3_flt" in dep_df.columns else np.full(n_dep, np.nan)
    valid_aobt = ~np.isnan(aobt_epoch)
    t_push = np.where(valid_aobt, aobt_epoch, mvt_epoch - 900)

    same_stand_prev_dt = np.full(n_dep, -1.0, dtype=np.float32)  # -1 = no prev
    same_stand_is_2nd = np.zeros(n_dep, dtype=np.int8)
    same_stand_busy_15m = np.zeros(n_dep, dtype=np.float32)
    same_stand_busy_30m = np.zeros(n_dep, dtype=np.float32)

    # Group by airport + stand, sort by pushback time
    airports = np.unique(apt_arr)
    for apt in airports:
        apt_mask = (apt_arr == apt)
        apt_idx = np.where(apt_mask)[0]
        apt_stands = stand_arr[apt_mask]
        apt_push = t_push[apt_mask]

        unique_stands = np.unique(apt_stands)
        for stand in unique_stands:
            if stand in ("UNK", ""):
                continue
            s_mask = (apt_stands == stand)
            s_idx = apt_idx[s_mask]
            s_push = apt_push[s_mask]

            if len(s_push) < 2:
                continue

            sort_order = np.argsort(s_push)
            s_idx_sorted = s_idx[sort_order]
            s_push_sorted = s_push[sort_order]

            for i in range(len(s_push_sorted)):
                orig_idx = s_idx_sorted[i]
                t_i = s_push_sorted[i]

                # Previous departure from same stand
                if i > 0:
                    dt = t_i - s_push_sorted[i - 1]
                    same_stand_prev_dt[orig_idx] = dt
                    if dt < 1200:  # within 20 min
                        same_stand_is_2nd[orig_idx] = 1

                # Count departures in past 15min and 30min (before this aircraft)
                before = s_push_sorted[:i]
                n_15m = np.sum(before >= t_i - 900)
                n_30m = np.sum(before >= t_i - 1800)
                same_stand_busy_15m[orig_idx] = n_15m
                same_stand_busy_30m[orig_idx] = n_30m

    feats = pl.DataFrame({
        "MVT_ID_mvt": dep_df["MVT_ID_mvt"],
        "same_stand_prev_dt": same_stand_prev_dt,
        "same_stand_is_2nd": same_stand_is_2nd,
        "same_stand_busy_15m": same_stand_busy_15m,
        "same_stand_busy_30m": same_stand_busy_30m,
    })
    return df.join(feats, on="MVT_ID_mvt", how="left")


if __name__ == "__main__":
    from src import features as feat_module
    RAW_DIR = PROJECT_ROOT / "data" / "raw"

    print("Testing stand sequence features on 1 month of training data...")
    t0 = __import__("time").time()
    df = feat_module.load_training_data(RAW_DIR, months=["2025-06"])
    df = feat_module.compute_congestion_metrics(df)
    df = compute_stand_sequence_features(df)

    dep = df.filter(pl.col("PHASE_mvt") == "DEP")

    # Correlation with taxi time
    for col in ["same_stand_prev_dt", "same_stand_is_2nd", "same_stand_busy_15m", "same_stand_busy_30m"]:
        valid = dep.filter(pl.col(col).is_not_null() & pl.col("TAXITIME_SEC_mvt").is_not_null() & (pl.col(col) >= 0))
        if len(valid) > 100:
            corr = np.corrcoef(valid[col].to_numpy(), valid["TAXITIME_SEC_mvt"].to_numpy())[0,1]
            n_nonzero = (valid[col].to_numpy() > 0).sum()
            print(f"  {col}: corr={corr:.4f}  n_nonzero={n_nonzero:,}  n_total={len(valid):,}")

    # Distribution of same_stand_is_2nd
    n_2nd = dep["same_stand_is_2nd"].fill_null(0).sum()
    print(f"\n  same_stand_is_2nd: {n_2nd:,}/{len(dep):,} flights ({n_2nd/len(dep)*100:.1f}%)")
    taxi_2nd = dep.filter(pl.col("same_stand_is_2nd") == 1)["TAXITIME_SEC_mvt"].drop_nulls().to_numpy()
    taxi_1st = dep.filter(pl.col("same_stand_is_2nd") == 0)["TAXITIME_SEC_mvt"].drop_nulls().to_numpy()
    print(f"  Mean taxi (1st): {taxi_1st.mean():.1f}s  Mean taxi (2nd in <20min): {taxi_2nd.mean():.1f}s  DIFF: +{taxi_2nd.mean()-taxi_1st.mean():.1f}s")
    print(f"\nDone in {__import__('time').time()-t0:.1f}s")
