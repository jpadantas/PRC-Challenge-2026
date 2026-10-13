"""
Train Next-Gen Tabular ResNet Offset Regressor (195 Features) on NVIDIA GeForce RTX 4060 GPU with CUDA.
Integrates tactical flight plan milestones (d_ai, d_al, d_li, d_ei, d_si), empirical 12-month UTT lookup,
and airline market segment categories.
Saves models/tabular_resnet_offset_nextgen.pt and scratch/resnet_offset_nextgen_test_preds.npy.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import time
import pickle
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features
from src.models_neural import train_tabular_resnet, predict_tabular_resnet

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
SCRATCH_DIR = PROJECT_ROOT / "scratch"
ALL_12_MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]


def extract_advanced_matrix(df: pl.DataFrame, utt_stats: pl.DataFrame, target_encs: dict | None = None) -> tuple[pd.DataFrame, pl.DataFrame]:
    X_pl, _, _ = features.extract_features(df, target_encodings=target_encs, attach_weather=True)
    dep = df.filter(pl.col("PHASE_mvt") == "DEP")

    dep = dep.join(utt_stats, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")

    aobt = dep["AOBT_3_flt"]
    sched = dep["SCHED_TIME_UTC_mvt"]
    iobt = dep["IOBT_flt"]
    lobt = dep["LOBT_flt"]
    eobt = dep["EOBT_1_flt"]

    d_ai = (aobt - iobt).dt.total_seconds().clip(-7200, 14400).fill_null(0.0).to_numpy()
    d_al = (aobt - lobt).dt.total_seconds().clip(-7200, 14400).fill_null(0.0).to_numpy()
    d_li = (lobt - iobt).dt.total_seconds().clip(-3600, 7200).fill_null(0.0).to_numpy()
    d_ei = (eobt - iobt).dt.total_seconds().clip(-3600, 7200).fill_null(0.0).to_numpy()
    d_si = (sched - iobt).dt.total_seconds().clip(-7200, 14400).fill_null(0.0).to_numpy()

    utt_p10 = dep["utt_p10"].fill_null(600.0).to_numpy()
    utt_p25 = dep["utt_p25"].fill_null(750.0).to_numpy()
    utt_p50 = dep["utt_p50"].fill_null(950.0).to_numpy()

    X_df = X_pl.to_pandas()
    X_df["tactical_d_ai"] = d_ai
    X_df["tactical_d_al"] = d_al
    X_df["tactical_d_li"] = d_li
    X_df["tactical_d_ei"] = d_ei
    X_df["tactical_d_si"] = d_si
    X_df["full_year_utt_p10"] = utt_p10
    X_df["full_year_utt_p25"] = utt_p25
    X_df["full_year_utt_p50"] = utt_p50
    X_df["mkt_segment"] = dep["MARKET_SEGMENT_flt"].to_pandas().astype("category")
    X_df["flight_type"] = dep["FLIGHT_TYPE_flt"].to_pandas().astype("category")
    X_df["wk_tbl_cat"] = dep["WK_TBL_CAT_flt"].to_pandas().astype("category")

    for c in X_df.select_dtypes(include=["object"]).columns:
        X_df[c] = X_df[c].astype("category")

    return X_df, dep


def main():
    print("=" * 80)
    print("🧠 TRAINING NEXT-GEN TABULAR RESNET OFFSET REGRESSOR (195 FEATURES) ON CUDA GPU")
    print("=" * 80)

    t0 = time.time()
    utt_stats = pl.read_parquet(SCRATCH_DIR / "empirical_utt_lookup.parquet")
    with open(MODELS_DIR / "encs_v75_nextgen.pkl", "rb") as f:
        encs = pickle.load(f)
    print(f"Loaded UTT profiles ({len(utt_stats):,}) and Target Encodings.")

    print(f"[{time.strftime('%H:%M:%S')}] Loading all 12 months of 2025...")
    df = features.load_training_data(RAW_DIR, months=ALL_12_MONTHS)
    print(f"Loaded {len(df):,} total rows in {time.time() - t0:.1f}s")

    print(f"[{time.strftime('%H:%M:%S')}] Computing domain features...")
    df = features.compute_congestion_metrics(df)
    df = features.compute_surface_accumulation_features(df)
    df = features.compute_apron_contention_features(df)
    df = features.compute_runway_sequence_features(df)

    dep_df = df.filter(pl.col("PHASE_mvt") == "DEP")
    print(f"[{time.strftime('%H:%M:%S')}] Extracting advanced feature matrix (195 features)...")
    X_pd, dep_ext = extract_advanced_matrix(df, utt_stats, target_encs=encs)

    aobt_col = dep_ext["AOBT_3_flt"]
    block_col = dep_ext["BLOCK_TIME_UTC_mvt"]
    sched_col = dep_ext["SCHED_TIME_UTC_mvt"]
    adep_col = dep_ext["ADEP_mvt"].to_numpy()

    has_aobt = aobt_col.is_not_null().to_numpy()
    y_offset = (block_col - aobt_col).dt.total_seconds().to_numpy()

    d_bs = (block_col - sched_col).dt.total_seconds().to_numpy()
    is_lirf_default = (adep_col == "LIRF") & (np.abs(d_bs) <= 15)

    valid_mask = has_aobt & (~is_lirf_default) & (y_offset >= -3600) & (y_offset <= 3600)
    X_train = X_pd.iloc[valid_mask].reset_index(drop=True)
    y_train = y_offset[valid_mask]
    print(f"ResNet Training Matrix: {len(X_train):,} rows ({np.mean(valid_mask):.2%} of departures), {X_train.shape[1]} features.")
    print(f"  Target stats: mean={np.mean(y_train):.2f}s, std={np.std(y_train):.2f}s, median={np.median(y_train):.2f}s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[{time.strftime('%H:%M:%S')}] Training Tabular ResNet on {device.upper()} (12 epochs, batch_size=4096)...")
    model, save_dict = train_tabular_resnet(
        X_train, y_train, epochs=12, batch_size=4096, lr=2e-3, d_model=512, n_blocks=4, device=device,
        min_val=-3600.0, max_val=3600.0
    )

    torch.save(save_dict, MODELS_DIR / "tabular_resnet_offset_nextgen.pt")
    print(f"💾 Model saved to models/tabular_resnet_offset_nextgen.pt")

    # Generate Test Predictions on ranking.parquet
    print(f"\n[{time.strftime('%H:%M:%S')}] Generating Test Predictions on ranking test set...")
    rank_df = pl.read_parquet(RAW_DIR / "ranking.parquet")
    rank_df = features.compute_congestion_metrics(rank_df)
    rank_df = features.compute_surface_accumulation_features(rank_df)
    rank_df = features.compute_apron_contention_features(rank_df)
    rank_df = features.compute_runway_sequence_features(rank_df)

    X_test_pd, dep_test_ext = extract_advanced_matrix(rank_df, utt_stats, target_encs=encs)
    resnet_preds = predict_tabular_resnet(save_dict, X_test_pd, device=device, min_clip=-3600.0, max_clip=3600.0)

    out_npy = SCRATCH_DIR / "resnet_offset_nextgen_test_preds.npy"
    np.save(out_npy, resnet_preds)
    print(f"💾 Saved Next-Gen Tabular ResNet offset test predictions: {out_npy}")
    print(f"   Stats: mean={resnet_preds.mean():.2f}s, std={resnet_preds.std():.2f}s, min={resnet_preds.min():.2f}s, max={resnet_preds.max():.2f}s")
    print(f"🎉 Next-Gen ResNet execution completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
