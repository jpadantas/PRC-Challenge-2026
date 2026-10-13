"""
Train Tabular ResNet 2.0 Offset Regressor with all 182 physical, aerodynamic, METAR, and OSM taxiway network features on NVIDIA GPU CUDA.
Predicts delta_AOBT = BLOCK_TIME_UTC_mvt - AOBT_3_flt.
Saves models/tabular_resnet_offset.pt and scratch/resnet_offset_test_preds.npy.
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


def main():
    print("=" * 80)
    print("🧠 TRAINING TABULAR RESNET 2.0 OFFSET REGRESSOR (182 FEATURES) ON GPU CUDA")
    print("=" * 80)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading all 12 months of 2025...")
    df = features.load_training_data(RAW_DIR, months=ALL_12_MONTHS)
    print(f"Loaded {len(df):,} total rows in {time.time() - t0:.1f}s")

    print(f"[{time.strftime('%H:%M:%S')}] Computing domain features...")
    df = features.compute_congestion_metrics(df)
    df = features.compute_surface_accumulation_features(df)
    df = features.compute_apron_contention_features(df)
    df = features.compute_runway_sequence_features(df)

    dep_df = df.filter(pl.col("PHASE_mvt") == "DEP")
    print(f"[{time.strftime('%H:%M:%S')}] Computing target encodings across {len(dep_df):,} departures...")
    encs = features.compute_target_encodings(dep_df)

    print(f"[{time.strftime('%H:%M:%S')}] Extracting all 182 features...")
    X_pl, _, _ = features.extract_features(df, target_encodings=encs, attach_weather=True)
    X_pd = X_pl.to_pandas()

    # Target: delta_offset = BLOCK_TIME - AOBT_3
    aobt_col = dep_df["AOBT_3_flt"]
    block_col = dep_df["BLOCK_TIME_UTC_mvt"]
    sched_col = dep_df["SCHED_TIME_UTC_mvt"]
    adep_col = dep_df["ADEP_mvt"].to_numpy()

    has_aobt = aobt_col.is_not_null().to_numpy()
    y_offset = (block_col - aobt_col).dt.total_seconds().to_numpy()

    # Filter out LIRF schedule-defaults (|BLOCK - SCHED| <= 15s)
    d_bs = (block_col - sched_col).dt.total_seconds().to_numpy()
    is_lirf_default = (adep_col == "LIRF") & (np.abs(d_bs) <= 15)

    valid_mask = has_aobt & (~is_lirf_default) & (y_offset >= -3600) & (y_offset <= 3600)

    X_train = X_pd.iloc[valid_mask].reset_index(drop=True)
    y_train = y_offset[valid_mask]
    print(f"ResNet Training Matrix ready: {len(X_train):,} rows ({np.mean(valid_mask):.2%} of departures), {X_train.shape[1]} features.")
    print(f"  Target stats: mean={np.mean(y_train):.2f}s, std={np.std(y_train):.2f}s, median={np.median(y_train):.2f}s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[{time.strftime('%H:%M:%S')}] Launching Tabular ResNet Offset Training on {device.upper()} (12 epochs)...")
    model, save_dict = train_tabular_resnet(
        X_train, y_train, epochs=12, batch_size=4096, lr=2e-3, d_model=512, n_blocks=4, device=device,
        min_val=-3600.0, max_val=3600.0
    )

    torch.save(save_dict, MODELS_DIR / "tabular_resnet_offset.pt")
    with open(MODELS_DIR / "encs_resnet_offset.pkl", "wb") as f:
        pickle.dump(encs, f)

    print(f"\n[{time.strftime('%H:%M:%S')}] Generating Test Predictions on ranking.parquet...")
    rank_df = pl.read_parquet(RAW_DIR / "ranking.parquet")
    rank_df = features.compute_congestion_metrics(rank_df)
    rank_df = features.compute_surface_accumulation_features(rank_df)
    rank_df = features.compute_apron_contention_features(rank_df)
    rank_df = features.compute_runway_sequence_features(rank_df)

    dep_rank = rank_df.filter(pl.col("PHASE_mvt") == "DEP")
    X_test_pl, _, _ = features.extract_features(dep_rank, target_encodings=encs, attach_weather=True)
    X_test_pd = X_test_pl.to_pandas()

    resnet_preds = predict_tabular_resnet(save_dict, X_test_pd, device=device, min_clip=-3600.0, max_clip=3600.0)
    out_npy = SCRATCH_DIR / "resnet_offset_test_preds.npy"
    np.save(out_npy, resnet_preds)
    print(f"💾 Saved Tabular ResNet offset test predictions: {out_npy}")
    print(f"   Stats: mean={resnet_preds.mean():.2f}s, std={resnet_preds.std():.2f}s, min={resnet_preds.min():.2f}s, max={resnet_preds.max():.2f}s")
    print(f"🎉 Total Tabular ResNet Offset Pipeline execution time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
