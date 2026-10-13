"""
Train Full-Year PyTorch Tabular Neural Network on all 12 Months of 2025 (2,085,000+ Movements)
with Entity Embeddings, True Surface Traffic (N_active), and GPU CUDA Acceleration.
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
from src.models_neural import train_neural_foundation, predict_neural_foundation

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
ALL_12_MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]


def train_neural():
    print("=" * 80)
    print("🚀 TRAINING PYTORCH TABULAR NEURAL NETWORK (2.08M MOVEMENTS, CUDA ACCELERATED)")
    print("=" * 80)

    t0 = time.time()
    # 1. Load full year data
    print(f"[{time.strftime('%H:%M:%S')}] Loading all 12 months of 2025...")
    df = features.load_training_data(RAW_DIR, months=ALL_12_MONTHS)
    print(f"Loaded {len(df):,} total rows in {time.time() - t0:.1f}s")

    # 2. Compute Congestion, Surface Accumulation & Dynamic Runway Sequencing
    print(f"[{time.strftime('%H:%M:%S')}] Computing congestion and runway sequence features...")
    df = features.compute_congestion_metrics(df)
    df = features.compute_surface_accumulation_features(df)
    df = features.compute_runway_sequence_features(df)

    # 3. Target encodings on departures
    dep_df = df.filter(pl.col("PHASE_mvt") == "DEP")
    print(f"[{time.strftime('%H:%M:%S')}] Computing target encodings across {len(dep_df):,} departures...")
    encs = features.compute_target_encodings(dep_df)

    # 4. Feature extraction
    print(f"[{time.strftime('%H:%M:%S')}] Extracting all features for neural network...")
    X_pl, _, y = features.extract_features(df, target_encodings=encs, attach_weather=True)

    X_pd = X_pl.to_pandas()
    print(f"Feature matrix ready: {len(X_pd):,} departure rows, {X_pd.shape[1]} features.")

    # 5. Train Neural Network on GPU CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[{time.strftime('%H:%M:%S')}] Launching PyTorch Tabular Neural Training on {device.upper()}...")
    model, save_dict = train_neural_foundation(
        X_pd, y, epochs=12, batch_size=4096, lr=2.5e-3, device=device
    )

    # Save target encodings for neural model
    with open(MODELS_DIR / "encs_neural_fullyear.pkl", "wb") as f:
        pickle.dump(encs, f)

    # 6. Generate Test Predictions immediately
    print(f"\n[{time.strftime('%H:%M:%S')}] Generating Test Predictions on ranking.parquet...")
    rank_df = pl.read_parquet(RAW_DIR / "ranking.parquet")
    rank_df = features.compute_congestion_metrics(rank_df)
    rank_df = features.compute_surface_accumulation_features(rank_df)
    rank_df = features.compute_runway_sequence_features(rank_df)

    dep_rank = rank_df.filter(pl.col("PHASE_mvt") == "DEP")
    X_test_pl, _, _ = features.extract_features(dep_rank, target_encodings=encs, attach_weather=True)
    X_test_pd = X_test_pl.to_pandas()

    neural_preds = predict_neural_foundation(save_dict, X_test_pd, device=device)
    out_npy = PROJECT_ROOT / "scratch" / "neural_deep_test_preds.npy"
    np.save(out_npy, neural_preds)
    print(f"💾 Saved Neural Deep test predictions: {out_npy} (mean={neural_preds.mean():.1f}s, max={neural_preds.max():.0f}s)")
    print(f"🎉 Total Neural Pipeline execution time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    train_neural()
