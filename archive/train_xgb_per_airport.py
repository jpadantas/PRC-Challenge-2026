"""
Train Per-Airport XGBoost Models (10 specialized models, one per airport).
Each model is trained on ALL 12 months for its specific airport only,
then predictions are combined for the full test set.

Airports: EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM

Strategy: per-airport models capture idiosyncratic taxiway topologies
that a global model consistently underfits. Final ensemble blends
per-airport specialist with global model using confidence weights.
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
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models" / "per_airport"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_DIR = PROJECT_ROOT / "scratch"

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
ALL_12_MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]


def train_per_airport():
    t0 = time.time()
    print("=" * 80)
    print("TRAINING PER-AIRPORT XGBoost SPECIALISTS (10 airports x 12 months)")
    print("=" * 80)

    # 1. Load full year data ONCE
    print(f"[{time.strftime('%H:%M:%S')}] Loading all 12 months of 2025...")
    df_all = features.load_training_data(RAW_DIR, months=ALL_12_MONTHS)
    print(f"Loaded {len(df_all):,} rows in {time.time() - t0:.1f}s")

    # 2. Compute expensive global features once (congestion, surface, sequencing)
    print(f"[{time.strftime('%H:%M:%S')}] Computing global features...")
    df_all = features.compute_congestion_metrics(df_all)
    df_all = features.compute_surface_accumulation_features(df_all)
    df_all = features.compute_apron_contention_features(df_all)
    df_all = features.compute_runway_sequence_features(df_all)
    print(f"Global features done in {time.time() - t0:.1f}s")

    # 3. Global target encodings (trained on all airports — gives global priors)
    dep_all = df_all.filter(pl.col("PHASE_mvt") == "DEP")
    global_encs = features.compute_target_encodings(dep_all)

    results = {}
    for apt in AIRPORTS:
        print(f"\n{'='*60}")
        print(f"  AIRPORT: {apt}")
        print(f"{'='*60}")
        t_apt = time.time()

        # Filter to this airport only
        df_apt = df_all.filter(
            pl.when(pl.col("PHASE_mvt") == "DEP")
            .then(pl.col("ADEP_mvt") == apt)
            .otherwise(pl.col("ADES_mvt") == apt)
        )
        dep_apt = df_apt.filter(pl.col("PHASE_mvt") == "DEP")
        n_dep = len(dep_apt)
        print(f"  Airport-filtered: {n_dep:,} departures")

        if n_dep < 2000:
            print(f"  Skipping {apt}: insufficient data ({n_dep} rows)")
            continue

        # Airport-specific target encodings (fine-tuned on local data)
        # Blend global priors with local priors for rare combos
        local_encs = features.compute_target_encodings(dep_apt)

        # Extract features with local encodings (better fit for airport-specific routes)
        X_pl, _, y = features.extract_features(df_apt, target_encodings=local_encs, attach_weather=True)
        X = X_pl.to_pandas()
        for c in X.select_dtypes(include=["object", "category"]).columns:
            X[c] = X[c].astype("category")

        cols = list(X.columns)
        cat_dtypes = {c: X[c].dtype for c in X.select_dtypes(include="category").columns}

        valid_mask = (y >= 60) & (y <= 150000)
        X_train = X.iloc[valid_mask].reset_index(drop=True)
        y_train = y[valid_mask]
        print(f"  Training matrix: {len(X_train):,} rows, {len(cols)} features")

        # Per-airport XGBoost: more trees allowed since dataset is smaller
        # Deeper trees to capture airport-specific interactions
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=10,          # Deeper: airport-specific quirks need more depth
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.5,
            reg_lambda=2.0,
            min_child_weight=5,
            enable_categorical=True,
            tree_method="hist",
            device="cuda",
            random_state=42,
        )
        model.fit(X_train, y_train)

        preds_train = np.clip(model.predict(X_train), 60, 88500)
        train_rmse = np.sqrt(np.mean((y_train - preds_train) ** 2))
        print(f"  Train RMSE: {train_rmse:.2f}s | Time: {time.time() - t_apt:.1f}s")

        # Save model + metadata
        model.save_model(str(MODELS_DIR / f"xgb_{apt}.json"))
        with open(MODELS_DIR / f"encs_{apt}.pkl", "wb") as f:
            pickle.dump(local_encs, f)
        with open(MODELS_DIR / f"cols_{apt}.pkl", "wb") as f:
            pickle.dump(cols, f)
        with open(MODELS_DIR / f"cat_dtypes_{apt}.pkl", "wb") as f:
            pickle.dump(cat_dtypes, f)

        results[apt] = {"train_rmse": train_rmse, "n_dep": n_dep}

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE. Results by airport:")
    for apt, r in sorted(results.items()):
        print(f"  {apt}: RMSE={r['train_rmse']:.2f}s  n={r['n_dep']:,}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    train_per_airport()
