"""
Train Dedicated LightGBM Regressors for LIRF Missing AOBT flights.
Learns to distinguish true ground delays (where TAXITIME ~ MVT - SCHED) from gate delays (where TAXITIME ~ normal surface time).
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import pickle
import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

WINTER_FILES = [
    "training_2025-01-01_2025-02-01.parquet",
    "training_2025-02-01_2025-03-01.parquet",
    "training_2025-11-01_2025-12-01.parquet",
    "training_2025-12-01_2026-01-01.parquet",
]

SUMMER_FILES = [
    "training_2025-05-01_2025-06-01.parquet",
    "training_2025-06-01_2025-07-01.parquet",
    "training_2025-07-01_2025-08-01.parquet",
    "training_2025-08-01_2025-09-01.parquet",
]


def extract_lirf_features(df: pl.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    mvt = df["MVT_TIME_UTC_mvt"]
    sched = df["SCHED_TIME_UTC_mvt"]
    d_sched = (mvt - sched).dt.total_seconds().to_numpy()
    
    feats = {
        "d_sched": d_sched,
        "log_d_sched": np.log1p(np.clip(d_sched, 0, None)),
        "hour": mvt.dt.hour(),
        "dow": mvt.dt.weekday(),
        "month": mvt.dt.month(),
        "sched_hour": sched.dt.hour(),
        "is_morning_push": (sched.dt.hour() >= 6) & (sched.dt.hour() <= 9),
        "is_evening_push": (sched.dt.hour() >= 17) & (sched.dt.hour() <= 20),
        "runway": df["RUNWAY_mvt"].cast(pl.Utf8).fill_null("UNK").to_list(),
        "aircraft_type": df["AIRCRAFT_TYPE_mvt"].cast(pl.Utf8).fill_null("UNK").to_list(),
    }
    
    pdf = pd.DataFrame(feats)
    for c in ["runway", "aircraft_type"]:
        pdf[c] = pdf[c].astype("category")
        
    y = df["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64) if "TAXITIME_SEC_mvt" in df.columns else None
    return pdf, y, d_sched


def train_seasonal_lirf(files: list[str], season: str):
    print(f"\n--- Training Dedicated LIRF Model ({season.upper()}) ---")
    dfs = []
    for f in files:
        path = RAW_DIR / f
        if path.exists():
            df = pl.read_parquet(path).filter(
                (pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF") & (pl.col("AOBT_3_flt").is_null())
            )
            dfs.append(df)
            
    lirf_df = pl.concat(dfs)
    print(f"Loaded {len(lirf_df):,} LIRF missing AOBT records for {season}")
    
    X_df, y, d_sched = extract_lirf_features(lirf_df)
    
    # Train LightGBM model
    model = lgb.LGBMRegressor(
        n_estimators=120,
        learning_rate=0.04,
        num_leaves=25,
        min_child_samples=10,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    model.fit(X_df, y)
    
    preds = np.clip(model.predict(X_df), 600, 60000)
    rmse = np.sqrt(mean_squared_error(y, preds))
    print(f"  • {season.capitalize()} LIRF Regressor Train RMSE: {rmse:.1f}s")
    
    # Save model
    out_path = MODELS_DIR / f"lirf_lgb_{season}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / f"lirf_cols_{season}.pkl", "wb") as f:
        pickle.dump(list(X_df.columns), f)
    print(f"  • Saved LIRF model to: {out_path}")


if __name__ == "__main__":
    train_seasonal_lirf(WINTER_FILES, "winter")
    train_seasonal_lirf(SUMMER_FILES, "summer")
    print("\n✅ Dedicated LIRF models successfully created!")
