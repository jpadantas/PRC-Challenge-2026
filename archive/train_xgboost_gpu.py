"""
High-Speed XGBoost GPU Seasonal Model for PRC Data Challenge 2026.
Trains on RTX 4060 using CUDA histogram tree method.
Incorporates full arrival taxi congestion feature suite.
Outputs test predictions to scratch/xgboost_arrival_test_preds.npy.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import time
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features_fast import compute_target_encodings_clean, extract_features_seasonal

RAW_DIR = PROJECT_ROOT / "data" / "raw"
SCRATCH_DIR = PROJECT_ROOT / "scratch"
SCRATCH_DIR.mkdir(exist_ok=True)

WINTER_MONTHS = ["2025-01", "2025-02", "2025-11", "2025-12"]
SUMMER_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08"]


def load_season_files(months: list[str]) -> pl.DataFrame:
    files = []
    for m in months:
        year, month = int(m[:4]), int(m[5:7])
        if month < 12:
            p = RAW_DIR / f"training_{m}-01_{year}-{month+1:02d}-01.parquet"
        else:
            p = RAW_DIR / f"training_{m}-01_{year+1}-01-01.parquet"
        files.append(p)
    return pl.concat([pl.read_parquet(f) for f in files])


def align_categories(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    test_aligned = test_df.copy()
    for col in train_df.select_dtypes(include=["category"]).columns:
        cats = list(train_df[col].cat.categories)
        if col in test_aligned.columns:
            test_aligned[col] = pd.Categorical(test_aligned[col], categories=cats)
    return test_aligned


def train_xgb_season(season_name: str, months: list[str], df_test: pl.DataFrame, test_mask: np.ndarray) -> np.ndarray:
    print(f"\n=======================================================")
    print(f"🚀 Training XGBoost GPU ({season_name.upper()}) on RTX 4060")
    print(f"=======================================================")
    t0 = time.time()

    df_train = load_season_files(months)
    dep_train = df_train.filter(pl.col("PHASE_mvt") == "DEP")
    print(f"Loaded {len(dep_train):,} {season_name} training departures in {time.time()-t0:.1f}s")

    encs = compute_target_encodings_clean(dep_train)
    X_train_df, dep_train_out = extract_features_seasonal(df_train, encs)

    y_train_raw = dep_train_out["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    has_aobt_train = X_train_df["has_aobt"].to_numpy() == 1
    da_train = X_train_df["da"].to_numpy()

    clean_mask = (y_train_raw >= 60) & (y_train_raw <= 7200) & has_aobt_train & (da_train >= 120) & (da_train <= 7200)
    X_tr = X_train_df[clean_mask].reset_index(drop=True)
    y_tr = y_train_raw[clean_mask]
    print(f"Clean training samples: {len(X_tr):,} with {X_tr.shape[1]} features")

    X_test_all, _ = extract_features_seasonal(df_test, encs)
    X_test_season = X_test_all[test_mask].copy().reset_index(drop=True)
    X_test_season = align_categories(X_tr, X_test_season)

    cat_cols = list(X_tr.select_dtypes(include=["category"]).columns)
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype("category")
        X_test_season[c] = X_test_season[c].astype("category")

    print(f"Fitting XGBoost GPU regressor (1200 trees, depth 8, lr 0.035)...")
    m_xgb = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.035,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.80,
        min_child_weight=30,
        tree_method="hist",
        device="cuda",
        enable_categorical=True,
        random_state=42,
    )
    t_fit = time.time()
    m_xgb.fit(X_tr, y_tr)
    print(f"Fit completed in {time.time()-t_fit:.1f}s")

    preds = m_xgb.predict(X_test_season)
    print(f"Predicted {len(preds):,} test samples: min={preds.min():.1f}, mean={preds.mean():.1f}, max={preds.max():.1f}")
    return preds


def main():
    t_start = time.time()
    print("=" * 70)
    print("XGBOOST GPU SEASONAL PRE-TRAINING")
    print("=" * 70)

    df_test = pl.read_parquet(RAW_DIR / "ranking.parquet")
    rank_dep = df_test.filter(pl.col("PHASE_mvt") == "DEP")
    mvt_test = rank_dep["MVT_TIME_UTC_mvt"]

    jan_mask = (mvt_test.dt.month() == 1).to_numpy()
    jul_mask = (mvt_test.dt.month() == 7).to_numpy()

    p_jan = train_xgb_season("winter", WINTER_MONTHS, df_test, jan_mask)
    p_jul = train_xgb_season("summer", SUMMER_MONTHS, df_test, jul_mask)

    p_xgb = np.zeros(len(rank_dep), dtype=np.float64)
    p_xgb[jan_mask] = p_jan
    p_xgb[jul_mask] = p_jul

    out_file = SCRATCH_DIR / "xgboost_arrival_test_preds.npy"
    np.save(out_file, p_xgb)
    print(f"\n✅ Successfully saved XGBoost predictions to {out_file} ({out_file.stat().st_size:,} bytes)")
    print(f"Total pipeline time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
