"""
Disruptive Breakthrough Submission Engine v87:
Dual-Engine Multi-Architecture Ensemble (LightGBM Leaf-Wise + CatBoost Symmetric Trees)
Direct TAXITIME Formulation + Exact LIRF Deterministic Tail Rule.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import time
import json
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from minio import Minio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features_fast import compute_target_encodings_clean, extract_features_seasonal

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

TEAM_NAME = "inspiring-kitten"
VERSION_NUM = 87
SUBMISSION_BUCKET = f"prc-2026-{TEAM_NAME}"
S3_ENDPOINT = "s3.opensky-network.org"

WINTER_MONTHS = ["2025-01", "2025-02", "2025-11", "2025-12"]
SUMMER_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08"]


def load_s3_client() -> Minio:
    with open(CREDENTIALS_FILE, "r") as f:
        creds = json.load(f)
    return Minio(
        S3_ENDPOINT,
        access_key=creds["accessKey"],
        secret_key=creds["secretKey"],
        secure=True,
    )


def load_season_files(months: list[str]) -> pl.DataFrame:
    files = []
    for m in months:
        year, month = int(m[:4]), int(m[5:7])
        if month < 12:
            p = RAW_DIR / f"training_{m}-01_{year}-{month+1:02d}-01.parquet"
        else:
            p = RAW_DIR / f"training_{m}-01_{year+1}-01-01.parquet"
        files.append(p)
    dfs = [pl.read_parquet(f) for f in files]
    return pl.concat(dfs)


def align_categories(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    test_aligned = test_df.copy()
    for col in train_df.select_dtypes(include=["category"]).columns:
        cats = list(train_df[col].cat.categories)
        if col in test_aligned.columns:
            test_aligned[col] = pd.Categorical(test_aligned[col], categories=cats)
    return test_aligned


def train_seasonal_dual_models(season_name: str, months: list[str], df_test: pl.DataFrame, test_season_mask: np.ndarray) -> np.ndarray:
    print("\n" + "=" * 80)
    print(f"❄️ / ☀️ TRAINING {season_name.upper()} DUAL-ENGINE (LightGBM + CatBoost)")
    print("=" * 80)

    # 1. Load seasonal training data
    t0 = time.time()
    df_train = load_season_files(months)
    dep_train = df_train.filter(pl.col("PHASE_mvt") == "DEP")
    print(f"[{season_name.upper()}] Loaded {len(df_train):,} total movements ({len(dep_train):,} departures) in {time.time()-t0:.1f}s")

    # 2. Compute encodings
    encs = compute_target_encodings_clean(dep_train)
    X_train_df, dep_train_out = extract_features_seasonal(df_train, encs)

    y_train_raw = dep_train_out["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    has_aobt_train = X_train_df["has_aobt"].to_numpy() == 1
    da_train = X_train_df["da"].to_numpy()

    clean_mask = (y_train_raw >= 60) & (y_train_raw <= 7200) & has_aobt_train & (da_train >= 60) & (da_train <= 7200)
    X_tr = X_train_df[clean_mask].reset_index(drop=True)
    y_tr = y_train_raw[clean_mask]

    # Test features
    X_test_all, dep_test_out = extract_features_seasonal(df_test, encs)
    X_test_season = X_test_all[test_season_mask].copy().reset_index(drop=True)
    X_test_season = align_categories(X_tr, X_test_season)

    # 1. LightGBM (Leaf-wise trees)
    print(f"[{season_name.upper()}] 1/2 Training LightGBM...")
    t_lgb = time.time()
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.035,
        "num_leaves": 127,
        "max_depth": 10,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "min_child_samples": 40,
        "n_estimators": 1200,
        "random_state": 42,
        "n_jobs": 6,
        "verbose": -1,
    }
    m_lgb = lgb.LGBMRegressor(**lgb_params)
    m_lgb.fit(X_tr, y_tr)
    p_lgb = m_lgb.predict(X_test_season)
    print(f"[{season_name.upper()}] LightGBM finished in {time.time()-t_lgb:.1f}s")

    # 2. CatBoost (Symmetric oblivious trees)
    print(f"[{season_name.upper()}] 2/2 Training CatBoost...")
    t_cb = time.time()
    cat_cols = list(X_tr.select_dtypes(include=["category"]).columns)
    
    # Fill NAs in category cols for CatBoost
    X_tr_cb = X_tr.copy()
    X_te_cb = X_test_season.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(str).fillna("UNK")
        X_te_cb[c] = X_te_cb[c].astype(str).fillna("UNK")

    cb_params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 8,
        "loss_function": "RMSE",
        "l2_leaf_reg": 5.0,
        "random_seed": 42,
        "thread_count": 6,
        "verbose": 200,
    }
    m_cb = CatBoostRegressor(**cb_params)
    m_cb.fit(X_tr_cb, y_tr, cat_features=cat_cols)
    p_cb = m_cb.predict(X_te_cb)
    print(f"[{season_name.upper()}] CatBoost finished in {time.time()-t_cb:.1f}s")

    # Blend: 55% LightGBM + 45% CatBoost
    p_blend = 0.55 * p_lgb + 0.45 * p_cb
    print(f"[{season_name.upper()}] Ensemble Blend: Mean={np.mean(p_blend):.2f}s, Median={np.median(p_blend):.0f}s, Std={np.std(p_blend):.2f}s")
    return p_blend


def validate_and_save(version_num: int, preds: np.ndarray, rank_dep: pl.DataFrame, template: pl.DataFrame) -> Path:
    sub_df = pl.DataFrame({
        "MVT_ID_mvt": rank_dep["MVT_ID_mvt"],
        "TAXITIME_SEC_mvt": preds,
    })
    final_sub = template.select(["MVT_ID_mvt"]).join(sub_df, on="MVT_ID_mvt", how="left")
    assert len(final_sub) == len(template), f"Length mismatch: {len(final_sub)} vs {len(template)}"
    assert final_sub["TAXITIME_SEC_mvt"].null_count() == 0, "Contains nulls"
    assert (final_sub["MVT_ID_mvt"] == template["MVT_ID_mvt"]).all(), "ID alignment mismatch"
    assert (final_sub["TAXITIME_SEC_mvt"] >= 5).all() and (final_sub["TAXITIME_SEC_mvt"] <= 88500).all(), "Values out of bounds"

    out_path = SUBMISSIONS_DIR / f"{TEAM_NAME}_v{version_num}.parquet"
    final_sub.write_parquet(out_path)
    vals = final_sub["TAXITIME_SEC_mvt"].to_numpy()
    print(f"\n✅ Validated & Saved {out_path.name} ({out_path.stat().st_size:,} bytes)")
    print(f"Global Stats: Mean={np.mean(vals):.2f}s, Median={np.median(vals):.0f}s, Min={np.min(vals)}, Max={np.max(vals)}, >7200s={np.sum(vals > 7200)}, >15k={np.sum(vals > 15000)}")
    return out_path


def upload_to_s3(version_num: int, file_path: Path):
    print(f"\n📦 Uploading {file_path.name} to S3 Bucket ({SUBMISSION_BUCKET})...")
    client = load_s3_client()
    obj_name = f"{TEAM_NAME}_v{version_num}.parquet"
    client.fput_object(SUBMISSION_BUCKET, obj_name, str(file_path))
    print(f"🚀 Successfully uploaded {obj_name} to OpenSky S3!")


def main():
    t_global = time.time()
    print("=" * 80)
    print(f"🏆 BUILDING MULTI-ARCHITECTURE ENSEMBLE SUBMISSION v{VERSION_NUM}")
    print("=" * 80)

    # 1. Load ranking test set
    df_test = pl.read_parquet(RAW_DIR / "ranking.parquet")
    template = pl.read_parquet(RAW_DIR / "submitting.parquet")

    rank_dep = df_test.filter(pl.col("PHASE_mvt") == "DEP")
    mvt_test = rank_dep["MVT_TIME_UTC_mvt"]
    jan_mask = (mvt_test.dt.month() == 1).to_numpy()
    jul_mask = (mvt_test.dt.month() == 7).to_numpy()
    n_test = len(rank_dep)

    print(f"Test departures: Total={n_test:,} | Jan (Winter)={np.sum(jan_mask):,} | Jul (Summer)={np.sum(jul_mask):,}")

    # 2. Train Winter Models & Predict Jan 2026 Test
    preds_jan = train_seasonal_dual_models("winter", WINTER_MONTHS, df_test, jan_mask)

    # 3. Train Summer Models & Predict Jul 2026 Test
    preds_jul = train_seasonal_dual_models("summer", SUMMER_MONTHS, df_test, jul_mask)

    # Stitch direct predictions
    preds_direct = np.zeros(n_test, dtype=np.float64)
    preds_direct[jan_mask] = preds_jan
    preds_direct[jul_mask] = preds_jul

    # 4. Dedicated Missing-AOBT Predictions (reuse robust model)
    from src.train_breakthrough_v86 import train_missing_aobt_engine
    sample_files = [
        RAW_DIR / "training_2025-01-01_2025-02-01.parquet",
        RAW_DIR / "training_2025-02-01_2025-03-01.parquet",
        RAW_DIR / "training_2025-07-01_2025-08-01.parquet",
        RAW_DIR / "training_2025-08-01_2025-09-01.parquet",
    ]
    df_sample_train = pl.concat([pl.read_parquet(f) for f in sample_files])
    preds_missing = train_missing_aobt_engine(df_sample_train, df_test)

    # 5. Routing Logic:
    aobt_test = rank_dep["AOBT_3_flt"]
    has_aobt_test = aobt_test.is_not_null().to_numpy()
    adep_test = rank_dep["ADEP_mvt"].to_numpy()
    sched_test = rank_dep["SCHED_TIME_UTC_mvt"]
    d_ms_test = (mvt_test - sched_test).dt.total_seconds().to_numpy()

    final_preds = np.where(has_aobt_test, preds_direct, preds_missing)

    # 6. Exact Deterministic LIRF Extreme Tail Rule
    lirf_extreme_mask = (adep_test == "LIRF") & (~has_aobt_test) & (d_ms_test > 7200)
    print(f"\n🎯 Applying Exact Deterministic LIRF Rule to {np.sum(lirf_extreme_mask)} extreme delay flights...")
    final_preds[lirf_extreme_mask] = d_ms_test[lirf_extreme_mask]

    final_preds_clipped = np.clip(np.round(final_preds), 60, 88500).astype(np.int32)

    # 7. Validate & Save
    sub_file = validate_and_save(VERSION_NUM, final_preds_clipped, rank_dep, template)

    # 8. Upload to S3
    upload_to_s3(VERSION_NUM, sub_file)

    print("\n" + "=" * 80)
    print(f"🎉 ENSEMBLE PIPELINE v{VERSION_NUM} COMPLETED IN {time.time()-t_global:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
