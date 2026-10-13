"""
Meta-Stacker & Candidate Generator for v22:
Combines Deep XGBoost (178s RMSE), Deep CatBoost (237s RMSE), Deep LightGBM (208s RMSE),
and Champion Anchor v21 (308.64s) with Tail-Shield Physical Horizon Protection.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import time
import pickle
import json
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from minio import Minio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src import features

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
TEAM_NAME = "inspiring-kitten"
SUBMISSION_BUCKET = f"prc-2026-{TEAM_NAME}"
S3_ENDPOINT = "s3.opensky-network.org"


def generate_v22_candidate():
    print("=" * 80)
    print("👑 GENERATING SUBMISSION v22: 4-FRAMEWORK DEEP FOUNDATION META-STACK")
    print("=" * 80)
    t0 = time.time()

    # 1. Load test data & template
    print("Loading test data (ranking.parquet)...")
    ranking_df = pl.read_parquet(RAW_DIR / "ranking.parquet")
    ranking_df = features.compute_congestion_metrics(ranking_df)
    ranking_df = features.compute_surface_accumulation_features(ranking_df)
    ranking_df = features.compute_runway_sequence_features(ranking_df)

    dep_df = ranking_df.filter(pl.col("PHASE_mvt") == "DEP")
    template = pl.read_parquet(RAW_DIR / "submitting.parquet")
    n_test = len(dep_df)
    print(f"Loaded {n_test:,} test departures in {time.time() - t0:.1f}s")

    # 2. Model 1: Deep CatBoost Full-Year Inference (Train RMSE: 235.13s)
    print("\n[1/5] Running Deep CatBoost Full-Year Inference (Train RMSE: 235.13s)...")
    t_cb = time.time()
    with open(MODELS_DIR / "encs_cb_fullyear_deep.pkl", "rb") as f:
        encs_cb = pickle.load(f)
    with open(MODELS_DIR / "cols_cb_fullyear_deep.pkl", "rb") as f:
        cols_cb = pickle.load(f)

    cb_deep = CatBoostRegressor()
    cb_deep.load_model(str(MODELS_DIR / "cb_fullyear_deep.cbm"))

    from src.models_catboost import prepare_catboost_df
    X_pl_cb, _, _ = features.extract_features(dep_df, target_encodings=encs_cb, attach_weather=True)
    X_cb = prepare_catboost_df(X_pl_cb)
    for c in cols_cb:
        if c in X_cb.columns and str(X_cb[c].dtype) in ["category", "object"]:
            X_cb[c] = X_cb[c].astype(str).fillna("UNK")
        elif c not in X_cb.columns:
            X_cb[c] = 0.0

    preds_cb_deep = np.clip(cb_deep.predict(X_cb[cols_cb]), 60, 88500)
    print(f"  • Deep CatBoost inference done in {time.time() - t_cb:.1f}s (mean={preds_cb_deep.mean():.1f}s, max={preds_cb_deep.max():.0f}s)")
    np.save(PROJECT_ROOT / "scratch" / "cb_deep_test_preds.npy", preds_cb_deep)

    # 3. Model 2: Deep LightGBM Full-Year Inference (Train RMSE: 206.68s)
    print("\n[2/5] Running Deep LightGBM Full-Year Inference (Train RMSE: 206.68s)...")
    t_lgb = time.time()
    with open(MODELS_DIR / "encs_lgb_fullyear_deep.pkl", "rb") as f:
        encs_lgb = pickle.load(f)
    with open(MODELS_DIR / "cols_lgb_fullyear_deep.pkl", "rb") as f:
        cols_lgb = pickle.load(f)
    with open(MODELS_DIR / "lgb_fullyear_deep.pkl", "rb") as f:
        lgb_deep = pickle.load(f)

    X_pl_lgb, _, _ = features.extract_features(dep_df, target_encodings=encs_lgb, attach_weather=True)
    X_lgb = X_pl_lgb.to_pandas()
    for c in cols_lgb:
        if c in X_lgb.columns and str(X_lgb[c].dtype) in ["object", "category"]:
            X_lgb[c] = X_lgb[c].astype("category")
        elif c not in X_lgb.columns:
            X_lgb[c] = 0.0

    preds_lgb_deep = np.clip(lgb_deep.predict(X_lgb[cols_lgb]), 60, 88500)
    print(f"  • Deep LightGBM inference done in {time.time() - t_lgb:.1f}s (mean={preds_lgb_deep.mean():.1f}s, max={preds_lgb_deep.max():.0f}s)")
    np.save(PROJECT_ROOT / "scratch" / "lgb_deep_test_preds.npy", preds_lgb_deep)

    # 4. Model 3: Deep XGBoost Full-Year Inference (Train RMSE: 173.58s)
    print("\n[3/5] Running Deep XGBoost Full-Year Inference (Train RMSE: 173.58s)...")
    t_xgb = time.time()
    with open(MODELS_DIR / "encs_xgb_fullyear_deep.pkl", "rb") as f:
        encs_xgb = pickle.load(f)
    with open(MODELS_DIR / "cols_xgb_fullyear_deep.pkl", "rb") as f:
        cols_xgb = pickle.load(f)
    with open(MODELS_DIR / "cat_dtypes_xgb_fullyear_deep.pkl", "rb") as f:
        cats_xgb = pickle.load(f)

    xgb_deep = xgb.XGBRegressor()
    xgb_deep.load_model(str(MODELS_DIR / "xgb_fullyear_deep.json"))

    X_pl_xgb, _, _ = features.extract_features(dep_df, target_encodings=encs_xgb, attach_weather=True)
    X_xgb = X_pl_xgb.to_pandas()
    for c in cols_xgb:
        if c in cats_xgb:
            dt = cats_xgb[c]
            X_xgb[c] = pd.Categorical(X_xgb[c], categories=dt.categories, ordered=dt.ordered)
        elif c not in X_xgb.columns:
            X_xgb[c] = 0.0

    preds_xgb_deep = np.clip(xgb_deep.predict(X_xgb[cols_xgb]), 60, 88500)
    print(f"  • Deep XGBoost inference done in {time.time() - t_xgb:.1f}s (mean={preds_xgb_deep.mean():.1f}s, max={preds_xgb_deep.max():.0f}s)")
    np.save(PROJECT_ROOT / "scratch" / "xgb_deep_test_preds.npy", preds_xgb_deep)

    # 5. Load Anchor Models (v18 and v21)
    print("\n[4/5] Loading Champion Anchor Predictions (v18 & v21)...")
    p18 = pl.read_parquet(SUBMISSIONS_DIR / f"{TEAM_NAME}_v18.parquet")["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    p21 = pl.read_parquet(SUBMISSIONS_DIR / f"{TEAM_NAME}_v21.parquet")["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)

    # 6. Load PyTorch Neural Deep Predictions
    neural_npy = PROJECT_ROOT / "scratch" / "neural_deep_test_preds.npy"
    if neural_npy.exists():
        preds_neural = np.load(neural_npy).astype(np.float64)
        print(f"  • Loaded PyTorch Neural Deep predictions (mean={preds_neural.mean():.1f}s)")
    else:
        preds_neural = p21.copy()

    # 7. Convex Meta-Stacking with Tail Shield
    print("\n[5/5] Executing Convex Meta-Stacking & Tail Shield...")
    high_mask = p18 > 15000
    n_high = np.sum(high_mask)
    print(f"  • Extreme Radar Horizon: {n_high} flights in LIRF with p18 > 15,000s.")
    print(f"  • Applying Tail Shield: preserving 100% v18 on extreme horizon to guarantee unclipped tail.")

    # Multi-framework weights on normal flights (<= 15,000s)
    # v24: GBDT Meta-Stack with True Surface Features (40% v21 + 25% XGB Deep + 20% CB Deep + 15% LGB Deep)
    w_21 = 0.40
    w_xgb = 0.25
    w_cb = 0.20
    w_lgb = 0.15

    print(f"  • Meta-stacking weights (v24): v21={w_21:.2f}, XGB Deep={w_xgb:.2f}, CatBoost Deep={w_cb:.2f}, LightGBM Deep={w_lgb:.2f}")
    preds_v24_float = w_21 * p21 + w_xgb * preds_xgb_deep + w_cb * preds_cb_deep + w_lgb * preds_lgb_deep
    preds_v24_float[high_mask] = p18[high_mask]
    final_v24 = np.clip(np.round(preds_v24_float), 5, 88500).astype(np.int32)

    # Build v24 submission
    sub_df_24 = pl.DataFrame({
        "MVT_ID_mvt": dep_df["MVT_ID_mvt"],
        "TAXITIME_SEC_mvt": final_v24,
    })
    final_sub_24 = template.select(["MVT_ID_mvt"]).join(sub_df_24, on="MVT_ID_mvt", how="left")

    assert len(final_sub_24) == len(template)
    assert final_sub_24["TAXITIME_SEC_mvt"].null_count() == 0
    assert (final_sub_24["MVT_ID_mvt"] == template["MVT_ID_mvt"]).all()
    assert (final_sub_24["TAXITIME_SEC_mvt"] >= 5).all() and (final_sub_24["TAXITIME_SEC_mvt"] <= 88500).all()

    filename_24 = f"{TEAM_NAME}_v24.parquet"
    out_path_24 = SUBMISSIONS_DIR / filename_24
    final_sub_24.write_parquet(out_path_24)
    print(f"\n💾 Saved v24 Candidate: {out_path_24} ({out_path_24.stat().st_size / 1e6:.2f} MB)")

    # v25: Multi-Paradigm Ensemble (85% v24 + 15% PyTorch Neural Deep)
    w_g = 0.85
    w_n = 0.15
    print(f"\n  • Multi-Paradigm Ensemble (v25): {w_g*100:.0f}% GBDT Meta-Stack (v24) + {w_n*100:.0f}% PyTorch Neural Deep")
    preds_v25_float = w_g * preds_v24_float + w_n * preds_neural
    preds_v25_float[high_mask] = p18[high_mask]
    final_v25 = np.clip(np.round(preds_v25_float), 5, 88500).astype(np.int32)

    sub_df_25 = pl.DataFrame({
        "MVT_ID_mvt": dep_df["MVT_ID_mvt"],
        "TAXITIME_SEC_mvt": final_v25,
    })
    final_sub_25 = template.select(["MVT_ID_mvt"]).join(sub_df_25, on="MVT_ID_mvt", how="left")

    assert len(final_sub_25) == len(template)
    assert final_sub_25["TAXITIME_SEC_mvt"].null_count() == 0
    assert (final_sub_25["MVT_ID_mvt"] == template["MVT_ID_mvt"]).all()
    assert (final_sub_25["TAXITIME_SEC_mvt"] >= 5).all() and (final_sub_25["TAXITIME_SEC_mvt"] <= 88500).all()

    filename_25 = f"{TEAM_NAME}_v25.parquet"
    out_path_25 = SUBMISSIONS_DIR / filename_25
    final_sub_25.write_parquet(out_path_25)
    print(f"💾 Saved v25 Candidate: {out_path_25} ({out_path_25.stat().st_size / 1e6:.2f} MB)")

    vals_24 = final_sub_24["TAXITIME_SEC_mvt"].to_numpy()
    vals_25 = final_sub_25["TAXITIME_SEC_mvt"].to_numpy()
    print("\n" + "=" * 80)
    print("📊 SUBMISSION CANDIDATE AUDIT METRICS:")
    print(f"  • v24 (GBDT Meta-Stack): Mean={vals_24.mean():.1f}s | Median={np.median(vals_24):.0f}s | Corr vs v21={np.corrcoef(vals_24, p21)[0, 1]:.6f}")
    print(f"  • v25 (Multi-Paradigm):  Mean={vals_25.mean():.1f}s | Median={np.median(vals_25):.0f}s | Corr vs v21={np.corrcoef(vals_25, p21)[0, 1]:.6f}")
    print("=" * 80)

    return out_path_24, out_path_25


if __name__ == "__main__":
    generate_v22_candidate()

