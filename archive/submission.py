"""
Inference and Submission Pipeline for PRC Data Challenge 2026.

Supports standalone isolated models (LightGBM, XGBoost, CatBoost) as well as Ensembles.
Verifies format against submitting.parquet, and uploads to OpenSky S3 bucket.
"""

import json
import pickle
import time
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
from minio import Minio
import xgboost as xgb
from catboost import CatBoostRegressor

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features
from src.models_catboost import prepare_catboost_df

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

TEAM_NAME = "inspiring-kitten"
SUBMISSION_BUCKET = f"prc-2026-{TEAM_NAME}"
S3_ENDPOINT = "s3.opensky-network.org"


def load_s3_client() -> Minio:
    """Load credentials and create MinIO client."""
    with open(CREDENTIALS_FILE, "r") as f:
        creds = json.load(f)
    return Minio(
        S3_ENDPOINT,
        access_key=creds["accessKey"],
        secret_key=creds["secretKey"],
        secure=True,
    )


def predict_pure_xgboost(dep_df: pl.DataFrame, season: str) -> np.ndarray:
    """Predict departures purely using standalone XGBoost."""
    with open(MODELS_DIR / f"encs_{season}.pkl", "rb") as f:
        encs = pickle.load(f)
    with open(MODELS_DIR / f"cols_{season}.pkl", "rb") as f:
        cols = pickle.load(f)
    with open(MODELS_DIR / f"cat_dtypes_{season}.pkl", "rb") as f:
        cat_dtypes = pickle.load(f)
        
    model = xgb.XGBRegressor()
    model.load_model(str(MODELS_DIR / f"xgb_{season}.json"))
    
    X_pl, _, _ = features.extract_features(dep_df, target_encodings=encs)
    X_pd = X_pl.select(cols).to_pandas()
    
    for c, dtype in cat_dtypes.items():
        if c in X_pd.columns:
            X_pd[c] = pd.Categorical(X_pd[c], categories=dtype.categories, ordered=dtype.ordered)
            
    raw_preds = model.predict(X_pd)
    return np.clip(np.round(raw_preds), 60, 15000).astype(np.int32)


def predict_pure_lightgbm(dep_df: pl.DataFrame, season: str) -> np.ndarray:
    """Predict departures purely using standalone LightGBM."""
    with open(MODELS_DIR / f"encs_{season}.pkl", "rb") as f:
        encs = pickle.load(f)
    with open(MODELS_DIR / f"cols_{season}.pkl", "rb") as f:
        cols = pickle.load(f)
    with open(MODELS_DIR / f"lgb_{season}.pkl", "rb") as f:
        model = pickle.load(f)
        
    X_pl, _, _ = features.extract_features(dep_df, target_encodings=encs)
    X_pd = X_pl.select(cols).to_pandas()
    for c in X_pd.select_dtypes(include="category").columns:
        X_pd[c] = X_pd[c].astype("category")
        
    raw_preds = model.predict(X_pd)
    return np.clip(np.round(raw_preds), 60, 15000).astype(np.int32)


def predict_pure_catboost(dep_df: pl.DataFrame, season: str) -> np.ndarray:
    """Predict departures purely using standalone CatBoost."""
    with open(MODELS_DIR / f"encs_{season}.pkl", "rb") as f:
        encs = pickle.load(f)
    with open(MODELS_DIR / f"cols_{season}.pkl", "rb") as f:
        cols = pickle.load(f)
        
    model = CatBoostRegressor()
    model.load_model(str(MODELS_DIR / f"cb_{season}.cbm"))
    
    X_pl, _, _ = features.extract_features(dep_df, target_encodings=encs)
    X_cb = prepare_catboost_df(X_pl)
    
    raw_preds = model.predict(X_cb[cols])
    return np.clip(np.round(raw_preds), 60, 15000).astype(np.int32)


def generate_submission(version: int = 7, model_type: str = "ensemble", upload: bool = True):
    print(f"\n{'=' * 65}")
    print(f"📦 Generating Submission {TEAM_NAME}_v{version}.parquet [MODEL: {model_type.upper()}]")
    print(f"{'=' * 65}")

    # 1. Load ranking data
    print("Loading test dataset (ranking.parquet)...")
    ranking_df = features.load_ranking_data(RAW_DIR)

    # 2. Compute congestion and runway sequence on the full ranking dataset (ARR + DEP)
    ranking_df = features.compute_congestion_metrics(ranking_df)
    ranking_df = features.compute_runway_sequence_features(ranking_df)

    # 3. Filter departures
    dep_df = ranking_df.filter(pl.col("PHASE_mvt") == "DEP")
    print(f"Total test departures: {len(dep_df):,}")

    jan_dep = dep_df.filter(pl.col("MVT_TIME_UTC_mvt").dt.month() == 1)
    jul_dep = dep_df.filter(pl.col("MVT_TIME_UTC_mvt").dt.month() == 7)
    print(f"  • January departures: {len(jan_dep):,}")
    print(f"  • July departures:    {len(jul_dep):,}")

    if model_type == "xgb":
        print("\nPredicting January 2026 with Pure XGBoost...")
        preds_jan = predict_pure_xgboost(jan_dep, season="winter")
        print("Predicting July 2026 with Pure XGBoost...")
        preds_jul = predict_pure_xgboost(jul_dep, season="summer")
    elif model_type == "lgb":
        print("\nPredicting January 2026 with Pure LightGBM...")
        preds_jan = predict_pure_lightgbm(jan_dep, season="winter")
        print("Predicting July 2026 with Pure LightGBM...")
        preds_jul = predict_pure_lightgbm(jul_dep, season="summer")
    elif model_type == "cb":
        print("\nPredicting January 2026 with Pure CatBoost...")
        preds_jan = predict_pure_catboost(jan_dep, season="winter")
        print("Predicting July 2026 with Pure CatBoost...")
        preds_jul = predict_pure_catboost(jul_dep, season="summer")
    elif model_type == "ensemble":
        from src.ensemble import TwoRegimeEnsemble
        ensemble = TwoRegimeEnsemble()
        print("\nPredicting January 2026 with Two-Regime Ensemble...")
        preds_jan = ensemble.predict_season(jan_dep, season="winter")
        print("Predicting July 2026 with Two-Regime Ensemble...")
        preds_jul = ensemble.predict_season(jul_dep, season="summer")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    jan_result = pl.DataFrame({
        "MVT_ID_mvt": jan_dep["MVT_ID_mvt"],
        "TAXITIME_SEC_mvt": preds_jan,
    })
    jul_result = pl.DataFrame({
        "MVT_ID_mvt": jul_dep["MVT_ID_mvt"],
        "TAXITIME_SEC_mvt": preds_jul,
    })

    # Combine predictions in exact order of submitting.parquet
    template = features.load_submission_template(RAW_DIR)
    combined_preds = pl.concat([jan_result, jul_result])

    final_sub = template.select(["MVT_ID_mvt"]).join(combined_preds, on="MVT_ID_mvt", how="left")

    # Sanity checks
    assert len(final_sub) == len(template), f"Length mismatch: {len(final_sub)} vs {len(template)}"
    assert final_sub["TAXITIME_SEC_mvt"].null_count() == 0, f"Found {final_sub['TAXITIME_SEC_mvt'].null_count()} null predictions!"
    assert (final_sub["MVT_ID_mvt"] == template["MVT_ID_mvt"]).all(), "Order mismatch with submitting template!"

    pred_vals = final_sub["TAXITIME_SEC_mvt"].to_numpy()
    print("\n📊 Prediction Statistics (Global):")
    print(f"  • Total Rows: {len(final_sub):,}")
    print(f"  • Min Taxi Time: {pred_vals.min():.0f}s ({pred_vals.min()/60:.1f}m)")
    print(f"  • Median Taxi Time: {np.median(pred_vals):.0f}s ({np.median(pred_vals)/60:.1f}m)")
    print(f"  • Mean Taxi Time: {pred_vals.mean():.1f}s ({pred_vals.mean()/60:.1f}m)")
    print(f"  • Max Taxi Time: {pred_vals.max():.0f}s ({pred_vals.max()/60:.1f}m)")
    print(f"  • Count > 3,600s: {np.sum(pred_vals > 3600):,}")
    print(f"  • Count > 7,200s: {np.sum(pred_vals > 7200):,}")
    print(f"  • Count > 15,000s: {np.sum(pred_vals > 15000):,}")

    filename = f"{TEAM_NAME}_v{version}.parquet"
    out_path = SUBMISSIONS_DIR / filename
    final_sub.write_parquet(out_path)
    print(f"\n✅ Submission file saved to: {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    if upload:
        print(f"\n☁️ Uploading {filename} to S3 bucket '{SUBMISSION_BUCKET}'...")
        client = load_s3_client()
        client.fput_object(
            bucket_name=SUBMISSION_BUCKET,
            object_name=filename,
            file_path=str(out_path),
        )
        print(f"🚀 Upload complete! Object: s3://{SUBMISSION_BUCKET}/{filename}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PRC Challenge 2026 Submission Pipeline")
    parser.add_argument("version", type=int, nargs="?", default=12, help="Submission version number (default: 12)")
    parser.add_argument("--model", type=str, default="ensemble", choices=["xgb", "lgb", "cb", "ensemble"], help="Model type to generate (default: ensemble)")
    parser.add_argument("--no-upload", action="store_true", help="Do not upload to S3")
    args = parser.parse_args()

    generate_submission(version=args.version, model_type=args.model, upload=not args.no_upload)
