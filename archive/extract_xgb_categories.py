"""
Extract and verify categorical dtypes for existing XGBoost models.
"""

import pickle
from pathlib import Path
import pandas as pd
import polars as pl
import xgboost as xgb

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features
from src.models_xgboost import prepare_xgboost_df

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"


def extract_and_save_cat_dtypes(season: str, months: list[str]):
    print(f"Extracting categorical dtypes for {season} ({months})...")
    df = features.load_training_data(RAW_DIR, months=months)
    df = features.compute_congestion_metrics(df)
    dep_df = df.filter(pl.col("PHASE_mvt") == "DEP")
    encs = features.compute_target_encodings(dep_df)
    X_pl, _, y = features.extract_features(df, target_encodings=encs)
    X = prepare_xgboost_df(X_pl)

    valid_mask = (y >= 0) & (y <= 25000)
    X_train = X.iloc[valid_mask].reset_index(drop=True)

    cat_dtypes = {c: X_train[c].dtype for c in X_train.select_dtypes(include="category").columns}
    out_path = MODELS_DIR / f"cat_dtypes_{season}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(cat_dtypes, f)
    print(f"Saved {len(cat_dtypes)} categorical dtypes to {out_path}")

    # Verify on test sample
    print(f"Verifying XGBoost {season} prediction on ranking.parquet with aligned categories...")
    model = xgb.XGBRegressor()
    model.load_model(str(MODELS_DIR / f"xgb_{season}.json"))

    # Load ranking
    ranking_df = features.load_ranking_data(RAW_DIR)
    ranking_df = features.compute_congestion_metrics(ranking_df)
    test_month = 1 if season == "winter" else 7
    dep_test = ranking_df.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt").dt.month() == test_month))

    with open(MODELS_DIR / f"cols_xgb_{season}.pkl", "rb") as f:
        cols = pickle.load(f)

    X_test_pl, _, _ = features.extract_features(dep_test, target_encodings=encs)
    X_test = X_test_pl.select(cols).to_pandas()

    for c, dtype in cat_dtypes.items():
        if c in X_test.columns:
            X_test[c] = pd.Categorical(X_test[c], categories=dtype.categories, ordered=dtype.ordered)

    preds = model.predict(X_test)
    print(f"✅ SUCCESS! Generated {len(preds):,} predictions for {season}. Sample: {preds[:5]}")


if __name__ == "__main__":
    extract_and_save_cat_dtypes("winter", ["2025-01", "2025-02", "2025-11", "2025-12"])
    extract_and_save_cat_dtypes("summer", ["2025-05", "2025-06", "2025-07", "2025-08"])
