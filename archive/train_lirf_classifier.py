"""
Train Stage-1 Probabilistic Binary Classifier for Rome Fiumicino (LIRF) Missing Departures:
Predicts whether a flight without AOBT defaulted to schedule (BLOCK == SCHED) or had normal physical taxi.
Saves model into models/lirf_stage1_classifier.pkl.
"""

from pathlib import Path
import sys
import pickle
import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_and_save_lirf_classifier():
    print("=" * 70)
    print("🚀 TRAINING LIRF STAGE-1 PROBABILISTIC CLASSIFIER")
    print("=" * 70)

    # Load all 12 training months
    dfs = [pl.read_parquet(f).filter(pl.col("PHASE_mvt") == "DEP") for f in sorted(RAW_DIR.glob("training_*.parquet"))]
    all_dep = pl.concat(dfs)

    mvt = all_dep["MVT_TIME_UTC_mvt"]
    aobt = all_dep["AOBT_3_flt"]
    sched = all_dep["SCHED_TIME_UTC_mvt"]
    block = all_dep["BLOCK_TIME_UTC_mvt"]

    d_aobt = (mvt - aobt).dt.total_seconds().to_numpy()
    d_sched = (mvt - sched).dt.total_seconds().to_numpy()
    d_block_sched = (block - sched).dt.total_seconds().to_numpy()

    has_aobt = (~np.isnan(d_aobt)) & (d_aobt >= 60) & (d_aobt <= 14400)
    miss_mask = ~has_aobt

    lirf_miss = (all_dep["ADEP_mvt"].to_numpy() == "LIRF") & miss_mask
    sub = all_dep.filter(pl.Series("m", lirf_miss))

    ds = d_sched[lirf_miss]
    dbs = d_block_sched[lirf_miss]
    y_label = (np.abs(dbs) <= 60).astype(np.int32)

    hour = sub["MVT_TIME_UTC_mvt"].dt.hour().to_numpy()
    dow = sub["MVT_TIME_UTC_mvt"].dt.weekday().to_numpy()
    month = sub["MVT_TIME_UTC_mvt"].dt.month().to_numpy()

    ops = sub["AIRCRAFT_OPERATOR_flt"].cast(pl.Utf8).fill_null("UNK").to_numpy()
    actype = sub["AIRCRAFT_TYPE_mvt"].cast(pl.Utf8).fill_null("UNK").to_numpy()
    runway = sub["RUNWAY_mvt"].cast(pl.Utf8).fill_null("UNK").to_numpy()
    stand = sub["STAND_mvt"].cast(pl.Utf8).fill_null("UNK").to_numpy()
    segment = sub["MARKET_SEGMENT_flt"].cast(pl.Utf8).fill_null("UNK").to_numpy()

    X_df = pd.DataFrame({
        "hour": hour,
        "dow": dow,
        "month": month,
        "d_sched": ds,
        "log_d_sched": np.log1p(np.clip(ds, 0, None)),
        "is_delay_gt_2h": (ds > 7200).astype(int),
        "is_delay_gt_5h": (ds > 18000).astype(int),
        "is_delay_gt_12h": (ds > 43200).astype(int),
        "is_delay_gt_20h": (ds > 72000).astype(int),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "operator": ops,
        "actype": actype,
        "runway": runway,
        "stand": stand,
        "market_segment": segment
    })

    cat_cols = ["operator", "actype", "runway", "stand", "market_segment"]
    for c in cat_cols:
        X_df[c] = X_df[c].astype("category")

    clf = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=10,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    clf.fit(X_df, y_label)
    train_probs = clf.predict_proba(X_df)[:, 1]
    auc = roc_auc_score(y_label, train_probs)
    print(f"✅ LIRF Stage-1 Classifier trained on {len(X_df):,} rows | Train ROC-AUC: {auc:.4f}")

    # Save classifier model and categories
    with open(MODELS_DIR / "lirf_stage1_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(MODELS_DIR / "lirf_classifier_cols.pkl", "wb") as f:
        pickle.dump(list(X_df.columns), f)
    with open(MODELS_DIR / "lirf_classifier_cats.pkl", "wb") as f:
        pickle.dump({c: X_df[c].dtype for c in cat_cols}, f)

    print("📁 Saved lirf_stage1_classifier.pkl, lirf_classifier_cols.pkl, lirf_classifier_cats.pkl successfully.")

if __name__ == "__main__":
    train_and_save_lirf_classifier()
