"""
Train Stage-1 Probabilistic Binary Classifier for Rome Fiumicino (LIRF):
Predicts whether a flight defaulted to schedule (BLOCK == SCHED) or had normal physical taxi.
Saves model into models/lirf_stage1_classifier.pkl.
"""

from pathlib import Path
import sys
import pickle
import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_lirf_dual_regime_classifier():
    print("=" * 80)
    print("🚀 TRAINING LIRF STAGE-1 DUAL-REGIME CLASSIFIER (FULL 2025 DATASET)")
    print("=" * 80)

    # 1. Load all 12 training months for LIRF departures
    files = sorted(RAW_DIR.glob("training_*.parquet"))
    dfs = [pl.read_parquet(f).filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF")) for f in files]
    lirf_all = pl.concat(dfs)
    n_total = len(lirf_all)
    print(f"Total LIRF departures loaded: {n_total:,}")

    # 2. Ground Truth Target: Did BLOCK_TIME default to SCHED_TIME (within 15s)?
    mvt = lirf_all["MVT_TIME_UTC_mvt"]
    sched = lirf_all["SCHED_TIME_UTC_mvt"]
    block = lirf_all["BLOCK_TIME_UTC_mvt"]
    aobt = lirf_all["AOBT_3_flt"]
    eobt = lirf_all["EOBT_1_flt"]
    iobt = lirf_all["IOBT_flt"]
    lobt = lirf_all["LOBT_flt"]

    d_bs = (block - sched).dt.total_seconds().to_numpy()
    y_target = (np.abs(d_bs) <= 15).astype(np.int32)
    pos_count = np.sum(y_target)
    print(f"LIRF Schedule-Default Positive Class: {pos_count:,} / {n_total:,} ({pos_count / n_total:.2%})")

    # 3. Engineer High-Discrimination Features
    d_ms = (mvt - sched).dt.total_seconds().to_numpy()
    d_ma = (mvt - aobt).dt.total_seconds().fill_null(1000.0).to_numpy()
    d_as = (aobt - sched).dt.total_seconds().fill_null(0.0).to_numpy()
    d_ae = (aobt - eobt).dt.total_seconds().fill_null(0.0).to_numpy()
    d_es = (eobt - sched).dt.total_seconds().fill_null(0.0).to_numpy()
    d_is = (iobt - sched).dt.total_seconds().fill_null(0.0).to_numpy()
    d_ls = (lobt - sched).dt.total_seconds().fill_null(0.0).to_numpy()

    has_aobt = aobt.is_not_null().cast(pl.Int8).to_numpy()
    has_eobt = eobt.is_not_null().cast(pl.Int8).to_numpy()

    stand_str = lirf_all["STAND_mvt"].cast(pl.Utf8).fill_null("UNK")
    pier = stand_str.str.slice(0, 1)

    hour = mvt.dt.hour().to_numpy()
    dow = mvt.dt.weekday().to_numpy()
    month = mvt.dt.month().to_numpy()

    feat_df = pd.DataFrame({
        "ades": lirf_all["ADES_mvt"].to_pandas().astype("category"),
        "runway": lirf_all["RUNWAY_mvt"].to_pandas().astype("category"),
        "stand": stand_str.to_pandas().astype("category"),
        "pier": pier.to_pandas().astype("category"),
        "actype": lirf_all["AIRCRAFT_TYPE_mvt"].to_pandas().astype("category"),
        "operator": lirf_all["AIRCRAFT_OPERATOR_flt"].to_pandas().astype("category"),
        "month": month,
        "hour": hour,
        "dow": dow,
        "d_ms": d_ms,
        "d_ma": d_ma,
        "d_as": d_as,
        "d_ae": d_ae,
        "d_es": d_es,
        "d_is": d_is,
        "d_ls": d_ls,
        "has_aobt": has_aobt,
        "has_eobt": has_eobt,
        "is_delay_gt_1h": (d_ms > 3600).astype(int),
        "is_delay_gt_3h": (d_ms > 10800).astype(int),
        "is_delay_gt_6h": (d_ms > 21600).astype(int),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
    })

    for c in feat_df.select_dtypes(include=["object", "category"]).columns:
        feat_df[c] = feat_df[c].astype("category")

    # 4. Out-of-Time Holdout Evaluation (Nov & Dec)
    val_mask = (month >= 11)
    X_tr, y_tr = feat_df[~val_mask], y_target[~val_mask]
    X_va, y_va = feat_df[val_mask], y_target[val_mask]

    print(f"\nTraining on Jan-Oct ({len(X_tr):,} rows), Validating on Nov-Dec ({len(X_va):,} rows)...")
    clf = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_tr, y_tr)

    val_probs = clf.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, val_probs)
    brier = brier_score_loss(y_va, val_probs)
    print(f"✅ Nov-Dec Validation ROC-AUC: {auc:.4f} | Brier Score: {brier:.4f}")

    # 5. Fit Full-Year Production Model
    print(f"\nFitting Final Production Model on all {len(feat_df):,} LIRF flights...")
    prod_clf = lgb.LGBMClassifier(
        n_estimators=450,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    prod_clf.fit(feat_df, y_target)

    # Save model
    model_path = MODELS_DIR / "lirf_stage1_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(prod_clf, f)
    print(f"💾 Production LIRF Stage-1 Classifier saved to: {model_path}")

    # Top Feature Importances
    imp = prod_clf.feature_importances_
    cols = feat_df.columns
    print("\nTop 10 Most Important Features:")
    for rank, idx in enumerate(np.argsort(imp)[::-1][:10], 1):
        print(f"  #{rank:2d}: {cols[idx]:20s} (Importance: {imp[idx]})")

    return prod_clf


if __name__ == "__main__":
    train_lirf_dual_regime_classifier()
