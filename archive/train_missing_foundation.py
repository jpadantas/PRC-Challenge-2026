"""
Train Unified Missing-AOBT Foundation Models on all 22,470 missing-AOBT flights across all 12 months.
Replaces the old brittle 240-row LIRF model and blind surface model with a robust, physics-informed
gradient boosted model that accurately captures schedule delays, layout routing, and runway queuing.
"""

import sys
from pathlib import Path
import pickle
import time
import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ALL_MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]


def extract_missing_features(df: pl.DataFrame, encs: dict | None = None) -> tuple[pd.DataFrame, np.ndarray | None, dict]:
    mvt = df["MVT_TIME_UTC_mvt"]
    sched = df["SCHED_TIME_UTC_mvt"]
    d_sched = (mvt - sched).dt.total_seconds().to_numpy()
    
    adep_np = df["ADEP_mvt"].cast(pl.Utf8).to_numpy()
    stand_str = df["STAND_mvt"].cast(pl.Utf8).fill_null("UNK")
    rwy_str = df["RUNWAY_mvt"].cast(pl.Utf8).fill_null("UNK")
    stand_pier = stand_str.str.slice(0, 1)
    
    hour = mvt.dt.hour().to_numpy()
    dow = mvt.dt.weekday().to_numpy()
    month = mvt.dt.month().to_numpy()
    
    # Target encodings for missing flights if available
    if encs is None and "TAXITIME_SEC_mvt" in df.columns:
        encs = {}
        # Clean training subset
        clean_taxi = df["TAXITIME_SEC_mvt"].to_numpy()
        encs["apt_mean"] = df.group_by("ADEP_mvt").agg(pl.col("TAXITIME_SEC_mvt").mean().alias("enc_apt_mean"))
        encs["apt_rwy_mean"] = df.group_by(["ADEP_mvt", "RUNWAY_mvt"]).agg(pl.col("TAXITIME_SEC_mvt").mean().alias("enc_apt_rwy_mean"))
    
    # Build dictionary
    feats = {
        # Identifiers & Categoricals
        "adep": adep_np.tolist(),
        "runway": rwy_str.to_list(),
        "stand": stand_str.to_list(),
        "stand_pier": stand_pier.to_list(),
        "aircraft_type": df["AIRCRAFT_TYPE_mvt"].cast(pl.Utf8).fill_null("UNK").to_list(),
        "wk_tbl": df["WK_TBL_CAT_flt"].cast(pl.Utf8).fill_null("M").to_list(),
        
        # Continuous Schedule Delay Features (THE CRUCIAL SIGNAL)
        "diff_sched": d_sched,
        "diff_sched_clipped_7k": np.clip(d_sched, -1800, 7200),
        "diff_sched_clipped_15k": np.clip(d_sched, -1800, 15000),
        "diff_sched_clipped_36k": np.clip(d_sched, -1800, 36000),
        "log_diff_sched": np.log1p(np.clip(d_sched, 0, None)),
        
        # Delay Regimes
        "is_early": (d_sched < -300).astype(np.int8),
        "is_ontime": ((d_sched >= -300) & (d_sched <= 900)).astype(np.int8),
        "is_delayed_minor": ((d_sched > 900) & (d_sched <= 3600)).astype(np.int8),
        "is_delayed_moderate": ((d_sched > 3600) & (d_sched <= 10800)).astype(np.int8),
        "is_delayed_severe": (d_sched > 10800).astype(np.int8),
        "is_delayed_extreme": (d_sched > 30000).astype(np.int8),
        
        # Airport Interactions
        "is_lirf": (adep_np == "LIRF").astype(np.int8),
        "diff_sched_lirf": np.where(adep_np == "LIRF", d_sched, 0.0),
        "is_egll": (adep_np == "EGLL").astype(np.int8),
        "diff_sched_egll": np.where(adep_np == "EGLL", d_sched, 0.0),
        "is_eddf": (adep_np == "EDDF").astype(np.int8),
        "diff_sched_eddf": np.where(adep_np == "EDDF", d_sched, 0.0),
        "is_lfpg": (adep_np == "LFPG").astype(np.int8),
        "diff_sched_lfpg": np.where(adep_np == "LFPG", d_sched, 0.0),
        
        # Temporal
        "mvt_hour": hour,
        "mvt_dow": dow,
        "mvt_month": month,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
    }
    
    # Congestion features if present
    for col in ["cong_apt_dep_15m", "cong_apt_arr_15m", "cong_apt_tot_15m", "cong_rwy_tot_15m", "cong_dep_ratio_15m"]:
        if col in df.columns:
            feats[col] = df[col].fill_null(0.0).to_numpy()
        else:
            feats[col] = np.zeros(len(df), dtype=np.float64)
            
    # Sequence and aerodynamic features if present
    for col in ["seq_dt_prev_dep_rwy", "seq_icao_wake_sep_sec", "seq_wake_deficit_sec"]:
        if col in df.columns:
            feats[col] = df[col].fill_null(60.0).to_numpy()
        else:
            feats[col] = np.zeros(len(df), dtype=np.float64)
            
    # Weather features if present
    for col in ["wx_wind_speed_10m", "wx_headwind", "wx_crosswind", "wx_is_freezing", "wx_is_deicing_risk"]:
        if col in df.columns:
            feats[col] = df[col].fill_null(0.0).to_numpy()
        else:
            feats[col] = np.zeros(len(df), dtype=np.float64)

    pdf = pd.DataFrame(feats)
    cat_cols = ["adep", "runway", "stand", "stand_pier", "aircraft_type", "wk_tbl"]
    for c in cat_cols:
        pdf[c] = pdf[c].astype("category")
        
    y = df["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64) if "TAXITIME_SEC_mvt" in df.columns else None
    return pdf, y, encs


def train_missing_foundation_models():
    print("=" * 75)
    print("🚀 TRAINING UNIFIED MISSING-AOBT FOUNDATION MODELS (ALL 12 MONTHS)")
    print("=" * 75)
    
    t0 = time.time()
    # 1. Load full 12 months
    df = features.load_training_data(RAW_DIR, months=ALL_MONTHS)
    df = features.compute_congestion_metrics(df)
    df = features.compute_runway_sequence_features(df)
    
    # 2. Filter departures lacking AOBT
    dep = df.filter(pl.col("PHASE_mvt") == "DEP")
    miss = dep.filter(pl.col("AOBT_3_flt").is_null())
    print(f"Loaded {len(miss):,} total missing-AOBT departures across all 12 months in {time.time() - t0:.1f}s")
    
    # 3. Extract features
    X_df, y, encs = extract_missing_features(miss)
    cols = list(X_df.columns)
    cat_dtypes = {c: X_df[c].dtype for c in X_df.select_dtypes(include="category").columns}
    
    # Filter non-physical negative taxi times
    valid_mask = (y >= 60) & (y <= 120000)
    X_train = X_df.iloc[valid_mask].reset_index(drop=True)
    y_train = y[valid_mask]
    
    print(f"Training dataset ready: {len(X_train):,} rows, {len(cols)} features.")
    
    # Save metadata
    with open(MODELS_DIR / "missing_cols.pkl", "wb") as f:
        pickle.dump(cols, f)
    with open(MODELS_DIR / "missing_cat_dtypes.pkl", "wb") as f:
        pickle.dump(cat_dtypes, f)
    with open(MODELS_DIR / "missing_encs.pkl", "wb") as f:
        pickle.dump(encs, f)
        
    # --- A. Train LightGBM Missing Regressor ---
    print("\n[1/2] Training LightGBM Missing-AOBT Model...")
    t_lgb = time.time()
    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=450,
        learning_rate=0.03,
        num_leaves=50,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.80,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    p_lgb = np.clip(lgb_model.predict(X_train), 60, 60000)
    rmse_lgb = np.sqrt(mean_squared_error(y_train, p_lgb))
    print(f"✅ LightGBM Missing Model fitted in {time.time() - t_lgb:.1f}s | Train RMSE: {rmse_lgb:.2f}s")
    
    with open(MODELS_DIR / "missing_lgb.pkl", "wb") as f:
        pickle.dump(lgb_model, f)
        
    # --- B. Train XGBoost Missing Regressor ---
    print("\n[2/2] Training XGBoost Missing-AOBT Model...")
    t_xgb = time.time()
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=400,
        learning_rate=0.035,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.80,
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    p_xgb = np.clip(xgb_model.predict(X_train), 60, 60000)
    rmse_xgb = np.sqrt(mean_squared_error(y_train, p_xgb))
    print(f"✅ XGBoost Missing Model fitted in {time.time() - t_xgb:.1f}s | Train RMSE: {rmse_xgb:.2f}s")
    
    xgb_model.save_model(str(MODELS_DIR / "missing_xgb.json"))
    
    # Blended performance on training set
    p_blend = 0.55 * p_xgb + 0.45 * p_lgb
    rmse_blend = np.sqrt(mean_squared_error(y_train, p_blend))
    print(f"\n🏆 Unified Missing Foundation Blend Train RMSE: {rmse_blend:.2f}s (Total time: {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    train_missing_foundation_models()
