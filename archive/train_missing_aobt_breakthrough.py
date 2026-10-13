"""
Missing AOBT Two-Stage Ground-Hold Breakthrough Engine.
Trained on all 12 months of Eurocontrol training data (22,470 missing AOBT flights).
Stage 1: 5-Fold Cross-Validated Classifier (LightGBM + CatBoost GPU) predicting BLOCK == SCHED default.
Stage 2: Regime-Specific Regressors for real taxi vs defaulted ground holds.
Evaluates Out-of-Fold Cross-Validation RMSE and generates test set predictions.
Thermal-safe: limits parallel threads and inserts cooling pauses.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import time
import subprocess
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models" / "missing_aobt"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_DIR = PROJECT_ROOT / "scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

MONTHS_ALL = [f"2025-{m:02d}" for m in range(1, 13)]


def get_gpu_temp() -> tuple[float, float]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        parts = out.strip().split(",")
        return float(parts[0]), float(parts[1])
    except Exception:
        return 0.0, 0.0


def load_all_missing_departures() -> tuple[pl.DataFrame, pl.DataFrame]:
    files = []
    for m in MONTHS_ALL:
        year, month = int(m[:4]), int(m[5:7])
        if month < 12:
            p = RAW_DIR / f"training_{m}-01_{year}-{month+1:02d}-01.parquet"
        else:
            p = RAW_DIR / f"training_{m}-01_{year+1}-01-01.parquet"
        files.append(p)
    df_raw = pl.concat([pl.read_parquet(f) for f in files])
    dep_missing = df_raw.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("AOBT_3_flt").is_null())
    return df_raw, dep_missing


def extract_features(df_dep: pl.DataFrame) -> pd.DataFrame:
    mvt = df_dep["MVT_TIME_UTC_mvt"]
    sched = df_dep["SCHED_TIME_UTC_mvt"]
    iobt = df_dep["IOBT_flt"]
    lobt = df_dep["LOBT_flt"]
    eobt = df_dep["EOBT_1_flt"]

    ds = (mvt - sched).dt.total_seconds().cast(pl.Float64).to_numpy()

    has_eobt = eobt.is_not_null().cast(pl.Int8).to_numpy()
    has_lobt = lobt.is_not_null().cast(pl.Int8).to_numpy()
    has_iobt = iobt.is_not_null().cast(pl.Int8).to_numpy()

    d_me = np.where(eobt.is_not_null().to_numpy(), (mvt - eobt).dt.total_seconds().to_numpy(), np.nan)
    d_ml = np.where(lobt.is_not_null().to_numpy(), (mvt - lobt).dt.total_seconds().to_numpy(), np.nan)
    d_mi = np.where(iobt.is_not_null().to_numpy(), (mvt - iobt).dt.total_seconds().to_numpy(), np.nan)
    d_es = np.where(eobt.is_not_null().to_numpy(), (eobt - sched).dt.total_seconds().to_numpy(), np.nan)
    d_ls = np.where(lobt.is_not_null().to_numpy(), (lobt - sched).dt.total_seconds().to_numpy(), np.nan)
    d_is = np.where(iobt.is_not_null().to_numpy(), (iobt - sched).dt.total_seconds().to_numpy(), np.nan)

    hour = mvt.dt.hour().to_numpy()
    minute = mvt.dt.minute().to_numpy()
    dow = mvt.dt.weekday().to_numpy()
    month = mvt.dt.month().to_numpy()

    stand_str = df_dep["STAND_mvt"].cast(pl.Utf8).fill_null("UNK")
    stand_pier = stand_str.str.slice(0, 1)

    X_df = pd.DataFrame({
        "adep": df_dep["ADEP_mvt"].cast(pl.Utf8).to_pandas().astype("category"),
        "ades": df_dep["ADES_mvt"].cast(pl.Utf8).fill_null("UNK").to_pandas().astype("category"),
        "runway": df_dep["RUNWAY_mvt"].cast(pl.Utf8).fill_null("UNK").to_pandas().astype("category"),
        "stand": stand_str.to_pandas().astype("category"),
        "stand_pier": stand_pier.to_pandas().astype("category"),
        "actype": df_dep["AIRCRAFT_TYPE_mvt"].cast(pl.Utf8).fill_null("UNK").to_pandas().astype("category"),
        "operator": df_dep["AIRCRAFT_OPERATOR_flt"].cast(pl.Utf8).fill_null("UNK").to_pandas().astype("category"),
        "flight_type": df_dep["FLIGHT_TYPE_flt"].cast(pl.Utf8).fill_null("UNK").to_pandas().astype("category"),
        "market_segment": df_dep["MARKET_SEGMENT_flt"].cast(pl.Utf8).fill_null("UNK").to_pandas().astype("category"),
        "ds": ds,
        "has_eobt": has_eobt,
        "has_lobt": has_lobt,
        "has_iobt": has_iobt,
        "d_me": d_me,
        "d_ml": d_ml,
        "d_mi": d_mi,
        "d_es": d_es,
        "d_ls": d_ls,
        "d_is": d_is,
        "hour": hour,
        "minute": minute,
        "dow": dow,
        "month": month,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
    })
    return X_df


def main():
    t_start = time.time()
    temp, pwr = get_gpu_temp()
    print("=" * 80)
    print(f"🚀 MISSING AOBT TWO-STAGE BREAKTHROUGH ENGINE (Initial Temp: {temp:.0f}°C, {pwr:.1f}W)")
    print("=" * 80)

    # 1. Load training data
    df_raw, dep_missing = load_all_missing_departures()
    print(f"Loaded {len(dep_missing):,} missing AOBT departures across all 12 months in {time.time()-t_start:.1f}s")

    block_tr = dep_missing["BLOCK_TIME_UTC_mvt"]
    sched_tr = dep_missing["SCHED_TIME_UTC_mvt"]
    taxi_tr = dep_missing["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()

    d_bs_tr = (block_tr - sched_tr).dt.total_seconds().to_numpy()
    y_is_default = (np.abs(d_bs_tr) <= 15).astype(int)
    print(f"Target distribution: {np.sum(y_is_default):,} defaults ({np.mean(y_is_default):.2%})")

    # Extract training features
    X_tr_df = extract_features(dep_missing)
    cat_cols = list(X_tr_df.select_dtypes(include=["category"]).columns)

    # 2. 5-Fold Cross-Validation for Stage 1 (Classifier) and Stage 2 (Regressors)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_prob_lgb = np.zeros(len(X_tr_df))
    oof_prob_cb = np.zeros(len(X_tr_df))
    oof_preds_final = np.zeros(len(X_tr_df))

    # Also fit regressors for non-default flights
    m_real_non_default = (y_is_default == 0)

    print("\n" + "=" * 80)
    print("🔄 5-FOLD CROSS-VALIDATION OF DUAL-REGIME ARCHITECTURE")
    print("=" * 80)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr_df)):
        t_fold = time.time()
        X_fold_tr, y_fold_tr = X_tr_df.iloc[tr_idx].reset_index(drop=True), y_is_default[tr_idx]
        X_fold_val, y_fold_val = X_tr_df.iloc[val_idx].reset_index(drop=True), y_is_default[val_idx]
        taxi_fold_val = taxi_tr[val_idx]
        ds_fold_val = X_fold_val["ds"].to_numpy()

        # LightGBM Classifier
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            subsample=0.80,
            colsample_bytree=0.80,
            random_state=42 + fold,
            n_jobs=4,
            verbose=-1,
        )
        clf_lgb.fit(X_fold_tr, y_fold_tr)
        p_val_lgb = clf_lgb.predict_proba(X_fold_val)[:, 1]
        oof_prob_lgb[val_idx] = p_val_lgb

        # CatBoost GPU Classifier
        X_tr_cb = X_fold_tr.copy()
        X_val_cb = X_fold_val.copy()
        for col in cat_cols:
            X_tr_cb[col] = X_tr_cb[col].astype(str).fillna("UNK")
            X_val_cb[col] = X_val_cb[col].astype(str).fillna("UNK")

        clf_cb = CatBoostClassifier(
            iterations=400,
            learning_rate=0.04,
            depth=6,
            cat_features=cat_cols,
            task_type="GPU",
            verbose=0,
            random_seed=42 + fold,
        )
        clf_cb.fit(X_tr_cb, y_fold_tr)
        p_val_cb = clf_cb.predict_proba(X_val_cb)[:, 1]
        oof_prob_cb[val_idx] = p_val_cb

        # Blend Classifiers
        p_val_blend = 0.50 * p_val_lgb + 0.50 * p_val_cb

        # Fit Regressor on non-default training flights
        tr_non_def = tr_idx[y_is_default[tr_idx] == 0]
        X_reg_tr = X_tr_df.iloc[tr_non_def].reset_index(drop=True)
        y_reg_tr = taxi_tr[tr_non_def]

        reg_lgb = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            subsample=0.80,
            colsample_bytree=0.80,
            random_state=42 + fold,
            n_jobs=4,
            verbose=-1,
        )
        reg_lgb.fit(X_reg_tr, y_reg_tr)
        pred_reg_val = reg_lgb.predict(X_fold_val)

        # Dual-regime logic:
        # If p_val_blend >= 0.40, predict ds, else predict reg_lgb
        pred_fold = np.where(p_val_blend >= 0.40, np.maximum(ds_fold_val, pred_reg_val), pred_reg_val)
        oof_preds_final[val_idx] = pred_fold

        auc_f = roc_auc_score(y_fold_val, p_val_blend)
        acc_f = accuracy_score(y_fold_val, (p_val_blend >= 0.50).astype(int))
        rmse_f = np.sqrt(np.mean((pred_fold - taxi_fold_val)**2))
        rmse_baseline_f = np.sqrt(np.mean((pred_reg_val - taxi_fold_val)**2))

        temp, _ = get_gpu_temp()
        t_el = time.time() - t_fold
        print(f"Fold {fold+1}/5 ({t_el:4.1f}s | {temp:.0f}°C): AUC={auc_f:.4f} | Acc={acc_f:.4f} | RMSE={rmse_f:.1f}s (vs baseline {rmse_baseline_f:.1f}s | Drop: {rmse_baseline_f - rmse_f:+.1f}s)")
        time.sleep(2.0)

    # Full OOF Summary
    oof_prob_blend = 0.50 * oof_prob_lgb + 0.50 * oof_prob_cb
    total_auc = roc_auc_score(y_is_default, oof_prob_blend)
    total_acc = accuracy_score(y_is_default, (oof_prob_blend >= 0.50).astype(int))
    total_rmse = np.sqrt(np.mean((oof_preds_final - taxi_tr)**2))

    # Baseline comparison: median per airport
    adep_all = X_tr_df["adep"].to_numpy()
    apt_meds = {a: np.median(taxi_tr[adep_all == a]) for a in np.unique(adep_all)}
    pred_simple_base = np.array([apt_meds[a] for a in adep_all])
    baseline_rmse = np.sqrt(np.mean((pred_simple_base - taxi_tr)**2))

    print("\n" + "=" * 80)
    print("🏆 OUT-OF-FOLD CROSS-VALIDATION SUMMARY ACROSS ALL 22,470 FLIGHTS:")
    print("=" * 80)
    print(f"  • Stage 1 Overall ROC AUC:  {total_auc:.4f}")
    print(f"  • Stage 1 Overall Accuracy: {total_acc:.4f}")
    print(f"  • Baseline RMSE on Missing: {baseline_rmse:.2f}s")
    print(f"  • Dual-Regime Model RMSE:   {total_rmse:.2f}s")
    print(f"  • Total Out-of-Fold Drop:   {baseline_rmse - total_rmse:+.2f}s RMSE reduction!")
    print("=" * 80)

    # 3. Fit Final Production Models on 100% of Data and Predict on Test Set
    print("\n📦 Fitting Final Production Models on 100% of Data...")
    clf_lgb_full = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4, verbose=-1)
    clf_lgb_full.fit(X_tr_df, y_is_default)

    X_tr_cb_full = X_tr_df.copy()
    for col in cat_cols:
        X_tr_cb_full[col] = X_tr_cb_full[col].astype(str).fillna("UNK")
    clf_cb_full = CatBoostClassifier(iterations=500, learning_rate=0.04, depth=6, cat_features=cat_cols, task_type="GPU", verbose=0, random_seed=42)
    clf_cb_full.fit(X_tr_cb_full, y_is_default)

    # Regressor on real non-default flights
    reg_lgb_full = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4, verbose=-1)
    reg_lgb_full.fit(X_tr_df.iloc[m_real_non_default].reset_index(drop=True), taxi_tr[m_real_non_default])

    # 4. Predict on Test Set
    rank = pl.read_parquet(RAW_DIR / "ranking.parquet")
    rank_dep = rank.filter(pl.col("PHASE_mvt") == "DEP")
    missing_test_mask = rank_dep["AOBT_3_flt"].is_null().to_numpy()
    dep_test_missing = rank_dep.filter(missing_test_mask)

    X_te_df = extract_features(dep_test_missing)
    # Align categories
    for col in cat_cols:
        cats = list(X_tr_df[col].cat.categories)
        X_te_df[col] = pd.Categorical(X_te_df[col], categories=cats)

    X_te_cb = X_te_df.copy()
    for col in cat_cols:
        X_te_cb[col] = X_te_cb[col].astype(str).fillna("UNK")

    p_test_prob_lgb = clf_lgb_full.predict_proba(X_te_df)[:, 1]
    p_test_prob_cb = clf_cb_full.predict_proba(X_te_cb)[:, 1]
    p_test_prob_blend = 0.50 * p_test_prob_lgb + 0.50 * p_test_prob_cb

    p_test_reg = reg_lgb_full.predict(X_te_df)
    ds_test = X_te_df["ds"].to_numpy()

    # Final dual-regime prediction for missing flights:
    p_test_missing_pred = np.where(p_test_prob_blend >= 0.40, np.maximum(ds_test, p_test_reg), p_test_reg)

    # Build full test array matching ranking.parquet
    v99 = pl.read_parquet(PROJECT_ROOT / "data" / "submissions" / "inspiring-kitten_v99.parquet")["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    p_full_test = v99.copy()
    p_full_test[missing_test_mask] = p_test_missing_pred

    out_npy = SCRATCH_DIR / "missing_aobt_breakthrough_test_preds.npy"
    np.save(out_npy, p_full_test)
    print(f"\n💾 Saved full test predictions to {out_npy}")
    print(f"   Missing flights modified: {np.sum(missing_test_mask):,} | Defaults predicted: {np.sum(p_test_prob_blend >= 0.40):,}")
    print(f"   Total Runtime: {time.time()-t_start:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
