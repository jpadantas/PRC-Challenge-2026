"""
Production Script: Train Dual-Engine Airport Specialists (LightGBM + CatBoost GPU)
Across All 10 European Hubs for Winter and Summer Seasons.
Saves test set predictions to scratch/dual_engine_specialists_test_preds.npy.
Thermal-safe: Monitors GPU temperatures and inserts cooling pauses.
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
import lightgbm as lgb
from catboost import CatBoostRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"
SCRATCH_DIR = PROJECT_ROOT / "scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

from src.features_fast import compute_target_encodings_clean, extract_features_seasonal

WINTER_MONTHS = ["2025-01", "2025-02", "2025-11", "2025-12"]
SUMMER_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08"]
ALL_HUBS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


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


def train_season_dual_engine(season_name: str, months: list[str], df_test: pl.DataFrame, test_mask: np.ndarray) -> np.ndarray:
    print("\n" + "=" * 80)
    print(f"✈️ TRAINING DUAL-ENGINE SPECIALISTS (LGB + CB GPU) FOR {season_name.upper()} SEASON")
    print("=" * 80)

    t0 = time.time()
    df_train = load_season_files(months)
    dep_train = df_train.filter(pl.col("PHASE_mvt") == "DEP")
    print(f"[{season_name.upper()}] Loaded {len(df_train):,} movements ({len(dep_train):,} departures) in {time.time()-t0:.1f}s")

    # Feature extraction
    t_feat = time.time()
    encs = compute_target_encodings_clean(dep_train)
    X_train_df, dep_train_out = extract_features_seasonal(df_train, encs)
    y_train_raw = dep_train_out["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    has_aobt_tr = X_train_df["has_aobt"].to_numpy() == 1
    da_tr = X_train_df["da"].to_numpy()

    clean_mask = (y_train_raw >= 60) & (y_train_raw <= 7200) & has_aobt_tr & (da_tr >= 120) & (da_tr <= 7200)
    X_tr = X_train_df[clean_mask].reset_index(drop=True)
    y_tr = y_train_raw[clean_mask]
    adep_tr = dep_train_out.filter(clean_mask)["ADEP_mvt"].to_numpy()
    print(f"[{season_name.upper()}] Features extracted in {time.time()-t_feat:.1f}s | Clean training rows: {len(X_tr):,}")

    # Test features
    X_test_all, dep_test_out = extract_features_seasonal(df_test, encs)
    X_test_season = X_test_all[test_mask].copy().reset_index(drop=True)
    X_test_season = align_categories(X_tr, X_test_season)
    adep_test_season = dep_test_out.filter(test_mask)["ADEP_mvt"].to_numpy()

    cat_cols = list(X_tr.select_dtypes(include=["category"]).columns)

    # CatBoost copies with string categoricals
    X_tr_cb = X_tr.copy()
    X_te_cb = X_test_season.copy()
    for col in cat_cols:
        X_tr_cb[col] = X_tr_cb[col].astype(str).fillna("UNK")
        X_te_cb[col] = X_te_cb[col].astype(str).fillna("UNK")

    p_season = np.zeros(len(X_test_season), dtype=np.float64)

    for apt in ALL_HUBS:
        t_apt = time.time()
        m_tr = (adep_tr == apt)
        m_te = (adep_test_season == apt)
        n_tr = np.sum(m_tr)
        n_te = np.sum(m_te)

        if n_te == 0:
            continue

        # Slices
        X_tr_apt_lgb = X_tr[m_tr].reset_index(drop=True)
        y_tr_apt = y_tr[m_tr]
        X_te_apt_lgb = X_test_season[m_te].reset_index(drop=True)

        X_tr_apt_cb = X_tr_cb[m_tr].reset_index(drop=True)
        X_te_apt_cb = X_te_cb[m_te].reset_index(drop=True)

        # 1. LightGBM Specialist
        m_lgb = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            boosting_type="gbdt",
            learning_rate=0.035,
            num_leaves=63,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.80,
            min_child_samples=25,
            n_estimators=800,
            random_state=42,
            n_jobs=4,
            verbose=-1,
        )
        m_lgb.fit(X_tr_apt_lgb, y_tr_apt)
        pred_lgb = m_lgb.predict(X_te_apt_lgb)

        # 2. CatBoost GPU Specialist
        m_cb = CatBoostRegressor(
            iterations=700,
            learning_rate=0.035,
            depth=6,
            cat_features=cat_cols,
            task_type="GPU",
            verbose=0,
            random_seed=42,
        )
        m_cb.fit(X_tr_apt_cb, y_tr_apt)
        pred_cb = m_cb.predict(X_te_apt_cb)

        # 3. 50/50 Dual-Engine Specialist Blend
        pred_dual = 0.50 * pred_lgb + 0.50 * pred_cb
        p_season[m_te] = pred_dual

        temp, pwr = get_gpu_temp()
        t_el = time.time() - t_apt
        print(f"   • [{apt:4s}] {n_tr:,} train -> {n_te:,} test in {t_el:4.1f}s ({temp:.0f}°C, {pwr:.1f}W) | mean={np.mean(pred_dual):.1f}s")

        # 2-second thermal dissipation pause
        time.sleep(2.0)

    return p_season


def main():
    t_start = time.time()
    temp_init, pwr_init = get_gpu_temp()
    print("=" * 80)
    print(f"🚀 TRAINING PRODUCTION DUAL-ENGINE SPECIALISTS (Temp: {temp_init:.0f}°C, {pwr_init:.1f}W)")
    print("=" * 80)

    df_test = pl.read_parquet(RAW_DIR / "ranking.parquet")
    rank_dep = df_test.filter(pl.col("PHASE_mvt") == "DEP")
    mvt_test = rank_dep["MVT_TIME_UTC_mvt"]

    jan_mask = (mvt_test.dt.month() == 1).to_numpy()
    jul_mask = (mvt_test.dt.month() == 7).to_numpy()

    p_jan = train_season_dual_engine("winter", WINTER_MONTHS, df_test, jan_mask)
    p_jul = train_season_dual_engine("summer", SUMMER_MONTHS, df_test, jul_mask)

    p_dual_full = np.zeros(len(rank_dep), dtype=np.float64)
    p_dual_full[jan_mask] = p_jan
    p_dual_full[jul_mask] = p_jul

    out_npy = SCRATCH_DIR / "dual_engine_specialists_test_preds.npy"
    np.save(out_npy, p_dual_full)
    print("\n" + "=" * 80)
    print(f"💾 Successfully saved dual-engine specialist predictions to {out_npy}!")
    print(f"   Shape: {p_dual_full.shape}, Mean: {p_dual_full.mean():.2f}s, Std: {p_dual_full.std():.2f}s")
    temp_end, _ = get_gpu_temp()
    print(f"   Total Runtime: {time.time()-t_start:.1f}s | Final GPU Temp: {temp_end:.0f}°C")
    print("=" * 80)


if __name__ == "__main__":
    main()
