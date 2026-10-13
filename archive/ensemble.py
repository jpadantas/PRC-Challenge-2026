"""
Two-Regime Seasonal Ensemble for PRC Data Challenge 2026.

Architecture:
  • Regime 1 (Valid AOBT, ~98.5% of flights):
      Calibrated 3-Way Ensemble (55% XGBoost + 35% LightGBM + 10% CatBoost)
      predicting continuous unclipped taxi-out durations.
  • Regime 2A (Missing AOBT at Rome Fiumicino LIRF):
      Accounts for Italian ANSP radar pipeline defaulting BLOCK_TIME = SCHED_TIME.
      Predicts clip(diff_sched, stat_mean_apt, 60000).
  • Regime 2B (Missing AOBT outside LIRF, other 9 major hubs):
      Surface Layout & Flow Regressor (LightGBM on layout geometry, stand-runway routing,
      wake category, and rolling airport/runway congestion pressure).
"""

import pickle
import time
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features
from src.models_catboost import prepare_catboost_df
from src.models_surface import predict_surface_taxi

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

# Operators at LIRF with >= 95% BLOCK_TIME == SCHED_TIME default rate across 12 months
PURE_LIRF_OPERATORS = {
    "5689a9068938a8902f5282c8cc76c253b2be941fa9fe52c7886a117e",
    "c9dc9ccb6bc1b903e28b65f8cc2eb108e64c23f66c8aa5d045d43be6",
    "fc83d3690709b3f64688f1c58079b5c2aeb387f54c9359e1bf0bfa6f",
    "4db206f1ddf5ab8fd74db97d6f89ab921b7ce3709b1192e21e06f5c8",
    "99595156e5b49dbd5f61adcee5334e0a4f5f5127027d14dbba5bbd4c",
    "36b065acb417099ca2f0c1f2227c67923769c9dcabf21ebaa7d4615a",
    "b39f64fcf31a26c657acfdd4515735aa4b8408f972da54ca70f44bb5",
    "dad5535be9201dc37a9b100d59cdf9c3f3ab663c6c19f5dfdbb8a737",
    "896f2dd89b8e2807e67ff8fd25ea5aa8784d0b0467b7f1e40eb14080",
    "a0acbaa27f3def4904a71b9de26e80b27fb33580a8eb33a64708fffa",
}


class TwoRegimeEnsemble:
    """Production Two-Regime Ensemble for the PRC Data Challenge 2026."""

    def __init__(self, w_xgb: float = 0.55, w_lgb: float = 0.45, w_cb: float = 0.00):
        total = w_xgb + w_lgb + w_cb
        self.w_xgb = w_xgb / total
        self.w_lgb = w_lgb / total
        self.w_cb = w_cb / total
        self._load_all_models()

    def _load_all_models(self):
        print(f"📦 Loading TwoRegimeEnsemble Models (Weights: XGB={self.w_xgb:.2f}, LGB={self.w_lgb:.2f}, CB={self.w_cb:.2f})...")
        t0 = time.time()
        self.models = {}

        for season in ["winter", "summer"]:
            # 1. XGBoost
            xgb_m = xgb.XGBRegressor()
            xgb_m.load_model(str(MODELS_DIR / f"xgb_{season}.json"))
            with open(MODELS_DIR / f"cat_dtypes_{season}.pkl", "rb") as f:
                cat_dtypes = pickle.load(f)

            # 2. LightGBM
            with open(MODELS_DIR / f"lgb_{season}.pkl", "rb") as f:
                lgb_m = pickle.load(f)
            with open(MODELS_DIR / f"encs_{season}.pkl", "rb") as f:
                encs = pickle.load(f)
            with open(MODELS_DIR / f"cols_{season}.pkl", "rb") as f:
                cols = pickle.load(f)

            # 3. CatBoost (optional if w_cb > 0)
            cb_m = None
            cols_cb = None
            encs_cb = None
            if self.w_cb > 0:
                cb_m = CatBoostRegressor()
                cb_m.load_model(str(MODELS_DIR / f"cb_{season}.cbm"))
                with open(MODELS_DIR / f"cols_cb_{season}.pkl", "rb") as f:
                    cols_cb = pickle.load(f)
                with open(MODELS_DIR / f"encs_cb_{season}.pkl", "rb") as f:
                    encs_cb = pickle.load(f)

            self.models[season] = {
                "xgb": xgb_m,
                "cat_dtypes": cat_dtypes,
                "lgb": lgb_m,
                "encs": encs,
                "cols": cols,
                "cb": cb_m,
                "cols_cb": cols_cb,
                "encs_cb": encs_cb,
            }

        # Load Full-Year Foundation XGBoost
        xgb_fy = xgb.XGBRegressor()
        xgb_fy.load_model(str(MODELS_DIR / "xgb_fullyear.json"))
        with open(MODELS_DIR / "encs_fullyear.pkl", "rb") as f:
            encs_fy = pickle.load(f)
        with open(MODELS_DIR / "cols_fullyear.pkl", "rb") as f:
            cols_fy = pickle.load(f)
        with open(MODELS_DIR / "cat_dtypes_fullyear.pkl", "rb") as f:
            cat_dtypes_fy = pickle.load(f)

        self.fy_model = {
            "xgb": xgb_fy,
            "encs": encs_fy,
            "cols": cols_fy,
            "cat_dtypes": cat_dtypes_fy,
        }

        # Load Unified Missing Foundation Models (LightGBM + XGBoost)
        with open(MODELS_DIR / "missing_lgb.pkl", "rb") as f:
            missing_lgb = pickle.load(f)
        missing_xgb = xgb.XGBRegressor()
        missing_xgb.load_model(str(MODELS_DIR / "missing_xgb.json"))
        with open(MODELS_DIR / "missing_cols.pkl", "rb") as f:
            missing_cols = pickle.load(f)
        with open(MODELS_DIR / "missing_cat_dtypes.pkl", "rb") as f:
            missing_cat_dtypes = pickle.load(f)
        with open(MODELS_DIR / "missing_encs.pkl", "rb") as f:
            missing_encs = pickle.load(f)

        self.missing_model = {
            "lgb": missing_lgb,
            "xgb": missing_xgb,
            "cols": missing_cols,
            "cat_dtypes": missing_cat_dtypes,
            "encs": missing_encs,
        }

        # Load LIRF Stage-1 Probabilistic Classifier if present
        if (MODELS_DIR / "lirf_stage1_classifier.pkl").exists():
            with open(MODELS_DIR / "lirf_stage1_classifier.pkl", "rb") as f:
                self.lirf_clf = pickle.load(f)
            with open(MODELS_DIR / "lirf_classifier_cols.pkl", "rb") as f:
                self.lirf_clf_cols = pickle.load(f)
            with open(MODELS_DIR / "lirf_classifier_cats.pkl", "rb") as f:
                self.lirf_clf_cats = pickle.load(f)
            print("  • Loaded LIRF Stage-1 Classifier successfully.")
        else:
            self.lirf_clf = None

        print(f"✅ All seasonal, full-year & missing foundation models loaded successfully in {time.time() - t0:.2f}s.")

    def predict_season(self, dep_df: pl.DataFrame, season: str) -> np.ndarray:
        """Predict departure taxi times for a season using Two-Regime routing."""
        m_dict = self.models[season]
        n_total = len(dep_df)

        mvt = dep_df["MVT_TIME_UTC_mvt"]
        sched = dep_df["SCHED_TIME_UTC_mvt"]
        aobt = dep_df["AOBT_3_flt"]
        adep = dep_df["ADEP_mvt"].to_numpy()

        d_aobt = (mvt - aobt).dt.total_seconds().to_numpy()
        d_sched = (mvt - sched).dt.total_seconds().to_numpy()
        has_aobt = (~np.isnan(d_aobt)) & (d_aobt >= 60) & (d_aobt <= 14400)

        n_valid = np.sum(has_aobt)
        n_miss = np.sum(~has_aobt)
        print(f"  • Routing: Valid AOBT = {n_valid:,} ({n_valid/n_total*100:.2f}%) | Missing AOBT = {n_miss:,} ({n_miss/n_total*100:.2f}%)")

        # --- REGIME 1: Valid AOBT Flights (High-Precision Mega-Blend) ---
        X_pl, _, _ = features.extract_features(dep_df, target_encodings=m_dict["encs"])

        # A. Seasonal XGBoost prediction
        X_xgb = X_pl.select(m_dict["cols"]).to_pandas()
        for c, dtype in m_dict["cat_dtypes"].items():
            if c in X_xgb.columns:
                X_xgb[c] = pd.Categorical(X_xgb[c], categories=dtype.categories, ordered=dtype.ordered)
        p_xgb = m_dict["xgb"].predict(X_xgb)

        # B. Full-Year Foundation XGBoost prediction
        X_pl_fy, _, _ = features.extract_features(dep_df, target_encodings=self.fy_model["encs"], attach_weather=True)
        X_xgb_fy = X_pl_fy.select(self.fy_model["cols"]).to_pandas()
        for c, dtype in self.fy_model["cat_dtypes"].items():
            if c in X_xgb_fy.columns:
                X_xgb_fy[c] = pd.Categorical(X_xgb_fy[c], categories=dtype.categories, ordered=dtype.ordered)
        p_xgb_fy = self.fy_model["xgb"].predict(X_xgb_fy)

        # C. Seasonal LightGBM prediction
        X_lgb = X_pl.select(m_dict["cols"]).to_pandas()
        for c in X_lgb.select_dtypes(include="category").columns:
            X_lgb[c] = X_lgb[c].astype("category")
        p_lgb = m_dict["lgb"].predict(X_lgb)

        # High-Precision Mega-Blend (85% Full-Year XGB + 15% Seasonal XGB)
        p_blend_valid = 0.85 * p_xgb_fy + 0.15 * p_xgb

        # --- ASSEMBLE PREDICTIONS ---
        final_preds = np.zeros(n_total, dtype=np.float64)

        # 1. Valid AOBT: High-Precision Blend
        final_preds[has_aobt] = p_blend_valid[has_aobt]

        # 2. Missing AOBT: Unified Foundation Model (LightGBM + XGBoost Blend)
        miss_mask = ~has_aobt
        if n_miss > 0:
            from src.train_missing_foundation import extract_missing_features
            miss_sub = dep_df.filter(pl.Series(miss_mask))
            X_miss, _, _ = extract_missing_features(miss_sub, encs=self.missing_model["encs"])
            X_miss_df = X_miss[self.missing_model["cols"]].copy()

            # LightGBM missing
            X_miss_lgb = X_miss_df.copy()
            for c in X_miss_lgb.select_dtypes(include="category").columns:
                X_miss_lgb[c] = X_miss_lgb[c].astype("category")
            p_miss_lgb = np.clip(self.missing_model["lgb"].predict(X_miss_lgb), 60, 60000)

            # XGBoost missing
            X_miss_xgb = X_miss_df.copy()
            for c, dtype in self.missing_model["cat_dtypes"].items():
                if c in X_miss_xgb.columns:
                    X_miss_xgb[c] = pd.Categorical(X_miss_xgb[c], categories=dtype.categories, ordered=dtype.ordered)
            p_miss_xgb = np.clip(self.missing_model["xgb"].predict(X_miss_xgb), 60, 60000)

            p_miss_blend = 0.50 * p_miss_xgb + 0.50 * p_miss_lgb
            final_preds[miss_mask] = np.clip(p_miss_blend, 60, 88500)

        # --- REGIME 3: Deterministic Signatures & Physical Boundary Calibration ---
        # 1. Corrupted Radar Pushback (d_aobt < 60s) -> Airport-specific physical median
        CORRUPT_AOBT_FALLBACK = {
            "EHAM": 615.0,
            "LSZH": 540.0,
            "LTFM": 1500.0,
            "EGLL": 1250.0,
            "LEMD": 1110.0,
            "LEBL": 1100.0,
            "EDDF": 990.0,
            "EDDM": 1045.0,
            "LFPG": 1220.0,
            "LIRF": 1600.0,
        }
        corrupt_aobt = (~np.isnan(d_aobt)) & (d_aobt < 60)
        for apt, med_val in CORRUPT_AOBT_FALLBACK.items():
            m_cor = corrupt_aobt & (adep == apt)
            final_preds[m_cor] = med_val

        # 2. Pure LIRF operators (where BLOCK == SCHED with 99-100% probability)
        ops = dep_df["AIRCRAFT_OPERATOR_flt"].cast(pl.Utf8).fill_null("UNK").to_numpy()
        is_pure_lirf = (adep == "LIRF") & np.isin(ops, list(PURE_LIRF_OPERATORS)) & (d_sched >= 60) & (d_sched <= 88500)
        final_preds[is_pure_lirf] = d_sched[is_pure_lirf]

        # 3. LIRF Stage-1 Classifier for completely missing AOBT (Predicts Schedule Default vs Normal Taxi)
        if self.lirf_clf is not None:
            m_lirf_miss = np.isnan(d_aobt) & (adep == "LIRF") & (~is_pure_lirf)
            if np.sum(m_lirf_miss) > 0:
                sub_lirf = dep_df.filter(pl.Series(m_lirf_miss))
                mvt_l = sub_lirf["MVT_TIME_UTC_mvt"]
                sched_l = sub_lirf["SCHED_TIME_UTC_mvt"]
                ds_l = (mvt_l - sched_l).dt.total_seconds().to_numpy()
                hour_l = mvt_l.dt.hour().to_numpy()
                dow_l = mvt_l.dt.weekday().to_numpy()
                month_l = mvt_l.dt.month().to_numpy()

                ops_l = sub_lirf["AIRCRAFT_OPERATOR_flt"].cast(pl.Utf8).fill_null("UNK").to_numpy()
                actype_l = sub_lirf["AIRCRAFT_TYPE_mvt"].cast(pl.Utf8).fill_null("UNK").to_numpy()
                runway_l = sub_lirf["RUNWAY_mvt"].cast(pl.Utf8).fill_null("UNK").to_numpy()
                stand_l = sub_lirf["STAND_mvt"].cast(pl.Utf8).fill_null("UNK").to_numpy()
                segment_l = sub_lirf["MARKET_SEGMENT_flt"].cast(pl.Utf8).fill_null("UNK").to_numpy()

                X_clf = pd.DataFrame({
                    "hour": hour_l,
                    "dow": dow_l,
                    "month": month_l,
                    "d_sched": ds_l,
                    "log_d_sched": np.log1p(np.clip(ds_l, 0, None)),
                    "is_delay_gt_2h": (ds_l > 7200).astype(int),
                    "is_delay_gt_5h": (ds_l > 18000).astype(int),
                    "is_delay_gt_12h": (ds_l > 43200).astype(int),
                    "is_delay_gt_20h": (ds_l > 72000).astype(int),
                    "hour_sin": np.sin(2 * np.pi * hour_l / 24),
                    "hour_cos": np.cos(2 * np.pi * hour_l / 24),
                    "operator": ops_l,
                    "actype": actype_l,
                    "runway": runway_l,
                    "stand": stand_l,
                    "market_segment": segment_l,
                })
                for c, dt in self.lirf_clf_cats.items():
                    if c in X_clf.columns:
                        X_clf[c] = pd.Categorical(X_clf[c], categories=dt.categories, ordered=dt.ordered)

                probs_l = self.lirf_clf.predict_proba(X_clf)[:, 1]
                idx_lirf = np.where(m_lirf_miss)[0]
                for i, p_def in enumerate(probs_l):
                    if p_def > 0.60 and ds_l[i] >= 1800:
                        final_preds[idx_lirf[i]] = np.clip(ds_l[i], 60, 88500)

        # 4. Helicopter pad liftoff physics (bounded between 30s and 300s)
        actype_np = dep_df["AIRCRAFT_TYPE_mvt"].cast(pl.Utf8).fill_null("UNK").to_numpy()
        is_helo = np.isin(actype_np, list(features.HELICOPTER_TYPES))
        final_preds[is_helo] = np.clip(final_preds[is_helo], 30, 300)

        # 5. Airport-specific physical caps across ALL non-LIRF flights (valid + missing)
        caps = {
            "LEMD": 3600,
            "LEBL": 3600,
            "EDDF": 7000,
            "EDDM": 7000,
            "LTFM": 7500,
            "LSZH": 7200,
            "EGLL": 7200,
            "LFPG": 7200,
            "EHAM": 7200,
        }
        for apt, max_t in caps.items():
            m_cap = (adep == apt) & (~is_helo)
            final_preds[m_cap] = np.clip(final_preds[m_cap], 60, max_t)

        # Final global operational clipping and rounding
        return np.clip(np.round(final_preds), 5, 88500).astype(np.int32)


# Backward compatibility alias
SeasonalEnsemble = TwoRegimeEnsemble

if __name__ == "__main__":
    ens = TwoRegimeEnsemble()
    # Quick test on 1000 rows
    test_df = features.load_training_data(RAW_DIR, months=["2025-01"]).filter(pl.col("PHASE_mvt") == "DEP").head(1000)
    test_df = features.compute_congestion_metrics(test_df)
    preds = ens.predict_season(test_df, "winter")
    print(f"Sample predictions (first 10): {preds[:10]}")
    print(f"Prediction min: {preds.min()}, median: {np.median(preds)}, max: {preds.max()}")
