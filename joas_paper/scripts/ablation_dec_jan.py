"""Controlled out-of-time ablation of the JOAS paper (Tables 2 and 3): train on December 2025,
validate on all January 2025 departures.

Usage (from the repository root), one run per seed:
    python joas_paper/scripts/ablation_dec_jan.py joas_paper/scripts/results 42
"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import polars as pl, pandas as pd, numpy as np, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
OUT = Path(sys.argv[1]); SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
from src.features_fast import extract_features_seasonal, compute_target_encodings_clean
from src.features_physics_engine import compute_physics_features_fast
RAW = ROOT/"data"/"raw"
tr_raw = pl.read_parquet(RAW/"training_2025-12-01_2026-01-01.parquet")
va_raw = pl.read_parquet(RAW/"training_2025-01-01_2025-02-01.parquet")

def milestones(dep):
    y = dep["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    has = dep["AOBT_3_flt"].is_not_null().to_numpy()
    da = np.where(has, (dep["MVT_TIME_UTC_mvt"]-dep["AOBT_3_flt"]).dt.total_seconds().to_numpy(), np.nan).astype(float)
    ds = (dep["MVT_TIME_UTC_mvt"]-dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy().astype(float)
    apt = dep["ADEP_mvt"].to_numpy()
    clean = has & (da >= 120) & (da <= 7200)
    ext = has & (da > 7200)
    # Regime 2B as in src/train_and_predict_final_phase.py (no NM off-block time at all)
    lirf = (~has) & (apt == "LIRF") & (ds > 7200)
    miss = (~clean) & (~ext) & (~lirf)
    return y, da, ds, apt, clean, ext, lirf, miss

encs = compute_target_encodings_clean(tr_raw.filter(pl.col("PHASE_mvt")=="DEP"))
t0=time.time()
Xtr, dtr = extract_features_seasonal(tr_raw, encs)
Xva, dva = extract_features_seasonal(va_raw, encs)
ptr = compute_physics_features_fast(tr_raw).drop("MVT_ID_mvt").to_pandas()
t1=time.time(); pva = compute_physics_features_fast(va_raw).drop("MVT_ID_mvt").to_pandas(); t_phys=time.time()-t1
ov = [c for c in ptr.columns if c in Xtr.columns]
Xtr_p = pd.concat([Xtr.reset_index(drop=True), ptr.drop(columns=ov)], axis=1)
Xva_p = pd.concat([Xva.reset_index(drop=True), pva.drop(columns=ov)], axis=1)
for c in Xtr_p.columns:
    if str(Xtr_p[c].dtype) in ("category","object"):
        Xtr_p[c] = Xtr_p[c].astype("category")
        Xva_p[c] = pd.Categorical(Xva_p[c], categories=Xtr_p[c].cat.categories)
base_cols = list(Xtr.columns); phys_cols = [c for c in ptr.columns if c not in ov]
print("feat extraction s", time.time()-t0, "physics(Jan) s", t_phys, "n base", len(base_cols), "n phys new", len(phys_cols), phys_cols, flush=True)

ytr, datr, dstr, apttr, ctr, etr, ltr, mtr = milestones(dtr)
yva, dava, dsva, aptva, cva, eva, lva, mva = milestones(dva)
fit_mask = ctr & (ytr >= 60) & (ytr <= 7200)
off_tr = datr - ytr
med_off = float(np.median(off_tr[ctr]))
P = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=63, min_child_samples=30,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, seed=SEED, n_jobs=-1)
def fit(cols, label, mask, rounds=450):
    return lgb.train(P, lgb.Dataset(Xtr_p.loc[mask, cols], label=label[mask]), num_boost_round=rounds)
def rmse(p, m=None):
    m = np.ones_like(yva, bool) if m is None else m
    return float(np.sqrt(np.mean((yva[m]-p[m])**2)))

groups = {
 "queue_at_pushback": ["rwy_queue_at_push","apt_queue_at_push"],
 "wake_separation": ["prev_dep_wake_penalty","dt_to_prev_dep","is_behind_heavy","recat_eu"],
 "osm_taxi_network": ["osm_dist_m","sl_dist_m","tortuosity","utt_osm_seconds"],
 "pier_contention": ["pier_queue_15m"],
 "weather": [c for c in Xtr_p.columns if c.startswith("wx_") or c in ("snowfall","temperature_2m","wind_speed_10m","headwind","crosswind","is_tailwind")],
 "rolling_flow_15_30min": [c for c in Xtr_p.columns if c.startswith(("rwy_dep_q","rwy_all_q","apt_dep_q","apt_arr_q","arr_taxi_")) or c in ("dt_prev_dep_rwy","dt_prev_any_rwy","is_prev_arr")],
 "historical_encodings": [c for c in Xtr_p.columns if c.startswith("enc_")],
}
res = {"n_val": int(len(yva)), "regime_counts_val": dict(clean=int(cva.sum()), ext=int(eva.sum()), lirf=int(lva.sum()), miss=int(mva.sum())),
       "regime_counts_tr": dict(clean=int(ctr.sum()), ext=int(etr.sum()), lirf=int(ltr.sum()), miss=int(mtr.sum())),
       "median_offset_tr": med_off, "groups": groups, "t_phys_jan_s": t_phys}
all_cols = list(Xtr_p.columns)

# (A) direct taxi regression, all departures, target clipped
dmask = np.ones_like(ytr, bool)
mA = fit(base_cols, np.clip(ytr, 60, 7200), dmask)
pA = np.clip(mA.predict(Xva_p[base_cols]), 60, 7200)
res["A_direct_base"] = dict(full=rmse(pA), clean=rmse(pA, cva))

# (B) offset, base features ; (C) offset + physics
def offset_pred(cols):
    m = fit(cols, off_tr, fit_mask)
    return m, np.clip(dava - m.predict(Xva_p[cols]), 60, 7200)
mB, pB = offset_pred(base_cols)
mC, pC = offset_pred(all_cols)
NAIVE = 1050.0
def naive(pclean): return np.where(cva, pclean, NAIVE)
res["B_offset_base_naive"] = dict(full=rmse(naive(pB)), clean=rmse(pB, cva))
res["C_offset_phys_naive"] = dict(full=rmse(naive(pC)), clean=rmse(pC, cva))

# routing components on top of C
miss_fit = mtr
mM = fit(base_cols, np.clip(ytr, 60, 3600), miss_fit, rounds=300)
pM = np.clip(mM.predict(Xva_p[base_cols]), 60, 3600)
p_r1 = naive(pC); p_r1[mva] = pM[mva]
p_r2 = p_r1.copy(); p_r2[lva] = dsva[lva]
p_r3 = p_r2.copy(); p_r3[eva] = np.maximum(60, dava[eva] - med_off)
# alternative for ext: clipped 7200
p_r3_clip = p_r2.copy(); p_r3_clip[eva] = 7200.0
res["R1_plus_missing_specialist"] = dict(full=rmse(p_r1), miss=rmse(p_r1, mva))
res["R2_plus_lirf_identity"] = dict(full=rmse(p_r2), lirf=rmse(p_r2, lva))
res["R3_full_router"] = dict(full=rmse(p_r3), ext=rmse(p_r3, eva) if eva.any() else None, clean=rmse(p_r3, cva), miss=rmse(p_r3, mva), lirf=rmse(p_r3, lva))
res["R3_ext_clip7200"] = dict(full=rmse(p_r3_clip))
# leave-one-out regime components from full router
p_noL = p_r3.copy(); p_noL[lva] = p_r1[lva]
p_noM = p_r3.copy(); p_noM[mva] = NAIVE
p_noE = p_r3.copy(); p_noE[eva] = NAIVE
res["router_minus_lirf"] = rmse(p_noL); res["router_minus_missing_spec"] = rmse(p_noM); res["router_minus_ext"] = rmse(p_noE)
# squared-error share by regime under full router
se = (yva - p_r3)**2
res["se_share"] = {k: float(se[m].sum()/se.sum()) for k, m in dict(clean=cva, miss=mva, lirf=lva, ext=eva).items()}
res["lirf_identity_hit_rate_120s_val"] = float(np.mean(np.abs(yva[lva]-dsva[lva]) < 120)) if lva.any() else None

# drop-group ablations (clean-flight RMSE + full RMSE under full router)
abl = {}
for g, cols in groups.items():
    keep = [c for c in all_cols if c not in cols]
    _, pg = offset_pred(keep)
    pf = p_r3.copy(); pf[cva] = pg[cva]
    abl[g] = dict(clean=rmse(pg, cva), full=rmse(pf), n_removed=len([c for c in cols if c in all_cols]))
    print(g, abl[g], flush=True)
res["drop_group"] = abl

imp = pd.Series(mC.feature_importance("gain"), index=all_cols).sort_values(ascending=False)
imp.to_csv(OUT/f"importance_gain_s{SEED}.csv", header=["gain"])
res["corr_da_y_clean_val"] = float(np.corrcoef(dava[cva], yva[cva])[0,1])
json.dump(res, open(OUT/f"ablation_s{SEED}.json","w"), indent=2)
print(json.dumps({k:v for k,v in res.items() if k!="groups"}, indent=1))
