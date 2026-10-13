"""
Generate inspiring-kitten_v110.parquet (paper, Section 3.3; public RMSE 294.6465 s).

v110 = 0.5 * v104 + 0.5 * v109 on departures with a plausible NM off-block time
(120 <= da <= 7200 s, v104 <= 15,000 s, v109 > 60 s), and v104 on all other departures.
v109 is a seasonal LightGBM + XGBoost + CatBoost offset ensemble with the physics features
(public RMSE 326.0368 s). v110 is the probe that fixes the direction used by generate_v112.py.

Usage (from the repository root):
    python src/generate_v110.py
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import polars as pl
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Load base predictions
sub104 = pl.read_parquet(SUBMISSIONS_DIR / "inspiring-kitten_v104.parquet")
sub109 = pl.read_parquet(SUBMISSIONS_DIR / "inspiring-kitten_v109.parquet")
sub_template = pl.read_parquet(RAW_DIR / "submitting.parquet")

p104 = sub104["TAXITIME_SEC_mvt"].to_numpy().astype(float)
p109 = sub109["TAXITIME_SEC_mvt"].to_numpy().astype(float)

# Test departures in template order, to identify the regimes
df_ranking = pl.read_parquet(RAW_DIR / "final_ranking.parquet")
dep = df_ranking.filter(pl.col("PHASE_mvt") == "DEP")
dep_test = sub_template.select("MVT_ID_mvt").join(dep, on="MVT_ID_mvt", how="left")

has_aobt = dep_test["AOBT_3_flt"].is_not_null().to_numpy()
da = np.where(has_aobt, (dep_test["MVT_TIME_UTC_mvt"] - dep_test["AOBT_3_flt"]).dt.total_seconds().to_numpy(), np.nan)

# Regime 1 departures where both inputs are in a normal range
mask_clean = has_aobt & (da >= 120.0) & (da <= 7200.0) & (p104 <= 15000) & (p109 > 60)

w = 0.50
p110 = p104.copy()
p110[mask_clean] = w * p109[mask_clean] + (1 - w) * p104[mask_clean]
p110_int = np.clip(np.round(p110), 60, 88500).astype(np.int32)

df_v110 = pl.DataFrame({
    "MVT_ID_mvt": sub_template["MVT_ID_mvt"],
    "TAXITIME_SEC_mvt": p110_int,
})
assert len(df_v110) == 344841
assert df_v110["TAXITIME_SEC_mvt"].null_count() == 0
assert (df_v110["MVT_ID_mvt"] == sub_template["MVT_ID_mvt"]).all()

out_file = SUBMISSIONS_DIR / "inspiring-kitten_v110.parquet"
df_v110.write_parquet(out_file)
print(f"Blended departures: {int(mask_clean.sum()):,} of {len(df_v110):,}")
print(f"Saved: {out_file}")
