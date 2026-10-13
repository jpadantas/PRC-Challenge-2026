# PRC Data Challenge 2026: team `inspiring-kitten`

Taxi-out time prediction for departures at ten European airports, developed for the
[EUROCONTROL Performance Review Commission (PRC) Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/).

The method and its evaluation are described in the accompanying paper,
*Gradient Boosted Trees and Operational Regimes for Airport Taxi-Out Time Prediction*
([`joas_paper/main.pdf`](joas_paper/main.pdf)), written for the Journal of Open Aviation Science
(JOAS). **The paper has not been submitted yet** and is included here as a draft.

## Results

| Set | Departures | Months | RMSE (s) |
| :-- | --: | :-- | --: |
| Phase 1 public leaderboard, best submission (`v112`) | 344,841 | Jan + Jul 2026 | **290.14** |
| Phase 1, production pipeline alone (`v111`) | 344,841 | Jan + Jul 2026 | 319.79 |
| Controlled holdout, full router (train Dec 2025, test Jan 2025, 3 seeds) | 153,706 | Jan 2025 | 367.24 ± 0.45 |

The final Phase 2 file (`data/submissions/inspiring-kitten_final.parquet`, 670,790 departures)
contains `v112` for January and July and the production pipeline for February and June 2026.

## Method in brief

1. **Milestone identity.** The target is `TAXITIME = T_mvt - T_block`. The airport off-block time
   `T_block` is hidden at test time, but the Network Manager actual off-block time `AOBT_3_flt` is
   available. With `da = T_mvt - AOBT_3_flt`, the target becomes `TAXITIME = da - offset`, and the
   models learn the offset between the two off-block records.
2. **Deterministic regime router** (Fig. 3 of the paper), tested in this order:
   - Regime 1: NM off-block time present and `120 <= da <= 7200` s. Offset ensemble.
   - Regime 3: NM off-block time present and `da > 7200` s. Unclipped identity
     `da - median airport offset`.
   - Regime 2B: LIRF, no NM off-block time and `T_mvt - T_sched > 7200` s. Schedule identity
     `T_mvt - T_sched`.
   - Regime 2A: all other departures. Flight-plan LightGBM clipped to [60, 3600] s.
3. **Features.** NM milestone differences, rolling runway and airport traffic, historical
   encodings, weather (METAR and Open-Meteo, runway-aligned wind) and OpenStreetMap taxi distances
   (`src/features_fast.py`). A physics feature engine (`src/features_physics_engine.py`) adds the
   runway queue at pushback (an O(N log N) sweep), a RECAT-EU wake separation penalty, an
   unimpeded taxi time and pier contention.
4. **Models.** Regime 1 averages seasonal LightGBM, XGBoost (GPU) and CatBoost (GPU) models
   trained on the offset, with weights 0.40, 0.35 and 0.25.
5. **Leaderboard blending.** The public RMSE of any linear blend of scored submissions can be
   computed exactly from the public scores alone (Sec. 2.6 of the paper). This was used to build
   `v104` and to choose the step along `v110 - v104` that gives `v112` (`v110` itself is a fixed
   50/50 probe). The paper reports it as leaderboard adaptation, not as model improvement.

## How the final submission was built

1. `v104` (290.74 s): blend of six earlier submissions, built with the scripts in `archive/`.
2. `v109` (326.04 s): seasonal LightGBM + XGBoost + CatBoost offset ensemble with the physics features.
3. `v110` (294.65 s): `0.5 v104 + 0.5 v109` on Regime 1 departures where `v104 <= 15,000 s` and
   `v109 > 60 s`, `v104` elsewhere.
4. `v112` (290.14 s, best public score): `v104 - 0.45 (v110 - v104)`.
5. Final file: `v112` for January and July 2026, production pipeline
   (`src/train_and_predict_final_phase.py`) for February and June 2026.

`v104` and `v109` are versioned because they cannot be rebuilt from a fresh clone (see
[`archive/README.md`](archive/README.md)). Everything after them is rebuilt by the scripts in `src/`.

## Repository layout

```
src/                                            runnable pipeline
  download_data.py                              challenge data from the OpenSky S3 bucket
  download_metar.py, features_metar.py          METAR (Iowa Environmental Mesonet)
  download_weather.py, merge_metar_weather.py   Open-Meteo, merged with METAR
  features_osm_taxiway.py                       OpenStreetMap stand-to-runway distances
  features_fast.py                              base feature set (62 features)
  features_physics_engine.py                    physics feature engine (9 features)
  train_and_predict_final_phase.py              production pipeline (Phase 2 file and v111)
  generate_v110.py, generate_v112.py            blends leading to the best public submission
  update_final_submission_with_v112.py          final file (uploads only with --upload)
data/external/                                  open external datasets (see Data)
data/submissions/                               v104, v109, v110, v111, v112 and the final file
joas_paper/
  main.pdf                                      paper (draft, not yet submitted)
  scripts/ablation_dec_jan.py                   controlled ablation (Tables 2 and 3)
  scripts/make_figures.py                       Figs. 1, 2, 4 and 5 and the dataset statistics in the text
  scripts/results/                              ablation outputs for seeds 42, 7 and 2026
archive/                                        scripts behind the earlier submissions and the submission log
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

XGBoost and CatBoost are configured for a CUDA GPU in the production pipeline; on a CPU-only
machine, change `device="cuda"` / `task_type="GPU"` in `src/train_and_predict_final_phase.py`.

To download the challenge data, create `credentials.json` in the repository root with the access
keys provided by the organisers for the OpenSky S3 bucket. The file is gitignored; never commit it.

```json
{
  "url": "https://s3-console.opensky-network.org/api/v1/service-account-credentials",
  "accessKey": "YOUR_ACCESS_KEY",
  "secretKey": "YOUR_SECRET_KEY",
  "api": "s3v4",
  "path": "auto"
}
```

## Data

**Challenge data** (not redistributed here). Download it into `data/raw/` with:

```bash
python src/download_data.py
```

**External open data** (versioned in `data/external/`, or rebuilt with the scripts below):

| File | Source | Licence | Script | Used by |
| :-- | :-- | :-- | :-- | :-- |
| `metar_raw.parquet`, `metar_hourly.parquet` | [Iowa Environmental Mesonet ASOS/METAR archive](https://mesonet.agron.iastate.edu/request/download.phtml) | free public data | `src/download_metar.py`, `src/features_metar.py` | `src/`, `archive/` |
| `weather_hourly.parquet` | [Open-Meteo historical weather API](https://open-meteo.com/) merged with METAR | CC BY 4.0 | `src/download_weather.py`, `src/merge_metar_weather.py` | `src/`, `archive/` |
| `osm_taxiway_distances.parquet` | © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors (3,844 stand-runway pairs, ten airports) | ODbL 1.0 | `src/features_osm_taxiway.py --all` | `src/`, `archive/` |
| `airports.csv`, `runways.csv` | [OurAirports](https://github.com/davidmegginson/ourairports-data) | Public domain (Unlicense) | manual download | `archive/` |
| `aircraft_types.csv` | *What is flying?* database by Julien Broutier ([whatisflying-db](https://github.com/laegsgaardTroels/whatisflying-db)) | ODbL 1.0 | manual download | `archive/` |

The files used only by `archive/` fed earlier submissions that are part of `v104`, the anchor of
the final submission. Details, attribution and licence notes are in
[`data/external/README.md`](data/external/README.md).

## Reproducing the results

All commands run from the repository root, after downloading the challenge data.

```bash
# 1. External data (optional: the processed files are already in data/external/)
python src/download_metar.py && python src/features_metar.py
python src/download_weather.py && python src/merge_metar_weather.py
python src/features_osm_taxiway.py --all

# 2. Production pipeline: writes inspiring-kitten_final.parquet (all four months) and _v111.parquet
python src/train_and_predict_final_phase.py

# 3. Best public submission, from the versioned v104 and v109 (reproduces v110 and v112 exactly)
python src/generate_v110.py
python src/generate_v112.py

# 4. Final file = v112 for January and July + step 2 for February and June (no upload)
python src/update_final_submission_with_v112.py

# 5. Paper: controlled ablation (one run per seed), figures and statistics
python joas_paper/scripts/ablation_dec_jan.py joas_paper/scripts/results 42
python joas_paper/scripts/ablation_dec_jan.py joas_paper/scripts/results 7
python joas_paper/scripts/ablation_dec_jan.py joas_paper/scripts/results 2026
python joas_paper/scripts/make_figures.py
```

Steps 2 to 4 overwrite the versioned files in `data/submissions/`. Step 4 uploads to the team's
OpenSky bucket only when run with `--upload`.

## Licence

The code in this repository is released under the [GNU General Public License v3.0](LICENSE).
The external datasets in `data/external/` keep their original licences (table above).

## Citation

The paper has not been submitted yet. Until it is published, please cite the preprint:

```bibtex
@unpublished{dantas2026gradient,
  author = {Dantas, Joao P. A.},
  title  = {Gradient Boosted Trees and Operational Regimes for Airport Taxi-Out Time Prediction},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/jpadantas/PRC-Challenge-2026/blob/main/joas_paper/main.pdf}
}
```

## Author

Joao P. A. Dantas, Decision Support Systems Subdivision, Institute for Advanced Studies (IEAv),
São José dos Campos, Brazil. ORCID [0000-0003-0300-8027](https://orcid.org/0000-0003-0300-8027).
