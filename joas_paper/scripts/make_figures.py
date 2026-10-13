"""
Regenerate the data figures of the JOAS paper and print the dataset statistics
quoted in the text. Figure 3 (regime router) is drawn with TikZ in main.tex.

Usage (from the repository root):
    python joas_paper/scripts/make_figures.py [ablation_dir]

ablation_dir must contain importance_gain_s42.csv written by
joas_paper/scripts/ablation_dec_jan.py (default: joas_paper/scripts/results).
Figure 5 also needs the leaderboard submission files v104, v110 and v112
in data/submissions/.
"""
import sys
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, MultipleLocator

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SUBS = ROOT / "data" / "submissions"
OUT = ROOT / "joas_paper" / "figures"
ABL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "joas_paper" / "scripts" / "results"
OUT.mkdir(parents=True, exist_ok=True)

# JOAS is typeset on B5 paper with 142 mm text width; figures are drawn at final size.
TEXT_W = 142 / 25.4

# Captions in the JOAS class use Source Sans; use it in figures when available.
for f in glob.glob(str(Path.home() / "AppData/Local/Programs/MiKTeX/fonts/opentype/adobe/sourcesans/SourceSans3-*.otf")) \
        + glob.glob("/usr/share/texmf*/fonts/opentype/adobe/sourcesans*/SourceSans3-*.otf"):
    font_manager.fontManager.addfont(f)
for f in glob.glob(str(Path.home() / "AppData/Local/Programs/MiKTeX/fonts/opentype/public/inconsolata/Inconsolatazi4-Regular.otf"))         + glob.glob("/usr/share/texmf*/fonts/opentype/public/inconsolata/Inconsolatazi4-Regular.otf"):
    font_manager.fontManager.addfont(f)
MONO = next((f.name for f in font_manager.fontManager.ttflist if f.name.startswith("Inconsolata")), "DejaVu Sans Mono")
FAMILY = "Source Sans 3" if any(f.name == "Source Sans 3" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"

INK, INK2, MUTED, GRID = "#1a1a19", "#52514e", "#8a8984", "#e4e3df"
BLUE, BLUE_L, ORANGE = "#2a78d6", "#d4e4f7", "#eb6834"
plt.rcParams.update({
    "font.family": FAMILY, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "mathtext.fontset": "custom",
    "mathtext.rm": FAMILY, "mathtext.it": f"{FAMILY}:italic", "mathtext.bf": f"{FAMILY}:bold",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def panel(ax, letter):
    ax.text(-0.01, 1.04, f"({letter})", transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", color=INK, ha="right", va="bottom")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def thousands(v, _):
    return f"{v/1e3:.0f}k" if v else "0"


def load_departures(path_glob="training_*.parquet") -> pl.DataFrame:
    cols = ["ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
    files = sorted(glob.glob(str(RAW / path_glob)))
    df = pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")
    return df.filter(pl.col("PHASE_mvt") == "DEP")


def regimes(dep: pl.DataFrame):
    """Regime router of Sec. 2.4 (identical to src/train_and_predict_final_phase.py)."""
    has = dep["AOBT_3_flt"].is_not_null().to_numpy()
    da = np.where(has, (dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).dt.total_seconds().to_numpy(),
                  np.nan).astype(float)
    ds = (dep["MVT_TIME_UTC_mvt"] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy().astype(float)
    apt = dep["ADEP_mvt"].to_numpy()
    clean = has & (da >= 120) & (da <= 7200)
    ext = has & (da > 7200)
    lirf = ~has & (apt == "LIRF") & (ds > 7200)
    miss = ~clean & ~ext & ~lirf
    return da, ds, clean, ext, lirf, miss


def print_statistics(dep: pl.DataFrame):
    y = dep["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    da, ds, c, e, l, m = regimes(dep)
    off = da[c] - y[c]
    n = len(y)
    print(f"[train 2025] departures={n:,}  median={np.median(y):.0f}  mean={y.mean():.0f}  std={y.std():.0f}  "
          f">7200s={int((y > 7200).sum())}  max={y.max():.0f}")
    print(f"  regimes: R1={c.mean():.2%} R2A={m.mean():.2%} ({int(m.sum())}) R2B={int(l.sum())} R3={int(e.sum())}")
    print(f"  offset R1: median={np.median(off):.0f} IQR={np.percentile(off, [25, 75])} std={off.std():.0f}; "
          f"taxi std R1={y[c].std():.0f}; corr(da,y)={np.corrcoef(da[c], y[c])[0, 1]:.3f}")
    print(f"  R2B: |y-ds|<120s for {np.mean(np.abs(y[l] - ds[l]) < 120):.1%}, y>7200 for {int((y[l] > 7200).sum())}")
    print(f"  R2A: y>3600 {int((y[m] > 3600).sum())}, y>7200 {int((y[m] > 7200).sum())}")
    for name in ("ranking.parquet", "final_ranking.parquet"):
        t = load_departures(name)
        _, _, tc, te, tl, tm = regimes(t)
        print(f"[{name}] departures={len(t):,} R1={tc.mean():.2%} R2A={tm.mean():.2%} "
              f"R2B={int(tl.sum())} ({tl.mean():.3%}) R3={int(te.sum())}")


def fig_traffic(dep: pl.DataFrame):
    months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    counts = np.bincount(dep["MVT_TIME_UTC_mvt"].dt.month().to_numpy(), minlength=13)[1:]
    y = dep["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    apt = dep["ADEP_mvt"].to_numpy()
    airports = sorted(np.unique(apt), key=lambda a: np.median(y[apt == a]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXT_W, 2.15), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.32})
    a1.bar(range(12), counts, color=BLUE, width=0.68)
    a1.set_xticks(range(12), months)
    a1.tick_params(axis="x", length=0)
    a1.yaxis.set_major_formatter(FuncFormatter(thousands))
    a1.set_ylabel("Departures per month, 2025")
    a1.set_ylim(0, 215e3)
    panel(a1, "a")

    data = [y[apt == a] / 60 for a in airports]
    a2.boxplot(data, vert=False, whis=(5, 95), showfliers=False, widths=0.58, patch_artist=True,
               medianprops=dict(color=INK, linewidth=1.1),
               whiskerprops=dict(color=MUTED, linewidth=0.7), capprops=dict(color=MUTED, linewidth=0.7),
               boxprops=dict(facecolor=BLUE_L, edgecolor=BLUE, linewidth=0.7))
    a2.set_yticks(range(1, len(airports) + 1), airports)
    a2.tick_params(axis="y", length=0)
    a2.set_xlabel("Taxi-out time (min)")
    a2.grid(axis="y", visible=False)
    a2.grid(axis="x", visible=True)
    a2.xaxis.set_major_locator(MultipleLocator(5))
    a2.spines["left"].set_visible(False)
    panel(a2, "b")
    save(fig, "traffic_overview")


def fig_offset(dep: pl.DataFrame):
    y = dep["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    da, _, c, *_ = regimes(dep)
    off = da[c] - y[c]
    yc = y[c]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXT_W, 2.1), gridspec_kw={"wspace": 0.28})
    for ax, data, bins, color, xlabel, letter in (
        (a1, yc / 60, np.arange(0, 50.5, 0.5), BLUE, r"Taxi-out time $T_\mathrm{taxi}$ (min)", "a"),
        (a2, off / 60, np.arange(-25.5, 25.6, 1.0), ORANGE,
         r"Milestone offset $\Delta_\mathrm{offset}=da-T_\mathrm{taxi}$ (min)", "b"),
    ):
        ax.hist(data, bins=bins, color=color, linewidth=0)
        med = np.median(data)
        q1, q3 = np.percentile(data, [25, 75])
        ax.axvline(med, color=INK, lw=0.8, ls=(0, (3, 2)))
        ax.text(0.98, 0.95, f"median {med*60:.0f} s\nIQR [{q1*60:.0f}, {q3*60:.0f}] s\nstd {data.std()*60:.0f} s",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK2, linespacing=1.3)
        ax.set_xlabel(xlabel)
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))
        panel(ax, letter)
    a1.set_xlim(0, 50)
    a2.set_xlim(-25, 25)
    a1.set_ylabel("Departures")
    save(fig, "taxitime_physics")


PHYS = ["rwy_queue_at_push", "apt_queue_at_push", "prev_dep_wake_penalty", "dt_to_prev_dep",
        "is_behind_heavy", "utt_osm_seconds", "pier_queue_15m"]


def fig_importance():
    imp = pd.read_csv(ABL / "importance_gain_s42.csv", index_col=0)["gain"]
    imp = 100 * imp / imp.sum()
    rank = imp.rank(ascending=False, method="min").astype(int)
    top = imp.sort_values(ascending=False).head(15)
    phys = imp[PHYS].sort_values(ascending=False)
    names = list(top.index) + list(phys.index)
    vals = np.r_[top.values, phys.values]
    ypos = np.r_[np.arange(len(top)), len(top) + 0.8 + np.arange(len(phys))]
    cols = [BLUE] * len(top) + [ORANGE] * len(phys)

    fig, ax = plt.subplots(figsize=(0.82 * TEXT_W, 3.6))
    x0 = 0.005
    ax.hlines(ypos, x0, vals, color=cols, lw=1.1)
    ax.scatter(vals, ypos, s=16, color=cols, zorder=3, linewidths=0)
    ax.set_xscale("log")
    ax.set_xlim(x0, 60)
    ax.set_xticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
    ax.set_yticks(ypos, names, fontsize=7.5, family=MONO)
    for y, n in zip(ypos, names):
        ax.text(x0 / 1.15, y, f"#{rank[n]}", ha="right", va="center", fontsize=6.5, color=MUTED)
    ax.tick_params(axis="y", length=0, pad=22)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["left"].set_visible(False)
    for y, v in zip(ypos, vals):
        ax.text(v * 1.25, y, f"{v:.2f}" if v < 1 else f"{v:.1f}", va="center", fontsize=6.5, color=INK2)
    ax.text(60, len(top) - 0.1, "physics feature engine", ha="right", va="center", fontsize=7, color=ORANGE)
    ax.text(60, -1.1, "15 most important features", ha="right", va="center", fontsize=7, color=BLUE)
    ax.set_xlabel("Share of total split gain (%, log scale)")
    save(fig, "feature_importance")


def fig_parabola():
    p = {v: pl.read_parquet(SUBS / f"inspiring-kitten_{v}.parquet")["TAXITIME_SEC_mvt"].to_numpy().astype(float)
         for v in ("v104", "v110", "v112")}
    score = {"v104": 290.7445, "v110": 294.6465, "v112": 290.1429}
    d = p["v110"] - p["v104"]
    q = float(np.mean(d ** 2))
    m0, m1 = score["v104"] ** 2, score["v110"] ** 2
    c = (m1 - m0 - q) / 2
    kstar = -c / q

    def rmse(k):
        return np.sqrt(m0 + 2 * k * c + k ** 2 * q)

    k = np.linspace(-1.2, 1.2, 400)
    fig, ax = plt.subplots(figsize=(0.72 * TEXT_W, 2.45))
    ax.plot(k, rmse(k), color=BLUE, lw=1.4, zorder=2)
    ax.scatter([0, 1], [score["v104"], score["v110"]], s=22, color=INK, zorder=4, linewidths=0)
    ax.scatter([kstar], [rmse(kstar)], s=22, facecolor="white", edgecolor=BLUE, lw=1.1, zorder=4)
    ax.scatter([-0.45], [score["v112"]], s=60, marker="*", color=ORANGE, edgecolor=INK, lw=0.4, zorder=5)
    ax.annotate("v104 (fit)", (0, score["v104"]), xytext=(6, -9), textcoords="offset points", fontsize=7, color=INK2)
    ax.annotate("v110 (fit)", (1, score["v110"]), xytext=(-8, 4), textcoords="offset points", fontsize=7,
                color=INK2, ha="right")
    ax.annotate(f"vertex $k^*={kstar:.3f}$\npredicted {rmse(kstar):.4f} s", (kstar, rmse(kstar)),
                xytext=(-1.17, 291.55), fontsize=7, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.annotate(f"v112, $k=-0.45$ (check)\npredicted {rmse(-0.45):.4f} s\nserver {score['v112']:.4f} s",
                (-0.45, score["v112"]), xytext=(-0.25, 292.6), fontsize=7, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlabel(r"Step $k$ along $\delta=\hat{y}_\mathrm{v110}-\hat{y}_\mathrm{v104}$")
    ax.set_ylabel("Phase 1 public RMSE (s)")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(289.8, 296.0)
    ax.grid(axis="x", visible=True)
    save(fig, "parabolic_optimization")
    print(f"[parabola] q={q:.3f} c={c:.3f} k*={kstar:.4f} min={rmse(kstar):.4f} at -0.45: {rmse(-0.45):.4f}")


if __name__ == "__main__":
    print("font:", FAMILY)
    dep = load_departures()
    print_statistics(dep)
    fig_traffic(dep)
    fig_offset(dep)
    fig_importance()
    fig_parabola()
    print("figures written to", OUT)
