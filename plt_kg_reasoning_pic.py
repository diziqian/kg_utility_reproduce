#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fig4_compound_from_raw_v3.py

Generate a single Springer-style compound figure (Fig. 4) directly from raw STEP2/STEP3 CSVs.
Requirements implemented:
- Panels (a)(b): show scatter + dashed OLS fit only; in-panel text is minimized (Spearman rho and N only).
- Panels (c)(d): use their own shared y-limits based on mean ± 95% CI (not the full scatter y-range).
- Short panel titles.
Outputs:
- OUT_DIR/Fig4.png
- OUT_DIR/Fig4.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# 0) Global configuration
# ==========================================================
ZIP_PATH = "./result_kg_reproduce"
OUT_DIR = "./KG_reasoning_pic_4"
os.makedirs(OUT_DIR, exist_ok=True)

FILE_STEP2 = "STEP2_dataset_with_mechanisms.csv"
FILE_STEP3_PROD = "STEP3_product_mechanism_proxies.csv"
FILE_STEP3_COMP = "STEP3_product_competitor_summary.csv"

YCOL = "y_log10"
X_HEAT = "comp_expected_heat_total"
X_SRC_ENT = "comp_src_entropy"

FIGSIZE = (12, 9)
DPI = 300
Q = 5

POINT_ALPHA = 0.18
POINT_SIZE = 12

MARGIN_X = 0.05
MARGIN_Y_SCATTER = 0.08
MARGIN_Y_QUINT = 0.15

FIG_TITLE = "Competitor-neighborhood KG proxies and price gradients"  # Figure-level title

plt.rcParams["figure.dpi"] = DPI
plt.rcParams["savefig.dpi"] = DPI
plt.rcParams["axes.unicode_minus"] = False


# ==========================================================
# 1) Load data
# ==========================================================
def read_csv(path_: str) -> pd.DataFrame:
    if not os.path.exists(path_):
        raise FileNotFoundError(path_)
    return pd.read_csv(path_)

d2 = read_csv(os.path.join(ZIP_PATH, FILE_STEP2))
prod_mech = read_csv(os.path.join(ZIP_PATH, FILE_STEP3_PROD))
comp = read_csv(os.path.join(ZIP_PATH, FILE_STEP3_COMP))

# Build prod_key consistent with STEP3 files
d2["prod_key"] = d2.apply(
    lambda r: f"PROD::{str(r.get('name', '')).strip()}||SUP::{str(r.get('supplier', '')).strip()}",
    axis=1,
)

need_prod = [c for c in ["prod_key"] if c in prod_mech.columns]
need_comp = [c for c in ["prod_key", X_HEAT, X_SRC_ENT] if c in comp.columns]

df = d2.copy()
if need_prod:
    df = df.merge(prod_mech[need_prod], on="prod_key", how="left")
if need_comp:
    df = df.merge(comp[need_comp], on="prod_key", how="left")


# ==========================================================
# 2) Helper functions (no SciPy dependency)
# ==========================================================
def _finite_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def spearman_rho(x, y) -> float:
    """Spearman rho computed as Pearson correlation of ranks."""
    x, y = _finite_xy(x, y)
    if len(x) < 3:
        return np.nan
    rx = pd.Series(x).rank(method="average").values
    ry = pd.Series(y).rank(method="average").values
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

def quintile_stats(df_in: pd.DataFrame, xcol: str, ycol: str, q=5) -> pd.DataFrame:
    """Compute mean and 95% CI (±1.96*SE) of y within x-quantiles."""
    x = pd.to_numeric(df_in[xcol], errors="coerce")
    y = pd.to_numeric(df_in[ycol], errors="coerce")
    m = x.notna() & y.notna()
    x = x[m]
    y = y[m]
    if len(x) < q + 2:
        return pd.DataFrame()

    bins = pd.qcut(x, q, labels=False, duplicates="drop")
    tmp = pd.DataFrame({"q": bins, "y": y.values})
    stat = tmp.groupby("q", as_index=False).agg(mean=("y", "mean"), std=("y", "std"), n=("y", "size"))
    stat["se"] = stat["std"] / np.sqrt(stat["n"].clip(lower=1))
    stat["ci95"] = 1.96 * stat["se"]
    return stat

def compute_scatter_ylim(df_in: pd.DataFrame, ycol: str):
    y = pd.to_numeric(df_in.get(ycol, pd.Series(dtype=float)), errors="coerce").values
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return None
    y_min, y_max = float(np.min(y)), float(np.max(y))
    pad = (y_max - y_min) * MARGIN_Y_SCATTER if y_max > y_min else 0.5
    return (y_min - pad, y_max + pad)

def compute_quint_ylim(stats_list: list):
    """Shared y-limits for panels (c)(d) based on mean ± CI across both panels."""
    vals = []
    for st in stats_list:
        if st is None or st.empty:
            continue
        lo = (st["mean"] - st["ci95"]).astype(float).values
        hi = (st["mean"] + st["ci95"]).astype(float).values
        vals.append(np.nanmin(lo))
        vals.append(np.nanmax(hi))
    if not vals:
        return None
    y_min, y_max = float(np.min(vals)), float(np.max(vals))
    pad = (y_max - y_min) * MARGIN_Y_QUINT if y_max > y_min else 0.2
    return (y_min - pad, y_max + pad)


# ==========================================================
# 3) Pre-compute y-limits
# ==========================================================
YLIM_SCATTER = compute_scatter_ylim(df, YCOL)

stat_heat = quintile_stats(df, X_HEAT, YCOL, q=Q) if (X_HEAT in df.columns and YCOL in df.columns) else pd.DataFrame()
stat_ent  = quintile_stats(df, X_SRC_ENT, YCOL, q=Q) if (X_SRC_ENT in df.columns and YCOL in df.columns) else pd.DataFrame()
YLIM_QUINT = compute_quint_ylim([stat_heat, stat_ent])


# ==========================================================
# 4) Plot: 2x2 compound figure
# ==========================================================
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, constrained_layout=True)
ax_a, ax_b, ax_c, ax_d = axes.flatten()

# Figure-level title (kept concise; main explanation goes to caption in the paper)
fig.suptitle(FIG_TITLE, fontsize=16)

def plot_scatter_linear_mintext(ax, xcol, title, xlab, panel_tag):
    if xcol not in df.columns or YCOL not in df.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{panel_tag} Missing columns", ha="center", va="center", fontsize=12)
        return

    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[YCOL], errors="coerce")
    m = x.notna() & y.notna()
    x = x[m].values
    y = y[m].values

    if len(x) < 3:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{panel_tag} Not enough data", ha="center", va="center", fontsize=12)
        return

    ax.scatter(x, y, s=POINT_SIZE, alpha=POINT_ALPHA)

    # Dashed OLS fit line (for visualization only)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(np.min(x), np.max(x), 200)
    ys = slope * xs + intercept
    ax.plot(xs, ys, linewidth=2, linestyle="--")

    # Minimal in-panel stats: Spearman rho and N only
    rho = spearman_rho(x, y)
    ax.text(
        0.02, 0.06,
        f"Spearman ρ={rho:.3f}\nN={len(x)}",
        transform=ax.transAxes,
        ha="left", va="bottom", fontsize=10
    )

    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel("log10(price)")
    ax.margins(x=MARGIN_X, y=0.0)

    if YLIM_SCATTER is not None:
        ax.set_ylim(*YLIM_SCATTER)

    ax.text(0.02, 0.98, panel_tag, transform=ax.transAxes,
            ha="left", va="top", fontsize=14, fontweight="bold")


def plot_quintile_errorbar(ax, stat_df, title, panel_tag):
    if stat_df is None or stat_df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{panel_tag} Not enough data", ha="center", va="center", fontsize=12)
        return

    q_idx = stat_df["q"].astype(int).values
    mean = stat_df["mean"].astype(float).values
    ci = stat_df["ci95"].astype(float).values

    ax.errorbar(q_idx, mean, yerr=ci, fmt="o-", linewidth=2, capsize=4)
    ax.set_xticks(q_idx)
    ax.set_xlabel("Quintile (0=lowest, 4=highest)")
    ax.set_ylabel("Mean log10(price)")
    ax.set_title(title)
    ax.margins(x=MARGIN_X, y=0.0)

    # Shared y-limits for (c)(d)
    if YLIM_QUINT is not None:
        ax.set_ylim(*YLIM_QUINT)

    ax.text(0.02, 0.98, panel_tag, transform=ax.transAxes,
            ha="left", va="top", fontsize=14, fontweight="bold")


# Short panel titles (Springer-friendly)
plot_scatter_linear_mintext(
    ax_a,
    xcol=X_HEAT,
    title="Demand heat vs price",
    xlab="Competitor expected demand heat (KG-inferred)",
    panel_tag="(a)"
)

plot_scatter_linear_mintext(
    ax_b,
    xcol=X_SRC_ENT,
    title="Source entropy vs price",
    xlab="Competitor source entropy (KG-inferred)",
    panel_tag="(b)"
)

plot_quintile_errorbar(
    ax_c,
    stat_df=stat_heat,
    title="Mean price by heat quintile",
    panel_tag="(c)"
)

plot_quintile_errorbar(
    ax_d,
    stat_df=stat_ent,
    title="Mean price by entropy quintile",
    panel_tag="(d)"
)

try:
    fig.align_labels()
except Exception:
    pass


# ==========================================================
# 5) Save outputs (single Fig4 only)
# ==========================================================
out_png = os.path.join(OUT_DIR, "Fig4.png")
out_pdf = os.path.join(OUT_DIR, "Fig4.pdf")

fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(" -", out_png)
print(" -", out_pdf)