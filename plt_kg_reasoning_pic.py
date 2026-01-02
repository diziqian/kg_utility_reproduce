# Chart generation script
# Generate 4 paper figures in the target folder after running：
#  - fig4_price_competitor-side inferred demand (KG-inferred).png
#  - fig5_competitor-side source complexity (KG-inferred).png
#  - fig6_ mean_log10(price)_by_competitor_heat_quintile.png
#  - fig7_mean log10(price)_by_competitor_source_entropy_quintile.png

import zipfile
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

zip_path = "./result_kg_reproduce"
out_dir = "./KG_reasoning_pic_4"
os.makedirs(out_dir, exist_ok=True)


def read_csv(path_in_dir: str) -> pd.DataFrame:
    with open(path_in_dir) as f:
        return pd.read_csv(f)

def read_json(path_in_dir: str) -> dict:
    with open(path_in_dir) as f:
        return json.load(f)

# =========================
# 1) read data: STEP2 row-level data + STEP3 product mechanism & competitor summary
# =========================
d2 = read_csv(os.path.join(zip_path, "STEP2_dataset_with_mechanisms.csv"))
prod_mech = read_csv(os.path.join(zip_path, "STEP3_product_mechanism_proxies.csv"))
comp = read_csv(os.path.join(zip_path, "STEP3_product_competitor_summary.csv"))

# 构造 prod_key（与 STEP3 文件一致）
d2["prod_key"] = d2.apply(
    lambda r: f"PROD::{str(r['name']).strip()}||SUP::{str(r['supplier']).strip()}",
    axis=1,
)

# Merge required columns (automatically skip if some columns are missing)
need_prod = [
    "prod_key",
    "prod_ppr_expected_heat_total",
    "prod_ppr_expected_thickness",
    "prod_sim_pagerank",
    "prod_sim_deg",
    "prod_sim_strength",
    "prod_sim_clustering",
    "prod_sim_betweenness",
]
need_comp = [
    "prod_key",
    "comp_expected_heat_total",
    "comp_app_entropy",
    "comp_src_entropy",
    "comp_app_top1_prob",
    "comp_src_top1_prob",
]

need_prod = [c for c in need_prod if c in prod_mech.columns]
need_comp = [c for c in need_comp if c in comp.columns]

df = (
    d2.merge(prod_mech[need_prod], on="prod_key", how="left")
      .merge(comp[need_comp], on="prod_key", how="left")
)

# =========================
# 2) Helper functions: scatter plot + fitting line, quantile bar plot
# =========================
def scatter_with_fit(x, y, xlab, ylab, title, filename):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.6)

    # Univariate linear fitting (for trend visualization only)
    if len(x) > 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        ys = slope * xs + intercept
        plt.plot(xs, ys, linewidth=2)

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.show()

def quintile_means(df_in: pd.DataFrame, xcol: str, ycol: str = "y_log10", q: int = 5) -> pd.DataFrame:
    x = df_in[xcol].astype(float).values
    bins = pd.qcut(x, q, labels=False, duplicates="drop")
    tmp = df_in[[ycol]].copy()
    tmp["q"] = bins
    return tmp.groupby("q")[ycol].mean().reset_index()

def bar_quintile(means_df: pd.DataFrame, title: str, filename: str):
    plt.figure(figsize=(7, 4.5))
    plt.bar(means_df["q"].astype(int).values, means_df["y_log10"].astype(float).values)
    plt.xlabel("Quintile (0=lowest, 4=highest)")
    plt.ylabel("Mean log10(price)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.show()

# =========================
# 3) Fig.4-5: Price vs competitor side KG inference mechanism (supply-demand/structure)
# =========================
if "comp_expected_heat_total" in df.columns:
    scatter_with_fit(
        df["comp_expected_heat_total"], df["y_log10"],
        "Competitor expected demand heat (KG-inferred)",
        "log10(price)",
        "Price vs competitor-side inferred demand",
        "fig4_price_competitor-side inferred demand (KG-inferred).png",
    )

if "comp_src_entropy" in df.columns:
    scatter_with_fit(
        df["comp_src_entropy"], df["y_log10"],
        "Competitor source entropy (KG-inferred)",
        "log10(price)",
        "Price vs competitor-side source complexity",
        "fig5_competitor-side source complexity (KG-inferred).png",
    )

# =========================
# 4) Fig.6-7: Quantile mean (differences between upper/lower quintiles)
# =========================
if "comp_expected_heat_total" in df.columns:
    means_heat = quintile_means(df, "comp_expected_heat_total")
    bar_quintile(means_heat, "Mean log10(price) by competitor heat quintile", "fig6_ mean_log10(price)_by_competitor_heat_quintile.png")

if "comp_src_entropy" in df.columns:
    means_srcent = quintile_means(df, "comp_src_entropy")
    bar_quintile(means_srcent, "Mean log10(price) by competitor source entropy quintile", "fig7_mean log10(price)_by_competitor_source_entropy_quintile.png")


print("Saved figures to:", out_dir)
