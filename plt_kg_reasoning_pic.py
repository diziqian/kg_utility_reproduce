#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.unicode_minus'] = False


def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _finite_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def spearman_rho(x, y) -> float:
    x, y = _finite_xy(x, y)
    if len(x) < 3:
        return np.nan
    rx = pd.Series(x).rank(method='average').values
    ry = pd.Series(y).rank(method='average').values
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def quintile_stats(df: pd.DataFrame, xcol: str, ycol: str, q=5) -> pd.DataFrame:
    x = pd.to_numeric(df[xcol], errors='coerce')
    y = pd.to_numeric(df[ycol], errors='coerce')
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    if len(x) < q + 2:
        return pd.DataFrame(columns=['q', 'mean', 'std', 'n', 'se', 'ci95'])
    bins = pd.qcut(x, q, labels=False, duplicates='drop')
    tmp = pd.DataFrame({'q': bins, 'y': y.values})
    stat = tmp.groupby('q', as_index=False).agg(mean=('y', 'mean'), std=('y', 'std'), n=('y', 'size'))
    stat['se'] = stat['std'] / np.sqrt(stat['n'].clip(lower=1))
    stat['ci95'] = 1.96 * stat['se']
    return stat


def compute_scatter_ylim(df: pd.DataFrame, ycol: str):
    y = pd.to_numeric(df.get(ycol, pd.Series(dtype=float)), errors='coerce').values
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return None
    y_min, y_max = float(np.min(y)), float(np.max(y))
    pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.5
    return (y_min - pad, y_max + pad)


def compute_quint_ylim(stats_list):
    vals = []
    for st in stats_list:
        if st is None or st.empty:
            continue
        lo = (st['mean'] - st['ci95']).astype(float).values
        hi = (st['mean'] + st['ci95']).astype(float).values
        vals.append(np.nanmin(lo))
        vals.append(np.nanmax(hi))
    if not vals:
        return None
    y_min, y_max = float(np.min(vals)), float(np.max(vals))
    pad = (y_max - y_min) * 0.15 if y_max > y_min else 0.2
    return (y_min - pad, y_max + pad)


def build_fig6(out_dir: str, fig_dir: str | None = None) -> tuple[str, str]:
    if fig_dir is None:
        fig_dir = os.path.join(out_dir, 'KG_reasoning_pic_6')
    os.makedirs(fig_dir, exist_ok=True)

    # Prefer the stable verified output if available
    step0_path = os.path.join(out_dir, 'STEP0_dataset_with_demand_structure.csv')
    step2_path = os.path.join(out_dir, 'STEP2_dataset_with_mechanisms.csv')
    comp_path = os.path.join(out_dir, 'STEP3_product_competitor_summary.csv')

    if os.path.exists(step0_path):
        df = read_csv(step0_path)
    else:
        step2 = read_csv(step2_path)
        comp = read_csv(comp_path)
        if 'prod_key' not in step2.columns:
            step2['prod_key'] = step2.apply(lambda r: f"PROD::{str(r.get('name','')).strip()}||SUP::{str(r.get('supplier','')).strip()}", axis=1)
        keep = [c for c in ['prod_key', 'comp_expected_heat_total', 'comp_src_entropy'] if c in comp.columns]
        df = step2.merge(comp[keep], on='prod_key', how='left') if keep else step2.copy()

    # harmonize to ln columns
    if 'y_ln' not in df.columns and 'y_log10' in df.columns:
        df['y_ln'] = pd.to_numeric(df['y_log10'], errors='coerce') * np.log(10.0)
    YCOL = 'y_ln'
    X_HEAT = 'comp_expected_heat_total'
    X_SRC = 'comp_src_entropy'
    y_lim_scatter = compute_scatter_ylim(df, YCOL)
    stat_heat = quintile_stats(df, X_HEAT, YCOL, q=5) if X_HEAT in df.columns else pd.DataFrame()
    stat_src = quintile_stats(df, X_SRC, YCOL, q=5) if X_SRC in df.columns else pd.DataFrame()
    y_lim_quint = compute_quint_ylim([stat_heat, stat_src])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    def plot_scatter(ax, xcol, title, xlab, tag):
        if xcol not in df.columns or YCOL not in df.columns:
            ax.axis('off')
            ax.text(0.5, 0.5, f'{tag} Missing columns', ha='center', va='center')
            return
        x = pd.to_numeric(df[xcol], errors='coerce')
        y = pd.to_numeric(df[YCOL], errors='coerce')
        m = x.notna() & y.notna()
        x = x[m].values; y = y[m].values
        if len(x) < 3:
            ax.axis('off')
            ax.text(0.5, 0.5, f'{tag} Not enough data', ha='center', va='center')
            return
        ax.scatter(x, y, s=12, alpha=0.18)
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(np.min(x), np.max(x), 200)
        ax.plot(xs, slope * xs + intercept, linestyle='--', linewidth=2)
        rho = spearman_rho(x, y)
        ax.text(0.02, 0.06, f'Spearman ρ={rho:.3f}\nN={len(x)}', transform=ax.transAxes,
                ha='left', va='bottom', fontsize=10)
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel('ln(price)')
        ax.margins(x=0.05, y=0)
        if y_lim_scatter is not None:
            ax.set_ylim(*y_lim_scatter)

    def plot_quint(ax, stat, title, xlab, tag):
        if stat is None or stat.empty:
            ax.axis('off')
            ax.text(0.5, 0.5, f'{tag} Not enough data', ha='center', va='center')
            return
        x = stat['q'].astype(int).values
        y = stat['mean'].values.astype(float)
        e = stat['ci95'].values.astype(float)
        ax.errorbar(x, y, yerr=e, fmt='o-', capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel('Mean ln(price)')
        if y_lim_quint is not None:
            ax.set_ylim(*y_lim_quint)

    plot_scatter(ax_a, X_HEAT, '(a) Demand heat vs ln(price)', 'Competitor demand heat', '(a)')
    plot_scatter(ax_b, X_SRC, '(b) Source entropy vs ln(price)', 'Competitor source entropy', '(b)')
    plot_quint(ax_c, stat_heat, '(c) Quintiles: demand heat', 'Demand heat quintile', '(c)')
    plot_quint(ax_d, stat_src, '(d) Quintiles: source entropy', 'Source entropy quintile', '(d)')

    png = os.path.join(fig_dir, 'Fig6.png')
    pdf = os.path.join(fig_dir, 'Fig6.pdf')
    fig.savefig(png, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    return png, pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='./result_kg_reproduce', help='directory containing STEP outputs')
    ap.add_argument('--fig_dir', default=None, help='directory to save Fig6')
    ap.add_argument('--zip_path', default=None, help='legacy alias for out_dir; kept for backward compatibility')
    args = ap.parse_args()
    base_out = args.zip_path if args.zip_path else args.out_dir
    save_dir = args.fig_dir if args.fig_dir else args.out_dir
    png, pdf = build_fig6(base_out, save_dir)
    print(png)
    print(pdf)


if __name__ == '__main__':
    main()
