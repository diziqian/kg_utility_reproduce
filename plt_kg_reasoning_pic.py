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


def _num(s):
    return pd.to_numeric(s, errors='coerce')


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
    x = _num(df[xcol])
    y = _num(df[ycol])
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


def compute_ylim(series: pd.Series, pad_ratio=0.08):
    y = _num(series).dropna().values
    if len(y) == 0:
        return None
    y_min, y_max = float(np.min(y)), float(np.max(y))
    pad = (y_max - y_min) * pad_ratio if y_max > y_min else 0.5
    return (y_min - pad, y_max + pad)


def compute_quint_ylim(stats_list, pad_ratio=0.15):
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
    pad = (y_max - y_min) * pad_ratio if y_max > y_min else 0.2
    return (y_min - pad, y_max + pad)


def short_name(row: pd.Series) -> str:
    # Keep names compact for exported table; full names should be described in manuscript text.
    name = str(row.get('name', '')).strip()
    supplier = str(row.get('supplier', '')).strip()
    if name and supplier:
        return f'{name} | {supplier}'
    if name:
        return name
    if supplier:
        return supplier
    prod_key = str(row.get('prod_key', '')).strip()
    return prod_key if prod_key else 'unknown'


def pick_examples(df: pd.DataFrame, xcol: str, ycol: str, top_n=2, bottom_n=2) -> pd.DataFrame:
    tmp = df.copy()
    tmp[xcol] = _num(tmp[xcol])
    tmp[ycol] = _num(tmp[ycol])
    tmp = tmp[tmp[xcol].notna() & tmp[ycol].notna()].copy()
    if tmp.empty:
        return pd.DataFrame()

    # Avoid selecting only extreme y outliers as “representative” examples
    y_low, y_high = tmp[ycol].quantile([0.10, 0.90]).values
    core = tmp[(tmp[ycol] >= y_low) & (tmp[ycol] <= y_high)].copy()
    if len(core) < top_n + bottom_n:
        core = tmp.copy()

    low = core.nsmallest(bottom_n, xcol).copy()
    high = core.nlargest(top_n, xcol).copy()

    low = low.reset_index(drop=True)
    high = high.reset_index(drop=True)

    low['group'] = 'Low'
    high['group'] = 'High'
    low['rank'] = [f'L{i+1}' for i in range(len(low))]
    high['rank'] = [f'H{i+1}' for i in range(len(high))]

    out = pd.concat([low, high], axis=0, ignore_index=True)
    out['short_label'] = out.apply(short_name, axis=1)
    keep = [c for c in ['rank', 'group', 'short_label', 'name', 'supplier', 'prod_key', xcol, ycol] if c in out.columns]
    return out[keep].copy()


def add_binned_mean(ax, x: np.ndarray, y: np.ndarray, bins=10):
    if len(x) < max(20, bins * 2):
        return
    qs = np.linspace(0, 1, bins + 1)
    edges = np.quantile(x, qs)
    edges[0] = min(edges[0], x.min())
    edges[-1] = max(edges[-1], x.max())
    mids, means = [], []
    for i in range(len(edges) - 1):
        if i == len(edges) - 2:
            m = (x >= edges[i]) & (x <= edges[i + 1])
        else:
            m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() == 0:
            continue
        mids.append(np.mean([edges[i], edges[i + 1]]))
        means.append(np.mean(y[m]))
    if len(mids) >= 2:
        ax.plot(mids, means, marker='o', linewidth=1.8, markersize=3.5, label='Binned mean')


def add_example_markers(ax, examples: pd.DataFrame, xcol: str, ycol: str):
    if examples is None or examples.empty:
        return
    color_map = {'Low': 'tab:green', 'High': 'tab:red'}
    for _, r in examples.iterrows():
        x = float(r[xcol]); y = float(r[ycol])
        color = color_map.get(r['group'], 'black')
        ax.scatter([x], [y], s=48, color=color, edgecolor='black', linewidth=0.4, zorder=5)
        ax.annotate(
            r['rank'],
            xy=(x, y),
            xytext=(3, 3),
            textcoords='offset points',
            fontsize=8.5,
            fontweight='bold',
            color=color
        )


def add_example_box(ax, examples: pd.DataFrame, xcol: str, box_title: str):
    if examples is None or examples.empty:
        return
    lines = [box_title]
    for _, r in examples.iterrows():
        lines.append(f"{r['rank']}: {r['short_label']}")
    txt = "\n".join(lines)
    ax.text(
        0.985, 0.02, txt,
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=7.7,
        bbox=dict(boxstyle='round,pad=0.25', alpha=0.12)
    )


def export_example_tables(examples_heat: pd.DataFrame, examples_src: pd.DataFrame, out_dir: str) -> tuple[str, str]:
    heat_path = os.path.join(out_dir, 'Fig6_examples_heat.csv')
    src_path = os.path.join(out_dir, 'Fig6_examples_entropy.csv')
    examples_heat.to_csv(heat_path, index=False, encoding='utf-8-sig')
    examples_src.to_csv(src_path, index=False, encoding='utf-8-sig')
    return heat_path, src_path


def build_fig6_v3(out_dir: str, fig_dir: str | None = None) -> tuple[str, str, str, str]:
    if fig_dir is None:
        fig_dir = os.path.join(out_dir, 'KG_reasoning_pic_6_v3')
    os.makedirs(fig_dir, exist_ok=True)

    step0_path = os.path.join(out_dir, 'STEP0_dataset_with_demand_structure.csv')
    step2_path = os.path.join(out_dir, 'STEP2_dataset_with_mechanisms.csv')
    comp_path = os.path.join(out_dir, 'STEP3_product_competitor_summary.csv')

    if os.path.exists(step0_path):
        df = read_csv(step0_path)
    else:
        step2 = read_csv(step2_path)
        comp = read_csv(comp_path)
        if 'prod_key' not in step2.columns:
            step2['prod_key'] = step2.apply(
                lambda r: f"PROD::{str(r.get('name','')).strip()}||SUP::{str(r.get('supplier','')).strip()}",
                axis=1
            )
        keep = [c for c in ['prod_key', 'comp_expected_heat_total', 'comp_src_entropy'] if c in comp.columns]
        df = step2.merge(comp[keep], on='prod_key', how='left') if keep else step2.copy()

    if 'y_ln' not in df.columns and 'y_log10' in df.columns:
        df['y_ln'] = _num(df['y_log10']) * np.log(10.0)
    if 'name' not in df.columns:
        df['name'] = ''
    if 'supplier' not in df.columns:
        df['supplier'] = ''

    YCOL = 'y_ln'
    X_HEAT = 'comp_expected_heat_total'
    X_SRC = 'comp_src_entropy'

    y_lim_scatter = compute_ylim(df[YCOL], pad_ratio=0.08)
    stat_heat = quintile_stats(df, X_HEAT, YCOL, q=5) if X_HEAT in df.columns else pd.DataFrame()
    stat_src = quintile_stats(df, X_SRC, YCOL, q=5) if X_SRC in df.columns else pd.DataFrame()
    y_lim_quint = compute_quint_ylim([stat_heat, stat_src], pad_ratio=0.15)

    ex_heat = pick_examples(df, X_HEAT, YCOL, top_n=2, bottom_n=2) if X_HEAT in df.columns else pd.DataFrame()
    ex_src = pick_examples(df, X_SRC, YCOL, top_n=2, bottom_n=2) if X_SRC in df.columns else pd.DataFrame()
    heat_csv, src_csv = export_example_tables(ex_heat, ex_src, fig_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    def plot_density_scatter(ax, xcol, title, xlab, examples, box_title):
        if xcol not in df.columns or YCOL not in df.columns:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Missing columns', ha='center', va='center')
            return

        x = _num(df[xcol]); y = _num(df[YCOL])
        m = x.notna() & y.notna()
        x = x[m].values; y = y[m].values
        if len(x) < 3:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center')
            return

        hb = ax.hexbin(x, y, gridsize=28, mincnt=1, linewidths=0.15, cmap='Blues')
        cbar = fig.colorbar(hb, ax=ax, shrink=0.85)
        cbar.set_label('Count', fontsize=9)

        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(np.min(x), np.max(x), 200)
        ax.plot(xs, slope * xs + intercept, linestyle='--', linewidth=2.0, label='OLS trend')
        add_binned_mean(ax, x, y, bins=10)

        rho = spearman_rho(x, y)
        ax.text(
            0.02, 0.05,
            f'Spearman ρ={rho:.3f}\nN={len(x)}',
            transform=ax.transAxes,
            ha='left', va='bottom', fontsize=9.3,
            bbox=dict(boxstyle='round,pad=0.2', alpha=0.15)
        )

        add_example_markers(ax, examples, xcol, YCOL)
        add_example_box(ax, examples, xcol, box_title)

        # Dummy handles for low/high example markers
        ax.scatter([], [], s=48, color='tab:green', edgecolor='black', linewidth=0.4, label='Low example')
        ax.scatter([], [], s=48, color='tab:red', edgecolor='black', linewidth=0.4, label='High example')

        ax.set_title(title, fontsize=11.5, fontweight='bold')
        ax.set_xlabel(xlab, fontsize=10.5)
        ax.set_ylabel('ln(price)', fontsize=10.5)
        if y_lim_scatter is not None:
            ax.set_ylim(*y_lim_scatter)
        ax.legend(loc='upper left', fontsize=8.2, frameon=False)

    def plot_quint(ax, stat, title, xlab):
        if stat is None or stat.empty:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center')
            return
        x = stat['q'].astype(int).values
        y = stat['mean'].values.astype(float)
        e = stat['ci95'].values.astype(float)
        n = stat['n'].astype(int).values
        ax.errorbar(x, y, yerr=e, fmt='o-', capsize=4, linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in x])
        for xi, yi, ni in zip(x, y, n):
            ax.annotate(f'n={ni}', xy=(xi, yi), xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=8)
        ax.set_title(title, fontsize=11.5, fontweight='bold')
        ax.set_xlabel(xlab, fontsize=10.5)
        ax.set_ylabel('Mean ln(price)', fontsize=10.5)
        if y_lim_quint is not None:
            ax.set_ylim(*y_lim_quint)

    plot_density_scatter(
        ax_a, X_HEAT,
        '(a) Competitor-neighborhood demand heat vs ln(price)',
        'Competitor demand heat',
        ex_heat,
        'Demand-heat examples'
    )
    plot_density_scatter(
        ax_b, X_SRC,
        '(b) Source-side structural complexity vs ln(price)',
        'Source entropy',
        ex_src,
        'Entropy examples'
    )
    plot_quint(
        ax_c, stat_heat,
        '(c) Quintiles of competitor demand heat',
        'Demand heat quintile'
    )
    plot_quint(
        ax_d, stat_src,
        '(d) Quintiles of source-side structural complexity',
        'Source entropy quintile'
    )

    png = os.path.join(fig_dir, 'Fig6.png')
    pdf = os.path.join(fig_dir, 'Fig6.pdf')
    fig.savefig(png, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    return png, pdf, heat_csv, src_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='./result_kg_reproduce', help='directory containing STEP outputs')
    ap.add_argument('--fig_dir', default='./result_kg_reproduce/KG_reasoning_pic_6_v3', help='directory to save Fig6')
    args = ap.parse_args()
    png, pdf, heat_csv, src_csv = build_fig6_v3(args.out_dir, args.fig_dir)
    print(png)
    print(pdf)
    print(heat_csv)
    print(src_csv)


if __name__ == '__main__':
    main()
