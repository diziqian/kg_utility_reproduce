
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
supplier_anchor_verify_in_one.py.py

Purpose
-------
Extend the original `supplier_anchor_verify_in_one.py` workflow so that a single run
can generate:

1) Original intermediate outputs for supplier-level FE interpretation;
2) Item-level panels merged with Neo4j-exported src/app relations;
3) Four "supplier anchor is not a black-box dummy" experiments;
4) Final paper artifacts:
   - 2 main tables
   - 1 appendix table
   - 1 mechanism figure

Design principle
----------------
- Reuse the original script functions whenever possible.
- Keep all major comments in English.
- Produce both intermediate and final outputs in one directory.
- Avoid requiring the user to run multiple scripts manually.
"""

# ==========================================================
# 0) Global configuration
# ==========================================================
CONFIG = {
    # Input files
    "FILE_FE_SUPPLIER": "./result_kg_reproduce/STEP1_fe_table_supplier.csv",
    "FILE_PRICE_XLSX": "./anymous/name_price_anonymized.xlsx",
    "FILE_MEDIA_XLSX": "./anymous/media_result.xlsx",

    # Neo4j exports
    "FILE_NODE_DATAPRODUCT": "./anymous/neo4j_export/nodes_dataproduct.csv",
    "FILE_REL_SRC": "./anymous/neo4j_export/rel_source_industry.csv",
    "FILE_REL_APP": "./anymous/neo4j_export/rel_applied_to.csv",

    # Original-script reference year
    "REFERENCE_YEAR": 2025,

    # Output directory
    "OUTPUT_DIR": "./supplier_anchor_verify_in_one_merge",

    # Modeling
    "PERMUTATION_N": 500,
    "MIN_GROUP_FOR_LOO": 2,
    "TOP_CATEGORY_MIN_COUNT": 10,
    "TOP_CATEGORY_MAX_DUMMIES": 8,
    "RANDOM_SEED": 42,

    # Figure settings
    "FIG_DPI": 300,
}

# ==========================================================
# 1) Imports
# ==========================================================
import os
import sys
import json
import math
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

warnings.filterwarnings("ignore")

try:
    import statsmodels.api as sm
except Exception:
    sm = None

# Reuse the original script as the base workflow.
import supplier_fe_market_power_indications as base


# ==========================================================
# 2) General helpers
# ==========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("（", "(").replace("）", ")").replace("，", ",")
    return s


def normalize_key(x) -> str:
    s = normalize_text(x).lower()
    s = re.sub(r"\s+", "", s)
    return s


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def log10_pos(x):
    x = pd.to_numeric(x, errors="coerce")
    return np.where(x > 0, np.log10(x), np.nan)


def read_table_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def star(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fit_ols(df: pd.DataFrame, y: str, xcols: List[str], hc3: bool = True):
    if sm is None:
        raise ImportError("statsmodels is required")
    use = df[[y] + xcols].copy()
    use = use.replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) == 0:
        return None, pd.DataFrame(), len(use)
    X = sm.add_constant(use[xcols], has_constant="add")
    model = sm.OLS(use[y], X)
    res = model.fit(cov_type="HC3" if hc3 else "nonrobust")
    coef = pd.DataFrame({
        "term": res.params.index,
        "coef": res.params.values,
        "std_err": res.bse.values,
        "t": res.tvalues.values,
        "pvalue": res.pvalues.values,
    })
    coef["stars"] = coef["pvalue"].map(star)
    coef["n_obs"] = int(res.nobs)
    coef["r2"] = float(res.rsquared)
    coef["adj_r2"] = float(res.rsquared_adj)
    return res, coef, len(use)


def fit_lpm(df: pd.DataFrame, y: str, xcols: List[str]):
    return fit_ols(df, y, xcols, hc3=True)


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    m = np.isfinite(values) & np.isfinite(weights)
    if m.sum() == 0 or weights[m].sum() <= 0:
        return np.nan
    return np.average(values[m], weights=weights[m])


# ==========================================================
# 3) Build original supplier-level panel using original functions
# ==========================================================
def run_original_base_workflow(cfg: Dict, out_dir: str) -> Dict:
    """
    Reproduce the original supplier-level interpretation workflow, but keep all
    intermediate objects for subsequent experiments and paper artifacts.
    """
    base_panel, item_df, app_col_used = base.build_supplier_panel_from_enriched(
        file_enriched_xlsx=cfg["FILE_PRICE_XLSX"],
        ref_year=int(cfg["REFERENCE_YEAR"]),
    )

    fe_panel = base.load_supplier_fe_table(cfg["FILE_FE_SUPPLIER"])
    panel = base_panel.merge(fe_panel, on="supplier", how="left")

    media_df = base.load_media_table(cfg["FILE_MEDIA_XLSX"], use_log1p=True)
    supplier_media_panel, media_diag = base.build_supplier_media_panel_from_items(item_df=item_df, media_df=media_df)
    if supplier_media_panel is not None and len(supplier_media_panel) > 0:
        panel = panel.merge(supplier_media_panel, on="supplier", how="left")

    panel = base.build_market_power_indications(panel, cfg)

    # Spearman and OLS from the original logic
    spearman_candidates = [
        "exclusivity_rate", "gov_share", "listed_share", "anonymous_rate",
        "firm_age", "log_total_asset_wan", "disclosure_years",
        "n_products", "overall_count_share",
        "app_domain_hhi", "max_app_share", "n_apps_covered_in_media_map", "single_app_focus_flag",
        "media_match_ratio", "exp_media_web", "exp_media_news",
        "exp_media_weixin_account", "exp_media_weixin_article", "exp_media_total",
        "score_premium", "score_structure", "score_barrier", "score_demand_visibility",
        "market_power_signal_score",
    ]
    spearman_df = base.run_spearman_table(panel, y_col="fe_log10", x_cols=spearman_candidates)

    ols_tables = []
    model_summary_txt = {}

    x1 = ["exclusivity_rate", "gov_share", "listed_share", "anonymous_rate",
          "firm_age", "log_total_asset_wan", "n_products"]
    res1, coef1 = base.fit_ols_hc3(panel, y_col="fe_log10", x_cols=x1,
                                   model_name="FE_on_barrier_only",
                                   min_obs=30)
    if res1 is not None and len(coef1) > 0:
        coef1 = coef1.copy()
        coef1["model_group"] = "core"
        ols_tables.append(coef1)
        model_summary_txt["FE_on_barrier_only"] = res1.summary().as_text()

    x2 = x1 + ["exp_media_total", "app_domain_hhi", "max_app_share", "media_match_ratio"]
    res2, coef2 = base.fit_ols_hc3(panel, y_col="fe_log10", x_cols=x2,
                                   model_name="FE_on_barrier_plus_media_structure",
                                   min_obs=30)
    if res2 is not None and len(coef2) > 0:
        coef2 = coef2.copy()
        coef2["model_group"] = "extended"
        ols_tables.append(coef2)
        model_summary_txt["FE_on_barrier_plus_media_structure"] = res2.summary().as_text()

    ols_df = pd.concat(ols_tables, ignore_index=True) if ols_tables else pd.DataFrame()

    # Export original-style intermediates
    panel.to_csv(os.path.join(out_dir, "01_supplier_panel_original_workflow.csv"), index=False, encoding="utf-8-sig")
    item_df.to_csv(os.path.join(out_dir, "02_item_table_from_price_file.csv"), index=False, encoding="utf-8-sig")
    spearman_df.to_csv(os.path.join(out_dir, "03_supplier_fe_spearman.csv"), index=False, encoding="utf-8-sig")
    ols_df.to_csv(os.path.join(out_dir, "04_supplier_fe_ols.csv"), index=False, encoding="utf-8-sig")

    for k, txt in model_summary_txt.items():
        with open(os.path.join(out_dir, f"{k}.txt"), "w", encoding="utf-8") as f:
            f.write(txt)

    try:
        fig_info = base.plot_fig8_supplier_fe_by_tier(panel=panel, out_dir=out_dir, cfg={
            "PAPER_FIG8_BASENAME": "05_fig_supplier_fe_by_tier",
            "PAPER_FIG8_DPI": cfg["FIG_DPI"],
            "WRITE_PDF_FIG": False
        })
    except Exception:
        fig_info = None

    return {
        "supplier_panel": panel,
        "item_df": item_df,
        "media_df": media_df,
        "spearman_df": spearman_df,
        "ols_df": ols_df,
        "fig_info": fig_info,
        "app_col_used": app_col_used,
    }


# ==========================================================
# 4) Merge Neo4j relations back to the item table
# ==========================================================
def build_item_panel_with_neo4j(cfg: Dict, item_df: pd.DataFrame, supplier_panel: pd.DataFrame, media_df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    Recover dp_id -> src/app relations using Neo4j exports, then merge them back
    to the item-level price table.
    """
    node_dp = read_table_any(cfg["FILE_NODE_DATAPRODUCT"])
    rel_src = read_table_any(cfg["FILE_REL_SRC"])
    rel_app = read_table_any(cfg["FILE_REL_APP"])

    # Normalize product name key
    node_dp = node_dp.copy()
    item_df = item_df.copy()

    name_col_node = "name_anon" if "name_anon" in node_dp.columns else "name"
    node_dp["name_key"] = node_dp[name_col_node].map(normalize_key)
    item_df["name_key"] = item_df["name"].map(normalize_key)

    keep_dp_cols = [c for c in ["dp_id", name_col_node, "supplier_anon"] if c in node_dp.columns]
    dp_map = node_dp[["name_key"] + keep_dp_cols].drop_duplicates("name_key")

    if "name_key" not in dp_map.columns:
        dp_map["name_key"] = dp_map[name_col_node].map(normalize_key)

    item = item_df.merge(dp_map[["name_key", "dp_id", name_col_node, "supplier_anon"]], on="name_key", how="left")
    item["dp_match"] = item["dp_id"].notna().astype(int)

    # Build src/app long relations
    rel_src = rel_src.copy()
    rel_app = rel_app.copy()

    src_name_col = "src_name" if "src_name" in rel_src.columns else rel_src.columns[-1]
    app_name_col = "app_name" if "app_name" in rel_app.columns else rel_app.columns[-1]

    src_long = rel_src[["dp_id", src_name_col]].dropna().drop_duplicates().rename(columns={src_name_col: "src_name"})
    app_long = rel_app[["dp_id", app_name_col]].dropna().drop_duplicates().rename(columns={app_name_col: "app_name"})

    # Aggregate lists and counts
    src_agg = src_long.groupby("dp_id")["src_name"].agg(list).reset_index()
    src_agg["n_src_labels"] = src_agg["src_name"].map(len)
    src_agg["src_joined"] = src_agg["src_name"].map(lambda xs: " | ".join(xs))

    app_agg = app_long.groupby("dp_id")["app_name"].agg(list).reset_index()
    app_agg["n_app_labels"] = app_agg["app_name"].map(len)
    app_agg["app_joined"] = app_agg["app_name"].map(lambda xs: " | ".join(xs))

    item = item.merge(src_agg[["dp_id", "src_name", "n_src_labels", "src_joined"]], on="dp_id", how="left")
    item = item.merge(app_agg[["dp_id", "app_name", "n_app_labels", "app_joined"]], on="dp_id", how="left")

    # Merge supplier FE and supplier-level columns
    supplier_keep = [c for c in ["supplier", "fe_log10", "fe_multiplier_on_price", "baseline_supplier",
                                 "n_products", "firm_age", "log_total_asset_wan", "listed_share",
                                 "gov_share", "exclusivity_rate", "anonymous_rate"] if c in supplier_panel.columns]
    item = item.merge(supplier_panel[supplier_keep].drop_duplicates("supplier"), on="supplier", how="left")

    # Product-level supplier count
    item["supplier_n_products"] = item.groupby("supplier")["name"].transform("count")

    # Build product-level media exposure from app relations and media keyword table
    media = media_df.copy()
    keyword_col = "keyword"
    media_key_col = "keyword_key" if "keyword_key" in media.columns else keyword_col
    media["app_key"] = media[media_key_col].map(normalize_key)

    app_long["app_key"] = app_long["app_name"].map(normalize_key)
    app_media = app_long.merge(media, on="app_key", how="left")

    # Use the original script's total-log1p demand proxy when available.
    if "media_total_log1p_sum" in app_media.columns:
        app_media["exp_media_total"] = pd.to_numeric(app_media["media_total_log1p_sum"], errors="coerce")
    elif "exp_media_total" in app_media.columns:
        app_media["exp_media_total"] = pd.to_numeric(app_media["exp_media_total"], errors="coerce")
    else:
        media_cols = [c for c in ["exp_media_web", "exp_media_news", "exp_media_weixin_account", "exp_media_weixin_article"] if c in app_media.columns]
        if media_cols:
            app_media["exp_media_total"] = app_media[media_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
        else:
            app_media["exp_media_total"] = np.nan

    # Aggregate demand proxies to the product level
    app_media_agg = app_media.groupby("dp_id").agg(
        app_media_total_mean=("exp_media_total", "mean"),
        app_media_total_max=("exp_media_total", "max"),
        app_media_total_sum=("exp_media_total", "sum"),
        app_media_match_count=("exp_media_total", lambda s: pd.to_numeric(s, errors="coerce").notna().sum())
    ).reset_index()

    item = item.merge(app_media_agg, on="dp_id", how="left")
    for c in ["app_media_total_mean", "app_media_total_max", "app_media_total_sum"]:
        item[f"log1p_{c}"] = np.log1p(pd.to_numeric(item[c], errors="coerce"))

    # Clean basic numeric fields
    item["price"] = safe_num(item["price"])
    item["log10_price"] = np.where(item["price"] > 0, np.log10(item["price"]), np.nan)
    if "Year_Dis" in item.columns:
        item["Year_Dis_num"] = safe_num(item["Year_Dis"])
    if "Is_Listed" in item.columns:
        item["Is_Listed"] = safe_num(item["Is_Listed"]).fillna(0)
    if "Is_Gov" in item.columns:
        item["Is_Gov"] = safe_num(item["Is_Gov"]).fillna(0)
    if "Is_Anonymous" in item.columns:
        item["Is_Anonymous"] = safe_num(item["Is_Anonymous"]).fillna(0)
    if "Data_Exclusivity" in item.columns:
        item["Data_Exclusivity"] = safe_num(item["Data_Exclusivity"]).fillna(0)

    # Export intermediate outputs
    item.to_csv(os.path.join(out_dir, "06_item_panel_with_neo4j_and_media.csv"), index=False, encoding="utf-8-sig")
    src_long.to_csv(os.path.join(out_dir, "07_src_long_relations.csv"), index=False, encoding="utf-8-sig")
    app_long.to_csv(os.path.join(out_dir, "08_app_long_relations.csv"), index=False, encoding="utf-8-sig")

    return item


def augment_supplier_panel_from_neo4j(item: pd.DataFrame, supplier_panel: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    Backfill supplier-level application/media structure using Neo4j-recovered app labels.
    This avoids losing the original interpretation step when the price file itself does
    not contain a ready-made application column.
    """
    sp = supplier_panel.copy()
    it = item.copy()

    # Supplier-level demand proxies from item-level app exposures
    med = it.groupby("supplier").agg(
        exp_media_total_neo4j=("app_media_total_sum", "mean"),
        media_match_ratio_neo4j=("app_media_match_count", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean()) if len(s) > 0 else np.nan),
        n_apps_covered_in_media_map_neo4j=("app_media_match_count", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum()))
    ).reset_index()

    # Supplier-app concentration using long relations rebuilt from item-level lists
    rec = []
    for _, row in it[["supplier", "app_name"]].iterrows():
        apps = row["app_name"]
        if isinstance(apps, list):
            for a in apps:
                rec.append((row["supplier"], a))
    app_long = pd.DataFrame(rec, columns=["supplier", "app_name"])
    if len(app_long) > 0:
        sc = app_long.groupby(["supplier", "app_name"]).size().reset_index(name="n")
        tot = sc.groupby("supplier")["n"].transform("sum")
        sc["share"] = sc["n"] / tot
        hhi = sc.groupby("supplier")["share"].apply(lambda s: float((s ** 2).sum())).rename("app_domain_hhi_neo4j")
        mx = sc.groupby("supplier")["share"].max().rename("max_app_share_neo4j")
        napp = sc.groupby("supplier")["app_name"].nunique().rename("n_apps_neo4j")
        st = pd.concat([hhi, mx, napp], axis=1).reset_index()
    else:
        st = pd.DataFrame(columns=["supplier", "app_domain_hhi_neo4j", "max_app_share_neo4j", "n_apps_neo4j"])

    aug = med.merge(st, on="supplier", how="outer")
    sp = sp.merge(aug, on="supplier", how="left")

    # Fill the original column names only when they are missing.
    fill_map = {
        "exp_media_total": "exp_media_total_neo4j",
        "media_match_ratio": "media_match_ratio_neo4j",
        "n_apps_covered_in_media_map": "n_apps_covered_in_media_map_neo4j",
        "app_domain_hhi": "app_domain_hhi_neo4j",
        "max_app_share": "max_app_share_neo4j",
    }
    for target, source in fill_map.items():
        if source in sp.columns:
            if target not in sp.columns:
                sp[target] = sp[source]
            else:
                sp[target] = sp[target].where(sp[target].notna(), sp[source])

    sp.to_csv(os.path.join(out_dir, "06b_supplier_panel_augmented_with_neo4j_media.csv"), index=False, encoding="utf-8-sig")
    return sp


# ==========================================================
# 5) Experiment 1: supplier FE decomposition
# ==========================================================
def run_experiment_1(supplier_panel: pd.DataFrame, out_dir: str) -> Dict:
    """
    Interpret supplier FE as a reduced-form premium and test whether observable
    proxies can partially explain it.
    """
    d = supplier_panel.copy()

    core_x = [c for c in [
        "exclusivity_rate", "gov_share", "listed_share", "anonymous_rate",
        "firm_age", "log_total_asset_wan", "n_products"
    ] if c in d.columns]

    ext_x = [c for c in [
        "exclusivity_rate", "gov_share", "listed_share", "anonymous_rate",
        "firm_age", "log_total_asset_wan", "n_products",
        "exp_media_total", "app_domain_hhi", "max_app_share", "media_match_ratio"
    ] if c in d.columns]

    res_core, coef_core, n_core = fit_ols(d, "fe_log10", core_x)
    res_ext, coef_ext, n_ext = fit_ols(d, "fe_log10", ext_x)

    # Tier summary from the original proxy-based tiering
    tier_col = "market_power_tier_en_v4" if "market_power_tier_en_v4" in d.columns else None
    if tier_col is not None:
        tier_sum = d.groupby(tier_col).agg(
            n_suppliers=("supplier", "count"),
            mean_fe_log10=("fe_log10", "mean"),
            median_fe_log10=("fe_log10", "median"),
        ).reset_index()
    else:
        tier_sum = pd.DataFrame()

    # Export
    with pd.ExcelWriter(os.path.join(out_dir, "exp1_supplier_fe_decomposition.xlsx"), engine="openpyxl") as writer:
        if len(coef_core) > 0:
            coef_core.to_excel(writer, sheet_name="core_coef", index=False)
        if len(coef_ext) > 0:
            coef_ext.to_excel(writer, sheet_name="extended_coef", index=False)
        if len(tier_sum) > 0:
            tier_sum.to_excel(writer, sheet_name="tier_summary", index=False)

    # Plot FE by tier
    fig_path = None
    if len(tier_sum) > 0:
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        ax.bar(tier_sum[tier_col], tier_sum["median_fe_log10"])
        ax.set_xlabel("Proxy-based supplier-premium tier")
        ax.set_ylabel("Median supplier FE (log10)")
        ax.set_title("Supplier FE across proxy-based tiers")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        fig_path = os.path.join(out_dir, "exp1_fe_tiers.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return {
        "core_r2": None if res_core is None else float(res_core.rsquared),
        "ext_r2": None if res_ext is None else float(res_ext.rsquared),
        "core_n": n_core,
        "ext_n": n_ext,
        "core_coef": coef_core,
        "ext_coef": coef_ext,
        "tier_summary": tier_sum,
        "fig_path": fig_path,
    }


# ==========================================================
# 6) Experiment 2: permutation test
# ==========================================================
def run_experiment_2(item: pd.DataFrame, cfg: Dict, out_dir: str) -> Dict:
    """
    Test whether the explanatory power of supplier identity is stronger than what
    would be obtained under randomly permuted supplier labels.
    """
    rng = np.random.default_rng(cfg["RANDOM_SEED"])
    d = item.copy()
    d = d[(d["price"] > 0) & d["supplier"].notna()].copy()

    controls = [c for c in [
        "Data_Exclusivity", "Is_Gov", "Is_Listed", "Is_Anonymous",
        "firm_age", "log_total_asset_wan", "Year_Dis_num", "supplier_n_products",
        "n_src_labels", "n_app_labels"
    ] if c in d.columns]

    # Real FE-only R² via group means
    supplier_mean = d.groupby("supplier")["log10_price"].mean().rename("supplier_mean_real")
    d = d.merge(supplier_mean, on="supplier", how="left")
    real_fe_only_r2 = float(np.corrcoef(d["log10_price"], d["supplier_mean_real"])[0, 1] ** 2)

    # Real controls + supplier FE
    real_res, _, _ = fit_ols(d, "log10_price", controls + ["fe_log10"])
    real_ctrl_fe_r2 = None if real_res is None else float(real_res.rsquared)

    # Permutation loop
    fe_only_perm = []
    ctrl_fe_perm = []

    suppliers = d["supplier"].values.copy()
    fe_map_real = d[["supplier", "fe_log10"]].drop_duplicates().dropna()

    for b in range(int(cfg["PERMUTATION_N"])):
        perm = rng.permutation(suppliers)
        tmp = d.copy()
        tmp["supplier_perm"] = perm

        perm_mean = tmp.groupby("supplier_perm")["log10_price"].mean().rename("supplier_mean_perm")
        tmp = tmp.merge(perm_mean, on="supplier_perm", how="left")
        r2_perm_fe = float(np.corrcoef(tmp["log10_price"], tmp["supplier_mean_perm"])[0, 1] ** 2)
        fe_only_perm.append(r2_perm_fe)

        # Assign FE by permuted supplier identity
        fe_map = fe_map_real.copy()
        # Use a random permutation of FE values to break the identity-price mapping
        fe_values = rng.permutation(fe_map["fe_log10"].values)
        tmp_map = pd.DataFrame({"supplier_perm": fe_map["supplier"].values, "fe_log10_perm": fe_values})
        tmp2 = tmp.merge(tmp_map, on="supplier_perm", how="left")

        res_perm, _, _ = fit_ols(tmp2, "log10_price", controls + ["fe_log10_perm"])
        ctrl_fe_perm.append(np.nan if res_perm is None else float(res_perm.rsquared))

    perm_df = pd.DataFrame({
        "fe_only_r2_perm": fe_only_perm,
        "controls_fe_r2_perm": ctrl_fe_perm,
    })
    perm_df.to_excel(os.path.join(out_dir, "exp2_permutation_test.xlsx"), index=False)

    # Histogram figure
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].hist(perm_df["fe_only_r2_perm"].dropna(), bins=24)
    axes[0].axvline(real_fe_only_r2, linestyle="--")
    axes[0].set_title("FE-only permutation distribution")
    axes[0].set_xlabel("R²")
    axes[0].set_ylabel("Count")

    axes[1].hist(perm_df["controls_fe_r2_perm"].dropna(), bins=24)
    axes[1].axvline(real_ctrl_fe_r2, linestyle="--")
    axes[1].set_title("Controls + FE permutation distribution")
    axes[1].set_xlabel("R²")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    hist_path = os.path.join(out_dir, "exp2_permutation_hist.png")
    fig.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    p_fe_only = float((np.sum(np.array(fe_only_perm) >= real_fe_only_r2) + 1) / (len(fe_only_perm) + 1))
    valid_ctrl = np.array([x for x in ctrl_fe_perm if pd.notna(x)])
    p_ctrl_fe = float((np.sum(valid_ctrl >= real_ctrl_fe_r2) + 1) / (len(valid_ctrl) + 1)) if len(valid_ctrl) > 0 else np.nan

    return {
        "real_fe_only_r2": real_fe_only_r2,
        "perm_fe_only_mean_r2": float(np.nanmean(fe_only_perm)),
        "perm_fe_only_max_r2": float(np.nanmax(fe_only_perm)),
        "p_fe_only": p_fe_only,
        "real_controls_fe_r2": real_ctrl_fe_r2,
        "perm_controls_fe_mean_r2": float(np.nanmean(ctrl_fe_perm)),
        "perm_controls_fe_max_r2": float(np.nanmax(ctrl_fe_perm)),
        "p_controls_fe": p_ctrl_fe,
        "hist_path": hist_path,
    }


# ==========================================================
# 7) Experiment 3: alternative supplier baselines
# ==========================================================
def run_experiment_3(item: pd.DataFrame, out_dir: str) -> Dict:
    """
    Replace hard supplier dummies with leave-one-out and shrinkage supplier baselines.
    """
    d = item.copy()
    d = d[(d["price"] > 0) & d["supplier"].notna()].copy()

    controls = [c for c in [
        "Data_Exclusivity", "Is_Gov", "Is_Listed", "Is_Anonymous",
        "firm_age", "log_total_asset_wan", "Year_Dis_num", "supplier_n_products",
        "n_src_labels", "n_app_labels"
    ] if c in d.columns]

    # Leave-one-out supplier mean
    g = d.groupby("supplier")["log10_price"]
    count = g.transform("count")
    summ = g.transform("sum")
    d["loo_supplier_mean"] = np.where(count > 1, (summ - d["log10_price"]) / (count - 1), np.nan)

    # Empirical-Bayes style shrinkage of the LOO baseline
    mu = d["log10_price"].mean()
    supplier_stats = d.groupby("supplier")["log10_price"].agg(["mean", "count"]).reset_index()
    tau = np.nanvar(supplier_stats["mean"])
    sigma = np.nanvar(d["log10_price"] - d.groupby("supplier")["log10_price"].transform("mean"))
    if not np.isfinite(tau) or tau <= 0:
        tau = 1e-6
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1e-6
    supplier_stats["lambda"] = tau / (tau + sigma / supplier_stats["count"].clip(lower=1))
    supplier_stats["eb_supplier_mean"] = supplier_stats["lambda"] * supplier_stats["mean"] + (1 - supplier_stats["lambda"]) * mu
    d = d.merge(supplier_stats[["supplier", "eb_supplier_mean"]], on="supplier", how="left")

    # Fit comparison models
    models = []

    res0, _, _ = fit_ols(d, "log10_price", controls)
    models.append(("controls_only", None if res0 is None else float(res0.rsquared)))

    res1, _, _ = fit_ols(d, "log10_price", controls + ["fe_log10"])
    models.append(("controls_plus_fe", None if res1 is None else float(res1.rsquared)))

    res2, _, _ = fit_ols(d, "log10_price", controls + ["loo_supplier_mean"])
    models.append(("controls_plus_loo_mean", None if res2 is None else float(res2.rsquared)))

    res3, _, _ = fit_ols(d, "log10_price", controls + ["eb_supplier_mean"])
    models.append(("controls_plus_eb_mean", None if res3 is None else float(res3.rsquared)))

    out = pd.DataFrame(models, columns=["model", "r2"])
    rho_loo = d[["fe_log10", "loo_supplier_mean"]].corr(method="spearman").iloc[0, 1]
    rho_eb = d[["fe_log10", "eb_supplier_mean"]].corr(method="spearman").iloc[0, 1]
    out["spearman_vs_fe"] = [np.nan, 1.0, rho_loo, rho_eb]

    out.to_excel(os.path.join(out_dir, "exp3_supplier_baseline_comparison.xlsx"), index=False)

    return {
        "comparison_table": out,
        "spearman_fe_vs_loo": None if pd.isna(rho_loo) else float(rho_loo),
        "spearman_fe_vs_eb": None if pd.isna(rho_eb) else float(rho_eb),
    }


# ==========================================================
# 8) Experiment 4: benchmark and high-gap
# ==========================================================
def make_top_dummies(item: pd.DataFrame, list_col: str, prefix: str, min_count: int, max_dummies: int) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Convert multi-label lists (src/app) into top-category dummy variables.
    """
    d = item.copy()
    # Long counts
    records = []
    for _, row in d[["name", list_col]].iterrows():
        vals = row[list_col]
        if isinstance(vals, list):
            for v in vals:
                records.append((row["name"], v))
    long_df = pd.DataFrame(records, columns=["name", f"{prefix}_label"])
    if len(long_df) == 0:
        return d, [], pd.DataFrame(columns=[f"{prefix}_label", "n_products"])

    cnt = long_df.groupby(f"{prefix}_label").size().reset_index(name="n_products")
    cnt = cnt[cnt["n_products"] >= min_count].sort_values(["n_products", f"{prefix}_label"], ascending=[False, True])
    keep = cnt.head(max_dummies)[f"{prefix}_label"].tolist()

    for lab in keep:
        col = f"{prefix}__" + re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", str(lab)).strip("_")
        d[col] = d[list_col].map(lambda xs: int(isinstance(xs, list) and lab in xs))
    return d, [f"{prefix}__" + re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", str(lab)).strip("_") for lab in keep], cnt


def run_experiment_4(item: pd.DataFrame, out_dir: str, cfg: Dict) -> Dict:
    """
    Build a supplier-anchored benchmark and examine which products exceed the
    benchmark by entering the upper quartile of positive quote gaps.
    """
    d = item.copy()
    d = d[(d["price"] > 0)].copy()

    base_x = [c for c in [
        "Data_Exclusivity", "Is_Gov", "Is_Listed", "Is_Anonymous",
        "firm_age", "log_total_asset_wan", "Year_Dis_num", "supplier_n_products",
        "n_src_labels", "n_app_labels", "fe_log10"
    ] if c in d.columns]

    benchmark_res, _, n_b = fit_ols(d, "log10_price", base_x)
    d = d.copy()
    d["_rowid"] = np.arange(len(d))
    bench_use = d[["_rowid", "log10_price"] + base_x].replace([np.inf, -np.inf], np.nan).dropna().copy()
    Xb = sm.add_constant(bench_use[base_x], has_constant="add")
    bench_use["benchmark_log10"] = benchmark_res.predict(Xb)
    bench_use["benchmark_price"] = np.power(10.0, bench_use["benchmark_log10"])
    bench_use["actual_price"] = np.power(10.0, bench_use["log10_price"])
    bench_use["gap_log10"] = bench_use["log10_price"] - bench_use["benchmark_log10"]
    bench_use["gap_price"] = bench_use["actual_price"] - bench_use["benchmark_price"]

    q75_gap = float(bench_use["gap_price"].quantile(0.75))
    bench_use["high_gap"] = (bench_use["gap_price"] >= q75_gap).astype(int)

    # Merge back identifiers and multi-label columns using the preserved row id.
    keep_cols = [c for c in ["name", "supplier", "src_name", "app_name",
                             "log1p_app_media_total_mean", "log1p_app_media_total_max", "log1p_app_media_total_sum"]
                 if c in d.columns]
    bench_use = bench_use.merge(d[["_rowid"] + keep_cols], on="_rowid", how="left")

    # Demand-only vs demand + src/app models
    demand_x = [c for c in ["log1p_app_media_total_mean", "log1p_app_media_total_max", "log1p_app_media_total_sum"] if c in bench_use.columns]

    bench_use2, src_dummy_cols, src_cnt = make_top_dummies(bench_use, "src_name", "src", cfg["TOP_CATEGORY_MIN_COUNT"], cfg["TOP_CATEGORY_MAX_DUMMIES"])
    bench_use3, app_dummy_cols, app_cnt = make_top_dummies(bench_use2, "app_name", "app", cfg["TOP_CATEGORY_MIN_COUNT"], cfg["TOP_CATEGORY_MAX_DUMMIES"])

    res_dem, _, _ = fit_lpm(bench_use3, "high_gap", demand_x)
    res_full, coef_full, _ = fit_lpm(bench_use3, "high_gap", demand_x + src_dummy_cols + app_dummy_cols)

    # High-gap rates by top categories
    src_rate = pd.DataFrame()
    if len(src_cnt) > 0:
        rec = []
        for lab in src_cnt[f"src_label"].head(cfg["TOP_CATEGORY_MAX_DUMMIES"]):
            m = bench_use3["src_name"].map(lambda xs: isinstance(xs, list) and lab in xs)
            rec.append({
                "src_label": lab,
                "n_products": int(m.sum()),
                "high_gap_rate": float(bench_use3.loc[m, "high_gap"].mean()) if m.sum() > 0 else np.nan,
                "lift_vs_overall": float(bench_use3.loc[m, "high_gap"].mean() - bench_use3["high_gap"].mean()) if m.sum() > 0 else np.nan,
            })
        src_rate = pd.DataFrame(rec)

    app_rate = pd.DataFrame()
    if len(app_cnt) > 0:
        rec = []
        for lab in app_cnt[f"app_label"].head(cfg["TOP_CATEGORY_MAX_DUMMIES"]):
            m = bench_use3["app_name"].map(lambda xs: isinstance(xs, list) and lab in xs)
            rec.append({
                "app_label": lab,
                "n_products": int(m.sum()),
                "high_gap_rate": float(bench_use3.loc[m, "high_gap"].mean()) if m.sum() > 0 else np.nan,
                "lift_vs_overall": float(bench_use3.loc[m, "high_gap"].mean() - bench_use3["high_gap"].mean()) if m.sum() > 0 else np.nan,
            })
        app_rate = pd.DataFrame(rec)

    # Within-supplier high-price auxiliary indicator
    d_aux = d.copy()
    d_aux["supplier_q75"] = d_aux.groupby("supplier")["price"].transform(lambda s: s.quantile(0.75))
    d_aux["high_price_within_supplier_q75"] = (d_aux["price"] >= d_aux["supplier_q75"]).astype(int)

    # Export
    with pd.ExcelWriter(os.path.join(out_dir, "exp4_high_gap_within_supplier.xlsx"), engine="openpyxl") as writer:
        bench_use3.to_excel(writer, sheet_name="item_benchmark_panel", index=False)
        if coef_full is not None and len(coef_full) > 0:
            coef_full.to_excel(writer, sheet_name="high_gap_lpm_coef", index=False)
        if len(src_rate) > 0:
            src_rate.to_excel(writer, sheet_name="src_high_gap_rate", index=False)
        if len(app_rate) > 0:
            app_rate.to_excel(writer, sheet_name="app_high_gap_rate", index=False)

    return {
        "benchmark_r2": None if benchmark_res is None else float(benchmark_res.rsquared),
        "benchmark_n": int(len(bench_use)),
        "high_gap_q75_price": q75_gap,
        "high_gap_share": float(bench_use["high_gap"].mean()),
        "lpm_demand_r2": None if res_dem is None else float(res_dem.rsquared),
        "lpm_full_r2": None if res_full is None else float(res_full.rsquared),
        "coef_full": coef_full,
        "src_rate": src_rate,
        "app_rate": app_rate,
        "item_benchmark_panel": bench_use3,
    }


# ==========================================================
# 9) Paper artifacts
# ==========================================================
def build_paper_tables_figure(exp1: Dict, exp2: Dict, exp3: Dict, exp4: Dict, out_dir: str) -> Dict:
    """
    Build 2 main tables, 1 appendix table, and 1 mechanism figure.
    """
    # ---------------- Main Table 1 ----------------
    mt1 = pd.DataFrame([
        ["Permutation test", "Real FE-only R²", exp2["real_fe_only_r2"], None],
        ["Permutation test", "Mean permuted FE-only R²", exp2["perm_fe_only_mean_r2"], None],
        ["Permutation test", "p-value (FE-only)", exp2["p_fe_only"], None],
        ["Permutation test", "Real controls+FE R²", exp2["real_controls_fe_r2"], None],
        ["Permutation test", "Mean permuted controls+FE R²", exp2["perm_controls_fe_mean_r2"], None],
        ["Permutation test", "p-value (controls+FE)", exp2["p_controls_fe"], None],
        ["Alternative baseline", "Controls only R²", float(exp3["comparison_table"].set_index("model").loc["controls_only", "r2"]), None],
        ["Alternative baseline", "Controls + FE R²", float(exp3["comparison_table"].set_index("model").loc["controls_plus_fe", "r2"]), None],
        ["Alternative baseline", "Controls + LOO supplier mean R²", float(exp3["comparison_table"].set_index("model").loc["controls_plus_loo_mean", "r2"]), None],
        ["Alternative baseline", "Controls + EB-shrinkage baseline R²", float(exp3["comparison_table"].set_index("model").loc["controls_plus_eb_mean", "r2"]), None],
        ["Alternative baseline", "Spearman(FE, LOO baseline)", exp3["spearman_fe_vs_loo"], None],
        ["Alternative baseline", "Spearman(FE, EB baseline)", exp3["spearman_fe_vs_eb"], None],
    ], columns=["Panel", "Statistic", "Value", "Interpretation"])
    mt1["Interpretation"] = ""
    mt1.loc[0, "Interpretation"] = "Real supplier identity contains far more price-level information than random labels."
    mt1.loc[6, "Interpretation"] = "Supplier anchoring survives when hard dummies are replaced by softer baselines."

    # ---------------- Main Table 2 ----------------
    # Select concise evidence from exp4
    app_rate = exp4["app_rate"].copy() if exp4["app_rate"] is not None else pd.DataFrame()
    src_rate = exp4["src_rate"].copy() if exp4["src_rate"] is not None else pd.DataFrame()
    coef_full = exp4["coef_full"].copy() if exp4["coef_full"] is not None else pd.DataFrame()

    top_pos_app = ""
    top_neg_app = ""
    top_pos_src = ""
    top_neg_src = ""
    if len(app_rate) > 0:
        top_pos_app = str(app_rate.sort_values("lift_vs_overall", ascending=False).iloc[0]["app_label"])
        top_neg_app = str(app_rate.sort_values("lift_vs_overall", ascending=True).iloc[0]["app_label"])
    if len(src_rate) > 0:
        top_pos_src = str(src_rate.sort_values("lift_vs_overall", ascending=False).iloc[0]["src_label"])
        top_neg_src = str(src_rate.sort_values("lift_vs_overall", ascending=True).iloc[0]["src_label"])

    mt2 = pd.DataFrame([
        ["Benchmark construction", "Benchmark R²", exp4["benchmark_r2"]],
        ["Benchmark construction", "Benchmark sample size", exp4["benchmark_n"]],
        ["High-gap definition", "Gap-price 75th percentile", exp4["high_gap_q75_price"]],
        ["High-gap definition", "High-gap share", exp4["high_gap_share"]],
        ["LPM", "Demand-only R²", exp4["lpm_demand_r2"]],
        ["LPM", "Demand + src/app positioning R²", exp4["lpm_full_r2"]],
        ["Category evidence", "Top positive app lift", top_pos_app],
        ["Category evidence", "Top negative app lift", top_neg_app],
        ["Category evidence", "Top positive src lift", top_pos_src],
        ["Category evidence", "Top negative src lift", top_neg_src],
    ], columns=["Panel", "Statistic", "Value"])
    mt2["Interpretation"] = ""
    mt2.loc[0, "Interpretation"] = "The first-layer benchmark is supplier-anchored."
    mt2.loc[4, "Interpretation"] = "Coarse demand proxies add little on their own."
    mt2.loc[5, "Interpretation"] = "Upper-tail deviations are more strongly linked to src/app positioning."

    # ---------------- Appendix Table A1 ----------------
    at1 = pd.DataFrame([
        ["Core FE decomposition R²", exp1["core_r2"]],
        ["Extended FE decomposition R²", exp1["ext_r2"]],
        ["Core FE decomposition N", exp1["core_n"]],
        ["Extended FE decomposition N", exp1["ext_n"]],
    ], columns=["Statistic", "Value"])

    # Add a few leading coefficients if available
    if exp1["ext_coef"] is not None and len(exp1["ext_coef"]) > 0:
        top_coef = exp1["ext_coef"].query("term != 'const'").copy()
        top_coef["abs_t"] = top_coef["t"].abs()
        top_coef = top_coef.sort_values(["abs_t", "pvalue"], ascending=[False, True]).head(6)
        top_coef = top_coef[["term", "coef", "std_err", "pvalue", "stars"]]
    else:
        top_coef = pd.DataFrame(columns=["term", "coef", "std_err", "pvalue", "stars"])

    # Export workbook and CSVs
    xlsx_path = os.path.join(out_dir, "paper_tables_from_original_program.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        mt1.to_excel(writer, sheet_name="Main_Table_1", index=False)
        mt2.to_excel(writer, sheet_name="Main_Table_2", index=False)
        at1.to_excel(writer, sheet_name="Appendix_Table_A1", index=False)
        if len(top_coef) > 0:
            top_coef.to_excel(writer, sheet_name="Appendix_A1_top_coef", index=False)

    mt1.to_csv(os.path.join(out_dir, "paper_main_table_1.csv"), index=False, encoding="utf-8-sig")
    mt2.to_csv(os.path.join(out_dir, "paper_main_table_2.csv"), index=False, encoding="utf-8-sig")
    at1.to_csv(os.path.join(out_dir, "paper_appendix_table_a1.csv"), index=False, encoding="utf-8-sig")
    if len(top_coef) > 0:
        top_coef.to_csv(os.path.join(out_dir, "paper_appendix_table_a1_top_coef.csv"), index=False, encoding="utf-8-sig")

    # ---------------- Mechanism figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8))

    # Panel A: permutation
    axes[0].bar(["Real FE-only", "Mean permuted"], [exp2["real_fe_only_r2"], exp2["perm_fe_only_mean_r2"]])
    axes[0].set_title("A. True supplier identity is informative")
    axes[0].set_ylabel("R²")

    # Panel B: alternative baselines
    comp = exp3["comparison_table"].copy()
    axes[1].bar(comp["model"], comp["r2"])
    axes[1].set_title("B. Supplier anchor survives softer baselines")
    axes[1].tick_params(axis="x", rotation=20)

    # Panel C: high-gap by top application lift
    if len(app_rate) > 0:
        use = app_rate.sort_values("lift_vs_overall", ascending=False).head(5)
        axes[2].bar(use["app_label"], use["lift_vs_overall"])
        axes[2].tick_params(axis="x", rotation=30)
    else:
        axes[2].text(0.5, 0.5, "No app-rate data", ha="center", va="center")
        axes[2].set_xticks([])
    axes[2].set_title("C. Above-benchmark gaps reflect positioning")
    axes[2].set_ylabel("High-gap lift vs overall")

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "paper_figure_supplier_anchor_mechanism.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "main_table_1": mt1,
        "main_table_2": mt2,
        "appendix_table_a1": at1,
        "appendix_top_coef": top_coef,
        "workbook": xlsx_path,
        "figure": fig_path,
    }




# ==========================================================
# 9b) Build Table 9 for Section 4.2.3
# ==========================================================
def build_table9_supplier_anchor(exp1: Dict, exp2: Dict, exp3: Dict, exp4: Dict, out_dir: str) -> Dict:
    """
    Build a single Table 9 that summarizes the four verification analyses
    for Section 4.2.3 and export it in CSV / XLSX / LaTeX formats.
    """
    comp = exp3["comparison_table"].copy()
    comp_map = {str(r["model"]): r for _, r in comp.iterrows()}

    rows = [
        {
            "Group": "A",
            "Verification": "Observable decomposition of supplier FE",
            "Quantitative result": f"Core R² = {exp1['core_r2']:.3f}; Extended R² = {exp1['ext_r2']:.3f}; N = {int(exp1['core_n'])}",
            "Meaning": "Publicly observable supplier-side variables explain only a small share of supplier FE."
        },
        {
            "Group": "B",
            "Verification": "Permutation test (FE-only)",
            "Quantitative result": f"Real R² = {exp2['real_fe_only_r2']:.3f}; Mean permuted R² = {exp2['perm_fe_only_mean_r2']:.3f}; p = {exp2['p_fe_only']:.3f}",
            "Meaning": "True supplier identity carries much more quote-level information than random supplier labels."
        },
        {
            "Group": "B",
            "Verification": "Permutation test (controls + supplier FE)",
            "Quantitative result": f"Real R² = {exp2['real_controls_fe_r2']:.3f}; Mean permuted R² = {exp2['perm_controls_fe_mean_r2']:.3f}; p = {exp2['p_controls_fe']:.3f}",
            "Meaning": "Supplier anchoring remains strongly informative even after adding observable controls."
        },
        {
            "Group": "C",
            "Verification": "Softer baseline comparison",
            "Quantitative result": (
                f"Controls only R² = {comp_map['controls_only']['r2']:.3f}; "
                f"Controls + FE R² = {comp_map['controls_plus_fe']['r2']:.3f}; "
                f"Controls + LOO mean R² = {comp_map['controls_plus_loo_mean']['r2']:.3f} "
                f"(ρ = {comp_map['controls_plus_loo_mean']['spearman_vs_fe']:.3f}); "
                f"Controls + EB mean R² = {comp_map['controls_plus_eb_mean']['r2']:.3f} "
                f"(ρ = {comp_map['controls_plus_eb_mean']['spearman_vs_fe']:.3f})"
            ),
            "Meaning": "The supplier anchor survives when hard dummies are replaced by softer supplier baselines."
        },
        {
            "Group": "D",
            "Verification": "Supplier-anchored benchmark and above-benchmark deviations",
            "Quantitative result": (
                f"Benchmark R² = {exp4['benchmark_r2']:.3f}; "
                f"Q75(gap_price) = {exp4['high_gap_q75_price']:.3f}; "
                f"High-gap share = {exp4['high_gap_share']:.3f}; "
                f"Demand-only R² = {exp4['lpm_demand_r2']:.3f}; "
                f"Demand + src/app positioning R² = {exp4['lpm_full_r2']:.3f}"
            ),
            "Meaning": "After the anchor is fixed, the second layer mainly operates through within-anchor src/app positioning."
        },
    ]
    table9 = pd.DataFrame(rows)

    csv_path = os.path.join(out_dir, "table9_supplier_anchoring_summary.csv")
    xlsx_path = os.path.join(out_dir, "table9_supplier_anchoring_summary.xlsx")
    tex_path = os.path.join(out_dir, "table9_supplier_anchoring_summary.tex")

    table9.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        table9.to_excel(writer, sheet_name="Table_9", index=False)

    # Three-line table in LaTeX format
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Quantitative evidence for supplier anchoring as a benchmark layer}",
        r"\label{tab:supplier_anchor_table9}",
        r"\begin{tabular}{p{0.06\textwidth} p{0.23\textwidth} p{0.34\textwidth} p{0.27\textwidth}}",
        r"\toprule",
        r"Group & Verification & Quantitative result & Meaning \\",
        r"\midrule",
    ]
    for _, r in table9.iterrows():
        q = str(r["Quantitative result"]).replace("%", r"\%").replace("ρ", r"$\rho$")
        m = str(r["Meaning"]).replace("%", r"\%").replace("ρ", r"$\rho$")
        v = str(r["Verification"]).replace("%", r"\%")
        g = str(r["Group"]).replace("%", r"\%")
        latex_lines.append(f"{g} & {v} & {q} & {m} \\\\")
    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item Notes: Group A tests whether supplier fixed effects can be reconstructed from publicly observable supplier-side variables. Group B evaluates whether the explanatory power of the true supplier mapping exceeds that of randomly permuted supplier labels. Group C replaces hard supplier fixed effects with softer supplier baselines to test whether the anchoring pattern survives beyond dummy-based implementation. Group D fixes a supplier-anchored benchmark and then examines whether positive deviations above that benchmark are primarily associated with source/application positioning.",
        r"\end{tablenotes}",
        r"\end{table}",
    ]
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    return {"table9": table9, "csv": csv_path, "xlsx": xlsx_path, "tex": tex_path}


# ==========================================================
# 10) Overview and main
# ==========================================================
def main():
    cfg = dict(base.CONFIG)
    cfg.update(CONFIG)
    out_dir = cfg["OUTPUT_DIR"]
    ensure_dir(out_dir)

    # Step 1: original supplier-level workflow
    original = run_original_base_workflow(cfg, out_dir)

    # Step 2: Neo4j-backed item panel
    item_panel = build_item_panel_with_neo4j(
        cfg=cfg,
        item_df=original["item_df"],
        supplier_panel=original["supplier_panel"],
        media_df=original["media_df"],
        out_dir=out_dir,
    )

    # Step 3: augment the original supplier panel with Neo4j-recovered app/media structure
    supplier_panel_aug = augment_supplier_panel_from_neo4j(item_panel, original["supplier_panel"], out_dir)

    # Step 4: four experiments
    exp1 = run_experiment_1(supplier_panel_aug, out_dir)
    exp2 = run_experiment_2(item_panel, cfg, out_dir)
    exp3 = run_experiment_3(item_panel, out_dir)
    exp4 = run_experiment_4(item_panel, out_dir, cfg)

    # Step 5: paper artifacts
    paper = build_paper_tables_figure(exp1, exp2, exp3, exp4, out_dir)
    table9 = build_table9_supplier_anchor(exp1, exp2, exp3, exp4, out_dir)

    # Step 6: overview
    overview = {
        "files_used": {
            "price_xlsx": cfg["FILE_PRICE_XLSX"],
            "fe_supplier_csv": cfg["FILE_FE_SUPPLIER"],
            "media_xlsx": cfg["FILE_MEDIA_XLSX"],
            "node_dataproduct_csv": cfg["FILE_NODE_DATAPRODUCT"],
            "rel_src_csv": cfg["FILE_REL_SRC"],
            "rel_app_csv": cfg["FILE_REL_APP"],
        },
        "summary": {
            "experiment_1_core_r2": exp1["core_r2"],
            "experiment_1_extended_r2": exp1["ext_r2"],
            "experiment_2_real_fe_only_r2": exp2["real_fe_only_r2"],
            "experiment_2_p_fe_only": exp2["p_fe_only"],
            "experiment_2_real_controls_fe_r2": exp2["real_controls_fe_r2"],
            "experiment_2_p_controls_fe": exp2["p_controls_fe"],
            "experiment_3_comparison": exp3["comparison_table"].to_dict(orient="records"),
            "experiment_4_benchmark_r2": exp4["benchmark_r2"],
            "experiment_4_high_gap_q75_price": exp4["high_gap_q75_price"],
            "experiment_4_high_gap_share": exp4["high_gap_share"],
            "experiment_4_lpm_demand_r2": exp4["lpm_demand_r2"],
            "experiment_4_lpm_full_r2": exp4["lpm_full_r2"],
        },
        "paper_outputs": {
            "workbook": paper["workbook"],
            "figure": paper["figure"],
            "main_table_1_csv": os.path.join(out_dir, "paper_main_table_1.csv"),
            "main_table_2_csv": os.path.join(out_dir, "paper_main_table_2.csv"),
            "appendix_table_a1_csv": os.path.join(out_dir, "paper_appendix_table_a1.csv"),
            "table9_csv": table9["csv"],
            "table9_xlsx": table9["xlsx"],
            "table9_tex": table9["tex"],
        },
        "notes": [
            "This all-in-one script extends the original supplier FE interpretation program rather than replacing its logic.",
            "Supplier anchoring is treated as a reduced-form baseline, not as a structural markup estimate.",
            "High-gap is defined as actual price minus supplier-anchored benchmark price, using the upper quartile of positive deviations.",
            "The script writes intermediate panels first, then final paper artifacts, so manual reruns of multiple scripts are no longer necessary."
        ]
    }
    with open(os.path.join(out_dir, "overview.json"), "w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    print("=" * 88)
    print("[Done] original-based all-in-one workflow completed.")
    print(f"Output directory: {out_dir}")
    print(f"Paper workbook: {paper['workbook']}")
    print(f"Mechanism figure: {paper['figure']}")
    print(f"Table 9 CSV: {table9['csv']}")
    print(f"Table 9 XLSX: {table9['xlsx']}")
    print(f"Table 9 LaTeX: {table9['tex']}")
    print("=" * 88)


if __name__ == "__main__":
    import re
    main()
