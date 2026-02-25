#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
supplier_fe_market_power_indications.py

Purpose (for Section 4.2.3):
- Treat supplier FE (fe_log10) as the dependent variable (reduced-form supplier premium).
- Examine whether supplier FE co-varies with observable proxy variables plausibly related to
  market-power indications in exchange-listing environments.
- Use media_result.xlsx (application-domain demand proxies) instead of supplier-app detail tables.
- Export paper-mapped outputs for Section 4.2.3:
    * Table 11: Spearman rank correlations
    * Table 12: HC3-robust OLS associations
    * Fig. 8: Supplier FE across proxy-based indication tiers

Important scope:
- This script is interpretive/descriptive, not causal identification.
- supplier FE is NOT a structural markup estimate.
- "Market-power indication" tiers are sample-internal, proxy-based heuristic labels.

Minimum required inputs:
1) STEP1_fe_table_supplier.csv
   columns: supplier, fe_log10, fe_multiplier_on_price, baseline_supplier

2) Data_Products_Enriched.xlsx
   minimum columns: supplier, name, price
   optional columns:
   Data_Exclusivity, Is_Gov, Is_Listed, year_create, Total_Asset, Year_Dis, Is_Anonymous
   and one application-domain column (e.g., app / application / app_IndustryCategory / 应用行业)

3) media_result.xlsx
   columns:
   keyword, sogou_web_results, sina_news_results, weixin_article_results
   (optional: weixin_account_results)

Outputs:
- supplier_fe_market_power_panel_v4.csv / .xlsx
- Table11_supplier_fe_proxy_spearman_v4.csv / .xlsx
- Table12_supplier_fe_proxy_ols_hc3_v4.csv / .xlsx
- Fig8_supplier_fe_by_market_power_tier_v4.png / .pdf
- Fig8_data_supplier_fe_by_market_power_tier_v4.csv
- overview_v4.json
- paper_artifact_titles_4_2_3_v4.json
- model summary .txt files
"""

# ==========================================================
# 0) Global configuration (kept at the top as requested)
# ==========================================================
CONFIG = {
    # ---- Input files ----
    "FILE_FE_SUPPLIER": "./result_kg_reproduce/STEP1_fe_table_supplier.csv",
    "FILE_ENRICHED_XLSX": "./anymous/name_price_anonymized.xlsx",
    "FILE_MEDIA_XLSX": "./anymous/media_result.xlsx",

    # ---- Output directory ----
    "OUTPUT_DIR": "./supplier_fe_market_power_run",

    # ---- Time / year convention ----
    "REFERENCE_YEAR": 2025,

    # ---- FE strength thresholds (log10 price premium) ----
    # 0.50 ~ 3.16x; 0.20 ~ 1.58x
    "FE_STRONG_LOG10_THRESHOLD": 0.50,
    "FE_MODERATE_LOG10_THRESHOLD": 0.20,

    # ---- Sample-internal tiering quantiles ----
    "Q_HIGH": 0.85,
    "Q_LOW": 0.35,

    # ---- OLS minimum sample size ----
    "MIN_OBS_FOR_OLS": 30,

    # ---- Winsorization ----
    "WINSOR_LOWER": 0.01,
    "WINSOR_UPPER": 0.99,

    # ---- Media transform ----
    # log1p is used to reduce heavy-tail effects in media counts
    "MEDIA_USE_LOG1P": True,

    # ---- Plot options ----
    "PAPER_FIG8_DPI": 300,

    # ---- Paper artifact basenames (for Section 4.2.3 mapping) ----
    "PAPER_TABLE11_BASENAME": "Table11_supplier_fe_proxy_spearman",
    "PAPER_TABLE12_BASENAME": "Table12_supplier_fe_proxy_ols_hc3",
    "PAPER_FIG8_BASENAME": "Fig8_supplier_fe_by_market_power_tier",

    # ---- Export options ----
    "WRITE_XLSX": True,
    "WRITE_PDF_FIG": True,
}

# ==========================================================
# 1) Imports
# ==========================================================
import os
import re
import json
import warnings
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

try:
    import statsmodels.api as sm
except Exception:
    sm = None

warnings.filterwarnings("ignore")

# ==========================================================
# 2) Utility functions
# ==========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _normalize_text(x) -> str:
    """
    Normalize text for matching:
    - strip spaces
    - collapse consecutive spaces
    - keep original language characters
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_key_for_match(x) -> str:
    """
    Stronger normalization for merge keys:
    - strip spaces
    - lower-case English
    - normalize punctuation
    """
    s = _normalize_text(x)
    s = s.lower()
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("，", ",")
    s = re.sub(r"\s+", "", s)
    return s


def parse_total_asset_to_wan(x):
    """
    Parse total assets into 'wan RMB' when unit strings are available.
    Examples:
    - 1000万 -> 1000
    - 2.5亿 -> 25000
    - 35000000 -> returned as-is (unknown unit)
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip().replace(",", "").replace("，", "")
    if s == "":
        return np.nan

    m = re.search(r"([-+]?\d*\.?\d+)", s)
    if not m:
        return np.nan

    val = float(m.group(1))

    if "亿" in s:
        return val * 10000.0
    if "万" in s:
        return val

    return val


def winsorize_series(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    s = safe_to_numeric(s)
    if s.notna().sum() < 5:
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def percentile_rank(s: pd.Series) -> pd.Series:
    """
    Convert a numeric series to percentile ranks in [0, 1].
    NaNs are preserved.
    """
    s_num = safe_to_numeric(s)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    m = s_num.notna()
    if m.sum() == 0:
        return out
    out.loc[m] = s_num.loc[m].rank(method="average", pct=True)
    return out


def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_application_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try multiple common column names for application-domain labels.
    """
    candidates = [
        "app",
        "application",
        "application_industry",
        "app_industry",
        "app_IndustryCategory",
        "app_industrycategory",
        "应用行业",
        "应用产业",
        "关联产业",
        "应用领域",
        "行业应用",
        "keyword",  # fallback if enriched file already stores the keyword
    ]
    return _pick_first_existing(df, candidates)


# ==========================================================
# 3) Read and aggregate supplier-level base proxies (from enriched file)
# ==========================================================
def build_supplier_panel_from_enriched(file_enriched_xlsx: str, ref_year: int) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    """
    Returns:
    - supplier_panel (supplier-level base variables)
    - item_level_df (cleaned item-level records, used later for media aggregation)
    - app_col_used (application-domain column name if detected, else None)
    """
    df = pd.read_excel(file_enriched_xlsx)

    required_basic = ["supplier", "name", "price"]
    for c in required_basic:
        if c not in df.columns:
            raise ValueError(f"[Error] Data_Products_Enriched.xlsx missing required column: {c}")

    d = df.copy()
    d["supplier"] = d["supplier"].astype(str).str.strip()
    d["name"] = d["name"].astype(str).str.strip()
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    d = d[d["supplier"].notna() & (d["supplier"] != "")].copy()

    # Price variables
    d["price_pos"] = d["price"].where(d["price"] > 0, np.nan)
    d["y_log10"] = np.log10(d["price_pos"])

    # Optional columns (auto-created if absent)
    for col in ["Data_Exclusivity", "Is_Gov", "Is_Listed", "year_create", "Total_Asset", "Year_Dis", "Is_Anonymous"]:
        if col not in d.columns:
            d[col] = np.nan

    # Numeric conversion
    d["Data_Exclusivity"] = pd.to_numeric(d["Data_Exclusivity"], errors="coerce")
    d["Is_Gov"] = pd.to_numeric(d["Is_Gov"], errors="coerce")
    d["Is_Listed"] = pd.to_numeric(d["Is_Listed"], errors="coerce")
    d["Is_Anonymous"] = pd.to_numeric(d["Is_Anonymous"], errors="coerce")
    d["year_create"] = pd.to_numeric(d["year_create"], errors="coerce")

    # Firm age proxy (item-level, later aggregated to supplier-level)
    d["firm_age_item"] = np.where(
        d["year_create"].between(1900, ref_year, inclusive="both"),
        ref_year - d["year_create"] + 1,
        np.nan
    )

    d["Total_Asset_wan"] = d["Total_Asset"].apply(parse_total_asset_to_wan)
    d["Year_Dis_num"] = pd.to_numeric(d["Year_Dis"], errors="coerce")

    # Application-domain column detection (for media_result join later)
    app_col = _get_application_column(d)
    if app_col is None:
        d["_app_domain_raw"] = np.nan
    else:
        d["_app_domain_raw"] = d[app_col]

    d["_app_domain_raw"] = d["_app_domain_raw"].apply(_normalize_text)
    d["_app_domain_key"] = d["_app_domain_raw"].apply(_normalize_key_for_match)

    # Supplier counts
    sup_counts_all = d.groupby("supplier", dropna=False)["name"].count().rename("n_products")
    sup_counts_priced = d.groupby("supplier", dropna=False)["price_pos"].apply(lambda x: int(x.notna().sum())).rename("n_products_priced")

    total_products = max(int(d["name"].count()), 1)

    def agg_supplier(g: pd.DataFrame) -> pd.Series:
        y = pd.to_numeric(g["y_log10"], errors="coerce")
        price = pd.to_numeric(g["price_pos"], errors="coerce")

        out = {
            "price_mean": float(price.mean()) if price.notna().any() else np.nan,
            "price_median": float(price.median()) if price.notna().any() else np.nan,
            "y_log10_mean": float(y.mean()) if y.notna().any() else np.nan,
            "y_log10_median": float(y.median()) if y.notna().any() else np.nan,

            "exclusivity_rate": float(pd.to_numeric(g["Data_Exclusivity"], errors="coerce").mean())
                if g["Data_Exclusivity"].notna().any() else np.nan,
            "gov_share": float(pd.to_numeric(g["Is_Gov"], errors="coerce").mean())
                if g["Is_Gov"].notna().any() else np.nan,
            "listed_share": float(pd.to_numeric(g["Is_Listed"], errors="coerce").mean())
                if g["Is_Listed"].notna().any() else np.nan,
            "anonymous_rate": float(pd.to_numeric(g["Is_Anonymous"], errors="coerce").mean())
                if g["Is_Anonymous"].notna().any() else np.nan,

            "firm_age": float(pd.to_numeric(g["firm_age_item"], errors="coerce").median())
                if g["firm_age_item"].notna().any() else np.nan,
            "total_asset_wan": float(pd.to_numeric(g["Total_Asset_wan"], errors="coerce").median())
                if g["Total_Asset_wan"].notna().any() else np.nan,
            "disclosure_years": float(pd.to_numeric(g["Year_Dis_num"], errors="coerce").median())
                if g["Year_Dis_num"].notna().any() else np.nan,

            "n_apps_in_enriched": int(g["_app_domain_key"].replace("", np.nan).dropna().nunique())
                if "_app_domain_key" in g.columns else np.nan,
        }
        return pd.Series(out)

    panel = d.groupby("supplier", dropna=False).apply(agg_supplier).reset_index()
    panel = panel.merge(sup_counts_all.reset_index(), on="supplier", how="left")
    panel = panel.merge(sup_counts_priced.reset_index(), on="supplier", how="left")

    panel["n_products"] = panel["n_products"].fillna(0).astype(int)
    panel["n_products_priced"] = panel["n_products_priced"].fillna(0).astype(int)
    panel["overall_count_share"] = panel["n_products"] / float(total_products)

    global_y_median = float(pd.to_numeric(d["y_log10"], errors="coerce").median())
    panel["global_y_log10_median"] = global_y_median
    panel["premium_log10_fallback"] = panel["y_log10_median"] - global_y_median

    return panel, d, app_col


# ==========================================================
# 4) Read supplier FE table
# ==========================================================
def load_supplier_fe_table(file_fe_supplier: str) -> pd.DataFrame:
    fe = pd.read_csv(file_fe_supplier)
    required = ["supplier", "fe_log10", "fe_multiplier_on_price", "baseline_supplier"]
    for c in required:
        if c not in fe.columns:
            raise ValueError(f"[Error] STEP1_fe_table_supplier.csv missing required column: {c}")

    f = fe.copy()
    f["supplier"] = f["supplier"].astype(str).str.strip()
    f["fe_log10"] = pd.to_numeric(f["fe_log10"], errors="coerce")
    f["fe_multiplier_on_price"] = pd.to_numeric(f["fe_multiplier_on_price"], errors="coerce")
    return f[required].drop_duplicates(subset=["supplier"]).reset_index(drop=True)


# ==========================================================
# 5) Read media_result.xlsx and aggregate media exposure to supplier level
# ==========================================================
def load_media_table(file_media_xlsx: str, use_log1p: bool = True) -> pd.DataFrame:
    if not os.path.exists(file_media_xlsx):
        raise FileNotFoundError(f"[Error] media_result.xlsx not found: {file_media_xlsx}")

    m = pd.read_excel(file_media_xlsx)
    required = ["keyword", "sogou_web_results", "sina_news_results", "weixin_article_results"]
    for c in required:
        if c not in m.columns:
            raise ValueError(f"[Error] media_result.xlsx missing required column: {c}")

    if "weixin_account_results" not in m.columns:
        m["weixin_account_results"] = np.nan

    m = m.copy()
    m["keyword"] = m["keyword"].apply(_normalize_text)
    m["keyword_key"] = m["keyword"].apply(_normalize_key_for_match)

    for c in ["sogou_web_results", "sina_news_results", "weixin_account_results", "weixin_article_results"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")

    # Deduplicate by keyword_key (keep max counts by column)
    agg_dict = {
        "keyword": "first",
        "sogou_web_results": "max",
        "sina_news_results": "max",
        "weixin_account_results": "max",
        "weixin_article_results": "max",
    }
    m = m.sort_values(["keyword_key", "keyword"]).groupby("keyword_key", as_index=False).agg(agg_dict)

    # Media transforms
    if use_log1p:
        m["media_web_log1p"] = np.log1p(m["sogou_web_results"])
        m["media_news_log1p"] = np.log1p(m["sina_news_results"])
        m["media_weixin_account_log1p"] = np.log1p(m["weixin_account_results"])
        m["media_weixin_article_log1p"] = np.log1p(m["weixin_article_results"])
    else:
        m["media_web_log1p"] = m["sogou_web_results"]
        m["media_news_log1p"] = m["sina_news_results"]
        m["media_weixin_account_log1p"] = m["weixin_account_results"]
        m["media_weixin_article_log1p"] = m["weixin_article_results"]

    # Composite demand-visibility proxy
    m["media_total_log1p_sum"] = (
        m["media_web_log1p"].fillna(0.0)
        + m["media_news_log1p"].fillna(0.0)
        + m["media_weixin_account_log1p"].fillna(0.0)
        + m["media_weixin_article_log1p"].fillna(0.0)
    )

    return m


def build_supplier_media_panel_from_items(
    item_df: pd.DataFrame,
    media_df: pd.DataFrame
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Map item-level application domains to media keywords and aggregate demand exposure to supplier level.
    """
    if "_app_domain_key" not in item_df.columns:
        return None, {"media_join_status": "item_df_missing_app_domain_key"}

    d = item_df.copy()
    d["supplier"] = d["supplier"].astype(str).str.strip()
    d["_app_domain_key"] = d["_app_domain_key"].fillna("").astype(str)
    d["_app_domain_raw"] = d["_app_domain_raw"].fillna("").astype(str)

    # Merge item application domain with media keyword
    x = d.merge(
        media_df,
        left_on="_app_domain_key",
        right_on="keyword_key",
        how="left",
        suffixes=("", "_media")
    )

    x["item_weight"] = 1.0
    x["has_media_match"] = x["keyword_key"].notna().astype(int)

    # Supplier-level application-domain concentration proxy (based on item counts)
    app_counts = (
        d.loc[d["_app_domain_key"] != "", ["supplier", "_app_domain_key"]]
        .groupby(["supplier", "_app_domain_key"])
        .size()
        .rename("cnt")
        .reset_index()
    )

    if len(app_counts) > 0:
        tmp = app_counts.copy()
        tmp["supplier_total"] = tmp.groupby("supplier")["cnt"].transform("sum")
        tmp["share"] = np.where(tmp["supplier_total"] > 0, tmp["cnt"] / tmp["supplier_total"], np.nan)
        hhi = tmp.groupby("supplier")["share"].apply(
            lambda s: float(np.sum(np.square(s.dropna().values))) if s.notna().any() else np.nan
        ).rename("app_domain_hhi")
        max_share = tmp.groupby("supplier")["share"].max().rename("max_app_share")
        n_apps = tmp.groupby("supplier")["_app_domain_key"].nunique().rename("n_apps_covered_in_media_map")
        struct_panel = pd.concat([hhi, max_share, n_apps], axis=1).reset_index()
    else:
        struct_panel = pd.DataFrame(columns=["supplier", "app_domain_hhi", "max_app_share", "n_apps_covered_in_media_map"])

    # Supplier-level media exposure aggregation
    def agg_one_supplier(g: pd.DataFrame) -> pd.Series:
        out = {
            "media_item_rows": int(len(g)),
            "media_match_rows": int(g["has_media_match"].sum()) if "has_media_match" in g.columns else 0,
            "media_match_ratio": float(g["has_media_match"].mean()) if "has_media_match" in g.columns and len(g) > 0 else np.nan,
        }

        # Raw counts (means and medians over matched rows)
        for src, out_mean, out_median in [
            ("sogou_web_results", "exp_media_web_mean_raw", "exp_media_web_median_raw"),
            ("sina_news_results", "exp_media_news_mean_raw", "exp_media_news_median_raw"),
            ("weixin_account_results", "exp_media_weixin_account_mean_raw", "exp_media_weixin_account_median_raw"),
            ("weixin_article_results", "exp_media_weixin_article_mean_raw", "exp_media_weixin_article_median_raw"),
        ]:
            s = pd.to_numeric(g[src], errors="coerce")
            out[out_mean] = float(s.mean()) if s.notna().any() else np.nan
            out[out_median] = float(s.median()) if s.notna().any() else np.nan

        # Transformed exposure proxies (means and medians)
        for src, out_mean, out_median in [
            ("media_web_log1p", "exp_media_web", "exp_media_web_median"),
            ("media_news_log1p", "exp_media_news", "exp_media_news_median"),
            ("media_weixin_account_log1p", "exp_media_weixin_account", "exp_media_weixin_account_median"),
            ("media_weixin_article_log1p", "exp_media_weixin_article", "exp_media_weixin_article_median"),
            ("media_total_log1p_sum", "exp_media_total", "exp_media_total_median"),
        ]:
            s = pd.to_numeric(g[src], errors="coerce")
            out[out_mean] = float(s.mean()) if s.notna().any() else np.nan
            out[out_median] = float(s.median()) if s.notna().any() else np.nan

        return pd.Series(out)

    media_panel = x.groupby("supplier", dropna=False).apply(agg_one_supplier).reset_index()

    # Merge structural proxies derived from application-domain distribution
    media_panel = media_panel.merge(struct_panel, on="supplier", how="left")

    # Additional derived proxies
    if "n_apps_covered_in_media_map" in media_panel.columns:
        media_panel["single_app_focus_flag"] = (pd.to_numeric(media_panel["n_apps_covered_in_media_map"], errors="coerce") <= 1).astype(float)

    diagnostics = {
        "media_rows_total": int(len(media_df)),
        "item_rows_total": int(len(d)),
        "item_rows_with_nonempty_app": int((d["_app_domain_key"] != "").sum()),
        "item_rows_matched_media": int(x["has_media_match"].sum()) if "has_media_match" in x.columns else 0,
        "item_media_match_ratio_over_all_items": float(x["has_media_match"].mean()) if "has_media_match" in x.columns and len(x) > 0 else np.nan,
        "unique_app_keys_in_items": int(d.loc[d["_app_domain_key"] != "", "_app_domain_key"].nunique()),
        "unique_media_keywords": int(media_df["keyword_key"].nunique()) if "keyword_key" in media_df.columns else int(len(media_df)),
    }

    return media_panel, diagnostics


# ==========================================================
# 6) Build proxy-based market-power indications and sample-internal tiers
# ==========================================================
def build_market_power_indications(panel: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    p = panel.copy()

    # Premium variable: FE first, fallback for descriptive score only
    p["premium_log10_effect"] = p["fe_log10"]
    p["premium_source"] = "supplier_FE"
    fallback_mask = p["premium_log10_effect"].isna() & p["premium_log10_fallback"].notna()
    p.loc[fallback_mask, "premium_log10_effect"] = p.loc[fallback_mask, "premium_log10_fallback"]
    p.loc[fallback_mask, "premium_source"] = "fallback_y_median_minus_global_median"

    p["premium_multiplier_vs_baseline"] = np.where(
        p["premium_log10_effect"].notna(),
        np.power(10.0, p["premium_log10_effect"].astype(float)),
        np.nan
    )

    # Log size proxy
    p["log_total_asset_wan"] = np.where(
        pd.to_numeric(p["total_asset_wan"], errors="coerce") > 0,
        np.log10(pd.to_numeric(p["total_asset_wan"], errors="coerce")),
        np.nan
    )

    # Winsorization
    lower = float(cfg.get("WINSOR_LOWER", 0.01))
    upper = float(cfg.get("WINSOR_UPPER", 0.99))
    for c in [
        "n_products", "overall_count_share", "firm_age", "log_total_asset_wan",
        "app_domain_hhi", "max_app_share", "exp_media_total",
        "exp_media_web", "exp_media_news", "exp_media_weixin_article"
    ]:
        if c in p.columns:
            p[c] = winsorize_series(p[c], lower, upper)

    # Score A: premium strength
    p["score_premium"] = percentile_rank(p["premium_log10_effect"])

    # Score B: barrier/resource indications
    barrier_components = []
    for c in ["exclusivity_rate", "gov_share", "listed_share", "firm_age", "log_total_asset_wan", "disclosure_years"]:
        if c in p.columns:
            barrier_components.append(percentile_rank(p[c]).rename(f"rk_{c}"))
    if barrier_components:
        barrier_mat = pd.concat(barrier_components, axis=1)
        p["score_barrier"] = barrier_mat.mean(axis=1, skipna=True)
    else:
        p["score_barrier"] = np.nan

    # Score C: structure indications
    struct_components = []
    for c in ["app_domain_hhi", "max_app_share", "overall_count_share", "single_app_focus_flag"]:
        if c in p.columns:
            struct_components.append(percentile_rank(p[c]).rename(f"rk_{c}"))
    if struct_components:
        struct_mat = pd.concat(struct_components, axis=1)
        p["score_structure"] = struct_mat.mean(axis=1, skipna=True)
    else:
        p["score_structure"] = np.nan

    # Score D: demand visibility (separate concept, smaller weight)
    demand_components = []
    for c in ["exp_media_total", "exp_media_web", "exp_media_news", "exp_media_weixin_article", "media_match_ratio"]:
        if c in p.columns:
            demand_components.append(percentile_rank(p[c]).rename(f"rk_{c}"))
    if demand_components:
        demand_mat = pd.concat(demand_components, axis=1)
        p["score_demand_visibility"] = demand_mat.mean(axis=1, skipna=True)
    else:
        p["score_demand_visibility"] = np.nan

    # Composite proxy-based indication score
    p["market_power_signal_score"] = (
        0.45 * p["score_premium"].fillna(0.0) +
        0.25 * p["score_structure"].fillna(0.0) +
        0.20 * p["score_barrier"].fillna(0.0) +
        0.10 * p["score_demand_visibility"].fillna(0.0)
    )

    # Flags for interpretive descriptions
    fe_strong_thr = float(cfg["FE_STRONG_LOG10_THRESHOLD"])
    fe_mod_thr = float(cfg["FE_MODERATE_LOG10_THRESHOLD"])

    p["flag_premium_strong"] = (pd.to_numeric(p["premium_log10_effect"], errors="coerce") >= fe_strong_thr).astype(int)
    p["flag_premium_moderate"] = (pd.to_numeric(p["premium_log10_effect"], errors="coerce") >= fe_mod_thr).astype(int)

    for col in ["score_structure", "score_barrier", "score_demand_visibility"]:
        if col in p.columns:
            p[f"flag_{col}_strong"] = (pd.to_numeric(p[col], errors="coerce") >= 0.67).astype(int)
            p[f"flag_{col}_moderate"] = (pd.to_numeric(p[col], errors="coerce") >= 0.50).astype(int)
        else:
            p[f"flag_{col}_strong"] = 0
            p[f"flag_{col}_moderate"] = 0

    # Sample-internal tiers (heuristic)
    q_high = float(cfg["Q_HIGH"])
    q_low = float(cfg["Q_LOW"])
    high_cut = p["market_power_signal_score"].quantile(q_high) if p["market_power_signal_score"].notna().any() else np.nan
    low_cut = p["market_power_signal_score"].quantile(q_low) if p["market_power_signal_score"].notna().any() else np.nan

    tier_cn = []
    tier_en = []
    reasons = []

    for _, r in p.iterrows():
        score = r.get("market_power_signal_score", np.nan)
        fev = r.get("premium_log10_effect", np.nan)
        s_struct = r.get("score_structure", np.nan)
        s_barrier = r.get("score_barrier", np.nan)
        s_demand = r.get("score_demand_visibility", np.nan)

        parts = []

        if pd.notna(fev):
            if fev >= fe_strong_thr:
                parts.append("Strong FE premium (>=3.16x baseline)")
            elif fev >= fe_mod_thr:
                parts.append("Positive FE premium")
            else:
                parts.append("Weak/non-positive FE premium")
        else:
            parts.append("FE missing (fallback used for descriptive score only)")

        if pd.notna(s_struct):
            if s_struct >= 0.67:
                parts.append("Stronger application-domain concentration indications")
            elif s_struct >= 0.50:
                parts.append("Moderate application-domain concentration indications")
        if pd.notna(s_barrier):
            if s_barrier >= 0.67:
                parts.append("Stronger barrier/resource indications")
            elif s_barrier >= 0.50:
                parts.append("Moderate barrier/resource indications")
        if pd.notna(s_demand):
            if s_demand >= 0.67:
                parts.append("Higher demand-visibility exposure")
            elif s_demand >= 0.50:
                parts.append("Moderate demand-visibility exposure")

        if "max_app_share" in p.columns and pd.notna(r.get("max_app_share", np.nan)):
            parts.append(f"max_app_share={r['max_app_share']:.2f}")
        if "app_domain_hhi" in p.columns and pd.notna(r.get("app_domain_hhi", np.nan)):
            parts.append(f"app_domain_hhi={r['app_domain_hhi']:.3f}")

        if pd.isna(score):
            cn = "中市场势力迹象（样本内代理口径）"
            en = "Medium indication (sample-internal, proxy-based)"
        else:
            cond_high = (
                (pd.notna(high_cut) and score >= high_cut and pd.notna(fev) and fev >= fe_mod_thr)
                or (pd.notna(fev) and fev >= fe_strong_thr and (
                    (pd.notna(s_struct) and s_struct >= 0.67) or
                    (pd.notna(s_barrier) and s_barrier >= 0.67)
                ))
            )
            cond_low = (
                (pd.notna(low_cut) and score <= low_cut)
                and (pd.isna(fev) or fev < fe_mod_thr)
                and (pd.isna(s_struct) or s_struct < 0.50)
                and (pd.isna(s_barrier) or s_barrier < 0.50)
            )

            if cond_high:
                cn = "高市场势力迹象（样本内代理口径）"
                en = "High indication (sample-internal, proxy-based)"
            elif cond_low:
                cn = "低市场势力迹象（样本内代理口径）"
                en = "Low indication (sample-internal, proxy-based)"
            else:
                cn = "中市场势力迹象（样本内代理口径）"
                en = "Medium indication (sample-internal, proxy-based)"

        tier_cn.append(cn)
        tier_en.append(en)
        reasons.append("; ".join(parts) if parts else "Insufficient proxy information")

    p["market_power_tier_v4"] = tier_cn
    p["market_power_tier_en_v4"] = tier_en
    p["market_power_reason_v4"] = reasons

    return p


# ==========================================================
# 7) Spearman and HC3 OLS (supplier FE as dependent variable)
# ==========================================================
def run_spearman_table(panel: pd.DataFrame, y_col: str, x_cols: List[str]) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(panel[y_col], errors="coerce")

    for x in x_cols:
        if x not in panel.columns:
            continue
        xs = pd.to_numeric(panel[x], errors="coerce")
        m = y.notna() & xs.notna()
        n = int(m.sum())
        if n < 8:
            continue
        if xs[m].nunique(dropna=True) <= 1:
            continue

        if spearmanr is None:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = spearmanr(xs[m], y[m], nan_policy="omit")

        rows.append({
            "x": x,
            "n": n,
            "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
            "pvalue": float(pval) if pd.notna(pval) else np.nan,
        })

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out = out.sort_values(["pvalue", "spearman_rho"], ascending=[True, False]).reset_index(drop=True)
    return out


def _drop_constant_and_collinear(X: pd.DataFrame, tol=1e-12) -> pd.DataFrame:
    """
    Remove constant / near-constant columns and perform a greedy rank-increase selection
    to reduce rank-deficiency warnings.
    """
    Xc = X.copy()

    for c in Xc.columns:
        Xc[c] = pd.to_numeric(Xc[c], errors="coerce")

    keep = []
    for c in Xc.columns:
        s = Xc[c]
        if s.notna().sum() == 0:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    Xc = Xc[keep].copy()

    if Xc.shape[1] <= 1:
        return Xc

    selected = []
    current = None
    for c in Xc.columns:
        col = Xc[[c]].fillna(0.0).values.astype(float)
        if current is None:
            selected.append(c)
            current = col
            continue
        cand = np.hstack([current, col])
        r_old = np.linalg.matrix_rank(current, tol=tol)
        r_new = np.linalg.matrix_rank(cand, tol=tol)
        if r_new > r_old:
            selected.append(c)
            current = cand

    return Xc[selected].copy()


def fit_ols_hc3(panel: pd.DataFrame, y_col: str, x_cols: List[str], model_name: str, min_obs: int = 30):
    if sm is None:
        return None, pd.DataFrame()

    use_cols = [c for c in x_cols if c in panel.columns]
    if not use_cols:
        return None, pd.DataFrame()

    d = panel[[y_col] + use_cols].copy()
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(axis=0, how="any").copy()

    if len(d) < min_obs:
        return None, pd.DataFrame()

    y = d[y_col].astype(float)
    X = d[use_cols].astype(float)

    X = _drop_constant_and_collinear(X)
    if X.shape[1] == 0:
        return None, pd.DataFrame()

    X = sm.add_constant(X, has_constant="add")
    res = sm.OLS(y, X).fit(cov_type="HC3")

    rows = []
    for term in res.params.index:
        rows.append({
            "model": model_name,
            "term": term,
            "coef": float(res.params[term]),
            "std_err_hc3": float(res.bse[term]) if term in res.bse.index else np.nan,
            "t": float(res.tvalues[term]) if term in res.tvalues.index else np.nan,
            "pvalue": float(res.pvalues[term]) if term in res.pvalues.index else np.nan,
            "n_obs": int(res.nobs),
            "r2": float(res.rsquared),
            "adj_r2": float(res.rsquared_adj),
        })
    coef_df = pd.DataFrame(rows)
    return res, coef_df


# ==========================================================
# 8) Fig. 8 plotting function (for Section 4.2.3)
# ==========================================================
def plot_fig8_supplier_fe_by_tier(panel: pd.DataFrame, out_dir: str, cfg: Dict):
    """
    Fig. 8:
    Supplier FE across proxy-based market-power indication tiers (descriptive evidence)
    """
    required_cols = ["supplier", "fe_log10", "market_power_tier_en_v4", "market_power_tier_v4"]
    for c in required_cols:
        if c not in panel.columns:
            print(f"[Warning] Fig. 8 skipped: missing column {c}")
            return None

    d = panel.copy()
    d["fe_log10"] = pd.to_numeric(d["fe_log10"], errors="coerce")
    d = d.dropna(subset=["fe_log10", "market_power_tier_en_v4"]).copy()
    if len(d) == 0:
        print("[Warning] Fig. 8 skipped: no valid rows.")
        return None

    order_en = [
        "Low indication (sample-internal, proxy-based)",
        "Medium indication (sample-internal, proxy-based)",
        "High indication (sample-internal, proxy-based)",
    ]
    short_labels = ["Low indication", "Medium indication", "High indication"]

    d = d[d["market_power_tier_en_v4"].isin(order_en)].copy()
    if len(d) == 0:
        print("[Warning] Fig. 8 skipped: no rows in tier order.")
        return None

    grouped = []
    counts = []
    medians = []
    for lab in order_en:
        vals = d.loc[d["market_power_tier_en_v4"] == lab, "fe_log10"].dropna().values.astype(float)
        grouped.append(vals)
        counts.append(len(vals))
        medians.append(float(np.median(vals)) if len(vals) > 0 else np.nan)

    if sum(counts) == 0:
        print("[Warning] Fig. 8 skipped: empty groups.")
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.boxplot(
        grouped,
        labels=short_labels,
        showfliers=False,
        patch_artist=False,
        widths=0.55
    )

    rng = np.random.RandomState(42)
    for i, vals in enumerate(grouped, start=1):
        if len(vals) == 0:
            continue
        xj = rng.normal(loc=i, scale=0.045, size=len(vals))
        ax.scatter(xj, vals, alpha=0.65, s=16)
        med = float(np.median(vals))
        ax.text(i + 0.12, med, f"median={med:.2f}\n(n={len(vals)})", fontsize=8, va="center")

    ax.set_xlabel("Proxy-based market-power indication tier")
    ax.set_ylabel("Supplier FE (log10 premium relative to baseline supplier)")
    ax.set_title("Supplier FE across proxy-based market-power indication tiers")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    base = cfg.get("PAPER_FIG8_BASENAME", "Fig8_supplier_fe_by_market_power_tier_v4")
    dpi = int(cfg.get("PAPER_FIG8_DPI", 300))

    png_path = os.path.join(out_dir, f"{base}.png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")

    pdf_path = None
    if bool(cfg.get("WRITE_PDF_FIG", True)):
        pdf_path = os.path.join(out_dir, f"{base}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    fig_data_path = os.path.join(out_dir, "Fig8_data_supplier_fe_by_market_power_tier_v4.csv")
    d_out = d[["supplier", "fe_log10", "market_power_tier_v4", "market_power_tier_en_v4"]].copy()
    d_out.to_csv(fig_data_path, index=False, encoding="utf-8-sig")

    return {
        "png": png_path,
        "pdf": pdf_path,
        "data_csv": fig_data_path,
        "counts": {k: int(v) for k, v in zip(short_labels, counts)},
        "medians": {k: (float(m) if pd.notna(m) else None) for k, m in zip(short_labels, medians)},
    }


# ==========================================================
# 9) Main
# ==========================================================
def main():
    cfg = CONFIG
    out_dir = cfg["OUTPUT_DIR"]
    ensure_dir(out_dir)

    print("=" * 88)
    print("[Start] supplier FE × proxy-based market-power indications (v4, media_result version)")
    print(f"Output directory: {out_dir}")

    # ---------- STEP A: Supplier-level base panel + item-level data ----------
    base_panel, item_df, app_col_used = build_supplier_panel_from_enriched(
        file_enriched_xlsx=cfg["FILE_ENRICHED_XLSX"],
        ref_year=int(cfg["REFERENCE_YEAR"]),
    )

    # ---------- STEP B: Supplier FE ----------
    fe_panel = load_supplier_fe_table(cfg["FILE_FE_SUPPLIER"])
    panel = base_panel.merge(fe_panel, on="supplier", how="left")

    n_sup_total = int(panel["supplier"].nunique())
    n_sup_fe = int(panel["fe_log10"].notna().sum())
    fe_cov = float(n_sup_fe / n_sup_total) if n_sup_total > 0 else np.nan

    # ---------- STEP C: media_result-based demand exposure + app-domain concentration ----------
    media_df = load_media_table(
        file_media_xlsx=cfg["FILE_MEDIA_XLSX"],
        use_log1p=bool(cfg.get("MEDIA_USE_LOG1P", True))
    )
    supplier_media_panel, media_diag = build_supplier_media_panel_from_items(item_df=item_df, media_df=media_df)

    if supplier_media_panel is not None and len(supplier_media_panel) > 0:
        panel = panel.merge(supplier_media_panel, on="supplier", how="left")
    else:
        media_diag = dict(media_diag) if isinstance(media_diag, dict) else {}
        media_diag["warning"] = "supplier_media_panel is empty; media-based proxies unavailable"

    # ---------- STEP D: Build proxy-based indication panel ----------
    panel = build_market_power_indications(panel, cfg)

    # Sort for easier inspection
    sort_col = "premium_log10_effect" if "premium_log10_effect" in panel.columns else "y_log10_median"
    if sort_col in panel.columns:
        panel = panel.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # ---------- STEP E: Spearman (Table 11 source) ----------
    spearman_candidates = [
        # Barrier/resource proxies
        "exclusivity_rate", "gov_share", "listed_share", "anonymous_rate",
        "firm_age", "log_total_asset_wan", "disclosure_years",
        # Supplier footprint
        "n_products", "overall_count_share",
        # Application-domain structure proxies
        "app_domain_hhi", "max_app_share", "n_apps_covered_in_media_map", "single_app_focus_flag",
        # Demand visibility proxies from media_result.xlsx
        "media_match_ratio", "exp_media_web", "exp_media_news",
        "exp_media_weixin_account", "exp_media_weixin_article", "exp_media_total",
        # Composite scores
        "score_premium", "score_structure", "score_barrier", "score_demand_visibility",
        "market_power_signal_score",
    ]
    spearman_df = run_spearman_table(panel, y_col="fe_log10", x_cols=spearman_candidates)

    # ---------- STEP F: HC3 OLS (Table 12 source) ----------
    ols_coef_tables = []
    model_summaries = {}

    # Model 1: Barrier/resource proxies only
    x1 = [
        "exclusivity_rate", "gov_share", "listed_share", "anonymous_rate",
        "firm_age", "log_total_asset_wan", "n_products"
    ]
    res1, coef1 = fit_ols_hc3(
        panel, y_col="fe_log10", x_cols=x1,
        model_name="FE_on_barrier_only",
        min_obs=int(cfg["MIN_OBS_FOR_OLS"])
    )
    if res1 is not None and len(coef1) > 0:
        ols_coef_tables.append(coef1)
        model_summaries["FE_on_barrier_only"] = res1.summary().as_text()

    # Model 2: Barrier + media demand exposure + structure
    x2 = x1 + ["exp_media_total", "app_domain_hhi", "max_app_share", "media_match_ratio"]
    res2, coef2 = fit_ols_hc3(
        panel, y_col="fe_log10", x_cols=x2,
        model_name="FE_on_barrier_plus_media_structure",
        min_obs=int(cfg["MIN_OBS_FOR_OLS"])
    )
    if res2 is not None and len(coef2) > 0:
        ols_coef_tables.append(coef2)
        model_summaries["FE_on_barrier_plus_media_structure"] = res2.summary().as_text()

    # Model 3: Fuller proxy set (descriptive extension)
    x3 = [
        "exclusivity_rate", "gov_share", "listed_share", "anonymous_rate",
        "firm_age", "log_total_asset_wan", "disclosure_years",
        "n_products", "overall_count_share",
        "app_domain_hhi", "max_app_share", "n_apps_covered_in_media_map", "single_app_focus_flag",
        "media_match_ratio", "exp_media_web", "exp_media_news", "exp_media_weixin_article", "exp_media_total",
    ]
    res3, coef3 = fit_ols_hc3(
        panel, y_col="fe_log10", x_cols=x3,
        model_name="FE_on_full_proxy_set_media",
        min_obs=int(cfg["MIN_OBS_FOR_OLS"])
    )
    if res3 is not None and len(coef3) > 0:
        ols_coef_tables.append(coef3)
        model_summaries["FE_on_full_proxy_set_media"] = res3.summary().as_text()

    ols_df = pd.concat(ols_coef_tables, axis=0, ignore_index=True) if ols_coef_tables else pd.DataFrame(
        columns=["model", "term", "coef", "std_err_hc3", "t", "pvalue", "n_obs", "r2", "adj_r2"]
    )

    # ---------- STEP G: Overview ----------
    overview = {
        "n_suppliers_total": n_sup_total,
        "n_suppliers_with_fe": n_sup_fe,
        "supplier_fe_coverage_ratio": fe_cov,
        "app_column_used_in_enriched": app_col_used,
        "premium_source_counts": panel["premium_source"].value_counts(dropna=False).to_dict() if "premium_source" in panel.columns else {},
        "market_power_tier_v4_counts": panel["market_power_tier_v4"].value_counts(dropna=False).to_dict() if "market_power_tier_v4" in panel.columns else {},
        "media_join_diagnostics": media_diag,
        "notes": [
            "supplier FE (fe_log10) is treated as a reduced-form supplier premium, not a structural markup estimate.",
            "Tier labels are sample-internal and proxy-based, not formal antitrust market-power classifications.",
            "Results are interpretive/descriptive and supplementary for explaining supplier anchoring, not causal identification.",
            "Demand proxies are derived from media_result.xlsx at the application-domain keyword level and aggregated to supplier level."
        ],
    }

    # ---------- STEP H: Core exports ----------
    panel_csv = os.path.join(out_dir, "supplier_fe_market_power_panel_v4.csv")
    panel.to_csv(panel_csv, index=False, encoding="utf-8-sig")

    spearman_csv = os.path.join(out_dir, "supplier_fe_proxy_spearman_v4.csv")
    spearman_df.to_csv(spearman_csv, index=False, encoding="utf-8-sig")

    ols_csv = os.path.join(out_dir, "supplier_fe_proxy_ols_hc3_v4.csv")
    ols_df.to_csv(ols_csv, index=False, encoding="utf-8-sig")

    with open(os.path.join(out_dir, "overview_v4.json"), "w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    for mname, txt in model_summaries.items():
        with open(os.path.join(out_dir, f"{mname}.txt"), "w", encoding="utf-8") as f:
            f.write(txt)

    # ---------- STEP H2: Paper-mapped outputs for Section 4.2.3 ----------
    table11_base = cfg.get("PAPER_TABLE11_BASENAME", "Table11_supplier_fe_proxy_spearman_v4")
    table12_base = cfg.get("PAPER_TABLE12_BASENAME", "Table12_supplier_fe_proxy_ols_hc3_v4")

    table11_csv = os.path.join(out_dir, f"{table11_base}.csv")
    table11_xlsx = os.path.join(out_dir, f"{table11_base}.xlsx")
    table12_csv = os.path.join(out_dir, f"{table12_base}.csv")
    table12_xlsx = os.path.join(out_dir, f"{table12_base}.xlsx")

    spearman_df.to_csv(table11_csv, index=False, encoding="utf-8-sig")
    ols_df.to_csv(table12_csv, index=False, encoding="utf-8-sig")

    if cfg.get("WRITE_XLSX", True):
        with pd.ExcelWriter(table11_xlsx, engine="openpyxl") as writer:
            spearman_df.to_excel(writer, sheet_name="Table11", index=False)
        with pd.ExcelWriter(table12_xlsx, engine="openpyxl") as writer:
            ols_df.to_excel(writer, sheet_name="Table12", index=False)

    # Fig. 8 (paper figure for Section 4.2.3)
    fig8_info = plot_fig8_supplier_fe_by_tier(panel=panel, out_dir=out_dir, cfg=cfg)

    paper_titles = {
        "Table11": {
            "number": "Table 11",
            "title": "Supplier FE and proxy-based market-power indications: Spearman rank correlations (supplier-level panel)",
            "files": {
                "csv": table11_csv,
                "xlsx": table11_xlsx if cfg.get("WRITE_XLSX", True) else None,
            }
        },
        "Table12": {
            "number": "Table 12",
            "title": "Supplier FE and proxy-based market-power indications: HC3-robust OLS associations (supplier-level panel)",
            "files": {
                "csv": table12_csv,
                "xlsx": table12_xlsx if cfg.get("WRITE_XLSX", True) else None,
            }
        },
        "Fig8": {
            "number": "Fig. 8",
            "title": "Supplier FE across proxy-based market-power indication tiers (descriptive evidence)",
            "files": fig8_info if fig8_info is not None else {},
        }
    }
    with open(os.path.join(out_dir, "paper_artifact_titles_4_2_3_v4.json"), "w", encoding="utf-8") as f:
        json.dump(paper_titles, f, ensure_ascii=False, indent=2)

    # ---------- STEP H3: Consolidated Excel workbook ----------
    if cfg.get("WRITE_XLSX", True):
        xlsx_path = os.path.join(out_dir, "supplier_fe_market_power_results_v4.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            panel.to_excel(writer, sheet_name="supplier_panel_v4", index=False)
            spearman_df.to_excel(writer, sheet_name="spearman_v4", index=False)
            ols_df.to_excel(writer, sheet_name="ols_hc3_v4", index=False)
            media_df.to_excel(writer, sheet_name="media_keyword_table", index=False)

            key_cols = [c for c in [
                "supplier", "fe_log10", "fe_multiplier_on_price", "premium_multiplier_vs_baseline",
                "n_products", "price_mean", "price_median",
                "exclusivity_rate", "gov_share", "listed_share", "firm_age", "log_total_asset_wan",
                "app_domain_hhi", "max_app_share", "n_apps_covered_in_media_map",
                "media_match_ratio", "exp_media_web", "exp_media_news", "exp_media_weixin_article", "exp_media_total",
                "score_premium", "score_structure", "score_barrier", "score_demand_visibility",
                "market_power_signal_score", "market_power_tier_v4", "market_power_tier_en_v4", "market_power_reason_v4"
            ] if c in panel.columns]
            panel[key_cols].to_excel(writer, sheet_name="panel_key_cols", index=False)

            ov_df = pd.DataFrame({
                "key": list(overview.keys()),
                "value": [json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for v in overview.values()]
            })
            ov_df.to_excel(writer, sheet_name="overview", index=False)

    # ---------- STEP I: Console summary ----------
    print(f"- n_suppliers_total: {overview['n_suppliers_total']}")
    print(f"- n_suppliers_with_fe: {overview['n_suppliers_with_fe']}")
    print(f"- supplier_fe_coverage_ratio: {overview['supplier_fe_coverage_ratio']}")
    print(f"- app_column_used_in_enriched: {overview['app_column_used_in_enriched']}")
    print(f"- premium_source_counts: {overview['premium_source_counts']}")
    print(f"- market_power_tier_v4_counts: {overview['market_power_tier_v4_counts']}")
    print(f"- media_join_diagnostics: {overview['media_join_diagnostics']}")

    print("- Paper mapping for Section 4.2.3:")
    print(f"  * Table 11 -> {table11_csv}")
    print(f"  * Table 12 -> {table12_csv}")
    if fig8_info is not None:
        print(f"  * Fig. 8  -> {fig8_info.get('png')}")
        if fig8_info.get('pdf'):
            print(f"             {fig8_info.get('pdf')}")
        print(f"    Fig. 8 data -> {fig8_info.get('data_csv')}")
    else:
        print("  * Fig. 8  -> skipped (check logs / missing tier data)")

    print("=" * 88)
    print("[Done] supplier FE × proxy-based market-power indications (v4, media_result version)")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()