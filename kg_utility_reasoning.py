# -*- coding: utf-8 -*-
"""
kg_fe_metapath_ppr_causal_suite_v7_man.py

Pipeline (as requested):
STEP0) Load price labels, match to KG (Supplier-DataProduct-src/app industries), build base dataset.
       - Evaluate explanatory / predictive power decomposition for 4 blocks:
         (1) FE: supplier/src/app fixed effects
         (2) PRODUCT: text embedding + similarity features + topic_id
         (3) DEMAND: heat / expected_heat proxies
         (4) STRUCTURE: HHI/CR4/CR8/n_suppliers/monopoly/network/thickness proxies
       - Target: y = log10(price)
       - Metrics: CI95 coverage, MAE, MAPE, R2, RMSE, SMAPE
       - Use EXACT same train/test index splits as STEP1.
STEP1) Fixed-effect (FE) diagnosis + keep explainability of FE blocks (supplier/src/app):
       - FE ablation (8 scenarios) under KFold and supplier-GroupKFold
       - Shapley (3-block exact) + Drop-one
       - (Optional) Hierarchical Bayes crossed RE (auto-skip if PyMC unavailable)
       - Extract interpretable FE estimates (full-sample BayesianRidge) for supplier/src/app
STEP2) Meta-path / Personalized PageRank (PPR) reasoning stage (keep original reasoning as much as possible):
       - Original heterogeneous-graph PPR (Supplier-Product-App/Src) + media-heat prior
       - Meta-path constrained PPR on bipartite/projection graphs (SUP-APP, SUP-SRC, SRC-APP)
       - Build “mechanism proxies” (supply/demand/network structure) for:
            * supplier
            * src industry
            * app industry
       - Explain FE components (supplier FE / src FE / app FE) with mechanism proxies (entity-level CV, no leakage)
       - Predictive validation for product price (pipeline CV; preprocessing inside folds to avoid leakage)
STEP3) Product-level reasoning (similar to STEP2):
       - Build mechanism proxies for products (e.g., product-specific PPR, network metrics, embeddings)
       - Explain product-level variations using proxies
       - Predictive validation integration
STEP4) Mechanism-chain analysis (mechanism-consistent evidence; NOT “hard causality”):
       - Supplier-level causal chain (as before)
       - Src-industry-level causal chain
       - App-industry-level causal chain
STEP5) Main factor model analysis
       - supplier FE + Product/Competitor factor as main factors
       - the final model

ANTI-LEAKAGE RULES implemented:
- Any feature preprocessing that learns from X distribution (OHE vocab, multi-hot vocab, scaling) is inside CV pipelines.
- No price-derived constructs (e.g., “brand premium” from y) are used as predictors in CV tasks.
- Step2 explanatory regressions (FE -> mechanisms) are entity-level CV; features do not use y.

Output filenames are prefixed with STEP{n}_ as required.

"""

import os
import re
import ast
import json
import math
import hashlib
import warnings
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from py2neo import Graph

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy import stats

warnings.filterwarnings("ignore")


# =============================================================================
# External default configuration (ALL external variables at the top)
# =============================================================================
NEO4J_URI = ""
NEO4J_USER = ""
NEO4J_PASS = ""
NEO4J_DB = ""

EXCEL_PATH = "anymous/name_price_anonymized.xlsx"
MEDIA_PATH = "anymous/media_result.xlsx"  # optional; used for demand heat
OUTPUT_PATH = "result_kg_reproduce"
os.makedirs(OUTPUT_PATH, exist_ok=True)

PRICE_COL = "price"

# CV controls
RANDOM_STATE = 42
N_SPLITS = 5

# Predictive interval controls (Normal approx for BayesianRidge)
Z_95 = 1.959963984540054

# Demand transform: log1p(heat+1) style
USE_LOG1P_HEAT = True

# Original heterogeneous-graph PPR controls
PPR_ALPHA = 0.85
PPR_TOPK_APPS = 10
PPR_TOPK_SRCS = 10
PPR_PRIOR_STRENGTH = 1.0  # media-heat prior strength

# Meta-path (constrained) PPR controls
METAPATH_ALPHA = 0.85
METAPATH_TOPK = 10
METAPATH_PRIOR_STRENGTH_APPHEAT = 1.0

# =============================================================================
# Heat usage ablation (Structural vs Teleport-only heat bias)
#   - Structural reasoning: PPR/MPPR uses ONLY KG structure (no heat in teleport/personalization or edge weights).
#   - Heat-biased reasoning: heat ONLY enters personalization (teleport distribution), NOT post-hoc posterior reweighting.
# Notes:
#   - Existing infer_*_apps() methods in KGReasoner and MetaPathPPRReasoner keep legacy "post-hoc heat prior" behavior.
#   - New *_structural() and *_heat_teleport() methods implement the requested strict split.
# =============================================================================
HEAT_ABLATION_ENABLE_DEFAULT = True

# Teleport (personalization) mixture weight:
#   personalization = (1-mass)*delta(start_node) + mass*heat_prior(APP)
PPR_HEAT_TELEPORT_MASS = 0.35
METAPATH_HEAT_TELEPORT_MASS = 0.35

# =============================================================================
# HUB collapse mitigation patch (minimal invasive)
# =============================================================================
HUB_PATCH_ENABLE = True

# Degree debias strength: 0.0 = off; 0.5 = sqrt penalty; 1.0 = strong penalty
HUB_DEBIAS_BETA = 0.5

# Clip heat z-score to avoid prior dominating everything
HEAT_Z_CLIP = 2.0

# MetaPath bipartite edge reweighting (optional but recommended)
BIPARTITE_EDGE_LOG1P = True
BIPARTITE_EDGE_IDF = True
BIPARTITE_EDGE_IDF_SMOOTH = 1.0  # smoothing to avoid div-by-zero

# Posterior embedding dims (cheap deterministic embedding)
SUP_EMB_DIM_APP = 16
SUP_EMB_DIM_SRC = 8
SRC_EMB_DIM_APP = 16
APP_EMB_DIM_SRC = 16

# Hierarchical Bayes controls (optional; will auto-skip if PyMC is not available)
FIT_HIERARCHICAL_BAYES = True
PYMC_DRAWS = 1200
PYMC_TUNE = 1200
PYMC_CHAINS = 2
PYMC_TARGET_ACCEPT = 0.9

# Mechanism chain controls
BOOTSTRAP_B = 300
DML_SPLITS = 5

# FE extraction controls
FE_MIN_LABEL_FREQ_FOR_TABLE = 1

# Efficiency / safety
MAX_NX_PAGERANK_ITERS = 200
NX_PAGERANK_TOL = 1e-8

# ---- Text / similarity / topics (Method1/2/3) ----
USE_METHOD1_TEXT = True
USE_METHOD2_TEXTSIM_GRAPH = True
USE_METHOD3_TOPICS = True

TEXT_EMB_DIM = 50
TEXT_MIN_DF = 2

SIM_KNN_K = 10          # each product connects to topK neighbors
SIM_MIN_COS = 0.25      # filter weak similarity edges
TOPIC_K = 20            # kmeans topics


# =============================================================================
# Small utility functions
# =============================================================================
def safe_log10(x: float) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan

    if x <= 0:
        return np.nan

    return float(np.log10(x))

def inv_log10(arr: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(arr, dtype=float))

def log10_1p(x):
    """Compute log10(1+x) elementwise for scalars/arrays."""
    x = np.asarray(x, dtype=float)
    return np.log10(np.clip(x, 0.0, None) + 1.0)

def inv_log10_1p(x):
    """Inverse of log10(1+x): returns (10**x - 1) elementwise."""
    x = np.asarray(x, dtype=float)
    # clip to avoid overflow in extreme cases
    x = np.clip(x, -100.0, 100.0)
    return np.power(10.0, x) - 1.0

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom))

def ci95_coverage_log10(y_true_log10: np.ndarray, mu_log10: np.ndarray, std_log10: np.ndarray) -> float:
    lo = mu_log10 - Z_95 * std_log10
    hi = mu_log10 + Z_95 * std_log10
    y_true_log10 = np.asarray(y_true_log10, dtype=float)
    return float(np.mean((y_true_log10 >= lo) & (y_true_log10 <= hi)))

def ci95_coverage_price_from_log10(y_true_price: np.ndarray, mu_log10: np.ndarray, std_log10: np.ndarray) -> float:
    lo = inv_log10(mu_log10 - Z_95 * std_log10)
    hi = inv_log10(mu_log10 + Z_95 * std_log10)
    y_true_price = np.asarray(y_true_price, dtype=float)
    return float(np.mean((y_true_price >= lo) & (y_true_price <= hi)))

def rmse_from_log10(y_true, y_pred, eps=1e-8):
    y_true_log10 = np.asarray(y_true, dtype=float)
    y_pred_log10 = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(y_true_log10, y_pred_log10)))

def metrics_on_log10_and_price(y_true_log10: np.ndarray, y_pred_log10: np.ndarray, y_std_log10: Optional[np.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["R2_log10"] = float(r2_score(y_true_log10, y_pred_log10))
    out["MAE_log10"] = float(mean_absolute_error(y_true_log10, y_pred_log10))
    out["RMSE_log10"] = float(np.sqrt(mean_squared_error(y_true_log10, y_pred_log10)))
    out["MAPE_log10"] = mape(y_true_log10, y_pred_log10)
    out["SMAPE_log10"] = smape(y_true_log10, y_pred_log10)

    y_true_p = inv_log10(y_true_log10)
    y_pred_p = inv_log10(y_pred_log10)
    out["MAE_price"] = float(mean_absolute_error(y_true_p, y_pred_p))
    out["RMSE_price"] = float(np.sqrt(mean_squared_error(y_true_p, y_pred_p)))
    out["MAPE_price"] = mape(y_true_p, y_pred_p)
    out["SMAPE_price"] = smape(y_true_p, y_pred_p)

    if y_std_log10 is not None:
        out["CI95_coverage_log10"] = ci95_coverage_log10(y_true_log10, y_pred_log10, y_std_log10)
        out["CI95_coverage_price"] = ci95_coverage_price_from_log10(y_true_p, y_pred_log10, y_std_log10)
    else:
        out["CI95_coverage_log10"] = np.nan
        out["CI95_coverage_price"] = np.nan
    return out

def select_numeric_feature_cols(df: pd.DataFrame, id_col: str, drop_prefixes: Tuple[str, ...] = ()) -> List[str]:
    """Select numeric feature columns excluding id_col and optionally excluding prefix families."""
    cols = []
    for c in df.columns:
        if c == id_col:
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        if drop_prefixes and str(c).startswith(drop_prefixes):
            continue
        cols.append(c)
    return cols

def add_fractional_label_mechanism_features(
    df_rows: pd.DataFrame,
    list_col: str,
    mech_df: pd.DataFrame,
    mech_key: str,
    feature_cols: List[str],
    prefix: str,
) -> pd.DataFrame:
    """
    Convert label-level mechanism table to row-level numeric features by fractional averaging over list_col.

    For a row with k labels, each label has weight 1/k.
    Missing labels are ignored and weights are renormalized over matched labels.
    """
    if df_rows is None or df_rows.empty or mech_df is None or mech_df.empty or not feature_cols:
        return pd.DataFrame(index=df_rows.index)

    m = mech_df[[mech_key] + feature_cols].copy()
    m[mech_key] = m[mech_key].astype(str).str.strip()

    tmp = df_rows[[list_col]].copy()
    tmp["_row_id"] = np.arange(len(tmp))
    tmp["_labs"] = tmp[list_col].apply(parse_list_cell)
    tmp["_k"] = tmp["_labs"].apply(len).astype(int)

    tmp = tmp.explode("_labs")
    tmp = tmp[tmp["_k"] > 0].copy()
    tmp["_w"] = 1.0 / tmp["_k"].astype(float)
    tmp["_labs"] = tmp["_labs"].astype(str).str.strip()

    tmp = tmp.merge(m, left_on="_labs", right_on=mech_key, how="left")

    # matched label indicator for renormalization
    tmp["_has"] = tmp[mech_key].notna().astype(float)
    tmp["_w_has"] = tmp["_w"] * tmp["_has"]

    for c in feature_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        tmp[c] = tmp[c].fillna(0.0) * tmp["_w"]

    agg_num = tmp.groupby("_row_id")[feature_cols].sum()
    agg_den = tmp.groupby("_row_id")["_w_has"].sum().replace(0.0, np.nan)

    out = agg_num.div(agg_den, axis=0).fillna(0.0)

    out = out.reindex(range(len(df_rows))).fillna(0.0)
    out = out.add_prefix(prefix)
    out.index = df_rows.index
    return out

def parse_list_cell(x: Any) -> List[str]:
    """
    Parse list-like cells. Accepts:
    - real list
    - stringified list: "['a','b']"
    - delimited strings: "a;b" / "a,b" / "a|b"
    - empty / nan -> []
    """
    if x is None:
        return []

    if isinstance(x, list):
        return [str(t).strip() for t in x if str(t).strip()]

    if isinstance(x, float) and np.isnan(x):
        return []

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return []

    try:
        v = ast.literal_eval(s)

        if isinstance(v, list):
            return [str(t).strip() for t in v if str(t).strip()]

    except Exception:
        pass

    parts = [p.strip() for p in re.split(r"[;,|/]+", s) if p.strip()]
    return parts

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def _safe_col(prefix: str, raw: str) -> str:
    """Create safe column name for Chinese labels (hash + readable prefix)."""
    raw = "" if raw is None else str(raw)
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
    readable = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", raw)[:20]
    return f"{prefix}__{readable}__{h}"

def build_fractional_multihot_df(series_lists: pd.Series, prefix: str, min_freq: int = 1) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Turn list-valued labels into fractional multi-hot:
      If a row has k labels -> each label gets weight 1/k.
    """
    rows = [parse_list_cell(x) for x in series_lists.tolist()]
    from collections import Counter
    cnt = Counter([lab for r in rows for lab in r])
    labels = sorted([lab for lab, c in cnt.items() if c >= min_freq])
    mapping = {lab: _safe_col(prefix, lab) for lab in labels}

    X = pd.DataFrame(0.0, index=range(len(rows)), columns=[mapping[lab] for lab in labels], dtype=float)

    for i, r in enumerate(rows):
        active = [lab for lab in r if lab in mapping]

        if not active:
            continue

        w = 1.0 / float(len(active))

        for lab in active:
            X.at[i, mapping[lab]] = w

    return X, mapping

# =============================================================================
# Build fixed split, for for reproduction (CV-safe: fit on train fold only)
# =============================================================================
def build_fixed_splits(df: pd.DataFrame, n_splits: int, random_state: int) -> Tuple[List[Dict[str, List[int]]], List[Dict[str, List[int]]]]:
    """
    Build and return precomputed splits (so STEP0 and STEP1 share EXACT same indices).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds_k = [{"train_idx": tr.tolist(), "test_idx": te.tolist()} for tr, te in kf.split(df)]

    groups = df["supplier"].astype(str).fillna("NA").values
    gk = GroupKFold(n_splits=min(n_splits, max(2, len(np.unique(groups)))))
    folds_g = [{"train_idx": tr.tolist(), "test_idx": te.tolist()} for tr, te in gk.split(df, df["y_log10"].values, groups)]

    return folds_k, folds_g

def evaluate_fe_on_fixed_splits(
    X: pd.DataFrame,
    y_log10: np.ndarray,
    folds: List[Dict[str, List[int]]],
    tag: str,
    min_freq: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for scen_name, cfg in SCENARIOS_FE.items():
        pipe_proto = make_fe_pipeline(cfg["supplier"], cfg["src"], cfg["app"], min_freq=min_freq)
        for fold_id, fd in enumerate(folds, start=1):
            tr = np.array(fd["train_idx"], dtype=int)
            te = np.array(fd["test_idx"], dtype=int)

            Xtr = X.iloc[tr].copy()
            Xte = X.iloc[te].copy()
            ytr = y_log10[tr].astype(float)
            yte = y_log10[te].astype(float)

            pipe = clone(pipe_proto)
            pipe.fit(Xtr, ytr)
            yhat, ystd = pipe.predict(Xte, return_std=True)

            met = metrics_on_log10_and_price(yte, yhat, ystd)
            rows.append({"cv_tag": tag, "scenario": scen_name, "fold": fold_id, **met})

    cv_long = pd.DataFrame(rows)
    agg = {c: ["mean", "std"] for c in cv_long.columns if c not in {"cv_tag", "scenario", "fold"}}
    summary = cv_long.groupby(["cv_tag", "scenario"], as_index=False).agg(agg)
    summary.columns = [(c[0] if c[1] == "" else f"{c[0]}_{c[1]}") for c in summary.columns]
    return cv_long, summary

# =============================================================================
# Product block features (CV-safe: fit on train fold only)
# =============================================================================
def fit_text_embedder(train_text: np.ndarray, dim: int, min_df: int) -> Tuple[TfidfVectorizer, Optional[TruncatedSVD]]:
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=min_df)
    X = tfidf.fit_transform(train_text)

    n_samples, n_features = X.shape
    if n_samples < 3 or n_features < 3:
        return tfidf, None

    n_comp = int(min(dim, n_samples - 1, n_features - 1))
    n_comp = max(2, n_comp)
    svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
    svd.fit(X)
    return tfidf, svd

def transform_text_embedder(tfidf: TfidfVectorizer, svd: Optional[TruncatedSVD], text: np.ndarray) -> np.ndarray:
    X = tfidf.transform(text)
    if svd is None:
        return np.zeros((len(text), 0), dtype=float)
    return svd.transform(X).astype(float)

def knn_similarity_features(X_train: np.ndarray, X_any: np.ndarray, k: int) -> np.ndarray:
    """
    For each sample in X_any, compute similarities to top-k neighbors in TRAIN set (cosine).
    Output columns: [sim_mean, sim_max, sim_std, sim_sum]
    """
    if X_train.shape[0] < 2 or X_train.shape[1] == 0:
        return np.zeros((X_any.shape[0], 4), dtype=float)

    k_use = int(min(max(1, k), X_train.shape[0]))
    nn = NearestNeighbors(n_neighbors=k_use, metric="cosine", algorithm="brute")
    nn.fit(X_train)
    dist, _ = nn.kneighbors(X_any, return_distance=True)
    sim = 1.0 - dist
    sim_mean = sim.mean(axis=1)
    sim_max = sim.max(axis=1)
    sim_std = sim.std(axis=1)
    sim_sum = sim.sum(axis=1)
    return np.column_stack([sim_mean, sim_max, sim_std, sim_sum]).astype(float)

# =============================================================================
# Competitor / neighbor reasoning helpers (desc-based)
# =============================================================================
def knn_neighbors_exclude_self(X_train: np.ndarray, X_any: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (nbr_idx, nbr_sim) for each row in X_any, neighbors are chosen from TRAIN.
    Cosine similarity = 1 - cosine_distance.
    If X_any is the same array as X_train, this function excludes self-neighbor.
    """
    if X_train.shape[0] < 2 or X_train.shape[1] == 0 or X_any.shape[0] == 0:
        return np.zeros((X_any.shape[0], 0), dtype=int), np.zeros((X_any.shape[0], 0), dtype=float)

    n_train = X_train.shape[0]
    k_use = int(min(max(1, k), n_train - 1))
    nn = NearestNeighbors(n_neighbors=min(k_use + 1, n_train), metric="cosine", algorithm="brute")
    nn.fit(X_train)
    dist, idx = nn.kneighbors(X_any, return_distance=True)
    sim = 1.0 - dist

    # Exclude self if X_any points into X_train with exact match (typical for train->train)
    # We drop the first neighbor when it is self with sim ~ 1.
    if X_any.shape[0] == n_train and np.allclose(X_any, X_train):
        idx2 = []
        sim2 = []
        for i in range(n_train):
            # drop neighbor where idx==i and distance==0 (sim==1)
            order = [(idx[i, j], sim[i, j]) for j in range(idx.shape[1]) if int(idx[i, j]) != i]
            order = order[:k_use]
            if order:
                idx2.append([p[0] for p in order])
                sim2.append([p[1] for p in order])
            else:
                idx2.append([])
                sim2.append([])
        maxk = max([len(x) for x in idx2]) if idx2 else 0
        idx_arr = np.zeros((n_train, maxk), dtype=int)
        sim_arr = np.zeros((n_train, maxk), dtype=float)
        for i in range(n_train):
            for j in range(len(idx2[i])):
                idx_arr[i, j] = int(idx2[i][j])
                sim_arr[i, j] = float(sim2[i][j])
        return idx_arr, sim_arr

    # For general X_any, keep top-k_use (exclude the last column if nn returned k+1)
    idx = idx[:, :k_use]
    sim = sim[:, :k_use]
    return idx.astype(int), sim.astype(float)

def neighbor_label_posterior(
    neighbor_label_lists: List[List[str]],
    neighbor_weights: np.ndarray,
) -> Dict[str, float]:
    """
    Build a posterior over labels from neighbors.
    Each neighbor contributes weight * (1/k_labels) to each of its labels.
    """
    post: Dict[str, float] = {}
    for labs, w in zip(neighbor_label_lists, neighbor_weights):
        labs = [str(x).strip() for x in (labs or []) if str(x).strip()]
        if not labs:
            continue
        ww = float(max(w, 0.0)) / float(len(labs))
        for lab in labs:
            post[lab] = post.get(lab, 0.0) + ww

    s = float(sum(post.values()))
    if s <= 1e-12:
        return {}
    return {k: float(v / s) for k, v in sorted(post.items(), key=lambda kv: kv[1], reverse=True)}

def posterior_entropy(post: Dict[str, float]) -> float:
    if not post:
        return 0.0
    p = np.array(list(post.values()), dtype=float)
    p = p / (p.sum() + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))

def expected_from_heat_map(post: Dict[str, float], heat_map: Dict[str, float]) -> float:
    if not post:
        return 0.0
    return float(sum(float(w) * float(heat_map.get(k, 0.0)) for k, w in post.items()))

def topk_keys_json(post: Dict[str, float], topk: int = 10) -> str:
    if not post:
        return json.dumps({}, ensure_ascii=False)
    items = list(sorted(post.items(), key=lambda kv: kv[1], reverse=True))[: int(topk)]
    return json.dumps({k: float(v) for k, v in items}, ensure_ascii=False)

def fit_topics(train_emb: np.ndarray, k_topics: int) -> KMeans:
    n = train_emb.shape[0]
    if n < 10 or train_emb.shape[1] == 0:
        # fallback: 2 clusters if possible
        k_topics = int(min(2, max(1, n)))
    else:
        k_topics = int(min(k_topics, max(2, n // 5), n))
    km = KMeans(n_clusters=max(1, k_topics), random_state=RANDOM_STATE, n_init=10)
    km.fit(train_emb)
    return km

def _save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# =============================================================================
# Sparse->dense adapter (BayesianRidge expects dense arrays)
# =============================================================================
class ToDense(BaseEstimator, TransformerMixin):
    """Convert scipy sparse matrix to dense numpy array."""
    def __init__(self):
        pass

    def get_params(self, deep: bool = True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            from scipy import sparse
            if sparse.issparse(X):
                return X.toarray()
        except Exception:
            pass
        return np.asarray(X)

# =============================================================================
# MultiHotEncoder for ColumnTransformer (list-valued) - CV-safe vocab learning
# =============================================================================
class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible multi-hot encoder for list-valued categorical features.

    - normalize: if True, each active label gets weight 1/k within a row
    - min_freq: keep labels whose global frequency >= min_freq (fit-time)
    - prefix: used in get_feature_names_out() for interpretability

    Important:
    - implements get_params/set_params so sklearn.clone() works in CV.
    """

    def __init__(self, normalize: bool = True, min_freq: int = 1, prefix: str = ""):
        self.normalize = bool(normalize)
        self.min_freq = int(min_freq)
        self.prefix = str(prefix)

        self.vocab_ = None
        self.index_ = None

    def get_params(self, deep: bool = True):
        return {"normalize": self.normalize, "min_freq": self.min_freq, "prefix": self.prefix}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

        return self

    def fit(self, X, y=None):
        from collections import Counter

        col = self._to_series(X)

        counter = Counter()
        for cell in col:
            toks = parse_list_cell(cell)
            counter.update([t for t in toks if t])

        vocab = [t for t, c in counter.items() if c >= max(1, self.min_freq)]
        self.vocab_ = sorted(vocab)
        self.index_ = {t: i for i, t in enumerate(self.vocab_)}
        return self

    def transform(self, X):
        from scipy.sparse import csr_matrix

        col = self._to_series(X)
        n = len(col)
        m = 0 if (self.vocab_ is None) else len(self.vocab_)

        if m == 0:
            return csr_matrix((n, 0), dtype=np.float64)

        rows, cols, data = [], [], []
        for i, cell in enumerate(col):
            toks = [t for t in parse_list_cell(cell) if t in self.index_]

            if not toks:
                continue

            w = 1.0 / len(toks) if (self.normalize and len(toks) > 0) else 1.0

            for t in toks:
                rows.append(i)
                cols.append(self.index_[t])
                data.append(w)

        return csr_matrix((data, (rows, cols)), shape=(n, m), dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        if self.vocab_ is None:
            return np.array([], dtype=object)

        if self.prefix:
            return np.array([f"{self.prefix}__{t}" for t in self.vocab_], dtype=object)

        return np.array(list(self.vocab_), dtype=object)

    @staticmethod
    def _to_series(X):
        import numpy as np
        import pandas as pd

        if isinstance(X, pd.Series):
            return X

        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:

                raise ValueError("MultiHotEncoder expects a single-column DataFrame.")
            return X.iloc[:, 0]

        arr = np.asarray(X)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]

        return pd.Series(arr)

# =============================================================================
# Neo4j access layer
# =============================================================================
class Neo4jDB:
    def __init__(self, uri: str, user: str, password: str, db_name: str):
        self.g = None
        try:
            self.g = Graph(uri, auth=(user, password), name=db_name)
            self.g.run("RETURN 1").data()
            print(f"[OK] Connected to Neo4j database: {db_name}")
        except Exception as e:
            print(f"[FATAL] Neo4j connection failed: {e}")
            self.g = None

    def fetch_matched_multilabel(self, df_price: pd.DataFrame) -> pd.DataFrame:
        """
        Match (supplier, product) pairs from price table to KG and return src_list/app_list.
        """
        if not self.g:
            return pd.DataFrame()

        pairs = [{"name": str(r["name"]), "supplier": str(r["supplier"])} for _, r in df_price.iterrows()]
        cypher = """
        UNWIND $pairs AS row
        MATCH (s:Supplier {name: row.supplier})-[:provide_data]->(p:DataProduct {name: row.name})
        OPTIONAL MATCH (src:src_IndustryCategory)-[:source_industry]->(p)
        OPTIONAL MATCH (p)-[:applied_to]->(app:app_IndustryCategory)               
        RETURN p.name AS name,
                s.name AS supplier,
                p.desc AS desc,
                collect(DISTINCT src.name) AS src_list,
                collect(DISTINCT app.name) AS app_list
        """

        return pd.DataFrame(self.g.run(cypher, pairs=pairs).data())

    def fetch_supplier_product_app_src_edges(self, suppliers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Edges for reasoning graph: supplier -> product -> app/src.
        If suppliers is provided, restrict to those suppliers to reduce size.
        """
        if not self.g:
            return pd.DataFrame()

        if suppliers:
            cypher = """
            MATCH (s:Supplier)-[:provide_data]->(p:DataProduct)
            WHERE s.name IN $suppliers
            OPTIONAL MATCH (p)-[:applied_to]->(a:app_IndustryCategory)
            OPTIONAL MATCH (src:src_IndustryCategory)-[:source_industry]->(p)
            RETURN s.name AS supplier, p.name AS product,
                   collect(DISTINCT a.name) AS app_list,
                   collect(DISTINCT src.name) AS src_list
            """
            return pd.DataFrame(self.g.run(cypher, suppliers=suppliers).data())
        else:
            cypher = """
            MATCH (s:Supplier)-[:provide_data]->(p:DataProduct)
            OPTIONAL MATCH (p)-[:applied_to]->(a:app_IndustryCategory)
            OPTIONAL MATCH (src:src_IndustryCategory)-[:source_industry]->(p)
            RETURN s.name AS supplier, p.name AS product,
                   collect(DISTINCT a.name) AS app_list,
                   collect(DISTINCT src.name) AS src_list
            """
            return pd.DataFrame(self.g.run(cypher).data())

    def fetch_supplier_app_counts(self) -> pd.DataFrame:
        """
        For market structure by APP: count products per (supplier, app).
        """
        if not self.g:
            return pd.DataFrame()

        cypher = """
        MATCH (s:Supplier)-[:provide_data]->(p:DataProduct)-[:applied_to]->(a:app_IndustryCategory)
        RETURN a.name AS app, s.name AS supplier, count(DISTINCT p) AS prod_cnt
        """
        return pd.DataFrame(self.g.run(cypher).data())

    def fetch_supplier_src_counts(self) -> pd.DataFrame:
        """
        For structure by SRC: count products per (supplier, src).
        """
        if not self.g:
            return pd.DataFrame()

        cypher = """
        MATCH (src:src_IndustryCategory)-[:source_industry]->(p:DataProduct)<-[:provide_data]-(s:Supplier)
        RETURN src.name AS src, s.name AS supplier, count(DISTINCT p) AS prod_cnt
        """
        return pd.DataFrame(self.g.run(cypher).data())

    def fetch_src_app_counts(self) -> pd.DataFrame:
        """
        For SRC-APP meta-path graph: count products per (src, app).
        """
        if not self.g:
            return pd.DataFrame()

        cypher = """
        MATCH (src:src_IndustryCategory)-[:source_industry]->(p:DataProduct)-[:applied_to]->(a:app_IndustryCategory)
        RETURN src.name AS src, a.name AS app, count(DISTINCT p) AS prod_cnt
        """
        return pd.DataFrame(self.g.run(cypher).data())

# =============================================================================
# Media heat feature engineering
# =============================================================================
class MediaHeat:
    def __init__(self, media_path: str):
        self.df = pd.DataFrame()

        try:
            df = pd.read_excel(media_path).copy()
            if "keyword" not in df.columns:
                self.df = pd.DataFrame()
                return

            for c in ["sogou_web_results", "sina_news_results", "weixin_article_results"]:
                if c not in df.columns:
                    df[c] = 0.0

                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

            # Raw counts
            df["heat_web_raw"] = df["sogou_web_results"]
            df["heat_news_raw"] = df["sina_news_results"]
            df["heat_weixin_raw"] = df["weixin_article_results"]

            # log10_1p counts (single log transform HERE, and only HERE)
            df["heat_web_log"] = log10_1p(df["heat_web_raw"])
            df["heat_news_log"] = log10_1p(df["heat_news_raw"])
            df["heat_weixin_log"] = log10_1p(df["heat_weixin_raw"])

            # Total demand heat on log scale
            df["heat_total_log"] = df["heat_web_log"] + df["heat_news_log"] + df["heat_weixin_log"]

            self.df = df
        except Exception:
            self.df = pd.DataFrame()

    def build_maps(self) -> Dict[str, Dict[str, float]]:
        """
        Return keyword -> heat maps (all on log10+1 scale):
          - total_log
          - web_log
          - news_log
          - weixin_log
        """
        if self.df.empty or "keyword" not in self.df.columns:
            return {"total_log": {}, "web_log": {}, "news_log": {}, "weixin_log": {}}

        maps = {"total_log": {}, "web_log": {}, "news_log": {}, "weixin_log": {}}

        for _, r in self.df.iterrows():
            k = str(r["keyword"]).strip()
            if not k:
                continue

            maps["total_log"][k] = float(r.get("heat_total_log", 0.0))
            maps["web_log"][k] = float(r.get("heat_web_log", 0.0))
            maps["news_log"][k] = float(r.get("heat_news_log", 0.0))
            maps["weixin_log"][k] = float(r.get("heat_weixin_log", 0.0))

        return maps

# =============================================================================
# KG Reasoning (Original): heterogeneous graph PPR + optional media-heat prior
# =============================================================================
class KGReasoner:
    def __init__(self, df_edges: pd.DataFrame, app_heat_total: Dict[str, float], topk_apps: int, topk_srcs: int, use_supplier_in_product_node: bool = False):
        self.app_heat_total = app_heat_total or {}
        self.topk_apps = int(topk_apps)
        self.topk_srcs = int(topk_srcs)
        self.use_supplier_in_product_node = bool(use_supplier_in_product_node)
        self.G = nx.DiGraph()
        self._cache_pr: Dict[Tuple[str, float], Dict[str, float]] = {}
        self._build_graph(df_edges)
        self._prepare_heat_z()
        self._apply_hub_patch_degree_debias()
    def _prod_node(self, product: str, supplier: Optional[str] = None) -> str:
        p = str(product).strip()
        s = str(supplier).strip() if supplier is not None else ""
        if self.use_supplier_in_product_node and s:
            return f"PROD::{p}||SUP::{s}"
        return f"PROD::{p}"


    def _build_graph(self, df_edges: pd.DataFrame) -> None:
        if df_edges is None or df_edges.empty:
            return

        for _, r in df_edges.iterrows():
            s = str(r.get("supplier", "")).strip()
            p = str(r.get("product", "")).strip()
            if not s or not p:
                continue

            sup = f"SUP::{s}"
            prod = self._prod_node(p, s)

            # supplier <-> product
            self.G.add_edge(sup, prod, weight=1.0, etype="provide_data")
            self.G.add_edge(prod, sup, weight=1.0, etype="provide_data")

            # product <-> app
            for a in (r.get("app_list") or []):
                if a is None:
                    continue
                a = str(a).strip()
                if not a:
                    continue
                app = f"APP::{a}"
                self.G.add_edge(prod, app, weight=1.0, etype="applied_to")
                self.G.add_edge(app, prod, weight=1.0, etype="applied_to")

            # src <-> product
            for src in (r.get("src_list") or []):
                if src is None:
                    continue
                src = str(src).strip()
                if not src:
                    continue
                srcn = f"SRC::{src}"
                self.G.add_edge(srcn, prod, weight=1.0, etype="source_industry")
                self.G.add_edge(prod, srcn, weight=1.0, etype="source_industry")

    def _prepare_heat_z(self) -> None:
        apps = [n.replace("APP::", "") for n in self.G.nodes() if n.startswith("APP::")]
        heats = np.array([self.app_heat_total.get(a, 0.0) for a in apps], dtype=float)

        if len(heats) == 0 or np.std(heats) < 1e-12:
            self.heat_z = {a: 0.0 for a in apps}
            return

        z = (heats - heats.mean()) / (heats.std() + 1e-12)
        if HUB_PATCH_ENABLE and HEAT_Z_CLIP is not None:
            z = np.clip(z, -float(HEAT_Z_CLIP), float(HEAT_Z_CLIP))
        self.heat_z = {apps[i]: float(z[i]) for i in range(len(apps))}

    def _apply_hub_patch_degree_debias(self) -> None:
        """
        Minimal-invasive hub mitigation:
        reweight each edge by dividing destination degree^beta.
        """
        if not HUB_PATCH_ENABLE:
            return
        beta = float(HUB_DEBIAS_BETA)
        if beta <= 0.0:
            return

        deg = dict(self.G.degree())
        for u, v, dd in self.G.edges(data=True):
            w = float(dd.get("weight", 1.0))
            dv = float(max(deg.get(v, 1), 1))
            dd["weight"] = w / (dv ** beta)

    def _ppr(self, start_node: str, alpha: float) -> Dict[str, float]:
        if start_node not in self.G:
            return {}

        key = (start_node, float(alpha))
        if key in self._cache_pr:
            return self._cache_pr[key]

        personalization = {n: 0.0 for n in self.G.nodes()}
        personalization[start_node] = 1.0

        pr = nx.pagerank(
            self.G,
            alpha=alpha,
            personalization=personalization,
            weight="weight",
            max_iter=MAX_NX_PAGERANK_ITERS,
            tol=NX_PAGERANK_TOL,
        )

        out = {str(k): float(v) for k, v in pr.items()}
        self._cache_pr[key] = out

        return out


    # -------- heat-usage ablation helpers (STRICT split) --------
    def _extract_app_posterior_from_pr(self, pr: Dict[str, float], topk: Optional[int] = None) -> Dict[str, float]:
        """Extract and renormalize APP::* scores from a full PR map."""
        raw = {n.replace("APP::", ""): float(sc) for n, sc in pr.items() if str(n).startswith("APP::")}
        if not raw:
            return {}
        s = float(sum(raw.values()))
        if s <= 1e-12:
            return {}
        raw = {k: float(v / s) for k, v in raw.items()}
        k = int(topk) if topk is not None else int(self.topk_apps)
        return dict(sorted(raw.items(), key=lambda kv: kv[1], reverse=True)[:k])

    def _heat_prior_over_app_nodes(self, prior_strength: float) -> Dict[str, float]:
        """Build a heat-shaped prior distribution over APP nodes (keyed by APP name, not node id)."""
        apps = [n.replace("APP::", "") for n in self.G.nodes() if str(n).startswith("APP::")]
        if not apps:
            return {}
        w = np.array([float(np.exp(float(prior_strength) * float(self.heat_z.get(a, 0.0)))) for a in apps], dtype=float)
        s = float(w.sum())
        if s <= 1e-12:
            return {}
        w = w / s
        return {apps[i]: float(w[i]) for i in range(len(apps))}

    def _ppr_personalized(self, start_node: str, alpha: float, personalization: Dict[str, float], cache_tag: str) -> Dict[str, float]:
        """PPR with a custom personalization vector; cached separately from self._ppr()."""
        if start_node not in self.G:
            return {}

        key = (cache_tag, start_node, float(alpha))
        if key in self._cache_pr:
            return self._cache_pr[key]

        pers = {n: 0.0 for n in self.G.nodes()}
        for k, v in personalization.items():
            if k in pers:
                pers[k] = float(v)

        pr = nx.pagerank(
            self.G,
            alpha=alpha,
            personalization=pers,
            weight="weight",
            max_iter=MAX_NX_PAGERANK_ITERS,
            tol=NX_PAGERANK_TOL,
        )

        out = {str(k): float(v) for k, v in pr.items()}
        self._cache_pr[key] = out
        return out

    def infer_supplier_apps_structural(self, supplier: str, alpha: float) -> Dict[str, float]:
        """Structural reasoning: supplier -> APP via PPR using ONLY KG structure."""
        node = f"SUP::{supplier}"
        pr = self._ppr(node, alpha=alpha)
        return self._extract_app_posterior_from_pr(pr, topk=self.topk_apps)

    def infer_supplier_apps_heat_teleport(self, supplier: str, alpha: float, prior_strength: float, teleport_mass: float) -> Dict[str, float]:
        """Heat-biased reasoning: heat ONLY enters PPR personalization (teleport)."""
        node = f"SUP::{supplier}"
        teleport_mass = float(np.clip(float(teleport_mass), 0.0, 1.0))
        heat_prior = self._heat_prior_over_app_nodes(prior_strength=float(prior_strength))
        pers = {node: 1.0 - teleport_mass}
        for a, p in heat_prior.items():
            pers[f"APP::{a}"] = pers.get(f"APP::{a}", 0.0) + teleport_mass * float(p)
        pr = self._ppr_personalized(
            node,
            alpha=alpha,
            personalization=pers,
            cache_tag=f"teleheat_sup_apps__m{teleport_mass:.4f}__ps{float(prior_strength):.4f}",
        )
        return self._extract_app_posterior_from_pr(pr, topk=self.topk_apps)

    def infer_product_apps_structural(self, product: str, supplier: Optional[str], alpha: float) -> Dict[str, float]:
        """Structural reasoning: product -> APP via PPR using ONLY KG structure."""
        node = self._prod_node(product, supplier=supplier)
        pr = self._ppr(node, alpha=alpha)
        return self._extract_app_posterior_from_pr(pr, topk=self.topk_apps)

    def infer_product_apps_heat_teleport(self, product: str, supplier: Optional[str], alpha: float, prior_strength: float, teleport_mass: float) -> Dict[str, float]:
        """Heat-biased reasoning: heat ONLY enters PPR personalization (teleport)."""
        node = self._prod_node(product, supplier=supplier)
        teleport_mass = float(np.clip(float(teleport_mass), 0.0, 1.0))
        heat_prior = self._heat_prior_over_app_nodes(prior_strength=float(prior_strength))
        pers = {node: 1.0 - teleport_mass}
        for a, p in heat_prior.items():
            pers[f"APP::{a}"] = pers.get(f"APP::{a}", 0.0) + teleport_mass * float(p)
        pr = self._ppr_personalized(
            node,
            alpha=alpha,
            personalization=pers,
            cache_tag=f"teleheat_prod_apps__m{teleport_mass:.4f}__ps{float(prior_strength):.4f}",
        )
        return self._extract_app_posterior_from_pr(pr, topk=self.topk_apps)

    def infer_supplier_apps(self, supplier: str, alpha: float, prior_strength: float) -> Dict[str, float]:
        """
        Posterior over APP nodes for a supplier:
          posterior(app) ∝ PPR(app | supplier) * exp(prior_strength * heat_z(app))
        """
        node = f"SUP::{supplier}"
        pr = self._ppr(node, alpha=alpha)
        raw = {n.replace("APP::", ""): float(sc) for n, sc in pr.items() if n.startswith("APP::")}
        if not raw:
            return {}

        post = {}
        for a, sc in raw.items():
            post[a] = sc * float(np.exp(prior_strength * self.heat_z.get(a, 0.0)))

        s = sum(post.values())
        if s <= 0:
            return {}

        post = {a: v / s for a, v in post.items()}
        return dict(sorted(post.items(), key=lambda x: x[1], reverse=True)[: self.topk_apps])

    def infer_supplier_srcs(self, supplier: str, alpha: float) -> Dict[str, float]:
        node = f"SUP::{supplier}"
        pr = self._ppr(node, alpha=alpha)
        raw = {n.replace("SRC::", ""): float(sc) for n, sc in pr.items() if n.startswith("SRC::")}
        if not raw:
            return {}

        s = sum(raw.values())
        raw = {k: v / s for k, v in raw.items()}
        return dict(sorted(raw.items(), key=lambda x: x[1], reverse=True)[: self.topk_srcs])

    def infer_product_apps(self, product: str, supplier: Optional[str], alpha: float, prior_strength: float) -> Dict[str, float]:
        """
        Posterior over APP nodes for a product:
          posterior(app) ∝ PPR(app | product) * exp(prior_strength * heat_z(app))
        Note: If use_supplier_in_product_node=True, you must pass supplier to disambiguate products.
        """
        node = self._prod_node(product, supplier)
        pr = self._ppr(node, alpha=alpha)
        raw = {n.replace("APP::", ""): float(sc) for n, sc in pr.items() if n.startswith("APP::")}
        if not raw:
            return {}

        post = {}
        for a, sc in raw.items():
            post[a] = sc * float(np.exp(prior_strength * self.heat_z.get(a, 0.0)))

        s = sum(post.values())
        if s <= 0:
            return {}

        post = {a: v / s for a, v in post.items()}
        return dict(sorted(post.items(), key=lambda x: x[1], reverse=True)[: self.topk_apps])

    def infer_product_srcs(self, product: str, supplier: Optional[str], alpha: float) -> Dict[str, float]:
        node = self._prod_node(product, supplier)
        pr = self._ppr(node, alpha=alpha)
        raw = {n.replace("SRC::", ""): float(sc) for n, sc in pr.items() if n.startswith("SRC::")}
        if not raw:
            return {}

        s = sum(raw.values())
        raw = {k: v / s for k, v in raw.items()}
        return dict(sorted(raw.items(), key=lambda x: x[1], reverse=True)[: self.topk_srcs])

    @staticmethod
    def entropy(probs: List[float], eps: float = 1e-12) -> float:
        p = np.array(probs, dtype=float)
        p = p / (p.sum() + eps)
        return float(-np.sum(p * np.log(p + eps)))

    def expected_heat(self, post_apps: Dict[str, float]) -> float:
        return float(sum(prob * self.app_heat_total.get(app, 0.0) for app, prob in post_apps.items()))

# =============================================================================
# Meta-path / constrained PPR reasoning (NEW, requested)
# =============================================================================
class MetaPathPPRReasoner:
    """
    Meta-path constrained PPR on bipartite (or projected) graphs:
      - SUP<->APP with weight=prod_cnt
      - SUP<->SRC with weight=prod_cnt
      - SRC<->APP with weight=prod_cnt

    In bipartite graphs, a random walk naturally alternates node types, which is a simple
    implementation of path-constrained random walks for a given meta-path.
    """
    def __init__(
        self,
        df_sup_app: pd.DataFrame,
        df_sup_src: pd.DataFrame,
        df_src_app: pd.DataFrame,
        app_heat_total: Dict[str, float],
    ):
        self.app_heat_total = app_heat_total or {}
        self.G_sa = self._build_bipartite(df_sup_app, left_prefix="SUP::", right_prefix="APP::", left_col="supplier", right_col="app")
        self.G_ss = self._build_bipartite(df_sup_src, left_prefix="SUP::", right_prefix="SRC::", left_col="supplier", right_col="src")
        self.G_ca = self._build_bipartite(df_src_app, left_prefix="SRC::", right_prefix="APP::", left_col="src", right_col="app")
        self._cache: Dict[Tuple[str, str, float], Dict[str, float]] = {}
        self._prepare_app_heat_z()

    @staticmethod
    def _build_bipartite(
            df: pd.DataFrame,
            left_prefix: str,
            right_prefix: str,
            left_col: str,
            right_col: str,
    ) -> nx.Graph:
        """
        Optional hub mitigation for bipartite graphs:
        - log1p edge weights
        - IDF reweight on the right nodes (apps/src) to downweight hubs
        """
        G = nx.Graph()
        if df is None or df.empty:
            return G

        d = df.copy()
        d["prod_cnt"] = pd.to_numeric(d.get("prod_cnt", 0), errors="coerce").fillna(0.0).astype(float)

        # Build IDF on right side if enabled
        idf = None
        if HUB_PATCH_ENABLE and BIPARTITE_EDGE_IDF:
            right_tot = d.groupby(right_col)["prod_cnt"].sum()
            n_left = max(1, d[left_col].nunique())
            # Smooth IDF
            idf = {}
            for rname, tot in right_tot.items():
                # Using count-like proxy: higher total => lower idf
                val = math.log(
                    (n_left + float(BIPARTITE_EDGE_IDF_SMOOTH)) / (float(tot) + float(BIPARTITE_EDGE_IDF_SMOOTH)))
                idf[str(rname).strip()] = float(max(0.0, val))

        for _, r in d.iterrows():
            l_raw = str(r[left_col]).strip()
            r_raw = str(r[right_col]).strip()
            if not l_raw or not r_raw:
                continue

            l = f"{left_prefix}{l_raw}"
            rr = f"{right_prefix}{r_raw}"

            w = float(r["prod_cnt"])
            if w <= 0:
                continue

            if HUB_PATCH_ENABLE and BIPARTITE_EDGE_LOG1P:
                w = float(log10_1p(w))

            if idf is not None:
                w = w * float(idf.get(r_raw, 0.0) + 1.0)

            if w <= 0:
                continue

            if G.has_edge(l, rr):
                G[l][rr]["weight"] += w
            else:
                G.add_edge(l, rr, weight=w)

        return G

    def _prepare_app_heat_z(self) -> None:
        apps = set()

        for G in [self.G_sa, self.G_ca]:
            for n in G.nodes():
                if str(n).startswith("APP::"):
                    apps.add(str(n).replace("APP::", ""))

        apps = sorted(apps)
        heats = np.array([self.app_heat_total.get(a, 0.0) for a in apps], dtype=float)

        if len(heats) == 0 or np.std(heats) < 1e-12:
            self.app_heat_z = {a: 0.0 for a in apps}
        else:
            z = (heats - heats.mean()) / (heats.std() + 1e-12)
            if HUB_PATCH_ENABLE and HEAT_Z_CLIP is not None:
                z = np.clip(z, -float(HEAT_Z_CLIP), float(HEAT_Z_CLIP))
            self.app_heat_z = {apps[i]: float(z[i]) for i in range(len(apps))}

    def _ppr(self, G: nx.Graph, start_node: str, alpha: float) -> Dict[str, float]:
        if start_node not in G:
            return {}

        key = (id(G), start_node, float(alpha))
        if key in self._cache:
            return self._cache[key]

        personalization = {n: 0.0 for n in G.nodes()}
        personalization[start_node] = 1.0

        pr = nx.pagerank(
            G,
            alpha=alpha,
            personalization=personalization,
            weight="weight",
            max_iter=MAX_NX_PAGERANK_ITERS,
            tol=NX_PAGERANK_TOL,
        )

        out = {str(k): float(v) for k, v in pr.items()}
        self._cache[key] = out

        return out


    # -------- heat-usage ablation helpers (STRICT split) --------
    def _extract_app_posterior_from_pr(self, pr: Dict[str, float], topk: Optional[int] = None) -> Dict[str, float]:
        """Extract and renormalize APP::* scores from a full PR map."""
        raw = {n.replace("APP::", ""): float(sc) for n, sc in pr.items() if str(n).startswith("APP::")}
        if not raw:
            return {}
        s = float(sum(raw.values()))
        if s <= 1e-12:
            return {}
        raw = {k: float(v / s) for k, v in raw.items()}
        k = int(topk) if topk is not None else int(self.topk_apps)
        return dict(sorted(raw.items(), key=lambda kv: kv[1], reverse=True)[:k])

    def _heat_prior_over_apps(self, G: nx.Graph, prior_strength: float) -> Dict[str, float]:
        """Heat-shaped prior over APP nodes in a given bipartite graph (keyed by APP name)."""
        apps = [n.replace("APP::", "") for n in G.nodes() if str(n).startswith("APP::")]
        if not apps:
            return {}
        w = np.array([float(np.exp(float(prior_strength) * float(self.app_heat_z.get(a, 0.0)))) for a in apps], dtype=float)
        s = float(w.sum())
        if s <= 1e-12:
            return {}
        w = w / s
        return {apps[i]: float(w[i]) for i in range(len(apps))}

    def _ppr_personalized(self, G: nx.Graph, start_node: str, alpha: float, personalization: Dict[str, float], cache_tag: str) -> Dict[str, float]:
        """PPR with a custom personalization vector; cached separately from self._ppr()."""
        if start_node not in G:
            return {}
        key = (cache_tag, id(G), start_node, float(alpha))
        if key in self._cache:
            return self._cache[key]

        pers = {n: 0.0 for n in G.nodes()}
        for k, v in personalization.items():
            if k in pers:
                pers[k] = float(v)

        pr = nx.pagerank(
            G,
            alpha=alpha,
            personalization=pers,
            weight="weight",
            max_iter=MAX_NX_PAGERANK_ITERS,
            tol=NX_PAGERANK_TOL,
        )

        out = {str(k): float(v) for k, v in pr.items()}
        self._cache[key] = out
        return out

    def sup_to_apps_structural(self, supplier: str, alpha: float, topk: int) -> Dict[str, float]:
        """Structural reasoning: SUP -> APP via PPR using ONLY KG structure."""
        node = f"SUP::{supplier}"
        pr = self._ppr(self.G_sa, node, alpha)
        return self._extract_app_posterior_from_pr(pr, topk=topk)

    def src_to_apps_structural(self, src: str, alpha: float, topk: int) -> Dict[str, float]:
        """Structural reasoning: SRC -> APP via PPR using ONLY KG structure."""
        node = f"SRC::{src}"
        pr = self._ppr(self.G_ca, node, alpha)
        return self._extract_app_posterior_from_pr(pr, topk=topk)

    def sup_to_apps_heat_teleport(self, supplier: str, alpha: float, topk: int, prior_strength: float, teleport_mass: float) -> Dict[str, float]:
        """Heat-biased reasoning: heat ONLY enters PPR personalization (teleport)."""
        node = f"SUP::{supplier}"
        teleport_mass = float(np.clip(float(teleport_mass), 0.0, 1.0))
        heat_prior = self._heat_prior_over_apps(self.G_sa, prior_strength=float(prior_strength))
        pers = {node: 1.0 - teleport_mass}
        for a, p in heat_prior.items():
            pers[f"APP::{a}"] = pers.get(f"APP::{a}", 0.0) + teleport_mass * float(p)
        pr = self._ppr_personalized(
            self.G_sa,
            node,
            alpha=alpha,
            personalization=pers,
            cache_tag=f"teleheat_sup_apps__m{teleport_mass:.4f}__ps{float(prior_strength):.4f}",
        )
        return self._extract_app_posterior_from_pr(pr, topk=topk)

    def src_to_apps_heat_teleport(self, src: str, alpha: float, topk: int, prior_strength: float, teleport_mass: float) -> Dict[str, float]:
        """Heat-biased reasoning: heat ONLY enters PPR personalization (teleport)."""
        node = f"SRC::{src}"
        teleport_mass = float(np.clip(float(teleport_mass), 0.0, 1.0))
        heat_prior = self._heat_prior_over_apps(self.G_ca, prior_strength=float(prior_strength))
        pers = {node: 1.0 - teleport_mass}
        for a, p in heat_prior.items():
            pers[f"APP::{a}"] = pers.get(f"APP::{a}", 0.0) + teleport_mass * float(p)
        pr = self._ppr_personalized(
            self.G_ca,
            node,
            alpha=alpha,
            personalization=pers,
            cache_tag=f"teleheat_src_apps__m{teleport_mass:.4f}__ps{float(prior_strength):.4f}",
        )
        return self._extract_app_posterior_from_pr(pr, topk=topk)

    def _posterior_with_appheat_prior(self, raw_app: Dict[str, float], prior_strength: float) -> Dict[str, float]:
        if not raw_app:
            return {}

        post = {}
        for a, sc in raw_app.items():
            post[a] = sc * float(np.exp(prior_strength * self.app_heat_z.get(a, 0.0)))

        s = sum(post.values())
        if s <= 0:
            return {}

        return {k: v / s for k, v in post.items()}

    # -------- supplier-centered --------
    def sup_to_apps(self, supplier: str, alpha: float, topk: int, prior_strength: float) -> Dict[str, float]:
        node = f"SUP::{supplier}"
        pr = self._ppr(self.G_sa, node, alpha)
        raw = {n.replace("APP::", ""): sc for n, sc in pr.items() if n.startswith("APP::")}
        post = self._posterior_with_appheat_prior(raw, prior_strength=prior_strength)

        return dict(sorted(post.items(), key=lambda kv: kv[1], reverse=True)[:topk])

    def sup_to_srcs(self, supplier: str, alpha: float, topk: int) -> Dict[str, float]:
        node = f"SUP::{supplier}"
        pr = self._ppr(self.G_ss, node, alpha)
        raw = {n.replace("SRC::", ""): sc for n, sc in pr.items() if n.startswith("SRC::")}
        s = sum(raw.values())

        if s <= 0:
            return {}

        raw = {k: v / s for k, v in raw.items()}
        return dict(sorted(raw.items(), key=lambda kv: kv[1], reverse=True)[:topk])

    # -------- src-centered --------
    def src_to_apps(self, src: str, alpha: float, topk: int, prior_strength: float) -> Dict[str, float]:
        node = f"SRC::{src}"
        pr = self._ppr(self.G_ca, node, alpha)
        raw = {n.replace("APP::", ""): sc for n, sc in pr.items() if n.startswith("APP::")}
        post = self._posterior_with_appheat_prior(raw, prior_strength=prior_strength)
        return dict(sorted(post.items(), key=lambda kv: kv[1], reverse=True)[:topk])

    def src_to_sups(self, src: str, alpha: float, topk: int) -> Dict[str, float]:
        node = f"SRC::{src}"
        pr = self._ppr(self.G_ss, node, alpha)  # same bipartite, start from SRC
        raw = {n.replace("SUP::", ""): sc for n, sc in pr.items() if n.startswith("SUP::")}
        s = sum(raw.values())

        if s <= 0:
            return {}

        raw = {k: v / s for k, v in raw.items()}
        return dict(sorted(raw.items(), key=lambda kv: kv[1], reverse=True)[:topk])

    # -------- app-centered --------
    def app_to_sups(self, app: str, alpha: float, topk: int) -> Dict[str, float]:
        node = f"APP::{app}"
        pr = self._ppr(self.G_sa, node, alpha)
        raw = {n.replace("SUP::", ""): sc for n, sc in pr.items() if n.startswith("SUP::")}
        s = sum(raw.values())

        if s <= 0:
            return {}

        raw = {k: v / s for k, v in raw.items()}
        return dict(sorted(raw.items(), key=lambda kv: kv[1], reverse=True)[:topk])

    def app_to_srcs(self, app: str, alpha: float, topk: int) -> Dict[str, float]:
        node = f"APP::{app}"
        pr = self._ppr(self.G_ca, node, alpha)  # start from APP in SRC-APP
        raw = {n.replace("SRC::", ""): sc for n, sc in pr.items() if n.startswith("SRC::")}
        s = sum(raw.values())

        if s <= 0:
            return {}

        raw = {k: v / s for k, v in raw.items()}
        return dict(sorted(raw.items(), key=lambda kv: kv[1], reverse=True)[:topk])

    @staticmethod
    def entropy(probs: List[float], eps: float = 1e-12) -> float:
        p = np.array(probs, dtype=float)
        p = p / (p.sum() + eps)
        return float(-np.sum(p * np.log(p + eps)))

    def expected_app_heat(self, post_apps: Dict[str, float]) -> float:
        return float(sum(prob * self.app_heat_total.get(app, 0.0) for app, prob in post_apps.items()))

# =============================================================================
# Posterior embedding (cheap deterministic)
# =============================================================================
class PosteriorEmbedder:
    def __init__(self, dim: int):
        self.dim = int(dim)

    @staticmethod
    def _det_vec(label: str, dim: int) -> np.ndarray:
        h = hashlib.md5(label.encode("utf-8")).hexdigest()
        seed = int(h[:8], 16) % (2**31 - 1)
        rng = np.random.RandomState(seed)
        v = rng.normal(0.0, 1.0, size=dim)

        return v / (np.linalg.norm(v) + 1e-12)

    def embed(self, posterior: Dict[str, float]) -> np.ndarray:
        if not posterior:
            return np.zeros(self.dim, dtype=float)

        vec = np.zeros(self.dim, dtype=float)

        for lab, w in posterior.items():
            vec += float(w) * self._det_vec(str(lab), self.dim)

        return vec

# =============================================================================
# Data loading and merging (STEP0)
# =============================================================================
def load_price_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    for c in ["name", "price", "supplier"]:
        if c not in df.columns:
            raise ValueError("Excel must have columns: name, price, supplier")

    df["name"] = df["name"].astype(str).str.strip()
    df["supplier"] = df["supplier"].astype(str).str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["name", "supplier", "price"])
    df = df[df["price"] > 0].copy()
    df["y_log10"] = df["price"].apply(safe_log10)
    df = df.dropna(subset=["y_log10"])
    return df

def add_row_uid(df: pd.DataFrame) -> pd.DataFrame:
    # uid 只依赖稳定字段，别用 index
    def _uid(r):
        s = f"{str(r['supplier']).strip()}||{str(r['name']).strip()}||{float(r['price']):.8f}"
        return hashlib.md5(s.encode("utf-8")).hexdigest()
    out = df.copy()
    out["row_uid"] = out.apply(_uid, axis=1)
    return out

def build_matched_dataset(db: Neo4jDB, excel_path: str, out_dir: str) -> pd.DataFrame:
    df_price = load_price_excel(excel_path)
    df_kg = db.fetch_matched_multilabel(df_price)

    if df_kg.empty:
        raise RuntimeError("No matched (supplier, product) rows found in KG for your price table.")

    merged = pd.merge(
        df_price,
        df_kg,
        on=["name", "supplier"],
        how="inner",
        validate="many_to_one"
    )

    merged["src_list"] = merged["src_list"].apply(parse_list_cell)
    merged["app_list"] = merged["app_list"].apply(parse_list_cell)
    merged = add_row_uid(merged)

    merged.to_csv(os.path.join(out_dir, "STEP0_matched_dataset.csv"), index=False, encoding="utf-8-sig")
    return merged

# =============================================================================
# Optional: Hierarchical Bayes (crossed random effects) - auto-skip if no PyMC
# =============================================================================
def fit_hierarchical_bayes_crossed(df: pd.DataFrame, min_label_freq: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Crossed random-effects style model:
      y ~ intercept + u_supplier[supplier] + X_src*b_src + X_app*b_app + eps
    with hierarchical priors.
    """
    try:
        import pymc as pm
        import arviz as az
        try:
            import pytensor.tensor as pt
        except Exception:
            import aesara.tensor as pt
    except Exception as e:
        sig = pd.DataFrame([{"param": "SKIPPED", "mean": np.nan, "hdi_2.5%": np.nan, "hdi_97.5%": np.nan, "note": f"PyMC not available: {e}"}])
        share = pd.DataFrame([{"component": "SKIPPED", "mean": np.nan, "hdi_2.5%": np.nan, "hdi_97.5%": np.nan}])
        meta = {"ok": False, "reason": str(e)}
        return sig, share, meta

    d = df.copy()
    y = d["y_log10"].values.astype(np.float64)

    suppliers = sorted(d["supplier"].astype(str).unique().tolist())
    s2i = {s: i for i, s in enumerate(suppliers)}
    supplier_idx = d["supplier"].astype(str).map(s2i).values.astype(int)
    n_sup = len(suppliers)

    X_src, _ = build_fractional_multihot_df(d["src_list"], prefix="SRC", min_freq=min_label_freq)
    X_app, _ = build_fractional_multihot_df(d["app_list"], prefix="APP", min_freq=min_label_freq)
    X_src_np = X_src.values.astype(np.float64)
    X_app_np = X_app.values.astype(np.float64)

    n_src = X_src_np.shape[1]
    n_app = X_app_np.shape[1]

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0.0, sigma=5.0)
        sigma_y = pm.HalfNormal("sigma_y", sigma=1.0)

        sigma_supplier = pm.HalfNormal("sigma_supplier", sigma=1.0)
        sigma_src = pm.HalfNormal("sigma_src", sigma=1.0)
        sigma_app = pm.HalfNormal("sigma_app", sigma=1.0)

        u_sup_raw = pm.Normal("u_sup_raw", mu=0.0, sigma=sigma_supplier, shape=n_sup)
        u_sup = pm.Deterministic("u_sup", u_sup_raw - pt.mean(u_sup_raw))

        mu = intercept + u_sup[supplier_idx]

        if n_src > 0:
            b_src_raw = pm.Normal("b_src_raw", mu=0.0, sigma=sigma_src, shape=n_src)
            b_src = b_src_raw - pt.mean(b_src_raw)
            mu = mu + pt.dot(X_src_np, b_src)

        if n_app > 0:
            b_app_raw = pm.Normal("b_app_raw", mu=0.0, sigma=sigma_app, shape=n_app)
            b_app = b_app_raw - pt.mean(b_app_raw)
            mu = mu + pt.dot(X_app_np, b_app)

        pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y)

        idata = pm.sample(
            draws=PYMC_DRAWS,
            tune=PYMC_TUNE,
            chains=PYMC_CHAINS,
            target_accept=PYMC_TARGET_ACCEPT,
            progressbar=True
        )

    sig_sum = az.summary(
        idata,
        var_names=["sigma_supplier", "sigma_src", "sigma_app", "sigma_y"],
        hdi_prob=0.95
    ).reset_index().rename(columns={"index": "param"})

    post = idata.posterior
    u = post["u_sup_raw"].stack(sample=("chain", "draw")).values
    u = u - u.mean(axis=0, keepdims=True)
    comp_sup = u[supplier_idx, :]

    comp_src = np.zeros_like(comp_sup)
    comp_app = np.zeros_like(comp_sup)

    if n_src > 0:
        b = post["b_src_raw"].stack(sample=("chain", "draw")).values
        b = b - b.mean(axis=0, keepdims=True)
        comp_src = X_src_np @ b

    if n_app > 0:
        b = post["b_app_raw"].stack(sample=("chain", "draw")).values
        b = b - b.mean(axis=0, keepdims=True)
        comp_app = X_app_np @ b

    var_sup = np.var(comp_sup, axis=0)
    var_src = np.var(comp_src, axis=0)
    var_app = np.var(comp_app, axis=0)
    sigma_y = post["sigma_y"].stack(sample=("chain", "draw")).values
    var_eps = sigma_y ** 2

    denom = var_sup + var_src + var_app + var_eps + 1e-12
    share_sup = var_sup / denom
    share_src = var_src / denom
    share_app = var_app / denom
    share_eps = var_eps / denom

    def summarise(x: np.ndarray) -> Dict[str, float]:
        x = np.asarray(x)
        return {
            "mean": float(np.mean(x)),
            "hdi_2.5%": float(np.quantile(x, 0.025)),
            "hdi_97.5%": float(np.quantile(x, 0.975))
        }

    var_share = pd.DataFrame([
        {"component": "supplier", **summarise(share_sup)},
        {"component": "src", **summarise(share_src)},
        {"component": "app", **summarise(share_app)},
        {"component": "eps", **summarise(share_eps)},
    ])

    meta = {"ok": True, "n_suppliers": n_sup, "n_src_vocab": n_src, "n_app_vocab": n_app}
    return sig_sum, var_share, meta

def product_heat_from_app_list(app_list_any: Any, hmap: Dict[str, float]) -> float:
    apps = parse_list_cell(app_list_any)
    if not apps:
        return 0.0
    vals = [float(hmap.get(a, 0.0)) for a in apps]
    return float(np.mean(vals)) if vals else 0.0

def build_demand_structure_features(
    df_rows: pd.DataFrame,
    df_sup_app: pd.DataFrame,
    df_sup_src: pd.DataFrame,
    df_src_app: pd.DataFrame,
    heat_maps: Dict[str, Dict[str, float]],
    out_dir: str,
) -> pd.DataFrame:
    """
    Build DEMAND + STRUCTURE numeric features (global, no y usage).
    Returns df_enriched with new numeric columns.
    """
    d = df_rows.copy()

    # ---- Demand: product heat from app_list ----
    d["paper_heat_total"] = d["app_list"].apply(lambda x: product_heat_from_app_list(x, heat_maps["total_log"])).astype(float)
    d["paper_heat_web"] = d["app_list"].apply(lambda x: product_heat_from_app_list(x, heat_maps["web_log"])).astype(float)
    d["paper_heat_news"] = d["app_list"].apply(lambda x: product_heat_from_app_list(x, heat_maps["news_log"])).astype(float)
    d["paper_heat_weixin"] = d["app_list"].apply(lambda x: product_heat_from_app_list(x, heat_maps["weixin_log"])).astype(float)

    # ---- Structure: app-level stats + thickness ----
    df_app_stats = compute_category_market_structure(df_sup_app.rename(columns={"app": "app"}), cat_col="app", supplier_col="supplier")
    if not df_app_stats.empty:
        df_app_stats = df_app_stats.rename(columns={"app_total_products": "app_total_products"})
    df_thick = compute_market_thickness_for_app(df_app_stats, app_heat_total=heat_maps["total_log"]) if not df_app_stats.empty else pd.DataFrame(columns=["app", "heat", "thickness"])
    df_app_table = df_app_stats.merge(df_thick, on="app", how="left").fillna(0.0)

    df_app_table.to_csv(os.path.join(out_dir, "STEP0_app_structure_table.csv"), index=False, encoding="utf-8-sig")

    # ---- Structure: src-level stats (based on supplier-src counts) ----
    df_src_stats = compute_category_market_structure(df_sup_src.rename(columns={"src": "src"}), cat_col="src", supplier_col="supplier")
    if not df_src_stats.empty:
        df_src_stats = df_src_stats.rename(columns={"src_total_products": "src_total_products"})
    df_src_table = df_src_stats.copy()
    df_src_table.to_csv(os.path.join(out_dir, "STEP0_src_structure_table.csv"), index=False, encoding="utf-8-sig")

    # ---- Supplier-level structure: monopoly + network + expected_heat ----
    df_sup_mono = compute_supplier_monopoly_metrics(df_sup_app)
    df_sup_net = compute_supplier_network_metrics_sa(df_sup_app)
    df_sup_eheat = supplier_expected_heat(df_sup_app, app_heat_total=heat_maps["total_log"])

    df_supplier_table = df_sup_mono.merge(df_sup_net, on="supplier", how="outer").merge(df_sup_eheat, on="supplier", how="outer").fillna(0.0)
    df_supplier_table.to_csv(os.path.join(out_dir, "STEP0_supplier_structure_table.csv"), index=False, encoding="utf-8-sig")

    # merge supplier-level into rows
    d = d.merge(df_supplier_table, on="supplier", how="left").fillna(0.0)

    # row-level aggregation from app/src tables
    app_cols = [
        c for c in [
            "n_suppliers", "HHI", "CR4", "CR8",
            "app_total_products", "thickness", "heat",
            "tension_B_nsup", "tension_B_nprod",
        ]
        if c in df_app_table.columns
    ]

    src_cols = [c for c in ["n_suppliers", "HHI", "CR4", "CR8", "src_total_products"] if c in df_src_table.columns]

    # app mean features
    for c in app_cols:
        d[f"row_app_mean_{c}"] = d["app_list"].apply(lambda x: row_mean_from_label_table(x, df_app_table, "app", [c])[c]).astype(float)

    # src mean features
    for c in src_cols:
        d[f"row_src_mean_{c}"] = d["src_list"].apply(lambda x: row_mean_from_label_table(x, df_src_table, "src", [c])[c]).astype(float)

    # simple counts of labels
    d["n_app_labels"] = d["app_list"].apply(lambda x: len(parse_list_cell(x))).astype(float)
    d["n_src_labels"] = d["src_list"].apply(lambda x: len(parse_list_cell(x))).astype(float)


    # ---- Paper-aligned vars for 3.4.1 Basic pricing model ----
    # We implement the paper definitions with base-10 logs and a unified market scope:
    #   Bd = log10(1 + D)   (here proxied by paper_heat_total, assumed already on Bd scale)
    #   N_market = supplier count under ONE consistent market definition (used in Bs/HHI/T)
    #   Bs = log10(1 + N_market)
    #   HHI proxy = 1 / N_market   (equal-share conservative proxy)
    #   T = log10(1 + D * N_market)
    #
    # NOTE: To avoid inconsistent N across Bs/T/HHI, we compute a single N_market
    # (defaulting to supply-side src industries) and reuse it everywhere.
    d["paper_heat_total"] = pd.to_numeric(d["paper_heat_total"], errors="coerce").fillna(0.0).astype(float)

    # Unified market N (supply-side): mean aggregation over multi-src labels is already reflected in row_src_mean_n_suppliers.
    n_market = pd.to_numeric(d.get("row_src_mean_n_suppliers"), errors="coerce").fillna(0.0).astype(float)

    # Fallback: if src-side N is missing/zero but app-side exists, borrow app-side N as the single N_market for that row.
    if "row_app_mean_n_suppliers" in d.columns:
        n_app_fallback = pd.to_numeric(d["row_app_mean_n_suppliers"], errors="coerce").fillna(0.0).astype(float)
        n_market = np.where((n_market <= 0.0) & (n_app_fallback > 0.0), n_app_fallback, n_market)

    n_market = np.clip(n_market, 0.0, None)
    d["paper_N_market"] = n_market

    d["paper_Bs"] = log10_1p(n_market)
    d["paper_HHI_proxy"] = 1.0 / np.clip(n_market, 1.0, None)

    D_raw = inv_log10_1p(d["paper_heat_total"])
    d["paper_T"] = log10_1p(D_raw * n_market)

    d.to_csv(os.path.join(out_dir, "STEP0_dataset_with_demand_structure.csv"), index=False, encoding="utf-8-sig")
    return d

# =============================================================================
# STEP0: competitor graph outputs (full data, for UI inspection)
# =============================================================================
def step0_write_competitor_outputs_from_desc(
    df_rows: pd.DataFrame,
    heat_maps: Dict[str, Dict[str, float]],
    out_dir: str,
    k: int = 10,
    min_cos: float = 0.25,
    topk_labels: int = 10,
) -> pd.DataFrame:
    """
    Build desc-based product similarity edges (same schema as STEP2_product_similarity_graph_edges.csv),
    plus product-level competitor reasoning summaries and missing label completion.

    Returns a compact numeric table keyed by (name, supplier) for merging into STEP0/STEP3.
    """
    ensure_dir(out_dir)
    d = df_rows.copy()
    d["desc"] = d["desc"].fillna("").astype(str)

    # Build embedding on full dataset (for inspection output only)
    df_prod_emb, _ = build_product_text_embedding(d, text_col="desc", dim=TEXT_EMB_DIM, min_df=TEXT_MIN_DF)
    Gsim = build_product_similarity_graph(df_prod_emb, k=k, min_cos=min_cos)

    # Write edges in the SAME format as STEP2
    edges_out = [{"u": u, "v": v, "w": float(dd.get("weight", 0.0))} for u, v, dd in Gsim.edges(data=True)]
    pd.DataFrame(edges_out).to_csv(
        os.path.join(out_dir, "STEP0_product_similarity_graph_edges.csv"),
        index=False, encoding="utf-8-sig"
    )

    # Neighbor-based completion using KNN (full data)
    emb_cols = [c for c in df_prod_emb.columns if c.startswith("textemb_")]
    X = df_prod_emb[emb_cols].astype(float).values
    nbr_idx, nbr_sim = knn_neighbors_exclude_self(X, X, k=k)

    # Prepare label lists
    app_lists = [parse_list_cell(x) for x in d["app_list"].tolist()]
    src_lists = [parse_list_cell(x) for x in d["src_list"].tolist()]

    # Prepare competitor summary rows
    prod_keys = [f"PROD::{r['name']}||SUP::{r['supplier']}" for _, r in d.iterrows()]
    rows = []
    missing_app_rows = []
    missing_src_rows = []

    for i in range(len(d)):
        idx = nbr_idx[i]
        sim = nbr_sim[i]
        # build competitor posterior over apps/srcs from neighbors
        neigh_apps = [app_lists[j] for j in idx] if len(idx) else []
        neigh_srcs = [src_lists[j] for j in idx] if len(idx) else []

        post_app = neighbor_label_posterior(neigh_apps, sim)
        post_src = neighbor_label_posterior(neigh_srcs, sim)

        ent_app = posterior_entropy(post_app)
        ent_src = posterior_entropy(post_src)

        top1_app = max(post_app.values()) if post_app else 0.0
        top1_src = max(post_src.values()) if post_src else 0.0

        exp_heat_total = expected_from_heat_map(post_app, heat_maps.get("total_log", {}))

        # inferred missing = inferred_topk - observed
        obs_app = set(app_lists[i])
        obs_src = set(src_lists[i])

        inferred_app = [k for k, _ in sorted(post_app.items(), key=lambda kv: kv[1], reverse=True)[:topk_labels]]
        inferred_src = [k for k, _ in sorted(post_src.items(), key=lambda kv: kv[1], reverse=True)[:topk_labels]]

        miss_app = [x for x in inferred_app if x not in obs_app]
        miss_src = [x for x in inferred_src if x not in obs_src]

        # competitors topk
        comp = {}
        for j, w in zip(idx.tolist() if hasattr(idx, "tolist") else list(idx), sim.tolist() if hasattr(sim, "tolist") else list(sim)):
            if w <= 0:
                continue
            comp[prod_keys[int(j)]] = float(w)

        rows.append({
            "name": str(d.iloc[i]["name"]),
            "supplier": str(d.iloc[i]["supplier"]),
            "prod_key": prod_keys[i],
            "comp_app_entropy": float(ent_app),
            "comp_src_entropy": float(ent_src),
            "comp_app_top1_prob": float(top1_app),
            "comp_src_top1_prob": float(top1_src),
            "comp_expected_heat_total": float(exp_heat_total),
            "comp_top_products_json": topk_keys_json(comp, topk=k),
            "comp_inferred_apps_json": topk_keys_json(post_app, topk=topk_labels),
            "comp_inferred_srcs_json": topk_keys_json(post_src, topk=topk_labels),
        })

        missing_app_rows.append({
            "name": str(d.iloc[i]["name"]),
            "supplier": str(d.iloc[i]["supplier"]),
            "observed_count": int(len(obs_app)),
            "inferred_topk": json.dumps(inferred_app, ensure_ascii=False),
            "missing_topk": json.dumps(miss_app, ensure_ascii=False),
        })
        missing_src_rows.append({
            "name": str(d.iloc[i]["name"]),
            "supplier": str(d.iloc[i]["supplier"]),
            "observed_count": int(len(obs_src)),
            "inferred_topk": json.dumps(inferred_src, ensure_ascii=False),
            "missing_topk": json.dumps(miss_src, ensure_ascii=False),
        })

    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(os.path.join(out_dir, "STEP0_product_competitor_summary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(missing_app_rows).to_csv(os.path.join(out_dir, "STEP0_completion_product_missing_apps.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(missing_src_rows).to_csv(os.path.join(out_dir, "STEP0_completion_product_missing_srcs.csv"), index=False, encoding="utf-8-sig")

    # Return compact numeric columns for modeling (optional merge)
    keep_num = ["name", "supplier", "comp_app_entropy", "comp_src_entropy", "comp_app_top1_prob", "comp_src_top1_prob", "comp_expected_heat_total"]
    return df_sum[keep_num].copy()

def compute_supplier_network_metrics_sa(df_sup_app: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal network metrics on SUP-APP bipartite:
      degree, strength, pagerank
    """
    G = build_bipartite_graph_from_counts(df_sup_app, "SUP::", "APP::", "supplier", "app")
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["supplier"])

    sup_nodes = [n for n in G.nodes() if str(n).startswith("SUP::")]
    deg = {n: float(G.degree(n)) for n in sup_nodes}
    strength = {n: float(sum(G[n][nbr].get("weight", 1.0) for nbr in G.neighbors(n))) for n in sup_nodes}
    pr = nx.pagerank(G, weight="weight", max_iter=200, tol=1e-8)

    rows = []
    for n in sup_nodes:
        rows.append({
            "supplier": str(n).replace("SUP::", ""),
            "supplier_net_degree_sa": float(deg.get(n, 0.0)),
            "supplier_net_strength_sa": float(strength.get(n, 0.0)),
            "supplier_net_pagerank_sa": float(pr.get(n, 0.0)),
        })
    return pd.DataFrame(rows)

def supplier_expected_heat(df_sup_app: pd.DataFrame, app_heat_total: Dict[str, float]) -> pd.DataFrame:
    """
    expected_heat_total per supplier using empirical supplier-app distribution (counts-based).
    This is a minimal "expected_heat" proxy without heavy PPR.
    """
    if df_sup_app is None or df_sup_app.empty:
        return pd.DataFrame(columns=["supplier", "supplier_expected_heat_total", "supplier_app_entropy"])

    d = df_sup_app.copy()
    d["prod_cnt"] = pd.to_numeric(d["prod_cnt"], errors="coerce").fillna(0.0).astype(float)
    d["app"] = d["app"].astype(str).str.strip()
    d["supplier"] = d["supplier"].astype(str).str.strip()
    d = d[d["prod_cnt"] > 0].copy()
    if d.empty:
        return pd.DataFrame(columns=["supplier", "supplier_expected_heat_total", "supplier_app_entropy"])

    tot = d.groupby("supplier")["prod_cnt"].sum().rename("tot").reset_index()
    d = d.merge(tot, on="supplier", how="left")
    d["w"] = d["prod_cnt"] / np.maximum(d["tot"], 1e-12)
    d["heat"] = d["app"].map(lambda a: float(app_heat_total.get(a, 0.0)))

    def ent(p: np.ndarray) -> float:
        p = p / (p.sum() + 1e-12)
        return float(-np.sum(p * np.log(p + 1e-12)))

    out = d.groupby("supplier").apply(
        lambda g: pd.Series({
            "supplier_expected_heat_total": float(np.sum(g["w"].values * g["heat"].values)),
            "supplier_app_entropy": ent(g["w"].values.astype(float)),
        })
    ).reset_index()
    return out

def row_mean_from_label_table(list_any: Any, table: pd.DataFrame, key: str, cols: List[str]) -> Dict[str, float]:
    labs = parse_list_cell(list_any)
    if not labs:
        return {c: 0.0 for c in cols}
    sub = table[table[key].isin(labs)]
    if sub.empty:
        return {c: 0.0 for c in cols}
    out = {}
    for c in cols:
        out[c] = float(pd.to_numeric(sub[c], errors="coerce").fillna(0.0).mean())
    return out

# =============================================================================
# STEP0 -- stats price files
# =============================================================================
def _safe_to_float(x):
    """Convert to float; non-convertible -> NaN."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip()
        if x == "":
            return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

def make_stats_table(df: pd.DataFrame, cols):
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.dropna()
        rows.append({
            "variable": c,
            "count": int(s.shape[0]),
            "mean": float(s.mean()) if len(s) else np.nan,
            "std": float(s.std(ddof=1)) if len(s) else np.nan,
            "min": float(s.min()) if len(s) else np.nan,
            "P25": float(s.quantile(0.25)) if len(s) else np.nan,
            "P50": float(s.quantile(0.50)) if len(s) else np.nan,
            "P75": float(s.quantile(0.75)) if len(s) else np.nan,
            "max": float(s.max()) if len(s) else np.nan,
        })
    out = pd.DataFrame(rows)
    # 可选：统一小数位
    num_cols = ["mean", "std", "min", "P25", "P50", "P75", "max"]
    out[num_cols] = out[num_cols].round(6)
    return out

def stat_price(out_dir):
    df = pd.read_excel(EXCEL_PATH)

    # 标准化 price
    if PRICE_COL not in df.columns:
        raise KeyError(f"Column '{PRICE_COL}' not found in {XLSX_PATH}. Columns: {list(df.columns)}")

    df["price_clean"] = df[PRICE_COL].map(_safe_to_float)

    # log10(price)：<=0 的价格无法取对数，置为 NaN
    df["log10_price"] = np.where(df["price_clean"] > 0, np.log10(df["price_clean"]), np.nan)

    stats = make_stats_table(df, cols=["price_clean", "log10_price"])
    stats = stats.replace({"price_clean": "price", "log10_price": "log10(price)"})

    stats.to_csv(os.path.join(out_dir, "STEP0_price_stats.csv"))

# =============================================================================
# 5-block ablation (add COMP block for competitor effects)
# =============================================================================
def shapley_blocks_from_cv_long(cv_long: pd.DataFrame, blocks: List[str], metric_col: str = "R2_log10") -> pd.DataFrame:
    """
    Exact Shapley values per fold using all subset scores.
    Requires cv_long contains all subsets for each fold.
    """
    import itertools

    out_rows = []
    for fold, d in cv_long.groupby("fold"):
        score_map = {}
        for _, r in d.iterrows():
            scen = str(r["scenario"])
            if scen == "NONE":
                key = frozenset()
            else:
                key = frozenset([x.strip() for x in scen.split("+") if x.strip()])
            score_map[key] = float(r.get(metric_col, np.nan))

        # Fill missing subsets with NaN
        all_sets = [frozenset([blocks[i] for i in range(len(blocks)) if (mask & (1 << i))]) for mask in range(1 << len(blocks))]
        for s in all_sets:
            if s not in score_map:
                score_map[s] = np.nan

        perms = list(itertools.permutations(blocks))
        shap = {b: 0.0 for b in blocks}
        cnt = {b: 0 for b in blocks}

        for perm in perms:
            cur = frozenset()
            cur_sc = score_map.get(cur, np.nan)
            if not np.isfinite(cur_sc):
                continue
            for b in perm:
                nxt = frozenset(set(cur) | {b})
                nxt_sc = score_map.get(nxt, np.nan)
                if np.isfinite(nxt_sc):
                    shap[b] += (nxt_sc - cur_sc)
                    cnt[b] += 1
                    cur = nxt
                    cur_sc = nxt_sc
                else:
                    # If missing score, stop this permutation path
                    break

        for b in blocks:
            shap[b] = float(shap[b] / max(1, cnt[b]))

        out_rows.append({"fold": int(fold), **{f"shapley_{b}": shap[b] for b in blocks}})
    return pd.DataFrame(out_rows)

def dropone_blocks_from_cv_summary_ordered(
    cv_summary: pd.DataFrame,
    blocks: List[str],
    metric_mean_col: str = "R2_log10_mean",
) -> pd.DataFrame:
    """
    drop-one using the summary mean scores:
      delta = score(ALL) - score(ALL \\ {block})

    关键：scenario 名字按 blocks 的顺序拼接，确保与 scenario_grid_* 输出一致。
    """
    def scen_name(use_set: set) -> str:
        if not use_set:
            return "NONE"
        return "+".join([b for b in blocks if b in use_set])

    # map: scenario -> mean score
    score = {str(r["scenario"]): float(r.get(metric_mean_col, np.nan)) for _, r in cv_summary.iterrows()}

    full_set = set(blocks)
    full_scen = scen_name(full_set)
    full = score.get(full_scen, np.nan)

    rows = []
    for b in blocks:
        s2 = full_set - {b}
        scen2 = scen_name(s2)
        rows.append({
            "drop": b,
            "R2_full": float(full),
            "R2_dropped": float(score.get(scen2, np.nan)),
            "delta_R2": float(full - score.get(scen2, np.nan)),
        })
    return pd.DataFrame(rows)

def dropone_blocks_from_cv_summary(cv_summary: pd.DataFrame, blocks: List[str], metric_mean_col: str = "R2_log10_mean") -> pd.DataFrame:
    """
    drop-one using the summary mean scores:
      delta = score(ALL) - score(ALL \ {block})
    """
    # def scen_name(use_set: set) -> str:
    #     if not use_set:
    #         return "NONE"
    #     return "+".join([b for b in ["FE", "PROD", "DEM", "STR", "COMP"] if b in use_set])

    def scen_name(use_set: set) -> str:
        if not use_set:
            return "NONE"
        return "+".join([b for b in ["FE", "PROD", "DEM", "STR", ] if b in use_set])

    r2 = {str(r["scenario"]): float(r.get(metric_mean_col, np.nan)) for _, r in cv_summary.iterrows()}
    full = r2.get("FE+PROD+DEM+STR", np.nan)

    rows = []
    full_set = set(blocks)
    for b in blocks:
        s2 = full_set - {b}
        rows.append({
            "drop": b,
            "R2_full": float(full),
            "R2_dropped": float(r2.get(scen_name(s2), np.nan)),
            "delta_R2": float(full - r2.get(scen_name(s2), np.nan)),
        })
    return pd.DataFrame(rows)

# =============================================================================
# Model pipeline (simple linear BayesianRidge)
# =============================================================================
def _make_ohe(drop_first: bool = True):
    # sklearn version compatibility: sparse vs sparse_output
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True, drop="first" if drop_first else None)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True, drop="first" if drop_first else None)

def make_linear_pipeline_paper(
    use_fe: bool,
    use_product: bool,
    use_supply: bool,
    use_demand: bool,
    use_structure: bool,
    numeric_cols: List[str],
    min_label_freq: int,
) -> Pipeline:
    transformers = []

    # FE block
    if use_fe:
        transformers.append(("supplier_ohe", _make_ohe(drop_first=True), ["supplier"]))
        transformers.append(("src_mhot", MultiHotEncoder(normalize=True, min_freq=min_label_freq, prefix="src"), ["src_list"]))
        transformers.append(("app_mhot", MultiHotEncoder(normalize=True, min_freq=min_label_freq, prefix="app"), ["app_list"]))

    # product topic_id is categorical but only if product block enabled
    if use_product:
        if "topic_id" in numeric_cols:
            # should not happen; topic_id is categorical here
            pass
        transformers.append(("topic_ohe", _make_ohe(drop_first=True), ["topic_id"]))

    # numeric columns (union of enabled blocks)
    if numeric_cols:
        transformers.append(("num", Pipeline([("scaler", StandardScaler())]), numeric_cols))

    if not transformers:
        # intercept-only
        from sklearn.preprocessing import FunctionTransformer
        pre = FunctionTransformer(lambda X: np.ones((X.shape[0], 1)))
        return Pipeline([("pre", pre), ("model", BayesianRidge())])

    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    return Pipeline([("pre", pre), ("dense", ToDense()), ("model", BayesianRidge())])

# =============================================================================
# STEP0 (paper): FE / PRODUCT(with competitor) / DEMAND / SUPPLY / MARKET-STRUCTURE
# =============================================================================
def descriptive_stats_table(
    df: pd.DataFrame,
    cols: List[str],
    round_digits: int = 6,
) -> pd.DataFrame:
    """Compute descriptive statistics for selected columns.

    Output columns: variable, count, mean, std, min, P25, P50, P75, max.

    Notes
    -----
    - Non-numeric columns are coerced via pd.to_numeric(errors="coerce").
    - inf/-inf are treated as missing.
    - 'count' is the number of non-missing numeric observations.
    """
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        cnt = int(s.notna().sum())
        if cnt == 0:
            continue
        rows.append({
            "variable": c,
            "count": cnt,
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)),
            "min": float(s.min()),
            "P25": float(s.quantile(0.25)),
            "P50": float(s.quantile(0.50)),
            "P75": float(s.quantile(0.75)),
            "max": float(s.max()),
        })
    out = pd.DataFrame(rows)
    if not out.empty and round_digits is not None:
        for c in ["mean", "std", "min", "P25", "P50", "P75", "max"]:
            out[c] = out[c].round(round_digits)
    return out

def save_descriptive_stats_table(
    df: pd.DataFrame,
    cols: List[str],
    out_csv: str,
    round_digits: int = 6,
) -> pd.DataFrame:
    """Save descriptive statistics table to CSV (utf-8-sig)."""
    ensure_dir(os.path.dirname(out_csv) or ".")
    stats_df = descriptive_stats_table(df=df, cols=cols, round_digits=round_digits)
    stats_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return stats_df

def scenario_grid_5block_paper() -> List[Dict[str, Any]]:
    """
    Paper blocks:
      FE, PROD (incl. competitor sim stats), DEM, SUP, MKT
    """
    blocks = ["FE", "PROD", "DEM", "SUP", "MKT"]
    scenarios = []
    for mask in range(1 << len(blocks)):
        on = [(mask >> i) & 1 for i in range(len(blocks))]
        name = "+".join([b for b, f in zip(blocks, on) if f]) if any(on) else "NONE"
        scenarios.append({
            "name": name,
            "use_FE": on[0],
            "use_PROD": on[1],
            "use_DEM": on[2],
            "use_SUP": on[3],
            "use_MKT": on[4],
        })
    return scenarios

def evaluate_5block_paper(df: pd.DataFrame, folds: List[Tuple[np.ndarray, np.ndarray]], out_dir: str, min_label_freq: int = 3):
    """
    STEP0 block evaluation aligned to paper 3.4.1:
      - FE
      - PRODUCT (keeps competitor info inside product via prod_sim_* stats; same as evaluate_4block)
      - DEMAND (product_heat_* + supplier_expected_heat_* + paper_Bd / row_app_mean_heat)
      - SUPPLY (paper_Bs_* / paper_N_* + n_suppliers + total_products counts)
      - MARKET STRUCTURE (HHI/CRk/thickness/tension/monopoly/network + paper_HHI_invN_* + paper_T_*)
    """
    ensure_dir(out_dir)

    scenarios = scenario_grid_5block_paper()

    base_text_col = "desc"
    base_fe_cols = ["supplier", "src_list", "app_list"]

    cv_long = []

    # Demand pool
    demand_cols = ["paper_heat_total", "paper_heat_web", "paper_heat_news", "paper_heat_weixin"]
    demand_cols = sorted(set([c for c in demand_cols if c in df.columns]))

    # Supply pool (counts + Bs)
    supply_candidates = [
        "paper_Bs",
    ]

    supply_cols = [c for c in supply_candidates if c in df.columns]
    supply_cols = sorted(set(supply_cols))

    # add paper HHI & thickness (T) explicitly if present
    market_cols = [c for c in ["paper_HHI_proxy", "paper_T"] if c in df.columns]
    market_cols = sorted(set([c for c in market_cols if c in df.columns]))

    # --- Descriptive stats (paper-style) for variables used in STEP0 5-block evaluation
    # Saved before CV loops so the paper can report sample-level summaries.
    stats_cols = []
    for c in ["price", "y_log10"]:
        if c in df.columns:
            stats_cols.append(c)

    stats_cols += demand_cols + supply_cols + market_cols
    stats_cols = [c for c in dict.fromkeys(stats_cols) if c in df.columns]

    stats_path = os.path.join(out_dir, "STEP0_paper_5block_paper_variable_stats.csv")

    if not os.path.exists(stats_path):
        save_descriptive_stats_table(df=df, cols=stats_cols, out_csv=stats_path, round_digits=6)

    for fold_id, fd in enumerate(folds, start=1):
        tr = np.array(fd["train_idx"], dtype=int)
        te = np.array(fd["test_idx"], dtype=int)

        dtr = df.iloc[tr].copy()
        dte = df.iloc[te].copy()

        ytr = dtr["y_log10"].values.astype(float)
        yte = dte["y_log10"].values.astype(float)

        # -------------------------
        # PRODUCT block (fit on train only)
        # -------------------------
        tfidf, svd = fit_text_embedder(dtr[base_text_col].fillna("").astype(str).values, dim=TEXT_EMB_DIM,
                                       min_df=TEXT_MIN_DF)
        emb_tr = transform_text_embedder(tfidf, svd, dtr[base_text_col].fillna("").astype(str).values)
        emb_te = transform_text_embedder(tfidf, svd, dte[base_text_col].fillna("").astype(str).values)

        # similarity features (neighbors in train only)
        sim_tr = knn_similarity_features(emb_tr, emb_tr, k=SIM_KNN_K)
        sim_te = knn_similarity_features(emb_tr, emb_te, k=SIM_KNN_K)

        # topics (fit on train only)
        km = fit_topics(emb_tr, k_topics=TOPIC_K)
        topic_tr = km.predict(emb_tr).astype(int) if emb_tr.shape[0] > 0 and emb_tr.shape[1] > 0 else np.zeros(len(dtr),
                                                                                                               dtype=int)
        topic_te = km.predict(emb_te).astype(int) if emb_te.shape[0] > 0 and emb_te.shape[1] > 0 else np.zeros(len(dte),
                                                                                                               dtype=int)

        # attach product features into copies
        # embedding columns
        prod_emb_cols = [f"prod_emb_{i}" for i in range(emb_tr.shape[1])]
        for i, c in enumerate(prod_emb_cols):
            dtr[c] = emb_tr[:, i]
            dte[c] = emb_te[:, i]

        dtr["prod_sim_mean"] = sim_tr[:, 0]
        dtr["prod_sim_max"] = sim_tr[:, 1]
        dtr["prod_sim_std"] = sim_tr[:, 2]
        dtr["prod_sim_sum"] = sim_tr[:, 3]

        dte["prod_sim_mean"] = sim_te[:, 0]
        dte["prod_sim_max"] = sim_te[:, 1]
        dte["prod_sim_std"] = sim_te[:, 2]
        dte["prod_sim_sum"] = sim_te[:, 3]

        dtr["topic_id"] = topic_tr
        dte["topic_id"] = topic_te

        product_numeric_cols = prod_emb_cols + ["prod_sim_mean", "prod_sim_max", "prod_sim_std", "prod_sim_sum"]
        # topic_id is categorical, handled separately in pipeline

        for sc in scenarios:
            use_fe = bool(sc["use_FE"])
            use_prod = bool(sc["use_PROD"])
            use_dem = bool(sc["use_DEM"])
            use_sup = bool(sc["use_SUP"])
            use_mkt = bool(sc["use_MKT"])

            num_cols = []
            if use_prod:
                num_cols += product_numeric_cols

            if use_dem:
                num_cols += demand_cols

            if use_sup:
                num_cols += supply_cols

            if use_mkt:
                num_cols += market_cols

            # keep stable order & only existing columns
            num_cols = [c for c in dict.fromkeys(num_cols) if c in df.columns]

            Xtr = dtr[base_fe_cols + [base_text_col, "topic_id"] + num_cols].copy()
            Xte = dte[base_fe_cols + [base_text_col, "topic_id"] + num_cols].copy()

            Xtr["supplier"] = Xtr["supplier"].astype(str)
            Xte["supplier"] = Xte["supplier"].astype(str)
            Xtr["topic_id"] = Xtr["topic_id"].astype(int).astype(str)
            Xte["topic_id"] = Xte["topic_id"].astype(int).astype(str)

            pipe = make_linear_pipeline_paper(
                use_fe=use_fe,
                use_product=use_prod,
                use_supply=use_sup,
                use_demand=use_dem,
                use_structure=use_mkt,
                numeric_cols=num_cols,
                min_label_freq=min_label_freq,
            )

            pipe.fit(Xtr, ytr)
            yhat, ystd = pipe.predict(Xte, return_std=True)
            met = metrics_on_log10_and_price(yte, yhat, ystd)

            met.update({
                "fold": fold_id,
                "scenario": sc["name"],
                "use_FE": sc["use_FE"],
                "use_PROD": sc["use_PROD"],
                "use_DEM": sc["use_DEM"],
                "use_SUP": sc["use_SUP"],
                "use_MKT": sc["use_MKT"],
                "n_num_cols": int(len(num_cols)),
            })

            cv_long.append(met)

    df_cv_long = pd.DataFrame(cv_long)
    df_cv_long.to_csv(os.path.join(out_dir, "STEP0_paper_5block_paper_cv_long.csv"), index=False, encoding="utf-8-sig")

    # summary mean/std
    agg = {c: ["mean", "std"] for c in df_cv_long.columns if c not in {"fold", "scenario", "use_FE", "use_PROD", "use_DEM", "use_SUP", "use_MKT"}}
    summary = df_cv_long.groupby(["scenario", "use_FE", "use_PROD", "use_DEM", "use_SUP", "use_MKT"], as_index=False).agg(agg)

    summary.columns = [
        (c[0] if c[1] == "" else f"{c[0]}_{c[1]}") if isinstance(c, tuple) else c
        for c in summary.columns
    ]

    summary.to_csv(os.path.join(out_dir, "STEP0_paper_5block_paper_summary.csv"), index=False, encoding="utf-8-sig")

    blocks = ["FE", "PROD", "DEM", "SUP", "MKT"]
    df_shapley = shapley_blocks_from_cv_long(df_cv_long, blocks=blocks, metric_col="R2_log10")
    df_shapley.to_csv(os.path.join(out_dir, "STEP0_paper_5block_paper_shapley.csv"), index=False, encoding="utf-8-sig")

    df_drop = dropone_blocks_from_cv_summary_ordered(summary, blocks=blocks, metric_mean_col="R2_log10_mean")
    df_drop.to_csv(os.path.join(out_dir, "STEP0_paper_5block_paper_dropone.csv"), index=False, encoding="utf-8-sig")

    return df_cv_long, summary

# =============================================================================
# STEP1: FE main-effect suite (ablation + 2 CV + Shapley + drop-one + optional HB)
# =============================================================================
BLOCKS = ["supplier", "src", "app"]
SCENARIOS_FE: Dict[str, Dict[str, bool]] = {
    "no_fe": {"supplier": False, "src": False, "app": False},
    "fe_supplier": {"supplier": True, "src": False, "app": False},
    "fe_src": {"supplier": False, "src": True, "app": False},
    "fe_app": {"supplier": False, "src": False, "app": True},
    "fe_supplier_src": {"supplier": True, "src": True, "app": False},
    "fe_supplier_app": {"supplier": True, "src": False, "app": True},
    "fe_src_app": {"supplier": False, "src": True, "app": True},
    "fe_all": {"supplier": True, "src": True, "app": True},
}

def make_fe_pipeline(use_supplier: bool, use_src: bool, use_app: bool, min_freq: int = 1) -> Pipeline:
    transformers = []

    if use_supplier:
        transformers.append(("supplier_ohe",
                             OneHotEncoder(handle_unknown="ignore", sparse=True, drop="first"),
                             ["supplier"]))
    if use_src:
        transformers.append(("src_mhot",
                             MultiHotEncoder(normalize=True, min_freq=min_freq, prefix="src"),
                             ["src_list"]))
    if use_app:
        transformers.append(("app_mhot",
                             MultiHotEncoder(normalize=True, min_freq=min_freq, prefix="app"),
                             ["app_list"]))

    if not transformers:
        # intercept-only
        from sklearn.preprocessing import FunctionTransformer
        pre = FunctionTransformer(lambda X: np.ones((X.shape[0], 1)))
        return Pipeline([("pre", pre), ("model", BayesianRidge())])

    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    return Pipeline([("pre", pre), ("dense", ToDense()), ("model", BayesianRidge())])

def shapley_r2_from_fold(r2_map: Dict[str, float]) -> Dict[str, float]:
    """
    Exact Shapley for 3 blocks using 2^3 subset R2 values.
    """
    subset_to_key = {
        frozenset(): "no_fe",
        frozenset(["supplier"]): "fe_supplier",
        frozenset(["src"]): "fe_src",
        frozenset(["app"]): "fe_app",
        frozenset(["supplier", "src"]): "fe_supplier_src",
        frozenset(["supplier", "app"]): "fe_supplier_app",
        frozenset(["src", "app"]): "fe_src_app",
        frozenset(["supplier", "src", "app"]): "fe_all",
    }

    import itertools

    shap = {b: 0.0 for b in BLOCKS}
    perms = list(itertools.permutations(BLOCKS))

    for perm in perms:
        cur = set()
        cur_r2 = r2_map[subset_to_key[frozenset(cur)]]

        for b in perm:
            new = cur | {b}
            new_r2 = r2_map[subset_to_key[frozenset(new)]]
            shap[b] += (new_r2 - cur_r2)
            cur = new
            cur_r2 = new_r2

    for b in BLOCKS:
        shap[b] /= len(perms)

    return shap

def compute_shapley_table(cv_long: pd.DataFrame, tag: str) -> pd.DataFrame:
    out = []

    for fold, d in cv_long.groupby("fold"):
        r2_map = d.set_index("scenario")["R2_log10"].to_dict()
        shap = shapley_r2_from_fold(r2_map)
        total = sum(shap.values()) if abs(sum(shap.values())) > 1e-12 else np.nan
        out.append({
            "cv_tag": tag,
            "fold": int(fold),
            "shapley_supplier": shap["supplier"],
            "shapley_src": shap["src"],
            "shapley_app": shap["app"],
            "share_supplier": shap["supplier"] / total if total else np.nan,
            "share_src": shap["src"] / total if total else np.nan,
            "share_app": shap["app"] / total if total else np.nan,
        })

    return pd.DataFrame(out)

def compute_dropone(summary: pd.DataFrame, tag: str) -> pd.DataFrame:
    r2 = {row["scenario"]: row["R2_log10_mean"] for _, row in summary.iterrows()}
    full = r2.get("fe_all", np.nan)
    rows = [
        {"cv_tag": tag, "drop": "supplier", "R2_full": full, "R2_dropped": r2.get("fe_src_app", np.nan), "delta_R2": full - r2.get("fe_src_app", np.nan)},
        {"cv_tag": tag, "drop": "src", "R2_full": full, "R2_dropped": r2.get("fe_supplier_app", np.nan), "delta_R2": full - r2.get("fe_supplier_app", np.nan)},
        {"cv_tag": tag, "drop": "app", "R2_full": full, "R2_dropped": r2.get("fe_supplier_src", np.nan), "delta_R2": full - r2.get("fe_supplier_src", np.nan)},
    ]

    return pd.DataFrame(rows)

# =============================================================================
# STEP1: Full-sample FE estimates (interpretable tables for supplier/src/app)
# =============================================================================
def fit_fullsample_fe_tables(df: pd.DataFrame, min_label_freq: int, out_dir: str) -> Dict[str, Any]:
    """
    Fit a full-sample BayesianRidge with:
      - Supplier OHE (explicit categories + drop-first baseline)
      - Src/App fractional multi-hot

    This is for interpretability (full-sample FE estimates), not for CV prediction.
    Coefficients are in log10-price space.
    """
    d = df.copy()
    y = d["y_log10"].astype(float).values

    # ---- Supplier OHE with explicit baseline ----
    suppliers = sorted(d["supplier"].astype(str).unique().tolist())
    if not suppliers:
        suppliers = ["UNKNOWN_SUPPLIER"]
        d["supplier"] = "UNKNOWN_SUPPLIER"

    from sklearn.preprocessing import OneHotEncoder

    try:
        enc = OneHotEncoder(
            categories=[suppliers],
            drop="first",
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        enc = OneHotEncoder(
            categories=[suppliers],
            drop="first",
            handle_unknown="ignore",
            sparse=False,
        )

    X_sup = enc.fit_transform(d[["supplier"]].astype(str))
    sup_feature_names = enc.get_feature_names_out(["supplier"]).tolist()

    supplier_baseline = enc.categories_[0][0]  # the dropped first category

    X_sup_df = pd.DataFrame(X_sup, columns=sup_feature_names).reset_index(drop=True)

    # ---- Src/App fractional multi-hot ----
    X_src, src_map = build_fractional_multihot_df(d["src_list"], prefix="SRC", min_freq=min_label_freq)
    X_app, app_map = build_fractional_multihot_df(d["app_list"], prefix="APP", min_freq=min_label_freq)

    X = pd.concat([X_sup_df, X_src.reset_index(drop=True), X_app.reset_index(drop=True)], axis=1).fillna(0.0)
    model = BayesianRidge()
    model.fit(X.values.astype(float), y)

    coef = pd.Series(model.coef_, index=X.columns)
    intercept = float(model.intercept_)

    # Supplier table (baseline = 0, others = coef on their OHE column)
    sup_coef = {supplier_baseline: 0.0}
    for s in suppliers:
        if s == supplier_baseline:
            continue
        # feature name is like "supplier_<value>"
        fname = f"supplier_{s}"
        if fname in coef.index:
            sup_coef[s] = float(coef[fname])
        else:
            sup_coef[s] = 0.0

    df_sup = pd.DataFrame([{
        "supplier": s,
        "fe_log10": float(sup_coef.get(s, 0.0)),
        "fe_multiplier_on_price": float(10.0 ** float(sup_coef.get(s, 0.0))),
        "baseline_supplier": supplier_baseline,
    } for s in suppliers]).sort_values("fe_log10", ascending=False).reset_index(drop=True)

    # SRC table
    inv_src = {v: k for k, v in src_map.items()}
    df_src = pd.DataFrame([{
        "src": inv_src.get(col, col),
        "fe_log10": float(coef.get(col, 0.0)),
        "fe_multiplier_on_price": float(10.0 ** float(coef.get(col, 0.0))),
        "note": "fractional multi-hot (effect at weight=1.0)",
    } for col in X_src.columns]).sort_values("fe_log10", ascending=False).reset_index(drop=True)

    # APP table
    inv_app = {v: k for k, v in app_map.items()}
    df_app = pd.DataFrame([{
        "app": inv_app.get(col, col),
        "fe_log10": float(coef.get(col, 0.0)),
        "fe_multiplier_on_price": float(10.0 ** float(coef.get(col, 0.0))),
        "note": "fractional multi-hot (effect at weight=1.0)",
    } for col in X_app.columns]).sort_values("fe_log10", ascending=False).reset_index(drop=True)

    df_sup.to_csv(os.path.join(out_dir, "STEP1_fe_table_supplier.csv"), index=False, encoding="utf-8-sig")
    df_src.to_csv(os.path.join(out_dir, "STEP1_fe_table_src.csv"), index=False, encoding="utf-8-sig")
    df_app.to_csv(os.path.join(out_dir, "STEP1_fe_table_app.csv"), index=False, encoding="utf-8-sig")

    meta = {
        "ok": True,
        "intercept_log10": intercept,
        "n_rows": int(len(d)),
        "n_supplier": int(len(suppliers)),
        "n_src_vocab": int(X_src.shape[1]),
        "n_app_vocab": int(X_app.shape[1]),
        "supplier_baseline": supplier_baseline,
    }
    with open(os.path.join(out_dir, "STEP1_fe_table_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta

# =============================================================================
# STEP1: Simple conclusion text
# =============================================================================
def conclude_step1(summary_k: pd.DataFrame, drop_k: pd.DataFrame) -> str:
    def get_r2(summary: pd.DataFrame, scen: str) -> float:
        row = summary[summary["scenario"] == scen]
        return float(row["R2_log10_mean"].values[0]) if len(row) else float("nan")

    def drop_delta(drop_df: pd.DataFrame, who: str) -> float:
        r = drop_df[drop_df["drop"] == who]
        return float(r["delta_R2"].values[0]) if len(r) else float("nan")

    r2_sup_k = get_r2(summary_k, "fe_supplier")
    r2_all_k = get_r2(summary_k, "fe_all")

    d_sup_k = drop_delta(drop_k, "supplier")
    d_src_k = drop_delta(drop_k, "src")
    d_app_k = drop_delta(drop_k, "app")

    lines = []
    lines.append("STEP1 conclusion (FE diagnosis):")
    lines.append(f"- KFold: R2(fe_supplier)={r2_sup_k:.4f}, R2(fe_all)={r2_all_k:.4f}")
    lines.append(f"- Drop-one ΔR2 (KFold): supplier={d_sup_k:.4f}, src={d_src_k:.4f}, app={d_app_k:.4f}")

    return "\n".join(lines)

# =============================================================================
# STEP2: Market structure / structural reasoning features
# =============================================================================
def compute_category_market_structure(df_pair: pd.DataFrame, cat_col: str, supplier_col: str) -> pd.DataFrame:
    """
    Generic market structure for a category (cat_col) over suppliers:
      - total products
      - n_suppliers
      - HHI
      - CR4 / CR8
    df_pair must have columns: cat_col, supplier_col, prod_cnt
    """
    if df_pair is None or df_pair.empty:
        return pd.DataFrame()

    d = df_pair.copy()
    d["prod_cnt"] = pd.to_numeric(d["prod_cnt"], errors="coerce").fillna(0.0)
    d[cat_col] = d[cat_col].astype(str).str.strip()
    d[supplier_col] = d[supplier_col].astype(str).str.strip()

    totals = d.groupby(cat_col)["prod_cnt"].sum().reset_index().rename(columns={"prod_cnt": f"{cat_col}_total_products"})
    d = d.merge(totals, on=cat_col, how="left")
    d["share"] = d["prod_cnt"] / np.maximum(d[f"{cat_col}_total_products"], 1e-12)

    hhi = d.groupby(cat_col)["share"].apply(lambda s: float(np.sum(np.square(s.values)))).reset_index().rename(columns={"share": "HHI"})

    def crk(g, k: int) -> float:
        ss = np.sort(g["share"].values)[::-1]
        return float(np.sum(ss[:k])) if len(ss) > 0 else 0.0

    cr4 = d.groupby(cat_col).apply(lambda g: crk(g, 4)).reset_index().rename(columns={0: "CR4"})
    cr8 = d.groupby(cat_col).apply(lambda g: crk(g, 8)).reset_index().rename(columns={0: "CR8"})

    n_sup = d.groupby(cat_col)[supplier_col].nunique().reset_index().rename(columns={supplier_col: "n_suppliers"})
    out = totals.merge(n_sup, on=cat_col, how="left").merge(hhi, on=cat_col, how="left").merge(cr4, on=cat_col, how="left").merge(cr8, on=cat_col, how="left")

    return out

def compute_market_thickness_for_app(
    df_app_stats: pd.DataFrame,
    app_heat_total: Dict[str, float],
    heat_is_log: bool = True,   # True means heat is already on log1p(count) scale.
) -> pd.DataFrame:
    """
    Market thickness proxy by app:
      thickness = log1p(n_suppliers) + log1p(app_total_products) + heat_component

    If heat_is_log=True: use heat directly (already log scale).
    If heat_is_log=False: use log1p(max(heat_raw, 0)).
    """
    if df_app_stats is None or df_app_stats.empty:
        return pd.DataFrame(columns=["app", "heat", "thickness"])

    d = df_app_stats.copy()

    # Ensure required columns exist
    if "n_suppliers" not in d.columns:
        d["n_suppliers"] = 0.0

    if "app_total_products" not in d.columns:
        # Fallback if the total-products column was renamed unexpectedly
        for cand in ["app_total_products", "app_total_products_sum", "app_total_products_total"]:
            if cand in d.columns:
                d["app_total_products"] = d[cand]
                break
        if "app_total_products" not in d.columns:
            d["app_total_products"] = 0.0

    d["heat"] = d["app"].map(lambda a: float(app_heat_total.get(a, 0.0)))
    heat_component = d["heat"].values.astype(float)
    if not heat_is_log:
        heat_component = log10_1p(np.maximum(heat_component, 0.0))

    d["thickness"] = (
        log10_1p(np.maximum(d["n_suppliers"].values.astype(float), 0.0))
        + log10_1p(np.maximum(d["app_total_products"].values.astype(float), 0.0))
        + heat_component
    )

    # ---- demand–supply tension (log-scale imbalance) ----
    # heat_component is already on log scale when heat_is_log=True.
    # B = log(1 + D) - log(1 + N)
    d["tension_B_nsup"] = heat_component - log10_1p(np.maximum(d["n_suppliers"].values.astype(float), 0.0))
    d["tension_B_nprod"] = heat_component - log10_1p(np.maximum(d["app_total_products"].values.astype(float), 0.0))

    return d[["app", "heat", "thickness", "tension_B_nsup", "tension_B_nprod"]].copy()

def compute_supplier_monopoly_metrics(
    df_sup_app: pd.DataFrame,
    df_app_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build supplier-level monopoly / market-power metrics using SUP-APP counts.

    Requires df_sup_app columns: supplier, app, prod_cnt
    Optional df_app_stats provides app-level HHI/CRk etc for exposure measures.
    """
    if df_sup_app is None or df_sup_app.empty:
        return pd.DataFrame(columns=["supplier"])

    d = df_sup_app.copy()
    d["supplier"] = d["supplier"].astype(str).str.strip()
    d["app"] = d["app"].astype(str).str.strip()
    d["prod_cnt"] = pd.to_numeric(d["prod_cnt"], errors="coerce").fillna(0.0).astype(float)
    d = d[d["prod_cnt"] > 0].copy()
    if d.empty:
        return pd.DataFrame(columns=["supplier"])

    # app 市场总量
    app_tot = d.groupby("app")["prod_cnt"].sum().rename("app_total_products").reset_index()
    d = d.merge(app_tot, on="app", how="left")
    d["app_total_products"] = np.maximum(d["app_total_products"].values, 1e-12)

    # supplier 在 app 内的市场份额
    d["share_in_app"] = d["prod_cnt"] / d["app_total_products"]

    # supplier 自身 portfolio 权重（在各 app 的配置）
    sup_tot = d.groupby("supplier")["prod_cnt"].sum().rename("supplier_total_products").reset_index()
    d = d.merge(sup_tot, on="supplier", how="left")
    d["supplier_total_products"] = np.maximum(d["supplier_total_products"].values, 1e-12)
    d["w_portfolio"] = d["prod_cnt"] / d["supplier_total_products"]

    # portfolio market power：加权平均市场份额
    sup_power = d.groupby("supplier").apply(
        lambda g: pd.Series({
            "supplier_total_products_in_app": float(g["prod_cnt"].sum()),
            "supplier_num_app": int(g["app"].nunique()),
            "supplier_portfolio_weighted_share": float(np.sum(g["w_portfolio"].values * g["share_in_app"].values)),
            "supplier_portfolio_top_app_share": float(np.max(g["share_in_app"].values)),
        })
    ).reset_index()

    sup_power2 = d.groupby("supplier").apply(
        lambda g: pd.Series({
            "supplier_monopoly_hhi": float(np.sum(g["w_portfolio"].values * (g["share_in_app"].values ** 2))),
        })
    ).reset_index()

    # portfolio 集中度（在 app 上）
    def hhi_entropy(g):
        p = g["w_portfolio"].values.astype(float)
        p = p / (p.sum() + 1e-12)
        hhi = float(np.sum(p**2))
        ent = float(-np.sum(p * np.log10(p + 1e-12)))
        return pd.Series({"supplier_portfolio_app_hhi": hhi, "supplier_portfolio_app_entropy": ent})

    sup_div = d.groupby("supplier").apply(hhi_entropy).reset_index()
    out = sup_power.merge(sup_div, on="supplier", how="left")
    out = out.merge(sup_power2, on="supplier", how="left")

    # 暴露于集中市场：weighted app HHI / CR4 / CR8（可选）
    if df_app_stats is not None and not df_app_stats.empty:
        a = df_app_stats.copy()
        a["app"] = a["app"].astype(str).str.strip()
        keep_cols = [c for c in ["app", "HHI", "CR4", "CR8", "n_suppliers", "app_total_products"] if c in a.columns]
        a = a[keep_cols].drop_duplicates("app")
        d2 = d.merge(a, on="app", how="left")
        for c in ["HHI", "CR4", "CR8"]:
            if c not in d2.columns:
                d2[c] = 0.0
            d2[c] = pd.to_numeric(d2[c], errors="coerce").fillna(0.0).astype(float)

        expo = d2.groupby("supplier").apply(
            lambda g: pd.Series({
                "supplier_portfolio_weighted_app_HHI": float(np.sum(g["w_portfolio"].values * g["HHI"].values)),
                "supplier_portfolio_weighted_app_CR4": float(np.sum(g["w_portfolio"].values * g["CR4"].values)),
                "supplier_portfolio_weighted_app_CR8": float(np.sum(g["w_portfolio"].values * g["CR8"].values)),
            })
        ).reset_index()
        out = out.merge(expo, on="supplier", how="left")

    # --- canonical supplier-monopoly indicators (paper-facing) ---
    if "supplier_portfolio_weighted_share" in out.columns:
        out["supplier_monopoly"] = out["supplier_portfolio_weighted_share"].astype(float)

    if "supplier_portfolio_top_app_share" in out.columns:
        out["supplier_monopoly_topshare"] = out["supplier_portfolio_top_app_share"].astype(float)

    return out.fillna(0.0)

def build_product_similarity_graph(prod_emb_df: pd.DataFrame, k: int = 10, min_cos: float = 0.25) -> nx.Graph:
    """
    Build KNN similarity graph among products using cosine similarity on text embeddings.
    Nodes: PROD::{name}|SUP::{supplier}  (unique key)
    """
    emb_cols = [c for c in prod_emb_df.columns if c.startswith("textemb_")]
    d = prod_emb_df.copy()
    X = d[emb_cols].astype(float).values
    if len(d) < 5 or X.shape[1] == 0:
        return nx.Graph()

    # cosine distance => similarity = 1 - dist
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(d)), metric="cosine", algorithm="brute")
    nn.fit(X)
    dist, idx = nn.kneighbors(X, return_distance=True)

    G = nx.Graph()
    keys = [f"PROD::{r['name']}||SUP::{r['supplier']}" for _, r in d.iterrows()]
    for key in keys:
        G.add_node(key)

    for i in range(len(d)):
        for jpos in range(1, idx.shape[1]):  # skip itself
            j = int(idx[i, jpos])
            sim = float(1.0 - dist[i, jpos])
            if sim < min_cos:
                continue
            a, b = keys[i], keys[j]
            if a == b:
                continue
            if G.has_edge(a, b):
                if sim > G[a][b].get("weight", 0.0):
                    G[a][b]["weight"] = sim
            else:
                G.add_edge(a, b, weight=sim)
    return G

def build_topic_exposure(df_rows: pd.DataFrame, prod_emb_df: pd.DataFrame, k_topics: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return:
      - product_topics: name,supplier,topic_id
      - supplier_topic_share: supplier + topic_share_*
      - src_topic_share: src + src_topic_share_*
      - app_topic_share: app + app_topic_share_*
    """
    emb_cols = [c for c in prod_emb_df.columns if c.startswith("textemb_")]
    X = prod_emb_df[emb_cols].astype(float).values

    k_topics = int(min(k_topics, max(2, len(prod_emb_df) // 5)))
    k_topics = int(min(k_topics, max(2, len(prod_emb_df))))  # ensure <= n_samples
    km = KMeans(n_clusters=k_topics, random_state=RANDOM_STATE, n_init=10)

    topic = km.fit_predict(X)

    prod_topics = prod_emb_df[["name", "supplier"]].copy()
    prod_topics["topic_id"] = topic.astype(int)

    # supplier exposure
    sup = prod_topics.groupby(["supplier", "topic_id"]).size().rename("cnt").reset_index()
    sup_tot = sup.groupby("supplier")["cnt"].sum().rename("tot").reset_index()
    sup = sup.merge(sup_tot, on="supplier", how="left")
    sup["share"] = sup["cnt"] / np.maximum(sup["tot"], 1e-12)
    sup_wide = sup.pivot_table(index="supplier", columns="topic_id", values="share", fill_value=0.0)
    sup_wide.columns = [f"topic_share_{int(c)}" for c in sup_wide.columns]
    sup_wide = sup_wide.reset_index()

    # src/app exposure (fractional per row labels)
    base = df_rows.merge(prod_topics, on=["name", "supplier"], how="left")
    # SRC
    rows = []
    for _, r in base.iterrows():
        labs = parse_list_cell(r["src_list"])
        if not labs:
            continue
        w = 1.0 / len(labs)
        for lab in labs:
            rows.append({"src": lab, "topic_id": int(r["topic_id"]), "w": w})
    src_wide = pd.DataFrame()
    if rows:
        tmp = pd.DataFrame(rows)
        tmp = tmp.groupby(["src", "topic_id"])["w"].sum().reset_index()
        tot = tmp.groupby("src")["w"].sum().rename("tot").reset_index()
        tmp = tmp.merge(tot, on="src", how="left")
        tmp["share"] = tmp["w"] / np.maximum(tmp["tot"], 1e-12)
        src_wide = tmp.pivot_table(index="src", columns="topic_id", values="share", fill_value=0.0)
        src_wide.columns = [f"src_topic_share_{int(c)}" for c in src_wide.columns]
        src_wide = src_wide.reset_index()

    # APP
    rows = []
    for _, r in base.iterrows():
        labs = parse_list_cell(r["app_list"])
        if not labs:
            continue
        w = 1.0 / len(labs)
        for lab in labs:
            rows.append({"app": lab, "topic_id": int(r["topic_id"]), "w": w})
    app_wide = pd.DataFrame()
    if rows:
        tmp = pd.DataFrame(rows)
        tmp = tmp.groupby(["app", "topic_id"])["w"].sum().reset_index()
        tot = tmp.groupby("app")["w"].sum().rename("tot").reset_index()
        tmp = tmp.merge(tot, on="app", how="left")
        tmp["share"] = tmp["w"] / np.maximum(tmp["tot"], 1e-12)
        app_wide = tmp.pivot_table(index="app", columns="topic_id", values="share", fill_value=0.0)
        app_wide.columns = [f"app_topic_share_{int(c)}" for c in app_wide.columns]
        app_wide = app_wide.reset_index()

    return prod_topics, sup_wide, src_wide, app_wide

def compute_product_graph_features(G: nx.Graph, max_betweenness_n: int = 2000) -> pd.DataFrame:
    """
    Compute graph metrics for product similarity graph.
    If graph is large, skip betweenness to avoid heavy cost.
    """
    if G is None or G.number_of_nodes() == 0:
        return pd.DataFrame()

    deg = dict(G.degree())
    strength = {n: float(sum(G[n][nbr].get("weight", 1.0) for nbr in G.neighbors(n))) for n in G.nodes()}
    pr = nx.pagerank(G, weight="weight", max_iter=MAX_NX_PAGERANK_ITERS, tol=NX_PAGERANK_TOL)

    try:
        clust = nx.clustering(G, weight="weight")
    except Exception:
        clust = {n: 0.0 for n in G.nodes()}

    try:
        core = nx.core_number(G)
    except Exception:
        core = {n: 0 for n in G.nodes()}

    if G.number_of_nodes() <= max_betweenness_n:
        btw = nx.betweenness_centrality(G, weight="weight", normalized=True)
    else:
        btw = {n: 0.0 for n in G.nodes()}

    rows = []
    for n in G.nodes():
        rows.append({
            "prod_key": n,
            "prod_sim_deg": float(deg.get(n, 0.0)),
            "prod_sim_strength": float(strength.get(n, 0.0)),
            "prod_sim_pagerank": float(pr.get(n, 0.0)),
            "prod_sim_clustering": float(clust.get(n, 0.0)),
            "prod_sim_coreness": float(core.get(n, 0)),
            "prod_sim_betweenness": float(btw.get(n, 0.0)),
        })
    return pd.DataFrame(rows)

def compute_product_level_heat_components(df_products: pd.DataFrame, heat_maps: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    def mean_heat(apps_any: Any, hmap: Dict[str, float]) -> float:
        apps = parse_list_cell(apps_any)
        if not apps:
            return 0.0
        vals = [float(hmap.get(a, 0.0)) for a in apps]
        return float(np.mean(vals)) if vals else 0.0

    out = pd.DataFrame(index=df_products.index)
    out["paper_heat_total"] = df_products["app_list"].apply(lambda x: mean_heat(x, heat_maps["total_log"])).astype(float)
    out["paper_heat_web"] = df_products["app_list"].apply(lambda x: mean_heat(x, heat_maps["web_log"])).astype(float)
    out["paper_heat_news"] = df_products["app_list"].apply(lambda x: mean_heat(x, heat_maps["news_log"])).astype(float)
    out["paper_heat_weixin"] = df_products["app_list"].apply(lambda x: mean_heat(x, heat_maps["weixin_log"])).astype(float)

    return out

def build_product_text_embedding(df_rows, text_col="desc", dim=50, min_df=2):
    d = df_rows.copy()
    d[text_col] = d[text_col].fillna("").astype(str)

    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=min_df)
    X_tfidf = tfidf.fit_transform(d[text_col].values)

    # Guard for small matrices
    n_samples, n_features = X_tfidf.shape
    if n_samples < 3 or n_features < 3:
        emb = pd.DataFrame(np.zeros((len(d), 0)), index=range(len(d)))
        out = pd.concat([d[["name", "supplier"]].reset_index(drop=True), emb], axis=1)
        pipe = Pipeline([("tfidf", tfidf)])
        return out, pipe

    n_comp = int(min(dim, n_samples - 1, n_features - 1))
    n_comp = max(2, n_comp)

    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X = svd.fit_transform(X_tfidf)

    emb = pd.DataFrame(X, columns=[f"textemb_{i}" for i in range(X.shape[1])])
    out = pd.concat([d[["name", "supplier"]].reset_index(drop=True), emb], axis=1)

    pipe = Pipeline([("tfidf", tfidf), ("svd", svd)])
    return out, pipe

def build_bipartite_graph_from_counts(df_counts: pd.DataFrame, left_prefix: str, right_prefix: str, left_col: str, right_col: str) -> nx.Graph:
    """
    Build an undirected weighted bipartite graph with edge weight = prod_cnt.
    """
    G = nx.Graph()
    if df_counts is None or df_counts.empty:
        return G

    d = df_counts.copy()
    d["prod_cnt"] = pd.to_numeric(d["prod_cnt"], errors="coerce").fillna(0.0)

    for _, r in d.iterrows():
        l = f"{left_prefix}{str(r[left_col]).strip()}"
        rr = f"{right_prefix}{str(r[right_col]).strip()}"
        w = float(r["prod_cnt"])

        if w <= 0:
            continue

        if G.has_edge(l, rr):
            G[l][rr]["weight"] += w
        else:
            G.add_edge(l, rr, weight=w)

    return G

# =============================================================================
# STEP2: Reasoning feature builders (supplier/src/app)
# =============================================================================
def build_supplier_reasoning_features(
    suppliers: List[str],
    reasoner: KGReasoner,
    mp_reasoner: MetaPathPPRReasoner,
    app_thickness: Dict[str, float],
    heat_maps
) -> pd.DataFrame:
    def expected_from_map(post_apps: Dict[str, float], hmap: Dict[str, float]) -> float:
        return float(sum(prob * float(hmap.get(app, 0.0)) for app, prob in post_apps.items())) if post_apps else 0.0

    emb_app = PosteriorEmbedder(SUP_EMB_DIM_APP)
    emb_src = PosteriorEmbedder(SUP_EMB_DIM_SRC)

    rows = []
    for s in suppliers:
        # Original heterogeneous PPR
        post_app = reasoner.infer_supplier_apps(s, alpha=PPR_ALPHA, prior_strength=PPR_PRIOR_STRENGTH)
        post_src = reasoner.infer_supplier_srcs(s, alpha=PPR_ALPHA)

        # Meta-path constrained PPR
        mp_app = mp_reasoner.sup_to_apps(s, alpha=METAPATH_ALPHA, topk=METAPATH_TOPK, prior_strength=METAPATH_PRIOR_STRENGTH_APPHEAT)
        mp_src = mp_reasoner.sup_to_srcs(s, alpha=METAPATH_ALPHA, topk=METAPATH_TOPK)

        # Entropy / top1 / expected heat/thickness
        ent = KGReasoner.entropy(list(post_app.values())) if post_app else 0.0
        top1 = max(post_app.values()) if post_app else 0.0
        exp_heat_total = expected_from_map(post_app, heat_maps["total_log"])
        exp_heat_web = expected_from_map(post_app, heat_maps["web_log"])
        exp_heat_news = expected_from_map(post_app, heat_maps["news_log"])
        exp_heat_weixin = expected_from_map(post_app, heat_maps["weixin_log"])
        exp_thick = float(sum(prob * float(app_thickness.get(app, 0.0)) for app, prob in post_app.items())) if post_app else 0.0

        mp_ent = MetaPathPPRReasoner.entropy(list(mp_app.values())) if mp_app else 0.0
        mp_top1 = max(mp_app.values()) if mp_app else 0.0
        mp_exp_heat_total = expected_from_map(mp_app, heat_maps["total_log"])
        mp_exp_heat_web = expected_from_map(mp_app, heat_maps["web_log"])
        mp_exp_heat_news = expected_from_map(mp_app, heat_maps["news_log"])
        mp_exp_heat_weixin = expected_from_map(mp_app, heat_maps["weixin_log"])
        mp_exp_thick = float(sum(prob * float(app_thickness.get(app, 0.0)) for app, prob in mp_app.items())) if mp_app else 0.0

        v_app = emb_app.embed(post_app)
        v_src = emb_src.embed(post_src)
        v_mp_app = emb_app.embed(mp_app)
        v_mp_src = emb_src.embed(mp_src)

        row = {
            "supplier": s,
            # original
            "ppr_expected_heat_total": float(exp_heat_total),
            "ppr_expected_heat_web": float(exp_heat_web),
            "ppr_expected_heat_news": float(exp_heat_news),
            "ppr_expected_heat_weixin": float(exp_heat_weixin),
            "ppr_app_entropy": float(ent),
            "ppr_app_top1_prob": float(top1),
            "ppr_expected_thickness": float(exp_thick),
            "ppr_top_apps_json": json.dumps(post_app, ensure_ascii=False),
            "ppr_top_srcs_json": json.dumps(post_src, ensure_ascii=False),
            # meta-path
            "mp_ppr_expected_heat_total": float(mp_exp_heat_total),
            "mp_ppr_expected_heat_web": float(mp_exp_heat_web),
            "mp_ppr_expected_heat_news": float(mp_exp_heat_news),
            "mp_ppr_expected_heat_weixin": float(mp_exp_heat_weixin),
            "mp_ppr_app_entropy": float(mp_ent),
            "mp_ppr_app_top1_prob": float(mp_top1),
            "mp_ppr_expected_thickness": float(mp_exp_thick),
            "mp_ppr_top_apps_json": json.dumps(mp_app, ensure_ascii=False),
            "mp_ppr_top_srcs_json": json.dumps(mp_src, ensure_ascii=False),
        }

        for i in range(len(v_app)):
            row[f"supemb_app_{i}"] = float(v_app[i])

        for i in range(len(v_src)):
            row[f"supemb_src_{i}"] = float(v_src[i])

        for i in range(len(v_mp_app)):
            row[f"supemb_mp_app_{i}"] = float(v_mp_app[i])

        for i in range(len(v_mp_src)):
            row[f"supemb_mp_src_{i}"] = float(v_mp_src[i])

        rows.append(row)
    return pd.DataFrame(rows)

def build_src_reasoning_features(
    src_list: List[str],
    mp_reasoner: MetaPathPPRReasoner,
    app_thickness: Dict[str, float],
) -> pd.DataFrame:
    emb = PosteriorEmbedder(SRC_EMB_DIM_APP)
    rows = []
    for src in src_list:
        post_app = mp_reasoner.src_to_apps(src, alpha=METAPATH_ALPHA, topk=METAPATH_TOPK, prior_strength=METAPATH_PRIOR_STRENGTH_APPHEAT)
        post_sup = mp_reasoner.src_to_sups(src, alpha=METAPATH_ALPHA, topk=METAPATH_TOPK)

        ent = MetaPathPPRReasoner.entropy(list(post_app.values())) if post_app else 0.0
        top1 = max(post_app.values()) if post_app else 0.0
        exp_heat = mp_reasoner.expected_app_heat(post_app) if post_app else 0.0
        exp_thick = float(sum(prob * float(app_thickness.get(app, 0.0)) for app, prob in post_app.items())) if post_app else 0.0

        v = emb.embed(post_app)
        row = {
            "src": src,
            "src_mp_expected_heat": float(exp_heat),
            "src_mp_app_entropy": float(ent),
            "src_mp_app_top1_prob": float(top1),
            "src_mp_expected_thickness": float(exp_thick),
            "src_mp_top_apps_json": json.dumps(post_app, ensure_ascii=False),
            "src_mp_top_sups_json": json.dumps(post_sup, ensure_ascii=False),
        }
        for i in range(len(v)):
            row[f"srcemb_mp_app_{i}"] = float(v[i])
        rows.append(row)
    return pd.DataFrame(rows)

def build_app_reasoning_features(
    app_list: List[str],
    mp_reasoner: MetaPathPPRReasoner,
) -> pd.DataFrame:
    emb = PosteriorEmbedder(APP_EMB_DIM_SRC)
    rows = []
    for app in app_list:
        post_src = mp_reasoner.app_to_srcs(app, alpha=METAPATH_ALPHA, topk=METAPATH_TOPK)
        post_sup = mp_reasoner.app_to_sups(app, alpha=METAPATH_ALPHA, topk=METAPATH_TOPK)

        ent = MetaPathPPRReasoner.entropy(list(post_src.values())) if post_src else 0.0
        top1 = max(post_src.values()) if post_src else 0.0

        v = emb.embed(post_src)
        row = {
            "app": app,
            "app_mp_src_entropy": float(ent),
            "app_mp_src_top1_prob": float(top1),
            "app_mp_top_srcs_json": json.dumps(post_src, ensure_ascii=False),
            "app_mp_top_sups_json": json.dumps(post_sup, ensure_ascii=False),
        }
        for i in range(len(v)):
            row[f"appemb_mp_src_{i}"] = float(v[i])
        rows.append(row)
    return pd.DataFrame(rows)

# =============================================================================
# STEP2: Predictive validation for product prices (CV-safe pipeline)
# =============================================================================
def make_step2_price_pipeline(
    include_supplier_fe: bool,
    include_src: bool,
    include_app: bool,
    numeric_cols: List[str],
    min_freq: int,
) -> Pipeline:
    transformers = []

    if include_supplier_fe:
        transformers.append(("supplier_ohe",
                             OneHotEncoder(handle_unknown="ignore", sparse=True, drop="first"),
                             ["supplier"]))
    if include_src:
        transformers.append(("src_mhot",
                             MultiHotEncoder(normalize=True, min_freq=min_freq, prefix="src"),
                             ["src_list"]))
    if include_app:
        transformers.append(("app_mhot",
                             MultiHotEncoder(normalize=True, min_freq=min_freq, prefix="app"),
                             ["app_list"]))

    if numeric_cols:
        transformers.append(("num",
                             Pipeline([("to_df", FunctionPassthrough()), ("scaler", StandardScaler())]),
                             numeric_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    return Pipeline([
        ("pre", pre),
        ("dense", ToDense()),
        ("model", BayesianRidge())
    ])

class FunctionPassthrough(BaseEstimator, TransformerMixin):
    """A no-op transformer to keep numeric columns in ColumnTransformer pipelines."""
    def fit(self, X, y=None): return self
    def transform(self, X): return np.asarray(X, dtype=float)

# =============================================================================
# STEP2: Explain FE by mechanism proxies (entity-level CV)
# =============================================================================
def crossfit_fe_to_mechanisms(
    df_rows: pd.DataFrame,
    entity_type: str,                 # "supplier" / "src" / "app" / "product"
    mech_df: pd.DataFrame,            # entity-level mechanism proxies
    mech_key: str,                    # entity id column name in mech_df (e.g. "supplier"/"src"/"app"/"prod_key")
    mech_feature_cols: List[str],
    min_freq: int,
    k: int,
    out_path_prefix: str,
) -> Dict[str, Any]:
    """
    Cross-fit FE->mechanism with OOF residual targets on train rows.

    Outer fold: entity split.
      Stage1: fit baseline model on outer-train rows; compute residuals on outer-test rows => fe_test
      Stage1(train target): compute OOF predictions inside outer-train rows => residuals => fe_train
      Stage2: fit mechanisms->fe on fe_train entities, predict fe_test entities
    """
    assert entity_type in {"supplier", "src", "app", "product"}

    d = df_rows.copy()
    if "y_log10" not in d.columns:
        raise ValueError("df_rows must contain y_log10.")
    y = d["y_log10"].astype(float).values

    base_cols = ["supplier", "src_list", "app_list"]

    if entity_type == "supplier":
        entities = sorted(d["supplier"].astype(str).unique().tolist())
    elif entity_type == "product":
        # By convention we use a unique product key (e.g. "PROD::{name}||SUP::{supplier}")
        # so we can do entity-split CV without name collisions.
        if mech_key in d.columns:
            d[mech_key] = d[mech_key].astype(str).str.strip()
        elif ("name" in d.columns) and ("supplier" in d.columns):
            d[mech_key] = d.apply(
                lambda r: f"PROD::{str(r['name']).strip()}||SUP::{str(r['supplier']).strip()}",
                axis=1,
            )
        elif "product" in d.columns:
            d[mech_key] = d["product"].astype(str).str.strip()
        else:
            raise ValueError("For entity_type='product', df_rows must contain mech_key or (name,supplier) or product.")
        entities = sorted(d[mech_key].astype(str).unique().tolist())
    elif entity_type == "src":
        entities = sorted(set([x for lst in d["src_list"].tolist() for x in parse_list_cell(lst)]))
    else:
        entities = sorted(set([x for lst in d["app_list"].tolist() for x in parse_list_cell(lst)]))

    if len(entities) < max(10, k * 2):
        meta = {"ok": False, "reason": f"Too few entities for crossfit ({entity_type} n={len(entities)})."}
        with open(out_path_prefix + "_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta

    # mechanism table
    mech = mech_df[[mech_key] + mech_feature_cols].copy()
    mech[mech_key] = mech[mech_key].astype(str).str.strip()
    mech = mech.drop_duplicates(mech_key).set_index(mech_key)

    ent_k = min(k, max(2, len(entities) // 2))
    ent_kf = KFold(n_splits=ent_k, shuffle=True, random_state=RANDOM_STATE)

    def oof_predict(pipe_proto: Pipeline, Xdf: pd.DataFrame, yarr: np.ndarray, n_splits: int) -> np.ndarray:
        """OOF predictions for rows in Xdf."""
        n = len(Xdf)
        if n < 10:
            # fallback: in-sample (rare) to avoid crashing; still no y leakage outside this fold
            p = clone(pipe_proto)
            p.fit(Xdf, yarr)
            return p.predict(Xdf)

        inner_k = min(n_splits, max(2, n // 5))
        kf = KFold(n_splits=inner_k, shuffle=True, random_state=RANDOM_STATE)
        yhat = np.full(n, np.nan, dtype=float)

        for tr, te in kf.split(Xdf):
            p = clone(pipe_proto)
            p.fit(Xdf.iloc[tr], yarr[tr])
            yhat[te] = p.predict(Xdf.iloc[te])

        # fill any nan (edge cases)
        if np.any(~np.isfinite(yhat)):
            p = clone(pipe_proto)
            p.fit(Xdf, yarr)
            yhat[~np.isfinite(yhat)] = p.predict(Xdf.iloc[~np.isfinite(yhat)])

        return yhat

    pred_rows = []
    coef_rows = []

    for fold, (tr_e_idx, te_e_idx) in enumerate(ent_kf.split(entities), start=1):
        train_ents = set([entities[i] for i in tr_e_idx])
        test_ents = set([entities[i] for i in te_e_idx])

        # -------- Stage1: choose baseline spec (exclude target block) --------
        if entity_type == "supplier":
            # baseline: src + app (NO supplier)
            base_pipe_proto = make_fe_pipeline(use_supplier=False, use_src=True, use_app=True, min_freq=min_freq)

            train_row_mask = d["supplier"].astype(str).isin(train_ents)
            test_row_mask = d["supplier"].astype(str).isin(test_ents)

            # fit outer baseline
            base_pipe = clone(base_pipe_proto)
            base_pipe.fit(d.loc[train_row_mask, base_cols], y[train_row_mask])
            yhat_test = base_pipe.predict(d.loc[test_row_mask, base_cols])
            resid_test = y[test_row_mask] - yhat_test

            tmp_test = d.loc[test_row_mask, ["supplier"]].copy()
            tmp_test["resid"] = resid_test
            fe_test = tmp_test.groupby("supplier")["resid"].mean().rename("fe_target").reset_index()

            # OOF residuals on outer-train rows => fe_train
            yhat_train_oof = oof_predict(base_pipe_proto, d.loc[train_row_mask, base_cols], y[train_row_mask], n_splits=k)
            resid_train = y[train_row_mask] - yhat_train_oof
            tmp_train = d.loc[train_row_mask, ["supplier"]].copy()
            tmp_train["resid"] = resid_train
            fe_train = tmp_train.groupby("supplier")["resid"].mean().rename("fe_target").reset_index()

        elif entity_type == "product":
            # baseline: supplier + src + app (NO product FE)
            base_pipe_proto = make_fe_pipeline(use_supplier=True, use_src=True, use_app=True, min_freq=min_freq)

            test_row_mask = d[mech_key].astype(str).isin(test_ents)
            train_row_mask = ~test_row_mask

            base_pipe = clone(base_pipe_proto)
            base_pipe.fit(d.loc[train_row_mask, base_cols], y[train_row_mask])
            yhat_test = base_pipe.predict(d.loc[test_row_mask, base_cols])
            resid_test = y[test_row_mask] - yhat_test

            tmp_test = d.loc[test_row_mask, [mech_key]].copy()
            tmp_test["resid"] = resid_test
            fe_test = tmp_test.groupby(mech_key)["resid"].mean().rename("fe_target").reset_index()

            # fe_train using OOF residuals on outer-train rows
            yhat_train_oof = oof_predict(base_pipe_proto, d.loc[train_row_mask, base_cols], y[train_row_mask], n_splits=k)
            resid_train = y[train_row_mask] - yhat_train_oof

            tmp_train = d.loc[train_row_mask, [mech_key]].copy()
            tmp_train["resid"] = resid_train
            fe_train = tmp_train.groupby(mech_key)["resid"].mean().rename("fe_target").reset_index()

        elif entity_type == "src":
            # baseline: supplier + app (NO src)
            base_pipe_proto = make_fe_pipeline(use_supplier=True, use_src=False, use_app=True, min_freq=min_freq)

            test_row_mask = d["src_list"].apply(lambda x: len(set(parse_list_cell(x)) & test_ents) > 0)
            train_row_mask = ~test_row_mask

            base_pipe = clone(base_pipe_proto)
            base_pipe.fit(d.loc[train_row_mask, base_cols], y[train_row_mask])
            yhat_test = base_pipe.predict(d.loc[test_row_mask, base_cols])
            resid_test = y[test_row_mask] - yhat_test

            # fe_test from residuals allocated to test src labels
            rows = []
            sub = d.loc[test_row_mask, ["src_list"]].copy()
            sub["resid"] = resid_test
            for _, r in sub.iterrows():
                labs = [lab for lab in parse_list_cell(r["src_list"]) if lab in test_ents]
                if not labs:
                    continue
                w = 1.0 / len(labs)
                for lab in labs:
                    rows.append({"src": lab, "resid_w": float(r["resid"]) * w, "w": w})
            fe_test = pd.DataFrame(columns=["src", "fe_target"])
            if rows:
                tmp = pd.DataFrame(rows)
                fe_test = tmp.groupby("src").apply(
                    lambda g: pd.Series({"fe_target": float(g["resid_w"].sum() / (g["w"].sum() + 1e-12))})
                ).reset_index()

            # fe_train using OOF residuals on outer-train rows
            yhat_train_oof = oof_predict(base_pipe_proto, d.loc[train_row_mask, base_cols], y[train_row_mask], n_splits=k)
            resid_train = y[train_row_mask] - yhat_train_oof

            rows = []
            sub = d.loc[train_row_mask, ["src_list"]].copy()
            sub["resid"] = resid_train
            for _, r in sub.iterrows():
                labs = parse_list_cell(r["src_list"])
                if not labs:
                    continue
                w = 1.0 / len(labs)
                for lab in labs:
                    rows.append({"src": lab, "resid_w": float(r["resid"]) * w, "w": w})

            fe_train = pd.DataFrame(columns=["src", "fe_target"])
            if rows:
                tmp = pd.DataFrame(rows)
                fe_train = tmp.groupby("src").apply(
                    lambda g: pd.Series({"fe_target": float(g["resid_w"].sum() / (g["w"].sum() + 1e-12))})
                ).reset_index()

        else:  # app
            # baseline: supplier + src (NO app)
            base_pipe_proto = make_fe_pipeline(use_supplier=True, use_src=True, use_app=False, min_freq=min_freq)

            test_row_mask = d["app_list"].apply(lambda x: len(set(parse_list_cell(x)) & test_ents) > 0)
            train_row_mask = ~test_row_mask

            base_pipe = clone(base_pipe_proto)
            base_pipe.fit(d.loc[train_row_mask, base_cols], y[train_row_mask])
            yhat_test = base_pipe.predict(d.loc[test_row_mask, base_cols])
            resid_test = y[test_row_mask] - yhat_test

            rows = []
            sub = d.loc[test_row_mask, ["app_list"]].copy()
            sub["resid"] = resid_test
            for _, r in sub.iterrows():
                labs = [lab for lab in parse_list_cell(r["app_list"]) if lab in test_ents]
                if not labs:
                    continue
                w = 1.0 / len(labs)
                for lab in labs:
                    rows.append({"app": lab, "resid_w": float(r["resid"]) * w, "w": w})

            fe_test = pd.DataFrame(columns=["app", "fe_target"])
            if rows:
                tmp = pd.DataFrame(rows)
                fe_test = tmp.groupby("app").apply(
                    lambda g: pd.Series({"fe_target": float(g["resid_w"].sum() / (g["w"].sum() + 1e-12))})
                ).reset_index()

            # fe_train using OOF residuals
            yhat_train_oof = oof_predict(base_pipe_proto, d.loc[train_row_mask, base_cols], y[train_row_mask], n_splits=k)
            resid_train = y[train_row_mask] - yhat_train_oof

            rows = []
            sub = d.loc[train_row_mask, ["app_list"]].copy()
            sub["resid"] = resid_train
            for _, r in sub.iterrows():
                labs = parse_list_cell(r["app_list"])
                if not labs:
                    continue
                w = 1.0 / len(labs)
                for lab in labs:
                    rows.append({"app": lab, "resid_w": float(r["resid"]) * w, "w": w})

            fe_train = pd.DataFrame(columns=["app", "fe_target"])
            if rows:
                tmp = pd.DataFrame(rows)
                fe_train = tmp.groupby("app").apply(
                    lambda g: pd.Series({"fe_target": float(g["resid_w"].sum() / (g["w"].sum() + 1e-12))})
                ).reset_index()

        if fe_test.empty or fe_train.empty:
            continue

        # -------- Stage2: mechanisms -> fe_target --------
        fe_train[mech_key] = fe_train[mech_key].astype(str).str.strip()
        fe_test[mech_key] = fe_test[mech_key].astype(str).str.strip()

        train_idx = [e for e in fe_train[mech_key].tolist() if e in mech.index]
        test_idx = [e for e in fe_test[mech_key].tolist() if e in mech.index]

        if len(train_idx) < 8 or len(test_idx) < 2:
            continue

        fe_train_i = fe_train.set_index(mech_key).loc[train_idx]
        fe_test_i = fe_test.set_index(mech_key).loc[test_idx]

        Xtr = mech.loc[train_idx, mech_feature_cols].astype(float).values
        ytr = fe_train_i["fe_target"].astype(float).values
        Xte = mech.loc[test_idx, mech_feature_cols].astype(float).values
        yte = fe_test_i["fe_target"].astype(float).values

        pipe2 = Pipeline([("scaler", StandardScaler()), ("model", BayesianRidge())])
        pipe2.fit(Xtr, ytr)
        ypred = pipe2.predict(Xte)

        pred_rows.append({
            "fold": fold,
            "R2": float(r2_score(yte, ypred)) if len(yte) >= 2 else np.nan,
            "MAE": float(mean_absolute_error(yte, ypred)),
            "RMSE": float(np.sqrt(mean_squared_error(yte, ypred))),
            "n_test_entities": int(len(test_idx)),
        })
        coef_rows.append(pipe2.named_steps["model"].coef_)

    df_cv = pd.DataFrame(pred_rows)
    df_cv.to_csv(out_path_prefix + "_cv.csv", index=False, encoding="utf-8-sig")

    if coef_rows:
        coef_mean = np.mean(np.vstack(coef_rows), axis=0)
        df_coef = pd.DataFrame({
            "feature": mech_feature_cols,
            "coef_mean": coef_mean,
            "abs_coef_mean": np.abs(coef_mean),
        }).sort_values("abs_coef_mean", ascending=False).reset_index(drop=True)
    else:
        df_coef = pd.DataFrame(columns=["feature", "coef_mean", "abs_coef_mean"])
    df_coef.to_csv(out_path_prefix + "_coef.csv", index=False, encoding="utf-8-sig")

    meta = {
        "ok": True,
        "entity_type": entity_type,
        "n_entities": int(len(entities)),
        "n_folds": int(ent_k),
        "cv_R2_mean": float(df_cv["R2"].mean()) if not df_cv.empty else np.nan,
        "cv_MAE_mean": float(df_cv["MAE"].mean()) if not df_cv.empty else np.nan,
        "cv_RMSE_mean": float(df_cv["RMSE"].mean()) if not df_cv.empty else np.nan,
        "top_features": df_coef.head(10).to_dict(orient="records") if not df_coef.empty else [],
    }
    with open(out_path_prefix + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta

# =============================================================================
# STEP3: run step3 product inference
# =============================================================================
def cv_predict_pipeline_price(pipe, df_rows, y_col="y_log10", splits=None):
    """
    Produce out-of-fold predictions (and std if supported) for a given pipeline and splits.
    """
    y = df_rows[y_col].values
    n = len(df_rows)
    oof_mu = np.zeros(n, dtype=float)
    oof_std = np.zeros(n, dtype=float)

    for fold_id, (tr, te) in enumerate(splits):
        X_tr = df_rows.iloc[tr]
        X_te = df_rows.iloc[te]
        y_tr = y[tr]
        pipe.fit(X_tr, y_tr)
        try:
            mu, std = pipe.predict(X_te, return_std=True)
            oof_mu[te] = mu
            oof_std[te] = std
        except TypeError:
            mu = pipe.predict(X_te)
            oof_mu[te] = mu
            oof_std[te] = 0.0

    return oof_mu, oof_std

def linear_feature_importance_from_pipeline(
    pipe: Pipeline,
) -> pd.DataFrame:
    """Extract global linear feature importance (coefficients) from a fitted sklearn Pipeline.

    Works for the Step2/Step3 price pipelines where the last step is a linear model (e.g., BayesianRidge)
    and the first step is a ColumnTransformer named "pre".

    Returns a dataframe sorted by abs(coef) desc with columns:
      - feature: transformed feature name
      - group:   feature group (transformer name before '__')
      - coef:    raw coefficient
      - abs_coef: absolute coefficient
    """
    if not isinstance(pipe, Pipeline):
        raise TypeError(f"pipe must be a sklearn Pipeline, got {type(pipe)}")

    # Try to find the preprocessor and model steps
    pre = pipe.named_steps.get("pre", None)
    model = pipe.named_steps.get("model", None)

    if pre is None:
        # fallback: assume first step is preprocessor
        pre = pipe.steps[0][1]
    if model is None:
        # fallback: assume last step is model
        model = pipe.steps[-1][1]

    if not hasattr(model, "coef_"):
        raise ValueError(f"Model {type(model)} has no coef_; cannot extract linear importances")

    # Get feature names from preprocessor (sklearn >= 1.0)
    if not hasattr(pre, "get_feature_names_out"):
        raise ValueError(
            f"Preprocessor {type(pre)} has no get_feature_names_out(); cannot map coefficients to names"
        )

    feat_names = list(pre.get_feature_names_out())
    coef = np.asarray(model.coef_).reshape(-1)

    if len(feat_names) != len(coef):
        raise ValueError(
            f"Feature name length ({len(feat_names)}) != coef length ({len(coef)}). "
            "Did the pipeline change feature dimensionality after 'pre'?"
        )

    def _group(n: str) -> str:
        # sklearn feature names look like: "<transformer>__<original_feature_or_token>"
        return n.split("__", 1)[0] if "__" in n else "(unknown)"

    df_imp = pd.DataFrame(
        {
            "feature": feat_names,
            "group": [_group(n) for n in feat_names],
            "coef": coef.astype(float),
        }
    )
    df_imp["abs_coef"] = df_imp["coef"].abs()
    df_imp = df_imp.sort_values(["abs_coef", "feature"], ascending=[False, True]).reset_index(drop=True)
    return df_imp

def step3_write_competitor_outputs_from_desc(
    df_rows: pd.DataFrame,
    heat_maps: Dict[str, Dict[str, float]],
    out_dir: str,
) -> pd.DataFrame:
    """
    Same logic as STEP0 competitor outputs, but write STEP3-prefixed files for clarity.
    """
    tmp_dir = out_dir
    df_comp = step0_write_competitor_outputs_from_desc(
        df_rows=df_rows,
        heat_maps=heat_maps,
        out_dir=tmp_dir,
        k=SIM_KNN_K,
        min_cos=SIM_MIN_COS,
        topk_labels=METAPATH_TOPK,
    )

    # Rename files for STEP3 (copy-by-write; simplest for beginners)
    # If you want strict no-duplication, remove this and accept STEP0_* naming.
    try:
        for src, dst in [
            ("STEP0_product_similarity_graph_edges.csv", "STEP3_product_similarity_graph_edges.csv"),
            ("STEP0_product_competitor_summary.csv", "STEP3_product_competitor_summary.csv"),
            ("STEP0_completion_product_missing_apps.csv", "STEP3_completion_product_missing_apps.csv"),
            ("STEP0_completion_product_missing_srcs.csv", "STEP3_completion_product_missing_srcs.csv"),
        ]:
            p1 = os.path.join(tmp_dir, src)
            p2 = os.path.join(tmp_dir, dst)
            if os.path.exists(p1):
                pd.read_csv(p1).to_csv(p2, index=False, encoding="utf-8-sig")
    except Exception:
        pass

    return df_comp

def build_step3_product_mechanism_proxies(
    df_rows: pd.DataFrame,
    df_edges: pd.DataFrame,
    heat_maps: Dict[str, Dict[str, float]],
    app_thickness_map: Optional[Dict[str, float]] = None,
    k_topics: int = 10,
    text_emb_dim: int = TEXT_EMB_DIM,
    text_min_df: int = TEXT_MIN_DF,
    sim_k: int = SIM_KNN_K,
    sim_min_cos: float = SIM_MIN_COS,
    ppr_alpha: float = PPR_ALPHA,
    ppr_prior_strength: float = PPR_PRIOR_STRENGTH,
    ppr_topk_apps: int = PPR_TOPK_APPS,
    ppr_topk_srcs: int = PPR_TOPK_SRCS,
    heat_ablation: bool = HEAT_ABLATION_ENABLE_DEFAULT,
    ppr_heat_teleport_mass: float = PPR_HEAT_TELEPORT_MASS,
    metapath_heat_teleport_mass: float = METAPATH_HEAT_TELEPORT_MASS,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build product-level mechanism proxies for STEP3 crossfit.

    Output mech_df columns:
      - prod_key (unique product id: "PROD::{name}||SUP::{supplier}")
      - numeric proxy columns (text embedding, similarity-graph metrics, heat summaries, KG-PPR summaries)
    """
    d = df_rows.copy()
    if "prod_key" not in d.columns:
        if ("name" in d.columns) and ("supplier" in d.columns):
            d["prod_key"] = d.apply(
                lambda r: f"PROD::{str(r['name']).strip()}||SUP::{str(r['supplier']).strip()}",
                axis=1,
            )
        elif "product" in d.columns:
            d["prod_key"] = d["product"].astype(str).str.strip()
        else:
            raise ValueError("df_rows must contain (name,supplier) or product to build product proxies.")

    # ---- 1) Text embedding (product representation) ----
    df_u = d[["name", "supplier", "desc", "prod_key"]].drop_duplicates("prod_key").copy()
    df_prod_emb, _ = build_product_text_embedding(df_u, text_col="desc", dim=text_emb_dim, min_df=text_min_df)
    df_prod_emb["prod_key"] = df_u["prod_key"].values

    mech = df_prod_emb[["prod_key"] + [c for c in df_prod_emb.columns if c.startswith("textemb_")]].copy()

    # ---- 2) Similarity-graph features ----
    try:
        Gsim = build_product_similarity_graph(df_prod_emb, k=sim_k, min_cos=sim_min_cos)
        df_graph = compute_product_graph_features(Gsim)
        mech = mech.merge(df_graph, on="prod_key", how="left")
    except Exception:
        pass

    # ---- 3) Heat summaries (mean app heat for product's app list) ----
    try:
        df_products = df_u[["prod_key", "app_list"]].copy()
        df_heat = compute_product_level_heat_components(df_products, heat_maps=heat_maps)
        df_heat["prod_key"] = df_products["prod_key"].values
        mech = mech.merge(df_heat, on="prod_key", how="left")
    except Exception:
        pass

    # ---- 4) Topic dummies (coarse semantic proxy) ----
    try:
        prod_topics, _, _, _ = build_topic_exposure(d, df_prod_emb, k_topics=k_topics)
        prod_topics["prod_key"] = prod_topics.apply(
            lambda r: f"PROD::{str(r['name']).strip()}||SUP::{str(r['supplier']).strip()}",
            axis=1,
        )
        topic_oh = pd.get_dummies(prod_topics["topic_id"].astype(int), prefix="prod_topic")
        df_topic = pd.concat([prod_topics[["prod_key"]], topic_oh], axis=1).groupby("prod_key").sum().reset_index()
        mech = mech.merge(df_topic, on="prod_key", how="left")
    except Exception:
        pass

    # ---- 5) KG-based PPR summaries from product node ----
    try:
        de = df_edges.copy()
        if "app_list" in de.columns:
            de["app_list"] = de["app_list"].apply(parse_list_cell)
        if "src_list" in de.columns:
            de["src_list"] = de["src_list"].apply(parse_list_cell)

        kg = KGReasoner(
            df_edges=de,
            app_heat_total=heat_maps.get("total_log", {}),
            topk_apps=ppr_topk_apps,
            topk_srcs=ppr_topk_srcs,
            use_supplier_in_product_node=True,
        )

        rows = []
        for _, r in df_u[["name", "supplier", "prod_key"]].iterrows():
            post_app = kg.infer_product_apps(
                product=str(r["name"]),
                supplier=str(r["supplier"]),
                alpha=ppr_alpha,
                prior_strength=ppr_prior_strength,

            )
            # --- Heat ablation (STRICT): structural vs teleport-only heat bias ---
            post_app_struct = {}
            post_app_tele = {}
            if heat_ablation:
                try:
                    post_app_struct = kg.infer_product_apps_structural(
                        product=str(r["name"]),
                        supplier=str(r["supplier"]),
                        alpha=ppr_alpha,
                    )
                except Exception:
                    post_app_struct = {}
                try:
                    post_app_tele = kg.infer_product_apps_heat_teleport(
                        product=str(r["name"]),
                        supplier=str(r["supplier"]),
                        alpha=ppr_alpha,
                        prior_strength=ppr_prior_strength,
                        teleport_mass=ppr_heat_teleport_mass,
                    )
                except Exception:
                    post_app_tele = {}
            post_src = kg.infer_product_srcs(
                product=str(r["name"]),
                supplier=str(r["supplier"]),
                alpha=ppr_alpha,
            )
            exp_heat = kg.expected_heat(post_app) if post_app else 0.0
            exp_heat_struct = kg.expected_heat(post_app_struct) if post_app_struct else 0.0
            exp_heat_tele = kg.expected_heat(post_app_tele) if post_app_tele else 0.0
            ent_app = kg.entropy(list(post_app.values())) if post_app else 0.0
            top1 = max(post_app.values()) if post_app else 0.0
            if app_thickness_map and post_app:
                exp_thick = float(sum(float(p) * float(app_thickness_map.get(a, 0.0)) for a, p in post_app.items()))
            else:
                exp_thick = 0.0

            rows.append({
                "prod_key": str(r["prod_key"]),
                "prod_ppr_expected_heat_total": float(exp_heat),
                "prod_ppr_expected_heat_total_struct": float(exp_heat_struct) if heat_ablation else float(exp_heat),
                "prod_ppr_expected_heat_total_tele": float(exp_heat_tele) if heat_ablation else float(exp_heat),
                "prod_ppr_expheat_delta_tele_minus_struct": float(exp_heat_tele - exp_heat_struct) if heat_ablation else 0.0,
                "prod_ppr_app_entropy": float(ent_app),
                "prod_ppr_app_top1_prob": float(top1),
                "prod_ppr_expected_thickness": float(exp_thick),
                "prod_ppr_n_top_apps": int(len(post_app)),
                "prod_ppr_n_top_srcs": int(len(post_src)),
            })

        df_ppr = pd.DataFrame(rows)
        mech = mech.merge(df_ppr, on="prod_key", how="left")
    except Exception:
        pass

        # ---- MPPR (Meta-path constrained PPR) summaries for PRODUCT ----
        # Goal: produce prod_mp_ppr_* features, analogous to STEP2 mp_ppr_* but at product-level.
        # We build meta-path bipartite counts from df_edges (no extra Neo4j query).
    try:
        from collections import defaultdict

        app_heat_total = heat_maps.get("total_log", {}) if isinstance(heat_maps, dict) else {}

        # -------- Build df_sup_app / df_sup_src / df_src_app from df_edges --------
        de = df_edges.copy()
        # df_edges expected columns: supplier, product, app_list, src_list (from Neo4j fetch_supplier_product_app_src_edges)
        if "supplier" in de.columns:
            de["supplier"] = de["supplier"].astype(str).str.strip()
        if "product" in de.columns:
            de["product"] = de["product"].astype(str).str.strip()

        # Ensure lists
        if "app_list" in de.columns:
            de["app_list"] = de["app_list"].apply(parse_list_cell)
        else:
            de["app_list"] = [[] for _ in range(len(de))]

        if "src_list" in de.columns:
            de["src_list"] = de["src_list"].apply(parse_list_cell)
        else:
            de["src_list"] = [[] for _ in range(len(de))]

        sup_app_cnt = defaultdict(float)
        sup_src_cnt = defaultdict(float)
        src_app_cnt = defaultdict(float)

        # Count DISTINCT product occurrences per pair (supplier, app/src) and (src, app)
        for _, r in de.iterrows():
            sup = str(r.get("supplier", "")).strip()
            prod = str(r.get("product", "")).strip()
            if not sup or not prod:
                continue

            apps = [str(x).strip() for x in (r.get("app_list") or []) if str(x).strip()]
            srcs = [str(x).strip() for x in (r.get("src_list") or []) if str(x).strip()]

            # De-duplicate within product
            apps_u = list(dict.fromkeys(apps))
            srcs_u = list(dict.fromkeys(srcs))

            for a in apps_u:
                sup_app_cnt[(sup, a)] += 1.0
            for s in srcs_u:
                sup_src_cnt[(sup, s)] += 1.0
            for s in srcs_u:
                for a in apps_u:
                    src_app_cnt[(s, a)] += 1.0

        df_sup_app = pd.DataFrame(
            [{"supplier": k[0], "app": k[1], "prod_cnt": float(v)} for k, v in sup_app_cnt.items()]
        )
        df_sup_src = pd.DataFrame(
            [{"supplier": k[0], "src": k[1], "prod_cnt": float(v)} for k, v in sup_src_cnt.items()]
        )
        df_src_app = pd.DataFrame(
            [{"src": k[0], "app": k[1], "prod_cnt": float(v)} for k, v in src_app_cnt.items()]
        )

        # Instantiate meta-path reasoner (same as STEP2 pattern)
        mp_reasoner_prod = MetaPathPPRReasoner(
            df_sup_app=df_sup_app,
            df_sup_src=df_sup_src,
            df_src_app=df_src_app,
            app_heat_total=app_heat_total,
        )

        def _avg_posteriors(posts):
            """Average-fuse multiple posterior dicts into one (then renormalize)."""
            acc = defaultdict(float)
            n = 0
            for post in posts:
                if not post:
                    continue
                n += 1
                for k, v in post.items():
                    acc[str(k)] += float(v)
            if n <= 0:
                return {}
            # average then normalize
            for k in list(acc.keys()):
                acc[k] /= float(n)
            s = float(sum(acc.values()))
            if s <= 1e-12:
                return {}
            out = {k: float(v / s) for k, v in acc.items()}
            out = dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))
            return out

        def _topk(post, k):
            if not post:
                return {}
            items = list(sorted(post.items(), key=lambda kv: kv[1], reverse=True))[: int(k)]
            s = float(sum(v for _, v in items))
            if s <= 1e-12:
                return {}
            return {k0: float(v0 / s) for k0, v0 in items}

        # Use product-level observed src_list/app_list from df_rows (not df_edges)
        df_u2 = d[["name", "supplier", "src_list", "app_list", "prod_key"]].drop_duplicates("prod_key").copy()
        df_u2["src_list"] = df_u2["src_list"].apply(parse_list_cell)
        df_u2["app_list"] = df_u2["app_list"].apply(parse_list_cell)

        mp_rows = []
        for _, r in df_u2.iterrows():
            sup = str(r.get("supplier", "")).strip()
            srcs = [str(x).strip() for x in (r.get("src_list") or []) if str(x).strip()]
            apps = [str(x).strip() for x in (r.get("app_list") or []) if str(x).strip()]

            # Build product MPPR posterior over apps:
            #   - posts_app: legacy (post-hoc heat prior inside mp_reasoner_prod.*_to_apps)
            #   - posts_app_struct: STRICT structural (no heat anywhere)
            #   - posts_app_tele: STRICT heat-biased (heat ONLY in teleport/personalization)
            posts_app = []
            posts_app_struct = []
            posts_app_tele = []
            for s in srcs:
                posts_app.append(
                    mp_reasoner_prod.src_to_apps(
                        s,
                        alpha=METAPATH_ALPHA,
                        topk=METAPATH_TOPK,
                        prior_strength=METAPATH_PRIOR_STRENGTH_APPHEAT,
                    )
                )
                if heat_ablation:
                    try:
                        posts_app_struct.append(
                            mp_reasoner_prod.src_to_apps_structural(
                                s,
                                alpha=METAPATH_ALPHA,
                                topk=METAPATH_TOPK,
                            )
                        )
                    except Exception:
                        pass
                    try:
                        posts_app_tele.append(
                            mp_reasoner_prod.src_to_apps_heat_teleport(
                                s,
                                alpha=METAPATH_ALPHA,
                                topk=METAPATH_TOPK,
                                prior_strength=METAPATH_PRIOR_STRENGTH_APPHEAT,
                                teleport_mass=metapath_heat_teleport_mass,
                            )
                        )
                    except Exception:
                        pass

            # Optional: include supplier prior (comment out if you want "pure product src-driven" MPPR)
            if sup:
                posts_app.append(
                    mp_reasoner_prod.sup_to_apps(
                        sup,
                        alpha=METAPATH_ALPHA,
                        topk=METAPATH_TOPK,
                        prior_strength=METAPATH_PRIOR_STRENGTH_APPHEAT,
                    )
                )
                if heat_ablation:
                    try:
                        posts_app_struct.append(
                            mp_reasoner_prod.sup_to_apps_structural(
                                sup,
                                alpha=METAPATH_ALPHA,
                                topk=METAPATH_TOPK,
                            )
                        )
                    except Exception:
                        pass
                    try:
                        posts_app_tele.append(
                            mp_reasoner_prod.sup_to_apps_heat_teleport(
                                sup,
                                alpha=METAPATH_ALPHA,
                                topk=METAPATH_TOPK,
                                prior_strength=METAPATH_PRIOR_STRENGTH_APPHEAT,
                                teleport_mass=metapath_heat_teleport_mass,
                            )
                        )
                    except Exception:
                        pass

            prod_mp_app = _topk(_avg_posteriors(posts_app), METAPATH_TOPK)
            prod_mp_app_struct = _topk(_avg_posteriors(posts_app_struct), METAPATH_TOPK) if heat_ablation else {}
            prod_mp_app_tele = _topk(_avg_posteriors(posts_app_tele), METAPATH_TOPK) if heat_ablation else {}

            # Also build product MPPR posterior over srcs (from apps)
            posts_src = []
            for a in apps:
                posts_src.append(
                    mp_reasoner_prod.app_to_srcs(
                        a,
                        alpha=METAPATH_ALPHA,
                        topk=METAPATH_TOPK,
                    )
                )
            if sup:
                posts_src.append(
                    mp_reasoner_prod.sup_to_srcs(
                        sup,
                        alpha=METAPATH_ALPHA,
                        topk=METAPATH_TOPK,
                    )
                )
            prod_mp_src = _topk(_avg_posteriors(posts_src), METAPATH_TOPK)

            exp_heat = mp_reasoner_prod.expected_app_heat(prod_mp_app) if prod_mp_app else 0.0
            exp_heat_struct = mp_reasoner_prod.expected_app_heat(prod_mp_app_struct) if prod_mp_app_struct else 0.0
            exp_heat_tele = mp_reasoner_prod.expected_app_heat(prod_mp_app_tele) if prod_mp_app_tele else 0.0
            ent_app = MetaPathPPRReasoner.entropy(list(prod_mp_app.values())) if prod_mp_app else 0.0
            top1 = max(prod_mp_app.values()) if prod_mp_app else 0.0

            if app_thickness_map and prod_mp_app:
                exp_thick = float(
                    sum(float(p) * float(app_thickness_map.get(a, 0.0)) for a, p in prod_mp_app.items())
                )
            else:
                exp_thick = 0.0

            mp_rows.append({
                "prod_key": str(r["prod_key"]),
                "prod_mp_ppr_expected_heat_total": float(exp_heat),
                "prod_mp_ppr_expected_heat_total_struct": float(exp_heat_struct) if heat_ablation else float(exp_heat),
                "prod_mp_ppr_expected_heat_total_tele": float(exp_heat_tele) if heat_ablation else float(exp_heat),
                "prod_mp_ppr_expheat_delta_tele_minus_struct": float(exp_heat_tele - exp_heat_struct) if heat_ablation else 0.0,
                "prod_mp_ppr_app_entropy": float(ent_app),
                "prod_mp_ppr_app_top1_prob": float(top1),
                "prod_mp_ppr_expected_thickness": float(exp_thick),
                "prod_mp_ppr_n_top_apps": int(len(prod_mp_app)),
                "prod_mp_ppr_n_top_srcs": int(len(prod_mp_src)),
                # debug / inspection (non-numeric; won’t enter feature_cols)
                "prod_mp_ppr_top_apps_json": json.dumps(prod_mp_app, ensure_ascii=False),
                "prod_mp_ppr_top_srcs_json": json.dumps(prod_mp_src, ensure_ascii=False),
            })

        df_mp = pd.DataFrame(mp_rows)
        mech = mech.merge(df_mp, on="prod_key", how="left")

    except Exception:
        # keep pipeline robust; MPPR is optional
        pass

    # cleanup
    mech = mech.fillna(0.0)
    feature_cols = [
        c for c in mech.columns
        if (c != "prod_key") and (pd.api.types.is_numeric_dtype(mech[c]))
    ]
    return mech, feature_cols

def write_heat_ablation_expheat_price_gradient(
    df_rows: pd.DataFrame,
    prod_mech_df: pd.DataFrame,
    out_dir: str,
    folds_k: Optional[List[Dict[str, List[int]]]] = None,
    folds_g: Optional[List[Dict[str, List[int]]]] = None,
    price_col: str = "price",
    supplier_col: str = "supplier",
    prod_key_col: str = "prod_key",
) -> None:
    """
    Compare STRICT structural vs STRICT heat-teleport (personalization-only) ExpHeat signals
    against price gradients (log10 price). Outputs:
      - HEAT_ABLATION__expheat_price_gradient_summary.csv
      - HEAT_ABLATION__expheat_price_gradient_folds.csv
      - HEAT_ABLATION__expheat_price_gradient_pairs.csv
    """
    ensure_dir(out_dir)

    base = df_rows.copy()
    if prod_key_col not in base.columns:
        return
    if price_col not in base.columns:
        return
    base = base[[prod_key_col, price_col, supplier_col]].copy() if supplier_col in base.columns else base[[prod_key_col, price_col]].copy()
    base["_row_idx"] = np.arange(len(base), dtype=int)

    # price -> log10(price)
    base[price_col] = pd.to_numeric(base[price_col], errors="coerce")
    base = base.dropna(subset=[price_col])
    base["y_log10"] = np.log10(np.maximum(base[price_col].values.astype(float), 1e-9))

    d = base.merge(prod_mech_df, on=prod_key_col, how="left", suffixes=("", "_mech"))

    def _pearson(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return np.nan
        x = x[m]; y = y[m]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    def _spearman(x, y):
        x = pd.Series(x).rank(method="average")
        y = pd.Series(y).rank(method="average")
        return _pearson(x.values, y.values)

    def _ols_slope(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return np.nan
        x = x[m]; y = y[m]
        vx = float(np.var(x))
        if vx < 1e-12:
            return np.nan
        cov = float(np.mean((x - x.mean()) * (y - y.mean())))
        return float(cov / vx)

    def _demean_by_group(x, y, g):
        df = pd.DataFrame({"x": x, "y": y, "g": g})
        df = df.dropna()
        if df.empty:
            return np.array([]), np.array([])
        df["x_dm"] = df["x"] - df.groupby("g")["x"].transform("mean")
        df["y_dm"] = df["y"] - df.groupby("g")["y"].transform("mean")
        return df["x_dm"].values, df["y_dm"].values

    methods = [
        ("ppr", "prod_ppr_expected_heat_total_struct", "prod_ppr_expected_heat_total_tele"),
        ("mp_ppr", "prod_mp_ppr_expected_heat_total_struct", "prod_mp_ppr_expected_heat_total_tele"),
    ]

    summary_rows = []
    pair_rows = []

    for method, col_struct, col_tele in methods:
        if col_struct not in d.columns or col_tele not in d.columns:
            continue
        x_s = d[col_struct].values
        x_t = d[col_tele].values
        y = d["y_log10"].values

        # pair agreement
        pair_rows.append({
            "method": method,
            "pearson_struct_vs_tele": _pearson(x_s, x_t),
            "spearman_struct_vs_tele": _spearman(x_s, x_t),
        })

        for mode, x in [("structural", x_s), ("teleport", x_t)]:
            pr = _pearson(x, y)
            sr = _spearman(x, y)
            slope = _ols_slope(x, y)

            slope_within = np.nan
            if supplier_col in d.columns:
                xdm, ydm = _demean_by_group(pd.Series(x), pd.Series(y), d[supplier_col].astype(str))
                if len(xdm) >= 3:
                    slope_within = _ols_slope(xdm, ydm)

            summary_rows.append({
                "method": method,
                "mode": mode,
                "n": int(np.isfinite(x).sum()),
                "pearson_x_y": pr,
                "spearman_x_y": sr,
                "slope_ols": slope,
                "slope_within_supplier": slope_within,
            })

    df_pairs = pd.DataFrame(pair_rows)
    df_pairs.to_csv(os.path.join(out_dir, "step3_HEAT_ABLATION__expheat_price_gradient_pairs.csv"), index=False)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, "step3_HEAT_ABLATION__expheat_price_gradient_summary.csv"), index=False)

    # Fold stability
    fold_rows = []
    valid_row_idx = set(d["_row_idx"].dropna().astype(int).tolist())

    def _fold_iter(folds):
        if not folds:
            return []
        out = []
        for k, f in enumerate(folds):
            tr = np.array(f["train_idx"], dtype=int)
            te = np.array(f["test_idx"], dtype=int)
            out.append((k, tr, te))
        return out

    for fold_type, folds in [("kfold", folds_k), ("groupkfold_supplier", folds_g)]:
        for method, col_struct, col_tele in methods:
            if col_struct not in d.columns or col_tele not in d.columns:
                continue
            for mode, col in [("structural", col_struct), ("teleport", col_tele)]:
                for k, tr, te in _fold_iter(folds):
                    tr = np.array([i for i in tr.tolist() if i in valid_row_idx], dtype=int)
                    te = np.array([i for i in te.tolist() if i in valid_row_idx], dtype=int)
                    if tr.size < 3 or te.size < 3:
                        continue
                    d_tr = d[d["_row_idx"].isin(tr)]
                    d_te = d[d["_row_idx"].isin(te)]
                    x_tr = d_tr[col].values
                    y_tr = d_tr["y_log10"].values
                    x_te = d_te[col].values
                    y_te = d_te["y_log10"].values

                    slope_tr = _ols_slope(x_tr, y_tr)
                    pr_te = _pearson(x_te, y_te)
                    fold_rows.append({
                        "fold_type": fold_type,
                        "fold_id": int(k),
                        "method": method,
                        "mode": mode,
                        "n_train": int(np.isfinite(x_tr).sum()),
                        "n_test": int(np.isfinite(x_te).sum()),
                        "train_slope": slope_tr,
                        "test_pearson": pr_te,
                    })

    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(os.path.join(out_dir, "step3_HEAT_ABLATION__expheat_price_gradient_folds.csv"), index=False)

def run_step3_product_inference(
    df_rows,
    out_dir,
    n_splits=5,
    random_state=42,
    min_label_freq=5,
    supplier_col="supplier",
    app_list_col="app_list",
    src_list_col="src_list",
    text_col="desc",
    y_col="y_log10",
    splits_kfold=None,
    scenarios=None,
    numeric_cols_for_price=[],
):
    """
    STEP3: Use STEP2 pipelines to produce product-level inference outputs.
    Default scenarios: B + C.
    Outputs:
      - STEP3_oof_predictions__{scenario}__kfold.csv
      - STEP3_oof_predictions__{scenario}__groupkfold_supplier.csv
      - STEP3_inference_summary.json
    """
    ensure_dir(out_dir)

    if scenarios is None:
        scenarios = ["B_mechanisms_with_supplier"]

    if splits_kfold is None:
        splits_kfold = list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(df_rows))

    summary = {"scenarios": {}, "n_rows": int(len(df_rows))}

    # build pipelines same as STEP2
    for sc in scenarios:
        if sc == "B_mechanisms_with_supplier":
            pipe = make_step2_price_pipeline(
                include_supplier_fe=True,
                include_src=True,
                include_app=True,
                numeric_cols= numeric_cols_for_price,
                min_freq=min_label_freq,
            )
        else:
            # fallback: use STEP2 defaults
            pipe = make_step2_price_pipeline(
                include_supplier_fe=False,
                include_src=True,
                include_app=True,
                numeric_cols=[],
                min_freq=min_label_freq,
            )

        # KFold OOF
        df_out = df_rows.copy()
        df_out.drop(columns=["ppr_top_apps_json", "ppr_top_srcs_json", "mp_ppr_top_apps_json", "mp_ppr_top_srcs_json", ], errors="ignore", inplace=True)
        mu_k, std_k = cv_predict_pipeline_price(pipe, df_out, y_col=y_col, splits=splits_kfold)
        df_out["oof_mu_log10"] = mu_k
        df_out["oof_std_log10"] = std_k
        df_out["oof_mu_price"] = (10 ** mu_k)
        df_out["y_true_price"] = (10 ** df_rows[y_col].values)
        df_out.to_csv(os.path.join(out_dir, f"STEP3_oof_predictions__{sc}__kfold.csv"), index=False, encoding="utf-8-sig")

        summary["scenarios"][sc] = {
            "kfold_rmse": float(rmse_from_log10(df_rows[y_col].values, mu_k)),
            "kfold_r2": float(r2_score(df_rows[y_col].values, mu_k)),
        }

        # Also export a simple, global explanation (linear coefficients) for each scenario.
        # This is the Product-side "reasoning" counterpart to STEP2's entity-level FE explanation.
        if out_dir:
            try:
                pipe_full = make_step2_price_pipeline(
                    include_supplier_fe=True,
                    include_src=True,
                    include_app=True,
                    numeric_cols=numeric_cols_for_price,
                    min_freq=min_label_freq,
                )

                pipe_full.fit(df_out, df_out[y_col].values)

                df_imp = linear_feature_importance_from_pipeline(pipe_full)
                imp_path = os.path.join(out_dir, f"STEP3_feature_importance__{sc}.csv")
                df_imp.to_csv(imp_path, index=False, encoding="utf-8")

                topk = 30
                summary["scenarios"][sc]["top_linear_features"] = [
                    {
                        "feature": r["feature"],
                        "group": r["group"],
                        "coef": float(r["coef"]),
                        "abs_coef": float(r["abs_coef"]),
                    }
                    for r in df_imp.head(topk).to_dict(orient="records")
                ]
            except Exception as e:
                summary["scenarios"][sc]["feature_importance_error"] = repr(e)

    with open(os.path.join(out_dir, "STEP3_inference_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # return a short text for conclusions
    lines = ["STEP3 Product inference summary:"]
    for sc, met in summary["scenarios"].items():
        lines.append(f"  - {sc}: KFold RMSE={met['kfold_rmse']:.4f}, R2={met['kfold_r2']:.4f} ")
    return "\n".join(lines)

# =============================================================================
# STEP4: Mechanism-chain analyzers (mediation, DML)
# =============================================================================
class MechanismChainAnalyzer:
    def __init__(self, random_state: int = 42):
        self.random_state = int(random_state)

    @staticmethod
    def _ols_coef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(X, y, rcond=None)[0]

    def mediation_bootstrap(self, df: pd.DataFrame, T: str, M: str, Y: str, controls: List[str], B: int = 300) -> Dict[str, Any]:
        rng = np.random.RandomState(self.random_state)
        cols = [T, M, Y] + (controls or [])
        d0 = df[cols].dropna().copy()
        if len(d0) < 30:
            return {"ok": False, "reason": "Too few rows for mediation."}

        t = d0[T].astype(float).values
        m = d0[M].astype(float).values
        y = d0[Y].astype(float).values
        w = d0[controls].astype(float).values if controls else np.zeros((len(d0), 0))

        def once(idx):
            t1, m1, y1 = t[idx], m[idx], y[idx]
            w1 = w[idx] if (w.shape[1] > 0) else np.zeros((len(idx), 0))
            X_M = np.column_stack([np.ones(len(idx)), t1, w1])
            a = self._ols_coef(X_M, m1)[1]
            X_Y = np.column_stack([np.ones(len(idx)), m1, t1, w1])
            coefY = self._ols_coef(X_Y, y1)
            b = coefY[1]
            direct = coefY[2]
            return a * b, direct

        idx_all = np.arange(len(d0))
        ind0, dir0 = once(idx_all)

        inds, dirs = [], []
        for _ in range(B):
            idx = rng.choice(idx_all, size=len(idx_all), replace=True)
            ind, dire = once(idx)
            inds.append(ind)
            dirs.append(dire)

        return {
            "ok": True,
            "indirect_effect": float(ind0),
            "direct_effect": float(dir0),
            "indirect_CI95": (float(np.percentile(inds, 2.5)), float(np.percentile(inds, 97.5))),
            "direct_CI95": (float(np.percentile(dirs, 2.5)), float(np.percentile(dirs, 97.5))),
            "B": int(B),
        }

    def dml_continuous_treatment(self, df: pd.DataFrame, T: str, Y: str, W: List[str], n_splits: int = 5) -> Dict[str, Any]:
        cols = [T, Y] + (W or [])
        d0 = df[cols].dropna().copy()
        if len(d0) < 50:
            return {"ok": False, "reason": "Too few rows for DML."}

        t = d0[T].astype(float).values
        y = d0[Y].astype(float).values
        w = d0[W].astype(float).values if W else np.zeros((len(d0), 0))

        kf = KFold(n_splits=min(n_splits, max(2, len(d0)//10)), shuffle=True, random_state=self.random_state)

        y_res_all, t_res_all = [], []
        for tr, te in kf.split(w):
            wtr, wte = w[tr], w[te]
            ytr, yte = y[tr], y[te]
            ttr, tte = t[tr], t[te]

            gY = Ridge(alpha=1.0, random_state=self.random_state)
            gT = Ridge(alpha=1.0, random_state=self.random_state)
            gY.fit(wtr, ytr)
            gT.fit(wtr, ttr)

            y_res_all.append(yte - gY.predict(wte))
            t_res_all.append(tte - gT.predict(wte))

        y_res = np.concatenate(y_res_all)
        t_res = np.concatenate(t_res_all)
        X = np.column_stack([np.ones(len(t_res)), t_res])
        beta = np.linalg.lstsq(X, y_res, rcond=None)[0][1]
        return {"ok": True, "dml_slope": float(beta)}

# =========================
# STEP4: summarize STEP4 causal-chain outputs
# Produces:
#   - STEP4_mechanism_chain_summary.csv
#   - STEP4_mechanism_chain_summary.md
# =========================
def _safe_json_load(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _ci_excludes_zero(ci):
    try:
        lo, hi = ci
        return (lo is not None) and (hi is not None) and (float(lo) > 0 or float(hi) < 0)
    except Exception:
        return False

def summarize_mechanism_chain_outputs(out_dir: str) -> pd.DataFrame:
    """
    Read STEP4_*_{mediation,dml}.json for supplier/product (if they exist),
    and write a compact summary table + a markdown report.
    """
    levels = ["supplier", "product"]
    rows = []

    for lvl in levels:
        med_path = os.path.join(out_dir, f"STEP4_{lvl}_mediation.json")
        dml_path = os.path.join(out_dir, f"STEP4_{lvl}_dml.json")

        med = _safe_json_load(med_path)
        dml = _safe_json_load(dml_path)

        # ---- Mediation ----
        if isinstance(med, dict):
            ind = med.get("indirect_effect", None)
            ind_ci = med.get("indirect_CI95", [None, None])
            dir_ = med.get("direct_effect", None)
            dir_ci = med.get("direct_CI95", [None, None])

            rows.append({
                "level": lvl,
                "method": "mediation",
                "term": "indirect_effect (a*b)",
                "estimate": ind,
                "ci95_low": (ind_ci or [None, None])[0],
                "ci95_high": (ind_ci or [None, None])[1],
                "ci_excludes_0": _ci_excludes_zero(ind_ci),
                "notes": "Mechanism-consistent evidence only (not causal identification).",
            })
            rows.append({
                "level": lvl,
                "method": "mediation",
                "term": "direct_effect (c')",
                "estimate": dir_,
                "ci95_low": (dir_ci or [None, None])[0],
                "ci95_high": (dir_ci or [None, None])[1],
                "ci_excludes_0": _ci_excludes_zero(dir_ci),
                "notes": "Direct association after controlling mediator.",
            })

        # ---- DML ----
        if isinstance(dml, dict):
            rows.append({
                "level": lvl,
                "method": "dml",
                "term": "dml_slope",
                "estimate": dml.get("dml_slope", None),
                "ci95_low": None,
                "ci95_high": None,
                "ci_excludes_0": None,
                "notes": "Cross-fit partialling-out slope (no CI in current implementation).",
            })

    df_sum = pd.DataFrame(rows)

    # write CSV
    csv_path = os.path.join(out_dir, "STEP4_mechanism_chain_summary.csv")
    df_sum.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # write a short markdown report
    md_path = os.path.join(out_dir, "STEP4_mechanism_chain_summary.md")
    lines = []
    lines.append("# STEP4 Mechanism-chain summary (robustness only)\n")
    lines.append("This summary aggregates mediation / DML outputs. Interpret as mechanism-consistent evidence, not mechanism identification.\n")

    for lvl in ["supplier", "product"]:
        sub = df_sum[df_sum["level"] == lvl]
        if sub.empty:
            continue
        lines.append(f"## {lvl}\n")
        for _, r in sub.iterrows():
            est = r.get("estimate", None)
            lo = r.get("ci95_low", None)
            hi = r.get("ci95_high", None)
            sig = r.get("ci_excludes_0", None)
            term = r.get("term", "")
            meth = r.get("method", "")
            if lo is None and hi is None:
                lines.append(f"- **{meth} / {term}**: {est}  \n  - note: {r.get('notes','')}\n")
            else:
                lines.append(f"- **{meth} / {term}**: {est} (CI95: [{lo}, {hi}]) | CI excludes 0: {sig}  \n  - note: {r.get('notes','')}\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return df_sum

# ----------------------
# STEP5: Run Final Key Factor Model
# ----------------------
def parse_labels(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            return [str(t).strip() for t in arr if str(t).strip()]
        except Exception:
            pass
    parts = re.split(r"[;,|/]\s*|\s*,\s*", s)
    return [p.strip() for p in parts if p.strip()]

def fit_embed(texts, dim=16, min_df=2, max_features=3000, random_state=42):
    tfidf = TfidfVectorizer(min_df=min_df, max_features=max_features)
    X = tfidf.fit_transform(texts)
    if X.shape[1] <= 1:
        return tfidf, None
    k = min(dim, X.shape[1]-1)
    svd = TruncatedSVD(n_components=k, random_state=random_state)
    svd.fit(X)
    return tfidf, svd

def transform_embed(tfidf, svd, texts):
    X = tfidf.transform(texts)
    if svd is None:
        return np.zeros((X.shape[0], 1))
    return svd.transform(X)

def knn(train_emb, query_emb, k=5, exclude_self=False):
    n_neighbors = min(k+1 if exclude_self else k, train_emb.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(train_emb)
    dists, idx = nn.kneighbors(query_emb, return_distance=True)
    sims = 1.0 - dists
    if exclude_self:
        idx = idx[:, 1:]
        sims = sims[:, 1:]
    return idx, sims

def neighbor_entropy(train_labels, nbr_idx, nbr_sim):
    ent = np.empty(len(nbr_idx), dtype=float)
    top1 = np.empty(len(nbr_idx), dtype=float)
    support = np.empty(len(nbr_idx), dtype=float)
    for i, (ids, sims) in enumerate(zip(nbr_idx, nbr_sim)):
        weights = {}
        total = 0.0
        for j, sim in zip(ids, sims):
            labs = train_labels[int(j)]
            if not labs:
                continue
            w = float(sim) / max(len(labs), 1)
            for lab in labs:
                weights[lab] = weights.get(lab, 0.0) + w
                total += w
        if total <= 0:
            ent[i] = np.nan
            top1[i] = np.nan
            support[i] = 0.0
        else:
            probs = np.fromiter(weights.values(), dtype=float) / total
            ent[i] = float(-(probs * np.log(probs + 1e-12)).sum())
            top1[i] = float(probs.max())
            support[i] = float(len(probs))
    return ent, top1, support

def compute_metrics(y_log_true, y_log_pred, y_price_true, y_price_pred):
    """calculate various evaluation metrics for the regression model,
    including the 95% confidence interval coverage."""

    # calculate R2
    denom = np.sum((y_log_true - y_log_true.mean()) ** 2)
    r2 = 1 - np.sum((y_log_true - y_log_pred) ** 2) / denom if denom > 0 else np.nan

    # calculate error metrics
    rmse_log = float(np.sqrt(np.mean((y_log_true - y_log_pred) ** 2)))
    mae_log = float(np.mean(np.abs(y_log_true - y_log_pred)))

    rmse_price = float(np.sqrt(np.mean((y_price_true - y_price_pred) ** 2)))
    mae_price = float(np.mean(np.abs(y_price_true - y_price_pred)))

    # calculate MAPE_price
    y_price_true_nonzero = y_price_true.copy()
    y_price_true_nonzero[y_price_true_nonzero == 0] = 1e-10  # 避免除以零
    mape_true = float(np.mean(np.abs((y_price_true - y_price_pred) / y_price_true_nonzero)))

    # calculate MAPE和SMAPE on log10(price)
    mape_log_val = mape(y_log_true, y_log_pred)
    smape_true = smape(y_price_true, y_price_pred)
    smape_log_val = smape(y_log_true, y_log_pred)

    # calculate CI95_coverage
    err = y_log_pred - y_log_true

    if len(err) > 1:
        # method 1: confidence interval based on the t-distribution
        n = len(err)
        err_mean = np.mean(err)
        err_std = np.std(err, ddof=1)  # 样本标准差
        t_critical = stats.t.ppf(0.975, df=n - 1)
        margin = t_critical * err_std / np.sqrt(n)
        lower_bound = err_mean - margin
        upper_bound = err_mean + margin

        # check if 0 is within the confidence interval
        # (because the expected value of the error should be 0).
        ci_contains_zero = 1 if (lower_bound <= 0 <= upper_bound) else 0

        # method 2: prediction interval coverage based on quantiles (more common method)
        # calculate the upper and lower bounds of the prediction interval
        alpha = 0.05  # 95% confidence level
        lower_quantile = np.percentile(err, alpha / 2 * 100)
        upper_quantile = np.percentile(err, (1 - alpha / 2) * 100)

        # calculate the proportion of actual values
        # that fall within the prediction interval.
        in_interval = ((err >= lower_quantile) & (err <= upper_quantile))
        coverage_rate = float(np.mean(in_interval))

        # method 3: prediction interval based on normal approximation
        # assumes the errors follow a normal distribution
        if err_std > 0:
            z_critical = stats.norm.ppf(0.975)
            normal_lower = -z_critical * err_std
            normal_upper = z_critical * err_std
            normal_coverage = float(np.mean((err >= normal_lower) & (err <= normal_upper)))
        else:
            normal_coverage = 1.0 if len(err) == 1 else np.nan

        # use the results from method 2
        # as the confidence interval coverage
        ci_coverage = coverage_rate

    else:
        # one sample cannot calculate confidence interval
        ci_coverage = np.nan
        ci_contains_zero = np.nan

    return {
        "R2_log10": float(r2),
        "RMSE_log10": rmse_log,
        "MAE_log10": mae_log,
        "RMSE_price": rmse_price,
        "MAE_price": mae_price,
        "MAPE_price": mape_true,  # modified MAPE
        "MAPE_price_log": mape_log_val,
        "SMAPE_price": smape_true,
        "SMAPE_price_log10": smape_log_val,
        "CI95_coverage_log10": ci_coverage,  # reak CI95
        "CI95_contains_zero": ci_contains_zero,  # new, is zero in confidence interval coverage
    }

def build_X(train_df, test_df, dim=16, knn_k=5, alpha=1.0, random_state=42):
    tfidf, svd = fit_embed(train_df["desc"].fillna("").astype(str).values,
                           dim=dim, min_df=2, max_features=3000, random_state=random_state)
    emb_tr = transform_embed(tfidf, svd, train_df["desc"].fillna("").astype(str).values)
    emb_te = transform_embed(tfidf, svd, test_df["desc"].fillna("").astype(str).values)

    nbr_tr_idx, nbr_tr_sim = knn(emb_tr, emb_tr, k=knn_k, exclude_self=True)
    nbr_te_idx, nbr_te_sim = knn(emb_tr, emb_te, k=knn_k, exclude_self=False)

    def sim_stats(s):
        return np.vstack([s.mean(1), s.max(1), s.std(1), s.sum(1)]).T

    sim_tr = sim_stats(nbr_tr_sim)
    sim_te = sim_stats(nbr_te_sim)

    tr_app = [parse_labels(x) for x in train_df["app_list"].values]
    tr_src = [parse_labels(x) for x in train_df["src_list"].values]

    app_ent_tr, app_top1_tr, app_sup_tr = neighbor_entropy(tr_app, nbr_tr_idx, nbr_tr_sim)
    app_ent_te, app_top1_te, app_sup_te = neighbor_entropy(tr_app, nbr_te_idx, nbr_te_sim)

    src_ent_tr, src_top1_tr, src_sup_tr = neighbor_entropy(tr_src, nbr_tr_idx, nbr_tr_sim)
    src_ent_te, src_top1_te, src_sup_te = neighbor_entropy(tr_src, nbr_te_idx, nbr_te_sim)

    X_num_tr = np.hstack([
        emb_tr, sim_tr,
        app_ent_tr[:, None], app_top1_tr[:, None], app_sup_tr[:, None],
        src_ent_tr[:, None], src_top1_tr[:, None], src_sup_tr[:, None],
    ])
    X_num_te = np.hstack([
        emb_te, sim_te,
        app_ent_te[:, None], app_top1_te[:, None], app_sup_te[:, None],
        src_ent_te[:, None], src_top1_te[:, None], src_sup_te[:, None],
    ])

    col_means = np.nanmean(X_num_tr, axis=0)
    col_means[~np.isfinite(col_means)] = 0.0

    def impute(X):
        X2 = X.copy()
        m = ~np.isfinite(X2)
        if m.any():
            X2[m] = np.take(col_means, np.where(m)[1])
        return X2

    X_num_tr = impute(X_num_tr)
    X_num_te = impute(X_num_te)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_sup_tr = ohe.fit_transform(train_df[["supplier"]].astype(str))
    X_sup_te = ohe.transform(test_df[["supplier"]].astype(str))

    X_tr = np.hstack([X_sup_tr, X_num_tr])
    X_te = np.hstack([X_sup_te, X_num_te])
    return X_tr, X_te

def summarize_over_folds(res):
    metrics_cols = [c for c in res.columns if c not in {"cv", "fold", "n_train", "n_test"}]
    out = []
    for cv, sub in res.groupby("cv"):
        k = len(sub)
        for met in metrics_cols:
            vals = sub[met].dropna().astype(float).values
            if len(vals) == 0:
                continue
            mu = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if k > 1 else 0.0
            se = sd / math.sqrt(k) if k > 1 else float("nan")

            if k > 1 and np.isfinite(se):
                tcrit = float(stats.t.ppf(0.975, df=k - 1))
                margin = tcrit * se  # 95% 置信区间半径
            else:
                margin = np.nan

            out.append({
                "cv": cv,
                "metric": met,
                "mean": mu,
                "std": sd,
                "ci95_margin": margin,
                "k_folds": k
            })
    return pd.DataFrame(out)

def load_saved_splits(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        folds = json.load(f)
    # folds: [{"train_idx":[...],"test_idx":[...]}, ...]
    splits = [(np.array(d["train_idx"], dtype=int), np.array(d["test_idx"], dtype=int)) for d in folds]
    return splits

def run_cv_with_splits(df, splits, cv_name, dim=16, knn_k=5, alpha=1.0, random_state=42):
    rows = []
    for fold, (tr, te) in enumerate(splits, start=1):
        dtr = df.iloc[tr].copy()
        dte = df.iloc[te].copy()
        X_tr, X_te = build_X(dtr, dte, dim=dim, knn_k=knn_k, alpha=alpha, random_state=random_state)

        model = Ridge(alpha=alpha, random_state=random_state)
        model.fit(X_tr, dtr["y_log10"].astype(float).values)
        pred_log = model.predict(X_te)
        pred_price = np.power(10.0, pred_log)

        m = compute_metrics(
            dte["y_log10"].astype(float).values, pred_log,
            dte["price"].astype(float).values, pred_price
        )
        m.update({"cv": cv_name, "fold": fold, "n_train": len(tr), "n_test": len(te)})
        rows.append(m)
    return pd.DataFrame(rows)

# STEP5: final model
def run_fe_product_model(
        out_dir,
        dataset_in_dir=os.path.join(OUTPUT_PATH, "STEP0_dataset_with_demand_structure.csv"),
        splits_kfold_json=os.path.join(OUTPUT_PATH, "STEP0_splits_kfold.json"),
        splits_group_json=os.path.join(OUTPUT_PATH, "STEP0_splits_groupkfold_supplier.json"),
):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(dataset_in_dir)

    req = ["supplier","desc","y_log10","price","src_list","app_list"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in dataset: {miss}")

    # Key: Do not sort or filter the df; keep the original row order
    splits_k = load_saved_splits(splits_kfold_json)
    splits_g = load_saved_splits(splits_group_json)

    res_k = run_cv_with_splits(df, splits_k, "KFold")
    res_g = run_cv_with_splits(df, splits_g, "GroupKFold")

    res_all = pd.concat([res_k, res_g], ignore_index=True)
    summary = summarize_over_folds(res_all)

    fold_path = os.path.join(out_dir, "STEP5_supplierFE_prodcomp_by_fold.csv")
    sum_path  = os.path.join(out_dir, "STEP5_supplierFE_prodcomp_summary.csv")
    res_all.to_csv(fold_path, index=False, encoding="utf-8")
    summary.to_csv(sum_path, index=False, encoding="utf-8")

def run_all_steps(
    excel_path: str,
    media_path: str,
    out_dir: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_pass: str,
    neo4j_db: str,
    n_splits: int,
    random_state: int,
    min_label_freq: int,
    heat_ablation: bool = HEAT_ABLATION_ENABLE_DEFAULT,
    ppr_heat_teleport_mass: float = PPR_HEAT_TELEPORT_MASS,
    metapath_heat_teleport_mass: float = METAPATH_HEAT_TELEPORT_MASS,
) -> None:
    ensure_dir(out_dir)

    stat_price(out_dir=out_dir)

    print("\n==============================")
    print("[STEP0] Load data and connect Neo4j")
    print("==============================")

    # ---- Connect KG ----
    db = Neo4jDB(neo4j_uri, neo4j_user, neo4j_pass, neo4j_db)
    # if db.g is None:
    #     raise RuntimeError("Neo4j is not available. Please check URI/user/pass/db.")

    # ---- Media heat ----
    mh = MediaHeat(media_path)
    heat_maps = mh.build_maps()
    _save_json(os.path.join(out_dir, "STEP0_media_heat_maps_meta.json"), {
        "has_media_file": bool(mh.df is not None and not mh.df.empty),
        "heat_keys": list(heat_maps.keys()),
    })

    # ---- STEP0: matched dataset ----
    df0 = build_matched_dataset(db, excel_path, out_dir)

    # Precompute splits ONCE (shared by STEP0 + STEP1 + STEP2/3 if needed)
    folds_k, folds_g = build_fixed_splits(df0, n_splits=n_splits, random_state=random_state)
    _save_json(os.path.join(out_dir, "STEP0_splits_kfold.json"), folds_k)
    _save_json(os.path.join(out_dir, "STEP0_splits_groupkfold_supplier.json"), folds_g)

    # ---- Fetch count tables for structure/mechanisms ----
    sup_list = sorted(df0["supplier"].astype(str).unique().tolist())
    df_edges = db.fetch_supplier_product_app_src_edges(suppliers=sup_list)

    df_sup_app = db.fetch_supplier_app_counts()
    df_sup_src = db.fetch_supplier_src_counts()
    df_src_app = db.fetch_src_app_counts()

    df_sup_app.to_csv(os.path.join(out_dir, "STEP0_sup_app_counts.csv"), index=False, encoding="utf-8-sig")
    df_sup_src.to_csv(os.path.join(out_dir, "STEP0_sup_src_counts.csv"), index=False, encoding="utf-8-sig")
    df_src_app.to_csv(os.path.join(out_dir, "STEP0_src_app_counts.csv"), index=False, encoding="utf-8-sig")
    df_edges.to_csv(os.path.join(out_dir, "STEP0_edges_supplier_product_app_src.csv"), index=False, encoding="utf-8-sig")

    # ---- STEP0: demand + structure columns (no y leakage) ----
    df1 = build_demand_structure_features(
        df_rows=df0,
        df_sup_app=df_sup_app,
        df_sup_src=df_sup_src,
        df_src_app=df_src_app,
        heat_maps=heat_maps,
        out_dir=out_dir,
    )

    # Optional inspection outputs
    try:
        step0_write_competitor_outputs_from_desc(df1, heat_maps=heat_maps, out_dir=out_dir, k=SIM_KNN_K, min_cos=SIM_MIN_COS)
    except Exception as e:
        write_text(os.path.join(out_dir, "STEP0_competitor_outputs_error.txt"), str(e))

    # ---- STEP0: 4-block and 5-block evaluation using EXACT same splits ----
    try:
        evaluate_5block_paper(df1, folds=folds_k, out_dir=out_dir, min_label_freq=min_label_freq)
    except Exception as e:
        write_text(os.path.join(out_dir, "STEP0_block_eval_error.txt"), str(e))

    print("\n[STEP0_5BLOCK] 5-block explainability (structure + demand + text + complemen)")

    print("\n[Done] Key outputs:")
    print(" - STEP0_matched_dataset.csv")
    print(" - STEP0_dataset_with_demand_structure.csv")
    print(" - STEP0_5block_cv_long.csv")
    print(" - STEP0_5block_cv_summary.csv")

    # ---- STEP1: FE diagnosis using EXACT same splits ----
    print("\n==============================")
    print("[STEP1] Fixed-effect diagnosis + FE tables")
    print("==============================")

    X_fe = df1[["supplier", "src_list", "app_list"]].copy()
    y = df1["y_log10"].values.astype(float)

    cv_long_k, summary_k = evaluate_fe_on_fixed_splits(X_fe, y, folds_k, tag="KFold_fixed", min_freq=min_label_freq)

    cv_long_k.to_csv(os.path.join(out_dir, "STEP1_fe_cv_long_kfold.csv"), index=False, encoding="utf-8-sig")
    summary_k.to_csv(os.path.join(out_dir, "STEP1_fe_cv_summary_kfold.csv"), index=False, encoding="utf-8-sig")

    shap_k = compute_shapley_table(cv_long_k, tag="KFold_fixed")
    shap_k.to_csv(os.path.join(out_dir, "STEP1_fe_shapley_kfold.csv"), index=False, encoding="utf-8-sig")

    drop_k = compute_dropone(summary_k, tag="KFold_fixed")
    drop_k.to_csv(os.path.join(out_dir, "STEP1_fe_dropone_kfold.csv"), index=False, encoding="utf-8-sig")

    concl = conclude_step1(summary_k, drop_k)
    write_text(os.path.join(out_dir, "STEP1_conclusion.txt"), concl)

    # Full-sample interpretable FE tables
    try:
        fit_fullsample_fe_tables(df1, min_label_freq=min_label_freq, out_dir=out_dir)
    except Exception as e:
        write_text(os.path.join(out_dir, "STEP1_fullsample_fe_error.txt"), str(e))

    # Optional Hierarchical Bayes
    if FIT_HIERARCHICAL_BAYES:
        try:
            sig, share, meta = fit_hierarchical_bayes_crossed(df1, min_label_freq=min_label_freq)
            sig.to_csv(os.path.join(out_dir, "STEP1_hb_sigma_summary.csv"), index=False, encoding="utf-8-sig")
            share.to_csv(os.path.join(out_dir, "STEP1_hb_variance_share.csv"), index=False, encoding="utf-8-sig")
            _save_json(os.path.join(out_dir, "STEP1_hb_meta.json"), meta)
        except Exception as e:
            write_text(os.path.join(out_dir, "STEP1_hb_error.txt"), str(e))

    # ---- STEP2: reasoning + mechanism proxies ----
    print("\n==============================")
    print("[STEP2] Meta-path / Personalized PageRank reasoning + mechanism proxies + FE explanation")
    print("==============================")

    try:
        app_heat_total = heat_maps.get("total_log", {})
        kg_reasoner = KGReasoner(df_edges=df_edges, app_heat_total=app_heat_total, topk_apps=PPR_TOPK_APPS, topk_srcs=PPR_TOPK_SRCS)

        mp_reasoner = MetaPathPPRReasoner(
            df_sup_app=df_sup_app,
            df_sup_src=df_sup_src,
            df_src_app=df_src_app,
            app_heat_total=app_heat_total,
        )

        # Build app thickness table
        df_app_stats = compute_category_market_structure(df_sup_app.rename(columns={"app": "app"}), cat_col="app", supplier_col="supplier")
        df_thick = compute_market_thickness_for_app(df_app_stats, app_heat_total=app_heat_total, heat_is_log=True)
        df_app_struct = df_app_stats.merge(df_thick, on="app", how="left").fillna(0.0)
        df_app_struct.to_csv(os.path.join(out_dir, "STEP2_app_structure_thickness.csv"), index=False, encoding="utf-8-sig")
        app_thickness_map = {str(r["app"]): float(r.get("thickness", 0.0)) for _, r in df_app_struct.iterrows()}

        # Supplier/src/app mechanism proxies
        suppliers = sorted(df1["supplier"].astype(str).unique().tolist())
        df_sup_mech = build_supplier_reasoning_features(
            suppliers=suppliers,
            reasoner=kg_reasoner,
            mp_reasoner=mp_reasoner,
            app_thickness=app_thickness_map,
            heat_maps=heat_maps,
        )

        df_sup_mech.to_csv(os.path.join(out_dir, "STEP2_mechanisms_supplier.csv"), index=False, encoding="utf-8-sig")

        srcs = sorted(set([x for lst in df1["src_list"].tolist() for x in parse_list_cell(lst)]))
        apps = sorted(set([x for lst in df1["app_list"].tolist() for x in parse_list_cell(lst)]))

        df_src_mech = build_src_reasoning_features(src_list=srcs, mp_reasoner=mp_reasoner, app_thickness=app_thickness_map)
        df_app_mech = build_app_reasoning_features(app_list=apps, mp_reasoner=mp_reasoner)

        df_src_mech.to_csv(os.path.join(out_dir, "STEP2_mechanisms_src.csv"), index=False, encoding="utf-8-sig")
        df_app_mech.to_csv(os.path.join(out_dir, "STEP2_mechanisms_app.csv"), index=False, encoding="utf-8-sig")

        # Merge row-level mechanism features
        d2 = df1.merge(df_sup_mech, on="supplier", how="left").fillna(0.0)

        sup_num_cols = select_numeric_feature_cols(df_sup_mech, id_col="supplier")
        src_num_cols = select_numeric_feature_cols(df_src_mech, id_col="src")
        app_num_cols = select_numeric_feature_cols(df_app_mech, id_col="app")

        X_src_row = add_fractional_label_mechanism_features(
            df_rows=d2,
            list_col="src_list",
            mech_df=df_src_mech,
            mech_key="src",
            feature_cols=src_num_cols,
            prefix="row_src_mech__",
        )

        X_app_row = add_fractional_label_mechanism_features(
            df_rows=d2,
            list_col="app_list",
            mech_df=df_app_mech,
            mech_key="app",
            feature_cols=app_num_cols,
            prefix="row_app_mech__",
        )

        d2 = pd.concat([d2, X_src_row, X_app_row], axis=1).fillna(0.0)
        d2.to_csv(os.path.join(out_dir, "STEP2_dataset_with_mechanisms.csv"), index=False, encoding="utf-8-sig")

        # Predictive validation (STEP2)
        mech_cols = [c for c in d2.columns if (c.startswith("ppr_") or c.startswith("mp_ppr_") or c.startswith("supemb_") or c.startswith("row_src_mech__") or c.startswith("row_app_mech__"))]
        mech_cols = [c for c in mech_cols if np.issubdtype(d2[c].dtype, np.number)]

        scenarios = [
            ("B_mechanisms_with_supplier", True),
            ("C_mechanisms_no_supplier_coldstart", False),
        ]
        eval_rows = []
        for name, use_supplier in scenarios:
            pipe = make_step2_price_pipeline(
                include_supplier_fe=use_supplier,
                include_src=True,
                include_app=True,
                numeric_cols=mech_cols,
                min_freq=min_label_freq,
            )
            # Use the same fixed splits via wrappers
            # Here we do KFold and GroupKFold evaluation through precomputed splits for strict equality.
            mu_k, std_k = cv_predict_pipeline_price(pipe, d2, y_col="y_log10", splits=[(np.array(f["train_idx"]), np.array(f["test_idx"])) for f in folds_k])

            eval_rows.append({
                "scenario": name,
                "kfold_R2_log10": float(r2_score(d2["y_log10"].values, mu_k)),
                "kfold_RMSE_log10": float(rmse_from_log10(d2["y_log10"].values, mu_k)),
            })

        pd.DataFrame(eval_rows).to_csv(os.path.join(out_dir, "STEP2_price_predictive_validation.csv"), index=False, encoding="utf-8-sig")

        # Explain FE by mechanisms (entity-level CV); prefer crossfit if possible
        # Supplier
        try:
            crossfit_fe_to_mechanisms(
                df_rows=df1,
                entity_type="supplier",
                mech_df=df_sup_mech,
                mech_key="supplier",
                mech_feature_cols=sup_num_cols,
                min_freq=min_label_freq,
                k=n_splits,
                out_path_prefix=os.path.join(out_dir, "STEP2_crossfit_fe_to_mech_supplier"),
            )
        except Exception as e:
            write_text(os.path.join(out_dir, "STEP2_crossfit_supplier_error.txt"), str(e))

        # Src
        try:
            crossfit_fe_to_mechanisms(
                df_rows=df1,
                entity_type="src",
                mech_df=df_src_mech,
                mech_key="src",
                mech_feature_cols=src_num_cols,
                min_freq=min_label_freq,
                k=n_splits,
                out_path_prefix=os.path.join(out_dir, "STEP2_crossfit_fe_to_mech_src"),
            )
        except Exception as e:
            write_text(os.path.join(out_dir, "STEP2_crossfit_src_error.txt"), str(e))

        # App
        try:
            crossfit_fe_to_mechanisms(
                df_rows=df1,
                entity_type="app",
                mech_df=df_app_mech,
                mech_key="app",
                mech_feature_cols=app_num_cols,
                min_freq=min_label_freq,
                k=n_splits,
                out_path_prefix=os.path.join(out_dir, "STEP2_crossfit_fe_to_mech_app"),
            )
        except Exception as e:
            write_text(os.path.join(out_dir, "STEP2_crossfit_app_error.txt"), str(e))

    except Exception as e:
        write_text(os.path.join(out_dir, "STEP2_error.txt"), str(e))

    # ---- STEP3: product inference (OOF predictions) ----
    print("\n==============================")
    print("[STEP3] product inference (OOF predictions)")
    print("==============================")

    try:
        if "d2" in locals():
            run_step3_product_inference(
                df_rows=d2,
                out_dir=out_dir,
                n_splits=n_splits,
                random_state=random_state,
                min_label_freq=min_label_freq,
                splits_kfold=[(np.array(f["train_idx"]), np.array(f["test_idx"])) for f in folds_k],
                numeric_cols_for_price=[
                    c for c in d2.columns
                    if (
                               c.startswith("row_src_mech__")
                               or c.startswith("row_app_mech__")
                               or c.startswith("ppr_")
                               or c.startswith("mp_ppr_")
                               or c.startswith("supemb_")
                       )
                       and (not c.endswith("_json"))
                ]
            )

            step3_write_competitor_outputs_from_desc(d2, heat_maps=heat_maps, out_dir=out_dir)
            # ---- STEP3.1: product-level FE -> mechanisms crossfit (outputs *_cv.csv, *_coef.csv, *_meta.json) ----
            try:
                d2_prod = d2.copy()
                d2_prod["prod_key"] = d2_prod.apply(
                    lambda r: f"PROD::{str(r['name']).strip()}||SUP::{str(r['supplier']).strip()}",
                    axis=1,
                )

                df_prod_mech, prod_mech_cols = build_step3_product_mechanism_proxies(
                    df_rows=d2_prod,
                    df_edges=df_edges,
                    heat_maps=heat_maps,
                    app_thickness_map=app_thickness_map if "app_thickness_map" in locals() else None,
                    heat_ablation=heat_ablation,
                    ppr_heat_teleport_mass=ppr_heat_teleport_mass,
                    metapath_heat_teleport_mass=metapath_heat_teleport_mass,
                )
                df_prod_mech.to_csv(os.path.join(out_dir, "STEP3_product_mechanism_proxies.csv"), index=False, encoding="utf-8")
                if heat_ablation:
                    try:
                        write_heat_ablation_expheat_price_gradient(
                            df_rows=d2_prod,
                            prod_mech_df=df_prod_mech,
                            out_dir=out_dir,
                            folds_k=folds_k,
                            folds_g=folds_g,
                        )
                    except Exception as _e:
                        write_text(os.path.join(out_dir, "HEAT_ABLATION__error.txt"), str(_e))

                crossfit_fe_to_mechanisms(
                    df_rows=d2_prod,
                    entity_type="product",
                    mech_df=df_prod_mech,
                    mech_key="prod_key",
                    mech_feature_cols=prod_mech_cols,
                    min_freq=min_label_freq,
                    k=n_splits,
                    out_path_prefix=os.path.join(out_dir, "STEP3_crossfit_fe_to_mech_product"),
                )
            except Exception as e:
                write_text(os.path.join(out_dir, "STEP3_crossfit_product_error.txt"), str(e))

    except Exception as e:
        write_text(os.path.join(out_dir, "STEP3_error.txt"), str(e))

    # ---- STEP4: mechanism-chain analysis (lightweight, evidence-based) ----
    print("\n==============================")
    print("[STEP4] mechanism-chain analysis (lightweight, evidence-based)")
    print("==============================")

    # =========================
    # PATCH: STEP4 product-level causal-chain analysis
    # =========================
    cca = MechanismChainAnalyzer(random_state=random_state)

    try:
        if "d2" in locals():
            dp = d2.copy()
            dp["y_log10"] = dp["y_log10"].astype(float)

            # Ensure prod_key exists (same convention as STEP3 proxies)
            if "prod_key" not in dp.columns:
                if ("name" in dp.columns) and ("supplier" in dp.columns):
                    dp["prod_key"] = dp.apply(
                        lambda r: f"PROD::{str(r['name']).strip()}||SUP::{str(r['supplier']).strip()}",
                        axis=1,
                    )
                elif "product" in dp.columns:
                    dp["prod_key"] = dp["product"].astype(str).str.strip()

            # 1) Merge STEP3 product mechanism proxies (contains prod_ppr_expected_heat_total/thickness, prod_sim_* etc.)
            prod_mech_path = os.path.join(out_dir, "STEP3_product_mechanism_proxies.csv")
            if os.path.exists(prod_mech_path):
                df_prod_mech = pd.read_csv(prod_mech_path)
                if "prod_key" in df_prod_mech.columns:
                    dp = dp.merge(df_prod_mech, on="prod_key", how="left")

            # 2) Aggregate at product level (one row per prod_key)
            prod_cols = [c for c in dp.columns if (c.startswith("prod_") or c.startswith("comp_"))]
            prod_cols = [c for c in prod_cols if (c in dp.columns and np.issubdtype(dp[c].dtype, np.number))]
            agg_p = dp.groupby("prod_key").agg({**{"y_log10": "mean"}, **{c: "mean" for c in prod_cols}}).reset_index()

            def _pick_first(cols):
                for c in cols:
                    if c in agg_p.columns:
                        return c
                return None

            # Prefer MPPR product names if you added them; otherwise fall back to PPR.
            T = _pick_first(["prod_mp_ppr_expected_thickness", "prod_ppr_expected_thickness"])
            M = _pick_first(["prod_mp_ppr_expected_heat_total", "prod_ppr_expected_heat_total"])

            # ---- Mediation (mechanism-chain) ----
            # Example: market thickness (supply/structure) -> heat (demand) -> price
            if T is not None and M is not None:
                med_p = cca.mediation_bootstrap(
                    df=agg_p,
                    T=T,
                    M=M,
                    Y="y_log10",
                    controls=[c for c in ["comp_app_entropy", "comp_src_entropy"] if c in agg_p.columns],
                    B=BOOTSTRAP_B,
                )
                _save_json(os.path.join(out_dir, "STEP4_product_mediation.json"), med_p)

            # ---- DML (robustness check for partial effect) ----
            # Treatment: thickness; Outcome: price; Controls: heat + competitor pressure + product network metrics if available
            if T is not None:
                W = []
                for c in [
                    M,
                    "comp_expected_heat_total",
                    "comp_app_entropy",
                    "comp_src_entropy",
                    "prod_sim_pagerank",
                    "prod_sim_strength",
                    "prod_sim_deg",
                    "prod_sim_mean",
                    "prod_sim_max",
                    "prod_sim_std",
                    "prod_sim_sum",
                ]:
                    if c is not None and c in agg_p.columns:
                        W.append(c)

                dml_p = cca.dml_continuous_treatment(
                    df=agg_p,
                    T=T,
                    Y="y_log10",
                    W=W,
                    n_splits=DML_SPLITS,
                )
                _save_json(os.path.join(out_dir, "STEP4_product_dml.json"), dml_p)
    except Exception as e:
        write_text(os.path.join(out_dir, "STEP4_product_error.txt"), str(e))

    try:
        # Supplier-level aggregation: mean y, plus selected mechanisms if present
        if "d2" in locals():
            ds = d2.copy()
            ds["y_log10"] = ds["y_log10"].astype(float)
            sup_cols = [c for c in ds.columns if c.startswith("supplier_") or c.startswith("ppr_") or c.startswith("mp_ppr_")]
            sup_cols = [c for c in sup_cols if np.issubdtype(ds[c].dtype, np.number)]
            agg = ds.groupby("supplier").agg({**{"y_log10": "mean"}, **{c: "mean" for c in sup_cols}}).reset_index()

            # Example mediation: T=mp_ppr_expected_thickness -> M=mp_ppr_expected_heat_total -> Y=y_log10
            if "mp_ppr_expected_thickness" in agg.columns and "mp_ppr_expected_heat_total" in agg.columns:
                med = cca.mediation_bootstrap(
                    df=agg,
                    T="mp_ppr_expected_thickness",
                    M="mp_ppr_expected_heat_total",
                    Y="y_log10",
                    controls=[],
                    B=BOOTSTRAP_B,
                )
                _save_json(os.path.join(out_dir, "STEP4_supplier_mediation.json"), med)

            # Example DML: T=supplier_monopoly -> Y=y_log10 controlling for heat/thickness if exist
            if "supplier_monopoly" in agg.columns:
                W = [c for c in ["mp_ppr_expected_heat_total", "mp_ppr_expected_thickness"] if c in agg.columns]
                dml = cca.dml_continuous_treatment(df=agg, T="supplier_monopoly", Y="y_log10", W=W, n_splits=DML_SPLITS)
                _save_json(os.path.join(out_dir, "STEP4_supplier_dml.json"), dml)

    except Exception as e:
        write_text(os.path.join(out_dir, "STEP4_supplier_error.txt"), str(e))

    summarize_mechanism_chain_outputs(out_dir)

    # Final Key Factor Model
    run_fe_product_model(out_dir=os.path.join(out_dir))

# =============================================================================
# Paper tables & appendices export (Table 1–11; Appendix 3–6)
# =============================================================================
def _round_numeric_df(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Round numeric columns to fixed decimals (for paper-ready tables)."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(decimals)
    return out

def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required output file: {path}")
    return pd.read_csv(path)

def _export_sheet_and_csv(
    writer: pd.ExcelWriter,
    sheet_name: str,
    df: pd.DataFrame,
    csv_dir: str,
    csv_name: str,
) -> None:
    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(os.path.join(csv_dir, csv_name), index=False, encoding="utf-8-sig")

def export_tables_1_to_11_and_appendices_3_4_5_6(
        out_dir: str,
        decimals: int = 3,
        excel_name: str = "paper_tables_v7_all_with_app6.xlsx",
        csv_subdir: str = "paper_tables_csv",
) -> str:
    """
    Export paper-ready tables (Table 1–11; Appendix Table 3–6) from pipeline outputs.

    Notes (v7 patch):
    - FE ablation summary is read from STEP1_fe_cv_summary_kfold.csv
      (older drafts sometimes referred to STEP1_fe_ablation_summary_kfold.csv).
    - Output directory is {out_dir}/table_out/ by default.
    - Shapley tables (Table 4 & Table 7) are exported as fold-averaged contributions to match the manuscript.
    - Table 10 is reconstructed from step3_HEAT_ABLATION__* outputs.
    """
    import os
    import numpy as np
    import pandas as pd

    table_root = os.path.join(out_dir, "table_out")
    csv_dir = os.path.join(table_root, csv_subdir)
    os.makedirs(csv_dir, exist_ok=True)
    excel_path = os.path.join(table_root, excel_name)

    # --------------------------
    # Helpers
    # --------------------------
    def rd(name: str) -> pd.DataFrame:
        return _safe_read_csv(os.path.join(out_dir, name))

    def export(writer, sheet: str, df: pd.DataFrame, csv_name: str) -> None:
        df2 = _round_numeric_df(df.copy(), decimals=decimals)
        _export_sheet_and_csv(writer, sheet, df2, csv_dir, csv_name)

    def spearman_p_from_rho(rho: float, n: int) -> float:
        # Approximate p-value for Spearman rho via t approximation.
        try:
            import math
            from scipy.stats import t as tdist  # type: ignore
            if n is None or n < 3 or rho is None or not np.isfinite(rho):
                return float("nan")
            denom = max(1e-12, 1.0 - float(rho) ** 2)
            t = float(rho) * math.sqrt((n - 2) / denom)
            return float(2.0 * tdist.sf(abs(t), df=n - 2))
        except Exception:
            return float("nan")

    # --------------------------
    # Load core files
    # --------------------------
    df_price_stats = rd("STEP0_price_stats.csv")  # Table 1
    df_step0 = rd("STEP0_paper_5block_paper_summary.csv")  # Table 2–3
    df_dropone = rd("STEP0_paper_5block_paper_dropone.csv")  # Table 5
    df_shapley = rd("STEP0_paper_5block_paper_shapley.csv")  # Table 4 (fold-level)

    df_fe_cv = rd("STEP1_fe_cv_summary_kfold.csv")  # Table 6
    df_fe_sh = rd("STEP1_fe_shapley_kfold.csv")  # Table 7 (fold-level)
    df_fe_drop = rd("STEP1_fe_dropone_kfold.csv")  # Table 8

    df_cv_sup = rd("STEP2_crossfit_fe_to_mech_supplier_cv.csv")  # Table 9
    df_cv_src = rd("STEP2_crossfit_fe_to_mech_src_cv.csv")  # Table 9
    df_cv_app = rd("STEP2_crossfit_fe_to_mech_app_cv.csv")  # Table 9
    df_cv_prod = rd("STEP3_crossfit_fe_to_mech_product_cv.csv")  # Table 9

    df_heat_pairs = rd("step3_HEAT_ABLATION__expheat_price_gradient_pairs.csv")  # Table 10
    df_heat_grad = rd("step3_HEAT_ABLATION__expheat_price_gradient_summary.csv")  # Table 10

    df_step5 = rd("STEP5_supplierFE_prodcomp_summary.csv")  # Table 11 + Appendix 5

    df_var_stats = rd("STEP0_paper_5block_paper_variable_stats.csv")  # Appendix 3
    df_chain = rd("STEP4_mechanism_chain_summary.csv")  # Appendix 4

    df_edges_all = rd("STEP0_edges_supplier_product_app_src.csv")  # Appendix 6 inputs
    df_price_rows = rd("STEP0_dataset_with_demand_structure.csv")  # Appendix 6 inputs

    # --------------------------
    # Baseline row
    # --------------------------
    baseline_scn = "FE+PROD+DEM+SUP+MKT"
    base_row = df_step0.loc[df_step0["scenario"] == baseline_scn]
    if base_row.empty:
        raise ValueError(f"Baseline row '{baseline_scn}' not found in STEP0_paper_5block_paper_summary.csv")

    baseline_r2 = float(base_row["R2_log10_mean"].iloc[0])
    baseline_map = {
        "CI95": float(base_row["CI95_coverage_log10_mean"].iloc[0]),
        "MAE": float(base_row["MAE_log10_mean"].iloc[0]),
        "MAPE": float(base_row["MAPE_price_mean"].iloc[0]),
        "R2": float(base_row["R2_log10_mean"].iloc[0]),
        "RMSE": float(base_row["RMSE_log10_mean"].iloc[0]),
        "SMAPE": float(base_row["SMAPE_log10_mean"].iloc[0]),
    }
    metrics = ["CI95", "MAE", "MAPE", "R2", "RMSE", "SMAPE"]

    # --------------------------
    # Export all tables
    # --------------------------
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Table 1
        t1 = df_price_stats.rename(columns={"stat": "variable"}).copy()
        export(writer, "Table1_price_dist", t1, "Table1_price_dist.csv")

        # Table 2
        t2 = pd.DataFrame(
            {
                "metric": metrics,
                "mean": [
                    baseline_map["CI95"],
                    baseline_map["MAE"],
                    baseline_map["MAPE"],
                    baseline_map["R2"],
                    baseline_map["RMSE"],
                    baseline_map["SMAPE"],
                ],
                "std": [
                    float(base_row["CI95_coverage_log10_std"].iloc[0]),
                    float(base_row["MAE_log10_std"].iloc[0]),
                    float(base_row["MAPE_price_std"].iloc[0]),
                    float(base_row["R2_log10_std"].iloc[0]),
                    float(base_row["RMSE_log10_std"].iloc[0]),
                    float(base_row["SMAPE_log10_std"].iloc[0]),
                ],
            }
        )
        export(writer, "Table2_baseline", t2, "Table2_baseline.csv")

        # Table 3 (R2 only)
        keep_models = [
            "DEM", "DEM+MKT", "DEM+SUP", "DEM+SUP+MKT",
            "PROD", "PROD+DEM", "PROD+DEM+MKT", "PROD+DEM+SUP", "PROD+DEM+SUP+MKT",
            "PROD+MKT", "PROD+SUP", "PROD+SUP+MKT",
            "SUP", "SUP+MKT",
            "MKT",
            "FE", "FE+SUP", "FE+MKT", "FE+DEM", "FE+DEM+MKT", "FE+DEM+SUP", "FE+SUP+MKT",
            "FE+PROD", "FE+PROD+DEM", "FE+PROD+MKT", "FE+PROD+SUP",
            "FE+PROD+DEM+MKT", "FE+PROD+DEM+SUP", "FE+PROD+SUP+MKT",
            "FE+PROD+DEM+SUP+MKT",
        ]
        dfa = df_step0[df_step0["scenario"].isin(keep_models)].copy().sort_values("scenario")
        t3 = pd.DataFrame(
            {
                "model": dfa["scenario"].astype(str).values,
                "metric": "R2",
                "mean": dfa["R2_log10_mean"].astype(float).values,
                "std": dfa["R2_log10_std"].astype(float).values,
            }
        )
        t3["baseline_mean"] = baseline_r2
        t3["delta_vs_baseline"] = t3["mean"] - baseline_r2
        export(writer, "Table3_ablation_R2", t3, "Table3_ablation_R2.csv")

        # Table 4 (fold-averaged Shapley)
        shap_cols = [c for c in df_shapley.columns if c.startswith("shapley_")]
        shap_means = df_shapley[shap_cols].mean(axis=0).to_dict()
        order = ["shapley_FE", "shapley_PROD", "shapley_DEM", "shapley_SUP", "shapley_MKT"]
        t4 = pd.DataFrame(
            [(c.replace("shapley_", ""), float(shap_means[c])) for c in order if c in shap_means],
            columns=["metric", "result"],
        )
        export(writer, "Table4_shapley", t4, "Table4_shapley.csv")

        # Table 5
        export(writer, "Table5_dropone", df_dropone.copy(), "Table5_dropone.csv")

        # Table 6 (FE ablation; R2 only)
        dfk = df_fe_cv.copy()
        if "cv_tag" in dfk.columns:
            dfk = dfk[dfk["cv_tag"] == "KFold_fixed"].copy()
        t6 = pd.DataFrame(
            {
                "scenario": dfk["scenario"].astype(str).values,
                "metric": "R2",
                "mean": dfk["R2_log10_mean"].astype(float).values,
                "std": dfk["R2_log10_std"].astype(float).values,
            }
        )
        t6 = t6[t6["scenario"] != "no_fe"].copy()
        t6["baseline_mean"] = baseline_r2
        t6["delta_vs_baseline"] = t6["mean"] - baseline_r2
        export(writer, "Table6_FE_ablation_R2", t6, "Table6_FE_ablation_R2.csv")

        # Table 7 (FE Shapley; fold-averaged)
        fe_cols = [c for c in df_fe_sh.columns if c.startswith("shapley_")]
        fe_means = df_fe_sh[fe_cols].mean(axis=0).to_dict()
        fe_order = ["shapley_supplier", "shapley_app", "shapley_src"]
        t7 = pd.DataFrame(
            [(c.replace("shapley_", ""), float(fe_means[c])) for c in fe_order if c in fe_means],
            columns=["metric", "result"],
        )
        export(writer, "Table7_FE_shapley", t7, "Table7_FE_shapley.csv")

        # Table 8
        export(writer, "Table8_FE_dropone", df_fe_drop.copy(), "Table8_FE_dropone.csv")

        # Table 9 (KG reasoning CV; mean over folds)
        def cv_mean(df: pd.DataFrame) -> dict:
            return {"cv R2": float(df["R2"].mean()), "MAE": float(df["MAE"].mean()), "RMSE": float(df["RMSE"].mean())}

        t9 = pd.DataFrame(
            [
                {"metric": "supplier", **cv_mean(df_cv_sup)},
                {"metric": "src", **cv_mean(df_cv_src)},
                {"metric": "app", **cv_mean(df_cv_app)},
                {"metric": "product", **cv_mean(df_cv_prod)},
            ]
        )
        export(writer, "Table9_KG_mechanism_cv", t9, "Table9_KG_mechanism_cv.csv")

        # ---------------------------------------------------------
        # Table 10 (heat ablation robustness) - Optimized Version
        # ---------------------------------------------------------
        # 1. Load consistency pairs
        cons_dict = {}
        for _, row in df_heat_pairs.iterrows():
            m_key = "PPR" if row["method"] == "ppr" else "MPPR"
            cons_dict[m_key] = {
                "r": float(row["pearson_struct_vs_tele"]),
                "rho": float(row["spearman_struct_vs_tele"])
            }

        rows = []
        # 2. Process each method (ppr -> PPR, mp_ppr -> MPPR)
        method_map = {"ppr": "PPR", "mp_ppr": "MPPR"}

        for m_raw, m_disp in method_map.items():
            dfm = df_heat_grad[df_heat_grad["method"] == m_raw].copy()
            if dfm.empty: continue

            # Extract modes
            s = dfm[dfm["mode"] == "structural"].iloc[0]
            b = dfm[dfm["mode"] == "teleport"].iloc[0]  # "teleport" refers to heat-biased mode

            # Consistency from pairs file
            c = cons_dict.get(m_disp, {"r": np.nan, "rho": np.nan})

            # Helper to calculate p-value and scaled slope
            def get_stats(row_data):
                rho = float(row_data["spearman_x_y"])
                n = int(row_data["n"])
                p = spearman_p_from_rho(rho, n)
                # Note: Scaling slope by 1/ln(10) to match Table 10's reporting units
                slope = float(row_data["slope_ols"]) / np.log(10)
                return rho, p, slope

            s_rho, s_p, s_slope = get_stats(s)
            b_rho, b_p, b_slope = get_stats(b)

            rows.append({
                "Method": m_disp,
                "Inferred consistency (r / ρ)": f"{c['r']:.3f} / {c['rho']:.3f}",
                "Price assoc. (Structural: ρ (p) / slope)": f"{s_rho:.3f} ({s_p:.3f}) / {s_slope:.3f}",
                "Price assoc. (Biased: ρ (p) / slope)": f"{b_rho:.3f} ({b_p:.3f}) / {b_slope:.3f}"
            })

        t10 = pd.DataFrame(rows)
        export(writer, "Table10_heat_ablation", t10, "Table10_heat_ablation.csv")

        # Table 11 (key factor model; KFold)
        # STEP5_supplierFE_prodcomp_summary.csv is stored in long format: (cv, metric, mean, std, ...).
        dfk = df_step5[df_step5["cv"] == "KFold"].copy()

        metric_map = {
            "CI95": "CI95_coverage_log10",
            "MAE": "MAE_log10",
            "MAPE": "MAPE_price",
            "R2": "R2_log10",
            "RMSE": "RMSE_log10",
            "SMAPE": "SMAPE_price_log10",
        }

        rows = []
        for m in metrics:
            mm = metric_map.get(m, None)
            if mm is None:
                continue
            r = dfk[dfk["metric"] == mm]
            if r.empty:
                continue
            rows.append({"metric": m, "mean": float(r["mean"].iloc[0]), "std": float(r["std"].iloc[0])})

        t11 = pd.DataFrame(rows)
        t11["baseline_mean"] = t11["metric"].map(baseline_map)
        t11["delta_vs_baseline"] = t11["mean"] - t11["baseline_mean"]
        export(writer, "Table11_keyfactor", t11, "Table11_keyfactor.csv")

        # Appendix Table 3
        export(writer, "AppendixTable3_var_stats", df_var_stats.copy(), "AppendixTable3_var_stats.csv")

        # Appendix Table 4
        export(writer, "AppendixTable4_mech_chain", df_chain.copy(), "AppendixTable4_mech_chain.csv")

        # Appendix Table 5 (GroupKFold)
        dfg = df_step5[df_step5["cv"] == "GroupKFold"].copy()

        metric_map = {
            "CI95": "CI95_coverage_log10",
            "MAE": "MAE_log10",
            "MAPE": "MAPE_price",
            "R2": "R2_log10",
            "RMSE": "RMSE_log10",
            "SMAPE": "SMAPE_price_log10",
        }

        rows = []
        for m in metrics:
            mm = metric_map.get(m, None)
            if mm is None:
                continue
            r = dfg[dfg["metric"] == mm]
            if r.empty:
                continue
            rows.append({"metric": m, "mean": float(r["mean"].iloc[0]), "std": float(r["std"].iloc[0])})

        app5 = pd.DataFrame(rows)
        export(writer, "AppendixTable5_groupkfold", app5, "AppendixTable5_groupkfold.csv")

        # Appendix Table 6 (KG ontology/coverage statistics on priced subgraph)
        price_keys = set(
            ("PROD::" + df_price_rows["name"].astype(str) + "||SUP::" + df_price_rows["supplier"].astype(str)).tolist()
        )
        df_edges_all = df_edges_all.copy()
        df_edges_all["prod_key"] = "PROD::" + df_edges_all["product"].astype(str) + "||SUP::" + df_edges_all[
            "supplier"].astype(str)
        df_edges = df_edges_all[df_edges_all["prod_key"].isin(price_keys)].copy()

        import ast
        def parse_list(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except Exception:
                    return []
            return [] if pd.isna(x) else [x]

        df_edges["src_list_parsed"] = df_edges["src_list"].apply(parse_list)
        df_edges["app_list_parsed"] = df_edges["app_list"].apply(parse_list)

        n_products = int(df_edges["prod_key"].nunique())
        n_suppliers = int(df_edges["supplier"].nunique())
        n_src = int(pd.Series([s for lst in df_edges["src_list_parsed"] for s in lst]).nunique())
        n_app = int(pd.Series([a for lst in df_edges["app_list_parsed"] for a in lst]).nunique())

        e_provide = int(len(df_edges))
        e_src = int(df_edges["src_list_parsed"].apply(len).sum())
        e_app = int(df_edges["app_list_parsed"].apply(len).sum())

        prod_per_supplier = df_edges.groupby("supplier")["prod_key"].nunique()
        app_per_prod = df_edges["app_list_parsed"].apply(len)
        src_per_prod = df_edges["src_list_parsed"].apply(len)

        def five_num(s: pd.Series) -> dict:
            return {
                "mean": float(s.mean()),
                "P25": float(s.quantile(0.25)),
                "P50": float(s.quantile(0.50)),
                "P75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }

        deg_supplier = five_num(prod_per_supplier)
        deg_app = five_num(app_per_prod)
        deg_src = five_num(src_per_prod)

        app6 = pd.DataFrame(
            [
                {"Panel": "A. Node coverage", "Statistic": "Products (priced pairs)", "Value": n_products},
                {"Panel": "A. Node coverage", "Statistic": "Suppliers (connected to priced pairs)",
                 "Value": n_suppliers},
                {"Panel": "A. Node coverage", "Statistic": "src industries (connected to priced pairs)",
                 "Value": n_src},
                {"Panel": "A. Node coverage", "Statistic": "app industries (connected to priced pairs)",
                 "Value": n_app},
                {"Panel": "B. Edge coverage", "Statistic": "Triples: provide_data", "Value": e_provide},
                {"Panel": "B. Edge coverage", "Statistic": "Triples: source_industry", "Value": e_src},
                {"Panel": "B. Edge coverage", "Statistic": "Triples: applied_to", "Value": e_app},
                {
                    "Panel": "C. Degree & label sparsity",
                    "Statistic": "Products per supplier (mean; P25/P50/P75; max)",
                    "Value": f'{deg_supplier["mean"]:.2f}; {deg_supplier["P25"]:.0f}/{deg_supplier["P50"]:.0f}/{deg_supplier["P75"]:.0f}; {deg_supplier["max"]:.0f}',
                },
                {
                    "Panel": "C. Degree & label sparsity",
                    "Statistic": "App labels per product (mean; P25/P50/P75; max)",
                    "Value": f'{deg_app["mean"]:.2f}; {deg_app["P25"]:.0f}/{deg_app["P50"]:.0f}/{deg_app["P75"]:.0f}; {deg_app["max"]:.0f}',
                },
                {
                    "Panel": "C. Degree & label sparsity",
                    "Statistic": "Src labels per product (mean; P25/P50/P75; max)",
                    "Value": f'{deg_src["mean"]:.2f}; {deg_src["P25"]:.0f}/{deg_src["P50"]:.0f}/{deg_src["P75"]:.0f}; {deg_src["max"]:.0f}',
                },
            ]
        )
        export(writer, "AppendixTable6_KG_stats", app6, "AppendixTable6_KG_stats.csv")

    print(f"[export_tables] Excel: {excel_path}")
    print(f"[export_tables] CSVs:  {csv_dir}")
    return excel_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", type=str, default=EXCEL_PATH)
    parser.add_argument("--media", type=str, default=MEDIA_PATH)
    parser.add_argument("--out", type=str, default=OUTPUT_PATH)

    parser.add_argument("--neo4j_uri", type=str, default=NEO4J_URI)
    parser.add_argument("--neo4j_user", type=str, default=NEO4J_USER)
    parser.add_argument("--neo4j_pass", type=str, default=NEO4J_PASS)
    parser.add_argument("--neo4j_db", type=str, default=NEO4J_DB)

    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--min_label_freq", type=int, default=FE_MIN_LABEL_FREQ_FOR_TABLE)
    # Heat usage ablation (Structural vs Teleport-only heat bias)
    parser.add_argument("--heat_ablation", action="store_true", default=HEAT_ABLATION_ENABLE_DEFAULT)
    parser.add_argument("--ppr_heat_teleport_mass", type=float, default=PPR_HEAT_TELEPORT_MASS)
    parser.add_argument("--metapath_heat_teleport_mass", type=float, default=METAPATH_HEAT_TELEPORT_MASS)

    args = parser.parse_args()

    run_all_steps(
        excel_path=args.excel,
        media_path=args.media,
        out_dir=args.out,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_pass=args.neo4j_pass,
        neo4j_db=args.neo4j_db,
        n_splits=args.n_splits,
        random_state=args.seed,
        min_label_freq=args.min_label_freq,
        heat_ablation=bool(args.heat_ablation),
        ppr_heat_teleport_mass=float(args.ppr_heat_teleport_mass),
        metapath_heat_teleport_mass=float(args.metapath_heat_teleport_mass),
    )

    export_tables_1_to_11_and_appendices_3_4_5_6(args.out, decimals=3)

if __name__ == "__main__":
    main()

