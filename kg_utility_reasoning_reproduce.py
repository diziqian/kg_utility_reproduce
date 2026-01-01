#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kg_utility_reasoning_reproduce.py (v6.3)

This version fixes the regression where Table export fails due to missing
  - STEP0_paper_5block_paper_summary.csv
and also ensures Table10 inputs exist:
  - step3_HEAT_ABLATION__expheat_price_gradient_pairs.csv

Key ideas (for exact reproducibility under anonymized desc):
1) CSV-only KG backend: replace Neo4jDB with CSVNeo4jDB reading anymous/neo4j_export/*.csv
2) Reuse exact fixed splits shipped in anymous:
     STEP0_splits_kfold.json
     STEP0_splits_groupkfold_supplier.json
3) STEP0 5-block CV: override fit_text_embedder/transform_text_embedder so that
   evaluate_5block_paper() uses precomputed fold-specific embeddings from:
     STEP0_fold_textemb_cache.npz  (fold1_emb..foldK_emb, each [n_rows, dim])
   This avoids "empty vocabulary" from anonymized/empty desc.
4) PRODUCT embedding elsewhere: override build_product_text_embedding to read:
     STEP0_product_textemb.csv/.npz  (numeric-only)
5) Post-run safeguards:
   - If STEP0_paper_5block_paper_summary.csv is missing, call core.evaluate_5block_paper() again explicitly.
   - If step3_HEAT_ABLATION__expheat_price_gradient_pairs.csv is missing, call
     core.write_heat_ablation_expheat_price_gradient() explicitly.

Usage:
  python kg_utility_reasoning_reproduce.py \
    --input anymous.zip \
    --core kg_utility_reasoning.py \
    --out_dir result_kg_reproduce
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
import types
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import re

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def is_zip(p: str) -> bool:
    return os.path.isfile(p) and p.lower().endswith(".zip")

def read_json(p: str) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_zip(zpath: str, out_dir: str) -> str:
    ensure_dir(out_dir)
    with zipfile.ZipFile(zpath, "r") as z:
        z.extractall(out_dir)
    return out_dir

def _exists_any(root: str, names: List[str]) -> bool:
    return any(os.path.exists(os.path.join(root, n)) for n in names)

def _find_first_recursive(root: str, target_names: List[str]) -> Optional[str]:
    # deterministic walk
    for cur, dirs, files in os.walk(root):
        dirs.sort()
        files_sorted = sorted(files)
        for tn in target_names:
            if tn in files_sorted:
                return os.path.join(cur, tn)
    return None

def find_anymous_root(d: str) -> str:
    """
    Locate the true anymous root that contains the *required* artifacts.

    We require:
      - name_price_anonymized.xlsx
      - media_result.xlsx
      - neo4j_export/ (dir)
      - STEP0_fold_textemb_cache.npz   (for STEP0 CV)
      - STEP0_product_textemb.csv or .npz (for PRODUCT block)

    Some users accidentally create a nested structure like:
      anymous/anymous/<files>
    This finder will resolve the deepest correct root.
    """
    d = os.path.abspath(d)

    def ok(root: str) -> bool:
        if not (os.path.exists(os.path.join(root, "name_price_anonymized.xlsx"))
                and os.path.exists(os.path.join(root, "media_result.xlsx"))
                and os.path.isdir(os.path.join(root, "neo4j_export"))):
            return False
        if not os.path.exists(os.path.join(root, "STEP0_fold_textemb_cache.npz")):
            return False
        if not _exists_any(root, ["STEP0_product_textemb.csv", "STEP0_product_textemb.npz"]):
            return False
        return True

    # direct
    if ok(d):
        return d

    # common nested
    cand = os.path.join(d, "anymous")
    if ok(cand):
        return cand

    # search subdirs (deterministic)
    best = None
    for cur, dirs, files in os.walk(d):
        dirs.sort()
        if ok(cur):
            # prefer deeper path (more specific) to avoid picking an outer folder missing embeddings
            if best is None or len(cur) > len(best):
                best = cur
    if best is not None:
        return best

    # last resort: maybe embeddings exist deeper; show helpful diagnostics
    emb = _find_first_recursive(d, ["STEP0_product_textemb.csv", "STEP0_product_textemb.npz"])
    fold = _find_first_recursive(d, ["STEP0_fold_textemb_cache.npz"])
    raise FileNotFoundError(
        "Cannot locate anymous root under: %s.\n" % d +
        "Found embedding candidate: %s\n" % emb +
        "Found fold-cache candidate: %s\n" % fold +
        "Expected to find, in the same folder: name_price_anonymized.xlsx, media_result.xlsx, neo4j_export/, "
        "STEP0_fold_textemb_cache.npz, STEP0_product_textemb.(csv|npz)"
    )


def load_core_module(core_script: str):
    spec = importlib.util.spec_from_file_location("kg_core", os.path.abspath(core_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load core script: {core_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _read_npz_from_path_or_zip(path: str) -> np.lib.npyio.NpzFile:
    # path may be real file or already extracted; here assume real file
    return np.load(path, allow_pickle=True)


def load_product_embedding_table(any_root: str) -> Tuple[pd.DataFrame, List[str]]:
    csv_path = os.path.join(any_root, "STEP0_product_textemb.csv")
    npz_path = os.path.join(any_root, "STEP0_product_textemb.npz")
    # Some exports may place files under a nested anymous/ folder; search recursively as fallback.
    if (not os.path.exists(csv_path)) and (not os.path.exists(npz_path)):
        csv_hit = _find_first_recursive(any_root, ["STEP0_product_textemb.csv"])
        npz_hit = _find_first_recursive(any_root, ["STEP0_product_textemb.npz"])
        if csv_hit:
            csv_path = csv_hit
        if npz_hit:
            npz_path = npz_hit
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        emb_cols = [c for c in df.columns if c.startswith("textemb_")]
        if "name" not in df.columns or "supplier" not in df.columns or not emb_cols:
            raise ValueError(f"Bad embedding CSV schema: {csv_path}, cols={list(df.columns)}")
        df = df[["name", "supplier"] + emb_cols].copy()
        df["name"] = df["name"].astype(str)
        df["supplier"] = df["supplier"].astype(str)
        df = df.drop_duplicates(["name", "supplier"]).reset_index(drop=True)
        return df, emb_cols
    if os.path.exists(npz_path):
        npz = np.load(npz_path, allow_pickle=True)
        keys = npz["keys"]  # (n,2)
        emb_cols = [str(x) for x in npz["emb_cols"].tolist()]
        mat = npz["mat"].astype(float)
        df = pd.DataFrame(keys, columns=["name", "supplier"])
        for j, c in enumerate(emb_cols):
            df[c] = mat[:, j]
        df["name"] = df["name"].astype(str)
        df["supplier"] = df["supplier"].astype(str)
        df = df.drop_duplicates(["name", "supplier"]).reset_index(drop=True)
        return df, emb_cols
    raise FileNotFoundError(f"Missing STEP0_product_textemb.csv/.npz under {any_root}")

def load_fold_textemb_cache(any_root: str) -> Tuple[Dict[int, np.ndarray], int, int]:
    """
    Compatible with multiple NPZ schemas.

    Supported key patterns:
      - fold1_emb, fold2_emb, ...   (newer)
      - fold1, fold2, ...           (older)
    Optional:
      - n_rows, dim_default         (if missing, infer from fold1 array shape)
    """
    npz_path = os.path.join(any_root, "STEP0_fold_textemb_cache.npz")
    if not os.path.exists(npz_path):
        hit = _find_first_recursive(any_root, ["STEP0_fold_textemb_cache.npz"])
        if hit:
            npz_path = hit
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing fold text embedding cache: {npz_path}")

    npz = np.load(npz_path, allow_pickle=True)

    # Collect fold arrays (fold1_emb or fold1)
    fold2emb: Dict[int, np.ndarray] = {}
    for k in npz.files:
        m = re.match(r"^fold(\d+)(?:_emb)?$", str(k))
        if not m:
            continue
        fid = int(m.group(1))
        arr = np.asarray(npz[k], dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{k} must be 2D [n_rows, dim], got shape={arr.shape}")
        fold2emb[fid] = arr

    if not fold2emb:
        raise ValueError(f"No fold arrays found in {npz_path}. keys={npz.files}")

    # Infer n_rows / dim_default if absent
    if "n_rows" in npz.files:
        n_rows = int(np.asarray(npz["n_rows"]).reshape(-1)[0])
    else:
        n_rows = int(next(iter(fold2emb.values())).shape[0])

    if "dim_default" in npz.files:
        dim = int(np.asarray(npz["dim_default"]).reshape(-1)[0])
    else:
        dim = int(next(iter(fold2emb.values())).shape[1])

    return fold2emb, n_rows, dim

# -----------------------------
# CSV-backed DB
# -----------------------------
class CSVNeo4jDB:
    def __init__(self, uri: str, user: str, password: str, neo_dir: str):
        self.neo_dir = os.path.abspath(str(neo_dir))
        if not os.path.isdir(self.neo_dir):
            raise FileNotFoundError(f"neo4j_export dir not found: {self.neo_dir}")

        self.nodes_dp = pd.read_csv(os.path.join(self.neo_dir, "nodes_dataproduct.csv"))
        self.nodes_supplier = pd.read_csv(os.path.join(self.neo_dir, "nodes_supplier.csv"))

        self.rel_provide = pd.read_csv(os.path.join(self.neo_dir, "rel_provides_data.csv"))
        self.rel_app = pd.read_csv(os.path.join(self.neo_dir, "rel_applied_to.csv"))
        self.rel_src = pd.read_csv(os.path.join(self.neo_dir, "rel_source_industry.csv"))

        # Normalize edge schemas (export may use *_name columns)
        if "src" not in self.rel_src.columns and "src_name" in self.rel_src.columns:
            self.rel_src = self.rel_src.rename(columns={"src_name": "src"})
        if "app" not in self.rel_app.columns and "app_name" in self.rel_app.columns:
            self.rel_app = self.rel_app.rename(columns={"app_name": "app"})

        # Normalize columns to match core expectations
        ren = {}
        if "name_anon" in self.nodes_dp.columns and "name" not in self.nodes_dp.columns:
            ren["name_anon"] = "name"
        if "supplier_anon" in self.nodes_dp.columns and "supplier" not in self.nodes_dp.columns:
            ren["supplier_anon"] = "supplier"
        if "desc_anon" in self.nodes_dp.columns and "desc" not in self.nodes_dp.columns:
            ren["desc_anon"] = "desc"
        self.nodes_dp = self.nodes_dp.rename(columns=ren)

        if "dp_id" not in self.nodes_dp.columns:
            raise ValueError(f"nodes_dataproduct.csv missing dp_id, cols={list(self.nodes_dp.columns)}")
        if "name" not in self.nodes_dp.columns or "supplier" not in self.nodes_dp.columns:
            raise ValueError(f"nodes_dataproduct.csv missing name/supplier, cols={list(self.nodes_dp.columns)}")

        # Ensure desc exists (may be empty; STEP0 uses cached embeddings anyway)
        if "desc" not in self.nodes_dp.columns:
            self.nodes_dp["desc"] = ""

        for c in ["dp_id", "name", "supplier", "desc"]:
            self.nodes_dp[c] = self.nodes_dp[c].astype(str)

        # Build dp_id -> label lists
        # (Some exports name columns as src_name/app_name; we normalized above.)
        if "dp_id" not in self.rel_src.columns:
            raise ValueError(f"rel_source_industry.csv missing dp_id, cols={list(self.rel_src.columns)}")
        if "dp_id" not in self.rel_app.columns:
            raise ValueError(f"rel_applied_to.csv missing dp_id, cols={list(self.rel_app.columns)}")
        if "src" not in self.rel_src.columns:
            # allow empty schema
            self.rel_src["src"] = ""
        if "app" not in self.rel_app.columns:
            self.rel_app["app"] = ""

        self.rel_src["dp_id"] = self.rel_src["dp_id"].astype(str)
        self.rel_app["dp_id"] = self.rel_app["dp_id"].astype(str)
        self.rel_src["src"] = self.rel_src["src"].astype(str)
        self.rel_app["app"] = self.rel_app["app"].astype(str)

        self._src_map = (
            self.rel_src.dropna(subset=["dp_id"])
            .groupby("dp_id")["src"]
            .apply(self._uniq_sorted)
            .to_dict()
        )
        self._app_map = (
            self.rel_app.dropna(subset=["dp_id"])
            .groupby("dp_id")["app"]
            .apply(self._uniq_sorted)
            .to_dict()
        )

        # Counts tables:
        # Some anymous exports do NOT include precomputed counts_*.csv.
        # Compute counts from raw edges + nodes_dataproduct (supplier attribute).
        dp_sup = self.nodes_dp[["dp_id", "supplier"]].copy()
        dp_sup["dp_id"] = dp_sup["dp_id"].astype(str)
        dp_sup["supplier"] = dp_sup["supplier"].astype(str)

        app_edges = self.rel_app[["dp_id", "app"]].copy()
        src_edges = self.rel_src[["dp_id", "src"]].copy()

        # supplier-app counts
        tmp = app_edges.merge(dp_sup, on="dp_id", how="left")
        tmp = tmp.dropna(subset=["supplier", "app"])
        self.counts_sup_app = (
            tmp.groupby(["app", "supplier"], as_index=False)["dp_id"]
            .nunique()
            .rename(columns={"dp_id": "prod_cnt"})
        )

        # supplier-src counts
        tmp2 = src_edges.merge(dp_sup, on="dp_id", how="left")
        tmp2 = tmp2.dropna(subset=["supplier", "src"])
        self.counts_sup_src = (
            tmp2.groupby(["src", "supplier"], as_index=False)["dp_id"]
            .nunique()
            .rename(columns={"dp_id": "prod_cnt"})
        )

        # src-app counts
        tmp3 = src_edges.merge(app_edges, on="dp_id", how="inner")
        tmp3 = tmp3.dropna(subset=["src", "app"])
        self.counts_src_app = (
            tmp3.groupby(["src", "app"], as_index=False)["dp_id"]
            .nunique()
            .rename(columns={"dp_id": "prod_cnt"})
        )
    @staticmethod
    def _uniq_sorted(x: pd.Series) -> List[str]:
        s = set()
        for v in x.dropna().tolist():
            t = str(v).strip()
            if t:
                s.add(t)
        return sorted(s)

    @staticmethod
    def _union_lists(series: pd.Series) -> List[str]:
        s = set()
        for lst in series.tolist():
            if isinstance(lst, list):
                for v in lst:
                    t = str(v).strip()
                    if t:
                        s.add(t)
        return sorted(s)

    def fetch_matched_multilabel(self, df_price: pd.DataFrame) -> pd.DataFrame:
        dp = self.nodes_dp[["dp_id", "name", "supplier", "desc"]].copy()
        dp["src_list"] = dp["dp_id"].map(lambda i: self._src_map.get(str(i), []))
        dp["app_list"] = dp["dp_id"].map(lambda i: self._app_map.get(str(i), []))

        # Collapse duplicates (name,supplier) into one row with unioned label lists
        kg = dp.groupby(["name", "supplier"], as_index=False).agg(
            desc=("desc", "first"),
            src_list=("src_list", self._union_lists),
            app_list=("app_list", self._union_lists),
        )

        keys = df_price[["name", "supplier"]].astype(str).drop_duplicates()
        kg = kg.merge(keys, on=["name", "supplier"], how="inner")
        return kg[["name", "supplier", "desc", "src_list", "app_list"]]

    def fetch_supplier_product_app_src_edges(self, suppliers: Optional[List[str]] = None) -> pd.DataFrame:
        dp = self.nodes_dp[["dp_id", "name", "supplier"]].copy()
        if suppliers is not None:
            sup_set = set([str(s) for s in suppliers])
            dp = dp[dp["supplier"].astype(str).isin(sup_set)].copy()
        dp["src_list"] = dp["dp_id"].map(lambda i: self._src_map.get(str(i), []))
        dp["app_list"] = dp["dp_id"].map(lambda i: self._app_map.get(str(i), []))
        edges = dp.groupby(["supplier", "name"], as_index=False).agg(
            app_list=("app_list", self._union_lists),
            src_list=("src_list", self._union_lists),
        ).rename(columns={"name": "product"})
        return edges[["supplier", "product", "app_list", "src_list"]]

    def fetch_supplier_app_counts(self) -> pd.DataFrame:
        return self.counts_sup_app[["app", "supplier", "prod_cnt"]].copy()

    def fetch_supplier_src_counts(self) -> pd.DataFrame:
        return self.counts_sup_src[["src", "supplier", "prod_cnt"]].copy()

    def fetch_src_app_counts(self) -> pd.DataFrame:
        return self.counts_src_app[["src", "app", "prod_cnt"]].copy()


# -----------------------------
# Monkeypatches
# -----------------------------

def patch_build_X_with_product_textemb(core, df_emb_map: pd.DataFrame, emb_cols: List[str]):
    """
    Fix empty-vocabulary crash in run_fe_product_model() under anonymized/empty desc.

    core.build_X() (used by run_fe_product_model) fits TF-IDF on train_df['desc'].
    Under anonymization, desc may become empty or unique tokens -> TF-IDF raises:
        ValueError: empty vocabulary

    We replace core.build_X with a version that uses exported numeric text embeddings
    from STEP0_product_textemb.csv/.npz, matched by (name, supplier), and keeps the
    rest of the feature construction identical (KNN stats + neighbor entropy + supplier OHE).
    """
    # Build lookup map
    dfm = df_emb_map.copy()
    dfm["name"] = dfm["name"].astype(str)
    dfm["supplier"] = dfm["supplier"].astype(str)
    use_cols = [c for c in emb_cols if c.startswith("textemb_")]
    # Ensure stable order textemb_0..textemb_{k}
    def _emb_sort_key(c):
        m = re.match(r"^textemb_(\d+)$", str(c))
        return int(m.group(1)) if m else 10**9
    use_cols = sorted(use_cols, key=_emb_sort_key)

    # Dict: (name,supplier) -> np.ndarray
    key2vec = {}
    for _, r in dfm[["name","supplier"] + use_cols].dropna(subset=["name","supplier"]).iterrows():
        key2vec[(r["name"], r["supplier"])] = r[use_cols].to_numpy(dtype=float)

    # Import helpers from core
    knn = core.knn
    parse_labels = core.parse_labels
    neighbor_entropy = core.neighbor_entropy
    OneHotEncoder = core.OneHotEncoder

    def build_X_cached(train_df, test_df, dim=16, knn_k=5, alpha=1.0, random_state=42):
        # Build embeddings from lookup
        def mat_from_df(d: pd.DataFrame) -> np.ndarray:
            rows = []
            for n, s in zip(d["name"].astype(str).values, d["supplier"].astype(str).values):
                v = key2vec.get((n, s))
                if v is None:
                    rows.append(np.zeros((len(use_cols),), dtype=float))
                else:
                    rows.append(v)
            M = np.vstack(rows) if rows else np.zeros((0, len(use_cols)), dtype=float)
            k = min(int(dim), M.shape[1]) if M.ndim == 2 else int(dim)
            if M.shape[1] < k:
                pad = np.zeros((M.shape[0], k - M.shape[1]), dtype=float)
                M = np.hstack([M, pad])
            return M[:, :k]

        emb_tr = mat_from_df(train_df)
        emb_te = mat_from_df(test_df)

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

        # Supplier one-hot
        # sklearn compatibility handled elsewhere (patch_onehotencoder_compat)
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_sup_tr = ohe.fit_transform(train_df[["supplier"]].astype(str))
        X_sup_te = ohe.transform(test_df[["supplier"]].astype(str))

        X_tr = np.hstack([X_num_tr, X_sup_tr])
        X_te = np.hstack([X_num_te, X_sup_te])

        return X_tr, X_te

    core.build_X = build_X_cached  # type: ignore

def patch_onehotencoder_compat(core):
    try:
        import sklearn.preprocessing as skp
    except Exception:
        return

    Orig = getattr(core, "OneHotEncoder", None)
    if Orig is None:
        return

    def OneHotEncoder_compat(*args, **kwargs):
        if "sparse" in kwargs:
            try:
                return skp.OneHotEncoder(*args, **kwargs)
            except TypeError:
                kwargs["sparse_output"] = kwargs.pop("sparse")
                return skp.OneHotEncoder(*args, **kwargs)
        return skp.OneHotEncoder(*args, **kwargs)

    core.OneHotEncoder = OneHotEncoder_compat  # type: ignore


def patch_fixed_splits(core, folds_k, folds_g):
    def build_fixed_splits_patched(df: pd.DataFrame, n_splits: int, random_state: int):
        n = len(df)
        max_idx = max(max(fd["train_idx"] + fd["test_idx"]) for fd in folds_k)
        if max_idx >= n:
            raise ValueError(
                f"[SplitsMismatch] splits max_idx={max_idx} >= n_rows={n}. "
                f"Ensure STEP0_splits_kfold.json corresponds to the current matched dataset."
            )
        return folds_k, folds_g
    core.build_fixed_splits = build_fixed_splits_patched  # type: ignore


def patch_product_text_embedding(core, df_emb_map: pd.DataFrame, emb_cols: List[str]):
    df_map = df_emb_map.copy()
    cols = list(emb_cols)

    def build_product_text_embedding_patched(df_rows, text_col="desc", dim=50, min_df=2, **kwargs):
        d = df_rows.copy()
        if "name" not in d.columns or "supplier" not in d.columns:
            raise KeyError("build_product_text_embedding expects columns: name, supplier")
        keys = d[["name", "supplier"]].astype(str).reset_index(drop=True)

        want = [f"textemb_{i}" for i in range(int(dim))]
        src_cols = [c for c in want if c in cols]

        out = keys.merge(
            df_map[["name", "supplier"] + src_cols].drop_duplicates(["name", "supplier"]),
            on=["name", "supplier"],
            how="left",
        )
        for c in want:
            if c not in out.columns:
                out[c] = 0.0
        out[want] = out[want].fillna(0.0).astype(float)
        return out[["name", "supplier"] + want], None

    core.build_product_text_embedding = build_product_text_embedding_patched  # type: ignore


def patch_step0_fold_text_embedding(core, folds_k, fold2emb: Dict[int, np.ndarray]):
    """
    evaluate_5block_paper() calls, per fold:
      tfidf, svd = fit_text_embedder(train_text, dim, min_df)
      emb_tr = transform_text_embedder(tfidf, svd, train_text)
      emb_te = transform_text_embedder(tfidf, svd, test_text)

    Patch fit/transform so that emb_tr/emb_te come from fold2emb[fold_id][train_idx/test_idx].
    """
    orig_fit = getattr(core, "fit_text_embedder", None)
    orig_transform = getattr(core, "transform_text_embedder", None)
    if not callable(orig_fit) or not callable(orig_transform):
        raise RuntimeError("Core missing fit_text_embedder/transform_text_embedder; cannot patch STEP0 text CV.")

    K = len(folds_k)
    state = {
        "call_count": 0,
        "fid": 0,
        "train_len": None,
        "test_len": None,
        "emb_tr": None,
        "emb_te": None,
    }

    def fit_text_embedder_patched(train_text, dim=50, min_df=2, **kwargs):
        state["call_count"] += 1
        # cycle 1..K to stay safe even if core re-enters evaluate_5block_paper
        fid = ((state["call_count"] - 1) % K) + 1
        state["fid"] = fid

        if fid in fold2emb:
            tr = np.array(folds_k[fid - 1]["train_idx"], dtype=int)
            te = np.array(folds_k[fid - 1]["test_idx"], dtype=int)
            emb_all = np.asarray(fold2emb[fid], dtype=float)
            state["train_len"] = int(len(tr))
            state["test_len"] = int(len(te))
            state["emb_tr"] = emb_all[tr, :int(dim)]
            state["emb_te"] = emb_all[te, :int(dim)]
            return "__CACHED__", "__CACHED__"
        # fallback
        return orig_fit(train_text, dim=dim, min_df=min_df)

    def transform_text_embedder_patched(tfidf, svd, texts, **kwargs):
        if tfidf == "__CACHED__" and svd == "__CACHED__":
            n = len(texts)
            tl = state.get("train_len")
            if tl is not None and n == tl:
                return np.asarray(state["emb_tr"], dtype=float)
            # otherwise treat as test
            return np.asarray(state["emb_te"], dtype=float)
        return orig_transform(tfidf, svd, texts)

    core.fit_text_embedder = fit_text_embedder_patched  # type: ignore
    core.transform_text_embedder = transform_text_embedder_patched  # type: ignore


def call_run_all_steps_compat(core, **kwargs):
    sig = inspect.signature(core.run_all_steps)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    missing = [k for k in accepted if k not in filtered and sig.parameters[k].default is inspect._empty]
    if missing:
        raise TypeError(f"core.run_all_steps missing required args: {missing}. Provided keys={sorted(filtered.keys())}")
    return core.run_all_steps(**filtered)


def _load_df0_for_step0(any_root: str, out_dir: str) -> pd.DataFrame:
    # Prefer out_dir if core already wrote STEP0_matched_dataset.csv; otherwise use anymous snapshot.
    cand1 = os.path.join(out_dir, "STEP0_matched_dataset.csv")
    cand2 = os.path.join(any_root, "STEP0_matched_dataset_anonymized.csv")
    path = cand1 if os.path.exists(cand1) else cand2
    df0 = pd.read_csv(path)
    return df0


def _ensure_prod_key(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "prod_key" not in d.columns and ("name" in d.columns) and ("supplier" in d.columns):
        d["prod_key"] = d.apply(
            lambda r: f"PROD::{str(r['name']).strip()}||SUP::{str(r['supplier']).strip()}",
            axis=1,
        )
    return d


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./anymous", help="Path to anymous folder OR anymous.zip")
    ap.add_argument("--core", default="./kg_utility_reasoning.py", help="Path to kg_utility_reasoning.py")
    ap.add_argument("--out_dir", default="./result_kg_reproduce", help="Output directory for reproduction run")
    ap.add_argument("--price_xlsx", default="./anymous/name_price_anonymized.xlsx", help="Override price xlsx (default: <anymous>/name_price_anonymized.xlsx)")
    ap.add_argument("--media_xlsx", default="./anymous/media_result.xlsx", help="Override media xlsx (default: <anymous>/media_result.xlsx)")
    ap.add_argument("--neo_dir", default="./anymous/neo4j_export", help="Override neo4j_export dir (default: <anymous>/neo4j_export)")
    ap.add_argument("--min_label_freq", type=int, default=-1, help="Override min label freq (default: core constant or 3)")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # Resolve anymous root
    # if is_zip(args.input):
    #     extract_dir = os.path.join(args.out_dir, "_extracted_anymous")
    #     if os.path.exists(extract_dir):
    #         shutil.rmtree(extract_dir)
    #     extract_zip(args.input, extract_dir)
    #     any_root = find_anymous_root(extract_dir)
    # else:
    #     any_root = find_anymous_root(args.input)

    any_root = args.input

    price_xlsx = args.price_xlsx.strip() or os.path.join(any_root, "name_price_anonymized.xlsx")
    media_xlsx = args.media_xlsx.strip() or os.path.join(any_root, "media_result.xlsx")
    neo_dir = args.neo_dir.strip() or os.path.join(any_root, "neo4j_export")

    for p in [price_xlsx, media_xlsx]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")
    if not os.path.isdir(neo_dir):
        raise FileNotFoundError(f"Missing required directory: {neo_dir}")

    # Load meta + splits + embeddings
    meta_path = os.path.join(any_root, "anymous_meta.json")
    meta = read_json(meta_path) if os.path.exists(meta_path) else {}
    splits_k_path = os.path.join(any_root, "STEP0_splits_kfold.json")
    splits_g_path = os.path.join(any_root, "STEP0_splits_groupkfold_supplier.json")
    folds_k = read_json(splits_k_path)
    folds_g = read_json(splits_g_path)

    df_emb_map, emb_cols = load_product_embedding_table(any_root)
    fold2emb, cache_n_rows, cache_dim = load_fold_textemb_cache(any_root)

    # Provide dummy py2neo if absent (CSV-only)
    try:
        import py2neo  # noqa: F401
    except Exception:
        dummy = types.ModuleType("py2neo")
        class _DummyGraph:
            def __init__(self, *a, **kw):
                raise RuntimeError("Neo4j Graph is disabled in reproduce mode (CSV-only).")
        dummy.Graph = _DummyGraph
        sys.modules["py2neo"] = dummy

    core = load_core_module(args.core)

    # Align core's global paths to avoid reading default ./price_files/name_price.xlsx
    try:
        core.EXCEL_PATH = price_xlsx  # type: ignore
        core.MEDIA_PATH = media_xlsx  # type: ignore
    except Exception:
        pass

    # Monkeypatches
    core.Neo4jDB = CSVNeo4jDB  # type: ignore
    patch_onehotencoder_compat(core)
    patch_fixed_splits(core, folds_k, folds_g)
    patch_step0_fold_text_embedding(core, folds_k, fold2emb)
    patch_product_text_embedding(core, df_emb_map, emb_cols)
    patch_build_X_with_product_textemb(core, df_emb_map, emb_cols)

    n_splits = int(meta.get("n_splits", len(folds_k) or 5))
    random_state = int(meta.get("random_state", getattr(core, "RANDOM_STATE", 7)))
    min_label_freq = int(args.min_label_freq) if int(args.min_label_freq) >= 0 else int(getattr(core, "FE_MIN_LABEL_FREQ_FOR_TABLE", 3))

    # Run full pipeline (heat_ablation MUST be True for Table10 input files)
    call_run_all_steps_compat(
        core,
        excel_path=price_xlsx,
        media_path=media_xlsx,
        out_dir=str(args.out_dir),
        neo4j_uri="",
        neo4j_user="",
        neo4j_pass="",
        neo4j_db=str(neo_dir),
        n_splits=n_splits,
        random_state=random_state,
        min_label_freq=min_label_freq,
        heat_ablation=True,
        ppr_heat_teleport_mass=float(meta.get("ppr_heat_teleport_mass", getattr(core, "PPR_HEAT_TELEPORT_MASS", 0.35))),
        metapath_heat_teleport_mass=float(meta.get("metapath_heat_teleport_mass", getattr(core, "METAPATH_HEAT_TELEPORT_MASS", 0.35))),
    )

    # -----------------------
    # Safeguard #1: STEP0 paper summary (Table 2-3 inputs)
    # -----------------------
    step0_sum = os.path.join(args.out_dir, "STEP0_paper_5block_paper_summary.csv")
    if not os.path.exists(step0_sum):
        df0 = _load_df0_for_step0(any_root, args.out_dir)
        # evaluate_5block_paper expects list[dict] folds, not tuple folds (implementation uses fd["train_idx"])
        if hasattr(core, "evaluate_5block_paper"):
            core.evaluate_5block_paper(df0, folds_k, str(args.out_dir), min_label_freq=min_label_freq)  # type: ignore
        if not os.path.exists(step0_sum):
            err_path = os.path.join(args.out_dir, "STEP0_block_eval_error.txt")
            msg = ""
            if os.path.exists(err_path):
                with open(err_path, "r", encoding="utf-8", errors="ignore") as f:
                    msg = f.read().strip()
            raise FileNotFoundError(
                f"STEP0_paper_5block_paper_summary.csv still missing after explicit evaluate_5block_paper().\n"
                f"STEP0_block_eval_error.txt:\n{msg}"
            )

    # -----------------------
    # Safeguard #2: STEP3 heat ablation (Table10 inputs)
    # -----------------------
    heat_pairs = os.path.join(args.out_dir, "step3_HEAT_ABLATION__expheat_price_gradient_pairs.csv")
    if not os.path.exists(heat_pairs):
        mech_path = os.path.join(args.out_dir, "STEP3_product_mechanism_proxies.csv")
        if os.path.exists(mech_path) and hasattr(core, "write_heat_ablation_expheat_price_gradient"):
            df0 = _ensure_prod_key(_load_df0_for_step0(any_root, args.out_dir))
            prod_mech = pd.read_csv(mech_path)
            core.write_heat_ablation_expheat_price_gradient(  # type: ignore
                df_rows=df0,
                prod_mech_df=prod_mech,
                out_dir=str(args.out_dir),
                folds_k=folds_k,
                folds_g=folds_g,
                price_col="price",
                supplier_col="supplier",
                prod_key_col="prod_key",
            )
        if not os.path.exists(heat_pairs):
            raise FileNotFoundError(
                "Missing required output file: step3_HEAT_ABLATION__expheat_price_gradient_pairs.csv\n"
                "Please confirm STEP3_product_mechanism_proxies.csv exists and contains the *_struct/*_tele columns."
            )

    # Export paper tables
    if hasattr(core, "export_tables_1_to_11_and_appendices_3_4_5_6"):
        core.export_tables_1_to_11_and_appendices_3_4_5_6(str(args.out_dir), decimals=3)
    else:
        raise RuntimeError("Core missing export_tables_1_to_11_and_appendices_3_4_5_6()")

    print(f"[OK] Reproduce run finished. Outputs at: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()