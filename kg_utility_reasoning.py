# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, math, argparse
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

RANDOM_STATE = 42
N_SPLITS = 5
FE_MIN_LABEL_FREQ_FOR_TABLE = 3
OUTPUT_PATH = './result_kg_reproduce'
EXCEL_PATH = './anymous/name_price_anonymized.xlsx'
MEDIA_PATH = './anymous/media_result.xlsx'

BLOCKS = ["Supplier FE", "Src FE", "App FE", "Product", "Demand", "Supply", "Market"]
BLOCK_SHORT = {
    "Supplier FE": "supplier",
    "Src FE": "src",
    "App FE": "app",
    "Product": "product",
    "Demand": "demand",
    "Supply": "supply",
    "Market": "market",
}


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def write_text(path: str, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def read_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_asset_num(x: Any) -> float:
    if pd.isna(x):
        return 0.0
    s = str(x).replace(',', '').strip()
    try:
        if s.endswith('万'):
            return float(s[:-1])
        if s.endswith('亿'):
            return float(s[:-1]) * 10000.0
        return float(s)
    except Exception:
        return 0.0


def parse_list_cell(x: Any) -> List[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    if not s or s.lower() in {'nan','none','null'}:
        return []
    if s.startswith('[') and s.endswith(']'):
        try:
            import ast
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return [str(t).strip() for t in v if str(t).strip()]
        except Exception:
            pass
    return [t.strip() for t in s.split('|') if t.strip()]


def mlb_transform_no_warn(mlb: MultiLabelBinarizer, seqs) -> np.ndarray:
    classes = list(getattr(mlb, "classes_", []))
    if not classes:
        return np.zeros((len(seqs), 0), dtype=float)
    idx = {c:i for i,c in enumerate(classes)}
    X = np.zeros((len(seqs), len(classes)), dtype=float)
    for r, labs in enumerate(seqs):
        if labs is None:
            continue
        for lab in labs:
            j = idx.get(lab)
            if j is not None:
                X[r, j] = 1.0
    return X

def safe_ln(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return np.nan
    return np.log(x) if x > 0 else np.nan


def metrics_on_ln_and_price(y_ln: np.ndarray, pred_ln: np.ndarray) -> Dict[str, float]:
    y_ln = np.asarray(y_ln, dtype=float)
    pred_ln = np.asarray(pred_ln, dtype=float)
    y = np.exp(y_ln)
    pred = np.exp(pred_ln)
    eps = 1e-12
    return {
        'R2': float(r2_score(y_ln, pred_ln)),
        'MAE': float(mean_absolute_error(y_ln, pred_ln)),
        'RMSE': float(np.sqrt(mean_squared_error(y_ln, pred_ln))),
        'MAPE': float(np.mean(np.abs((y - pred) / np.maximum(np.abs(y), eps)))),
        'SMAPE': float(np.mean(np.abs(y - pred) / np.maximum((np.abs(y) + np.abs(pred)) / 2.0, eps))),
    }


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method='average').values
    yr = pd.Series(y).rank(method='average').values
    if np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


@dataclass
class Encoders:
    supplier_ohe: Optional[OneHotEncoder]
    src_mlb: Optional[MultiLabelBinarizer]
    app_mlb: Optional[MultiLabelBinarizer]
    scalers: Dict[str, StandardScaler]


class DataBuilder:
    def __init__(self, excel_path: str, media_path: str, neo4j_dir: str):
        self.excel_path = excel_path
        self.media_path = media_path
        self.neo4j_dir = neo4j_dir

    def load(self) -> pd.DataFrame:
        price = pd.read_excel(self.excel_path)
        price = price.copy()
        price['price'] = pd.to_numeric(price['price'], errors='coerce')
        price = price[price['price'].gt(0)].copy()
        price['supplier'] = price['supplier'].astype(str)
        price['name'] = price['name'].astype(str)
        price['y_ln'] = price['price'].apply(safe_ln)
        price = price.dropna(subset=['y_ln']).reset_index(drop=True)
        price['Total_Asset_num'] = price.get('Total_Asset', 0).apply(parse_asset_num)

        nodes_dp = pd.read_csv(os.path.join(self.neo4j_dir, 'nodes_dataproduct.csv'))
        rel_app = pd.read_csv(os.path.join(self.neo4j_dir, 'rel_applied_to.csv'))
        rel_src = pd.read_csv(os.path.join(self.neo4j_dir, 'rel_source_industry.csv'))
        nodes_dp = nodes_dp.rename(columns={'name_anon': 'name', 'supplier_anon': 'supplier', 'desc_anon': 'desc'})
        rel_app = rel_app.rename(columns={'app_name': 'app'})
        rel_src = rel_src.rename(columns={'src_name': 'src'})
        dp = nodes_dp[['dp_id', 'name', 'supplier', 'desc']].copy()
        dp['name'] = dp['name'].astype(str)
        dp['supplier'] = dp['supplier'].astype(str)

        app_map = rel_app.groupby('dp_id')['app'].apply(lambda s: sorted({str(v).strip() for v in s.dropna() if str(v).strip()})).to_dict()
        src_map = rel_src.groupby('dp_id')['src'].apply(lambda s: sorted({str(v).strip() for v in s.dropna() if str(v).strip()})).to_dict()
        dp['app_list'] = dp['dp_id'].map(app_map).apply(lambda x: x if isinstance(x, list) else [])
        dp['src_list'] = dp['dp_id'].map(src_map).apply(lambda x: x if isinstance(x, list) else [])
        dp = dp.sort_values('dp_id').drop_duplicates(['name', 'supplier'], keep='first')

        df = price.merge(dp[['name', 'supplier', 'desc', 'app_list', 'src_list']], on=['name', 'supplier'], how='left')
        df['desc'] = df['desc'].fillna('')
        df['app_list'] = df['app_list'].apply(lambda x: x if isinstance(x, list) else [])
        df['src_list'] = df['src_list'].apply(lambda x: x if isinstance(x, list) else [])

        # Precomputed product embeddings and competitor summaries
        emb = pd.read_csv(os.path.join(os.path.dirname(self.excel_path), 'STEP0_product_textemb.csv'))
        emb['name'] = emb['name'].astype(str)
        emb['supplier'] = emb['supplier'].astype(str)
        df = df.merge(emb, on=['name', 'supplier'], how='left')

        comp = pd.read_csv(os.path.join(os.path.dirname(self.excel_path), 'STEP0_product_competitor_summary.csv'))
        comp['name'] = comp['name'].astype(str)
        comp['supplier'] = comp['supplier'].astype(str)
        df = df.merge(comp, on=['name', 'supplier'], how='left', suffixes=('', '_comp'))
        if 'prod_key' not in df.columns:
            df['prod_key'] = 'PROD::' + df['name'] + '||SUP::' + df['supplier']

        # Media heat
        media = pd.read_excel(self.media_path)
        media = media.rename(columns={'keyword': 'app'})
        for c in ['sogou_web_results', 'sina_news_results', 'weixin_article_results']:
            media[c] = pd.to_numeric(media[c], errors='coerce').fillna(0.0)
        media['heat_total'] = np.log1p(media['sogou_web_results'] + media['sina_news_results'] + media['weixin_article_results'])
        media['heat_web'] = np.log1p(media['sogou_web_results'])
        media['heat_news'] = np.log1p(media['sina_news_results'])
        media['heat_weixin'] = np.log1p(media['weixin_article_results'])
        app_heat = media.set_index('app')[['heat_total', 'heat_web', 'heat_news', 'heat_weixin']].to_dict('index')

        def app_heat_agg(apps: List[str], key: str, agg='mean') -> float:
            vals = [float(app_heat.get(a, {}).get(key, 0.0)) for a in apps]
            if not vals:
                return 0.0
            if agg == 'max':
                return float(np.max(vals))
            if agg == 'sum':
                return float(np.sum(vals))
            return float(np.mean(vals))

        for key in ['heat_total', 'heat_web', 'heat_news', 'heat_weixin']:
            df[f'demand_{key}_mean'] = df['app_list'].apply(lambda x: app_heat_agg(x, key, 'mean'))
        df['demand_heat_total_max'] = df['app_list'].apply(lambda x: app_heat_agg(x, 'heat_total', 'max'))
        df['demand_heat_total_sum'] = df['app_list'].apply(lambda x: app_heat_agg(x, 'heat_total', 'sum'))

        # Supply / market counts
        app_edges = rel_app.merge(dp[['dp_id', 'supplier']], on='dp_id', how='left')
        src_edges = rel_src.merge(dp[['dp_id', 'supplier']], on='dp_id', how='left')
        app_supplier_cnt = app_edges.groupby('app')['supplier'].nunique().to_dict()
        src_supplier_cnt = src_edges.groupby('src')['supplier'].nunique().to_dict()
        src_product_cnt = src_edges.groupby('src')['dp_id'].nunique().to_dict()
        app_product_cnt = app_edges.groupby('app')['dp_id'].nunique().to_dict()

        def agg_from_map(labels, mp, agg='mean', default=0.0):
            vals = [float(mp.get(v, default)) for v in labels]
            if not vals:
                return 0.0
            if agg == 'max':
                return float(np.max(vals))
            if agg == 'sum':
                return float(np.sum(vals))
            return float(np.mean(vals))

        df['supply_src_supplier_mean'] = df['src_list'].apply(lambda x: agg_from_map(x, src_supplier_cnt, 'mean'))
        df['supply_src_supplier_max'] = df['src_list'].apply(lambda x: agg_from_map(x, src_supplier_cnt, 'max'))
        df['supply_src_product_mean'] = df['src_list'].apply(lambda x: agg_from_map(x, src_product_cnt, 'mean'))
        df['market_app_supplier_mean'] = df['app_list'].apply(lambda x: agg_from_map(x, app_supplier_cnt, 'mean'))
        df['market_app_product_mean'] = df['app_list'].apply(lambda x: agg_from_map(x, app_product_cnt, 'mean'))
        df['market_hhi_proxy_mean'] = df['app_list'].apply(lambda x: agg_from_map(x, {k: 1.0/max(v,1) for k,v in app_supplier_cnt.items()}, 'mean'))
        df['market_thickness_mean'] = df['app_list'].apply(lambda x: agg_from_map(x, app_product_cnt, 'mean'))

        # Product block columns from precomputed embeddings + competitor summaries
        text_cols = [c for c in df.columns if c.startswith('textemb_')]
        comp_cols = [c for c in ['comp_app_entropy','comp_src_entropy','comp_app_top1_prob','comp_src_top1_prob','comp_expected_heat_total'] if c in df.columns]
        for c in text_cols + comp_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

        # Ensure lists are clean
        df['src_list'] = df['src_list'].apply(lambda x: x if isinstance(x, list) else parse_list_cell(x))
        df['app_list'] = df['app_list'].apply(lambda x: x if isinstance(x, list) else parse_list_cell(x))
        df['topic_id'] = '0'
        self.text_cols = text_cols
        self.comp_cols = comp_cols
        return df


def block_columns(df: pd.DataFrame, builder: Optional[DataBuilder]=None) -> Dict[str, List[str]]:
    text_cols = [c for c in df.columns if c.startswith("textemb_")]
    comp_cols = [c for c in ["comp_app_entropy","comp_src_entropy","comp_app_top1_prob","comp_src_top1_prob","comp_expected_heat_total"] if c in df.columns]
    product_cols = text_cols + comp_cols
    demand_cols = [c for c in df.columns if c.startswith('demand_')]
    supply_cols = [c for c in ['Data_Exclusivity','Is_Gov','Is_Listed','year_create','Year_Dis','Total_Asset_num','supply_src_supplier_mean','supply_src_supplier_max','supply_src_product_mean'] if c in df.columns]
    market_cols = [c for c in ['market_app_supplier_mean','market_app_product_mean','market_hhi_proxy_mean','market_thickness_mean'] if c in df.columns]
    return {'Product': product_cols, 'Demand': demand_cols, 'Supply': supply_cols, 'Market': market_cols}


def fit_block_encoders(train_df: pd.DataFrame, numeric_block_cols: Dict[str, List[str]]) -> Encoders:
    supplier_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    supplier_ohe.fit(train_df[['supplier']].astype(str))
    src_mlb = MultiLabelBinarizer()
    src_mlb.fit(train_df['src_list'])
    app_mlb = MultiLabelBinarizer()
    app_mlb.fit(train_df['app_list'])
    scalers = {}
    for block, cols in numeric_block_cols.items():
        if cols:
            sc = StandardScaler()
            tmp = train_df[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
            sc.fit(tmp)
            scalers[block] = sc
    return Encoders(supplier_ohe, src_mlb, app_mlb, scalers)


def transform_blocks(df: pd.DataFrame, enc: Encoders, numeric_block_cols: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    out['Supplier FE'] = enc.supplier_ohe.transform(df[['supplier']].astype(str)) if enc.supplier_ohe else np.zeros((len(df), 0))
    out['Src FE'] = mlb_transform_no_warn(enc.src_mlb, df['src_list']) if enc.src_mlb else np.zeros((len(df), 0))
    out['App FE'] = mlb_transform_no_warn(enc.app_mlb, df['app_list']) if enc.app_mlb else np.zeros((len(df), 0))
    for block in ['Product','Demand','Supply','Market']:
        cols = numeric_block_cols.get(block, [])
        if cols:
            Xdf = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
            out[block] = enc.scalers[block].transform(Xdf) if block in enc.scalers else Xdf.values
        else:
            out[block] = np.zeros((len(df), 0))
    return out


def join_blocks(block_mats: Dict[str, np.ndarray], chosen: List[str]) -> np.ndarray:
    if not chosen:
        return np.ones((next(iter(block_mats.values())).shape[0], 1))
    mats = [block_mats[b] for b in chosen]
    mats = [m for m in mats if m.ndim == 2 and m.shape[1] > 0]
    if not mats:
        return np.ones((next(iter(block_mats.values())).shape[0], 1))
    return np.concatenate(mats, axis=1)


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, chosen: List[str], numeric_block_cols: Dict[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
    enc = fit_block_encoders(train_df, numeric_block_cols)
    tr_blocks = transform_blocks(train_df, enc, numeric_block_cols)
    te_blocks = transform_blocks(test_df, enc, numeric_block_cols)
    Xtr = join_blocks(tr_blocks, chosen)
    Xte = join_blocks(te_blocks, chosen)
    ytr = train_df['y_ln'].values.astype(float)
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    pred_tr = model.predict(Xtr)
    return pred, pred_tr


def fullsample_metrics(df: pd.DataFrame, chosen: List[str], numeric_block_cols: Dict[str, List[str]]) -> Dict[str, float]:
    pred, _ = fit_predict(df, df, chosen, numeric_block_cols)
    return metrics_on_ln_and_price(df['y_ln'].values, pred)


def cv_metrics(df: pd.DataFrame, splits: List[Tuple[np.ndarray,np.ndarray]], chosen: List[str], numeric_block_cols: Dict[str, List[str]]) -> Dict[str, float]:
    rows = []
    for tr, te in splits:
        train_df = df.iloc[tr].copy(); test_df = df.iloc[te].copy()
        pred, _ = fit_predict(train_df, test_df, chosen, numeric_block_cols)
        rows.append(metrics_on_ln_and_price(test_df['y_ln'].values, pred))
    out = {}
    for k in rows[0].keys():
        vals = [r[k] for r in rows]
        out[k] = float(np.mean(vals))
        out[k+'_SD'] = float(np.std(vals, ddof=0))
    return out


def scenario_subsets(blocks: List[str]) -> List[Tuple[str,...]]:
    out = [tuple()]
    for r in range(1, len(blocks)+1):
        out.extend(combinations(blocks, r))
    return out


def fullsample_r2_for_subset(df: pd.DataFrame, subset: Tuple[str,...], numeric_block_cols: Dict[str,List[str]]) -> float:
    return fullsample_metrics(df, list(subset), numeric_block_cols)['R2']


def shapley_fullsample(df: pd.DataFrame, blocks: List[str], numeric_block_cols: Dict[str,List[str]]) -> pd.DataFrame:
    subsets = scenario_subsets(blocks)
    cache = {s: fullsample_r2_for_subset(df, s, numeric_block_cols) for s in subsets}
    rows = []
    n = len(blocks)
    fac = math.factorial
    for b in blocks:
        phi = 0.0
        others = [x for x in blocks if x != b]
        for r in range(len(others)+1):
            for S in combinations(others, r):
                S = tuple(S)
                Sw = tuple(list(S)+[b])
                w = fac(len(S))*fac(n-len(S)-1)/fac(n)
                phi += w * (cache[tuple(sorted(Sw, key=blocks.index))] - cache[S])
        rows.append({'component': b, 'Shapley': float(phi)})
    return pd.DataFrame(rows)


def build_tables(df: pd.DataFrame, out_dir: str) -> Dict[str,pd.DataFrame]:
    ensure_dir(out_dir)
    nb = block_columns(df)
    print('[build_tables] Table 1', flush=True)
    # Table1
    t1 = pd.DataFrame([
        {'Var':'price','Obs':len(df),'Mean':df['price'].mean(),'SD':df['price'].std(ddof=1),'Min':df['price'].min(),'P25':df['price'].quantile(0.25),'P50':df['price'].quantile(0.5),'P75':df['price'].quantile(0.75),'Max':df['price'].max()},
        {'Var':'ln(price)','Obs':len(df),'Mean':df['y_ln'].mean(),'SD':df['y_ln'].std(ddof=1),'Min':df['y_ln'].min(),'P25':df['y_ln'].quantile(0.25),'P50':df['y_ln'].quantile(0.5),'P75':df['y_ln'].quantile(0.75),'Max':df['y_ln'].max()},
    ])

    # Splits
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    random_splits = [(tr, te) for tr, te in kf.split(df)]
    groups = df['supplier'].astype(str).values
    gkf = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))
    group_splits = [(tr, te) for tr, te in gkf.split(df, df['y_ln'].values, groups)]

    print('[build_tables] Table 2', flush=True)
    # Table2 Panel A independent full-sample fit
    rows = []
    rows.append({'Panel':'A. Independent full-sample fit','Component':'Intercept only','Scenario':'NONE', **fullsample_metrics(df, [], nb)})
    for b in BLOCKS:
        rows.append({'Panel':'A. Independent full-sample fit','Component':b,'Scenario':BLOCK_SHORT[b], **fullsample_metrics(df, [b], nb)})
    cumulative = []
    cum = []
    for b in BLOCKS:
        cum = cum + [b]
        cumulative.append((b, cum.copy()))
    for comp, chosen in cumulative:
        rows.append({'Panel':'B. Conditional additions','Component':comp,'Scenario':' + '.join(chosen), **fullsample_metrics(df, chosen, nb)})
    t2 = pd.DataFrame(rows)

    print('[build_tables] Table 3', flush=True)
    # Table3 full-sample ablation/drop-one/shapley
    rows = []
    for b in BLOCKS:
        rows.append({'Panel':'A. Single-block full-sample fit','Component':b, **fullsample_metrics(df, [b], nb)})
    full_r2 = fullsample_metrics(df, BLOCKS, nb)['R2']
    for b in BLOCKS:
        chosen = [x for x in BLOCKS if x != b]
        met = fullsample_metrics(df, chosen, nb)
        rows.append({'Panel':'B. Drop-one from full model','Component':b, 'R2':met['R2'], 'Delta_R2': full_r2 - met['R2'], 'MAE':met['MAE'], 'RMSE':met['RMSE']})
    shap = shapley_fullsample(df, BLOCKS, nb)
    for _, r in shap.iterrows():
        rows.append({'Panel':'C. Shapley (full sample)','Component':r['component'], 'R2':r['Shapley']})
    t3 = pd.DataFrame(rows)

    print('[build_tables] Table 4', flush=True)
    # Table4 KFold
    rows = []
    rows.append({'Panel':'A. Independent K-fold fit','Component':'Intercept only','Scenario':'NONE', **cv_metrics(df, random_splits, [], nb)})
    for b in BLOCKS:
        rows.append({'Panel':'A. Independent K-fold fit','Component':b,'Scenario':BLOCK_SHORT[b], **cv_metrics(df, random_splits, [b], nb)})
    cum = []
    for b in BLOCKS:
        cum = cum + [b]
        rows.append({'Panel':'B. Conditional additions (K-fold)','Component':b,'Scenario':' + '.join(cum), **cv_metrics(df, random_splits, cum, nb)})
    t4 = pd.DataFrame(rows)

    print('[build_tables] Table 5', flush=True)
    # Table5 Group KFold boundary
    rows = []
    cum = []
    for b in BLOCKS:
        cum = cum + [b]
        met_r = cv_metrics(df, random_splits, cum, nb)
        met_g = cv_metrics(df, group_splits, cum, nb)
        rows.append({'Component':b,'Scenario':' + '.join(cum),'Random K-fold R2':met_r['R2'],'Random K-fold SD':met_r['R2_SD'],'Group K-fold R2':met_g['R2'],'Group K-fold SD':met_g['R2_SD'],'Delta (Group-Random)':met_g['R2']-met_r['R2'],'Group MAE':met_g['MAE'],'Group RMSE':met_g['RMSE']})
    t5 = pd.DataFrame(rows)

    print('[build_tables] Table 6', flush=True)
    # Table6 mechanism proxies across entity levels
    rows = []
    # Supplier entity
    sup = df.groupby('supplier').agg(y_ln=('y_ln','mean'), n_prod=('name','count'), demand=('demand_heat_total_mean','mean'), hhi=('market_hhi_proxy_mean','mean'), thick=('market_thickness_mean','mean')).reset_index()
    for ent, d, cols in [
        ('Supplier', sup, ['n_prod','demand','hhi','thick']),
        ('Src', df.explode('src_list').groupby('src_list').agg(y_ln=('y_ln','mean'), supplier_n=('supplier','nunique'), demand=('demand_heat_total_mean','mean'), prod_n=('name','count')).reset_index(), ['supplier_n','demand','prod_n']),
        ('App', df.explode('app_list').groupby('app_list').agg(y_ln=('y_ln','mean'), supplier_n=('supplier','nunique'), heat=('demand_heat_total_mean','mean'), prod_n=('name','count')).reset_index(), ['supplier_n','heat','prod_n']),
        ('Product', df[['prod_key','y_ln','comp_expected_heat_total','comp_app_entropy','comp_src_entropy','comp_app_top1_prob','comp_src_top1_prob']].drop_duplicates('prod_key'), ['comp_expected_heat_total','comp_app_entropy','comp_src_entropy','comp_app_top1_prob','comp_src_top1_prob']),
    ]:
        d = d.copy().fillna(0)
        if len(d) < 5:
            rows.append({'Entity level': ent, 'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan})
            continue
        splits = [(tr, te) for tr, te in KFold(n_splits=min(5, len(d)), shuffle=True, random_state=RANDOM_STATE).split(d)]
        mets = []
        for tr, te in splits:
            sc = StandardScaler().fit(d.iloc[tr][cols])
            Xtr = sc.transform(d.iloc[tr][cols]); Xte = sc.transform(d.iloc[te][cols])
            ytr = d.iloc[tr]['y_ln'].values; yte = d.iloc[te]['y_ln'].values
            m = Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(Xtr, ytr)
            pred = m.predict(Xte)
            mets.append(metrics_on_ln_and_price(yte, pred))
        rows.append({'Entity level': ent, 'R2': float(np.mean([m['R2'] for m in mets])), 'MAE': float(np.mean([m['MAE'] for m in mets])), 'RMSE': float(np.mean([m['RMSE'] for m in mets]))})
    t6 = pd.DataFrame(rows)

    print('[build_tables] Table 7', flush=True)
    # Table7 structural vs heat-biased robustness
    rows = []
    prod = df[['prod_key','supplier','y_ln','comp_expected_heat_total','demand_heat_total_mean']].drop_duplicates('prod_key').fillna(0)
    if len(prod) > 2:
        r = float(np.corrcoef(prod['comp_expected_heat_total'], prod['demand_heat_total_mean'])[0,1])
        rho = spearman_rho(prod['comp_expected_heat_total'].values, prod['demand_heat_total_mean'].values)
        slope_s = LinearRegression().fit(prod[['comp_expected_heat_total']], prod['y_ln']).coef_[0]
        slope_b = LinearRegression().fit(prod[['demand_heat_total_mean']], prod['y_ln']).coef_[0]
        rows.append({'Entity level':'Product','Consistency (r / ρ)':f'{r:.3f} / {rho:.3f}','Structural assoc. (ρ / slope)':f'{spearman_rho(prod["comp_expected_heat_total"].values, prod["y_ln"].values):.3f} / {slope_s:.3f}','Heat-biased assoc. (ρ / slope)':f'{spearman_rho(prod["demand_heat_total_mean"].values, prod["y_ln"].values):.3f} / {slope_b:.3f}'})
    sup2 = df.groupby('supplier').agg(y_ln=('y_ln','mean'), structural=('comp_expected_heat_total','mean'), biased=('demand_heat_total_mean','mean')).reset_index().fillna(0)
    if len(sup2) > 2:
        r = float(np.corrcoef(sup2['structural'], sup2['biased'])[0,1])
        rho = spearman_rho(sup2['structural'].values, sup2['biased'].values)
        slope_s = LinearRegression().fit(sup2[['structural']], sup2['y_ln']).coef_[0]
        slope_b = LinearRegression().fit(sup2[['biased']], sup2['y_ln']).coef_[0]
        rows.append({'Entity level':'Supplier','Consistency (r / ρ)':f'{r:.3f} / {rho:.3f}','Structural assoc. (ρ / slope)':f'{spearman_rho(sup2["structural"].values, sup2["y_ln"].values):.3f} / {slope_s:.3f}','Heat-biased assoc. (ρ / slope)':f'{spearman_rho(sup2["biased"].values, sup2["y_ln"].values):.3f} / {slope_b:.3f}'})
    t7 = pd.DataFrame(rows)

    return {'Table1':t1,'Table2':t2,'Table3':t3,'Table4':t4,'Table5':t5,'Table6':t6,'Table7':t7}


def style_ws(ws):
    fill = PatternFill(fill_type='solid', fgColor='D9E2F3')
    thin = Side(style='thin', color='BFBFBF')
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.fill = fill
        cell.border = Border(bottom=thin)
    ws.freeze_panes = 'A2'
    for col in ws.columns:
        mx = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(max(mx+2, 10), 42)


def export_tables_1_to_11_and_appendices_1_to_5(out_dir: str, decimals: int = 3, excel_name: str = 'Paper_All_Tables_Merged.xlsx', csv_subdir: str = 'paper_tables_csv') -> str:
    csv_dir = os.path.join(out_dir, 'table_out', csv_subdir)
    ensure_dir(csv_dir)
    excel_path = os.path.join(out_dir, 'table_merge', excel_name)
    ensure_dir(os.path.dirname(excel_path))
    print('[export] write workbook', flush=True)
    tables = read_json(os.path.join(out_dir, 'table_manifest.json'))
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for sheet, csvname in tables.items():
            df = pd.read_csv(os.path.join(csv_dir, csvname))
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].round(decimals)
            df.to_excel(writer, sheet_name=sheet, index=False)
    print('[export] style workbook', flush=True)
    wb = load_workbook(excel_path)
    for ws in wb.worksheets:
        style_ws(ws)
    wb.save(excel_path)
    return excel_path


def run_all_steps(excel_path: str, media_path: str, out_dir: str, neo4j_uri: str, neo4j_user: str, neo4j_pass: str, neo4j_db: str, n_splits: int, random_state: int, min_label_freq: int, heat_ablation: bool=True, ppr_heat_teleport_mass: float=0.35, metapath_heat_teleport_mass: float=0.35) -> None:
    ensure_dir(out_dir)
    global EXCEL_PATH, MEDIA_PATH, RANDOM_STATE, N_SPLITS, FE_MIN_LABEL_FREQ_FOR_TABLE
    EXCEL_PATH, MEDIA_PATH = excel_path, media_path
    RANDOM_STATE, N_SPLITS, FE_MIN_LABEL_FREQ_FOR_TABLE = random_state, n_splits, min_label_freq

    print('\n==============================')
    print('[STEP0] Load data and build dataset')
    print('==============================')
    builder = DataBuilder(excel_path, media_path, neo4j_db)
    df = builder.load()
    df.to_csv(os.path.join(out_dir, 'STEP0_dataset_with_demand_structure.csv'), index=False, encoding='utf-8-sig')
    # stats and splits
    stats = pd.DataFrame([
        {'variable':'price','count':len(df),'mean':df['price'].mean(),'std':df['price'].std(ddof=1),'min':df['price'].min(),'max':df['price'].max()},
        {'variable':'ln(price)','count':len(df),'mean':df['y_ln'].mean(),'std':df['y_ln'].std(ddof=1),'min':df['y_ln'].min(),'max':df['y_ln'].max()},
    ])
    stats.to_csv(os.path.join(out_dir,'STEP0_price_stats.csv'), index=False, encoding='utf-8-sig')
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds_k = [{'train_idx':tr.tolist(),'test_idx':te.tolist()} for tr,te in kf.split(df)]
    save_json(os.path.join(out_dir,'STEP0_splits_kfold.json'), folds_k)
    groups = df['supplier'].astype(str).values
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    folds_g = [{'train_idx':tr.tolist(),'test_idx':te.tolist()} for tr,te in gkf.split(df, df['y_ln'].values, groups)]
    save_json(os.path.join(out_dir,'STEP0_splits_groupkfold_supplier.json'), folds_g)

    print('\n==============================')
    print('[STEP1] Build manuscript tables')
    print('==============================')
    tabs = build_tables(df, out_dir)
    csv_dir = os.path.join(out_dir, 'table_out', 'paper_tables_csv')
    ensure_dir(csv_dir)
    manifest = {}
    for k, v in tabs.items():
        csvname = f'{k}.csv'
        v.to_csv(os.path.join(csv_dir, csvname), index=False, encoding='utf-8-sig')
        manifest[k.replace('Table','Table_')] = csvname
    save_json(os.path.join(out_dir, 'table_manifest.json'), manifest)
    export_tables_1_to_11_and_appendices_1_to_5(out_dir)
    print('[OK] all tables generated')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--excel', default=EXCEL_PATH)
    ap.add_argument('--media', default=MEDIA_PATH)
    ap.add_argument('--out', default=OUTPUT_PATH)
    ap.add_argument('--neo4j_db', default='./anymous/neo4j_export')
    ap.add_argument('--n_splits', type=int, default=N_SPLITS)
    ap.add_argument('--seed', type=int, default=RANDOM_STATE)
    ap.add_argument('--min_label_freq', type=int, default=FE_MIN_LABEL_FREQ_FOR_TABLE)
    args = ap.parse_args()
    run_all_steps(args.excel, args.media, args.out, '', '', '', args.neo4j_db, args.n_splits, args.seed, args.min_label_freq)

if __name__ == '__main__':
    main()
