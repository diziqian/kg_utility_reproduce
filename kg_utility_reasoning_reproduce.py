#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, math, argparse, importlib.util, subprocess, sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def find_anymous_root(d: str) -> str:
    d = os.path.abspath(d)
    if os.path.exists(os.path.join(d, 'name_price_anonymized.xlsx')):
        return d
    cand = os.path.join(d, 'anymous')
    if os.path.exists(os.path.join(cand, 'name_price_anonymized.xlsx')):
        return cand
    for cur, _, files in os.walk(d):
        if 'name_price_anonymized.xlsx' in files and 'media_result.xlsx' in files and os.path.isdir(os.path.join(cur, 'neo4j_export')):
            return cur
    raise FileNotFoundError(f'Cannot locate valid data root under: {d}')


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def style_ws(ws):
    fill = PatternFill(fill_type='solid', fgColor='D9E2F3')
    thin = Side(style='thin', color='BFBFBF')
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.fill = fill
        cell.border = Border(bottom=thin)
    ws.freeze_panes = 'A2'
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            if isinstance(cell.value, float):
                cell.number_format = '0.000'
    for col in ws.columns:
        mx = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(max(mx+2, 10), 42)


def parse_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
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


def fmt3(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(3)
    return out


def build_appendix_tables(any_root: str, out_dir: str) -> Dict[str, str]:
    neo_dir = os.path.join(any_root, 'neo4j_export')
    nodes_dp = pd.read_csv(os.path.join(neo_dir, 'nodes_dataproduct.csv')).rename(columns={'name_anon':'name','supplier_anon':'supplier','desc_anon':'desc'})
    rel_app = pd.read_csv(os.path.join(neo_dir, 'rel_applied_to.csv')).rename(columns={'app_name':'app'})
    rel_src = pd.read_csv(os.path.join(neo_dir, 'rel_source_industry.csv')).rename(columns={'src_name':'src'})
    price = pd.read_excel(os.path.join(any_root, 'name_price_anonymized.xlsx')).copy()
    price['price'] = pd.to_numeric(price['price'], errors='coerce')
    price = price[price['price'].gt(0)].copy()
    price['name'] = price['name'].astype(str)
    price['supplier'] = price['supplier'].astype(str)
    price = price.drop_duplicates(['name','supplier'])
    dp = nodes_dp[['dp_id','name','supplier']].copy()
    dp['name'] = dp['name'].astype(str)
    dp['supplier'] = dp['supplier'].astype(str)
    priced_dp = price.merge(dp[['dp_id','name','supplier']], on=['name','supplier'], how='inner')
    priced_ids = set(priced_dp['dp_id'].astype(str))

    # Appendix C1
    def degree_stats(start_ids, end_ids, edges_df, start_col, end_col):
        outdeg = edges_df.groupby(start_col)[end_col].count().reindex(start_ids, fill_value=0)
        indeg = edges_df.groupby(end_col)[start_col].count().reindex(end_ids, fill_value=0)
        def pack(s):
            s = pd.Series(s, dtype=float)
            return f"{s.mean():.3f}; {int(s.quantile(0.25))}/{int(s.quantile(0.5))}/{int(s.quantile(0.75))}; {int(s.max())}"
        return pack(outdeg), pack(indeg)

    provide = priced_dp[['supplier','dp_id']].rename(columns={'supplier':'start','dp_id':'end'})
    src_edges = rel_src[rel_src['dp_id'].astype(str).isin(priced_ids)][['src','dp_id']].rename(columns={'src':'start','dp_id':'end'})
    app_edges = rel_app[rel_app['dp_id'].astype(str).isin(priced_ids)][['dp_id','app']].rename(columns={'dp_id':'start','app':'end'})

    rows_c1 = []
    for rel_name, schema, edf, start_set, end_set in [
        ('provide_data','Supplier → DataProduct', provide, sorted(provide['start'].astype(str).unique()), sorted(provide['end'].astype(str).unique())),
        ('source_industry','src_IndustryCategory → DataProduct', src_edges, sorted(src_edges['start'].astype(str).unique()), sorted(src_edges['end'].astype(str).unique())),
        ('applied_to','DataProduct → app_IndustryCategory', app_edges, sorted(app_edges['start'].astype(str).unique()), sorted(app_edges['end'].astype(str).unique())),
    ]:
        sdeg, edeg = degree_stats(start_set, end_set, edf, 'start', 'end')
        rows_c1.append({
            'Relation (r)': rel_name,
            'Schema (Start → End)': schema,
            '#Edges': len(edf),
            '#Start': len(start_set),
            '#End': len(end_set),
            'Start out-degree (mean; P25/P50/P75; max)': sdeg,
            'End in-degree (mean; P25/P50/P75; max)': edeg,
        })
    c1 = pd.DataFrame(rows_c1)

    # Appendix C2
    app_counts = rel_app[rel_app['dp_id'].astype(str).isin(priced_ids)].groupby('dp_id').size()
    src_counts = rel_src[rel_src['dp_id'].astype(str).isin(priced_ids)].groupby('dp_id').size()
    prod_per_supplier = priced_dp.groupby('supplier').size()
    def pack_stat(s):
        s = pd.Series(s, dtype=float)
        return f"{s.mean():.3f}; {int(s.quantile(0.25))}/{int(s.quantile(0.5))}/{int(s.quantile(0.75))}; {int(s.max())}"
    c2 = pd.DataFrame([
        {'Panel':'A. Node coverage','Statistic':'Products (with price)','Value':len(priced_ids)},
        {'Panel':'A. Node coverage','Statistic':'Suppliers (connected to priced products)','Value':priced_dp['supplier'].nunique()},
        {'Panel':'A. Node coverage','Statistic':'src industries (connected to priced products)','Value':src_edges['start'].nunique()},
        {'Panel':'A. Node coverage','Statistic':'app industries (connected to priced products)','Value':app_edges['end'].nunique()},
        {'Panel':'B. Edge coverage','Statistic':'Triples: provide_data','Value':len(provide)},
        {'Panel':'B. Edge coverage','Statistic':'Triples: source_industry','Value':len(src_edges)},
        {'Panel':'B. Edge coverage','Statistic':'Triples: applied_to','Value':len(app_edges)},
        {'Panel':'C. Degree & label sparsity','Statistic':'Products per supplier (mean; P25/P50/P75; max)','Value':pack_stat(prod_per_supplier)},
        {'Panel':'C. Degree & label sparsity','Statistic':'App labels per product (mean; P25/P50/P75; max)','Value':pack_stat(app_counts.reindex(sorted(priced_ids), fill_value=0))},
        {'Panel':'C. Degree & label sparsity','Statistic':'Src labels per product (mean; P25/P50/P75; max)','Value':pack_stat(src_counts.reindex(sorted(priced_ids), fill_value=0))},
    ])

    # Appendix D1 from anymous data only
    media = pd.read_excel(os.path.join(any_root, 'media_result.xlsx')).rename(columns={'keyword':'app'})
    for c in ['sogou_web_results','sina_news_results','weixin_article_results']:
        media[c] = pd.to_numeric(media[c], errors='coerce').fillna(0.0)
    media['heat_web'] = np.log1p(media['sogou_web_results'])
    media['heat_news'] = np.log1p(media['sina_news_results'])
    media['heat_weixin'] = np.log1p(media['weixin_article_results'])
    media['heat_total'] = media['heat_web'] + media['heat_news'] + media['heat_weixin']
    heat_map = media.set_index('app')[['heat_web','heat_news','heat_weixin','heat_total']].to_dict('index')

    # build priced dataset with app labels
    rel_app2 = rel_app[rel_app['dp_id'].astype(str).isin(priced_ids)][['dp_id','app']].copy()
    rel_src2 = rel_src[rel_src['dp_id'].astype(str).isin(priced_ids)][['dp_id','src']].copy()
    app_map = rel_app2.groupby('dp_id')['app'].apply(list).to_dict()
    src_map = rel_src2.groupby('dp_id')['src'].apply(list).to_dict()
    d = priced_dp[['dp_id','supplier']].copy()
    d['app_list'] = d['dp_id'].map(app_map).apply(lambda x: x if isinstance(x,list) else [])
    d['src_list'] = d['dp_id'].map(src_map).apply(lambda x: x if isinstance(x,list) else [])

    # supplier counts on full listing graph including unpriced
    full_dp = nodes_dp[['dp_id','supplier']].copy()
    full_app_edges = rel_app.merge(full_dp[['dp_id','supplier']], on='dp_id', how='left')
    app_supplier_cnt = full_app_edges.groupby('app')['supplier'].nunique().to_dict()

    def mean_across(labels, mp):
        vals = [float(mp.get(v, 0.0)) for v in labels]
        return float(np.mean(vals)) if vals else 0.0
    def mean_heat(labels, key):
        vals = [float(heat_map.get(v, {}).get(key, 0.0)) for v in labels]
        return float(np.mean(vals)) if vals else 0.0

    d['heat_news'] = d['app_list'].apply(lambda x: mean_heat(x, 'heat_news'))
    d['heat_total'] = d['app_list'].apply(lambda x: mean_heat(x, 'heat_total'))
    d['heat_web'] = d['app_list'].apply(lambda x: mean_heat(x, 'heat_web'))
    d['heat_weixin'] = d['app_list'].apply(lambda x: mean_heat(x, 'heat_weixin'))
    nmarket = d['app_list'].apply(lambda x: mean_across(x, app_supplier_cnt))
    d['HHI_proxy'] = 1.0 / np.maximum(nmarket, 1.0)
    d['Bs'] = np.log1p(nmarket)
    d['T'] = np.log1p(nmarket * np.maximum(np.expm1(d['heat_total']), 0.0))

    def desc_row(var):
        s = pd.to_numeric(d[var], errors='coerce')
        return {'Var':var,'Obs':int(s.notna().sum()),'Mean':s.mean(),'SD':s.std(ddof=1),'Min':s.min(),'P25':s.quantile(0.25),'P50':s.quantile(0.5),'P75':s.quantile(0.75),'Max':s.max()}
    d1 = pd.DataFrame([desc_row(v) for v in ['heat_news','heat_total','heat_web','heat_weixin','Bs','HHI_proxy','T']])
    d1 = fmt3(d1)

    csv_dir = os.path.join(out_dir, 'table_out', 'paper_tables_csv')
    ensure_dir(csv_dir)
    c1_path = os.path.join(csv_dir, 'Appendix_C1.csv'); fmt3(c1).to_csv(c1_path, index=False)
    c2_path = os.path.join(csv_dir, 'Appendix_C2.csv'); c2.to_csv(c2_path, index=False)
    d1_path = os.path.join(csv_dir, 'Appendix_D1.csv'); d1.to_csv(d1_path, index=False)
    return {'App_Table_C1':'Appendix_C1.csv','App_Table_C2':'Appendix_C2.csv','App_Table_D1':'Appendix_D1.csv'}


def merge_excel(out_dir: str, appendix_manifest: Dict[str, str]) -> str:
    csv_dir = os.path.join(out_dir, 'table_out', 'paper_tables_csv')
    manifest = json.load(open(os.path.join(out_dir, 'table_manifest.json'), 'r', encoding='utf-8'))
    for k, v in appendix_manifest.items():
        manifest[k] = v
    merged_dir = os.path.join(out_dir, 'table_merge')
    ensure_dir(merged_dir)
    merged_path = os.path.join(merged_dir, 'Paper_All_Tables_Merged.xlsx')
    with pd.ExcelWriter(merged_path, engine='openpyxl') as writer:
        for sheet, csvname in manifest.items():
            df = pd.read_csv(os.path.join(csv_dir, csvname))
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].round(3)
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    wb = load_workbook(merged_path)
    for ws in wb.worksheets:
        style_ws(ws)
    wb.save(merged_path)
    return merged_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='./anymous')
    ap.add_argument('--core', default='./kg_utility_reasoning.py')
    ap.add_argument('--fig_script', default='./plt_kg_reasoning_pic.py')
    ap.add_argument('--out_dir', default='./result_kg_reproduce')
    ap.add_argument('--min_label_freq', type=int, default=3)
    args = ap.parse_args()

    root = find_anymous_root(args.input)
    print(f'>>> Root data directory determined as: {root}')
    price_xlsx = os.path.join(root, 'name_price_anonymized.xlsx')
    media_xlsx = os.path.join(root, 'media_result.xlsx')
    neo_dir = os.path.join(root, 'neo4j_export')

    core = load_module('kg_core_fixed', args.core)
    print('>>> Loading core reproduction script...')
    core.run_all_steps(
        excel_path=price_xlsx,
        media_path=media_xlsx,
        out_dir=args.out_dir,
        neo4j_uri='', neo4j_user='', neo4j_pass='', neo4j_db=neo_dir,
        n_splits=5,
        random_state=42,
        min_label_freq=args.min_label_freq,
        heat_ablation=True,
        ppr_heat_teleport_mass=0.35,
        metapath_heat_teleport_mass=0.35,
    )
    print('>>> Reproducing Fig. 6...')
    subprocess.run([sys.executable, args.fig_script, '--out_dir', args.out_dir, '--fig_dir', os.path.join(args.out_dir, 'KG_reasoning_pic_6')], check=True)
    print('>>> Building Appendix C/D tables from anymous data...')
    app_manifest = build_appendix_tables(root, args.out_dir)
    print('>>> Merging workbook...')
    merged = merge_excel(args.out_dir, app_manifest)
    print(f'[Complete] {merged}')

if __name__ == '__main__':
    main()
