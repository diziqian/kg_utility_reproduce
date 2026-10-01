#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, math, argparse, importlib.util, subprocess, sys, shutil, hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


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


# ---------------------------------------------------------------------------
# Manuscript-facing table output structure
# ---------------------------------------------------------------------------
# The generated Excel workbooks intentionally contain TABLE CONTENT ONLY.
# Table captions/titles and manuscript Notes are not embedded because their
# wording may continue to be refined during revision. Panel headings are kept
# where they are part of the internal table structure.
MAIN_TABLE_SHEETS = [
    'Table_1', 'Table_2', 'Table_3', 'Table_4',
    'Table_5', 'Table_6', 'Table_7',
]

APPENDIX_TABLE_SHEETS = [
    'Appendix_Table_C1', 'Appendix_Table_C2', 'Appendix_Table_C3',
    'Appendix_Table_C4', 'Appendix_Table_D1',
    'Appendix_Table_E1', 'Appendix_Table_E2', 'Appendix_Table_E3',
]

def _cell_is_number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def _is_panel_row(ws, row_idx: int, first_data_col: int = 1) -> bool:
    v = ws.cell(row_idx, first_data_col).value
    if not (isinstance(v, str) and v.startswith('Panel ')):
        return False
    # A manuscript panel heading is represented in the CSV by text in the first
    # column and blank cells thereafter.
    for c in range(first_data_col + 1, ws.max_column + 1):
        if ws.cell(row_idx, c).value not in (None, ''):
            return False
    return True


def _column_is_text_heavy(header: str) -> bool:
    h = str(header or '').lower()
    keys = [
        'component', 'scenario', 'specification', 'characteristic', 'sample category',
        'stage', 'audit/source', 'schema', 'statistic', 'variable', 'definition',
        'interpretation', 'features', 'relation', 'method', 'entity level',
    ]
    return any(k in h for k in keys)


def _apply_word_like_sheet_style(
    ws,
    *,
    kind: str,
    header_row: int,
    first_data_row: int,
    last_data_row: int,
):
    """
    Match the current Word tables as closely as practical in Excel, while
    keeping the workbook free of provisional table captions and Notes:
      - Times New Roman, 10 pt;
      - white background and hidden worksheet gridlines;
      - no vertical rules;
      - thin top/bottom rules around the column header;
      - thin bottom rule at the end of each table/panel block;
      - bold panel headings; main-text panels use the light-gray panel cue that
        appears in the manuscript, Appendix panels remain white;
      - text columns left aligned, numerical columns centered;
      - wrapped text and manuscript-scale row heights.

    Table captions/titles and manuscript Notes are deliberately NOT written.
    They should be added only after their wording is finalized in the paper.
    """
    thin = Side(style='thin', color='000000')
    white = PatternFill(fill_type='solid', fgColor='FFFFFF')
    panel_fill_main = PatternFill(fill_type='solid', fgColor='F2F2F2')
    no_border = Border()

    ws.sheet_view.showGridLines = False
    ws.freeze_panes = ws.cell(header_row + 1, 1).coordinate

    # Base style for used region.
    for row in ws.iter_rows(min_row=header_row, max_row=last_data_row):
        for cell in row:
            cell.fill = white
            cell.font = Font(name='Times New Roman', size=10, color='000000')
            cell.border = no_border
            cell.alignment = Alignment(vertical='center', wrap_text=True)

    # Header.
    for cell in ws[header_row]:
        cell.font = Font(name='Times New Roman', size=10, bold=True, color='000000')
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = Border(top=thin, bottom=thin)
        cell.fill = white
    ws.row_dimensions[header_row].height = 30

    # Body + panel rows.
    for r in range(first_data_row, last_data_row + 1):
        if _is_panel_row(ws, r):
            if ws.max_column > 1:
                txt = ws.cell(r, 1).value
                for c in range(2, ws.max_column + 1):
                    ws.cell(r, c).value = None
                ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=ws.max_column)
                ws.cell(r, 1).value = txt
            pc = ws.cell(r, 1)
            pc.font = Font(name='Times New Roman', size=10, bold=True, italic=True, color='000000')
            pc.alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
            pc.fill = panel_fill_main if kind == 'main' else white
            pc.border = Border(bottom=thin)
            ws.row_dimensions[r].height = 21
            continue

        for c in range(1, ws.max_column + 1):
            cell = ws.cell(r, c)
            header = ws.cell(header_row, c).value
            if c == 1 or _column_is_text_heavy(header):
                cell.alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
            else:
                cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

            if _cell_is_number(cell.value):
                h = str(header or '').lower()
                if 'share' in h:
                    cell.number_format = '0.0%'
                elif abs(float(cell.value) - round(float(cell.value))) < 1e-12 and ('n' == h.strip() or 'count' in h or 'obs' in h):
                    cell.number_format = '0'
                else:
                    cell.number_format = '0.000'
        ws.row_dimensions[r].height = 20

    # Bottom rule, matching Word's table-ending horizontal rule.
    if last_data_row >= first_data_row:
        for c in range(1, ws.max_column + 1):
            ws.cell(last_data_row, c).border = Border(bottom=thin)

    # Word-like column widths. Text-heavy columns are deliberately wider.
    for c in range(1, ws.max_column + 1):
        header = ws.cell(header_row, c).value
        max_len = 0
        for r in range(header_row, last_data_row + 1):
            v = ws.cell(r, c).value
            if v is not None:
                max_len = max(max_len, len(str(v)))
        if _column_is_text_heavy(header) or c == 1:
            width = min(max(max_len * 0.72 + 2, 18), 48 if kind == 'main' else 58)
        else:
            width = min(max(max_len + 2, 10), 20)
        ws.column_dimensions[get_column_letter(c)].width = width

    # Print layout: wide tables are landscape, otherwise portrait.
    ws.page_setup.orientation = 'landscape' if ws.max_column >= 6 else 'portrait'
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_options.horizontalCentered = True
    ws.page_margins.left = 0.3
    ws.page_margins.right = 0.3
    ws.page_margins.top = 0.5
    ws.page_margins.bottom = 0.5

def style_ws(ws, kind: str = 'main', title: Optional[str] = None):
    """
    Backward-compatible style entry point.

    For manuscript workbooks generated by write_manuscript_workbook(), the richer
    _apply_word_like_sheet_style() path is used. This function remains available
    for any legacy caller.
    """
    thin = Side(style='thin', color='000000')
    white = PatternFill(fill_type='solid', fgColor='FFFFFF')
    ws.sheet_view.showGridLines = False

    for row in ws.iter_rows():
        for cell in row:
            cell.fill = white
            cell.font = Font(name='Times New Roman', size=10, color='000000')
            cell.alignment = Alignment(vertical='center', wrap_text=True)
            cell.border = Border()

    if ws.max_row >= 1:
        for cell in ws[1]:
            cell.font = Font(name='Times New Roman', size=10, bold=True, color='000000')
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.border = Border(top=thin, bottom=thin)

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(
                horizontal='center' if cell.column > 1 else 'left',
                vertical='center',
                wrap_text=True,
            )
            if isinstance(cell.value, float):
                cell.number_format = '0.000'

    if ws.max_row >= 2:
        for cell in ws[ws.max_row]:
            cell.border = Border(bottom=thin)

    ws.freeze_panes = 'A2'
    for col in ws.columns:
        mx = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(max(mx + 2, 10), 48)

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

    # Appendix C1: auditable KG universe and price-modeling sample construction
    raw_price_path = os.path.join(any_root, 'name_price_all_anonymized.xlsx')
    if not os.path.exists(raw_price_path):
        raise FileNotFoundError(
            'Appendix C1 requires raw price file name_price.xlsx under the anonymized root. '
            'This file is needed to reproduce raw rows, zero-price records, and upper-tail exclusions.'
        )

    raw_price = pd.read_excel(raw_price_path).copy()
    raw_price['price'] = pd.to_numeric(raw_price['price'], errors='coerce')
    raw_price['name'] = raw_price['name'].astype(str).str.strip()
    raw_price['supplier'] = raw_price['supplier'].astype(str).str.strip()

    kg_total_products = int(nodes_dp['dp_id'].astype(str).nunique()) if 'dp_id' in nodes_dp.columns else int(nodes_dp[['name','supplier']].drop_duplicates().shape[0])
    raw_rows = int(len(raw_price))
    raw_unique_pairs = int(raw_price[['name','supplier']].drop_duplicates().shape[0])
    zero_price_records = int(raw_price['price'].eq(0).sum())
    upper_tail_removed = int(raw_price['price'].ge(300).sum())
    final_model_sample = int(price[['name','supplier']].drop_duplicates().shape[0])

    # price_dp and price_ids
    priced_dp = price.merge(dp[['dp_id','name','supplier']], on=['name','supplier'], how='inner')
    priced_ids = set(priced_dp['dp_id'].astype(str))

    # Appendix C1 content
    final_matched_to_kg = int(priced_dp[['name','supplier']].drop_duplicates().shape[0])

    c1 = pd.DataFrame([
        {
            'Stage': 'Broader KG API-listing universe',
            'Count': kg_total_products,
            'Audit/source basis': 'Unique API listings retained after cleaning and de-duplication for KG construction.'
        },
        {
            'Stage': 'Raw price-file rows',
            'Count': raw_rows,
            'Audit/source basis': 'Rows in the raw price file used for price cleaning.'
        },
        {
            'Stage': 'Unique product-supplier pairs in raw price file',
            'Count': raw_unique_pairs,
            'Audit/source basis': 'Product name and supplier name are used as the uniqueness criterion.'
        },
        {
            'Stage': 'Zero-price listings removed',
            'Count': zero_price_records,
            'Audit/source basis': 'Records with price = 0; excluded because they do not represent positive paid API posted quotes.'
        },
        {
            'Stage': 'Upper-tail listings removed',
            'Count': upper_tail_removed,
            'Audit/source basis': 'Records priced at 300 RMB per call or above.'
        },
        {
            'Stage': 'Final price-modeling sample',
            'Count': final_model_sample,
            'Audit/source basis': 'Unique API products with positive normalized posted prices after cleaning.'
        },
        {
            'Stage': 'Final sample matched back to KG',
            'Count': final_matched_to_kg,
            'Audit/source basis': 'Final modeling records matched to the KG through anonymized product and supplier labels.'
        },
    ])

    # Appendix C2
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

    rows_c2 = []
    for rel_name, schema, edf, start_set, end_set in [
        ('provide_data','Supplier → DataProduct', provide, sorted(provide['start'].astype(str).unique()), sorted(provide['end'].astype(str).unique())),
        ('source_industry','src_IndustryCategory → DataProduct', src_edges, sorted(src_edges['start'].astype(str).unique()), sorted(src_edges['end'].astype(str).unique())),
        ('applied_to','DataProduct → app_IndustryCategory', app_edges, sorted(app_edges['start'].astype(str).unique()), sorted(app_edges['end'].astype(str).unique())),
    ]:
        sdeg, edeg = degree_stats(start_set, end_set, edf, 'start', 'end')
        rows_c2.append({
            'Relation (r)': rel_name,
            'Schema (Start → End)': schema,
            '#Edges': len(edf),
            '#Start': len(start_set),
            '#End': len(end_set),
            'Start out-degree (mean; P25/P50/P75; max)': sdeg,
            'End in-degree (mean; P25/P50/P75; max)': edeg,
        })
    c2 = pd.DataFrame(rows_c2)

    # Appendix C3
    app_counts = rel_app[rel_app['dp_id'].astype(str).isin(priced_ids)].groupby('dp_id').size()
    src_counts = rel_src[rel_src['dp_id'].astype(str).isin(priced_ids)].groupby('dp_id').size()
    prod_per_supplier = priced_dp.groupby('supplier').size()
    def pack_stat(s):
        s = pd.Series(s, dtype=float)
        return f"{s.mean():.3f}; {int(s.quantile(0.25))}/{int(s.quantile(0.5))}/{int(s.quantile(0.75))}; {int(s.max())}"
    c3 = pd.DataFrame([
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

    # Appendix D1 from the exact STEP0 dataset generated by the core program.
    # This guarantees that Appendix D describes the same paper-facing variables
    # created by the core script. Demand and market variables follow the manuscript/Appendix D definitions.
    step0_path = os.path.join(out_dir, 'STEP0_dataset_with_demand_structure.csv')
    if not os.path.exists(step0_path):
        raise FileNotFoundError(f'Cannot find core STEP0 dataset for Appendix D: {step0_path}')
    d = pd.read_csv(step0_path)

    appendix_d_map = {
        'heat_news': 'heat_news',
        'heat_total': 'heat_total',
        'heat_web': 'heat_web',
        'heat_weixin': 'heat_weixin',
        'Bs': 'Bs',
        'HHI_proxy': 'HHI_proxy',
        'T': 'T',
    }
    missing = [col for col in appendix_d_map.values() if col not in d.columns]
    if missing:
        raise KeyError(f'Missing expected STEP0 variables for Appendix D: {missing}')

    def desc_row(label, col):
        s = pd.to_numeric(d[col], errors='coerce')
        return {
            'Var': label,
            'Obs': int(s.notna().sum()),
            'Mean': s.mean(),
            'SD': s.std(ddof=1),
            'Min': s.min(),
            'P25': s.quantile(0.25),
            'P50': s.quantile(0.5),
            'P75': s.quantile(0.75),
            'Max': s.max(),
        }
    d1 = pd.DataFrame([desc_row(label, col) for label, col in appendix_d_map.items()])
    d1 = fmt3(d1)

    csv_dir = os.path.join(out_dir, 'table_out', 'paper_tables_csv')
    ensure_dir(csv_dir)

    c1_path = os.path.join(csv_dir, 'Appendix_C1.csv')
    c1.to_csv(c1_path, index=False, encoding='utf-8-sig')
    c2_path = os.path.join(csv_dir, 'Appendix_C2.csv');
    fmt3(c2).to_csv(c2_path, index=False)
    c3_path = os.path.join(csv_dir, 'Appendix_C3.csv');
    c3.to_csv(c3_path, index=False)
    d1_path = os.path.join(csv_dir, 'Appendix_D1.csv');
    d1.to_csv(d1_path, index=False)
    return {
        'App_Table_C1':'Appendix_C1.csv',
        'App_Table_C2':'Appendix_C2.csv',
        'App_Table_C3':'Appendix_C3.csv',
        'App_Table_D1':'Appendix_D1.csv'
    }



def _build_manifest(csv_dir: str) -> Dict[str, str]:
    lst_csv = sorted(os.listdir(csv_dir))
    manifest: Dict[str, str] = {}
    for c in lst_csv:
        if not c.endswith('.csv'):
            continue
        if c.startswith('Appendix_C4_PanelA_') or c.startswith('Appendix_C4_PanelB_'):
            # C4 helper files are used by the special C4 worksheet writer below,
            # not treated as standalone manuscript tables.
            continue
        if c.startswith('Appendix'):
            key = c.replace('Appendix_', 'Appendix_Table_')[:-4]
        else:
            key = c.replace('Table', 'Table_', 1)[:-4]
        manifest[key] = c
    return manifest


def _write_regular_sheet(writer, sheet_name: str, csv_path: str, kind: str):
    """Write a clean manuscript-style table sheet with no embedded caption or Notes."""
    df = pd.read_csv(csv_path)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].round(3)

    # Row 1 = column header; row 2+ = table body.
    # No table title/caption and no Notes are inserted into the workbook.
    df.to_excel(writer, sheet_name=sheet_name[:31], index=False, startrow=0)
    ws = writer.book[sheet_name[:31]]

    last_data_row = 1 + len(df)
    _apply_word_like_sheet_style(
        ws,
        kind=kind,
        header_row=1,
        first_data_row=2,
        last_data_row=last_data_row,
    )

def _write_appendix_c4_sheet(writer, csv_dir: str):
    """
    Write Appendix C.4 as two visually separate panels. The sheet deliberately
    contains no provisional Appendix-table caption and no manuscript Notes.
    Panel headings are retained because they define the table's internal structure.
    """
    a_path = os.path.join(csv_dir, 'Appendix_C4_PanelA_sample_counts.csv')
    b_path = os.path.join(csv_dir, 'Appendix_C4_PanelB_observable_characteristics.csv')
    if not os.path.exists(a_path) or not os.path.exists(b_path):
        raise FileNotFoundError(
            'Styled Appendix C.4 requires Appendix_C4_PanelA_sample_counts.csv and '
            'Appendix_C4_PanelB_observable_characteristics.csv.'
        )

    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)
    for d in (a, b):
        for c in d.columns:
            if pd.api.types.is_numeric_dtype(d[c]):
                d[c] = d[c].round(3)

    sheet_name = 'Appendix_Table_C4'
    ws = writer.book.create_sheet(sheet_name[:31])
    max_cols = max(len(a.columns), len(b.columns))

    # Panel A starts immediately at row 1.
    panel_a_title_row = 1
    a_header_row = 2
    a_start_data = 3
    ws.merge_cells(start_row=panel_a_title_row, start_column=1, end_row=panel_a_title_row, end_column=max_cols)
    ws.cell(panel_a_title_row, 1).value = 'Panel A. Posted-price availability and analytical-sample construction'
    for j, col in enumerate(a.columns, 1):
        ws.cell(a_header_row, j).value = col
    for i, row in enumerate(a.itertuples(index=False, name=None), a_start_data):
        for j, v in enumerate(row, 1):
            ws.cell(i, j).value = None if (isinstance(v, float) and np.isnan(v)) else v
    a_last = a_start_data + len(a) - 1

    # Panel B follows after one blank row.
    panel_b_title_row = a_last + 2
    b_header_row = panel_b_title_row + 1
    b_start_data = b_header_row + 1
    ws.merge_cells(start_row=panel_b_title_row, start_column=1, end_row=panel_b_title_row, end_column=max_cols)
    ws.cell(panel_b_title_row, 1).value = 'Panel B. Observable characteristics of the final price-modeling sample and other KG listings'
    for j, col in enumerate(b.columns, 1):
        ws.cell(b_header_row, j).value = col
    for i, row in enumerate(b.itertuples(index=False, name=None), b_start_data):
        for j, v in enumerate(row, 1):
            ws.cell(i, j).value = None if (isinstance(v, float) and np.isnan(v)) else v
    b_last = b_start_data + len(b) - 1

    # Manual Word-like styling because the sheet contains two independent headers.
    thin = Side(style='thin', color='000000')
    white = PatternFill(fill_type='solid', fgColor='FFFFFF')
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = 'A3'

    for pr in (panel_a_title_row, panel_b_title_row):
        c = ws.cell(pr, 1)
        c.font = Font(name='Times New Roman', size=10, bold=True, italic=True)
        c.alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
        c.fill = white
        c.border = Border(bottom=thin)
        ws.row_dimensions[pr].height = 21

    for hr in (a_header_row, b_header_row):
        for c in range(1, max_cols + 1):
            cell = ws.cell(hr, c)
            cell.fill = white
            cell.font = Font(name='Times New Roman', size=10, bold=True)
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.border = Border(top=thin, bottom=thin)
        ws.row_dimensions[hr].height = 30

    # Panel A body.
    for r in range(a_start_data, a_last + 1):
        for c in range(1, max_cols + 1):
            cell = ws.cell(r, c)
            cell.fill = white
            cell.font = Font(name='Times New Roman', size=10)
            cell.border = Border()
            cell.alignment = Alignment(
                horizontal='left' if c == 1 else 'center',
                vertical='center',
                wrap_text=True,
            )
            if _cell_is_number(cell.value):
                header = str(ws.cell(a_header_row, c).value or '').lower()
                if 'share' in header:
                    cell.number_format = '0.0%'
                elif header.strip() == 'n':
                    cell.number_format = '0'
                else:
                    cell.number_format = '0.000'
        ws.row_dimensions[r].height = 20
    for c in range(1, max_cols + 1):
        ws.cell(a_last, c).border = Border(bottom=thin)

    # Panel B body.
    for r in range(b_start_data, b_last + 1):
        for c in range(1, max_cols + 1):
            cell = ws.cell(r, c)
            cell.fill = white
            cell.font = Font(name='Times New Roman', size=10)
            cell.border = Border()
            cell.alignment = Alignment(
                horizontal='left' if c == 1 else 'center',
                vertical='center',
                wrap_text=True,
            )
            if _cell_is_number(cell.value):
                cell.number_format = '0.000'
        ws.row_dimensions[r].height = 20
    for c in range(1, max_cols + 1):
        ws.cell(b_last, c).border = Border(bottom=thin)

    # Widths chosen to resemble the current Appendix Word tables.
    widths = [50, 22, 25, 25, 22, 20]
    for c in range(1, max_cols + 1):
        ws.column_dimensions[get_column_letter(c)].width = widths[c-1] if c <= len(widths) else 18

    ws.page_setup.orientation = 'landscape'
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_margins.left = 0.3
    ws.page_margins.right = 0.3
    ws.page_margins.top = 0.5
    ws.page_margins.bottom = 0.5

def _write_manuscript_workbook(
    path: str,
    sheets: List[str],
    manifest: Dict[str, str],
    csv_dir: str,
):
    ensure_dir(os.path.dirname(path))
    # Create workbook through pandas/openpyxl, then style each worksheet.
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for sheet in sheets:
            if sheet == 'Appendix_Table_C4':
                _write_appendix_c4_sheet(writer, csv_dir)
                continue
            if sheet not in manifest:
                raise KeyError(f'Missing table in merge manifest: {sheet}')
            csv_path = os.path.join(csv_dir, manifest[sheet])
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f'Missing CSV for {sheet}: {csv_path}')
            _write_regular_sheet(
                writer,
                sheet_name=sheet,
                csv_path=csv_path,
                kind='appendix' if sheet.startswith('Appendix_') else 'main',
            )
    return path


def merge_excel(out_dir: str, appendix_manifest: Dict[str, str]) -> str:
    """
    Generate three Word-style, table-content-only workbooks:
      1) Main tables only;
      2) Appendix tables only;
      3) All tables merged.

    Table captions/titles and manuscript Notes are deliberately omitted so their
    wording can be finalized later in Word. The legacy merged filename is retained
    so downstream workflows do not break.
    """
    csv_dir = os.path.join(out_dir, 'table_out', 'paper_tables_csv')
    manifest = _build_manifest(csv_dir)

    required = MAIN_TABLE_SHEETS + APPENDIX_TABLE_SHEETS
    missing = [s for s in required if s not in manifest and s != 'Appendix_Table_C4']
    if missing:
        raise KeyError(f'Missing tables in merge manifest: {missing}')

    merged_dir = os.path.join(out_dir, 'table_merge')
    ensure_dir(merged_dir)

    main_path = os.path.join(merged_dir, 'Paper_Main_Tables_Style.xlsx')
    appendix_path = os.path.join(merged_dir, 'Paper_Appendix_Tables_Style.xlsx')
    merged_path = os.path.join(merged_dir, 'Paper_All_Tables_Merged.xlsx')

    _write_manuscript_workbook(
        main_path, MAIN_TABLE_SHEETS, manifest, csv_dir
    )
    _write_manuscript_workbook(
        appendix_path, APPENDIX_TABLE_SHEETS, manifest, csv_dir
    )
    _write_manuscript_workbook(
        merged_path, MAIN_TABLE_SHEETS + APPENDIX_TABLE_SHEETS, manifest, csv_dir
    )

    # Small manifest for audit / downstream copy-paste workflow.
    style_manifest = {
        'style_basis': {
            'font': 'Times New Roman 10 pt',
            'background': 'white',
            'borders': 'horizontal rules only; no vertical grid',
            'header': 'bold, centered, thin top and bottom rules',
            'main_panel_rows': 'bold italic with light-gray manuscript panel cue',
            'appendix_panel_rows': 'bold italic, white background',
            'numeric_alignment': 'center',
            'text_alignment': 'left',
            'gridlines': 'hidden',
            'embedded_table_titles': False,
            'embedded_manuscript_notes': False,
            'reason': 'captions and Notes remain author-editable until manuscript wording is finalized',
        },
        'files': {
            'main_tables': os.path.basename(main_path),
            'appendix_tables': os.path.basename(appendix_path),
            'all_tables': os.path.basename(merged_path),
        },
        'appendix_c4': {
            'panel_a': 'Posted-price availability and analytical-sample construction',
            'panel_b': 'Final 521 modeling listings vs other 6,165 KG listings',
            'supplier_metric_label': 'Supplier listing breadth in the full KG',
            'mean_and_median_both_retained': True,
        },
    }
    with open(
        os.path.join(merged_dir, 'TABLE_STYLE_AND_OUTPUT_MANIFEST.json'),
        'w', encoding='utf-8'
    ) as f:
        json.dump(style_manifest, f, ensure_ascii=False, indent=2)

    print(f'      Main-table workbook: {main_path}')
    print(f'      Appendix-table workbook: {appendix_path}')
    print(f'      Combined workbook: {merged_path}')
    return merged_path


# =============================================================================
# FINAL REVISION ADDITIONS
#   1) FULLY fold-inductive Table 4 / Table 5.
#      Every representation learned from the 521 labeled sample is re-fit inside
#      each fold: TF-IDF -> SVD -> competitor KNN -> encoders/scalers -> model.
#   2) Appendix C.4 posted-price-availability audit plus final-vs-other KG comparison requested by Reviewer #2(i).
#
# The broader 6,686-node KG/media context is held fixed only for price-independent
# market-context variables (Bs, heat_total, HHI_proxy, T). This is intentional:
# the paper studies intra-market benchmarking in an already-observed market.
# =============================================================================

# Strict-CV configuration. These globals are assigned by
# build_fully_strict_table4_table5() from the wrapper arguments.
PRICE_FILE: Path
MEDIA_FILE: Path
NODES_FILE: Path
REL_APP_FILE: Path
REL_SRC_FILE: Path
DESCRIPTION_FILE: Optional[Path] = None
DESCRIPTION_DP_ID_COL: Optional[str] = None
DESCRIPTION_NAME_COL: Optional[str] = None
DESCRIPTION_SUPPLIER_COL: Optional[str] = None
DESCRIPTION_TEXT_COL: Optional[str] = None
KFOLD_SPLITS_FILE: Path
GROUPKFOLD_SPLITS_FILE: Path

RANDOM_STATE = 42
N_SPLITS = 5
TEXT_EMB_DIM = 50
TEXT_MIN_DF = 2
TEXT_NGRAM_RANGE = (2, 4)
KNN_K = 10
MIN_COSINE_SIM = 0.25
BLOCKS = ["Supplier FE", "Src FE", "App FE", "Product", "Supply", "Demand", "Market"]
COMP_COLS = [
    "comp_app_entropy", "comp_src_entropy",
    "comp_app_top1_prob", "comp_src_top1_prob",
    "comp_expected_heat_total",
]

def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if ext in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def pick_col(df: pd.DataFrame, explicit: Optional[str], candidates: List[str], what: str) -> Optional[str]:
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(f"Configured {what} column '{explicit}' not found. Available: {df.columns.tolist()}")
        return explicit
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def normalize_str(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def load_splits(path: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out = []
    for d in obj:
        out.append((np.asarray(d["train_idx"], dtype=int), np.asarray(d["test_idx"], dtype=int)))
    return out


def metrics(y_ln: np.ndarray, pred_ln: np.ndarray) -> Dict[str, float]:
    y_ln = np.asarray(y_ln, dtype=float)
    pred_ln = np.asarray(pred_ln, dtype=float)
    y = np.exp(y_ln)
    p = np.exp(pred_ln)
    eps = 1e-12
    return {
        "R2": float(r2_score(y_ln, pred_ln)),
        "MAE": float(mean_absolute_error(y_ln, pred_ln)),
        "RMSE": float(np.sqrt(mean_squared_error(y_ln, pred_ln))),
        "MAPE": float(np.mean(np.abs((y - p) / np.maximum(np.abs(y), eps)))),
        "SMAPE": float(np.mean(np.abs(y - p) / np.maximum((np.abs(y) + np.abs(p)) / 2.0, eps))),
    }


def safe_mlb_transform(mlb: MultiLabelBinarizer, rows: List[List[str]]) -> np.ndarray:
    known = set(mlb.classes_)
    cleaned = [[v for v in xs if v in known] for xs in rows]
    return mlb.transform(cleaned)


def renorm_dict(d: Dict[str, float]) -> Dict[str, float]:
    items = [(str(k), float(v)) for k, v in d.items() if str(k).strip() and np.isfinite(float(v)) and float(v) > 0]
    s = sum(v for _, v in items)
    if s <= 1e-12:
        return {}
    return {k: v / s for k, v in items}


def posterior_from_neighbors(label_lists: List[List[str]], weights: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for labs, w in zip(label_lists, weights):
        labs = [str(x).strip() for x in (labs or []) if str(x).strip()]
        if not labs or float(w) <= 0:
            continue
        ww = float(w) / len(labs)
        for lab in labs:
            out[lab] = out.get(lab, 0.0) + ww
    return renorm_dict(out)


def entropy(post: Dict[str, float]) -> float:
    if not post:
        return 0.0
    p = np.asarray(list(post.values()), dtype=float)
    p = p / (p.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


# =============================================================================
# Data loading and ORIGINAL description attachment
# =============================================================================
def load_base_dataset() -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """Load the 521 modeling rows and price-independent market context."""
    price = pd.read_excel(PRICE_FILE).copy()
    price["price"] = pd.to_numeric(price["price"], errors="coerce")
    price = price[price["price"].gt(0)].copy().reset_index(drop=True)
    price["name"] = normalize_str(price["name"])
    price["supplier"] = normalize_str(price["supplier"])
    price["y_ln"] = np.log(price["price"].astype(float))
    if len(price) != 521:
        raise RuntimeError(f"Expected 521 final modeling rows, got {len(price)} from {PRICE_FILE}")

    nodes = pd.read_csv(NODES_FILE).copy()
    nodes = nodes.rename(columns={"name_anon": "name", "supplier_anon": "supplier", "desc_anon": "desc"})
    required = {"dp_id", "name", "supplier"}
    if not required.issubset(nodes.columns):
        raise RuntimeError(f"NODES_FILE must contain {required}; found {nodes.columns.tolist()}")
    nodes["dp_id"] = normalize_str(nodes["dp_id"])
    nodes["name"] = normalize_str(nodes["name"])
    nodes["supplier"] = normalize_str(nodes["supplier"])
    if "desc" not in nodes.columns:
        nodes["desc"] = ""

    app = pd.read_csv(REL_APP_FILE).rename(columns={"app_name": "app"}).copy()
    src = pd.read_csv(REL_SRC_FILE).rename(columns={"src_name": "src"}).copy()
    app["dp_id"] = normalize_str(app["dp_id"])
    src["dp_id"] = normalize_str(src["dp_id"])
    app["app"] = normalize_str(app["app"])
    src["src"] = normalize_str(src["src"])

    dp = nodes[["dp_id", "name", "supplier", "desc"]].copy()
    app_map = app.groupby("dp_id")["app"].apply(lambda s: sorted({v for v in s if v})).to_dict()
    src_map = src.groupby("dp_id")["src"].apply(lambda s: sorted({v for v in s if v})).to_dict()
    dp["app_list"] = dp["dp_id"].map(app_map).apply(lambda x: x if isinstance(x, list) else [])
    dp["src_list"] = dp["dp_id"].map(src_map).apply(lambda x: x if isinstance(x, list) else [])
    dp = dp.sort_values("dp_id").drop_duplicates(["name", "supplier"], keep="first")

    df = price.merge(dp[["dp_id", "name", "supplier", "desc", "app_list", "src_list"]],
                     on=["name", "supplier"], how="left", validate="one_to_one")
    if df["dp_id"].isna().any():
        bad = df.loc[df["dp_id"].isna(), ["name", "supplier"]]
        raise RuntimeError(f"{len(bad)} modeling rows failed to match KG nodes. Example:\n{bad.head(10)}")

    # External demand visibility: fixed, price-independent information.
    media = pd.read_excel(MEDIA_FILE).rename(columns={"keyword": "app"}).copy()
    for c in ["sogou_web_results", "sina_news_results", "weixin_article_results"]:
        media[c] = pd.to_numeric(media[c], errors="coerce").fillna(0.0)
    media["heat_total"] = (
        np.log1p(media["sogou_web_results"]) +
        np.log1p(media["sina_news_results"]) +
        np.log1p(media["weixin_article_results"])
    )
    heat_map = media.set_index("app")["heat_total"].to_dict()
    df["heat_total"] = df["app_list"].apply(
        lambda xs: float(np.mean([heat_map.get(a, 0.0) for a in xs])) if xs else 0.0
    )

    # Broader 6,686-node KG context: fixed, price-independent market information.
    src_edges = src.merge(dp[["dp_id", "supplier"]], on="dp_id", how="left")
    src_supplier_cnt = src_edges.groupby("src")["supplier"].nunique().to_dict()
    df["Nmarket"] = df["src_list"].apply(
        lambda xs: float(np.mean([src_supplier_cnt.get(s, 0.0) for s in xs])) if xs else 0.0
    ).clip(lower=1.0)
    df["Bs"] = np.log1p(df["Nmarket"])
    df["HHI_proxy"] = 1.0 / df["Nmarket"]
    df["T"] = np.log1p(df["Nmarket"] * np.expm1(df["heat_total"]).clip(lower=0.0))

    return df, heat_map, nodes


def attach_original_descriptions(df: pd.DataFrame, nodes: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Attach original descriptions without relying on anonymized text embeddings.

    Preferred matching order:
    1) non-empty desc already present in NODES_FILE, keyed by dp_id;
    2) DESCRIPTION_FILE keyed by dp_id;
    3) DESCRIPTION_FILE keyed by name + supplier.
    """
    d = df.copy()
    node_desc = nodes[["dp_id", "desc"]].copy()
    node_desc["desc"] = normalize_str(node_desc["desc"])
    node_nonempty = int(node_desc["desc"].ne("").sum())

    source_used = None
    match_mode = None

    if node_nonempty > 0:
        desc_map = node_desc.drop_duplicates("dp_id").set_index("dp_id")["desc"]
        d["desc_original"] = d["dp_id"].map(desc_map).fillna("").astype(str)
        source_used = str(NODES_FILE)
        match_mode = "dp_id from NODES_FILE"
    else:
        if DESCRIPTION_FILE is None:
            raise RuntimeError(
                "FULLY STRICT CV REQUIRES EITHER (A) VERIFIED STRICT FOLD CACHES OR (B) THE ORIGINAL DESCRIPTIONS ONCE.\n"
                "The current public NODES_FILE contains no usable descriptions because desc_anon was intentionally blanked during anonymization.\n"
                "If strict caches do not yet exist, rerun once on the private author machine with: \n"
                "  python kg_utility_reasoning_reproduce.py --description_file /path/to/original_descriptions.xlsx\n"
                "Preferred private format: columns dp_id + desc/description. The dp_id should match the public KG dp_id.\n"
                "Alternative: columns name + supplier + desc, where name/supplier match PRICE_FILE identifiers.\n"
                "After that run, the script writes two numeric strict fold caches that can be placed in anymous/ for GitHub reproduction; raw text need not be published."
            )
        desc_file = Path(DESCRIPTION_FILE)
        x = read_table(desc_file).copy()
        text_col = pick_col(
            x, DESCRIPTION_TEXT_COL,
            ["desc", "description", "description_text", "desc_anon", "简介", "描述", "内容"],
            "description text"
        )
        if text_col is None:
            raise RuntimeError(f"Cannot find a description-text column in {desc_file}. Columns: {x.columns.tolist()}")
        x[text_col] = normalize_str(x[text_col])

        dp_col = pick_col(x, DESCRIPTION_DP_ID_COL, ["dp_id", "data_product_id", "product_id"], "dp_id")
        if dp_col is not None:
            x[dp_col] = normalize_str(x[dp_col])
            if x[dp_col].duplicated().any():
                # Duplicate dp_id is only safe if description is identical; keep first after audit.
                nunq = x.groupby(dp_col)[text_col].nunique(dropna=False)
                conflict = nunq[nunq > 1]
                if len(conflict):
                    raise RuntimeError(f"DESCRIPTION_FILE has conflicting descriptions for {len(conflict)} dp_id values.")
            desc_map = x.drop_duplicates(dp_col).set_index(dp_col)[text_col]
            d["desc_original"] = d["dp_id"].map(desc_map).fillna("").astype(str)
            source_used = str(desc_file)
            match_mode = f"dp_id ({dp_col})"
        else:
            name_col = pick_col(x, DESCRIPTION_NAME_COL, ["name", "name_anon", "product_name", "产品名称", "数据名称"], "name")
            sup_col = pick_col(x, DESCRIPTION_SUPPLIER_COL, ["supplier", "supplier_anon", "seller", "供应商", "卖家"], "supplier")
            if name_col is None or sup_col is None:
                raise RuntimeError(
                    f"DESCRIPTION_FILE has no dp_id and no usable name+supplier keys. Columns: {x.columns.tolist()}"
                )
            x[name_col] = normalize_str(x[name_col])
            x[sup_col] = normalize_str(x[sup_col])
            x2 = x[[name_col, sup_col, text_col]].rename(columns={name_col: "name", sup_col: "supplier", text_col: "desc_original"})
            nunq = x2.groupby(["name", "supplier"])["desc_original"].nunique(dropna=False)
            conflict = nunq[nunq > 1]
            if len(conflict):
                raise RuntimeError(f"DESCRIPTION_FILE has conflicting descriptions for {len(conflict)} name+supplier keys.")
            x2 = x2.drop_duplicates(["name", "supplier"])
            x2["__description_key_matched__"] = True
            d = d.drop(columns=["desc_original"], errors="ignore").merge(x2, on=["name", "supplier"], how="left", validate="one_to_one")
            d["desc_original"] = d["desc_original"].fillna("").astype(str)
            source_used = str(desc_file)
            match_mode = f"name+supplier ({name_col}, {sup_col})"

    # Every modeling row must be matched to the description source. The text itself may
    # legitimately be empty for a small number of products; an empty string is then treated
    # as an observed missing description and transformed to a zero TF-IDF row. What is not
    # allowed is an unmatched row silently becoming empty because of a failed key join.
    matched_flag = d.get("__description_key_matched__")
    if matched_flag is None:
        # In the dp_id branches, mapping coverage is checked by whether the dp_id exists in
        # the source index, not by whether the description text is non-empty.
        if match_mode and match_mode.startswith("dp_id"):
            if source_used == str(NODES_FILE):
                source_keys = set(node_desc["dp_id"].astype(str))
            else:
                source_keys = set(x[dp_col].astype(str))
            matched = int(d["dp_id"].astype(str).isin(source_keys).sum())
        else:
            matched = int(d["desc_original"].notna().sum())
    else:
        matched = int(pd.Series(matched_flag).fillna(False).astype(bool).sum())

    if matched != len(d):
        raise RuntimeError(
            f"Description key matching is incomplete: {matched}/{len(d)} modeling rows matched. "
            "Fully strict CV requires every modeling row to be linked to its intended description record."
        )

    nonempty = int(d["desc_original"].fillna("").astype(str).str.strip().ne("").sum())
    if nonempty < 2:
        raise RuntimeError(
            f"Only {nonempty}/{len(d)} modeling rows contain non-empty description text. "
            "The public anonymized KG intentionally omits raw descriptions, so TF-IDF/SVD cannot be re-fit from it. "
            "Provide the private original description file once, or use strict fold-specific embedding caches generated from it."
        )

    info = {
        "description_source": source_used,
        "description_match_mode": match_mode,
        "modeling_rows": int(len(d)),
        "matched_description_keys": matched,
        "nonempty_descriptions": nonempty,
        "empty_descriptions": int(len(d) - nonempty),
    }
    d = d.drop(columns=["__description_key_matched__"], errors="ignore")
    return d, info


# =============================================================================
# Fully fold-specific Product block: TF-IDF -> SVD -> KNN -> comp_*
# =============================================================================
def fit_text_embedding_train_only(train_texts: List[str]) -> Tuple[TfidfVectorizer, TruncatedSVD, np.ndarray, Dict[str, int]]:
    tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=TEXT_NGRAM_RANGE,
        min_df=TEXT_MIN_DF,
    )
    Xtr_tfidf = tfidf.fit_transform(np.asarray(train_texts, dtype=str))
    n_samples, n_features = Xtr_tfidf.shape
    if n_features < 3 or n_samples < 3:
        raise RuntimeError(
            f"Train-fold TF-IDF matrix is too small for SVD: samples={n_samples}, features={n_features}."
        )
    n_comp = int(min(TEXT_EMB_DIM, n_samples - 1, n_features - 1))
    if n_comp < 2:
        raise RuntimeError(f"Train-fold SVD dimension would be {n_comp}; cannot proceed.")
    svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
    Xtr_emb = svd.fit_transform(Xtr_tfidf).astype(float)
    meta = {"tfidf_features": int(n_features), "svd_dim": int(n_comp), "train_rows": int(n_samples)}
    return tfidf, svd, Xtr_emb, meta


def transform_text(tfidf: TfidfVectorizer, svd: TruncatedSVD, texts: List[str]) -> np.ndarray:
    return svd.transform(tfidf.transform(np.asarray(texts, dtype=str))).astype(float)


def competitor_features_from_train(
    X_train: np.ndarray,
    X_query: np.ndarray,
    train_global_idx: np.ndarray,
    query_global_idx: np.ndarray,
    train_apps: List[List[str]],
    train_srcs: List[List[str]],
    heat_map: Dict[str, float],
    query_is_train: bool,
) -> pd.DataFrame:
    """For every query, competitor candidates are ONLY rows from the training fold."""
    if len(X_train) < 2:
        raise RuntimeError("Training fold has fewer than 2 observations.")

    n_neighbors = min(KNN_K + (1 if query_is_train else 0), len(X_train))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(X_train)
    dist, ids = nn.kneighbors(X_query, return_distance=True)
    sim = 1.0 - dist

    rows = []
    for qpos, gidx in enumerate(query_global_idx):
        pairs: List[Tuple[int, float]] = []
        for loc, w in zip(ids[qpos], sim[qpos]):
            loc = int(loc)
            w = float(w)
            # self exclusion only for train-query products
            if query_is_train and int(train_global_idx[loc]) == int(gidx):
                continue
            if w < MIN_COSINE_SIM:
                continue
            pairs.append((loc, w))
            if len(pairs) >= KNN_K:
                break

        weights = np.asarray([w for _, w in pairs], dtype=float)
        post_app = posterior_from_neighbors([train_apps[j] for j, _ in pairs], weights)
        post_src = posterior_from_neighbors([train_srcs[j] for j, _ in pairs], weights)

        rows.append({
            "comp_app_entropy": entropy(post_app),
            "comp_src_entropy": entropy(post_src),
            "comp_app_top1_prob": max(post_app.values()) if post_app else 0.0,
            "comp_src_top1_prob": max(post_src.values()) if post_src else 0.0,
            "comp_expected_heat_total": float(
                sum(float(p) * float(heat_map.get(app, 0.0)) for app, p in post_app.items())
            ) if post_app else 0.0,
            "n_valid_competitors": int(len(pairs)),
        })
    return pd.DataFrame(rows, index=query_global_idx)


def build_fold_data(
    df: pd.DataFrame,
    heat_map: Dict[str, float],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cv_name: str,
    fold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Construct train/test fold from raw descriptions with NO test-fitted text step."""
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    if len(set(train_idx.tolist()) & set(test_idx.tolist())):
        raise AssertionError(f"{cv_name} fold {fold}: train/test row overlap detected.")

    tr = df.iloc[train_idx].copy()
    te = df.iloc[test_idx].copy()

    # 1) fit text model on TRAIN descriptions only
    tfidf, svd, Xtr, txt_meta = fit_text_embedding_train_only(tr["desc_original"].fillna("").astype(str).tolist())
    # 2) TEST only transformed by train-fitted text model
    Xte = transform_text(tfidf, svd, te["desc_original"].fillna("").astype(str).tolist())

    emb_cols = [f"textemb_{i:02d}" for i in range(Xtr.shape[1])]
    tr.loc[:, emb_cols] = Xtr
    te.loc[:, emb_cols] = Xte

    # 3) competitor pool = TRAIN ONLY
    train_apps = tr["app_list"].tolist()
    train_srcs = tr["src_list"].tolist()
    tr_comp = competitor_features_from_train(
        Xtr, Xtr, train_idx, train_idx, train_apps, train_srcs, heat_map, query_is_train=True
    )
    te_comp = competitor_features_from_train(
        Xtr, Xte, train_idx, test_idx, train_apps, train_srcs, heat_map, query_is_train=False
    )
    for c in COMP_COLS:
        tr[c] = tr_comp.loc[train_idx, c].to_numpy()
        te[c] = te_comp.loc[test_idx, c].to_numpy()

    # audit: GroupKFold must contain no supplier overlap
    supplier_overlap = len(set(tr["supplier"]) & set(te["supplier"]))
    if cv_name == "Supplier GroupKFold" and supplier_overlap != 0:
        raise AssertionError(f"GroupKFold fold {fold}: supplier overlap={supplier_overlap}, expected 0.")

    audit = {
        "CV": cv_name,
        "fold": int(fold),
        "train_n": int(len(tr)),
        "test_n": int(len(te)),
        "row_overlap_n": 0,
        "supplier_overlap_n": int(supplier_overlap),
        "train_nonempty_desc": int(tr["desc_original"].str.strip().ne("").sum()),
        "test_nonempty_desc": int(te["desc_original"].str.strip().ne("").sum()),
        "tfidf_fit_rows": int(len(tr)),
        "tfidf_vocab_size": int(txt_meta["tfidf_features"]),
        "svd_fit_rows": int(len(tr)),
        "svd_dim": int(txt_meta["svd_dim"]),
        "knn_candidate_rows": int(len(tr)),
        "test_rows_used_to_fit_tfidf": 0,
        "test_rows_used_to_fit_svd": 0,
        "test_rows_in_knn_candidate_pool": 0,
        "heldout_y_used_in_feature_construction": False,
        "mean_train_competitors": float(tr_comp["n_valid_competitors"].mean()),
        "mean_test_competitors": float(te_comp["n_valid_competitors"].mean()),
    }
    return tr, te, audit


# =============================================================================
# Fold-specific encoders/scalers/model
# =============================================================================
def fit_predict_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, chosen: List[str], emb_cols: List[str]) -> np.ndarray:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    ohe.fit(train_df[["supplier"]].astype(str))

    src_mlb = MultiLabelBinarizer().fit(train_df["src_list"])
    app_mlb = MultiLabelBinarizer().fit(train_df["app_list"])

    numeric = {
        "Product": emb_cols + COMP_COLS,
        "Supply": ["Bs"],
        "Demand": ["heat_total"],
        "Market": ["HHI_proxy", "T"],
    }
    scalers: Dict[str, StandardScaler] = {}
    for b, cols in numeric.items():
        sc = StandardScaler()
        sc.fit(train_df[cols].astype(float))
        scalers[b] = sc

    def blocks(d: pd.DataFrame) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        out["Supplier FE"] = ohe.transform(d[["supplier"]].astype(str))
        out["Src FE"] = safe_mlb_transform(src_mlb, d["src_list"].tolist())
        out["App FE"] = safe_mlb_transform(app_mlb, d["app_list"].tolist())
        for b, cols in numeric.items():
            out[b] = scalers[b].transform(d[cols].astype(float))
        return out

    trb = blocks(train_df)
    teb = blocks(test_df)

    if not chosen:
        Xtr = np.ones((len(train_df), 1), dtype=float)
        Xte = np.ones((len(test_df), 1), dtype=float)
    else:
        Xtr = np.concatenate([trb[b] for b in chosen if trb[b].shape[1] > 0], axis=1)
        Xte = np.concatenate([teb[b] for b in chosen if teb[b].shape[1] > 0], axis=1)

    model = BayesianRidge()
    model.fit(Xtr, train_df["y_ln"].to_numpy(float))
    return model.predict(Xte)


def prepare_cv_folds(
    df: pd.DataFrame,
    heat_map: Dict[str, float],
    splits: List[Tuple[np.ndarray, np.ndarray]],
    cv_name: str,
) -> Tuple[List[Tuple[int, pd.DataFrame, pd.DataFrame, List[str]]], pd.DataFrame]:
    prepared = []
    audits = []
    for fold, (tr, te) in enumerate(splits, 1):
        print(f">>> [{cv_name}] building fully strict fold {fold}/{len(splits)}", flush=True)
        trdf, tedf, audit = build_fold_data(df, heat_map, tr, te, cv_name=cv_name, fold=fold)
        emb_cols = [c for c in trdf.columns if c.startswith("textemb_")]
        prepared.append((fold, trdf, tedf, emb_cols))
        audits.append(audit)
    return prepared, pd.DataFrame(audits)


# =============================================================================
# Privacy-preserving strict text-embedding caches
# =============================================================================
def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_keys_sha256(df: pd.DataFrame) -> str:
    keys = [str(x) for x in df["dp_id"].tolist()]
    return _sha256_text("\n".join(keys))


def _splits_sha256(splits: List[Tuple[np.ndarray, np.ndarray]]) -> str:
    serial = [
        {"train_idx": [int(x) for x in tr], "test_idx": [int(x) for x in te]}
        for tr, te in splits
    ]
    return _sha256_text(json.dumps(serial, separators=(",", ":"), sort_keys=True))


def generate_strict_textemb_cache(
    df: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    cache_path: Path,
    cv_name: str,
) -> Dict[str, Any]:
    """
    Generate privacy-preserving fold-specific embeddings from PRIVATE descriptions.

    For every fold:
      - TF-IDF is fit on TRAIN text only;
      - SVD is fit on TRAIN TF-IDF only;
      - TEST is transform-only;
      - the resulting 521 x d matrix is safe to publish because it contains no raw text.
    """
    payload: Dict[str, Any] = {}
    fold_meta = []
    for fold, (tr, te) in enumerate(splits, 1):
        tr = np.asarray(tr, dtype=int)
        te = np.asarray(te, dtype=int)
        tfidf, svd, Xtr, meta = fit_text_embedding_train_only(
            df.iloc[tr]["desc_original"].fillna("").astype(str).tolist()
        )
        Xte = transform_text(
            tfidf, svd,
            df.iloc[te]["desc_original"].fillna("").astype(str).tolist()
        )
        if Xtr.shape[1] != Xte.shape[1]:
            raise AssertionError(f"{cv_name} fold {fold}: train/test embedding dimension mismatch.")
        emb_all = np.zeros((len(df), Xtr.shape[1]), dtype=float)
        emb_all[tr, :] = Xtr
        emb_all[te, :] = Xte
        payload[f"fold{fold}"] = emb_all
        fold_meta.append({
            "fold": fold,
            "train_n": int(len(tr)),
            "test_n": int(len(te)),
            "tfidf_fit_rows": int(len(tr)),
            "svd_fit_rows": int(len(tr)),
            "tfidf_features": int(meta["tfidf_features"]),
            "svd_dim": int(meta["svd_dim"]),
            "test_rows_used_to_fit_tfidf": 0,
            "test_rows_used_to_fit_svd": 0,
        })

    metadata = {
        "cache_format": "strict_fold_textemb_v1",
        "cv_name": cv_name,
        "n_rows": int(len(df)),
        "row_keys_sha256": _row_keys_sha256(df),
        "splits_sha256": _splits_sha256(splits),
        "text_analyzer": "char",
        "ngram_range": list(TEXT_NGRAM_RANGE),
        "min_df": int(TEXT_MIN_DF),
        "max_svd_dim": int(TEXT_EMB_DIM),
        "random_state": int(RANDOM_STATE),
        "fit_scope": "TF-IDF and SVD fit on TRAIN fold only; TEST transform-only",
        "raw_text_in_cache": False,
        "folds": fold_meta,
    }
    payload["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return metadata


def load_strict_textemb_cache(
    cache_path: Path,
    df: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    cv_name: str,
) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    z = np.load(cache_path, allow_pickle=False)
    if "metadata_json" not in z.files:
        raise RuntimeError(
            f"{cache_path} is not a verified strict cache (missing metadata_json). "
            "Legacy STEP0_fold_textemb_cache.npz is not accepted as proof for fully strict Table 4/5."
        )
    meta = json.loads(str(np.asarray(z["metadata_json"]).item()))
    required = {
        "cache_format": "strict_fold_textemb_v1",
        "cv_name": cv_name,
        "n_rows": int(len(df)),
        "row_keys_sha256": _row_keys_sha256(df),
        "splits_sha256": _splits_sha256(splits),
    }
    for k, expected in required.items():
        if meta.get(k) != expected:
            raise RuntimeError(
                f"Strict cache validation failed for {cache_path.name}: {k}={meta.get(k)!r}, expected {expected!r}."
            )
    fold2emb: Dict[int, np.ndarray] = {}
    for fold in range(1, len(splits) + 1):
        key = f"fold{fold}"
        if key not in z.files:
            raise RuntimeError(f"Strict cache {cache_path} missing {key}.")
        a = np.asarray(z[key], dtype=float)
        if a.shape[0] != len(df) or a.ndim != 2 or a.shape[1] < 2:
            raise RuntimeError(f"Invalid {key} shape in {cache_path}: {a.shape}")
        fold2emb[fold] = a
    return fold2emb, meta


def build_fold_data_from_cached_embedding(
    df: pd.DataFrame,
    heat_map: Dict[str, float],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    emb_all: np.ndarray,
    cv_name: str,
    fold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Build one strict fold from a verified train-fitted text-embedding cache."""
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    if len(set(train_idx.tolist()) & set(test_idx.tolist())):
        raise AssertionError(f"{cv_name} fold {fold}: train/test row overlap detected.")

    tr = df.iloc[train_idx].copy()
    te = df.iloc[test_idx].copy()
    Xtr = np.asarray(emb_all[train_idx], dtype=float)
    Xte = np.asarray(emb_all[test_idx], dtype=float)
    emb_cols = [f"textemb_{i:02d}" for i in range(Xtr.shape[1])]
    tr.loc[:, emb_cols] = Xtr
    te.loc[:, emb_cols] = Xte

    train_apps = tr["app_list"].tolist()
    train_srcs = tr["src_list"].tolist()
    tr_comp = competitor_features_from_train(
        Xtr, Xtr, train_idx, train_idx, train_apps, train_srcs, heat_map, query_is_train=True
    )
    te_comp = competitor_features_from_train(
        Xtr, Xte, train_idx, test_idx, train_apps, train_srcs, heat_map, query_is_train=False
    )
    for c in COMP_COLS:
        tr[c] = tr_comp.loc[train_idx, c].to_numpy()
        te[c] = te_comp.loc[test_idx, c].to_numpy()

    supplier_overlap = len(set(tr["supplier"]) & set(te["supplier"]))
    if cv_name == "Supplier GroupKFold" and supplier_overlap != 0:
        raise AssertionError(f"GroupKFold fold {fold}: supplier overlap={supplier_overlap}, expected 0.")

    audit = {
        "CV": cv_name,
        "fold": int(fold),
        "train_n": int(len(tr)),
        "test_n": int(len(te)),
        "row_overlap_n": 0,
        "supplier_overlap_n": int(supplier_overlap),
        "train_nonempty_desc": np.nan,
        "test_nonempty_desc": np.nan,
        "tfidf_fit_rows": int(len(tr)),
        "tfidf_vocab_size": np.nan,
        "svd_fit_rows": int(len(tr)),
        "svd_dim": int(Xtr.shape[1]),
        "knn_candidate_rows": int(len(tr)),
        "test_rows_used_to_fit_tfidf": 0,
        "test_rows_used_to_fit_svd": 0,
        "test_rows_in_knn_candidate_pool": 0,
        "heldout_y_used_in_feature_construction": False,
        "mean_train_competitors": float(tr_comp["n_valid_competitors"].mean()),
        "mean_test_competitors": float(te_comp["n_valid_competitors"].mean()),
        "text_representation_source": "verified strict fold cache",
    }
    return tr, te, audit


def prepare_cv_folds_from_cache(
    df: pd.DataFrame,
    heat_map: Dict[str, float],
    splits: List[Tuple[np.ndarray, np.ndarray]],
    cv_name: str,
    fold2emb: Dict[int, np.ndarray],
) -> Tuple[List[Tuple[int, pd.DataFrame, pd.DataFrame, List[str]]], pd.DataFrame]:
    prepared, audits = [], []
    for fold, (tr, te) in enumerate(splits, 1):
        print(f">>> [{cv_name}] building strict cached fold {fold}/{len(splits)}", flush=True)
        trdf, tedf, audit = build_fold_data_from_cached_embedding(
            df, heat_map, tr, te, fold2emb[fold], cv_name=cv_name, fold=fold
        )
        emb_cols = [c for c in trdf.columns if c.startswith("textemb_")]
        prepared.append((fold, trdf, tedf, emb_cols))
        audits.append(audit)
    return prepared, pd.DataFrame(audits)

def run_scenario(prepared, chosen: List[str], cv_name: str, scenario: str) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows = []
    for fold, trdf, tedf, emb_cols in prepared:
        pred = fit_predict_fold(trdf, tedf, chosen, emb_cols)
        m = metrics(tedf["y_ln"].to_numpy(float), pred)
        rows.append({"CV": cv_name, "fold": fold, "Scenario": scenario, **m})
    fr = pd.DataFrame(rows)
    summary: Dict[str, float] = {}
    for c in ["R2", "MAE", "RMSE", "MAPE", "SMAPE"]:
        summary[c] = float(fr[c].mean())
        summary[c + "_SD"] = float(fr[c].std(ddof=0))
    return summary, fr


# =============================================================================
# Build strict Table 4 and Table 5
# =============================================================================
def get_splits(df: pd.DataFrame) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]], str]:
    if KFOLD_SPLITS_FILE.exists() and GROUPKFOLD_SPLITS_FILE.exists():
        k = load_splits(KFOLD_SPLITS_FILE)
        g = load_splits(GROUPKFOLD_SPLITS_FILE)
        source = "stored STEP0 split JSON files"
    else:
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        k = [(tr, te) for tr, te in kf.split(df)]
        groups = df["supplier"].astype(str).to_numpy()
        gkf = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))
        g = [(tr, te) for tr, te in gkf.split(df, df["y_ln"].to_numpy(float), groups)]
        source = "regenerated from RANDOM_STATE=42"
    return k, g, source


def build_tables(
    df: pd.DataFrame, heat_map: Dict[str, float],
    k_cache: Optional[Dict[int, np.ndarray]] = None,
    g_cache: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    k_splits, g_splits, split_source = get_splits(df)

    if k_cache is None:
        prep_k, audit_k = prepare_cv_folds(df, heat_map, k_splits, "KFold")
    else:
        prep_k, audit_k = prepare_cv_folds_from_cache(df, heat_map, k_splits, "KFold", k_cache)
    if g_cache is None:
        prep_g, audit_g = prepare_cv_folds(df, heat_map, g_splits, "Supplier GroupKFold")
    else:
        prep_g, audit_g = prepare_cv_folds_from_cache(df, heat_map, g_splits, "Supplier GroupKFold", g_cache)
    audit = pd.concat([audit_k, audit_g], ignore_index=True)
    audit["split_source"] = split_source

    # Table 4
    t4_rows = []
    fold_metrics = []

    m, fr = run_scenario(prep_k, [], "KFold", "None")
    t4_rows.append({"Panel": "Panel A. Independent K-fold fit", "Component": "Intercept only", "Scenario": "None", **m})
    fold_metrics.append(fr)

    for b in BLOCKS:
        m, fr = run_scenario(prep_k, [b], "KFold", b)
        t4_rows.append({"Panel": "Panel A. Independent K-fold fit", "Component": b, "Scenario": b, **m})
        fold_metrics.append(fr)

    cum: List[str] = []
    for b in BLOCKS:
        cum = cum + [b]
        scenario = " + ".join(cum)
        m, fr = run_scenario(prep_k, cum, "KFold", scenario)
        t4_rows.append({"Panel": "Panel B. Conditional additions (K-fold)", "Component": b, "Scenario": scenario, **m})
        fold_metrics.append(fr)
    t4 = pd.DataFrame(t4_rows)

    # Table 5
    t5_rows = []
    cum = []
    for b in BLOCKS:
        cum = cum + [b]
        scenario = " + ".join(cum)
        mr, fr_r = run_scenario(prep_k, cum, "KFold", scenario)
        mg, fr_g = run_scenario(prep_g, cum, "Supplier GroupKFold", scenario)
        fold_metrics.extend([fr_r, fr_g])
        t5_rows.append({
            "Component": b,
            "Scenario": scenario,
            "Random K-fold R2": mr["R2"],
            "Random K-fold SD": mr["R2_SD"],
            "Group K-fold R2": mg["R2"],
            "Group K-fold SD": mg["R2_SD"],
            "Delta (Group-Random)": mg["R2"] - mr["R2"],
            "Group MAE": mg["MAE"],
            "Group RMSE": mg["RMSE"],
        })
    t5 = pd.DataFrame(t5_rows)

    long = pd.concat(fold_metrics, ignore_index=True)
    return t4, t5, audit, long


# =============================================================================
# Output / assertions
# =============================================================================
def validate_leakage_audit(audit: pd.DataFrame) -> Dict[str, Any]:
    checks = {
        "all_row_overlap_zero": bool((audit["row_overlap_n"] == 0).all()),
        "all_test_rows_used_to_fit_tfidf_zero": bool((audit["test_rows_used_to_fit_tfidf"] == 0).all()),
        "all_test_rows_used_to_fit_svd_zero": bool((audit["test_rows_used_to_fit_svd"] == 0).all()),
        "all_test_rows_in_knn_candidate_pool_zero": bool((audit["test_rows_in_knn_candidate_pool"] == 0).all()),
        "all_heldout_y_not_used_in_feature_construction": bool((audit["heldout_y_used_in_feature_construction"] == False).all()),
        "groupkfold_supplier_overlap_zero": bool((audit.loc[audit["CV"] == "Supplier GroupKFold", "supplier_overlap_n"] == 0).all()),
    }
    checks["ALL_STRICT_CHECKS_PASS"] = bool(all(checks.values()))
    if not checks["ALL_STRICT_CHECKS_PASS"]:
        raise AssertionError(f"Strict leakage audit failed: {checks}")
    return checks


def _description_candidate_coverage(path: Path, df: pd.DataFrame) -> Tuple[int, int, str]:
    """Return (matched_keys, matched_nonempty_text, mode) for a candidate description file."""
    try:
        x = read_table(path).copy()
    except Exception:
        return 0, 0, "unreadable"
    text_col = pick_col(
        x, None,
        ["desc", "description", "description_text", "desc_anon", "简介", "描述", "内容"],
        "description text"
    )
    if text_col is None:
        return 0, 0, "no-text-column"
    x[text_col] = normalize_str(x[text_col])
    dp_col = pick_col(x, None, ["dp_id", "data_product_id", "product_id"], "dp_id")
    if dp_col is not None:
        x[dp_col] = normalize_str(x[dp_col])
        m = df[["dp_id"]].merge(x[[dp_col, text_col]].rename(columns={dp_col: "dp_id"}), on="dp_id", how="left")
        matched = int(m[text_col].notna().sum())
        nonempty = int(m[text_col].fillna("").astype(str).str.strip().ne("").sum())
        return matched, nonempty, f"dp_id:{dp_col}"
    name_col = pick_col(x, None, ["name", "name_anon", "product_name", "产品名称", "数据名称"], "name")
    sup_col = pick_col(x, None, ["supplier", "supplier_anon", "seller", "供应商", "卖家"], "supplier")
    if name_col is not None and sup_col is not None:
        x[name_col] = normalize_str(x[name_col]); x[sup_col] = normalize_str(x[sup_col])
        m = df[["name", "supplier"]].merge(
            x[[name_col, sup_col, text_col]].rename(columns={name_col: "name", sup_col: "supplier"}),
            on=["name", "supplier"], how="left"
        )
        matched = int(m[text_col].notna().sum())
        nonempty = int(m[text_col].fillna("").astype(str).str.strip().ne("").sum())
        return matched, nonempty, f"name+supplier:{name_col},{sup_col}"
    return 0, 0, "no-usable-key"


def _discover_description_file(any_root: str, df: pd.DataFrame) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    """Search likely private/raw folders and choose the candidate with highest 521-row coverage."""
    script_root = Path(__file__).resolve().parent
    data_root = Path(any_root).resolve()
    roots = [script_root, script_root / "private", script_root / "raw", script_root / "data",
             data_root, data_root.parent, data_root.parent / "private", data_root.parent / "raw"]
    seen, candidates = set(), []
    skip_names = {
        "name_price_anonymized.xlsx", "name_price_all_anonymized.xlsx", "media_result.xlsx",
        "nodes_dataproduct.csv", "STEP0_product_textemb.csv"
    }
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            files = list(root.glob("*.csv")) + list(root.glob("*.xlsx")) + list(root.glob("*.xls"))
            # One level below is enough for common private/raw/data layouts and avoids scanning huge trees.
            for sub in root.iterdir():
                if sub.is_dir() and sub.name not in {".git", "result_kg_reproduce", "neo4j_export"}:
                    files += list(sub.glob("*.csv")) + list(sub.glob("*.xlsx")) + list(sub.glob("*.xls"))
        except Exception:
            continue
        for f in files:
            try: rf = f.resolve()
            except Exception: continue
            if rf in seen or f.name in skip_names:
                continue
            seen.add(rf)
            matched, nonempty, mode = _description_candidate_coverage(f, df)
            if matched > 0:
                candidates.append({"path": str(f), "matched": matched, "nonempty": nonempty, "mode": mode})
    candidates.sort(key=lambda z: (z["matched"], z["nonempty"]), reverse=True)
    if candidates and candidates[0]["matched"] == len(df) and candidates[0]["nonempty"] >= 2:
        return Path(candidates[0]["path"]), candidates
    return None, candidates


def _write_description_requirements(df: pd.DataFrame, out_dir: str, candidates: List[Dict[str, Any]]) -> Tuple[Path, Path]:
    diag = Path(out_dir) / "strict_cv_requirements"
    diag.mkdir(parents=True, exist_ok=True)
    template = diag / "STRICT_CV_REQUIRED_DESCRIPTIONS.csv"
    req = df[["dp_id", "name", "supplier"]].copy()
    req["desc"] = ""
    req.to_csv(template, index=False, encoding="utf-8-sig")
    report = diag / "README_MISSING_ORIGINAL_DESCRIPTIONS.txt"
    lines = [
        "FULLY STRICT TABLE 4/5 CANNOT BE GENERATED FROM THE PUBLIC ANONYMIZED PACKAGE ALONE.",
        "",
        "Reason: anymous/neo4j_export/nodes_dataproduct.csv intentionally contains blank desc_anon.",
        "The repository README also states that raw product descriptions were removed and only semantic/intermediate variables were released.",
        "A strict fold-inductive TF-IDF/SVD rerun mathematically requires the description text once, on the author's private machine.",
        "",
        "Private one-time solution:",
        "  python kg_utility_reasoning_reproduce.py --description_file /path/to/original_descriptions.xlsx",
        "",
        "Preferred private file format: dp_id + desc (or description).",
        "Alternative: name + supplier + desc, provided the identifiers match the 521 modeling records.",
        "After one successful private run, the script writes TWO privacy-preserving strict caches into the anonymized data folder:",
        "  STEP0_strict_textemb_cache_kfold.npz",
        "  STEP0_strict_textemb_cache_groupkfold_supplier.npz",
        "These caches contain fold-specific numeric embeddings only, no raw text, and may be used by the public GitHub reproduction.",
        "",
        f"A 521-row key template was written to: {template}",
        "",
        "Auto-discovery candidates seen by the program:",
    ]
    if candidates:
        for c in candidates[:30]:
            lines.append(f"  matched={c['matched']}, nonempty={c['nonempty']}, mode={c['mode']}, path={c['path']}")
    else:
        lines.append("  (none)")
    report.write_text("\n".join(lines), encoding="utf-8")
    return template, report


def _resolve_original_description_file(any_root: str, explicit: Optional[str]) -> Optional[Path]:
    """Resolve an original description file without changing the analytical pipeline."""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f'--description_file does not exist: {p}')
        return p

    # If the KG node file itself contains non-empty desc_anon, attach_original_descriptions
    # will use it and no external file is required.
    script_root = Path(__file__).resolve().parent
    data_root = Path(any_root).resolve()
    candidates = [
        script_root / 'original_product_descriptions.xlsx',
        script_root / 'original_product_descriptions.csv',
        script_root / 'private' / 'original_product_descriptions.xlsx',
        script_root / 'private' / 'original_product_descriptions.csv',
        script_root / 'private' / 'product_descriptions.xlsx',
        script_root / 'private' / 'product_descriptions.csv',
        data_root / 'original_product_descriptions.xlsx',
        data_root / 'original_product_descriptions.csv',
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def build_fully_strict_table4_table5(
    any_root: str,
    out_dir: str,
    core_module,
    description_file: Optional[str] = None,
    prepare_caches_only: bool = False,
    allow_private_text: bool = False,
):
    """
    Rebuild manuscript Table 4 and Table 5 with full fold-inductive protection.

    Two execution paths are allowed and analytically equivalent:
    A) PRIVATE AUTHOR CACHE BUILD: original descriptions are explicitly supplied. The script fits
       TF-IDF/SVD inside each fold and writes privacy-preserving numeric strict fold caches.
    B) PUBLIC GITHUB RUN: raw descriptions are forbidden; the two strict fold caches produced
       by path A are present. The script validates their row/split hashes and uses them directly.

    The public anonymized package by itself cannot reconstruct raw descriptions; silently
    falling back to full-sample embeddings would no longer be fully strict and is forbidden.
    """
    global PRICE_FILE, MEDIA_FILE, NODES_FILE, REL_APP_FILE, REL_SRC_FILE
    global DESCRIPTION_FILE, KFOLD_SPLITS_FILE, GROUPKFOLD_SPLITS_FILE

    root = Path(any_root).resolve()
    PRICE_FILE = root / 'name_price_anonymized.xlsx'
    MEDIA_FILE = root / 'media_result.xlsx'
    NODES_FILE = root / 'neo4j_export' / 'nodes_dataproduct.csv'
    REL_APP_FILE = root / 'neo4j_export' / 'rel_applied_to.csv'
    REL_SRC_FILE = root / 'neo4j_export' / 'rel_source_industry.csv'

    k_in = root / 'STEP0_splits_kfold.json'
    g_in = root / 'STEP0_splits_groupkfold_supplier.json'
    k_out = Path(out_dir) / 'STEP0_splits_kfold.json'
    g_out = Path(out_dir) / 'STEP0_splits_groupkfold_supplier.json'
    KFOLD_SPLITS_FILE = k_in if k_in.exists() else k_out
    GROUPKFOLD_SPLITS_FILE = g_in if g_in.exists() else g_out

    strict_k_cache_path = root / 'STEP0_strict_textemb_cache_kfold.npz'
    strict_g_cache_path = root / 'STEP0_strict_textemb_cache_groupkfold_supplier.npz'

    print('>>> [STRICT CV] Loading 521 modeling rows and market context...', flush=True)
    df, heat_map, nodes = load_base_dataset()
    k_splits, g_splits, split_source = get_splits(df)

    cache_mode = strict_k_cache_path.exists() and strict_g_cache_path.exists()
    cache_meta = {}

    if cache_mode:
        print('>>> [STRICT CV] Using verified privacy-preserving strict fold caches; raw descriptions are not required.', flush=True)
        k_cache, k_meta = load_strict_textemb_cache(strict_k_cache_path, df, k_splits, 'KFold')
        g_cache, g_meta = load_strict_textemb_cache(strict_g_cache_path, df, g_splits, 'Supplier GroupKFold')
        cache_meta = {'kfold': k_meta, 'groupkfold': g_meta}
        desc_info = {
            'description_source': 'private descriptions used previously to generate verified strict fold caches',
            'description_match_mode': 'not needed in public cache mode',
            'modeling_rows': len(df),
            'matched_description_keys': len(df),
            'nonempty_descriptions': None,
            'empty_descriptions': None,
        }
    else:
        if not allow_private_text:
            raise RuntimeError(
                "PUBLIC STRICT CV MODE REQUIRES THE TWO VERIFIED NUMERIC FOLD CACHES. "
                "Raw descriptions are intentionally not read in public mode. Generate the caches once "
                "on the private author machine with --description_file ... --prepare_strict_caches_only, "
                "then commit ONLY STEP0_strict_textemb_cache_kfold.npz and "
                "STEP0_strict_textemb_cache_groupkfold_supplier.npz."
            )
        if not description_file:
            raise RuntimeError(
                "PRIVATE CACHE BUILD REQUIRES AN EXPLICIT --description_file. Automatic discovery is disabled "
                "to prevent accidental use or publication of plaintext descriptions."
            )
        DESCRIPTION_FILE = _resolve_original_description_file(any_root, description_file)
        candidates: List[Dict[str, Any]] = []
        print(f'>>> [STRICT CV][PRIVATE] Description source found: {DESCRIPTION_FILE}', flush=True)

        try:
            df, desc_info = attach_original_descriptions(df, nodes)
        except RuntimeError as e:
            template, report = _write_description_requirements(df, out_dir, candidates)
            legacy_k = root / 'STEP0_fold_textemb_cache.npz'
            extra = (
                f"\n\nA legacy KFold embedding cache exists at {legacy_k}, but it does not provide a "
                "verified Supplier GroupKFold train-only text representation and therefore cannot establish "
                "fully strict Table 5. The program deliberately refuses to downgrade the test silently."
                if legacy_k.exists() else ''
            )
            raise RuntimeError(
                str(e) +
                f"\n\nI created the exact 521-row description-key template here:\n  {template}"
                f"\nDetailed instructions:\n  {report}"
                "\n\nProvide the original descriptions ONCE on the private author machine, then rerun. "
                "The script will automatically generate two numeric strict caches that can be committed to GitHub; "
                "the raw descriptions themselves do not need to be published." + extra
            ) from None

        print(
            f">>> [STRICT CV] descriptions: {desc_info['nonempty_descriptions']}/{desc_info['modeling_rows']} non-empty; "
            f"matched={desc_info.get('matched_description_keys')}; source={desc_info['description_source']}; "
            f"match={desc_info['description_match_mode']}",
            flush=True,
        )
        print('>>> [STRICT CV] Generating privacy-preserving strict KFold text cache...', flush=True)
        k_meta = generate_strict_textemb_cache(df, k_splits, strict_k_cache_path, 'KFold')
        print('>>> [STRICT CV] Generating privacy-preserving strict Supplier GroupKFold text cache...', flush=True)
        g_meta = generate_strict_textemb_cache(df, g_splits, strict_g_cache_path, 'Supplier GroupKFold')
        k_cache, _ = load_strict_textemb_cache(strict_k_cache_path, df, k_splits, 'KFold')
        g_cache, _ = load_strict_textemb_cache(strict_g_cache_path, df, g_splits, 'Supplier GroupKFold')
        cache_meta = {'kfold': k_meta, 'groupkfold': g_meta}
        print(f'>>> [STRICT CV] Strict cache written: {strict_k_cache_path}', flush=True)
        print(f'>>> [STRICT CV] Strict cache written: {strict_g_cache_path}', flush=True)

    if prepare_caches_only:
        print('>>> [STRICT CV] --prepare_strict_caches_only requested; caches are ready. Table generation skipped.', flush=True)
        return None, None, None, {'STRICT_CACHES_READY': True}

    # Build Table 4/5 exclusively from verified fold-specific caches. This makes the private
    # and public runs identical after the cache-generation boundary.
    t4, t5, audit, _ = build_tables(df, heat_map, k_cache=k_cache, g_cache=g_cache)
    checks = validate_leakage_audit(audit)
    if not checks.get('ALL_STRICT_CHECKS_PASS', False):
        raise AssertionError(f'FULLY STRICT leakage checks failed: {checks}')

    csv_dir = os.path.join(out_dir, 'table_out', 'paper_tables_csv')
    ensure_dir(csv_dir)
    core_module.manuscript_table('Table4', t4).to_csv(
        os.path.join(csv_dir, 'Table4.csv'), index=False, encoding='utf-8-sig'
    )
    core_module.manuscript_table('Table5', t5).to_csv(
        os.path.join(csv_dir, 'Table5.csv'), index=False, encoding='utf-8-sig'
    )

    diag = Path(out_dir) / 'diagnostics_fully_strict_cv'
    diag.mkdir(parents=True, exist_ok=True)
    audit.to_csv(diag / 'FULLY_STRICT_fold_leakage_audit.csv', index=False, encoding='utf-8-sig')
    with open(diag / 'FULLY_STRICT_cache_and_description_meta.json', 'w', encoding='utf-8') as f:
        json.dump({
            'description_info': desc_info,
            'cache_mode': bool(cache_mode),
            'strict_kfold_cache': str(strict_k_cache_path),
            'strict_groupkfold_cache': str(strict_g_cache_path),
            'split_source': split_source,
            'cache_metadata': cache_meta,
            'leakage_checks': checks,
        }, f, ensure_ascii=False, indent=2)

    print('>>> [STRICT CV] ALL STRICT LEAKAGE CHECKS PASSED:', flush=True)
    for k, v in checks.items():
        print(f'      {k}: {v}', flush=True)
    print('      TF-IDF: fitted on TRAIN fold only when strict cache was generated', flush=True)
    print('      SVD: fitted on TRAIN fold only when strict cache was generated', flush=True)
    print('      TEST text: transform-only with TRAIN-fitted TF-IDF/SVD', flush=True)
    print('      strict cache validation: row-key hash + split hash + CV type verified', flush=True)
    print('      competitor candidate pool: TRAIN fold only', flush=True)
    print('      Supplier/Src/App encoders: TRAIN fold only', flush=True)
    print('      StandardScaler: TRAIN fold only', flush=True)
    print('      BayesianRidge: TRAIN y only', flush=True)
    print('      Supplier GroupKFold: zero supplier overlap', flush=True)
    print('      6,686-node KG/media market context: fixed and price-independent by design', flush=True)
    return t4, t5, audit, checks


def _smd_nonoverlap(x, y):
    x = pd.to_numeric(pd.Series(x), errors='coerce').dropna().astype(float)
    y = pd.to_numeric(pd.Series(y), errors='coerce').dropna().astype(float)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled = math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2.0)
    return 0.0 if pooled <= 1e-15 else float((x.mean() - y.mean()) / pooled)


def build_appendix_c4_sample_comparison(any_root: str, out_dir: str):
    """
    Reviewer #2(i): observable sample-selection comparison using ONLY the existing
    public anonymized files.

    Rationale
    ---------
    The public pre-filter price file contains 559 rows / 555 unique price-observed
    product-supplier pairs, but the 34 listings later excluded by price cleaning
    deliberately use EXCLUDED_* audit placeholders and therefore are not intended to
    provide a canonical KG-node crosswalk.

    We therefore separate two questions cleanly:

    Panel A -- Posted-price availability and sample construction
        Count-based comparison only, fully reproducible from the existing anonymized
        price files:
            broader KG universe                         6,686
            with standardizable posted-price info        555
            without standardizable posted-price info   6,131
            price-observed excluded by price cleaning     34
            final price-modeling sample                  521

        No attempt is made to map the 34 EXCLUDED_* audit placeholders back to KG
        nodes. This avoids fabricating a 555-node structural crosswalk that the public
        package does not contain.

    Panel B -- Observable KG characteristics of the estimation sample
        Non-overlapping structural comparison:
            final price-modeling sample                 521
            other KG listings                         6,165

        The 6,165 listings outside the final sample comprise the 6,131 listings
        without standardizable posted-price information plus the 34 price-observed
        listings excluded by the stated price-cleaning rules.

    This design directly documents price availability and then evaluates whether the
    actual estimation sample differs observably from the rest of the broader KG market.
    """
    neo = os.path.join(any_root, 'neo4j_export')

    nodes_path = os.path.join(neo, 'nodes_dataproduct.csv')
    app_path = os.path.join(neo, 'rel_applied_to.csv')
    src_path = os.path.join(neo, 'rel_source_industry.csv')
    final_price_path = os.path.join(any_root, 'name_price_anonymized.xlsx')
    raw_price_path = os.path.join(any_root, 'name_price_all_anonymized.xlsx')

    required = [nodes_path, app_path, src_path, final_price_path, raw_price_path]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            'Appendix C.4 requires the existing public anonymized files only. '
            'Missing:\n  ' + '\n  '.join(missing)
        )

    nodes = pd.read_csv(nodes_path).rename(columns={
        'name_anon': 'name',
        'supplier_anon': 'supplier',
        'desc_anon': 'description',
    })
    app = pd.read_csv(app_path).rename(columns={'app_name': 'app'})
    src = pd.read_csv(src_path).rename(columns={'src_name': 'src'})
    final_price = pd.read_excel(final_price_path).copy()
    raw_price = pd.read_excel(raw_price_path).copy()

    # ------------------------------------------------------------------
    # A. Audit the existing public sample files
    # ------------------------------------------------------------------
    universe = nodes[['dp_id', 'name', 'supplier']].copy()
    for c in ['dp_id', 'name', 'supplier']:
        universe[c] = universe[c].fillna('').astype(str).str.strip()

    if universe['dp_id'].duplicated().any():
        raise RuntimeError(
            'Appendix C.4 cannot be constructed safely because '
            'nodes_dataproduct.csv contains duplicate dp_id values.'
        )
    if universe.duplicated(['name', 'supplier']).any():
        raise RuntimeError(
            'Appendix C.4 cannot be constructed safely because '
            'nodes_dataproduct.csv contains duplicate anonymous product-supplier keys.'
        )

    kg_n = int(len(universe))
    if kg_n != 6686:
        raise RuntimeError(
            f'Appendix C.4 expected the broader KG universe to contain 6,686 '
            f'API listings, but found {kg_n}.'
        )

    for df, label in [(raw_price, 'name_price_all_anonymized.xlsx'),
                      (final_price, 'name_price_anonymized.xlsx')]:
        need = {'name', 'supplier', 'price'}
        if not need.issubset(df.columns):
            raise RuntimeError(
                f'{label} must contain columns {sorted(need)}; '
                f'available columns are {list(df.columns)}.'
            )
        df['name'] = df['name'].fillna('').astype(str).str.strip()
        df['supplier'] = df['supplier'].fillna('').astype(str).str.strip()
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

    if raw_price['price'].isna().any():
        raise RuntimeError(
            'name_price_all_anonymized.xlsx contains missing/non-numeric prices; '
            'Appendix C.4 sample-count audit cannot be reproduced exactly.'
        )
    if final_price['price'].isna().any():
        raise RuntimeError(
            'name_price_anonymized.xlsx contains missing/non-numeric prices.'
        )

    raw_rows = int(len(raw_price))
    raw_unique = raw_price.drop_duplicates(['name', 'supplier'], keep='first').copy()
    price_observed_n = int(len(raw_unique))
    zero_unique_n = int(raw_unique['price'].eq(0).sum())
    upper_unique_n = int(raw_unique['price'].ge(300).sum())

    final_unique = final_price.drop_duplicates(['name', 'supplier'], keep='first').copy()
    final_n = int(len(final_unique))

    no_standard_price_n = kg_n - price_observed_n
    price_cleaning_excluded_n = price_observed_n - final_n
    other_kg_n = kg_n - final_n

    expected_counts = {
        'raw price-file rows': (raw_rows, 559),
        'unique listings with standardizable posted-price information': (price_observed_n, 555),
        'zero-price unique listings': (zero_unique_n, 18),
        'upper-tail unique listings priced at 300 RMB/call or above': (upper_unique_n, 16),
        'final price-modeling sample': (final_n, 521),
        'listings without standardizable posted-price information': (no_standard_price_n, 6131),
        'price-observed listings excluded by price cleaning': (price_cleaning_excluded_n, 34),
        'other KG listings outside final modeling sample': (other_kg_n, 6165),
    }
    bad = [
        f'{k}: found {actual}, expected {expected}'
        for k, (actual, expected) in expected_counts.items()
        if actual != expected
    ]
    if bad:
        raise RuntimeError(
            'Appendix C.4 public-data integrity check failed:\n  '
            + '\n  '.join(bad)
        )

    # Final 521 must map completely to the public KG. This is the only
    # listing-level crosswalk required for the structural comparison.
    dp_lookup = universe[['dp_id', 'name', 'supplier']].copy()
    final_mapped = final_unique.merge(
        dp_lookup,
        on=['name', 'supplier'],
        how='left',
        validate='one_to_one',
        indicator=True,
    )
    unmatched_final = final_mapped.loc[
        final_mapped['_merge'] != 'both',
        ['name', 'supplier']
    ]
    if len(unmatched_final):
        raise RuntimeError(
            f'Appendix C.4 final-sample mapping failed for {len(unmatched_final)} '
            f'of the 521 anonymous product-supplier keys. Examples: '
            f'{unmatched_final.head(5).to_dict("records")}'
        )

    final_ids = set(final_mapped['dp_id'].astype(str))
    if len(final_ids) != 521:
        raise RuntimeError(
            f'Appendix C.4 expected 521 unique final-sample dp_id values, '
            f'but obtained {len(final_ids)}.'
        )

    # ------------------------------------------------------------------
    # Panel A. Posted-price availability and analytical-sample construction
    # ------------------------------------------------------------------
    panel_a = pd.DataFrame([
        {
            'Sample category': 'Broader KG universe',
            'N': kg_n,
            'Share of KG universe': 1.0,
            'Share of price-observed listings': np.nan,
        },
        {
            'Sample category': 'Listings with standardizable posted-price information',
            'N': price_observed_n,
            'Share of KG universe': price_observed_n / kg_n,
            'Share of price-observed listings': 1.0,
        },
        {
            'Sample category': 'Listings without standardizable posted-price information',
            'N': no_standard_price_n,
            'Share of KG universe': no_standard_price_n / kg_n,
            'Share of price-observed listings': np.nan,
        },
        {
            'Sample category': 'Price-observed listings excluded by price cleaning',
            'N': price_cleaning_excluded_n,
            'Share of KG universe': price_cleaning_excluded_n / kg_n,
            'Share of price-observed listings': price_cleaning_excluded_n / price_observed_n,
        },
        {
            'Sample category': 'Final price-modeling sample',
            'N': final_n,
            'Share of KG universe': final_n / kg_n,
            'Share of price-observed listings': final_n / price_observed_n,
        },
    ])

    # ------------------------------------------------------------------
    # Panel B. Observable KG characteristics: final 521 vs other 6,165
    # ------------------------------------------------------------------
    app['dp_id'] = app['dp_id'].fillna('').astype(str).str.strip()
    src['dp_id'] = src['dp_id'].fillna('').astype(str).str.strip()
    app['app'] = app['app'].fillna('').astype(str).str.strip()
    src['src'] = src['src'].fillna('').astype(str).str.strip()

    # Empty category labels, if any, are not substantive links.
    app_nonempty = app.loc[app['app'].ne('')].copy()
    src_nonempty = src.loc[src['src'].ne('')].copy()

    app_count = app_nonempty.groupby('dp_id')['app'].nunique()
    src_count = src_nonempty.groupby('dp_id')['src'].nunique()

    universe['app_link_count'] = universe['dp_id'].map(app_count).fillna(0).astype(int)
    universe['src_link_count'] = universe['dp_id'].map(src_count).fillna(0).astype(int)
    universe['multi_app'] = (universe['app_link_count'] > 1).astype(int)
    universe['multi_src'] = (universe['src_link_count'] > 1).astype(int)

    # Listing-level supplier breadth: each listing inherits the number of distinct
    # API listings associated with its supplier in the full 6,686-listing KG.
    supplier_breadth = universe.groupby('supplier')['dp_id'].nunique()
    universe['supplier_listing_breadth'] = (
        universe['supplier'].map(supplier_breadth).astype(int)
    )

    universe['is_final_model'] = universe['dp_id'].isin(final_ids)
    final_group = universe.loc[universe['is_final_model']].copy()
    other_group = universe.loc[~universe['is_final_model']].copy()

    if (len(final_group), len(other_group)) != (521, 6165):
        raise RuntimeError(
            'Appendix C.4 Panel B integrity check failed: expected '
            f'521/6165 final/other listings, got '
            f'{len(final_group)}/{len(other_group)}.'
        )

    specs = [
        ('Application-domain links per listing', 'app_link_count', 'mean'),
        ('Source-industry links per listing', 'src_link_count', 'mean'),
        ('Share linked to multiple application domains', 'multi_app', 'mean'),
        ('Share linked to multiple source industries', 'multi_src', 'mean'),
        ('Supplier listing breadth in the full KG: mean', 'supplier_listing_breadth', 'mean'),
        ('Supplier listing breadth in the full KG: median', 'supplier_listing_breadth', 'median'),
    ]

    def _stat(df, col, how):
        if how == 'median':
            return float(pd.to_numeric(df[col], errors='coerce').median())
        return float(pd.to_numeric(df[col], errors='coerce').mean())

    rows_b = []
    for label, col, how in specs:
        fval = _stat(final_group, col, how)
        oval = _stat(other_group, col, how)
        rows_b.append({
            'Characteristic': label,
            'Final price-modeling sample (N=521)': fval,
            'Other KG listings (N=6165)': oval,
            'Difference (final - other)': fval - oval,
            'SMD (final vs other)': (
                np.nan if how == 'median'
                else _smd_nonoverlap(final_group[col], other_group[col])
            ),
        })
    panel_b = pd.DataFrame(rows_b)

    # ------------------------------------------------------------------
    # Manuscript-facing combined Appendix C.4 CSV
    # ------------------------------------------------------------------
    # One superset schema is used so the file can be merged automatically into the
    # existing one-click workbook. Panel headers explain the column interpretation.
    cols = [
        'Characteristic / sample category',
        'N / Final modeling sample',
        'Share of KG universe / Other KG listings',
        'Share of price-observed listings',
        'Difference (final - other)',
        'SMD (final vs other)',
    ]
    paper_rows = []

    paper_rows.append({
        cols[0]: 'Panel A. Posted-price availability and analytical-sample construction'
    })
    for _, r in panel_a.iterrows():
        paper_rows.append({
            cols[0]: r['Sample category'],
            cols[1]: int(r['N']),
            cols[2]: r['Share of KG universe'],
            cols[3]: r['Share of price-observed listings'],
            cols[4]: np.nan,
            cols[5]: np.nan,
        })

    paper_rows.append({
        cols[0]: 'Panel B. Observable characteristics of the final price-modeling sample and other KG listings'
    })
    for _, r in panel_b.iterrows():
        paper_rows.append({
            cols[0]: r['Characteristic'],
            cols[1]: r['Final price-modeling sample (N=521)'],
            cols[2]: r['Other KG listings (N=6165)'],
            cols[3]: np.nan,
            cols[4]: r['Difference (final - other)'],
            cols[5]: r['SMD (final vs other)'],
        })

    c4 = pd.DataFrame(paper_rows, columns=cols)

    csv_dir = os.path.join(out_dir, 'table_out', 'paper_tables_csv')
    ensure_dir(csv_dir)

    c4_path = os.path.join(csv_dir, 'Appendix_C4.csv')
    panel_a_path = os.path.join(csv_dir, 'Appendix_C4_PanelA_sample_counts.csv')
    panel_b_path = os.path.join(csv_dir, 'Appendix_C4_PanelB_observable_characteristics.csv')
    audit_path = os.path.join(csv_dir, 'Appendix_C4_public_data_audit.json')

    c4.to_csv(c4_path, index=False, encoding='utf-8-sig')
    panel_a.to_csv(panel_a_path, index=False, encoding='utf-8-sig')
    panel_b.to_csv(panel_b_path, index=False, encoding='utf-8-sig')

    audit = {
        'source_files': {
            'raw_price_file': 'name_price_all_anonymized.xlsx',
            'final_price_file': 'name_price_anonymized.xlsx',
            'kg_nodes': 'neo4j_export/nodes_dataproduct.csv',
            'application_edges': 'neo4j_export/rel_applied_to.csv',
            'source_industry_edges': 'neo4j_export/rel_source_industry.csv',
        },
        'counts': {
            'kg_universe': kg_n,
            'raw_price_rows': raw_rows,
            'unique_price_observed': price_observed_n,
            'without_standardizable_posted_price': no_standard_price_n,
            'zero_price_unique': zero_unique_n,
            'upper_tail_ge_300_unique': upper_unique_n,
            'price_observed_excluded_by_cleaning': price_cleaning_excluded_n,
            'final_model_sample': final_n,
            'other_kg_listings': other_kg_n,
            'final_sample_matched_to_kg': len(final_ids),
        },
        'design': {
            'panel_a': (
                'Count-based posted-price availability and sample-construction audit. '
                'The 34 EXCLUDED_* audit placeholders are not mapped back to KG nodes.'
            ),
            'panel_b': (
                'Non-overlapping structural comparison of the final 521 estimation '
                'sample with the other 6,165 KG listings.'
            ),
            'other_kg_composition': (
                '6,131 listings without standardizable posted-price information + '
                '34 price-observed listings excluded by price cleaning.'
            ),
            'smd_definition': (
                'Difference in group means divided by the square root of the average '
                'of the two group variances. SMD is not reported for medians.'
            ),
        },
    }
    with open(audit_path, 'w', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    # Manuscript Notes are intentionally not generated here; wording remains author-editable.

    return panel_a, panel_b, c4_path


def _assert_public_privacy(any_root: str) -> None:
    """
    Public-release privacy gate. The GitHub reproduction must never consume or expose
    plaintext product descriptions. Public nodes may keep desc_anon empty (preferred)
    or an opaque token such as DESC_xxx; free text is rejected.
    """
    root = Path(any_root).resolve()
    nodes_path = root / "neo4j_export" / "nodes_dataproduct.csv"
    if nodes_path.exists():
        n = pd.read_csv(nodes_path, dtype=str).fillna("")
        if "desc_anon" in n.columns:
            vals = n["desc_anon"].astype(str).str.strip()
            nonempty = vals[vals.ne("")]
            if len(nonempty):
                ok = nonempty.str.fullmatch(r"DESC_[A-Za-z0-9_-]+")
                if not bool(ok.all()):
                    raise RuntimeError(
                        "PUBLIC PRIVACY CHECK FAILED: nodes_dataproduct.csv contains non-empty "
                        "desc_anon values that are not opaque DESC_* tokens. Plaintext descriptions "
                        "must never be committed to the public reproduction package. Keep desc_anon "
                        "empty (preferred) and publish only verified numeric strict fold caches."
                    )

    forbidden = [
        "original_product_descriptions.csv",
        "original_product_descriptions.xlsx",
        "original_product_descriptions_audit.csv",
        "STRICT_CV_TEXT_EXPORT_MANIFEST.json",
    ]
    leaked = [str(root / name) for name in forbidden if (root / name).exists()]
    if leaked:
        raise RuntimeError(
            "PUBLIC PRIVACY CHECK FAILED: private text/audit files were found inside the public anymous directory:\n  "
            + "\n  ".join(leaked)
            + "\nMove these files to ./private (gitignored)."
        )


def _write_strict_cv_status(out_dir: str, status: str, reason: str, strict_applied: bool) -> Path:
    """Write an explicit machine-readable status so a public run never silently downgrades strict CV."""
    diag = Path(out_dir) / 'diagnostics_fully_strict_cv'
    diag.mkdir(parents=True, exist_ok=True)
    payload = {
        'status': status,
        'strict_table4_table5_applied': bool(strict_applied),
        'reason': reason,
        'meaning': (
            'Table 4/5 are fully fold-inductive only when strict_table4_table5_applied=true. '
            'If false, the unchanged core Table 4/5 are retained and must not be described as fully strict.'
        ),
        'required_public_cache_files': [
            'anymous/STEP0_strict_textemb_cache_kfold.npz',
            'anymous/STEP0_strict_textemb_cache_groupkfold_supplier.npz',
        ],
    }
    path = diag / 'STRICT_CV_STATUS.json'
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def _filled_template_if_available(out_dir: str) -> Optional[str]:
    """Allow the generated 521-row template to be filled manually and reused directly."""
    p = Path(out_dir) / 'strict_cv_requirements' / 'STRICT_CV_REQUIRED_DESCRIPTIONS.csv'
    if not p.exists():
        return None
    try:
        x = pd.read_csv(p)
        text_col = pick_col(x, None, ['desc','description','description_text','desc_anon','简介','描述','内容'], 'description text')
        if text_col is None:
            return None
        nonempty = normalize_str(x[text_col]).ne('').sum()
        if int(nonempty) >= 2:
            return str(p)
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser(description='One-click reproduction for the revised manuscript with fully strict Table 4/5 CV.')
    ap.add_argument('--input', default='./anymous')
    ap.add_argument('--core', default='./kg_utility_reasoning.py')
    ap.add_argument('--fig_stats_script', default='./plt_price_distribution.py')
    ap.add_argument('--fig_script', default='./plt_kg_reasoning_pic.py')
    ap.add_argument('--out_dir', default='./result_kg_reproduce')
    ap.add_argument('--min_label_freq', type=int, default=3)
    ap.add_argument(
        '--description_file', default=None,
        help=(
            'PRIVATE AUTHOR MODE ONLY. Plaintext description file used solely with '
            '--prepare_strict_caches_only to generate publishable numeric fold caches. '
            'Preferred columns: dp_id + desc/description. Public one-click reproduction never reads this file.'
        )
    )
    ap.add_argument(
        '--strict_cv', choices=['auto','require','off'], default='require',
        help=(
            "auto: use fully strict Table 4/5 when verified strict caches are available; "
            "otherwise keep the unchanged core Table 4/5, write STRICT_CV_STATUS.json, and continue without pretending strictness. "
            "require (default): abort unless verified strict numeric caches exist. off: skip the strict replacement entirely."
        )
    )
    ap.add_argument('--prepare_strict_caches_only', action='store_true',
                    help='Use private original descriptions to generate the two privacy-preserving strict fold caches, then stop.')
    ap.add_argument('--skip_figures', action='store_true')
    args = ap.parse_args()

    root = find_anymous_root(args.input)
    print(f'>>> Root data directory determined as: {root}')

    if args.description_file and not args.prepare_strict_caches_only:
        raise RuntimeError(
            "For privacy, --description_file is accepted ONLY together with --prepare_strict_caches_only. "
            "The public reproduction path must use anonymized data plus verified numeric strict fold caches and must never read raw descriptions."
        )
    if not args.prepare_strict_caches_only:
        _assert_public_privacy(root)
    price_xlsx = os.path.join(root, 'name_price_anonymized.xlsx')
    media_xlsx = os.path.join(root, 'media_result.xlsx')
    neo_dir = os.path.join(root, 'neo4j_export')

    core = load_module('kg_core_final', args.core)

    if args.prepare_strict_caches_only:
        print('>>> Preparing strict numeric text caches only (PRIVATE AUTHOR RUN)...')
        if not args.description_file:
            raise RuntimeError(
                "--prepare_strict_caches_only requires an explicit --description_file. "
                "Keep that file under ./private or outside the Git repository."
            )
        desc_path = Path(args.description_file).expanduser().resolve()
        anon_path = Path(root).resolve()
        try:
            desc_path.relative_to(anon_path)
            raise RuntimeError(
                "Privacy violation: --description_file is inside the public anymous directory. "
                "Move it to ./private or outside the repository before generating caches."
            )
        except ValueError:
            pass
        build_fully_strict_table4_table5(
            root, args.out_dir, core, description_file=str(desc_path),
            prepare_caches_only=True, allow_private_text=True
        )
        print('\n[Complete] Strict caches prepared. You may now commit ONLY the two STEP0_strict_textemb_cache_*.npz files to GitHub; do not publish the raw description file.')
        return

    print('>>> [1/6] Running the unchanged core reproduction pipeline...')
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

    strict_applied = False
    strict_reason = ''
    if args.strict_cv == 'off':
        strict_reason = 'Strict CV replacement disabled by --strict_cv off; unchanged core Table 4/5 retained.'
        status_path = _write_strict_cv_status(args.out_dir, 'OFF', strict_reason, False)
        print(f'>>> [2/6] Strict Table 4/5 replacement skipped. Status: {status_path}')
    else:
        print('>>> [2/6] Attempting FULL end-to-end fold-inductive Table 4 / Table 5...')
        try:
            build_fully_strict_table4_table5(
                root, args.out_dir, core, description_file=None,
                prepare_caches_only=False, allow_private_text=False
            )
            strict_applied = True
            strict_reason = 'Fully strict Table 4/5 generated from verified privacy-preserving fold-specific numeric caches; public run used no raw descriptions.'
            status_path = _write_strict_cv_status(args.out_dir, 'FULLY_STRICT', strict_reason, True)
            print(f'>>> [STRICT CV] Status written: {status_path}')
        except RuntimeError as e:
            strict_reason = str(e)
            if args.strict_cv == 'require':
                _write_strict_cv_status(args.out_dir, 'MISSING_PREREQUISITES', strict_reason, False)
                raise
            # AUTO mode: do not silently fabricate a strict GroupKFold representation.
            # Keep the original core Table 4/5 and continue the rest of the one-click reproduction.
            status_path = _write_strict_cv_status(args.out_dir, 'NOT_AVAILABLE_PUBLIC_DATA_ONLY', strict_reason, False)
            print('\n' + '='*88)
            print('STRICT CV COULD NOT BE GENERATED FROM THE CURRENT PUBLIC DATA, BUT THE RUN WILL CONTINUE.')
            print('The unchanged core Table 4 / Table 5 are retained. They MUST NOT be labelled fully strict.')
            print('To obtain fully strict Table 4 / Table 5, generate the two caches in a PRIVATE author run,')
            print('then add ONLY the verified strict numeric cache files to anymous/.')
            print(f'Status/details: {status_path}')
            print('='*88 + '\n')

    print('>>> [3/6] Building Appendix C/D audit tables...')
    app_manifest = build_appendix_tables(root, args.out_dir)

    print('>>> [4/6] Building Reviewer #2(i) Appendix C.4 from existing public anonymized files...')
    _, _, c4_csv = build_appendix_c4_sample_comparison(root, args.out_dir)
    print(f'      Appendix C.4 CSV: {c4_csv}')
    print('      Panel A: 555 with standardizable posted-price information vs 6,131 without (count audit).')
    print('      Panel B: final 521 modeling listings vs other 6,165 KG listings (observable KG characteristics).')

    if not args.skip_figures:
        print('>>> [5/6] Reproducing figures...')
        subprocess.run([sys.executable, args.fig_stats_script, '--out_dir', args.out_dir, '--fig_dir', os.path.join(args.out_dir, 'Figs')], check=True)
        subprocess.run([sys.executable, args.fig_script, '--out_dir', args.out_dir, '--fig_dir', os.path.join(args.out_dir, 'Figs')], check=True)
    else:
        print('>>> [5/6] Figures skipped by request (--skip_figures).')

    print('>>> [6/6] Writing main + Appendix workbooks in the current Word-table style...')
    merged = merge_excel(args.out_dir, app_manifest)
    print('\n[Complete] One-click reproduction finished.')
    print(f'Final combined workbook: {merged}')
    print(f'Main tables: {os.path.join(args.out_dir, "table_merge", "Paper_Main_Tables_Style.xlsx")}')
    print(f'Appendix tables: {os.path.join(args.out_dir, "table_merge", "Paper_Appendix_Tables_Style.xlsx")}')
    if strict_applied:
        print('Final manuscript Table 4 / Table 5 are the fully strict fold-inductive versions.')
    else:
        print('WARNING: fully strict Table 4 / Table 5 were NOT available from the current public inputs; unchanged core Table 4 / Table 5 were retained.')
        print('See result_kg_reproduce/diagnostics_fully_strict_cv/STRICT_CV_STATUS.json before describing the CV implementation in the manuscript.')
    print('Only one new manuscript table is added relative to the prior table set: Appendix C.4.')


if __name__ == '__main__':
    main()
