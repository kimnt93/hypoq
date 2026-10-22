#!/usr/bin/env python3
"""
Comprehensive Listed Company Fundamentals & Financial Reporting Ingestion Pipeline.
Extracts Point-in-Time official financial reporting data:
1. Financial Accounting Taxonomy & Line Items Dictionary (Bảng danh mục chỉ tiêu BCTC TT202, TT49, TT334, TT199)
2. Financial Statements (Balance Sheet, Income Statement, Cash Flow Statements across Quarters and Years)
3. Financial Metrics & Key Financial Ratios (Chỉ số tài chính: P/E, P/B, ROE, ROA, Biên lợi nhuận, Đòn bẩy nợ, v.v.)
"""

import os
import time
import concurrent.futures
from datetime import datetime
import requests
import polars as pl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data/financials")
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 65 Core liquid constituents across all 10 ICB sectors
DEFAULT_SYMBOLS = [
    # Banking (16)
    "VCB", "BID", "CTG", "TCB", "MBB", "VPB", "ACB", "HDB", "STB", "SHB", "LPB", "VIB", "TPB", "MSB", "SSB", "EIB",
    # Real Estate & Construction (15)
    "VHM", "VIC", "VRE", "NVL", "KDH", "DIG", "DXG", "PDR", "CEO", "KBC", "CII", "CTD", "HBC", "BCM", "TCH",
    # Financial Services / Securities (8)
    "SSI", "VND", "VCI", "HCM", "MBS", "SHS", "FTS", "BSI",
    # Steel & Basic Materials (6)
    "HPG", "HSG", "NKG", "DGC", "DPM", "BMP",
    # Technology & Telecommunications (1)
    "FPT",
    # Retail & Consumer Goods (8)
    "MWG", "MSN", "VNM", "SAB", "PNJ", "KDC", "SBT", "MCH",
    # Energy, Oil & Gas, Infrastructure & Utilities (8)
    "GAS", "PLX", "PVD", "PVS", "BSR", "POW", "REE", "NT2",
    # Aviation & Logistics (3)
    "HVN", "VJC", "GMD"
]

# Standardized fundamental ratio codes
FUNDAMENTAL_RATIOS = [
    # Profitability (Khả năng sinh lời)
    "ROAE_TR_AVG5Q", "ROAA_TR_AVG5Q",
    "GROSS_MARGIN_QR", "GROSS_MARGIN_TR", "GROSS_MARGIN_YD",
    "NET_MARGIN_QR", "NET_MARGIN_TR", "NET_MARGIN_YD",
    "OPERATING_EBIT_MARGIN_QR", "OPERATING_EBIT_MARGIN_TR",
    "OPERATING_EBITDA_MARGIN_QR", "OPERATING_EBITDA_MARGIN_TR",
    # Financial Leverage & Solvency (Đòn bẩy tài chính & Khả năng thanh toán)
    "DEBT_TO_EQUITY_AQ", "EQUITY_TO_ASSET_AQ", "DEBT_EBITDA",
    "CURRENT_RATIO_AQ", "QUICK_RATIO_AQ",
    # Valuation Multiples & Per-Share Metrics (Định giá & Chỉ tiêu trên mỗi cổ phần)
    "PRICE_TO_EARNINGS", "PRICE_TO_BOOK", "PRICE_TO_SALES",
    "EPS_TR", "BVPS_CR", "DIVIDEND_YIELD", "MARKETCAP",
    # Operating Efficiency & Activity (Hiệu quả hoạt động)
    "INVENTORY_TURNOVER_TR_AVG5Q", "RECEIVABLE_TURNOVER_TR_AVG5Q", "PAYABLE_TURNOVER_TR_AVG5Q",
    # Scale & Growth (Quy mô & Tăng trưởng)
    "NET_SALES_QR", "NET_SALES_TR", "NET_SALES_TR_GRYOY",
    "TOTAL_ASSETS_AQ", "TOTAL_ASSETS_AQ_GRYOY"
]


def fetch_financial_models_taxonomy():
    print("\n>>> [1/3] Fetching Financial Accounting Taxonomy & Line Items Dictionary...")
    url = "https://api-finfo.vndirect.com.vn/v4/financial_models?size=10000"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            data = r.json().get("data", [])
            records = []
            seen_items = set()
            for m in data:
                icode = int(m.get("itemCode", 0))
                mtype = int(m.get("modelType", 0)) if m.get("modelType") is not None else 0
                key = (mtype, icode)
                if key in seen_items or icode == 0:
                    continue
                seen_items.add(key)
                records.append({
                    "model_type": mtype,
                    "model_type_name": str(m.get("modelTypeName", "")).strip(),
                    "model_vn_desc": str(m.get("modelVnDesc", "")).strip(),
                    "model_en_desc": str(m.get("modelEnDesc", "")).strip(),
                    "company_form": str(m.get("companyForm", "")).strip(),
                    "accounting_circular": str(m.get("note", "")).strip(),
                    "item_code": icode,
                    "item_vn_name": str(m.get("itemVnName", "")).strip(),
                    "item_en_name": str(m.get("itemEnName", "")).strip(),
                    "display_order": int(m.get("displayOrder", 0)) if m.get("displayOrder") is not None else 0,
                    "display_level": int(m.get("displayLevel", 1)) if m.get("displayLevel") is not None else 1,
                    "form_type": str(m.get("formType", "")).strip()
                })

            schema = {
                "model_type": pl.Int64,
                "model_type_name": pl.Utf8,
                "model_vn_desc": pl.Utf8,
                "model_en_desc": pl.Utf8,
                "company_form": pl.Utf8,
                "accounting_circular": pl.Utf8,
                "item_code": pl.Int64,
                "item_vn_name": pl.Utf8,
                "item_en_name": pl.Utf8,
                "display_order": pl.Int64,
                "display_level": pl.Int64,
                "form_type": pl.Utf8
            }
            df = pl.DataFrame(records, schema=schema).sort(["model_type", "display_order", "item_code"])
            out_path = os.path.join(DATA_DIR, "vn_financial_models_taxonomy.parquet")
            df.write_parquet(out_path)
            print(f"    - Saved vn_financial_models_taxonomy.parquet ({len(df):,} line items across {df['model_type'].n_unique()} accounting models)")
            return df
    except Exception as e:
        print(f"    ! Error fetching financial models: {e}")
    return pl.DataFrame()


def fetch_financial_ratios(symbols=None):
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    print(f"\n>>> [2/3] Fetching Financial Ratios & Fundamental Metrics for {len(symbols)} symbols...")

    ratio_query = ",".join(FUNDAMENTAL_RATIOS)
    all_rows = []

    def _fetch_sym(sym):
        url = f"https://api-finfo.vndirect.com.vn/v4/ratios?q=code:{sym}~ratioCode:{ratio_query}&size=3000"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                data = r.json().get("data", [])
                out = []
                for item in data:
                    rdate = item.get("reportDate")
                    if not rdate:
                        continue
                    dt = datetime.strptime(rdate[:10], "%Y-%m-%d")
                    year = dt.year
                    month = dt.month
                    quarter = (month - 1) // 3 + 1
                    
                    val = float(item.get("value")) if item.get("value") is not None else None
                    if val is None:
                        continue
                    
                    rcode = item.get("ratioCode", "")
                    cat = "OTHER"
                    if "MARGIN" in rcode or "ROA" in rcode or "ROE" in rcode:
                        cat = "PROFITABILITY"
                    elif "DEBT" in rcode or "EQUITY" in rcode:
                        cat = "LEVERAGE"
                    elif "RATIO" in rcode:
                        cat = "LIQUIDITY"
                    elif "PRICE" in rcode or "EPS" in rcode or "BVPS" in rcode or "MARKETCAP" in rcode:
                        cat = "VALUATION"
                    elif "TURNOVER" in rcode:
                        cat = "EFFICIENCY"
                    elif "SALES" in rcode or "ASSETS" in rcode:
                        cat = "SCALE_GROWTH"

                    out.append({
                        "symbol": sym,
                        "report_date": rdate[:10],
                        "year": int(year),
                        "quarter": int(quarter),
                        "ratio_code": rcode,
                        "ratio_name": str(item.get("itemName", "")).strip(),
                        "category": cat,
                        "value": float(val)
                    })
                return out
        except Exception as e:
            print(f"    ! Error fetching ratios for {sym}: {e}")
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_sym, s): s for s in symbols}
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                all_rows.extend(res)

    if all_rows:
        schema = {
            "symbol": pl.Utf8,
            "report_date": pl.Utf8,
            "year": pl.Int32,
            "quarter": pl.Int32,
            "ratio_code": pl.Utf8,
            "ratio_name": pl.Utf8,
            "category": pl.Utf8,
            "value": pl.Float64
        }
        df = pl.DataFrame(all_rows, schema=schema).sort(["symbol", "report_date", "ratio_code"])
        out_path = os.path.join(DATA_DIR, "vn_financial_ratios_quarterly_yearly.parquet")
        df.write_parquet(out_path)
        print(f"    - Saved vn_financial_ratios_quarterly_yearly.parquet ({len(df):,} rows across {df['symbol'].n_unique()} symbols)")
        return df
    return pl.DataFrame()


def fetch_financial_statements(symbols=None):
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    print(f"\n>>> [3/3] Fetching Financial Statements (Balance Sheet, Income, Cash Flow) for {len(symbols)} symbols...")

    # Load taxonomy dictionary for item descriptions
    taxo_path = os.path.join(DATA_DIR, "vn_financial_models_taxonomy.parquet")
    models_dict = {}
    if os.path.exists(taxo_path):
        df_taxo = pl.read_parquet(taxo_path)
        for row in df_taxo.iter_rows(named=True):
            icode = row["item_code"]
            if icode not in models_dict:
                models_dict[icode] = {
                    "statement_type": row["model_type_name"],
                    "item_name_vn": row["item_vn_name"],
                    "item_name_en": row["item_en_name"],
                    "display_level": row["display_level"]
                }

    all_rows = []

    def _fetch_sym(sym):
        # Fetch quarterly and annual statements since 2018
        url = f"https://api-finfo.vndirect.com.vn/v4/financial_statements?q=code:{sym}~fiscalDate:gte:2018-01-01&size=5000"
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                data = r.json().get("data", [])
                out = []
                for item in data:
                    fdate = item.get("fiscalDate")
                    if not fdate:
                        continue
                    dt = datetime.strptime(fdate[:10], "%Y-%m-%d")
                    year = dt.year
                    month = dt.month
                    quarter = (month - 1) // 3 + 1
                    
                    rtype = str(item.get("reportType", "QUARTER")).strip()
                    if rtype == "YEAR":
                        quarter = 0  # 0 indicates Full Year Annual Report

                    val = float(item.get("numericValue")) if item.get("numericValue") is not None else None
                    if val is None:
                        continue

                    icode = int(item.get("itemCode", 0))
                    info = models_dict.get(icode, {})

                    out.append({
                        "symbol": sym,
                        "fiscal_date": fdate[:10],
                        "year": int(year),
                        "quarter": int(quarter),
                        "report_type": rtype,
                        "statement_type": info.get("statement_type", "GENERAL"),
                        "item_code": icode,
                        "item_name_vn": info.get("item_name_vn", ""),
                        "item_name_en": info.get("item_name_en", ""),
                        "display_level": int(info.get("display_level", 1)),
                        "value": float(val),
                        "model_type": int(item.get("modelType", 0)) if item.get("modelType") is not None else 0
                    })
                return out
        except Exception as e:
            print(f"    ! Error fetching statements for {sym}: {e}")
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_sym, s): s for s in symbols}
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                all_rows.extend(res)

    if all_rows:
        schema = {
            "symbol": pl.Utf8,
            "fiscal_date": pl.Utf8,
            "year": pl.Int32,
            "quarter": pl.Int32,
            "report_type": pl.Utf8,
            "statement_type": pl.Utf8,
            "item_code": pl.Int64,
            "item_name_vn": pl.Utf8,
            "item_name_en": pl.Utf8,
            "display_level": pl.Int32,
            "value": pl.Float64,
            "model_type": pl.Int64
        }
        df = pl.DataFrame(all_rows, schema=schema).sort(["symbol", "fiscal_date", "item_code"])
        out_path = os.path.join(DATA_DIR, "vn_financial_statements_quarterly_yearly.parquet")
        df.write_parquet(out_path)
        print(f"    - Saved vn_financial_statements_quarterly_yearly.parquet ({len(df):,} rows across {df['symbol'].n_unique()} symbols)")
        return df
    return pl.DataFrame()


def run_all(symbols=None):
    t0 = time.time()
    print("=" * 80)
    print("STARTING COMPANY FUNDAMENTALS & FINANCIAL REPORTING INGESTION")
    print("=" * 80)
    df_taxo = fetch_financial_models_taxonomy()
    df_ratios = fetch_financial_ratios(symbols)
    df_stmts = fetch_financial_statements(symbols)
    print("=" * 80)
    print(f"COMPLETED ALL FUNDAMENTAL DATASETS IN {time.time() - t0:.2f}s!")
    print("=" * 80)


if __name__ == "__main__":
    run_all()
