#!/usr/bin/env python3
"""
Data ingestion and parquet builder for Macro & Quant Trading Research.
Collects, structures, and converts public macro & market datasets into Parquet format.
"""

import json
import time
import os
from datetime import datetime, timezone
import requests
import polars as pl
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MACRO_DIR = os.path.join(DATA_DIR, "macro")
MARKET_DIR = os.path.join(DATA_DIR, "market")
FIN_DIR = os.path.join(DATA_DIR, "financials")
CROSS_DIR = os.path.join(DATA_DIR, "cross_asset")

STRUCT_DIR = os.path.join(DATA_DIR, "market_structure")
LAZY_DATA_DIR = "/home/kim/Documents/vietquant/lazy-researcher/datasets/vn_market_structure"

for d in [MACRO_DIR, MARKET_DIR, FIN_DIR, CROSS_DIR, STRUCT_DIR]:
    os.makedirs(d, exist_ok=True)


def build_market_structure():
    print(">>> 5. Processing Market Structure & Free-Float Datasets (from lazy-researcher)...")
    if not os.path.exists(LAZY_DATA_DIR):
        print(f"    ! Warning: {LAZY_DATA_DIR} not found.")
        return

    # 5.1 Full free-float dataset
    ff_path = os.path.join(LAZY_DATA_DIR, "free_float.csv")
    if os.path.exists(ff_path):
        df_ff = pl.read_csv(ff_path)
        df_ff.write_parquet(os.path.join(STRUCT_DIR, "free_float.parquet"))
        print(f"    - Saved free_float.parquet ({len(df_ff):,} rows, {df_ff['symbol'].n_unique()} tickers across {df_ff['index'].n_unique()} indices)")

        # VN30 specific free-float slice
        df_vn30_ff = df_ff.filter(pl.col("index") == "VN30").sort(["symbol", "effective_date"])
        df_vn30_ff.write_parquet(os.path.join(STRUCT_DIR, "vn30_free_float_history.parquet"))
        print(f"    - Saved vn30_free_float_history.parquet ({len(df_vn30_ff):,} rows for VN30)")

    # 5.2 Observed free-float changes
    ff_changes_path = os.path.join(LAZY_DATA_DIR, "free_float_changes_observed.csv")
    if os.path.exists(ff_changes_path):
        df_ffc = pl.read_csv(ff_changes_path)
        df_ffc.write_parquet(os.path.join(STRUCT_DIR, "free_float_changes_observed.parquet"))
        print(f"    - Saved free_float_changes_observed.parquet ({len(df_ffc):,} rows)")

    # 5.3 Index constituents
    const_path = os.path.join(LAZY_DATA_DIR, "index_constituents.csv")
    if os.path.exists(const_path):
        df_const = pl.read_csv(const_path)
        df_const.write_parquet(os.path.join(STRUCT_DIR, "index_constituents.parquet"))
        print(f"    - Saved index_constituents.parquet ({len(df_const):,} rows)")

    # 5.4 Basket history
    basket_path = os.path.join(LAZY_DATA_DIR, "basket_history.csv")
    if os.path.exists(basket_path):
        df_basket = pl.read_csv(basket_path)
        df_basket.write_parquet(os.path.join(STRUCT_DIR, "basket_history.parquet"))
        print(f"    - Saved basket_history.parquet ({len(df_basket):,} rows)")
for d in [MACRO_DIR, MARKET_DIR, FIN_DIR, CROSS_DIR]:
    os.makedirs(d, exist_ok=True)


def build_widata_macro():
    print(">>> 1. Processing WiData Macro Datasets...")
    
    # 1.1 All raw macro series (long table)
    with open('/tmp/widata_cleaned_macro_series.json', 'r') as f:
        raw_series = json.load(f)

    long_rows = []
    for k, v in raw_series.items():
        col = v.get('column_name', k)
        tbl = v.get('table_name', '')
        tf = v.get('time_frame', 'daily')
        vt = v.get('value_type', 'value')
        for pt in v.get('data', []):
            val = float(pt['value']) if pt['value'] is not None else None
            long_rows.append({
                'series_id': k,
                'table_name': str(tbl),
                'column_name': str(col),
                'time_frame': str(tf),
                'value_type': str(vt),
                'timestamp': int(pt['timestamp']),
                'date': pt['date'],
                'value': val
            })

    df_long = pl.DataFrame(long_rows)
    df_long.write_parquet(os.path.join(MACRO_DIR, "widata_macro_all_series.parquet"))
    print(f"    - Saved widata_macro_all_series.parquet ({len(df_long):,} rows, {df_long['series_id'].n_unique()} series)")

    # 1.2 Vietnam Daily Monetary & Financial Conditions
    # Map key series into clean daily table
    vn_daily_series = {
        'macro_269_lai_suat_lien_nh_on': 'interbank_rate_on',
        'macro_250_lai_suat_dieu_hanh_tai_cap_von': 'sbv_refinancing_rate',
        'macro_250_lai_suat_dieu_hanh_chiet_khau': 'sbv_discount_rate',
        'macro_269_usd_nhtm_ban_ra': 'usdvnd_commercial_sell',
        'macro_269_usd_nhtm_mua_vao': 'usdvnd_commercial_buy',
        'macro_269_usd_nhnn_trung_tam': 'usdvnd_central_rate',
        'macro_269_usd_tu_do_ban_ra': 'usdvnd_freemarket_sell',
        'macro_410_vietnam_1y': 'vn_govbond_yield_1y',
        'macro_410_vietnam_10y': 'vn_govbond_yield_10y',
        'macro_302_nhom_mbb_acb_tcb_lai_suat_ky_han_12_thang': 'bank_deposit_rate_12m',
        'macro_269_bom_hut_rong_tong_luu_hanh': 'sbv_net_liquidity_injection',
        'macro_269_omo_dang_luu_hanh': 'omo_outstanding_balance',
        'macro_269_sbv_bills_dang_luu_hanh': 'sbv_bills_outstanding_balance',
        'macro_475_gia_vang_trong_nuoc_mua_vao': 'gold_sjc_domestic_vnd',
        'macro_475_gia_vang_the_gioi_dieu_chinh': 'gold_world_usd_adjusted'
    }

    sub_daily = df_long.filter(pl.col("series_id").is_in(list(vn_daily_series.keys())))
    if len(sub_daily) > 0:
        pivot_daily = sub_daily.pivot(on="series_id", index="date", values="value")
        rename_map = {k: v for k, v in vn_daily_series.items() if k in pivot_daily.columns}
        pivot_daily = pivot_daily.rename(rename_map).sort("date")
        pivot_daily.write_parquet(os.path.join(MACRO_DIR, "vietnam_monetary_daily.parquet"))
        print(f"    - Saved vietnam_monetary_daily.parquet ({len(pivot_daily):,} rows, {len(pivot_daily.columns)-1} features)")

    # 1.3 Vietnam Monthly Real Economy & Macro Series
    vn_monthly_series = {
        'vn_pmi_monthly': 'pmi_manufacturing',
        'macro_265_giatri': 'pmi_manufacturing_widata',
        'macro_225_cpi': 'cpi_index',
        'macro_225_cpi_yoy': 'cpi_yoy',
        'macro_225_thuc_pham': 'cpi_food',
        'macro_225_giao_thong': 'cpi_transport',
        'macro_225_nha_o_vlxd': 'cpi_housing',
        'macro_248_toan_nganh_cong_nghiep_yoy': 'iip_industrial_production_yoy',
        'macro_248_khai_khoang_yoy': 'iip_mining_yoy',
        'macro_218_tong_yoy': 'retail_sales_yoy',
        'macro_234_von_thuc_hien_ttm': 'fdi_disbursed_ttm',
        'macro_234_von_dang_ky_ttm': 'fdi_registered_ttm'
    }

    sub_monthly = df_long.filter(pl.col("series_id").is_in(list(vn_monthly_series.keys())))
    if len(sub_monthly) > 0:
        pivot_monthly = sub_monthly.pivot(on="series_id", index="date", values="value")
        rename_map = {k: v for k, v in vn_monthly_series.items() if k in pivot_monthly.columns}
        pivot_monthly = pivot_monthly.rename(rename_map).sort("date")
        pivot_monthly.write_parquet(os.path.join(MACRO_DIR, "vietnam_macro_monthly.parquet"))
        print(f"    - Saved vietnam_macro_monthly.parquet ({len(pivot_monthly):,} rows, {len(pivot_monthly.columns)-1} features)")

    # 1.4 US Macro Series
    us_series = {
        'macro_111_dff': 'us_fed_funds_rate_dff',
        'macro_111_dfedtaru': 'us_fed_target_upper',
        'macro_111_dgs3mo': 'us_treasury_3m',
        'macro_111_dgs2': 'us_treasury_2y',
        'macro_111_dgs10': 'us_treasury_10y',
        'macro_111_sofrvol': 'us_sofr_volume',
        'macro_111_effrvol': 'us_effr_volume',
        'macro_99_walcl': 'us_fed_total_assets_walcl',
        'macro_117_m2sl_yoy': 'us_m2_money_supply_yoy'
    }
    sub_us = df_long.filter(pl.col("series_id").is_in(list(us_series.keys())))
    if len(sub_us) > 0:
        pivot_us = sub_us.pivot(on="series_id", index="date", values="value")
        rename_map = {k: v for k, v in us_series.items() if k in pivot_us.columns}
        pivot_us = pivot_us.rename(rename_map).sort("date")
        pivot_us.write_parquet(os.path.join(MACRO_DIR, "us_macro_daily.parquet"))
        print(f"    - Saved us_macro_daily.parquet ({len(pivot_us):,} rows, {len(pivot_us.columns)-1} features)")

    # 1.5 Market Valuation P/E
    if os.path.exists('/tmp/widata_vnindex_pe_daily.json'):
        with open('/tmp/widata_vnindex_pe_daily.json') as f:
            pe_data = json.load(f)
        df_pe = pl.DataFrame(pe_data).rename({"ngay": "date", "mack": "symbol"}).sort("date")
        df_pe.write_parquet(os.path.join(MACRO_DIR, "vietnam_market_pe_daily.parquet"))
        print(f"    - Saved vietnam_market_pe_daily.parquet ({len(df_pe):,} rows)")


def build_financials():
    print(">>> 2. Processing Corporate Financial Results (KQKD)...")
    if os.path.exists('/tmp/kqkd_q3_2026.json'):
        with open('/tmp/kqkd_q3_2026.json') as f:
            kqkd_data = json.load(f)
        df_fin = pl.DataFrame(kqkd_data).rename({
            "mack": "symbol",
            "san": "exchange",
            "nganhcap4_vi": "industry_level4",
            "doanhthu": "revenue_bn_vnd",
            "doanhthu_lkdn": "revenue_ytd_bn_vnd",
            "doanhthu_ttckn": "revenue_growth_yoy_pct",
            "doanhthu_ttckq": "revenue_growth_qoq_pct",
            "loinhuan": "net_profit_bn_vnd",
            "loinhuan_lkdn": "net_profit_ytd_bn_vnd",
            "loinhuan_ttckn": "profit_growth_yoy_pct",
            "loinhuan_ttckq": "profit_growth_qoq_pct",
            "loinhuan_htkh": "target_plan_completion_pct",
            "pe": "pe_ratio",
            "vonhoa": "market_cap_bn_vnd",
            "close": "close_price",
            "smg": "smg_score",
            "trungbinh_vol_15ngay": "avg_vol_15d_kshares",
            "ngayup": "report_date"
        })
        df_fin.write_parquet(os.path.join(FIN_DIR, "vn_listed_q3_2026_financials.parquet"))
        print(f"    - Saved vn_listed_q3_2026_financials.parquet ({len(df_fin):,} companies)")


def fetch_vndirect_series(symbol, start_ts, end_ts):
    url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution=D&symbol={symbol}&from={start_ts}&to={end_ts}"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    if resp.status_code == 200:
        data = resp.json()
        if data.get("s") == "ok" or "t" in data:
            timestamps = data["t"]
            records = []
            for i in range(len(timestamps)):
                d_str = datetime.fromtimestamp(timestamps[i]).strftime("%Y-%m-%d")
                records.append({
                    "symbol": symbol,
                    "date": d_str,
                    "timestamp": timestamps[i],
                    "open": float(data["o"][i]),
                    "high": float(data["h"][i]),
                    "low": float(data["l"][i]),
                    "close": float(data["c"][i]),
                    "volume": float(data["v"][i])
                })
            return records
    return []


def build_market_data():
    print(">>> 3. Fetching Market & Derivative Data from VNDirect Open API...")
    end_ts = int(time.time())
    start_ts = end_ts - 365 * 11 * 86400 # 11 years (2015 - 2026)

    # 3.1 Indices: VNINDEX & VN30
    indices_records = []
    for idx_sym in ["VNINDEX", "VN30"]:
        recs = fetch_vndirect_series(idx_sym, start_ts, end_ts)
        print(f"    - Fetched {idx_sym}: {len(recs)} trading days")
        indices_records.extend(recs)
        time.sleep(0.5)

    if indices_records:
        df_idx = pl.DataFrame(indices_records).sort(["symbol", "date"])
        df_idx.write_parquet(os.path.join(MARKET_DIR, "vn_indices_daily.parquet"))
        print(f"    - Saved vn_indices_daily.parquet ({len(df_idx):,} rows)")

    # 3.2 Derivatives: VN30F1M
    f1m_records = fetch_vndirect_series("VN30F1M", start_ts, end_ts)
    print(f"    - Fetched VN30F1M: {len(f1m_records)} trading days")
    if f1m_records:
        df_f1m = pl.DataFrame(f1m_records).sort("date")
        df_f1m.write_parquet(os.path.join(MARKET_DIR, "vn30f1m_derivatives_daily.parquet"))
        print(f"    - Saved vn30f1m_derivatives_daily.parquet ({len(df_f1m):,} rows)")

    # 3.3 Leading Sector Constituent Equities
    top_stocks = [
        # Banking
        "VCB", "BID", "CTG", "TCB", "MBB", "VPB", "ACB",
        # Steel & Industrial
        "HPG", "HSG",
        # Tech & Telecom
        "FPT",
        # Retail & Consumer
        "MWG", "MSN", "VNM", "DGW",
        # Securities
        "SSI", "VND", "VCI", "HCM",
        # Real Estate
        "VHM", "VIC", "KDH", "NVL",
        # Energy & Utilities
        "GAS", "PVD", "PVS", "REE"
    ]

    equity_records = []
    for stock in top_stocks:
        recs = fetch_vndirect_series(stock, start_ts, end_ts)
        print(f"      * {stock}: {len(recs)} rows")
        equity_records.extend(recs)
        time.sleep(0.3)

    if equity_records:
        df_eq = pl.DataFrame(equity_records).sort(["symbol", "date"])
        df_eq.write_parquet(os.path.join(MARKET_DIR, "vn_top_equities_daily.parquet"))
        print(f"    - Saved vn_top_equities_daily.parquet ({len(df_eq):,} rows across {len(top_stocks)} tickers)")


def build_global_cross_asset():
    print(">>> 4. Fetching Global Macro & Cross-Asset Data from Yahoo Finance...")
    tickers = {
        "^GSPC": "sp500",
        "^DJI": "dow_jones",
        "^IXIC": "nasdaq",
        "^TNX": "us_treasury_10y_yield",
        "DX-Y.NYB": "dollar_index_dxy",
        "CL=F": "crude_oil_wti",
        "GC=F": "gold_futures",
        "^VIX": "cboe_vix",
        "BTC-USD": "bitcoin_usd",
        "VND=X": "usdvnd_fx_rate"
    }

    cross_records = []
    for tick, label in tickers.items():
        try:
            t = yf.Ticker(tick)
            h = t.history(period="10y")
            if len(h) > 0:
                print(f"    - {tick} ({label}): {len(h)} rows (latest: {h.index[-1].strftime('%Y-%m-%d')})")
                for dt, row in h.iterrows():
                    cross_records.append({
                        "ticker": tick,
                        "asset_name": label,
                        "date": dt.strftime("%Y-%m-%d"),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": float(row["Volume"]) if "Volume" in row and row["Volume"] is not None else None
                    })
        except Exception as e:
            print(f"    ! Error fetching {tick}: {e}")

    if cross_records:
        df_cross = pl.DataFrame(cross_records).sort(["asset_name", "date"])
        df_cross.write_parquet(os.path.join(CROSS_DIR, "global_macro_daily.parquet"))
        print(f"    - Saved global_macro_daily.parquet ({len(df_cross):,} rows across {len(tickers)} global assets)")

def build_sectors_and_industries():
    print(">>> 6. Fetching Industry Sectors & Stock Classifications (ICB)...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    # 6.1 Fetch ICB Industry Definitions
    url_ind = "https://api-finfo.vndirect.com.vn/v4/industry_classification?size=1000"
    try:
        res_ind = requests.get(url_ind, headers=headers, timeout=15).json().get("data", [])
        icb_rows = []
        stock_to_icb = {}
        for item in res_ind:
            lvl = str(item.get("industryLevel", ""))
            code = str(item.get("industryCode", ""))
            parent = str(item.get("higherLevelCode", "")) if item.get("higherLevelCode") else None
            en = item.get("englishName", "").strip()
            vi = item.get("vietnameseName", "").strip()
            cnt = int(item.get("totalCount", 0))
            raw_codes = item.get("codeList", "")
            code_list = [c.strip() for c in raw_codes.split(",") if c.strip()]
            
            icb_rows.append({
                "industry_code": code,
                "industry_level": int(lvl) if lvl.isdigit() else None,
                "higher_level_code": parent,
                "english_name": en,
                "vietnamese_name": vi,
                "total_count": cnt,
                "code_list": raw_codes
            })
            
            for ticker in code_list:
                if ticker not in stock_to_icb:
                    stock_to_icb[ticker] = {}
                stock_to_icb[ticker][f"icb_level{lvl}_code"] = code
                stock_to_icb[ticker][f"icb_level{lvl}_en"] = en
                stock_to_icb[ticker][f"icb_level{lvl}_vi"] = vi

        df_icb = pl.DataFrame(icb_rows)
        df_icb.write_parquet(os.path.join(MARKET_DIR, "vn_industry_sectors_icb.parquet"))
        print(f"    - Saved vn_industry_sectors_icb.parquet ({len(df_icb)} ICB definitions)")
    except Exception as e:
        print(f"    ! Error fetching ICB definitions: {e}")
        stock_to_icb = {}

    # 6.2 Fetch Listed Stocks & Map to Sector Hierarchy
    url_stocks = "https://api-finfo.vndirect.com.vn/v4/stocks?q=type:STOCK~floor:HOSE,HNX,UPCOM&size=3000"
    try:
        res_stocks = requests.get(url_stocks, headers=headers, timeout=15).json().get("data", [])
        stock_rows = []
        for s in res_stocks:
            ticker = s.get("code")
            icb_info = stock_to_icb.get(ticker, {})
            stock_rows.append({
                "symbol": ticker,
                "company_name": s.get("companyName"),
                "company_name_eng": s.get("companyNameEng"),
                "short_name": s.get("shortName"),
                "floor": s.get("floor"),
                "isin": s.get("isin"),
                "status": s.get("status"),
                "listed_date": s.get("listedDate"),
                "icb_level1_code": icb_info.get("icb_level1_code"),
                "icb_level1_en": icb_info.get("icb_level1_en"),
                "icb_level1_vi": icb_info.get("icb_level1_vi"),
                "icb_level2_code": icb_info.get("icb_level2_code"),
                "icb_level2_en": icb_info.get("icb_level2_en"),
                "icb_level2_vi": icb_info.get("icb_level2_vi"),
                "icb_level3_code": icb_info.get("icb_level3_code"),
                "icb_level3_en": icb_info.get("icb_level3_en"),
                "icb_level3_vi": icb_info.get("icb_level3_vi"),
                "icb_level4_code": icb_info.get("icb_level4_code"),
                "icb_level4_en": icb_info.get("icb_level4_en"),
                "icb_level4_vi": icb_info.get("icb_level4_vi"),
                "tax_code": s.get("taxCode"),
                "company_id": s.get("companyId"),
            })
        df_stocks = pl.DataFrame(stock_rows)
        df_stocks.write_parquet(os.path.join(MARKET_DIR, "vn_stock_industry_classification.parquet"))
        print(f"    - Saved vn_stock_industry_classification.parquet ({len(df_stocks)} listed stocks mapped)")
    except Exception as e:
        print(f"    ! Error fetching listed stocks: {e}")

    # 6.3 Fetch Historical Daily Sector Indices
    sector_indices = [
        "VNREAL", "VNFIN", "VNMAT", "VNIND", "VNIT", 
        "VNENE", "VNUTI", "VNCONS", "VNHEAL", "VNFINLEAD", "VNDIAMOND", "VNSI"
    ]
    end_time = int(time.time())
    start_time = end_time - 365 * 10 * 86400
    all_sector_bars = []
    for idx in sector_indices:
        url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution=D&symbol={idx}&from={start_time}&to={end_time}"
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                d = r.json()
                if isinstance(d, dict) and "t" in d and len(d["t"]) > 0:
                    for i in range(len(d["t"])):
                        all_sector_bars.append({
                            "symbol": idx,
                            "date": datetime.fromtimestamp(d["t"][i], timezone.utc).strftime("%Y-%m-%d"),
                            "timestamp": int(d["t"][i]),
                            "open": float(d["o"][i]),
                            "high": float(d["h"][i]),
                            "low": float(d["l"][i]),
                            "close": float(d["c"][i]),
                            "volume": float(d["v"][i])
                        })
        except Exception as e:
            print(f"    ! Error fetching sector index {idx}: {e}")

    if all_sector_bars:
        df_sec_idx = pl.DataFrame(all_sector_bars).sort(["symbol", "date"])
        df_sec_idx.write_parquet(os.path.join(MARKET_DIR, "vn_sector_indices_daily.parquet"))
        print(f"    - Saved vn_sector_indices_daily.parquet ({len(df_sec_idx):,} rows across {df_sec_idx['symbol'].n_unique()} sector indices)")

def build_corporate_and_microstructure():
    print(">>> 7. Ingesting Shareholders, Subsidiaries, Capital History, Foreign & Supply/Demand...")
    try:
        from scripts.ingest_corporate_and_microstructure import run_all
        run_all()
    except Exception as e:
        print(f"    ! Error running corporate & microstructure ingestion: {e}")


if __name__ == "__main__":
    print("Starting Data Pipeline Build...")
    start_time = time.time()
    build_widata_macro()
    build_financials()
    build_market_data()
    build_global_cross_asset()
    build_market_structure()
    build_sectors_and_industries()
    build_corporate_and_microstructure()
    elapsed = time.time() - start_time
    print(f"\nAll datasets built and saved to Parquet successfully in {elapsed:.2f} seconds!")
