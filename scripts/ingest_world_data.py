#!/usr/bin/env python3
"""
World Comprehensive Financial & Macroeconomic Ingestion Pipeline.
Inspired by WiFeed multi-domain institutional data taxonomy, covering global markets for free:
1. Macroeconomics & Real Economy: FRED Inflation (PCE), Unemployment, Financial Conditions (NFCI), M2, Debt/GDP, Term Spreads
2. Global GICS Sector Indices: 11 Sector SPDRs (Tech, Financials, Healthcare, Energy, Discretionary, Staples, Industrials, Materials, Utilities, Real Estate, Comm)
3. Global Factors & Smart Beta: Momentum (MTUM), Quality (QUAL), Low Vol (USMV), Value (VLUE), Growth (VUG), Small-Cap (IJR), Mid-Cap (IJH), Dividend (NOBL)
4. Global Credit Spreads, Fixed Income & Maritime Freight: IG Corporate (LQD), High Yield (HYG), Emerging Debt (EMB), TIPS (TIP), Treasuries (SHY, TLT), Muni (MUB), Dry Bulk Freight (BDRY)
5. Cross-Asset Implied Volatility: S&P VIX, VIX of VIX (VVIX), Crude Oil Vol (OVX), Gold Vol (GVZ)
6. Global Corporate Fundamentals & Institutional Valuation Ratios: 40+ World Mega-Caps across US, EU, JP, KR, CN, HK (P/E, P/B, EV/EBITDA, ROE, Margin, Debt/Equity)

All datasets are saved into data/macro/, data/market/, and data/financials/ with Zstandard compression.
"""

import os
import sys
import io
import time
import requests
import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.yf_common import fetch_ticker_dataframe

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MARKET_DIR = os.path.join(DATA_DIR, "market")
MACRO_DIR = os.path.join(DATA_DIR, "macro")
FINANCIALS_DIR = os.path.join(DATA_DIR, "financials")

for d in [MARKET_DIR, MACRO_DIR, FINANCIALS_DIR]:
    os.makedirs(d, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """Downloads historical macroeconomic series from FRED without requiring an API key."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200 and "observation_date" in r.text:
            df = pd.read_csv(io.StringIO(r.text))
            df = df[df[series_id] != "."].dropna()
            df[series_id] = df[series_id].astype(float)
            df = df.rename(columns={"observation_date": "date"})
            return df
    except Exception as e:
        print(f"      ! Error fetching FRED series {series_id}: {e}")
    return pd.DataFrame()


def build_global_macro_indicators():
    print("\n>>> [1/6] Ingesting FRED World & US Macroeconomic Indicators...")
    series_map = {
        "PCEPI": "pce_headline_inflation_index",
        "UNRATE": "unemployment_rate_pct",
        "NFCI": "chicago_fed_financial_conditions_index",
        "T10YIE": "treasury_10y_breakeven_inflation",
        "T10Y2Y": "term_spread_10y_2y",
        "M2SL": "m2_money_supply_billions",
        "GFDEGDQ188S": "public_debt_pct_gdp",
        "BAMLH0A0HYM2": "high_yield_option_adjusted_spread_bps",
    }

    merged = None
    for s_id, col_name in series_map.items():
        df_s = fetch_fred_series(s_id)
        if not df_s.empty:
            df_s = df_s.rename(columns={s_id: col_name})
            if merged is None:
                merged = df_s
            else:
                merged = pd.merge(merged, df_s, on="date", how="outer")
            print(f"      + FRED {s_id} ({col_name}): {len(df_s):,} records")

    if merged is not None and not merged.empty:
        merged = merged.sort_values("date").reset_index(drop=True)
        # Forward fill low-frequency macro series (quarterly/monthly) across trading calendar
        merged = merged.ffill()
        df_pl = (
            pl.DataFrame(merged, infer_schema_length=None)
            .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
            .sort("date")
            .unique(subset=["date"], keep="last")
        )
        out_path = os.path.join(MACRO_DIR, "global_macro_indicators_monthly.parquet")
        df_pl.write_parquet(out_path, compression="zstd")
        print(f"    - Saved {out_path} ({len(df_pl):,} rows, {len(df_pl.columns)-1} macro indicators)")


fetch_yf_batch = fetch_ticker_dataframe


def build_global_sectors():
    print("\n>>> [2/6] Ingesting 11 GICS Global Sector SPDR Indices...")
    sectors = {
        "XLK": {"sector_name": "Technology Select Sector", "sector_code": "XLK"},
        "XLF": {"sector_name": "Financial Select Sector", "sector_code": "XLF"},
        "XLV": {"sector_name": "Health Care Select Sector", "sector_code": "XLV"},
        "XLE": {"sector_name": "Energy Select Sector", "sector_code": "XLE"},
        "XLY": {"sector_name": "Consumer Discretionary Select Sector", "sector_code": "XLY"},
        "XLP": {"sector_name": "Consumer Staples Select Sector", "sector_code": "XLP"},
        "XLI": {"sector_name": "Industrial Select Sector", "sector_code": "XLI"},
        "XLB": {"sector_name": "Materials Select Sector", "sector_code": "XLB"},
        "XLU": {"sector_name": "Utilities Select Sector", "sector_code": "XLU"},
        "XLRE": {"sector_name": "Real Estate Select Sector", "sector_code": "XLRE"},
        "XLC": {"sector_name": "Communication Services Select Sector", "sector_code": "XLC"},
    }
    df_sec = fetch_yf_batch(sectors, period="10y")
    out_sec = os.path.join(MARKET_DIR, "global_sectors_daily.parquet")
    df_sec.write_parquet(out_sec, compression="zstd")
    print(f"    - Saved {out_sec} ({len(df_sec):,} rows across {df_sec['symbol'].n_unique()} GICS sectors)")


def build_global_factors():
    print("\n>>> [3/6] Ingesting Global Factors & Smart Beta Indices...")
    factors = {
        "MTUM": {"factor_name": "MSCI USA Momentum Factor", "style_category": "Momentum"},
        "QUAL": {"factor_name": "MSCI USA Quality Factor", "style_category": "Quality"},
        "USMV": {"factor_name": "MSCI USA Min Vol Factor", "style_category": "Low Volatility"},
        "VLUE": {"factor_name": "MSCI USA Value Factor", "style_category": "Value"},
        "VUG": {"factor_name": "Vanguard Growth ETF", "style_category": "Growth"},
        "IJR": {"factor_name": "iShares Core S&P Small-Cap ETF", "style_category": "Size / Small-Cap"},
        "IJH": {"factor_name": "iShares Core S&P Mid-Cap ETF", "style_category": "Size / Mid-Cap"},
        "NOBL": {"factor_name": "ProShares S&P 500 Dividend Aristocrats", "style_category": "Dividend Yield"},
    }
    df_fac = fetch_yf_batch(factors, period="10y")
    out_fac = os.path.join(MARKET_DIR, "global_factors_daily.parquet")
    df_fac.write_parquet(out_fac, compression="zstd")
    print(f"    - Saved {out_fac} ({len(df_fac):,} rows across {df_fac['symbol'].n_unique()} factor indices)")


def build_global_credit_spreads():
    print("\n>>> [4/6] Ingesting Global Credit Spreads, Fixed Income & Maritime Freight...")
    credit_assets = {
        "LQD": {"asset_name": "iShares iBoxx $ Investment Grade Corporate Bond", "asset_category": "Corporate IG"},
        "HYG": {"asset_name": "iShares iBoxx $ High Yield Corporate Bond", "asset_category": "Corporate HY"},
        "EMB": {"asset_name": "iShares J.P. Morgan USD Emerging Markets Bond", "asset_category": "Emerging Market Sovereign"},
        "TIP": {"asset_name": "iShares TIPS Bond ETF", "asset_category": "Inflation-Protected TIPS"},
        "SHY": {"asset_name": "iShares 1-3 Year Treasury Bond ETF", "asset_category": "Short-Term Treasuries"},
        "TLT": {"asset_name": "iShares 20+ Year Treasury Bond ETF", "asset_category": "Long-Term Treasuries"},
        "MUB": {"asset_name": "iShares National Muni Bond ETF", "asset_category": "Municipal Debt"},
        "BDRY": {"asset_name": "Breakwave Dry Bulk Shipping ETF (Baltic Dry Proxy)", "asset_category": "Maritime Freight / Supply Chain"},
    }
    df_credit = fetch_yf_batch(credit_assets, period="10y")
    out_credit = os.path.join(MACRO_DIR, "global_credit_spreads_daily.parquet")
    df_credit.write_parquet(out_credit, compression="zstd")
    print(f"    - Saved {out_credit} ({len(df_credit):,} rows across {df_credit['symbol'].n_unique()} credit & freight assets)")


def build_cross_asset_volatility():
    print("\n>>> [5/6] Ingesting Cross-Asset Implied Volatility (VIX, VVIX, OVX, GVZ)...")
    vol_tickers = {
        "^VIX": "vix_equity",
        "^VVIX": "vvix_vol_of_vol",
        "^OVX": "ovx_crude_oil",
        "^GVZ": "gvz_gold",
    }
    raw = yf.download(list(vol_tickers.keys()), period="10y", interval="1d", progress=False)
    close_df = raw["Close"].copy() if "Close" in raw else raw

    records = []
    close_df = close_df.dropna(how="all").reset_index()
    date_col = "Date" if "Date" in close_df.columns else close_df.columns[0]

    for _, row in close_df.iterrows():
        dt_str = pd.to_datetime(row[date_col]).strftime("%Y-%m-%d")
        records.append({
            "date": dt_str,
            "vix_equity": float(row.get("^VIX")) if pd.notnull(row.get("^VIX")) else None,
            "vvix_vol_of_vol": float(row.get("^VVIX")) if pd.notnull(row.get("^VVIX")) else None,
            "ovx_crude_oil": float(row.get("^OVX")) if pd.notnull(row.get("^OVX")) else None,
            "gvz_gold": float(row.get("^GVZ")) if pd.notnull(row.get("^GVZ")) else None,
        })

    df_vol = (
        pl.DataFrame(records, infer_schema_length=None)
        .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
        .sort("date")
        .unique(subset=["date"], keep="last")
    )
    out_vol = os.path.join(MACRO_DIR, "global_cross_asset_volatility_daily.parquet")
    df_vol.write_parquet(out_vol, compression="zstd")
    print(f"    - Saved {out_vol} ({len(df_vol):,} daily volatility surface records)")


def build_global_corporate_fundamentals():
    print("\n>>> [6/6] Ingesting Global Corporate Fundamentals & Valuation Ratios...")
    global_universe = [
        # US
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "UNH", "XOM", "PG", "JNJ", "HD", "BRK-B",
        # Europe
        "ASML", "SAP", "SHEL", "NVO", "AZN", "TTE", "MC.PA", "OR.PA", "SIE.DE", "NESN.SW", "NOVN.SW",
        # Japan
        "7203.T", "6758.T", "6861.T", "8035.T", "8306.T", "9984.T", "7974.T", "9983.T",
        # South Korea
        "005930.KS", "000660.KS", "005380.KS", "373220.KS", "005490.KS", "035420.KS",
        # China / Hong Kong
        "BABA", "PDD", "JD", "BIDU", "NIO", "0700.HK", "0005.HK", "1299.HK", "3690.HK", "1810.HK"
    ]

    fundamentals = []
    print(f"    - Fetching valuation & accounting ratios across {len(global_universe)} global corporations...")
    for sym in global_universe:
        try:
            tk = yf.Ticker(sym)
            inf = tk.info
            fundamentals.append({
                "symbol": sym,
                "company_name": inf.get("shortName") or inf.get("longName") or sym,
                "country": inf.get("country", ""),
                "sector": inf.get("sector", ""),
                "industry": inf.get("industry", ""),
                "market_cap": float(inf.get("marketCap")) if inf.get("marketCap") is not None else None,
                "enterprise_value": float(inf.get("enterpriseValue")) if inf.get("enterpriseValue") is not None else None,
                "pe_trailing": float(inf.get("trailingPE")) if inf.get("trailingPE") is not None else None,
                "pe_forward": float(inf.get("forwardPE")) if inf.get("forwardPE") is not None else None,
                "pb_ratio": float(inf.get("priceToBook")) if inf.get("priceToBook") is not None else None,
                "ps_ratio": float(inf.get("priceToSalesTrailing12Months")) if inf.get("priceToSalesTrailing12Months") is not None else None,
                "ev_to_ebitda": float(inf.get("enterpriseToEbitda")) if inf.get("enterpriseToEbitda") is not None else None,
                "profit_margin": float(inf.get("profitMargins")) if inf.get("profitMargins") is not None else None,
                "operating_margin": float(inf.get("operatingMargins")) if inf.get("operatingMargins") is not None else None,
                "roe": float(inf.get("returnOnEquity")) if inf.get("returnOnEquity") is not None else None,
                "roa": float(inf.get("returnOnAssets")) if inf.get("returnOnAssets") is not None else None,
                "debt_to_equity": float(inf.get("debtToEquity")) if inf.get("debtToEquity") is not None else None,
                "dividend_yield": float(inf.get("dividendYield")) if inf.get("dividendYield") is not None else None,
                "free_cashflow": float(inf.get("freeCashflow")) if inf.get("freeCashflow") is not None else None,
                "beta": float(inf.get("beta")) if inf.get("beta") is not None else None,
            })
            print(f"      + {sym}: MarketCap={inf.get('marketCap')}, TrailingPE={inf.get('trailingPE')}, ROE={inf.get('returnOnEquity')}")
        except Exception as e:
            print(f"      ! Error fetching {sym}: {e}")

    if fundamentals:
        df_fund = pl.DataFrame(fundamentals, infer_schema_length=None).sort("market_cap", descending=True)
        out_fund = os.path.join(FINANCIALS_DIR, "global_corporate_fundamentals.parquet")
        df_fund.write_parquet(out_fund, compression="zstd")
        print(f"    - Saved {out_fund} ({len(df_fund)} global corporate profiles with 20 fundamental features)")


def run_all():
    start = time.time()
    print("=" * 85)
    print("STARTING WORLD FINANCIAL DATA TAXONOMY INGESTION (MACRO, SECTORS, FACTORS, DEBT)")
    print("=" * 85)

    build_global_macro_indicators()
    build_global_sectors()
    build_global_factors()
    build_global_credit_spreads()
    build_cross_asset_volatility()
    build_global_corporate_fundamentals()

    elapsed = time.time() - start
    print("=" * 85)
    print(f"WORLD FINANCIAL DATA INGESTION COMPLETED IN {elapsed:.2f}s!")
    print("=" * 85)


if __name__ == "__main__":
    run_all()
