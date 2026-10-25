#!/usr/bin/env python3
"""
Global Financial Markets Ingestion Pipeline.
Acquires daily historical data for Asia, US, Europe, Global Commodities, and Crypto:
1. US Benchmark Indices & Mega-Cap Equities
2. European Benchmark Indices & Mega-Cap Equities
3. Asian Benchmark Indices & Corporate Titans
4. Global Cross-Asset & Factor ETFs
5. Global Commodities (Energy, Metals, Agriculture)
6. US Yield Curve & Volatility (Treasuries 3M/5Y/10Y/30Y, VIX)
7. European Macro & Foreign Exchange (EUR/USD, GBP/USD, EUR Crosses)
8. Asian Macro & Foreign Exchange (USD/JPY, USD/CNY, USD/KRW, USD/INR, etc.)
9. Cryptocurrency Top Assets & Macro Market Indicators (BTC, ETH, SOL, Volatility, Corrs)

All datasets are saved as native Parquet files with Zstandard compression under data/.
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.yf_common import fetch_ticker_dataframe
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MARKET_DIR = os.path.join(DATA_DIR, "market")
MACRO_DIR = os.path.join(DATA_DIR, "macro")
CRYPTO_DIR = os.path.join(DATA_DIR, "crypto")

for d in [MARKET_DIR, MACRO_DIR, CRYPTO_DIR]:
    os.makedirs(d, exist_ok=True)


def build_us_markets():
    print("\n--- [1/9] Ingesting US Benchmark Indices & Mega-Cap Equities ---")
    # 1.1 US Indices
    us_indices = {
        "^GSPC": {"index_name": "S&P 500 Index", "country": "US"},
        "^NDX": {"index_name": "Nasdaq 100 Index", "country": "US"},
        "^DJI": {"index_name": "Dow Jones Industrial Average", "country": "US"},
        "^RUT": {"index_name": "Russell 2000 Small-Cap Index", "country": "US"},
        "^MID": {"index_name": "S&P MidCap 400 Index", "country": "US"},
        "^VIX": {"index_name": "CBOE Volatility Index", "country": "US"},
    }
    df_indices = fetch_ticker_dataframe(us_indices, period="10y")
    out_idx = os.path.join(MARKET_DIR, "us_indices_daily.parquet")
    df_indices.write_parquet(out_idx, compression="zstd")
    print(f"    => Saved {out_idx} ({len(df_indices):,} rows across {df_indices['symbol'].n_unique()} indices)")

    # 1.2 US Equities
    us_equities = {
        # Tech & Semis
        "AAPL": {"company_name": "Apple Inc.", "sector": "Information Technology", "country": "US"},
        "MSFT": {"company_name": "Microsoft Corporation", "sector": "Information Technology", "country": "US"},
        "NVDA": {"company_name": "NVIDIA Corporation", "sector": "Information Technology", "country": "US"},
        "AVGO": {"company_name": "Broadcom Inc.", "sector": "Information Technology", "country": "US"},
        "AMD": {"company_name": "Advanced Micro Devices Inc.", "sector": "Information Technology", "country": "US"},
        "ORCL": {"company_name": "Oracle Corporation", "sector": "Information Technology", "country": "US"},
        "CRM": {"company_name": "Salesforce Inc.", "sector": "Information Technology", "country": "US"},
        "QCOM": {"company_name": "Qualcomm Inc.", "sector": "Information Technology", "country": "US"},
        "INTC": {"company_name": "Intel Corporation", "sector": "Information Technology", "country": "US"},
        # Consumer & Communication
        "AMZN": {"company_name": "Amazon.com Inc.", "sector": "Consumer Discretionary", "country": "US"},
        "TSLA": {"company_name": "Tesla Inc.", "sector": "Consumer Discretionary", "country": "US"},
        "HD": {"company_name": "Home Depot Inc.", "sector": "Consumer Discretionary", "country": "US"},
        "COST": {"company_name": "Costco Wholesale Corp.", "sector": "Consumer Staples", "country": "US"},
        "WMT": {"company_name": "Walmart Inc.", "sector": "Consumer Staples", "country": "US"},
        "PG": {"company_name": "Procter & Gamble Co.", "sector": "Consumer Staples", "country": "US"},
        "GOOGL": {"company_name": "Alphabet Inc.", "sector": "Communication Services", "country": "US"},
        "META": {"company_name": "Meta Platforms Inc.", "sector": "Communication Services", "country": "US"},
        # Financials
        "JPM": {"company_name": "JPMorgan Chase & Co.", "sector": "Financials", "country": "US"},
        "BAC": {"company_name": "Bank of America Corp.", "sector": "Financials", "country": "US"},
        "WFC": {"company_name": "Wells Fargo & Company", "sector": "Financials", "country": "US"},
        "GS": {"company_name": "Goldman Sachs Group Inc.", "sector": "Financials", "country": "US"},
        "MS": {"company_name": "Morgan Stanley", "sector": "Financials", "country": "US"},
        "V": {"company_name": "Visa Inc.", "sector": "Financials", "country": "US"},
        "MA": {"company_name": "Mastercard Inc.", "sector": "Financials", "country": "US"},
        "BRK-B": {"company_name": "Berkshire Hathaway Inc.", "sector": "Financials", "country": "US"},
        # Healthcare
        "LLY": {"company_name": "Eli Lilly and Company", "sector": "Healthcare", "country": "US"},
        "UNH": {"company_name": "UnitedHealth Group Inc.", "sector": "Healthcare", "country": "US"},
        "JNJ": {"company_name": "Johnson & Johnson", "sector": "Healthcare", "country": "US"},
        "PFE": {"company_name": "Pfizer Inc.", "sector": "Healthcare", "country": "US"},
        "ABBV": {"company_name": "AbbVie Inc.", "sector": "Healthcare", "country": "US"},
        # Energy & Industrials
        "XOM": {"company_name": "Exxon Mobil Corporation", "sector": "Energy", "country": "US"},
        "CVX": {"company_name": "Chevron Corporation", "sector": "Energy", "country": "US"},
        "CAT": {"company_name": "Caterpillar Inc.", "sector": "Industrials", "country": "US"},
        "GE": {"company_name": "General Electric Company", "sector": "Industrials", "country": "US"},
        "BA": {"company_name": "Boeing Company", "sector": "Industrials", "country": "US"},
    }
    df_eq = fetch_ticker_dataframe(us_equities, period="10y")
    out_eq = os.path.join(MARKET_DIR, "us_equities_daily.parquet")
    df_eq.write_parquet(out_eq, compression="zstd")
    print(f"    => Saved {out_eq} ({len(df_eq):,} rows across {df_eq['symbol'].n_unique()} mega-cap stocks)")


def build_eu_markets():
    print("\n--- [2/9] Ingesting European Benchmark Indices & Blue-Chips ---")
    # 2.1 EU Indices
    eu_indices = {
        "^STOXX50E": {"index_name": "Euro Stoxx 50 Index", "country": "EU"},
        "^GDAXI": {"index_name": "DAX 40 Index", "country": "Germany"},
        "^FTSE": {"index_name": "FTSE 100 Index", "country": "United Kingdom"},
        "^FCHI": {"index_name": "CAC 40 Index", "country": "France"},
        "^IBEX": {"index_name": "IBEX 35 Index", "country": "Spain"},
        "FTSEMIB.MI": {"index_name": "FTSE MIB Index", "country": "Italy"},
        "^SSMI": {"index_name": "Swiss Market Index SMI", "country": "Switzerland"},
        "^AEX": {"index_name": "AEX Index", "country": "Netherlands"},
    }
    df_indices = fetch_ticker_dataframe(eu_indices, period="10y")
    out_idx = os.path.join(MARKET_DIR, "eu_indices_daily.parquet")
    df_indices.write_parquet(out_idx, compression="zstd")
    print(f"    => Saved {out_idx} ({len(df_indices):,} rows across {df_indices['symbol'].n_unique()} European indices)")

    # 2.2 EU Equities
    eu_equities = {
        # Netherlands & Germany
        "ASML": {"company_name": "ASML Holding N.V.", "sector": "Semiconductors", "country": "Netherlands"},
        "SAP": {"company_name": "SAP SE", "sector": "Software", "country": "Germany"},
        "SIE.DE": {"company_name": "Siemens AG", "sector": "Industrials", "country": "Germany"},
        "ALV.DE": {"company_name": "Allianz SE", "sector": "Financials / Insurance", "country": "Germany"},
        "BMW.DE": {"company_name": "Bayerische Motoren Werke (BMW)", "sector": "Automotive", "country": "Germany"},
        # United Kingdom
        "SHEL": {"company_name": "Shell plc", "sector": "Energy", "country": "United Kingdom"},
        "AZN": {"company_name": "AstraZeneca PLC", "sector": "Healthcare", "country": "United Kingdom"},
        "BP": {"company_name": "BP p.l.c.", "sector": "Energy", "country": "United Kingdom"},
        "RIO": {"company_name": "Rio Tinto Group", "sector": "Materials / Mining", "country": "United Kingdom"},
        # France
        "TTE": {"company_name": "TotalEnergies SE", "sector": "Energy", "country": "France"},
        "MC.PA": {"company_name": "LVMH Moët Hennessy", "sector": "Consumer Luxury", "country": "France"},
        "OR.PA": {"company_name": "L'Oréal S.A.", "sector": "Consumer Staples", "country": "France"},
        "AIR.PA": {"company_name": "Airbus SE", "sector": "Aerospace & Defense", "country": "France"},
        "BNP.PA": {"company_name": "BNP Paribas S.A.", "sector": "Financials", "country": "France"},
        "SAN.PA": {"company_name": "Sanofi S.A.", "sector": "Healthcare", "country": "France"},
        # Denmark & Switzerland
        "NVO": {"company_name": "Novo Nordisk A/S", "sector": "Healthcare", "country": "Denmark"},
        "NESN.SW": {"company_name": "Nestlé S.A.", "sector": "Consumer Staples", "country": "Switzerland"},
        "NOVN.SW": {"company_name": "Novartis AG", "sector": "Healthcare", "country": "Switzerland"},
        "ROG.SW": {"company_name": "Roche Holding AG", "sector": "Healthcare", "country": "Switzerland"},
        "UBSG.SW": {"company_name": "UBS Group AG", "sector": "Financials", "country": "Switzerland"},
        # Italy
        "RACE": {"company_name": "Ferrari N.V.", "sector": "Consumer Luxury", "country": "Italy"},
    }
    df_eq = fetch_ticker_dataframe(eu_equities, period="10y")
    out_eq = os.path.join(MARKET_DIR, "eu_equities_daily.parquet")
    df_eq.write_parquet(out_eq, compression="zstd")
    print(f"    => Saved {out_eq} ({len(df_eq):,} rows across {df_eq['symbol'].n_unique()} European blue-chips)")


def build_asia_markets():
    print("\n--- [3/9] Ingesting Asian Benchmark Indices & Corporate Titans ---")
    # 3.1 Asia Indices
    asia_indices = {
        "^N225": {"index_name": "Nikkei 225", "country": "Japan"},
        "^HSI": {"index_name": "Hang Seng Index", "country": "Hong Kong"},
        "000001.SS": {"index_name": "SSE Composite Index", "country": "China"},
        "000300.SS": {"index_name": "CSI 300 Index", "country": "China"},
        "^KS11": {"index_name": "KOSPI Composite Index", "country": "South Korea"},
        "^TWII": {"index_name": "TAIEX Index", "country": "Taiwan"},
        "^NSEI": {"index_name": "Nifty 50 Index", "country": "India"},
        "^BSESN": {"index_name": "BSE Sensex 30", "country": "India"},
        "^STI": {"index_name": "Straits Times Index", "country": "Singapore"},
        "^AXJO": {"index_name": "S&P/ASX 200", "country": "Australia"},
    }
    df_indices = fetch_ticker_dataframe(asia_indices, period="10y")
    out_idx = os.path.join(MARKET_DIR, "asia_indices_daily.parquet")
    df_indices.write_parquet(out_idx, compression="zstd")
    print(f"    => Saved {out_idx} ({len(df_indices):,} rows across {df_indices['symbol'].n_unique()} Asian indices)")

    # 3.2 Asia Equities
    asia_equities = {
        "TSM": {"company_name": "Taiwan Semiconductor (TSMC)", "sector": "Semiconductors", "country": "Taiwan"},
        "BABA": {"company_name": "Alibaba Group Holding Ltd.", "sector": "Consumer Discretionary", "country": "China"},
        "TCEHY": {"company_name": "Tencent Holdings Ltd.", "sector": "Communication Services", "country": "China"},
        "005930.KS": {"company_name": "Samsung Electronics Co., Ltd.", "sector": "Technology", "country": "South Korea"},
        "7203.T": {"company_name": "Toyota Motor Corporation", "sector": "Automotive", "country": "Japan"},
        "HDB": {"company_name": "HDFC Bank Limited", "sector": "Financials", "country": "India"},
        "INFY": {"company_name": "Infosys Limited", "sector": "Information Technology", "country": "India"},
        "SONY": {"company_name": "Sony Group Corporation", "sector": "Consumer Electronics", "country": "Japan"},
        "9984.T": {"company_name": "SoftBank Group Corp.", "sector": "Financials/Tech", "country": "Japan"},
    }
    df_eq = fetch_ticker_dataframe(asia_equities, period="10y")
    out_eq = os.path.join(MARKET_DIR, "asia_equities_daily.parquet")
    df_eq.write_parquet(out_eq, compression="zstd")
    print(f"    => Saved {out_eq} ({len(df_eq):,} rows across {df_eq['symbol'].n_unique()} Asian corporate giants)")


def build_global_etfs():
    print("\n--- [4/9] Ingesting Global Cross-Asset & Factor ETFs ---")
    etfs = {
        "SPY": {"etf_name": "SPDR S&P 500 ETF Trust", "asset_class": "US Equity Large Cap"},
        "QQQ": {"etf_name": "Invesco QQQ Trust (Nasdaq 100)", "asset_class": "US Equity Tech Growth"},
        "IWM": {"etf_name": "iShares Russell 2000 ETF", "asset_class": "US Equity Small Cap"},
        "EFA": {"etf_name": "iShares MSCI EAFE ETF", "asset_class": "Developed Markets ex-US"},
        "EEM": {"etf_name": "iShares MSCI Emerging Markets ETF", "asset_class": "Emerging Markets Equity"},
        "ACWI": {"etf_name": "iShares MSCI ACWI ETF", "asset_class": "Global Equities All-Cap"},
        "FXI": {"etf_name": "iShares China Large-Cap ETF", "asset_class": "China Equities"},
        "EWJ": {"etf_name": "iShares MSCI Japan ETF", "asset_class": "Japan Equities"},
        "EWG": {"etf_name": "iShares MSCI Germany ETF", "asset_class": "Germany Equities"},
        "EWU": {"etf_name": "iShares MSCI United Kingdom ETF", "asset_class": "UK Equities"},
        "INDA": {"etf_name": "iShares MSCI India ETF", "asset_class": "India Equities"},
        "TLT": {"etf_name": "iShares 20+ Year Treasury Bond ETF", "asset_class": "US Long Treasuries"},
        "IEF": {"etf_name": "iShares 7-10 Year Treasury Bond ETF", "asset_class": "US Intermediate Treasuries"},
        "LQD": {"etf_name": "iShares iBoxx $ Investment Grade Corporate Bond", "asset_class": "Corporate IG Bonds"},
        "HYG": {"etf_name": "iShares iBoxx $ High Yield Corporate Bond ETF", "asset_class": "High Yield Credit"},
        "GLD": {"etf_name": "SPDR Gold Shares", "asset_class": "Precious Metals"},
        "VNQ": {"etf_name": "Vanguard Real Estate ETF", "asset_class": "Real Estate / REITs"},
    }
    df_etf = fetch_ticker_dataframe(etfs, period="10y")
    out_etf = os.path.join(MARKET_DIR, "global_etfs_daily.parquet")
    df_etf.write_parquet(out_etf, compression="zstd")
    print(f"    => Saved {out_etf} ({len(df_etf):,} rows across {df_etf['symbol'].n_unique()} global ETFs)")


def build_global_commodities():
    print("\n--- [5/9] Ingesting Global Commodities (Energy, Metals, Agriculture) ---")
    commodities = {
        "CL=F": {"commodity_name": "Crude Oil WTI Futures", "sector": "Energy"},
        "BZ=F": {"commodity_name": "Brent Crude Oil Futures", "sector": "Energy"},
        "NG=F": {"commodity_name": "Natural Gas Futures", "sector": "Energy"},
        "RB=F": {"commodity_name": "RBOB Gasoline Futures", "sector": "Energy"},
        "GC=F": {"commodity_name": "Gold 100oz Futures", "sector": "Precious Metals"},
        "SI=F": {"commodity_name": "Silver 5000oz Futures", "sector": "Precious Metals"},
        "HG=F": {"commodity_name": "Copper High Grade Futures", "sector": "Industrial Metals"},
        "PL=F": {"commodity_name": "Platinum Futures", "sector": "Precious Metals"},
        "ZC=F": {"commodity_name": "Corn Futures", "sector": "Agriculture"},
        "ZW=F": {"commodity_name": "Wheat Futures", "sector": "Agriculture"},
        "ZS=F": {"commodity_name": "Soybeans Futures", "sector": "Agriculture"},
        "KC=F": {"commodity_name": "Coffee 'C' Futures", "sector": "Soft Commodities"},
        "SB=F": {"commodity_name": "Sugar #11 Futures", "sector": "Soft Commodities"},
        "CT=F": {"commodity_name": "Cotton #2 Futures", "sector": "Soft Commodities"},
    }
    df_comm = fetch_ticker_dataframe(commodities, period="10y")
    out_comm = os.path.join(MACRO_DIR, "global_commodities_daily.parquet")
    df_comm.write_parquet(out_comm, compression="zstd")
    print(f"    => Saved {out_comm} ({len(df_comm):,} rows across {df_comm['symbol'].n_unique()} global commodities)")


def build_us_yield_curve():
    print("\n--- [6/9] Ingesting US Yield Curve & Term Structure (3M, 5Y, 10Y, 30Y, VIX) ---")
    yield_tickers = {
        "^IRX": "tbill_3m",
        "^FVX": "tnote_5y",
        "^TNX": "tnote_10y",
        "^TYX": "tbond_30y",
        "^VIX": "cboe_vix",
    }
    raw = yf.download(list(yield_tickers.keys()), period="10y", interval="1d", progress=False)
    close_df = raw["Close"].copy() if "Close" in raw else raw

    pivot_data = []
    close_df = close_df.dropna(how="all").reset_index()
    date_col = "Date" if "Date" in close_df.columns else close_df.columns[0]
    
    for _, row in close_df.iterrows():
        dt_str = pd.to_datetime(row[date_col]).strftime("%Y-%m-%d")
        t3m = float(row.get("^IRX")) if pd.notnull(row.get("^IRX")) else None
        t5y = float(row.get("^FVX")) if pd.notnull(row.get("^FVX")) else None
        t10y = float(row.get("^TNX")) if pd.notnull(row.get("^TNX")) else None
        t30y = float(row.get("^TYX")) if pd.notnull(row.get("^TYX")) else None
        vix = float(row.get("^VIX")) if pd.notnull(row.get("^VIX")) else None

        spread_10y_3m = (t10y - t3m) if (t10y is not None and t3m is not None) else None
        spread_10y_5y = (t10y - t5y) if (t10y is not None and t5y is not None) else None

        pivot_data.append({
            "date": dt_str,
            "tbill_3m": t3m,
            "tnote_5y": t5y,
            "tnote_10y": t10y,
            "tbond_30y": t30y,
            "yield_spread_10y_3m": spread_10y_3m,
            "yield_spread_10y_5y": spread_10y_5y,
            "cboe_vix": vix,
        })

    df_yc = (
        pl.DataFrame(pivot_data, infer_schema_length=None)
        .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
        .sort("date")
        .unique(subset=["date"], keep="last")
    )
    out_yc = os.path.join(MACRO_DIR, "us_yield_curve_daily.parquet")
    df_yc.write_parquet(out_yc, compression="zstd")
    print(f"    => Saved {out_yc} ({len(df_yc):,} rows of daily US yield curve observations)")


def build_eu_macro():
    print("\n--- [7/9] Ingesting European Foreign Exchange & Macro Series ---")
    fx_pairs = {
        "EURUSD=X": {"currency_pair": "EUR/USD", "region": "Europe"},
        "GBPUSD=X": {"currency_pair": "GBP/USD", "region": "United Kingdom"},
        "EURGBP=X": {"currency_pair": "EUR/GBP", "region": "Cross Europe"},
        "EURCHF=X": {"currency_pair": "EUR/CHF", "region": "Europe / Switzerland"},
        "EURJPY=X": {"currency_pair": "EUR/JPY", "region": "Europe / Japan"},
    }
    df_fx = fetch_ticker_dataframe(fx_pairs, period="10y")
    out_fx = os.path.join(MACRO_DIR, "eu_macro_daily.parquet")
    df_fx.write_parquet(out_fx, compression="zstd")
    print(f"    => Saved {out_fx} ({len(df_fx):,} rows across {df_fx['symbol'].n_unique()} European FX pairs)")


def build_asia_macro():
    print("\n--- [8/9] Ingesting Asian Foreign Exchange & Currency Regimes ---")
    fx_asia = {
        "USDJPY=X": {"currency_pair": "USD/JPY", "country": "Japan"},
        "USDCNY=X": {"currency_pair": "USD/CNY", "country": "China"},
        "USDKRW=X": {"currency_pair": "USD/KRW", "country": "South Korea"},
        "USDINR=X": {"currency_pair": "USD/INR", "country": "India"},
        "USDSGD=X": {"currency_pair": "USD/SGD", "country": "Singapore"},
        "USDHKD=X": {"currency_pair": "USD/HKD", "country": "Hong Kong"},
        "AUDUSD=X": {"currency_pair": "AUD/USD", "country": "Australia"},
        "USDVND=X": {"currency_pair": "USD/VND", "country": "Vietnam"},
    }
    df_fx = fetch_ticker_dataframe(fx_asia, period="10y")
    out_fx = os.path.join(MACRO_DIR, "asia_macro_daily.parquet")
    df_fx.write_parquet(out_fx, compression="zstd")
    print(f"    => Saved {out_fx} ({len(df_fx):,} rows across {df_fx['symbol'].n_unique()} Asian currencies)")


def build_crypto_markets():
    print("\n--- [9/9] Ingesting Cryptocurrency Markets & Cross-Asset Metrics ---")
    crypto_assets = {
        "BTC-USD": {"asset_name": "Bitcoin", "category": "Store of Value / L1"},
        "ETH-USD": {"asset_name": "Ethereum", "category": "Smart Contract L1"},
        "SOL-USD": {"asset_name": "Solana", "category": "High-Throughput L1"},
        "BNB-USD": {"asset_name": "Binance Coin", "category": "Exchange / L1"},
        "XRP-USD": {"asset_name": "Ripple", "category": "Cross-Border Settlement"},
        "ADA-USD": {"asset_name": "Cardano", "category": "PoS L1"},
        "DOGE-USD": {"asset_name": "Dogecoin", "category": "Meme / Currency"},
        "AVAX-USD": {"asset_name": "Avalanche", "category": "Smart Contract L1"},
        "LINK-USD": {"asset_name": "Chainlink", "category": "Oracle / Middleware"},
        "DOT-USD": {"asset_name": "Polkadot", "category": "Interoperability"},
        "NEAR-USD": {"asset_name": "Near Protocol", "category": "Sharded L1"},
    }
    df_crypto_raw = fetch_ticker_dataframe(crypto_assets, period="10y")

    # Add 30-day realized volatility per crypto asset
    # Realized vol = std(returns, 30) * sqrt(365)
    crypto_augmented = []
    for sym in df_crypto_raw["symbol"].unique():
        sub = df_crypto_raw.filter(pl.col("symbol") == sym).sort("date")
        ret_series = sub["returns"].to_pandas()
        vol30 = ret_series.rolling(30).std() * np.sqrt(365)
        sub = sub.with_columns(pl.Series("volatility_30d", vol30.values, nan_to_null=True))
        crypto_augmented.append(sub)

    df_crypto = pl.concat(crypto_augmented).sort(["symbol", "date"])
    out_crypto = os.path.join(CRYPTO_DIR, "crypto_top_assets_daily.parquet")
    df_crypto.write_parquet(out_crypto, compression="zstd")
    print(f"    => Saved {out_crypto} ({len(df_crypto):,} rows across {df_crypto['symbol'].n_unique()} top cryptocurrencies)")

    # Macro Crypto Aggregate Indicators
    # Pivot BTC, ETH, and compute cross correlations with S&P 500 and Gold
    print("    - Computing aggregate crypto market indicators, ETH/BTC ratio, and macro spillovers...")
    try:
        btc_sub = df_crypto.filter(pl.col("symbol") == "BTC-USD").select(["date", "close", "volume", "returns"]).rename({
            "close": "btc_close", "volume": "btc_volume", "returns": "btc_returns"
        })
        eth_sub = df_crypto.filter(pl.col("symbol") == "ETH-USD").select(["date", "close", "volume", "returns"]).rename({
            "close": "eth_close", "volume": "eth_volume", "returns": "eth_returns"
        })

        def _as_date(expr):
            return expr.cast(pl.Utf8).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)

        def _rolling_corr_trading_days(other_df, other_col, out_col):
            """30-observation rolling correlation of BTC with an asset that trades on weekdays only.
            Computed on the dates both trade (no weekend nulls), then matched back by date."""
            joined = btc_sub.select(["date", "btc_returns"]).join(other_df, on="date", how="inner").sort("date")
            return joined.select(["date", pl.rolling_corr(pl.col("btc_returns"), pl.col(other_col), window_size=30).alias(out_col)])

        combined = btc_sub.join(eth_sub, on="date", how="inner")
        sp500_file = os.path.join(MARKET_DIR, "us_indices_daily.parquet")
        gold_file = os.path.join(MACRO_DIR, "global_commodities_daily.parquet")
        for fpath, sym, col, out_col in [
            (sp500_file, "^GSPC", "sp500_returns", "btc_sp500_corr_30d"),
            (gold_file, "GC=F", "gold_returns", "btc_gold_corr_30d"),
        ]:
            if not os.path.exists(fpath):
                continue
            other = (
                pl.read_parquet(fpath).filter(pl.col("symbol") == sym)
                .select([_as_date(pl.col("date")).alias("date"), pl.col("returns").alias(col)])
                .drop_nulls()
            )
            combined = combined.join(_rolling_corr_trading_days(other, col, out_col), on="date", how="left")

        comb_pd = combined.sort("date").to_pandas()
        comb_pd["eth_btc_ratio"] = comb_pd["eth_close"] / np.where(comb_pd["btc_close"] == 0, np.nan, comb_pd["btc_close"])
        comb_pd["crypto_top_volume_sum"] = comb_pd["btc_volume"] + comb_pd["eth_volume"]
        comb_pd["btc_vol_30d"] = comb_pd["btc_returns"].rolling(30).std() * np.sqrt(365)
        comb_pd["eth_vol_30d"] = comb_pd["eth_returns"].rolling(30).std() * np.sqrt(365)

        select_cols = [
            "date", "btc_close", "eth_close", "eth_btc_ratio", "crypto_top_volume_sum",
            "btc_vol_30d", "eth_vol_30d", "btc_sp500_corr_30d", "btc_gold_corr_30d"
        ]
        comb_pd = comb_pd[[c for c in select_cols if c in comb_pd.columns]]

        df_mkt = (
            pl.DataFrame(comb_pd)
            .with_columns(pl.col("date").cast(pl.Date))
            .sort("date")
            .unique(subset=["date"], keep="last")
        )
        out_mkt = os.path.join(CRYPTO_DIR, "crypto_market_daily.parquet")
        df_mkt.write_parquet(out_mkt, compression="zstd")
        print(f"    => Saved {out_mkt} ({len(df_mkt):,} rows of daily aggregate crypto indicators)")
    except Exception as e:
        print(f"    ! Error computing crypto market aggregates: {e}")


def run_all():
    start = time.time()
    print("=" * 85)
    print("STARTING GLOBAL FINANCIAL MARKETS INGESTION (ASIA, US, EU, COMMODITIES, CRYPTO)")
    print("=" * 85)

    build_us_markets()
    build_eu_markets()
    build_asia_markets()
    build_global_etfs()
    build_global_commodities()
    build_us_yield_curve()
    build_eu_macro()
    build_asia_macro()
    build_crypto_markets()

    elapsed = time.time() - start
    print("=" * 85)
    print(f"GLOBAL INGESTION COMPLETE IN {elapsed:.2f}s!")
    print("=" * 85)


if __name__ == "__main__":
    run_all()
