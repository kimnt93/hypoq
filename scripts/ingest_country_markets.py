#!/usr/bin/env python3
"""
Country-Specific Financial Markets Ingestion Pipeline.
Extracts and builds discrete datasets for China (CN), Hong Kong (HK), Japan (JP), and South Korea (KR).
Covers indices, equities, and currency macro series:
- cn_indices_daily.parquet, cn_equities_daily.parquet, cn_macro_daily.parquet
- hk_indices_daily.parquet, hk_equities_daily.parquet, hk_macro_daily.parquet
- jp_indices_daily.parquet, jp_equities_daily.parquet, jp_macro_daily.parquet
- kr_indices_daily.parquet, kr_equities_daily.parquet, kr_macro_daily.parquet

All datasets are saved into data/market/ and data/macro/ with Zstandard compression.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MARKET_DIR = os.path.join(DATA_DIR, "market")
MACRO_DIR = os.path.join(DATA_DIR, "macro")

for d in [MARKET_DIR, MACRO_DIR]:
    os.makedirs(d, exist_ok=True)

def fetch_country_dataframe(tickers_meta: dict, period: str = "10y") -> pl.DataFrame:
    """Fetches and standardizes financial series into a clean Polars DataFrame."""
    records = []

    for tick, meta in tickers_meta.items():
        try:
            t = yf.Ticker(tick)
            df_tick = t.history(period=period, auto_adjust=False)

            if df_tick is not None and len(df_tick) > 0:
                df_tick = df_tick.reset_index()
                date_col = "Date" if "Date" in df_tick.columns else df_tick.columns[0]
                clean_dates = pd.to_datetime(df_tick[date_col]).dt.strftime("%Y-%m-%d").tolist()
                
                closes = df_tick["Close"].astype(float).values
                returns = np.zeros(len(closes))
                if len(closes) > 1:
                    returns[1:] = (closes[1:] - closes[:-1]) / np.where(closes[:-1] == 0, np.nan, closes[:-1])

                opens = df_tick["Open"].astype(float).values if "Open" in df_tick.columns else closes
                highs = df_tick["High"].astype(float).values if "High" in df_tick.columns else closes
                lows = df_tick["Low"].astype(float).values if "Low" in df_tick.columns else closes
                adj_closes = df_tick["Adj Close"].astype(float).values if "Adj Close" in df_tick.columns else closes
                volumes = df_tick["Volume"].astype(float).values if "Volume" in df_tick.columns else np.zeros(len(closes))

                for i in range(len(df_tick)):
                    rec = {
                        "date": clean_dates[i],
                        "symbol": tick,
                        "open": float(opens[i]) if pd.notnull(opens[i]) else None,
                        "high": float(highs[i]) if pd.notnull(highs[i]) else None,
                        "low": float(lows[i]) if pd.notnull(lows[i]) else None,
                        "close": float(closes[i]) if pd.notnull(closes[i]) else None,
                        "adj_close": float(adj_closes[i]) if pd.notnull(adj_closes[i]) else None,
                        "volume": float(volumes[i]) if pd.notnull(volumes[i]) else 0.0,
                        "returns": float(returns[i]) if pd.notnull(returns[i]) else 0.0,
                    }
                    rec.update(meta)
                    records.append(rec)

                label = meta.get("company_name", meta.get("index_name", meta.get("currency_pair", "")))
                print(f"      + {tick} ({label}): {len(df_tick):,} bars")
            else:
                print(f"      ! Warning: No data for {tick}")
        except Exception as err:
            print(f"      ! Error parsing {tick}: {err}")

    if not records:
        return pl.DataFrame()

    df_out = (
        pl.DataFrame(records)
        .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
        .sort(["symbol", "date"])
        .unique(subset=["symbol", "date"], keep="last")
    )
    return df_out


def build_japan_datasets():
    print("\n>>> [JP] Ingesting Japan Financial Markets (Indices, Equities, FX)...")
    # JP Indices
    jp_indices = {
        "^N225": {"index_name": "Nikkei 225 Index", "country": "Japan"},
    }
    df_idx = fetch_country_dataframe(jp_indices, period="10y")
    out_idx = os.path.join(MARKET_DIR, "jp_indices_daily.parquet")
    df_idx.write_parquet(out_idx, compression="zstd")
    print(f"    - Saved {out_idx} ({len(df_idx):,} rows)")

    # JP Equities
    jp_equities = {
        "7203.T": {"company_name": "Toyota Motor Corp.", "sector": "Automotive", "country": "Japan"},
        "6758.T": {"company_name": "Sony Group Corp.", "sector": "Consumer Electronics", "country": "Japan"},
        "6861.T": {"company_name": "Keyence Corp.", "sector": "Industrial Automation", "country": "Japan"},
        "8035.T": {"company_name": "Tokyo Electron Ltd.", "sector": "Semiconductors", "country": "Japan"},
        "8306.T": {"company_name": "Mitsubishi UFJ Financial Group", "sector": "Financials", "country": "Japan"},
        "9984.T": {"company_name": "SoftBank Group Corp.", "sector": "Tech Investment", "country": "Japan"},
        "7974.T": {"company_name": "Nintendo Co., Ltd.", "sector": "Gaming / Entertainment", "country": "Japan"},
        "9983.T": {"company_name": "Fast Retailing Co. (Uniqlo)", "sector": "Consumer Discretionary", "country": "Japan"},
        "6902.T": {"company_name": "Denso Corp.", "sector": "Auto Components", "country": "Japan"},
        "4063.T": {"company_name": "Shin-Etsu Chemical Co.", "sector": "Materials / Chemicals", "country": "Japan"},
        "8058.T": {"company_name": "Mitsubishi Corporation", "sector": "Trading Conglomerate", "country": "Japan"},
        "8001.T": {"company_name": "Itochu Corporation", "sector": "Trading Conglomerate", "country": "Japan"},
        "6501.T": {"company_name": "Hitachi, Ltd.", "sector": "Industrials & Tech", "country": "Japan"},
        "7267.T": {"company_name": "Honda Motor Co., Ltd.", "sector": "Automotive", "country": "Japan"},
        "9432.T": {"company_name": "Nippon Telegraph and Telephone (NTT)", "sector": "Telecommunications", "country": "Japan"},
        "4502.T": {"company_name": "Takeda Pharmaceutical Co.", "sector": "Healthcare", "country": "Japan"},
        "6367.T": {"company_name": "Daikin Industries, Ltd.", "sector": "Industrials / HVAC", "country": "Japan"},
        "7741.T": {"company_name": "HOYA Corporation", "sector": "Optical / Healthcare", "country": "Japan"},
    }
    df_eq = fetch_country_dataframe(jp_equities, period="10y")
    out_eq = os.path.join(MARKET_DIR, "jp_equities_daily.parquet")
    df_eq.write_parquet(out_eq, compression="zstd")
    print(f"    - Saved {out_eq} ({len(df_eq):,} rows across {df_eq['symbol'].n_unique()} Japanese equities)")

    # JP Macro
    jp_macro = {
        "USDJPY=X": {"currency_pair": "USD/JPY", "country": "Japan"},
        "EURJPY=X": {"currency_pair": "EUR/JPY", "country": "Japan"},
    }
    df_macro = fetch_country_dataframe(jp_macro, period="10y")
    out_macro = os.path.join(MACRO_DIR, "jp_macro_daily.parquet")
    df_macro.write_parquet(out_macro, compression="zstd")
    print(f"    - Saved {out_macro} ({len(df_macro):,} rows)")


def build_korea_datasets():
    print("\n>>> [KR] Ingesting South Korea Financial Markets (Indices, Equities, FX)...")
    # KR Indices
    kr_indices = {
        "^KS11": {"index_name": "KOSPI Composite Index", "country": "South Korea"},
        "^KQ11": {"index_name": "KOSDAQ Index", "country": "South Korea"},
    }
    df_idx = fetch_country_dataframe(kr_indices, period="10y")
    out_idx = os.path.join(MARKET_DIR, "kr_indices_daily.parquet")
    df_idx.write_parquet(out_idx, compression="zstd")
    print(f"    - Saved {out_idx} ({len(df_idx):,} rows)")

    # KR Equities
    kr_equities = {
        "005930.KS": {"company_name": "Samsung Electronics", "sector": "Semiconductors & Tech", "country": "South Korea"},
        "000660.KS": {"company_name": "SK Hynix Inc.", "sector": "Semiconductors", "country": "South Korea"},
        "005380.KS": {"company_name": "Hyundai Motor Company", "sector": "Automotive", "country": "South Korea"},
        "373220.KS": {"company_name": "LG Energy Solution", "sector": "EV Batteries", "country": "South Korea"},
        "005490.KS": {"company_name": "POSCO Holdings", "sector": "Steel & Materials", "country": "South Korea"},
        "035420.KS": {"company_name": "NAVER Corp.", "sector": "Internet & Search", "country": "South Korea"},
        "035720.KS": {"company_name": "Kakao Corp.", "sector": "Communication & Internet", "country": "South Korea"},
        "051910.KS": {"company_name": "LG Chem Ltd.", "sector": "Chemicals", "country": "South Korea"},
        "207940.KS": {"company_name": "Samsung Biologics", "sector": "Biotechnology", "country": "South Korea"},
        "000270.KS": {"company_name": "Kia Corporation", "sector": "Automotive", "country": "South Korea"},
        "068270.KS": {"company_name": "Celltrion, Inc.", "sector": "Biopharmaceuticals", "country": "South Korea"},
        "105560.KS": {"company_name": "KB Financial Group Inc.", "sector": "Financials", "country": "South Korea"},
        "055550.KS": {"company_name": "Shinhan Financial Group", "sector": "Financials", "country": "South Korea"},
        "012330.KS": {"company_name": "Hyundai Mobis", "sector": "Auto Parts", "country": "South Korea"},
        "028260.KS": {"company_name": "Samsung C&T Corporation", "sector": "Industrial Conglomerate", "country": "South Korea"},
        "066570.KS": {"company_name": "LG Electronics", "sector": "Consumer Electronics", "country": "South Korea"},
    }
    df_eq = fetch_country_dataframe(kr_equities, period="10y")
    out_eq = os.path.join(MARKET_DIR, "kr_equities_daily.parquet")
    df_eq.write_parquet(out_eq, compression="zstd")
    print(f"    - Saved {out_eq} ({len(df_eq):,} rows across {df_eq['symbol'].n_unique()} South Korean equities)")

    # KR Macro
    kr_macro = {
        "USDKRW=X": {"currency_pair": "USD/KRW", "country": "South Korea"},
    }
    df_macro = fetch_country_dataframe(kr_macro, period="10y")
    out_macro = os.path.join(MACRO_DIR, "kr_macro_daily.parquet")
    df_macro.write_parquet(out_macro, compression="zstd")
    print(f"    - Saved {out_macro} ({len(df_macro):,} rows)")


def build_hongkong_datasets():
    print("\n>>> [HK] Ingesting Hong Kong Financial Markets (Indices, Equities, FX)...")
    # HK Indices
    hk_indices = {
        "^HSI": {"index_name": "Hang Seng Index", "country": "Hong Kong"},
    }
    df_idx = fetch_country_dataframe(hk_indices, period="10y")
    out_idx = os.path.join(MARKET_DIR, "hk_indices_daily.parquet")
    df_idx.write_parquet(out_idx, compression="zstd")
    print(f"    - Saved {out_idx} ({len(df_idx):,} rows)")

    # HK Equities
    hk_equities = {
        "0700.HK": {"company_name": "Tencent Holdings Ltd.", "sector": "Internet & Gaming", "country": "Hong Kong"},
        "9988.HK": {"company_name": "Alibaba Group Holding Ltd.", "sector": "E-Commerce & Cloud", "country": "Hong Kong"},
        "0005.HK": {"company_name": "HSBC Holdings plc", "sector": "Financials / Banking", "country": "Hong Kong"},
        "1299.HK": {"company_name": "AIA Group Limited", "sector": "Insurance", "country": "Hong Kong"},
        "3690.HK": {"company_name": "Meituan", "sector": "Consumer Services", "country": "Hong Kong"},
        "1810.HK": {"company_name": "Xiaomi Corporation", "sector": "Consumer Electronics", "country": "Hong Kong"},
        "0941.HK": {"company_name": "China Mobile Limited", "sector": "Telecommunications", "country": "Hong Kong"},
        "2318.HK": {"company_name": "Ping An Insurance Group", "sector": "Insurance", "country": "Hong Kong"},
        "0388.HK": {"company_name": "Hong Kong Exchanges & Clearing", "sector": "Financial Exchanges", "country": "Hong Kong"},
        "9618.HK": {"company_name": "JD.com Inc. (HK)", "sector": "E-Commerce", "country": "Hong Kong"},
        "1024.HK": {"company_name": "Kuaishou Technology", "sector": "Social Media / Video", "country": "Hong Kong"},
        "2269.HK": {"company_name": "WuXi Biologics", "sector": "Biotechnology", "country": "Hong Kong"},
        "1211.HK": {"company_name": "BYD Company Limited (HK)", "sector": "Electric Vehicles", "country": "Hong Kong"},
        "0883.HK": {"company_name": "CNOOC Limited", "sector": "Energy / Oil & Gas", "country": "Hong Kong"},
        "2015.HK": {"company_name": "Li Auto Inc. (HK)", "sector": "Electric Vehicles", "country": "Hong Kong"},
        "0016.HK": {"company_name": "Sun Hung Kai Properties", "sector": "Real Estate", "country": "Hong Kong"},
    }
    df_eq = fetch_country_dataframe(hk_equities, period="10y")
    out_eq = os.path.join(MARKET_DIR, "hk_equities_daily.parquet")
    df_eq.write_parquet(out_eq, compression="zstd")
    print(f"    - Saved {out_eq} ({len(df_eq):,} rows across {df_eq['symbol'].n_unique()} Hong Kong equities)")

    # HK Macro
    hk_macro = {
        "USDHKD=X": {"currency_pair": "USD/HKD", "country": "Hong Kong"},
    }
    df_macro = fetch_country_dataframe(hk_macro, period="10y")
    out_macro = os.path.join(MACRO_DIR, "hk_macro_daily.parquet")
    df_macro.write_parquet(out_macro, compression="zstd")
    print(f"    - Saved {out_macro} ({len(df_macro):,} rows)")


def build_china_datasets():
    print("\n>>> [CN] Ingesting China Financial Markets (Indices, Equities, FX)...")
    # CN Indices
    cn_indices = {
        "000001.SS": {"index_name": "SSE Composite Index", "country": "China"},
        "000300.SS": {"index_name": "CSI 300 Index", "country": "China"},
    }
    df_idx = fetch_country_dataframe(cn_indices, period="10y")
    out_idx = os.path.join(MARKET_DIR, "cn_indices_daily.parquet")
    df_idx.write_parquet(out_idx, compression="zstd")
    print(f"    - Saved {out_idx} ({len(df_idx):,} rows)")

    # CN Equities (Mainland A-shares & Global Chinese Leaders)
    cn_equities = {
        "600519.SS": {"company_name": "Kweichow Moutai Co., Ltd.", "sector": "Consumer Staples / Liquor", "country": "China"},
        "601398.SS": {"company_name": "Industrial and Commercial Bank of China (ICBC)", "sector": "Financials", "country": "China"},
        "300750.SZ": {"company_name": "Contemporary Amperex Technology (CATL)", "sector": "EV Batteries & Energy", "country": "China"},
        "BABA": {"company_name": "Alibaba Group (ADR)", "sector": "E-Commerce", "country": "China"},
        "PDD": {"company_name": "PDD Holdings Inc. (Temu)", "sector": "E-Commerce", "country": "China"},
        "JD": {"company_name": "JD.com Inc.", "sector": "E-Commerce", "country": "China"},
        "BIDU": {"company_name": "Baidu Inc.", "sector": "Artificial Intelligence / Search", "country": "China"},
        "NIO": {"company_name": "NIO Inc.", "sector": "Electric Vehicles", "country": "China"},
        "BYDDY": {"company_name": "BYD Company Ltd. (ADR)", "sector": "Electric Vehicles", "country": "China"},
        "LI": {"company_name": "Li Auto Inc.", "sector": "Electric Vehicles", "country": "China"},
        "601857.SS": {"company_name": "PetroChina Company Limited", "sector": "Energy", "country": "China"},
        "601288.SS": {"company_name": "Agricultural Bank of China", "sector": "Financials", "country": "China"},
        "600036.SS": {"company_name": "China Merchants Bank", "sector": "Financials", "country": "China"},
        "000858.SZ": {"company_name": "Wuliangye Yibin Co., Ltd.", "sector": "Consumer Staples", "country": "China"},
        "601988.SS": {"company_name": "Bank of China Limited", "sector": "Financials", "country": "China"},
        "600900.SS": {"company_name": "China Yangtze Power Co.", "sector": "Utilities", "country": "China"},
        "XPEV": {"company_name": "XPeng Inc. (ADR)", "sector": "Electric Vehicles", "country": "China"},
        "TCOM": {"company_name": "Trip.com Group Limited", "sector": "Travel & Consumer Services", "country": "China"},
    }
    df_eq = fetch_country_dataframe(cn_equities, period="10y")
    out_eq = os.path.join(MARKET_DIR, "cn_equities_daily.parquet")
    df_eq.write_parquet(out_eq, compression="zstd")
    print(f"    - Saved {out_eq} ({len(df_eq):,} rows across {df_eq['symbol'].n_unique()} Chinese corporate leaders)")

    # CN Macro
    cn_macro = {
        "USDCNY=X": {"currency_pair": "USD/CNY", "country": "China"},
    }
    df_macro = fetch_country_dataframe(cn_macro, period="10y")
    out_macro = os.path.join(MACRO_DIR, "cn_macro_daily.parquet")
    df_macro.write_parquet(out_macro, compression="zstd")
    print(f"    - Saved {out_macro} ({len(df_macro):,} rows)")


def run_all():
    start = time.time()
    print("=" * 85)
    print("STARTING COUNTRY-SPECIFIC MARKET INGESTION (CN, HK, JP, KR)")
    print("=" * 85)

    build_japan_datasets()
    build_korea_datasets()
    build_hongkong_datasets()
    build_china_datasets()

    elapsed = time.time() - start
    print("=" * 85)
    print(f"COUNTRY MARKETS INGESTION COMPLETED IN {elapsed:.2f}s!")
    print("=" * 85)


if __name__ == "__main__":
    run_all()
