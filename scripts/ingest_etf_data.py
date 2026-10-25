#!/usr/bin/env python3
"""
Vietnam ETFs Market Ingestion Pipeline.
Downloads daily OHLCV and volume for the listed Vietnamese ETFs plus VanEck VNM and Fubon 00885.

NAV, AUM and creation-basket holdings are NOT generated here: no free, key-less source publishes them
and simulating them would contaminate hypothesis tests. Agents may add them only from a real source.

Output:
- data/market/vn_etf_market_daily.parquet
"""

import os
import time
from datetime import datetime, timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import polars as pl
import yfinance as yf
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MARKET_DIR = os.path.join(DATA_DIR, "market")
STRUCT_DIR = os.path.join(DATA_DIR, "market_structure")
OUT_DIR = os.path.join(BASE_DIR, "figures_overview")

os.makedirs(MARKET_DIR, exist_ok=True)
os.makedirs(STRUCT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 14 Domestic Listed ETFs on HOSE / HNX + 2 Global Benchmarks
ETF_METADATA = {
    "E1VFVN30": {
        "name": "Quỹ ETF DCVFMVN30",
        "manager": "Dragon Capital VietFund Management (DCVFM)",
        "benchmark": "VN30 Index",
        "exchange": "HOSE",
        "isin": "VN0E1VFVN306",
        "fee_pct": 0.65,
        "custodian": "Standard Chartered Bank Vietnam"
    },
    "FUEVFVND": {
        "name": "Quỹ ETF DCVFMVN DIAMOND",
        "manager": "Dragon Capital VietFund Management (DCVFM)",
        "benchmark": "VN DIAMOND Index",
        "exchange": "HOSE",
        "isin": "VN0FUEVFVND5",
        "fee_pct": 0.80,
        "custodian": "Vietcombank"
    },
    "FUESSVFL": {
        "name": "Quỹ ETF SSIAM VNFIN LEAD",
        "manager": "SSI Asset Management (SSIAM)",
        "benchmark": "VNFIN LEAD Index",
        "exchange": "HOSE",
        "isin": "VN0FUESSVFL3",
        "fee_pct": 0.70,
        "custodian": "Vietcombank"
    },
    "FUESSV50": {
        "name": "Quỹ ETF SSIAM VNX50",
        "manager": "SSI Asset Management (SSIAM)",
        "benchmark": "VNX50 Index",
        "exchange": "HOSE",
        "isin": "VN0FUESSV504",
        "fee_pct": 0.75,
        "custodian": "Vietcombank"
    },
    "FUESSV30": {
        "name": "Quỹ ETF SSIAM VN30",
        "manager": "SSI Asset Management (SSIAM)",
        "benchmark": "VN30 Index",
        "exchange": "HOSE",
        "isin": "VN0FUESSV306",
        "fee_pct": 0.60,
        "custodian": "Vietcombank"
    },
    "FUEVN100": {
        "name": "Quỹ ETF VinaCapital VN100",
        "manager": "VinaCapital Fund Management",
        "benchmark": "VN100 Index",
        "exchange": "HOSE",
        "isin": "VN0FUEVN1002",
        "fee_pct": 0.67,
        "custodian": "BIDV"
    },
    "FUEKIV30": {
        "name": "Quỹ ETF KIM Growth VN30",
        "manager": "KIM Vietnam Fund Management",
        "benchmark": "VN30 Index",
        "exchange": "HOSE",
        "isin": "VN0FUEKIV304",
        "fee_pct": 0.55,
        "custodian": "Standard Chartered Bank Vietnam"
    },
    "FUEKIVFS": {
        "name": "Quỹ ETF KIM Growth VNFINSELECT",
        "manager": "KIM Vietnam Fund Management",
        "benchmark": "VNFINSELECT Index",
        "exchange": "HOSE",
        "isin": "VN0FUEKIVFS6",
        "fee_pct": 0.60,
        "custodian": "Standard Chartered Bank Vietnam"
    },
    "FUEKIVND": {
        "name": "Quỹ ETF KIM Growth VN DIAMOND",
        "manager": "KIM Vietnam Fund Management",
        "benchmark": "VN DIAMOND Index",
        "exchange": "HOSE",
        "isin": "VN0FUEKIVND2",
        "fee_pct": 0.65,
        "custodian": "Standard Chartered Bank Vietnam"
    },
    "FUEMAV30": {
        "name": "Quỹ ETF MAFM VN30",
        "manager": "Mirae Asset Global Investments Vietnam",
        "benchmark": "VN30 Index",
        "exchange": "HOSE",
        "isin": "VN0FUEMAV307",
        "fee_pct": 0.50,
        "custodian": "BIDV"
    },
    "FUEMAVND": {
        "name": "Quỹ ETF MAFM VNDIAMOND",
        "manager": "Mirae Asset Global Investments Vietnam",
        "benchmark": "VN DIAMOND Index",
        "exchange": "HOSE",
        "isin": "VN0FUEMAVND5",
        "fee_pct": 0.60,
        "custodian": "BIDV"
    },
    "FUEDCMID": {
        "name": "Quỹ ETF DCVFMVNMIDCAP",
        "manager": "Dragon Capital VietFund Management (DCVFM)",
        "benchmark": "VNMIDCAP Index",
        "exchange": "HOSE",
        "isin": "VN0FUEDCMID9",
        "fee_pct": 0.80,
        "custodian": "Standard Chartered Bank Vietnam"
    },
    "FUEIP100": {
        "name": "Quỹ ETF IPA Partner VN100",
        "manager": "IPA Asset Management (IPAAM)",
        "benchmark": "VN100 Index",
        "exchange": "HOSE",
        "isin": "VN0FUEIP1007",
        "fee_pct": 0.60,
        "custodian": "BIDV"
    },
    "E1SSHN30": {
        "name": "Quỹ ETF SSIAM HNX30",
        "manager": "SSI Asset Management (SSIAM)",
        "benchmark": "HNX30 Index",
        "exchange": "HNX",
        "isin": "VN0E1SSHN304",
        "fee_pct": 0.80,
        "custodian": "Vietcombank"
    }
}


def fetch_etf_market_data():
    print(">>> [1/3] Fetching Daily OHLCV Bars for All Vietnamese ETFs...")
    end_time = int(time.time())
    start_time = end_time - 365 * 10 * 86400
    all_bars = []

    # 1. Domestic Listed ETFs via VNDirect
    for ticker, meta in ETF_METADATA.items():
        url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution=D&symbol={ticker}&from={start_time}&to={end_time}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict) and "t" in data and len(data["t"]) > 0:
                    n = len(data["t"])
                    for i in range(n):
                        c_val = float(data["c"][i]) * 1000.0  # normalize thousand VND to VND
                        o_val = float(data["o"][i]) * 1000.0
                        h_val = float(data["h"][i]) * 1000.0
                        l_val = float(data["l"][i]) * 1000.0
                        vol = float(data["v"][i])
                        val_vnd = c_val * vol

                        all_bars.append({
                            "symbol": ticker,
                            "etf_name": meta["name"],
                            "benchmark_index": meta["benchmark"],
                            "exchange": meta["exchange"],
                            "date": datetime.fromtimestamp(data["t"][i], timezone.utc).strftime("%Y-%m-%d"),
                            "timestamp": int(data["t"][i]),
                            "open": o_val,
                            "high": h_val,
                            "low": l_val,
                            "close": c_val,
                            "volume": vol,
                            "value_vnd": val_vnd
                        })
                    print(f"    - Ingested {ticker:10s} ({meta['name']}): {n:,} daily bars")
        except Exception as e:
            print(f"    ! Error fetching {ticker}: {e}")

    # 2. Global Cross-Border Vietnam ETFs via Yahoo Finance
    global_etfs = {
        "VNM": ("VanEck Vietnam ETF", "MVIS Vietnam Index", "NYSE Arca"),
        "00885.TW": ("Fubon FTSE Vietnam ETF", "FTSE Vietnam 30 Index", "TWSE")
    }
    for g_tick, (g_name, g_bench, g_exch) in global_etfs.items():
        try:
            t = yf.Ticker(g_tick)
            h = t.history(period="10y")
            if len(h) > 0:
                for dt, row in h.iterrows():
                    all_bars.append({
                        "symbol": g_tick.split(".")[0],
                        "etf_name": g_name,
                        "benchmark_index": g_bench,
                        "exchange": g_exch,
                        "date": dt.strftime("%Y-%m-%d"),
                        "timestamp": int(dt.timestamp()),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": float(row["Volume"]),
                        "value_vnd": float(row["Close"] * row["Volume"] * 25000.0)  # estimate VND equivalent
                    })
                print(f"    - Ingested {g_tick:10s} ({g_name}): {len(h):,} daily bars")
        except Exception as e:
            print(f"    ! Error fetching {g_tick}: {e}")

    schema = {
        "symbol": pl.Utf8,
        "etf_name": pl.Utf8,
        "benchmark_index": pl.Utf8,
        "exchange": pl.Utf8,
        "date": pl.Utf8,
        "timestamp": pl.Int64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "value_vnd": pl.Float64
    }
    df_bars = pl.DataFrame(all_bars, schema=schema).sort(["symbol", "date"])
    out_path = os.path.join(MARKET_DIR, "vn_etf_market_daily.parquet")
    df_bars.write_parquet(out_path)
    print(f"    - Saved vn_etf_market_daily.parquet ({len(df_bars):,} rows across {df_bars['symbol'].n_unique()} ETFs)")
    return df_bars


def run_all():
    print("=" * 80)
    print("STARTING VIETNAMESE ETFS COMPLETE DATASET INGESTION")
    print("=" * 80)
    fetch_etf_market_data()
    print("=" * 80)
    print("COMPLETED ALL ETF INGESTION STEPS!")
    print("=" * 80)


if __name__ == "__main__":
    run_all()
