#!/usr/bin/env python3
"""
Shared Yahoo Finance daily-bar downloader used by every global ingestion script.

Rules (data must be true, never filled):
- prices are Yahoo's split-adjusted `Close` (auto_adjust=False); `adj_close` also includes dividends
- `returns` is the simple price return between consecutive available bars; the first bar has NULL returns
- missing values stay NULL (no zero filling)
- each ticker is retried with back-off; tickers that never return data are reported and skipped
"""

import time
from typing import Dict

import pandas as pd
import polars as pl
import yfinance as yf

RETRIES = 3
MIN_BARS = 60  # a ticker with fewer bars is a vendor gap, not a usable series


def _history(tick: str, period: str) -> pd.DataFrame:
    for attempt in range(RETRIES):
        try:
            df = yf.Ticker(tick).history(period=period, auto_adjust=False)
            if df is not None and len(df) > 0:
                return df
        except Exception as exc:
            print(f"      ! {tick}: {type(exc).__name__}: {exc}")
        time.sleep(2 ** attempt)
    return pd.DataFrame()


def fetch_ticker_dataframe(tickers_meta: Dict[str, dict], period: str = "10y") -> pl.DataFrame:
    """Downloads daily bars for {ticker: metadata}; metadata keys become extra columns."""
    frames = []
    for tick, meta in tickers_meta.items():
        raw = _history(tick, period)
        if raw.empty or "Close" not in raw.columns:
            print(f"      ! No data returned for {tick}")
            continue
        raw = raw.reset_index()
        date_col = "Date" if "Date" in raw.columns else raw.columns[0]
        raw["date"] = pd.to_datetime(raw[date_col]).dt.tz_localize(None).dt.normalize()
        raw = raw.dropna(subset=["Close"]).sort_values("date").drop_duplicates("date", keep="last")
        if len(raw) < MIN_BARS:
            print(f"      ! Skipping {tick}: only {len(raw)} bars available from Yahoo")
            continue
        close = raw["Close"].astype(float)
        out = pd.DataFrame({
            "date": raw["date"].dt.date,
            "symbol": tick,
            "open": raw.get("Open"),
            "high": raw.get("High"),
            "low": raw.get("Low"),
            "close": close,
            "adj_close": raw["Adj Close"] if "Adj Close" in raw.columns else close,
            "volume": raw["Volume"] if "Volume" in raw.columns else None,
            "returns": close.pct_change(),
        })
        for k, v in meta.items():
            out[k] = v
        frames.append(out)
        label = meta.get("name") or meta.get("company_name") or meta.get("index_name") or meta.get("asset_name") or ""
        print(f"      + {tick} ({label}): {len(out):,} bars")
    if not frames:
        return pl.DataFrame()
    df = pl.from_pandas(pd.concat(frames, ignore_index=True), nan_to_null=True)
    float_cols = [c for c in ("open", "high", "low", "close", "adj_close", "volume", "returns") if c in df.columns]
    return (
        df.with_columns([pl.col(c).cast(pl.Float64) for c in float_cols])
        .with_columns(pl.col("date").cast(pl.Date))
        .sort(["symbol", "date"])
    )
