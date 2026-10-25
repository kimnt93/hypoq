#!/usr/bin/env python3
"""
Parquet dataset integrity and quality audit.

For every file under data/ it reports rows, columns, size, date range and checks:
- jurisdiction prefix standard (cn_, hk_, jp_, kr_, asia_, us_, eu_, global_, crypto_, vn_)
- file size < 100 MB (GitHub limit)
- duplicate primary keys (date + symbol when both exist)
- OHLC sanity (high >= low, positive prices, close inside [low, high] with a small tolerance)
- per-symbol coverage (symbols with < 50% of the median row count, or ending > 14 days before the dataset end)
- staleness of the dataset end date versus today (daily datasets only)
- columns that are entirely null

Exit code is 1 when any ERROR is found; WARN items are reported but do not fail the audit.
"""

import glob
import os
import sys
from datetime import date

import polars as pl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PREFIXES = ("cn_", "hk_", "jp_", "kr_", "asia_", "us_", "eu_", "global_", "crypto_", "vn_")
STALE_DAYS = 10
# Real, documented non-positive prices: WTI crude futures settled at -$37.63 on 2020-04-20 (storage squeeze).
REAL_NEGATIVE_PRICE_SYMBOLS = {"CL=F"}
SLOW_HINTS = ("monthly", "quarterly", "weekly", "history", "taxonomy", "classification", "sectors_icb", "nodes",
              "edges", "fundamentals", "financials", "subsidiaries", "shareholders", "charter", "basket",
              "free_float", "posts", "questions", "threads", "stackexchange", "8k", "annual", "communications")


def audit_file(path: str):
    name = os.path.basename(path)
    errors, warns = [], []
    df = pl.read_parquet(path)
    size_mb = os.path.getsize(path) / (1024 * 1024)

    if not name.startswith(PREFIXES):
        errors.append("file name lacks a jurisdiction prefix")
    if size_mb >= 100:
        errors.append(f"file is {size_mb:.0f} MB (>= 100 MB)")
    if len(df) == 0:
        errors.append("empty dataset")
        return df, errors, warns, None, None

    for c in df.columns:
        if df[c].null_count() == len(df):
            errors.append(f"column '{c}' is entirely null")

    date_col = "date" if "date" in df.columns else None
    dmin = dmax = None
    if date_col and df.schema[date_col] in (pl.Date, pl.Datetime):
        dmin, dmax = df[date_col].min(), df[date_col].max()
        if hasattr(dmax, "date"):
            dmin, dmax = dmin.date(), dmax.date()
        if not any(h in name for h in SLOW_HINTS) and (date.today() - dmax).days > STALE_DAYS:
            warns.append(f"stale: ends {dmax} ({(date.today() - dmax).days} days ago)")

    key = [c for c in ("symbol", "ticker", "asset_name", "series_id", "date") if c in df.columns]
    key = [c for c in key if c != "ticker" or "symbol" not in df.columns]
    if date_col and len(key) >= 1 and not {"post_id", "index", "url", "accession_number"} & set(df.columns):
        dups = len(df) - df.select(key).unique().height
        if dups:
            errors.append(f"{dups:,} duplicate rows on key {key}")

    price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    if len(price_cols) == 4 and df.schema["close"] in (pl.Float64, pl.Float32):
        id_col = next((c for c in ("symbol", "ticker") if c in df.columns), None)
        checked = df.filter(~pl.col(id_col).is_in(list(REAL_NEGATIVE_PRICE_SYMBOLS))) if id_col else df
        nonpos = checked.filter((pl.col("close") <= 0) | (pl.col("high") <= 0) | (pl.col("low") <= 0)).height
        if nonpos:
            errors.append(f"{nonpos:,} rows with non-positive prices")
        bad_hl = df.filter(pl.col("high") < pl.col("low") * 0.999).height
        if bad_hl:
            errors.append(f"{bad_hl:,} rows with high < low")
        outside = df.filter((pl.col("close") > pl.col("high") * 1.001) | (pl.col("close") < pl.col("low") * 0.999)).height
        if outside > 0.001 * len(df):
            warns.append(f"{outside:,} rows with close outside [low, high]")

    sym = "symbol" if "symbol" in df.columns else None
    if sym and date_col and dmax is not None:
        per = df.group_by(sym).agg(pl.len().alias("n"), pl.col(date_col).max().alias("last"))
        med = per["n"].median()
        short = per.filter(pl.col("n") < 0.5 * med)
        if 0 < short.height <= 40 and med > 250:
            warns.append(f"{short.height} symbols with < 50% of median rows ({short[sym].to_list()[:6]})")
        last = per["last"]
        ended = per.filter(last.cast(pl.Date) < dmax - pl.duration(days=14)) if last.dtype in (pl.Date, pl.Datetime) else per.head(0)
        if 0 < ended.height and not any(h in name for h in SLOW_HINTS):
            warns.append(f"{ended.height} symbols end > 14 days before dataset end ({ended[sym].to_list()[:6]})")
    return df, errors, warns, dmin, dmax


def verify() -> int:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "**/*.parquet"), recursive=True))
    print(f"Found {len(files)} Parquet files across {DATA_DIR}")
    print("-" * 118)
    print(f"{'Dataset':<58} | {'Rows':>9} | {'Cols':>4} | {'Size':>9} | {'First':>10} | {'Last':>10}")
    print("-" * 118)
    total_rows = total_bytes = n_err = n_warn = 0
    findings = []
    for f in files:
        rel = os.path.relpath(f, BASE_DIR)
        size = os.path.getsize(f)
        total_bytes += size
        try:
            df, errors, warns, dmin, dmax = audit_file(f)
        except Exception as exc:
            print(f"{rel:<58} | [ERROR: {exc}]")
            n_err += 1
            continue
        total_rows += len(df)
        size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.2f} MB"
        print(f"{rel:<58} | {len(df):>9,} | {len(df.columns):>4} | {size_str:>9} | {str(dmin or ''):>10} | {str(dmax or ''):>10}")
        for e in errors:
            findings.append(("ERROR", rel, e))
        for w in warns:
            findings.append(("WARN", rel, w))
        n_err += len(errors)
        n_warn += len(warns)
    print("-" * 118)
    print(f"TOTAL: {len(files)} Parquet files | {total_rows:,} rows | {total_bytes / (1024 * 1024):.2f} MB on disk")
    if findings:
        print("\nFindings:")
        for level, rel, msg in findings:
            print(f"  [{level}] {rel}: {msg}")
    print(f"\n{n_err} error(s), {n_warn} warning(s)")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(verify())
