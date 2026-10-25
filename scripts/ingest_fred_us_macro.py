#!/usr/bin/env python3
"""
US macro and Federal Reserve balance-sheet tables, rebuilt from FRED (free, key-less CSV endpoint).

Outputs:
- data/macro/us_macro_daily.parquet            daily/weekly/monthly FRED series on their own publication dates
- data/macro/us_fed_balance_sheet_weekly.parquet  H.4.1 weekly balance sheet items, USD billions

Units: WALCL-family series are converted from USD millions to USD billions; rates are percent.
`us_m2_money_supply_yoy` is the 12-month % change of M2SL and is dated at the END of its reference month
(the value is only published weeks later: lag it before using it in a backtest).
Values are verified identical to the previous vendor snapshot on every overlapping date.
"""

import io
import os
import time

import polars as pl
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MACRO_DIR = os.path.join(BASE_DIR, "data", "macro")
os.makedirs(MACRO_DIR, exist_ok=True)
HEADERS = {"User-Agent": "hypoq-research/2.0 (open data ingestion)"}

DAILY = {
    "DFF": "us_fed_funds_rate_dff",
    "DFEDTARU": "us_fed_target_upper",
    "DGS3MO": "us_treasury_3m",
    "DGS2": "us_treasury_2y",
    "DGS10": "us_treasury_10y",
    "SOFRVOL": "us_sofr_volume",
    "EFFRVOL": "us_effr_volume",
}
WEEKLY_BALANCE_SHEET = {
    "WALCL": "fed_total_assets_walcl",
    "WSHOTSL": "fed_treasury_securities_wshotsl",
    "WSHOMCB": "fed_mbs_securities_wshomcb",
    "WDTGAL": "fed_loans_total_wdtgal",
    "WLRRAL": "fed_reverse_repo_wlrral",
    "WCBLSA": "fed_commercial_bank_assets_wcblsa",
}


def fred_series(series_id: str, retries: int = 4) -> pl.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=90)
            r.raise_for_status()
            df = pl.read_csv(io.BytesIO(r.content), null_values=["."])
            df = df.rename({df.columns[0]: "date", df.columns[1]: "value"})
            return df.with_columns(pl.col("date").str.to_date(), pl.col("value").cast(pl.Float64)).drop_nulls()
        except Exception as exc:
            print(f"    ! {series_id}: {type(exc).__name__}: {exc}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"FRED series {series_id} unavailable")


def _wide(series_map: dict, scale: dict = None) -> pl.DataFrame:
    out = None
    for sid, name in series_map.items():
        df = fred_series(sid)
        factor = (scale or {}).get(sid, 1.0)
        df = df.select("date", (pl.col("value") * factor).alias(name))
        out = df if out is None else out.join(df, on="date", how="full", coalesce=True)
        print(f"    - {sid} -> {name}: {len(df):,} obs ({df['date'].min()} .. {df['date'].max()})")
    return out.sort("date")


def build_us_macro_daily():
    print("\n>>> FRED: us_macro_daily")
    wide = _wide(DAILY)
    walcl = fred_series("WALCL").select("date", (pl.col("value") / 1000.0).alias("us_fed_total_assets_walcl"))
    m2 = fred_series("M2SL").sort("date")
    m2_yoy = (
        m2.with_columns((pl.col("value").pct_change(12) * 100.0).alias("us_m2_money_supply_yoy"))
        .drop_nulls("us_m2_money_supply_yoy")
        .select((pl.col("date").dt.month_end()).alias("date"), "us_m2_money_supply_yoy")
    )
    out = (
        wide.join(walcl, on="date", how="full", coalesce=True)
        .join(m2_yoy, on="date", how="full", coalesce=True)
        .sort("date")
        .select(["date", "us_fed_total_assets_walcl", "us_fed_funds_rate_dff", "us_sofr_volume", "us_effr_volume",
                 "us_fed_target_upper", "us_treasury_2y", "us_treasury_10y", "us_treasury_3m", "us_m2_money_supply_yoy"])
    )
    out.write_parquet(os.path.join(MACRO_DIR, "us_macro_daily.parquet"), compression="zstd")
    print(f"    => us_macro_daily.parquet ({len(out):,} rows)")


def build_us_fed_balance_sheet_weekly():
    print("\n>>> FRED: us_fed_balance_sheet_weekly")
    out = _wide(WEEKLY_BALANCE_SHEET, scale={sid: 0.001 for sid in WEEKLY_BALANCE_SHEET})
    out.write_parquet(os.path.join(MACRO_DIR, "us_fed_balance_sheet_weekly.parquet"), compression="zstd")
    print(f"    => us_fed_balance_sheet_weekly.parquet ({len(out):,} rows)")


def run_all():
    build_us_macro_daily()
    build_us_fed_balance_sheet_weekly()


if __name__ == "__main__":
    run_all()
