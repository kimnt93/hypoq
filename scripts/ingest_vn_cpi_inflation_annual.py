#!/usr/bin/env python3
"""
Ingest script: World Bank annual CPI inflation (FP.CPI.TOTL.ZG) for a country.
Source : https://api.worldbank.org (free, no key)
Output : data/macro/vn_cpi_inflation_annual.parquet
"""
import os

import polars as pl
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "macro", "vn_cpi_inflation_annual.parquet")
URL = "https://api.worldbank.org/v2/country/VNM/indicator/FP.CPI.TOTL.ZG"


def fetch_and_build() -> pl.DataFrame:
    resp = requests.get(URL, params={"format": "json", "per_page": 200}, timeout=60)
    resp.raise_for_status()
    records = resp.json()[1]
    df = (
        pl.DataFrame(
            [{"year": int(r["date"]), "cpi_inflation_pct": r["value"]} for r in records],
            schema={"year": pl.Int32, "cpi_inflation_pct": pl.Float64},
        )
        .drop_nulls("cpi_inflation_pct")           # missing years stay missing
        .with_columns(pl.date(pl.col("year"), 12, 31).alias("date"))  # reference-period end, NOT an availability date
        .sort("date")
    )
    if df.is_empty():
        raise RuntimeError("World Bank returned no data; refusing to write an empty dataset")
    df.write_parquet(OUTPUT_FILE, compression="zstd")
    print(f">>> {len(df)} rows -> {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    fetch_and_build()
