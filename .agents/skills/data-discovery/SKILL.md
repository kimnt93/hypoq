---
name: data-discovery
description: Discover, inspect schemas, temporal ranges, row counts, and summary statistics across Parquet datasets, ingest new data series via scripts, synchronize DATA.md documentation, and perform continuous incremental updates.
version: 1.1.0
author: kim / vnresearch
tags: [data, discovery, parquet, schema, datasets, ingestion, documentation, incremental, updates]
---

# Data Discovery & Continuous Ingestion Skill

## Overview
This skill defines the complete operational protocol for managing the quantitative data lake in `data/`:
1. **Discovering & Auditing**: Inspecting schemas, row counts, null rates, and date ranges across the six core data lake domains (`macro/`, `market/`, `market_structure/`, `financials/`, `cross_asset/`, `crypto/`).
2. **Ingesting New Datasets**: Writing standalone or modular ingestion scripts under `scripts/ingest_*.py` when researchers request new financial or macroeconomic series.
3. **Continuous Incremental Updates**: Appending new observations without rewriting historical data from scratch, ensuring zero look-ahead bias and deduplication by primary keys.
4. **Documentation Synchronization**: Keeping `DATA.md` strictly synchronized with column definitions, data types, sources, units, and row counts.

---

## 1. Dataset Jurisdiction Naming Standard
Every empirical dataset stored under `data/` must strictly use an ISO country or geographic scope prefix:
* **`asia_`**: Asian regional indices, equities, and currencies (e.g. `data/market/asia_indices_daily.parquet`, `data/market/asia_equities_daily.parquet`, `data/macro/asia_macro_daily.parquet`).
* **`us_`**: United States Federal Reserve, Treasury term structure, and equities (e.g. `data/market/us_indices_daily.parquet`, `data/market/us_equities_daily.parquet`, `data/macro/us_yield_curve_daily.parquet`).
* **`eu_`**: European Central Bank, sovereign yields, EU indices, and FX series (e.g. `data/market/eu_indices_daily.parquet`, `data/macro/eu_macro_daily.parquet`).
* **`global_`**: International multi-asset cross-border benchmarks, factor ETFs, and commodities (e.g. `data/market/global_etfs_daily.parquet`, `data/macro/global_commodities_daily.parquet`, `data/cross_asset/global_macro_daily.parquet`).
* **`crypto_`**: Cryptocurrency markets, Layer 1 benchmarks, and crypto-macro aggregate indicators (e.g. `data/crypto/crypto_top_assets_daily.parquet`, `data/crypto/crypto_market_daily.parquet`).
* **`vn_`**: Vietnam domestic financial, corporate, and macroeconomic series.
---

## 2. Discovery & Auditing Protocols

### Pattern 1: Zero-Copy Schema & Metadata Inspection
Never load multi-gigabyte files into RAM just to check columns or types. Use Polars lazy scanning:
```python
import polars as pl

# Zero-copy schema inspection
lf = pl.scan_parquet("data/market/vn_etf_market_daily.parquet")
print("Columns & Types:", lf.schema)

# Preview first 5 rows
preview = lf.limit(5).collect()
print(preview)
```

### Pattern 2: Temporal Coverage & Entity Summary
```python
import polars as pl

df = pl.read_parquet(
    "data/market/vn_etf_market_daily.parquet",
    columns=["date", "symbol", "close", "volume"]
)

summary = df.group_by("symbol").agg([
    pl.col("date").min().alias("start_date"),
    pl.col("date").max().alias("end_date"),
    pl.len().alias("n_observations"),
    pl.col("close").null_count().alias("null_closes"),
    pl.col("close").mean().alias("mean_close")
]).sort("n_observations", descending=True)

print(summary)
```

### Pattern 3: Missing Value and Nullness Audit
```python
import polars as pl

df = pl.read_parquet("data/macro/vn_monetary_daily.parquet")
null_summary = df.select([
    pl.col(c).null_count().alias(f"{c}_nulls") for c in df.columns
])
print(null_summary)
```

---

## 3. Ingesting New Datasets via Scripts

When a user requests a new data series (e.g., Vietnam Corporate Bond Yields, Credit Growth Targets, Shipping Freight Rates, or Sovereign CDS Spreads):

### Step 1: Create Dedicated Ingestion Script
Create a new script under `scripts/ingest_<domain>_<dataset_name>.py` (or extend `scripts/build_datasets.py`):
```python
#!/usr/bin/env python3
"""
Ingest script for Vietnam Corporate Bond Index.
Source: HNX / FiinPro / WiData API
Output: data/market/vn_corporate_bonds_daily.parquet
"""

import os
import requests
import polars as pl
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "market")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, "vn_corporate_bonds_daily.parquet")

def fetch_and_build():
    print(f">>> Ingesting corporate bond series into {OUTPUT_FILE}...")
    
    # 1. Fetch raw data from API or parse document using anymd
    # raw_records = fetch_from_api(...)
    raw_records = [
        {"date": "2026-01-02", "bond_index": 105.4, "yield_to_maturity": 7.85, "volume_vnd": 450e9},
        {"date": "2026-01-05", "bond_index": 105.6, "yield_to_maturity": 7.82, "volume_vnd": 510e9},
    ]

    # 2. Structure into Polars DataFrame with strict typing
    df = (
        pl.DataFrame(raw_records)
        .with_columns([
            pl.col("date").str.to_date(),
            pl.col("bond_index").cast(pl.Float64),
            pl.col("yield_to_maturity").cast(pl.Float64),
            pl.col("volume_vnd").cast(pl.Float64),
        ])
        .sort("date")
    )

    # 3. Serialize to Parquet with Zstandard compression
    df.write_parquet(OUTPUT_FILE, compression="zstd")
    print(f">>> Successfully serialized {len(df)} rows to {OUTPUT_FILE}")
    return df

if __name__ == "__main__":
    fetch_and_build()
```

### Step 2: Register in Master Ingestion Pipeline
Add the new ingest step to `scripts/download_all.py` so that `make download` runs it automatically:
```python
from scripts.ingest_corporate_bonds import fetch_and_build as build_corporate_bonds

def run_download_all():
    # ...
    build_corporate_bonds()
```

---

## 4. Continuous & Incremental Ingestion Recipe

When new data needs to be continued or updated without full historical re-downloads:

```python
import os
import polars as pl

def update_or_append_parquet(
    parquet_path: str,
    new_data: pl.DataFrame,
    primary_keys: list = ["date", "symbol"],
    compression: str = "zstd"
) -> pl.DataFrame:
    """
    Incrementally appends new observations to an existing Parquet dataset.
    Deduplicates records by primary keys, keeping the latest observation.
    """
    if os.path.exists(parquet_path):
        existing_df = pl.read_parquet(parquet_path)
        combined_df = (
            pl.concat([existing_df, new_data], how="vertical_relaxed")
            .unique(subset=primary_keys, keep="last")
            .sort(primary_keys)
        )
        added_rows = len(combined_df) - len(existing_df)
        print(f"Updated {parquet_path}: +{added_rows} new rows (Total: {len(combined_df)})")
    else:
        combined_df = new_data.sort(primary_keys)
        print(f"Created new dataset {parquet_path} ({len(combined_df)} rows)")

    combined_df.write_parquet(parquet_path, compression=compression)
    return combined_df
```

---

## 5. Synchronizing Dataset Documentation (`DATA.md`)

Every added or updated dataset **must** be immediately documented in `DATA.md` following the standardized format:

### 1. Update Inventory Table in Section 1
Update the directory inventory tree and total row count:
```text
data/
├── market/
│   ├── vn_corporate_bonds_daily.parquet # 1,245 rows x 4 cols | Bond index, YTM, volume
```

### 2. Add Standardized Single-Table Specification
```markdown
#### X. `vn_corporate_bonds_daily.parquet`
* **Coverage**: 2020-01-02 to 2026-10-23 (1,245 daily observations, 4 columns).
* **Source**: Hanoi Stock Exchange (HNX) / FiinPro.
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `bond_index` | `Float64` | Index | Benchmark Corporate Bond Clean Price Index |
  | `yield_to_maturity` | `Float64` | % / year | Volume-weighted average yield to maturity |
  | `volume_vnd` | `Float64` | VND | Total daily corporate bond trading value |
```

---

## 6. Validation Checklist Before Committing
1. **Schema Integrity**: Run `make verify` (`python scripts/verify_datasets.py`) to confirm row counts, non-empty columns, and clean Parquet metadata.
2. **Zero Look-Ahead Bias**: Ensure that low-frequency macro or corporate event dates reflect official public release timestamps (`\tau_{\text{info}} \le \tau_{\text{market}}`).
3. **No Duplicate Rows**: Confirm uniqueness across primary key columns (`(date, symbol)` or `date`).
4. **Zstandard Compression**: Confirm all output files use `compression="zstd"`.
5. **Docs Sync**: Ensure `DATA.md` has the new table specification and correct total inventory counts.
