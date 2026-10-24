---
name: chunked-data-exploration
description: Explore, scan, filter, and analyze partitioned and chunked Parquet datasets (>100MB or multi-part splits) using Polars LazyFrames, predicate pushdown, partition pruning, and out-of-core streaming without memory bloat.
version: 1.0.0
author: AI Quantitative Research Network
tags: [parquet, chunking, out-of-core, streaming, polars, lazyframe, predicate-pushdown, big-data]
---

# Chunked Parquet Data Exploration & Multi-Part Streaming Skill

## Overview
This skill defines the operational protocol for autonomous AI agents to explore, query, and join **partitioned, chunked, or multi-part Parquet datasets** without loading entire files into host RAM.

### Core Constraints & GitHub Limit Invariant
1. **GitHub 100MB File Boundary**: Any single file exceeding 100MB is blocked by GitHub. When empirical quantitative series expand beyond ~90MB, they are automatically partitioned into balanced part chunks (`dataset_part001.parquet`, `dataset_part002.parquet`, etc.) or temporal directories (`dataset/year=YYYY/*.parquet`).
2. **Never Read Entire Chunked Tables into RAM**: Never use `pl.read_parquet(...)` across a list of large multi-part files.
3. **Always Scan Lazily**: Always instantiate `pl.scan_parquet(...)` and filter with predicate pushdown before calling `.collect(streaming=True)`.

---

## 1. Multi-Part Scanning & Automatic Chunk Discovery

Polars natively supports glob expressions to scan all parts as a single continuous virtual table:

```python
import polars as pl
import glob
import os

def scan_chunked_dataset(dataset_pattern: str) -> pl.LazyFrame:
    """
    Seamlessly scans single Parquet files or multi-part chunked patterns with zero RAM overhead.
    
    Examples:
        lf = scan_chunked_dataset("data/market/us_equities_daily*.parquet")
        lf = scan_chunked_dataset("data/market/chunked_orderbook/*.parquet")
    """
    matching_files = sorted(glob.glob(dataset_pattern))
    if not matching_files:
        raise FileNotFoundError(f"No Parquet files found matching pattern: {dataset_pattern}")
        
    print(f">>> Scanning {len(matching_files)} Parquet partition(s):")
    for f in matching_files[:3]:
        mb = os.path.getsize(f) / (1024 * 1024)
        print(f"    - {os.path.basename(f)} ({mb:.2f} MB)")
    if len(matching_files) > 3:
        print(f"    - ... and {len(matching_files) - 3} more parts.")

    # Polars scans all matching files in parallel with zero-copy
    lf = pl.scan_parquet(dataset_pattern)
    return lf
```

---

## 2. Predicate Pushdown & Partition Pruning

When filtering by symbol, date range, or region, Polars pushes filters down to the Parquet row-group level, reading only the necessary bytes from disk:

```python
import polars as pl
from datetime import date

def query_ticker_time_window(
    dataset_pattern: str,
    target_symbol: str,
    start_date: str,
    end_date: str
) -> pl.DataFrame:
    """
    Demonstrates zero-copy predicate pushdown across chunked partitions.
    """
    lf = pl.scan_parquet(dataset_pattern)
    
    # Filters are applied BEFORE decompression and loading into memory
    filtered_df = (
        lf.filter(
            (pl.col("symbol") == target_symbol) &
            (pl.col("date") >= pl.lit(start_date).str.to_date()) &
            (pl.col("date") <= pl.lit(end_date).str.to_date())
        )
        .sort("date")
        .collect()
    )
    
    print(f"Query returned {len(filtered_df):,} rows for {target_symbol} ({start_date} to {end_date})")
    return filtered_df
```

---

## 3. Streaming Out-of-Core Aggregations

For calculating summary metrics (mean returns, realized volatility, max drawdown, value traded) across millions of rows without memory spikes, activate Polars streaming mode:

```python
import polars as pl

def compute_chunked_summary_stats(dataset_pattern: str) -> pl.DataFrame:
    """
    Computes cross-sectional summary statistics using out-of-core streaming execution.
    """
    lf = pl.scan_parquet(dataset_pattern)
    
    summary = (
        lf.group_by("symbol")
        .agg([
            pl.len().alias("total_observations"),
            pl.col("date").min().alias("start_date"),
            pl.col("date").max().alias("end_date"),
            pl.col("close").last().alias("latest_close"),
            pl.col("returns").mean().alias("mean_daily_return"),
            (pl.col("returns").std() * 100 * (252**0.5)).alias("annualized_vol_pct"),
            pl.col("volume").mean().alias("mean_daily_volume"),
        ])
        .sort("total_observations", descending=True)
        .collect(streaming=True)  # Streams batches through RAM without full materialization
    )
    return summary
```

---

## 4. Multi-Dataset Point-in-Time (PIT) As-Of Joins on Chunked Data

To execute an econometric event study or factor backtest combining chunked market prices with low-frequency macro announcements:

```python
import polars as pl

def point_in_time_chunked_join(
    chunked_equity_pattern: str,
    macro_parquet_path: str,
    target_ticker: str
) -> pl.DataFrame:
    """
    Executes a Point-in-Time backward alignment between chunked equities and macro indicators.
    """
    # 1. Lazily scan chunked equity files and filter target ticker
    lf_equity = (
        pl.scan_parquet(chunked_equity_pattern)
        .filter(pl.col("symbol") == target_ticker)
        .sort("date")
    )
    
    # 2. Lazily scan macro indicators
    lf_macro = pl.scan_parquet(macro_parquet_path).sort("date")
    
    # 3. Perform backward as-of join with zero look-ahead bias
    df_joined = (
        lf_equity.join_asof(
            lf_macro,
            on="date",
            strategy="backward"
        )
        .collect()
    )
    return df_joined
```

---

## 5. Agent Verification Checklist
When discovering or generating data:
1. **Size Inspection**: Run `scripts/chunk_large_datasets.py` to confirm all files are $< 90$ MB.
2. **Schema Uniformity**: Verify that all `_part*.parquet` files share identical column names and strict data types.
3. **Glob Safety**: In Python research pipelines (`code/run_pipeline.py`), specify `path_or_glob = "data/domain/name*.parquet"` rather than fixed single file names whenever datasets may be chunked.
