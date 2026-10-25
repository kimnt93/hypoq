---
name: polars-fast-data
description: High-performance, zero-copy data manipulation using Polars, including point-in-time backward joins, rolling windows, and vectorized financial calculations.
version: 1.0.0
author: hypoq
tags: [polars, dataframe, performance, zero-copy, point-in-time]
---

# Polars Fast Data Skill

## Overview
This skill establishes optimal Polars patterns for financial econometrics and alpha research. Avoid avoidable allocations, Python-level for-loops, and Pandas overhead by using Polars zero-copy vectorized expressions and lazy evaluation.

## When to Use
- Merging daily asset prices with monthly macro indicators or quarterly accounting statements without look-ahead bias (`join_asof` backward).
- Computing rolling return volatility, moving averages, Sharpe ratios, and drawdowns.
- Replicating index constituent market weights ($w_{i,t} = \frac{P_{i,t} \cdot Q_{i,t} \cdot f_{i,t} \cdot c_{i,t}}{\sum P \cdot Q \cdot f \cdot c}$).
- Filtering large tick/order book datasets with predicate pushdown.

## Non-Negotiable Best Practices
1. **Never use `.to_pandas()`** unless interfacing with a library that strictly demands NumPy/Pandas (e.g. Statsmodels OLS). Convert only the final regression matrix at the very end.
2. **Strict Point-in-Time (PIT) Backward Joins**:
   ```python
   # CORRECT: Past disclosures matched to current/future dates
   aligned = df_daily.join_asof(
       df_macro,
       left_on="date",
       right_on="date",
       strategy="backward"
   )
   ```
3. **Column Projection**: When loading large files, pass `columns=[...]` to read only necessary fields.
4. **Lazy Execution**: Use `pl.scan_parquet(...)` for filtering and aggregations before `collect()`.

## Core Implementation Recipes

### Recipe 1: Computing Log Returns & Rolling Volatility
```python
import polars as pl

df = pl.read_parquet("data/market/vn_indices_daily.parquet")

returns_df = (
    df.filter(pl.col("symbol") == "VNINDEX")
    .sort("date")
    .with_columns([
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_return")
    ])
    .with_columns([
        (pl.col("log_return").rolling_std(window_size=20) * (252 ** 0.5)).alias("vol_20d_annualized"),
        (pl.col("log_return").rolling_mean(window_size=20) * 252).alias("mean_ret_20d_annualized")
    ])
)
```

### Recipe 2: Futures Basis & Cost of Carry Decomposition
$$\text{Basis}_t = \ln(F_t) - \ln(S_t), \quad \text{Carry}_t = r_{\text{interbank}, t} \times \frac{\text{DTE}}{365}$$
```python
import polars as pl

f_df = pl.read_parquet("data/market/vn_vn30f1m_derivatives_daily.parquet").select([
    pl.col("date").str.to_date(),
    pl.col("close").alias("f_close")
])

s_df = pl.read_parquet("data/market/vn_indices_daily.parquet").filter(
    pl.col("symbol") == "VN30"
).select([
    pl.col("date").str.to_date(),
    pl.col("close").alias("s_close")
])

basis_df = (
    f_df.join(s_df, on="date", how="inner")
    .sort("date")
    .with_columns([
        (pl.col("f_close").log() - pl.col("s_close").log()).alias("basis_log"),
        ((pl.col("f_close") - pl.col("s_close")) / pl.col("s_close") * 100).alias("basis_pct")
    ])
)
```

### Recipe 3: Free-Float Index Weight Replication
```python
import polars as pl

prices = pl.read_parquet("data/market/vn_top_equities_daily.parquet")
floats = pl.read_parquet("data/market_structure/vn_free_float.parquet")

# Join point-in-time float factor
weighted = (
    prices.join_asof(
        floats.sort("date"),
        left_on="date",
        right_on="date",
        by="symbol",
        strategy="backward"
    )
    .with_columns([
        (pl.col("close") * pl.col("shares_outstanding") * pl.col("free_float_ratio") * pl.col("capping_factor")).alias("float_market_cap")
    ])
    .with_columns([
        (pl.col("float_market_cap") / pl.col("float_market_cap").sum().over("date")).alias("index_weight")
    ])
)
```
