---
name: data-discovery
description: Discover, inspect schema, temporal ranges, row counts, and summary statistics across the 36 Parquet datasets in the quantitative data lake.
version: 1.0.0
author: kim / vnresearch
tags: [data, discovery, parquet, schema, datasets]
---

# Data Discovery Skill

## Overview
This skill provides standardized procedures to explore, audit, and inspect the 36 Parquet datasets archived in `data/` across five domains (`macro/`, `market/`, `market_structure/`, `financials/`, `cross_asset/`).

## When to Use
- At the start of an autonomous research task to identify candidate datasets for testing a hypothesis.
- To inspect column names, data types, date coverage, row counts, and missingness before drafting code.
- To verify that required variables (e.g., foreign trading, order book depth, interbank rates, free-float factors) exist and cover the target testing sample.

## Dataset Inventory Reference
```text
data/
├── macro/
│   ├── vietnam_monetary_daily.parquet           # Interbank rates, OMO, SBV rates, FX, Yields
│   ├── vietnam_macro_monthly.parquet            # PMI, CPI components, IIP, Retail sales, FDI
│   ├── vietnam_commodities_daily.parquet        # Construction Steel, Pork, Fuels, Coffee, Rice
│   ├── vietnam_trade_exports_monthly.parquet    # Merchandise Exports (Steel, Coffee, Total)
│   ├── vietnam_fdi_sectors_monthly.parquet      # Realized FDI by Sector
│   ├── vietnam_sbv_policy_rates_daily.parquet   # SBV Policy Rates & Required Reserves
│   ├── us_macro_daily.parquet                   # Fed Funds DFF, Yields (3M/2Y/10Y), M2, WALCL
│   ├── us_fed_balance_sheet_weekly.parquet      # Fed Balance Sheet Assets, Loans, MBS & RRP
│   └── vietnam_market_pe_daily.parquet          # Daily VN-Index P/E valuation (2015-2026)
├── market/
│   ├── vn_indices_daily.parquet                 # VNINDEX & VN30 daily OHLCV
│   ├── vn30f1m_derivatives_daily.parquet        # VN30F1M Futures daily OHLCV
│   ├── vn_top_equities_daily.parquet            # 55 constituent stocks daily OHLCV
│   ├── vn_stock_industry_classification.parquet # All listed stocks mapped to 4-tier ICB
│   ├── vn_industry_sectors_icb.parquet          # 185 ICB sector definitions & ticker lists
│   ├── vn_sector_indices_daily.parquet          # 12 official sector indices daily OHLCV
│   ├── vn_foreign_trading_daily.parquet         # Daily Foreign Trading (Buy, Sell, Room)
│   ├── vn_order_supply_demand_daily.parquet     # Daily Order Supply/Demand (Orders, Vol)
│   ├── vn_etf_market_daily.parquet              # 15 Listed Vietnamese ETFs Daily OHLCV
│   └── vn_etf_nav_aum_daily.parquet             # Daily NAV, AUM & Premium/Discount
├── market_structure/
│   ├── vn_etf_portfolio_holdings_daily.parquet  # Daily ETF Creation Baskets
│   ├── vn_corporate_network_edges.parquet       # Directed Edges (Loans, Subs, Owners)
│   ├── vn_corporate_network_nodes.parquet       # Nodes & Centrality (PageRank, Degree)
│   ├── vn_shareholders_history.parquet          # Major Shareholders & Ownership History
│   ├── vn_subsidiaries_affiliates.parquet       # Subsidiaries & Affiliates
│   ├── vn_charter_capital_history.parquet       # Historical Capital Changes & Issuances
│   ├── free_float.parquet                       # Historical free-float factor & capping factors
│   ├── vn30_free_float_history.parquet          # VN30-specific free float & index shares history
│   ├── free_float_changes_observed.parquet      # Discrete parameter changes
│   ├── index_constituents.parquet               # Historical basket constituents across all VN indices
│   └── basket_history.parquet                   # Basket rebalancing events and effective dates
├── financials/
│   ├── vn_financial_models_taxonomy.parquet     # Accounting line item definitions (TT200/TT202/TT49)
│   ├── vn_financial_statements_quarterly_yearly.parquet # Balance Sheet, Income, Cash Flow line items
│   ├── vn_financial_ratios_quarterly_yearly.parquet     # Standardized financial ratios (P/E, ROE, Margins)
│   └── vn_listed_q3_2026_financials.parquet     # Corporate quarterly results
└── cross_asset/
    └── global_macro_daily.parquet               # S&P500, US10Y, DXY, Gold, WTI, VIX, BTC, USDVND
```

## Discovery Patterns (Polars)

### Pattern 1: Inspect Schema and Metadata
```python
import polars as pl

# Scan parquet metadata without loading full dataset into RAM
lf = pl.scan_parquet("data/market/vn_indices_daily.parquet")
print("Schema:", lf.schema)

# Fetch first 5 rows
preview = lf.limit(5).collect()
print(preview)
```

### Pattern 2: Check Temporal Range and Tickers
```python
import polars as pl

df = pl.read_parquet(
    "data/market/vn_indices_daily.parquet",
    columns=["date", "symbol", "close"]
)

summary = df.group_by("symbol").agg([
    pl.col("date").min().alias("start_date"),
    pl.col("date").max().alias("end_date"),
    pl.len().alias("n_rows")
])
print(summary)
```

### Pattern 3: Audit Missing Values
```python
import polars as pl

df = pl.read_parquet("data/macro/vietnam_monetary_daily.parquet")
null_counts = df.null_count()
print(null_counts)
```
