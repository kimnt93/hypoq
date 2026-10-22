# vnresearch — Autonomous Quantitative Financial & Macroeconomic Data Lake

[![GitHub repository](https://img.shields.io/badge/GitHub-kimnt93%2Fvnresearch-blue.svg?logo=github)](https://github.com/kimnt93/vnresearch.git)
[![Data Format](https://img.shields.io/badge/Format-Apache%20Parquet-orange.svg)](https://parquet.apache.org/)
[![Query Engine](https://img.shields.io/badge/Engine-Polars%20Vectorized-cd79ff.svg)](https://pola.rs/)
[![Document Parser](https://img.shields.io/badge/Parser-SylphxAI%2Fanymd-c3f53c.svg)](https://github.com/SylphxAI/anymd)

**vnresearch** is a standalone, open quantitative and macroeconomic data lake for emerging and frontier financial market research (focusing on Vietnam equities, index derivatives, corporate governance, and macroeconomic policy).

It provides clean, vectorized, Point-in-Time (PIT) datasets serialized in **Apache Parquet**, paired with autonomous ingestion pipelines that extract regulatory filings, announcements, and corporate disclosures using **[`anymd`](https://github.com/SylphxAI/anymd)**.

---

## 1. Quickstart & Automation (`Makefile`)

The repository includes a dedicated `Makefile` to acquire, parse, verify, and inspect all datasets:

```bash
# Clone the repository
git clone https://github.com/kimnt93/vnresearch.git
cd vnresearch

# Display all available commands
make help

# Run complete data acquisition & ingestion pipeline via anymd
make download

# Validate integrity, schemas, and row counts across all 27 Parquet files
make verify

# Render empirical overview and distribution visualizations
make visualize

# Parse any corporate report, regulatory PDF, or web document to clean Markdown
make parse-doc FILE=path/to/report.pdf [PAGES=1-5]

# Clean temporary download caches
make clean
```

---

## 2. Dataset Architecture & Inventory (`data/`)

All 27 datasets are serialized in **Apache Parquet** format for zero-copy, high-speed analytical queries:

```text
vnresearch/data/
├── macro/
│   ├── widata_macro_all_series.parquet      # 204,426 rows | 107 macro series unpivoted (1970–2026)
│   ├── vietnam_monetary_daily.parquet       # 5,471 rows x 16 cols | Interbank rates, OMO, SBV rates, FX, Yields
│   ├── vietnam_macro_monthly.parquet        # 298 rows x 13 cols | PMI, CPI components, IIP, Retail sales, FDI
│   ├── vietnam_market_pe_daily.parquet      # 2,877 rows x 3 cols | Daily VN-Index P/E valuation (2015–2026)
│   └── us_macro_daily.parquet               # 13,821 rows x 10 cols | Fed Funds DFF, 3M/2Y/10Y Yields, M2, WALCL
├── market/
│   ├── vn_indices_daily.parquet             # 4,570 rows x 8 cols | VNINDEX & VN30 daily OHLCV (2015–2026)
│   ├── vn30f1m_derivatives_daily.parquet    # 2,296 rows x 8 cols | VN30F1M Futures daily OHLCV (2017–2026)
│   ├── vn_top_equities_daily.parquet        # 136,302 rows x 8 cols | 55 constituent stocks across all sectors
│   ├── vn_stock_industry_classification.parquet # 1,971 rows x 22 cols | All listed stocks mapped to 4-tier ICB
│   ├── vn_industry_sectors_icb.parquet      # 185 rows x 7 cols | 185 ICB sector definitions & ticker lists
│   ├── vn_sector_indices_daily.parquet      # 26,113 rows x 8 cols | 12 official sector indices daily OHLCV (2017–2026)
│   ├── vn_foreign_trading_daily.parquet     # 130,837 rows x 12 cols | Daily Foreign Trading (GD NĐTNN: Buy, Sell, Room)
│   └── vn_order_supply_demand_daily.parquet # 83,731 rows x 11 cols | Daily Order Book Supply/Demand (Cung cầu: Orders, Vol)
├── market_structure/
│   ├── vn_shareholders_history.parquet      # 3,115 rows x 9 cols | Major Shareholders & Ownership History (Cổ đông lớn)
│   ├── vn_subsidiaries_affiliates.parquet   # 1,688 rows x 10 cols | Subsidiaries & Affiliates (Công ty con & liên kết)
│   ├── vn_charter_capital_history.parquet   # 4,391 rows x 13 cols | Historical Capital Changes & Issuances (Vốn điều lệ)
│   ├── free_float.parquet                   # 43,377 rows x 18 cols | Historical free-float factor & capping factors (2017–2026)
│   ├── vn30_free_float_history.parquet      # 690 rows x 18 cols | VN30-specific free float & index shares history
│   ├── free_float_changes_observed.parquet  # 10,178 rows x 22 cols | Tracking discrete parameter changes
│   ├── index_constituents.parquet           # 43,377 rows x 18 cols | Historical basket constituents across all VN indices
│   └── basket_history.parquet               # 261 rows x 10 cols | Basket rebalancing events and effective dates
├── financials/
│   ├── vn_financial_models_taxonomy.parquet # 3,370 rows x 12 cols | Full accounting line item definitions (TT202, TT49, TT334)
│   ├── vn_financial_statements_quarterly_yearly.parquet # 325,000 rows x 12 cols | Balance Sheet, Income, Cash Flow line items
│   ├── vn_financial_ratios_quarterly_yearly.parquet     # 189,000 rows x 8 cols  | Financial Ratios (P/E, P/B, ROE, ROA, D/E)
│   └── vn_listed_q3_2026_financials.parquet # 192 rows x 21 cols | Q3-2026 corporate results (Revenue, Profit, P/E)
└── cross_asset/
    └── global_macro_daily.parquet           # 26,358 rows x 8 cols | S&P500, US10Y, DXY, Gold, WTI, VIX, BTC, USDVND
```

Total database volume: **27 Parquet files | 1,400,197 total rows | 15.51 MB on disk**.  
For exact column schemas, mathematical formulas, and data provenance, see [`DATA.md`](./DATA.md).

---

## 3. Empirical Distribution & Overview Figures (`figures_overview/`)

Visualized via `make visualize`:

* **Figure 1**: [`fig1_shareholders_concentration.png`](./figures_overview/fig1_shareholders_concentration.png)  
  *Cross-sectional shareholder concentration and ownership breakdown across key blue-chips (State/Institutional vs. Individual executives).*
* **Figure 2**: [`fig2_corporate_structure_network.png`](./figures_overview/fig2_corporate_structure_network.png)  
  *Corporate group network hierarchy: Number of Subsidiaries (>50% control) vs. Affiliates/Associates (20-50% control) across leading conglomerates.*
* **Figure 3**: [`fig3_cumulative_foreign_flows.png`](./figures_overview/fig3_cumulative_foreign_flows.png)  
  *Cumulative net foreign trading value (`GD NĐTNN` Mua/Bán Ròng Lũy Kế 2018–2026) across key stocks (HPG, FPT, VCB, VHM, MSN, HVN, SSI, MWG).*
* **Figure 4**: [`fig4_order_imbalance_distribution.png`](./figures_overview/fig4_order_imbalance_distribution.png)  
  *Kernel density estimation of daily order book supply-demand imbalance ($\text{OIB} = \frac{\text{Bid} - \text{Offer}}{\text{Bid} + \text{Offer}}$).*
* **Figure 5**: [`fig5_charter_capital_events.png`](./figures_overview/fig5_charter_capital_events.png)  
  *Breakdown of capital adjustment event types (Rights issues, share dividends, private placements) and cumulative share expansion trajectories.*

---

## 4. Querying Datasets with Polars

```python
import polars as pl

# 1. Load Macro & Monetary Conditions
macro = pl.read_parquet("data/macro/vietnam_monetary_daily.parquet")

# 2. Load Free-Float History & Capping Factors
float_history = pl.read_parquet("data/market_structure/vn30_free_float_history.parquet")

# 3. Load Order Book Supply/Demand Imbalance
order_flow = pl.read_parquet("data/market/vn_order_supply_demand_daily.parquet")

# 4. Load Major Shareholders & Ownership History
holders = pl.read_parquet("data/market_structure/vn_shareholders_history.parquet")

# 5. Load Financial Statements & Ratios
statements = pl.read_parquet("data/financials/vn_financial_statements_quarterly_yearly.parquet")
ratios = pl.read_parquet("data/financials/vn_financial_ratios_quarterly_yearly.parquet")
```

---

## 5. License & Citations

All data processing scripts, schemas, and documentation are released under the MIT License. When using this data lake for empirical publications or quantitative backtesting, please cite:

```bibtex
@misc{nguyen2026vnresearch,
  author = {Nguyen, Kim},
  title = {vnresearch: Autonomous Quantitative Financial and Macroeconomic Data Lake for Emerging Markets},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/kimnt93/vnresearch}}
}
```
