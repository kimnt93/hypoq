# vnresearch — Autonomous Quantitative Financial & Macroeconomic Data Lake

[![GitHub repository](https://img.shields.io/badge/GitHub-kimnt93%2Fvnresearch-blue.svg?logo=github)](https://github.com/kimnt93/vnresearch.git)
[![Data Format](https://img.shields.io/badge/Format-Apache%20Parquet-orange.svg)](https://parquet.apache.org/)
[![Query Engine](https://img.shields.io/badge/Engine-Polars%20Vectorized-cd79ff.svg)](https://pola.rs/)
[![Document Parser](https://img.shields.io/badge/Parser-SylphxAI%2Fanymd-c3f53c.svg)](https://github.com/SylphxAI/anymd)

**vnresearch** is an autonomous quantitative financial, corporate network, and macroeconomic data lake for emerging and frontier market research (focusing on Vietnam equities, index derivatives, corporate ownership networks, and monetary policy).

It provides clean, Point-in-Time (PIT) datasets serialized in **Apache Parquet**, paired with autonomous ingestion pipelines that extract regulatory filings, announcements, and corporate disclosures using **[`anymd`](https://github.com/SylphxAI/anymd)**.

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

# Validate integrity, schemas, and row counts across all 29 Parquet files
make verify

# Render empirical overview and distribution visualizations (PNG only)
make visualize

# Parse any corporate report, regulatory PDF, or web document to clean Markdown
make parse-doc FILE=path/to/report.pdf [PAGES=1-5]

# Clean temporary download caches
make clean
```

---

## 2. Complete Dataset Architecture & Inventory (`data/`)

All 29 datasets are serialized in **Apache Parquet** format for zero-copy, high-speed analytical queries:

```text
vnresearch/data/
├── macro/
│   ├── widata_macro_all_series.parquet          # 204,426 rows | 107 macro series unpivoted (1970–2026)
│   ├── vietnam_monetary_daily.parquet           # 5,471 rows x 16 cols | Interbank rates, OMO, SBV rates, FX, Yields
│   ├── vietnam_macro_monthly.parquet            # 298 rows x 13 cols | PMI, CPI components, IIP, Retail sales, FDI
│   ├── vietnam_market_pe_daily.parquet          # 2,877 rows x 3 cols | Daily VN-Index P/E valuation (2015–2026)
│   ├── us_macro_daily.parquet                   # 13,821 rows x 10 cols | Fed Funds DFF, 3M/2Y/10Y Yields, M2, WALCL
│   └── vietnam_commodities_daily.parquet        # 2,285 rows x 17 cols | Construction Steel, Pork, Fuels, Coffee, Rice
├── market/
│   ├── vn_indices_daily.parquet                 # 4,570 rows x 8 cols | VNINDEX & VN30 daily OHLCV (2015–2026)
│   ├── vn30f1m_derivatives_daily.parquet        # 2,296 rows x 8 cols | VN30F1M Futures daily OHLCV (2017–2026)
│   ├── vn_top_equities_daily.parquet            # 136,302 rows x 8 cols | 55 constituent stocks across all sectors
│   ├── vn_stock_industry_classification.parquet # 1,971 rows x 22 cols | All listed stocks mapped to 4-tier ICB
│   ├── vn_industry_sectors_icb.parquet          # 185 rows x 7 cols | 185 ICB sector definitions & ticker lists
│   ├── vn_sector_indices_daily.parquet          # 26,113 rows x 8 cols | 12 official sector indices daily OHLCV
│   ├── vn_foreign_trading_daily.parquet         # 130,837 rows x 12 cols | Daily Foreign Trading (GD NĐTNN: Buy, Sell, Room)
│   ├── vn_order_supply_demand_daily.parquet     # 83,731 rows x 11 cols | Daily Order Book Supply/Demand (Cung cầu)
│   ├── vn_etf_market_daily.parquet              # 22,377 rows x 12 cols | 15 Listed Vietnamese ETFs Daily OHLCV (2014–2026)
│   └── vn_etf_nav_aum_daily.parquet             # 18,512 rows x 10 cols | Daily Net Asset Value (NAV), AUM & Premium/Discount
├── market_structure/
│   ├── vn_etf_portfolio_holdings_daily.parquet  # 58 rows x 8 cols | Daily ETF Constituent Creation Baskets (Danh mục hoán đổi)
│   ├── vn_corporate_network_edges.parquet       # 4,803 rows x 12 cols | Corporate Network Directed Edges (Loans, Subs, Owners)
│   ├── vn_corporate_network_nodes.parquet       # 3,473 rows x 8 cols | Corporate Network Nodes & Centrality (PageRank, Degree)
│   ├── vn_shareholders_history.parquet          # 3,115 rows x 9 cols | Major Shareholders & Ownership History (Cổ đông lớn)
│   ├── vn_subsidiaries_affiliates.parquet       # 1,688 rows x 10 cols | Subsidiaries & Affiliates (Công ty con & liên kết)
│   ├── vn_charter_capital_history.parquet       # 4,391 rows x 13 cols | Historical Capital Changes & Issuances (Vốn điều lệ)
│   ├── free_float.parquet                       # 43,377 rows x 18 cols | Historical free-float factor & capping factors
│   ├── vn30_free_float_history.parquet          # 690 rows x 18 cols | VN30-specific free float & index shares history
│   ├── free_float_changes_observed.parquet      # 10,178 rows x 22 cols | Tracking discrete parameter changes
│   ├── index_constituents.parquet               # 43,377 rows x 18 cols | Historical basket constituents across all VN indices
│   └── basket_history.parquet                   # 261 rows x 10 cols | Basket rebalancing events and effective dates
├── financials/
│   ├── vn_financial_models_taxonomy.parquet     # 3,370 rows x 12 cols | Full accounting line item definitions (TT202, TT49)
│   ├── vn_financial_statements_quarterly_yearly.parquet # 325,000 rows x 12 cols | Balance Sheet, Income, Cash Flow line items
│   ├── vn_financial_ratios_quarterly_yearly.parquet     # 189,000 rows x 8 cols  | Financial Ratios (P/E, P/B, ROE, ROA, D/E)
│   └── vn_listed_q3_2026_financials.parquet     # 192 rows x 21 cols | Q3-2026 corporate results (Revenue, Profit, P/E)
└── cross_asset/
    └── global_macro_daily.parquet               # 26,358 rows x 8 cols | S&P500, US10Y, DXY, Gold, WTI, VIX, BTC, USDVND
```

Total database volume: **33 Parquet files | 1,451,705 total rows | 16.45 MB on disk**.  
For exact column schemas, mathematical formulas, and data provenance, see [`DATA.md`](./DATA.md).

---

## 3. Empirical Distribution Figures (`figures_overview/`)

Visualized via `make visualize` (Output format: **PNG ONLY**, zero PDF):

* **Figure 1**: [`fig1_shareholders_concentration.png`](./figures_overview/fig1_shareholders_concentration.png)  
  *Cross-sectional shareholder concentration across key blue-chips (State/Institutional vs. Individual executives).*
* **Figure 2**: [`fig2_corporate_structure_network.png`](./figures_overview/fig2_corporate_structure_network.png)  
  *Corporate group hierarchy: Number of Subsidiaries (>50% control) vs. Affiliates/Associates (20-50% control).*
* **Figure 3**: [`fig3_cumulative_foreign_flows.png`](./figures_overview/fig3_cumulative_foreign_flows.png)  
  *Cumulative net foreign trading value (`GD NĐTNN` Mua/Bán Ròng Lũy Kế 2018–2026) across key blue-chips.*
* **Figure 4**: [`fig4_order_imbalance_distribution.png`](./figures_overview/fig4_order_imbalance_distribution.png)  
  *Kernel density estimation of daily order book supply-demand imbalance ($\text{OIB} = \frac{\text{Bid} - \text{Offer}}{\text{Bid} + \text{Offer}}$).*
* **Figure 5**: [`fig5_charter_capital_events.png`](./figures_overview/fig5_charter_capital_events.png)  
  *Breakdown of capital actions (Rights issues, share dividends, private placements) and share expansion.*
* **Figure 6**: [`fig6_corporate_network_graph.png`](./figures_overview/fig6_corporate_network_graph.png)  
  *Directed network graph of Vietnamese corporate cross-ownership, state control, and auditing relationships.*
* **Figure 7**: [`fig7_vietnam_commodity_prices.png`](./figures_overview/fig7_vietnam_commodity_prices.png)  
  *Domestic commodity trajectories: Petrolimex fuel cycles, Hòa Phát construction steel, regional live hog/pork, and Robusta coffee.*
* **Figure 8**: [`fig8_etf_market_dynamics.png`](./figures_overview/fig8_etf_market_dynamics.png)  
  *ETF asset trajectories, foreign premium to NAV across Diamond ETF, top constituent portfolio weights, and sector exposure.*

---

## 4. Querying Datasets: Polars vs. Pandas Solutions

### A. Point-in-Time Alignment (Macro to Daily Market Data)
* **Polars (Vectorized Zero-Copy)**:
  ```python
  import polars as pl
  
  df_idx = pl.read_parquet("data/market/vn_indices_daily.parquet").filter(pl.col("symbol") == "VNINDEX")
  df_macro = pl.read_parquet("data/macro/vietnam_macro_monthly.parquet")
  
  aligned = df_idx.join_asof(
      df_macro,
      left_on=pl.col("date").str.to_date(),
      right_on=pl.col("date").str.to_date(),
      strategy="backward"
  )
  ```
* **Pandas**:
  ```python
  import pandas as pd
  
  df_idx = pd.read_parquet("data/market/vn_indices_daily.parquet").query("symbol == 'VNINDEX'")
  df_macro = pd.read_parquet("data/macro/vietnam_macro_monthly.parquet")
  
  df_idx["date"] = pd.to_datetime(df_idx["date"])
  df_macro["date"] = pd.to_datetime(df_macro["date"])
  
  aligned = pd.merge_asof(df_idx.sort_values("date"), df_macro.sort_values("date"), on="date", direction="backward")
  ```

### B. Corporate Network Traversal & PageRank Centrality
* **Polars**:
  ```python
  import polars as pl
  
  nodes = pl.read_parquet("data/market_structure/vn_corporate_network_nodes.parquet")
  edges = pl.read_parquet("data/market_structure/vn_corporate_network_edges.parquet")
  
  # Top 10 most central systemic entities
  top_central = nodes.sort("pagerank", descending=True).head(10)
  
  # Direct subsidiary control edges for Vingroup
  vic_subs = edges.filter((pl.col("source_id") == "VIC") & (pl.col("relationship_type") == "SUBSIDIARY_CONTROL"))
  ```
* **Pandas**:
  ```python
  import pandas as pd
  
  nodes = pd.read_parquet("data/market_structure/vn_corporate_network_nodes.parquet")
  edges = pd.read_parquet("data/market_structure/vn_corporate_network_edges.parquet")
  
  top_central = nodes.sort_values("pagerank", ascending=False).head(10)
  vic_subs = edges[(edges["source_id"] == "VIC") & (edges["relationship_type"] == "SUBSIDIARY_CONTROL")]
  ```

---

## 5. Verified Data Sources & Reference Links

1. **WiData Financial & Macro Engine**: High-frequency macroeconomic indicators, interbank market rates, SBV bills/OMO operations.  
   *URL*: [https://widata.vn/vi-mo/vn/pmi-san-xuat](https://widata.vn/vi-mo/vn/pmi-san-xuat) | [https://widata.vn/ket-qua-kinh-doanh](https://widata.vn/ket-qua-kinh-doanh)
2. **State Bank of Vietnam (SBV / NHNN)**: Central bank reference exchange rates, refinancing rates, discount rates, interbank data.  
   *URL*: [https://www.sbv.gov.vn](https://www.sbv.gov.vn)
3. **VNDirect Financial Information (Finfo Open API)**: Listed stock registry, 4-tier ICB taxonomy, financial statements (TT200/TT202/TT49), financial ratios, daily order book supply/demand (`trading_statistics`), daily foreign trading (`foreigns`), shareholders, and subsidiaries.  
   *URL*: [https://dchart.vndirect.com.vn](https://dchart.vndirect.com.vn) | [https://www.vndirect.com.vn](https://www.vndirect.com.vn)
4. **Ho Chi Minh Stock Exchange (HOSE) & Hanoi Stock Exchange (HNX)**: Official Point-in-Time constituent rebalancing announcements, free-float factors ($f$), capping factors ($c$), and index shares.  
   *URL*: [https://www.hsx.vn](https://www.hsx.vn) | [https://www.hnx.vn](https://www.hnx.vn)
5. **Ministry of Finance (Bộ Tài chính)**: Statutory accounting circulars governing financial statement reporting lines (`TT200/2014/TT-BTC`, `TT202/2014/TT-BTC`, `TT334/2016/TT-BTC`, `TT199/2014/TT-BTC`).  
   *URL*: [https://mof.gov.vn](https://mof.gov.vn)
6. **Yahoo Finance Open Engine**: Multi-asset global macro pricing benchmarks: S&P 500 (`^GSPC`), US 10Y Yield (`^TNX`), US Dollar Index (`DX-Y.NYB`), WTI Crude Oil (`CL=F`), Gold (`GC=F`), VIX (`^VIX`), Bitcoin (`BTC-USD`), and USD/VND (`VND=X`).  
   *URL*: [https://finance.yahoo.com](https://finance.yahoo.com)
7. **Federal Reserve Bank of St. Louis (FRED)**: US Federal Reserve monetary policies, Treasuries, and money supply series.  
   *URL*: [https://fred.stlouisfed.org](https://fred.stlouisfed.org)
8. **SylphxAI anymd**: Fast native Rust document & PDF parser.  
   *URL*: [https://github.com/SylphxAI/anymd](https://github.com/SylphxAI/anymd)

---

## 6. License & Citations

All data processing scripts, schemas, and documentation are released under the MIT License:

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
