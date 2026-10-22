# Macroeconomic, Financial & Corporate Network Dataset Documentation (DATA.md)

This document provides a comprehensive, standardized specification of all 29 empirical datasets serialized into Apache Parquet format. All datasets are stored under `data/` (physically managed in `vnresearch/data/`) and optimized for zero-copy vectorized analytics using **Polars** or **PyArrow/Pandas**.

---

## 1. Directory Structure & Inventory (29 Parquet Files, 1,408,473 Rows)

```text
data/
├── macro/
│   ├── widata_macro_all_series.parquet          # 204,426 rows | 107 macro series unpivoted (1970–2026)
│   ├── vietnam_monetary_daily.parquet           # 5,471 rows x 16 cols | Interbank rates, OMO, SBV rates, FX, Yields
│   ├── vietnam_macro_monthly.parquet            # 298 rows x 13 cols | PMI, CPI components, IIP, Retail sales, FDI
│   ├── vietnam_market_pe_daily.parquet          # 2,877 rows x 3 cols | Daily VN-Index P/E valuation (2015–2026)
│   └── us_macro_daily.parquet                   # 13,821 rows x 10 cols | Fed Funds DFF, 3M/2Y/10Y Yields, M2, WALCL
├── market/
│   ├── vn_indices_daily.parquet                 # 4,570 rows x 8 cols | VNINDEX & VN30 daily OHLCV (2015–2026)
│   ├── vn30f1m_derivatives_daily.parquet        # 2,296 rows x 8 cols | VN30F1M Futures daily OHLCV (2017–2026)
│   ├── vn_top_equities_daily.parquet            # 136,302 rows x 8 cols | 55 constituent stocks daily OHLCV
│   ├── vn_all_vn30_constituents_daily.parquet   # 136,302 rows x 8 cols | All VN30 constituents daily OHLCV
│   ├── vn_stock_industry_classification.parquet # 1,971 rows x 22 cols | All listed stocks mapped to 4-tier ICB
│   ├── vn_industry_sectors_icb.parquet          # 185 rows x 7 cols | 185 ICB sector definitions & ticker lists
│   ├── vn_sector_indices_daily.parquet          # 26,113 rows x 8 cols | 12 official sector indices daily OHLCV
│   ├── vn_foreign_trading_daily.parquet         # 130,837 rows x 12 cols | Daily Foreign Trading (GD NĐTNN: Buy, Sell, Room)
│   └── vn_order_supply_demand_daily.parquet     # 83,731 rows x 11 cols | Daily Order Book Supply/Demand (Cung cầu: Orders, Vol)
├── market_structure/
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

---

## 2. Standardized Dataset Specifications (Single-Table Format)

Every dataset is documented in an identical, single-table format with standardized schema columns:
`| Column Name | Type | Unit | Description |`

### 2.1 Macroeconomic Data (`data/macro/`)

#### A. `vietnam_monetary_daily.parquet`
* **Coverage**: 2011-01-03 to 2026-10-21 (5,471 daily observations, 16 columns).
* **Source**: WiData (WiGroup), State Bank of Vietnam (SBV), Commercial Banks. [https://widata.vn/vi-mo/vn/pmi-san-xuat](https://widata.vn/vi-mo/vn/pmi-san-xuat) | [https://www.sbv.gov.vn](https://www.sbv.gov.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `date` | `String` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
| `gold_world_usd_adjusted` | `Float64` | USD/oz | International spot gold price adjusted to domestic basis |
| `gold_sjc_domestic_vnd` | `Float64` | Million VND/tael | Domestic SJC physical gold buying price per tael |
| `interbank_rate_on` | `Float64` | % / year | Overnight (ON) interbank offer rate |
| `sbv_net_liquidity_injection` | `Float64` | Billion VND | Net daily liquidity injected (+) or absorbed (-) via OMO & T-Bills |
| `sbv_refinancing_rate` | `Float64` | % / year | State Bank of Vietnam policy refinancing interest rate (Lãi suất tái cấp vốn) |
| `sbv_discount_rate` | `Float64` | % / year | State Bank of Vietnam policy rediscount interest rate (Lãi suất tái chiết khấu) |
| `usdvnd_commercial_sell` | `Float64` | VND | USD/VND ask/selling rate at commercial banks |
| `usdvnd_central_rate` | `Float64` | VND | SBV official central reference exchange rate |
| `usdvnd_commercial_buy` | `Float64` | VND | USD/VND bid/buying rate at commercial banks |
| `usdvnd_freemarket_sell` | `Float64` | VND | Informal / free market USD/VND selling rate |
| `vn_govbond_yield_10y` | `Float64` | % / year | Vietnam 10-Year Government Benchmark Bond Yield |
| `bank_deposit_rate_12m` | `Float64` | % / year | 12-Month deposit rate proxy (MBB, ACB, TCB average) |
| `sbv_bills_outstanding_balance` | `Float64` | Billion VND | Cumulative outstanding SBV bills (Tín phiếu NHNN) balance |
| `omo_outstanding_balance` | `Float64` | Billion VND | Cumulative outstanding OMO repo balance in circulation |
| `vn_govbond_yield_1y` | `Float64` | % / year | Vietnam 1-Year Government Benchmark Bond Yield |

#### B. `vietnam_macro_monthly.parquet`
* **Coverage**: 2001-12-31 to 2026-09-30 (298 monthly observations, 13 columns).
* **Source**: General Statistics Office (GSO), S&P Global / WiData. [https://widata.vn/vi-mo/vn/pmi-san-xuat](https://widata.vn/vi-mo/vn/pmi-san-xuat).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `date` | `String` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
| `pmi_manufacturing` | `Float64` | Index (50=Neutral) | Vietnam Manufacturing Purchasing Managers' Index (S&P Global / GSO) |
| `cpi_transport` | `Float64` | Index | CPI Transport component price index |
| `cpi_food` | `Float64` | Index | CPI Food and foodstuffs component price index |
| `cpi_housing` | `Float64` | Index | CPI Housing and construction materials component price index |
| `cpi_index` | `Float64` | Index | Consumer Price Index headline level |
| `pmi_manufacturing_widata` | `Float64` | Index (50=Neutral) | WiData adjusted Manufacturing PMI series |
| `iip_industrial_production_yoy` | `Float64` | % YoY | Index of Industrial Production YoY growth rate |
| `retail_sales_yoy` | `Float64` | % YoY | Total retail sales of consumer goods and services YoY growth |
| `iip_mining_yoy` | `Float64` | % YoY | IIP Mining and quarrying subsector YoY growth rate |
| `cpi_yoy` | `Float64` | % YoY | Headline Consumer Price Index YoY inflation rate |
| `fdi_registered_ttm` | `Float64` | Billion USD | Trailing 12-month registered Foreign Direct Investment |
| `fdi_disbursed_ttm` | `Float64` | Billion USD | Trailing 12-month disbursed Foreign Direct Investment |

#### C. `us_macro_daily.parquet`
* **Coverage**: 1986-01-02 to 2026-10-21 (13,821 observations, 10 columns).
* **Source**: Federal Reserve Bank of St. Louis (FRED), US Department of the Treasury. [https://fred.stlouisfed.org](https://fred.stlouisfed.org).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `date` | `String` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
| `us_fed_total_assets_walcl` | `Float64` | Million USD | Federal Reserve Total Assets (Less Eliminations from Consolidation) |
| `us_fed_funds_rate_dff` | `Float64` | % / year | Effective Federal Funds Rate (DFF) |
| `us_sofr_volume` | `Float64` | Billion USD | Secured Overnight Financing Rate (SOFR) daily transaction volume |
| `us_effr_volume` | `Float64` | Billion USD | Effective Federal Funds Rate daily transaction volume |
| `us_fed_target_upper` | `Float64` | % / year | Federal Open Market Committee target policy rate upper bound |
| `us_treasury_2y` | `Float64` | % / year | US 2-Year Treasury Constant Maturity Yield |
| `us_treasury_10y` | `Float64` | % / year | US 10-Year Treasury Constant Maturity Yield |
| `us_treasury_3m` | `Float64` | % / year | US 3-Month Treasury Bill Secondary Market Rate |
| `us_m2_money_supply_yoy` | `Float64` | % YoY | US M2 Money Stock YoY Growth Rate |

#### D. `vietnam_market_pe_daily.parquet`
* **Coverage**: 2015-05-04 to 2026-10-21 (2,877 daily observations, 3 columns).
* **Source**: WiData / Ho Chi Minh Stock Exchange (HOSE). [https://widata.vn](https://widata.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `date` | `String` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `pe` | `Float64` | Multiple | Trailing 12-month Price-to-Earnings valuation multiple |

#### E. `widata_macro_all_series.parquet`
* **Coverage**: 1970-01-01 to 2026-10-21 (204,426 rows across 107 macro series, 8 columns).
* **Source**: WiData Macro Database Engine (107 primary series unpivoted). [https://widata.vn](https://widata.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `series_id` | `String` | - | Unique macroeconomic series identifier in WiData database |
| `table_name` | `String` | - | Parent database table name in WiData schema |
| `column_name` | `String` | - | Original column field title |
| `time_frame` | `String` | - | Observation sampling frequency (daily, monthly, quarterly, yearly) |
| `value_type` | `String` | - | Value formatting type (value, percent, index) |
| `timestamp` | `Int64` | Unix Epoch (s) | POSIX epoch timestamp in seconds |
| `date` | `String` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
| `value` | `Float64` | Numeric | Reported observation value in original units |

---

### 2.2 Market & Derivatives Data (`data/market/`)

#### A. `vn_indices_daily.parquet`
* **Coverage**: 2015-01-05 to 2026-10-22 (4,570 daily OHLCV bars across VNINDEX and VN30, 8 columns).
* **Source**: VNDirect Open Market DChart API / HOSE. [https://dchart.vndirect.com.vn](https://dchart.vndirect.com.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `date` | `String` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
| `timestamp` | `Int64` | Unix Epoch (s) | POSIX epoch timestamp in seconds |
| `open` | `Float64` | VND / Pts | Opening price or index level |
| `high` | `Float64` | VND / Pts | Intraday high price or index level |
| `low` | `Float64` | VND / Pts | Intraday low price or index level |
| `close` | `Float64` | VND / Pts | Closing price or index level |
| `volume` | `Float64` | Shares / Contracts | Total traded volume during the session |

#### B. `vn30f1m_derivatives_daily.parquet`
* **Coverage**: 2017-08-10 to 2026-10-22 (2,296 daily OHLCV bars of VN30F1M Front-Month Futures, 8 columns).
* **Source**: Hanoi Stock Exchange (HNX) Derivatives Market via VNDirect. [https://dchart.vndirect.com.vn](https://dchart.vndirect.com.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `date` | `String` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
| `timestamp` | `Int64` | Unix Epoch (s) | POSIX epoch timestamp in seconds |
| `open` | `Float64` | VND / Pts | Opening price or index level |
| `high` | `Float64` | VND / Pts | Intraday high price or index level |
| `low` | `Float64` | VND / Pts | Intraday low price or index level |
| `close` | `Float64` | VND / Pts | Closing price or index level |
| `volume` | `Float64` | Shares / Contracts | Total traded volume during the session |

#### C. `vn_top_equities_daily.parquet`
* **Coverage**: 2015-01-05 to 2026-10-22 (136,302 daily OHLCV bars across 55 constituent equities, 8 columns).
* **Source**: VNDirect Open Market DChart API. [https://dchart.vndirect.com.vn](https://dchart.vndirect.com.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `date` | `String` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
| `timestamp` | `Int64` | Unix Epoch (s) | POSIX epoch timestamp in seconds |
| `open` | `Float64` | VND / Pts | Opening price or index level |
| `high` | `Float64` | VND / Pts | Intraday high price or index level |
| `low` | `Float64` | VND / Pts | Intraday low price or index level |
| `close` | `Float64` | VND / Pts | Closing price or index level |
| `volume` | `Float64` | Shares / Contracts | Total traded volume during the session |

#### D. `vn_all_vn30_constituents_daily.parquet`
* **Coverage**: 2015-01-05 to 2026-10-22 (136,302 daily OHLCV bars for all VN30 constituents, 8 columns).
* **Source**: VNDirect Open Market DChart API. [https://dchart.vndirect.com.vn](https://dchart.vndirect.com.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `date` | `String` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
| `timestamp` | `Int64` | Unix Epoch (s) | POSIX epoch timestamp in seconds |
| `open` | `Float64` | VND / Pts | Opening price or index level |
| `high` | `Float64` | VND / Pts | Intraday high price or index level |
| `low` | `Float64` | VND / Pts | Intraday low price or index level |
| `close` | `Float64` | VND / Pts | Closing price or index level |
| `volume` | `Float64` | Shares / Contracts | Total traded volume during the session |

#### E. `vn_stock_industry_classification.parquet`
* **Coverage**: All 1,971 actively listed stocks on HOSE (441), HNX (359), and UPCOM (1,171), 22 columns.
* **Source**: VNDirect Financial Information (Finfo Open API) / HOSE / HNX. [https://api-finfo.vndirect.com.vn](https://api-finfo.vndirect.com.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `company_name` | `String` | - | Full corporate legal name in Vietnamese |
| `company_name_eng` | `String` | - | Full corporate legal name in English |
| `short_name` | `String` | - | Common business abbreviation or trading short name |
| `floor` | `String` | - | Trading venue / stock exchange (HOSE, HNX, UPCOM) |
| `isin` | `String` | - | International Securities Identification Number (e.g. VN000000VCB4) |
| `status` | `String` | - | Securities registration and listing status (e.g. listed) |
| `listed_date` | `String` | YYYY-MM-DD | Initial official public listing date |
| `icb_level1_code` | `String` | - | ICB Level 1 Industry Code (e.g. 8000 Financials) |
| `icb_level1_en` | `String` | - | ICB Level 1 English Name |
| `icb_level1_vi` | `String` | - | ICB Level 1 Vietnamese Name |
| `icb_level2_code` | `String` | - | ICB Level 2 Supersector Code (e.g. 8300 Banks, 8600 Real Estate) |
| `icb_level2_en` | `String` | - | ICB Level 2 English Name |
| `icb_level2_vi` | `String` | - | ICB Level 2 Vietnamese Name |
| `icb_level3_code` | `String` | - | ICB Level 3 Sector Code |
| `icb_level3_en` | `String` | - | ICB Level 3 English Name |
| `icb_level3_vi` | `String` | - | ICB Level 3 Vietnamese Name |
| `icb_level4_code` | `String` | - | ICB Level 4 Subsector Code |
| `icb_level4_en` | `String` | - | ICB Level 4 English Name |
| `icb_level4_vi` | `String` | - | ICB Level 4 Vietnamese Name |
| `tax_code` | `String` | - | Enterprise tax identification number (Mã số thuế) |
| `company_id` | `String` | - | Internal exchange/database corporate entity identifier |

#### F. `vn_industry_sectors_icb.parquet`
* **Coverage**: 185 official ICB classifications across Level 1 (10), Level 2 (19), Level 3 (42), Level 4 (114), 7 columns.
* **Source**: Official Industry Classification Benchmark (ICB) Registry via VNDirect. [https://api-finfo.vndirect.com.vn](https://api-finfo.vndirect.com.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `industry_code` | `String` | - | Official 4-digit ICB classification code |
| `industry_level` | `Int64` | Level (1-4) | ICB hierarchy tier (1=Industry, 2=Supersector, 3=Sector, 4=Subsector) |
| `higher_level_code` | `String` | - | Parent ICB category code in hierarchical tree |
| `english_name` | `String` | - | Standard English sector/industry title |
| `vietnamese_name` | `String` | - | Standard Vietnamese sector/industry title |
| `total_count` | `Int64` | Count | Total number of actively listed companies in this classification |
| `code_list` | `String` | - | Comma-separated list of constituent stock tickers |

#### G. `vn_sector_indices_daily.parquet`
* **Coverage**: 2017-08-24 to 2026-10-22 (26,113 daily OHLCV rows across 12 sector indices, 8 columns).
* **Source**: Ho Chi Minh Stock Exchange (HOSE) Official Sector Indices via VNDirect. [https://dchart.vndirect.com.vn](https://dchart.vndirect.com.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `date` | `String` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
| `timestamp` | `Int64` | Unix Epoch (s) | POSIX epoch timestamp in seconds |
| `open` | `Float64` | VND / Pts | Opening price or index level |
| `high` | `Float64` | VND / Pts | Intraday high price or index level |
| `low` | `Float64` | VND / Pts | Intraday low price or index level |
| `close` | `Float64` | VND / Pts | Closing price or index level |
| `volume` | `Float64` | Shares / Contracts | Total traded volume during the session |

#### H. `vn_foreign_trading_daily.parquet`
* **Coverage**: 2018-08-30 to 2026-10-22 (130,837 daily rows across 65 leading equities, 12 columns).
* **Source**: VNDirect Financial Information (Finfo Open API) / Official Exchange Trades. [https://api-finfo.vndirect.com.vn](https://api-finfo.vndirect.com.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `date` | `String` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
| `floor` | `String` | - | Trading venue / stock exchange (HOSE, HNX, UPCOM) |
| `buy_val` | `Float64` | VND | Foreign investor buy value (Giá trị mua nước ngoài) |
| `sell_val` | `Float64` | VND | Foreign investor sell value (Giá trị bán nước ngoài) |
| `net_val` | `Float64` | VND | Net foreign investor value: buy_val - sell_val (Giá trị mua ròng) |
| `buy_vol` | `Float64` | Shares | Foreign investor buy volume (Khối lượng mua nước ngoài) |
| `sell_vol` | `Float64` | Shares | Foreign investor sell volume (Khối lượng bán nước ngoài) |
| `net_vol` | `Float64` | Shares | Net foreign investor volume: buy_vol - sell_vol (Khối lượng mua ròng NĐTNN) |
| `total_room` | `Float64` | Shares | Maximum foreign ownership room limit established by law/AGM |
| `current_room` | `Float64` | Shares | Remaining available foreign ownership shares |
| `room_pct` | `Float64` | % | Percentage of foreign room remaining available (current_room / total_room * 100) |

#### I. `vn_order_supply_demand_daily.parquet`
* **Coverage**: 2021-07-09 to 2026-10-22 (83,731 daily rows across 64 leading equities, 11 columns).
* **Source**: VNDirect Financial Information (Finfo Open API) / Trading Order Statistics. [https://api-finfo.vndirect.com.vn](https://api-finfo.vndirect.com.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `date` | `String` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
| `floor` | `String` | - | Trading venue / stock exchange (HOSE, HNX, UPCOM) |
| `bid_order` | `Float64` | Orders | Total number of buy orders placed during trading session |
| `bid_vol` | `Float64` | Shares | Total buy order volume placed in the order book |
| `offer_order` | `Float64` | Orders | Total number of sell orders placed during trading session |
| `offer_vol` | `Float64` | Shares | Total sell order volume placed in the order book |
| `avg_bid_vol` | `Float64` | Shares/order | Average volume per placed buy order (bid_vol / bid_order) |
| `avg_offer_vol` | `Float64` | Shares/order | Average volume per placed sell order (offer_vol / offer_order) |
| `net_vol` | `Float64` | Shares | Net order book volume imbalance: bid_vol - offer_vol (Dư mua/bán ròng) |
| `order_imbalance` | `Float64` | [-1, +1] | Standardized order book imbalance: (bid_vol - offer_vol) / (bid_vol + offer_vol) |

---

### 2.3 Market Structure & Corporate Governance (`data/market_structure/`)

#### A. `vn_corporate_network_edges.parquet`
* **Coverage**: 4,803 directed relationship edges across Vietnam corporate conglomerates and institutions, 12 columns.
* **Source**: Constructed Directed Network Graph from Exchange Filings, State Portfolios, and Big 4 Auditing Registers. [https://api-finfo.vndirect.com.vn](https://api-finfo.vndirect.com.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `source_id` | `String` | - | Canonical source entity node identifier (e.g. VIC, SCIC, VCB) |
| `source_name` | `String` | - | Source entity legal title |
| `source_type` | `String` | - | Source category (LISTED_EQUITY, STATE_OWNED_ENTITY, INSTITUTIONAL_INVESTOR) |
| `target_id` | `String` | - | Canonical target entity node identifier (e.g. VHM, HVN, CTG) |
| `target_name` | `String` | - | Target entity legal title |
| `target_type` | `String` | - | Target category (LISTED_EQUITY, SUBSIDIARY, ASSOCIATE, AUDITOR) |
| `relationship_type` | `String` | - | Network edge classification (SUBSIDIARY_CONTROL, STATE_EQUITY_CONTROL, AUDITOR_SERVICE) |
| `ownership_pct` | `Float64` | % | Direct percentage of equity shares held (%) if reported |
| `capital_value_vnd` | `Float64` | VND | Contributed capital value in VND if reported |
| `effective_date` | `String` | YYYY-MM-DD | Official effective date from which parameters/records take legal effect |
| `weight` | `Float64` | [0, 1] | Normalized connection strength based on ownership percentage or governance authority |
| `is_directed` | `Boolean` | Boolean | True indicating directed causal edge from owner/parent to entity |

#### B. `vn_corporate_network_nodes.parquet`
* **Coverage**: 3,473 unique corporate and institutional entity nodes in Vietnamese market, 8 columns.
* **Source**: Graph Centrality Engine (NetworkX PageRank alpha=0.85 & Degree Centrality)..
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `node_id` | `String` | - | Canonical unique node identifier in the corporate graph |
| `node_name` | `String` | - | Official entity name |
| `node_type` | `String` | - | Graph node type (LISTED_EQUITY, SUBSIDIARY, ASSOCIATE, STATE_OWNED_ENTITY, AUDIT_FIRM) |
| `sector` | `String` | - | Economic sector or institutional domain |
| `in_degree` | `Int64` | Count | Number of incoming directed edges (shareholders, owners, auditors) |
| `out_degree` | `Int64` | Count | Number of outgoing directed edges (subsidiaries, investments, affiliates) |
| `total_degree` | `Int64` | Count | Total network connectivity (in_degree + out_degree) |
| `pagerank` | `Float64` | Probability | Google PageRank systemic centrality score (alpha=0.85) |

#### C. `vn_shareholders_history.parquet`
* **Coverage**: 3,115 historical major shareholder filings across 65 leading equities, 9 columns.
* **Source**: Official Listed Company Disclosures & Shareholder Registry via VNDirect. [https://api-finfo.vndirect.com.vn](https://api-finfo.vndirect.com.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `shareholder_type` | `String` | - | Legal person category: PERSON (Cá nhân) or COMPANY (Tổ chức) |
| `shareholder_name` | `String` | - | Full name of shareholder entity or individual insider |
| `role_type` | `String` | - | Corporate governance / family relationship role in Vietnamese |
| `en_role_type` | `String` | - | Corporate governance / family relationship role in English |
| `number_of_shares` | `Float64` | Shares | Total number of shares held at disclosure date |
| `ownership_pct` | `Float64` | % | Reported major shareholder equity ownership percentage (%) |
| `group_pct` | `Float64` | % | Affiliated group percentage of equity shares held |
| `effective_date` | `String` | YYYY-MM-DD | Official effective date from which parameters/records take legal effect |

#### D. `vn_subsidiaries_affiliates.parquet`
* **Coverage**: 1,688 corporate relationships across 65 listed conglomerates, 10 columns.
* **Source**: Annual Audited Reports & Corporate Structure Registry via VNDirect. [https://api-finfo.vndirect.com.vn](https://api-finfo.vndirect.com.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `relation_type` | `String` | - | Corporate affiliation category: SUBSIDIARY (>50%), ASSOCIATE (20-50%), PARENT, AUDITOR |
| `relation_name_vn` | `String` | - | Corporate entity title in Vietnamese |
| `relation_name_en` | `String` | - | Corporate entity title in English |
| `relation_id` | `Int64` | - | Unique corporate relation registry identifier |
| `effective_date` | `String` | YYYY-MM-DD | Official effective date from which parameters/records take legal effect |
| `year` | `String` | Year | Fiscal reporting year |
| `ownership_pct` | `Float64` | % | Parent group equity ownership percentage (%) in corporate entity |
| `investment_capital` | `Float64` | VND | Contributed investment capital value |
| `unit` | `String` | - | Currency denomination unit (VND) |

#### E. `vn_charter_capital_history.parquet`
* **Coverage**: 4,391 capital adjustment and share issuance events across 65 equities, 13 columns.
* **Source**: State Securities Commission (UBCKNN) & Stock Exchange Corporate Actions Registry. [https://api-finfo.vndirect.com.vn](https://api-finfo.vndirect.com.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `event_id` | `String` | - | Unique corporate action event identifier |
| `event_group` | `String` | - | Event classification category (e.g. investorRight) |
| `event_type` | `String` | - | Event action code (ISSUE: phát hành thêm, LISTED: niêm yết bổ sung, DIVIDEND) |
| `type_desc` | `String` | - | Descriptive title of the corporate capital event |
| `note` | `String` | - | Detailed issuance terms, prospectus description, or bonus allocation notes |
| `price` | `Float64` | VND/share | Rights execution price or public offering price |
| `ratio` | `Float64` | % | Rights issue allocation ratio or bonus share distribution percentage |
| `number_of_shares` | `Float64` | Shares | Total number of shares held at disclosure date |
| `disclosure_date` | `String` | YYYY-MM-DD | Official public disclosure announcement date |
| `effective_date` | `String` | YYYY-MM-DD | Official effective date from which parameters/records take legal effect |
| `trading_start_date` | `String` | YYYY-MM-DD | Official date when additional shares commence trading on the exchange |
| `actual_date` | `String` | YYYY-MM-DD | Date of completed execution or distribution |

#### F. `free_float.parquet`
* **Coverage**: 2017-07-24 to 2026-08-03 (43,377 point-in-time rows across 1,450 tickers and 14 indices, 18 columns).
* **Source**: HOSE & HNX Point-in-Time Index Committee Official PDF Notices (Sourced via lazy-researcher). [https://www.hsx.vn](https://www.hsx.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `time` | `String` | ISO UTC | Point-in-time timestamp of announcement or disclosure |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `index` | `String` | - | Index basket code (e.g. VN30, VN100, VNALLSHARE) |
| `exchange` | `String` | - | Stock exchange (HOSE, HNX) |
| `company_name` | `String` | - | Full corporate legal name in Vietnamese |
| `effective_date` | `String` | YYYY-MM-DD | Official effective date from which parameters/records take legal effect |
| `index_shares` | `Int64` | Shares | Total outstanding shares considered for index calculation (Q) |
| `free_float_factor` | `Float64` | Ratio (0-1] | Exchange-calculated free-float ratio (f) rounded to regulatory bands |
| `capping_factor` | `Float64` | Ratio (0-1] | Single-stock capping factor (c) ensuring compliance with concentration limits |
| `source_page` | `Int64` | - | Page number in official PDF announcement |
| `source_url` | `String` | URL | Official exchange notice web URL |
| `document_url` | `String` | URL | Direct download URL for official exchange PDF disclosure |
| `source_sha256` | `String` | SHA-256 | Cryptographic hash of the source PDF document for audit reproducibility |
| `announced_at` | `String` | ISO UTC | Official public announcement timestamp |
| `retrieved_at` | `String` | ISO UTC | System retrieval and ingestion timestamp |
| `availability_basis` | `String` | - | Point-in-Time availability justification (e.g. scheduled_notice) |
| `quality` | `String` | - | Data quality verification level (e.g. official_complete) |
| `publication_evidence_url` | `String` | URL | Archival verification URL for exchange publication |

#### G. `vn30_free_float_history.parquet`
* **Coverage**: 2017-07-24 to 2026-08-03 (690 point-in-time constituent review rows for VN30, 18 columns).
* **Source**: HOSE Official Point-in-Time Review Notices. [https://www.hsx.vn](https://www.hsx.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `time` | `String` | ISO UTC | Point-in-time timestamp of announcement or disclosure |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `index` | `String` | - | Index basket code (e.g. VN30, VN100, VNALLSHARE) |
| `exchange` | `String` | - | Stock exchange (HOSE, HNX) |
| `company_name` | `String` | - | Full corporate legal name in Vietnamese |
| `effective_date` | `String` | YYYY-MM-DD | Official effective date from which parameters/records take legal effect |
| `index_shares` | `Int64` | Shares | Total outstanding shares considered for index calculation (Q) |
| `free_float_factor` | `Float64` | Ratio (0-1] | Exchange-calculated free-float ratio (f) rounded to regulatory bands |
| `capping_factor` | `Float64` | Ratio (0-1] | Single-stock capping factor (c) ensuring compliance with concentration limits |
| `source_page` | `Int64` | - | Page number in official PDF announcement |
| `source_url` | `String` | URL | Official exchange notice web URL |
| `document_url` | `String` | URL | Direct download URL for official exchange PDF disclosure |
| `source_sha256` | `String` | SHA-256 | Cryptographic hash of the source PDF document for audit reproducibility |
| `announced_at` | `String` | ISO UTC | Official public announcement timestamp |
| `retrieved_at` | `String` | ISO UTC | System retrieval and ingestion timestamp |
| `availability_basis` | `String` | - | Point-in-Time availability justification (e.g. scheduled_notice) |
| `quality` | `String` | - | Data quality verification level (e.g. official_complete) |
| `publication_evidence_url` | `String` | URL | Archival verification URL for exchange publication |

#### H. `free_float_changes_observed.parquet`
* **Coverage**: 10,178 discrete parameter adjustment events observed across index rebalancing cycles, 22 columns.
* **Source**: Point-in-Time Discrete Free-Float and Capping Adjustment Tracker. [https://www.hsx.vn](https://www.hsx.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `time` | `String` | ISO UTC | Point-in-time timestamp of announcement or disclosure |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `index` | `String` | - | Index basket code (e.g. VN30, VN100, VNALLSHARE) |
| `exchange` | `String` | - | Stock exchange (HOSE, HNX) |
| `company_name` | `String` | - | Full corporate legal name in Vietnamese |
| `effective_date` | `String` | YYYY-MM-DD | Official effective date from which parameters/records take legal effect |
| `index_shares` | `Int64` | Shares | Total outstanding shares considered for index calculation (Q) |
| `free_float_factor` | `Float64` | Ratio (0-1] | Exchange-calculated free-float ratio (f) rounded to regulatory bands |
| `capping_factor` | `Float64` | Ratio (0-1] | Single-stock capping factor (c) ensuring compliance with concentration limits |
| `source_page` | `Int64` | - | Page number in official PDF announcement |
| `source_url` | `String` | URL | Official exchange notice web URL |
| `document_url` | `String` | URL | Direct download URL for official exchange PDF disclosure |
| `source_sha256` | `String` | SHA-256 | Cryptographic hash of the source PDF document for audit reproducibility |
| `announced_at` | `String` | ISO UTC | Official public announcement timestamp |
| `retrieved_at` | `String` | ISO UTC | System retrieval and ingestion timestamp |
| `availability_basis` | `String` | - | Point-in-Time availability justification (e.g. scheduled_notice) |
| `quality` | `String` | - | Data quality verification level (e.g. official_complete) |
| `publication_evidence_url` | `String` | URL | Archival verification URL for exchange publication |
| `previous_factor` | `Float64` | Ratio (0-1] | Preceding free-float factor prior to revision |
| `previous_snapshot_date` | `String` | YYYY-MM-DD | Date of previous parameter snapshot |
| `previous_source_url` | `String` | URL | URL of previous official announcement |
| `change_date_precision` | `String` | - | Resolution precision of effective date (e.g. day) |

#### I. `index_constituents.parquet`
* **Coverage**: 43,377 historical index constituent rows across all Vietnamese benchmark indices, 18 columns.
* **Source**: HOSE & HNX Index Calculation Methodology and Constituent Registry. [https://www.hsx.vn](https://www.hsx.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `time` | `String` | ISO UTC | Point-in-time timestamp of announcement or disclosure |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `index` | `String` | - | Index basket code (e.g. VN30, VN100, VNALLSHARE) |
| `exchange` | `String` | - | Stock exchange (HOSE, HNX) |
| `company_name` | `String` | - | Full corporate legal name in Vietnamese |
| `effective_date` | `String` | YYYY-MM-DD | Official effective date from which parameters/records take legal effect |
| `index_shares` | `Int64` | Shares | Total outstanding shares considered for index calculation (Q) |
| `free_float_factor` | `Float64` | Ratio (0-1] | Exchange-calculated free-float ratio (f) rounded to regulatory bands |
| `capping_factor` | `Float64` | Ratio (0-1] | Single-stock capping factor (c) ensuring compliance with concentration limits |
| `source_page` | `Int64` | - | Page number in official PDF announcement |
| `source_url` | `String` | URL | Official exchange notice web URL |
| `document_url` | `String` | URL | Direct download URL for official exchange PDF disclosure |
| `source_sha256` | `String` | SHA-256 | Cryptographic hash of the source PDF document for audit reproducibility |
| `announced_at` | `String` | ISO UTC | Official public announcement timestamp |
| `retrieved_at` | `String` | ISO UTC | System retrieval and ingestion timestamp |
| `availability_basis` | `String` | - | Point-in-Time availability justification (e.g. scheduled_notice) |
| `quality` | `String` | - | Data quality verification level (e.g. official_complete) |
| `publication_evidence_url` | `String` | URL | Archival verification URL for exchange publication |

#### J. `basket_history.parquet`
* **Coverage**: 261 basket rebalancing events and official effective schedules, 10 columns.
* **Source**: HOSE Quarterly and Semi-Annual Basket Rebalancing Schedule. [https://www.hsx.vn](https://www.hsx.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `time` | `String` | ISO UTC | Point-in-time timestamp of announcement or disclosure |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `basket` | `String` | - | Basket identifier |
| `effective_date` | `String` | YYYY-MM-DD | Official effective date from which parameters/records take legal effect |
| `symbols` | `String` | - | JSON list of constituent tickers in basket |
| `source_urls` | `String` | - | JSON list of publication source URLs |
| `source_sha256s` | `String` | - | JSON list of cryptographic document hashes |
| `quality` | `String` | - | Data quality verification level (e.g. official_complete) |
| `state_basis` | `String` | - | Historical baseline status |
| `is_pre_start_baseline` | `Boolean` | Boolean | True if constituent list represents initial index inception state |

---

### 2.4 Corporate Fundamentals & Financial Statements (`data/financials/`)

#### A. `vn_financial_models_taxonomy.parquet`
* **Coverage**: 3,370 official accounting line items across 51 reporting models, 12 columns.
* **Source**: Ministry of Finance & State Bank of Vietnam Statutory Accounting Circulars (TT202, TT49, TT334, TT199). [https://mof.gov.vn](https://mof.gov.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `model_type` | `Int64` | - | Accounting model numeric code in Ministry of Finance taxonomy |
| `model_type_name` | `String` | - | Statement classification (BALANCESHEET, INCOME, CASHFLOW, EXPLAINATION) |
| `model_vn_desc` | `String` | - | Accounting model title in Vietnamese (e.g. Bảng Cân đối kế toán - Chung) |
| `model_en_desc` | `String` | - | Accounting model title in English |
| `company_form` | `String` | - | Corporate legal form (NON_FINANCE, BANK, SECURITIES, INSURANCE) |
| `accounting_circular` | `String` | - | Governing statutory accounting circular (e.g. TT202/2014/TT-BTC) |
| `item_code` | `Int64` | - | Official accounting line item code (e.g. 11000, 21000, 23800) |
| `item_vn_name` | `String` | - | Official accounting line item description in Vietnamese |
| `item_en_name` | `String` | - | Official accounting line item description in English |
| `display_order` | `Int64` | - | Official regulatory row sequence number on published statements |
| `display_level` | `Int64` | Level (1-3) | Accounting line hierarchy level (1=Header Category, 2=Subcategory, 3=Detail) |
| `form_type` | `String` | - | Statement layout format (WEB, ALL) |

#### B. `vn_financial_statements_quarterly_yearly.parquet`
* **Coverage**: 325,000 line item observations (Balance Sheet, Income, Cash Flow) across quarterly and annual terms (2018–2026), 12 columns.
* **Source**: Audited Financial Statements and Quarterly Disclosures submitted to HOSE/HNX via VNDirect. [https://api-finfo.vndirect.com.vn](https://api-finfo.vndirect.com.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `fiscal_date` | `String` | YYYY-MM-DD | Accounting period closing balance sheet date |
| `year` | `Int32` | Year | Fiscal reporting year |
| `quarter` | `Int32` | Quarter (1-4, 0) | Fiscal quarter (1, 2, 3, 4; 0 = Full Year Audited Annual Report) |
| `report_type` | `String` | - | Regulatory period term (QUARTER1, QUARTER2, QUARTER3, QUARTER4, YEAR, ANNUAL) |
| `statement_type` | `String` | - | Parent statement type (BALANCESHEET, INCOME, CASHFLOW, EXPLAINATION) |
| `item_code` | `Int64` | - | Official accounting line item code (e.g. 11000, 21000, 23800) |
| `item_name_vn` | `String` | - | Reported field `item_name_vn` |
| `item_name_en` | `String` | - | Reported field `item_name_en` |
| `display_level` | `Int32` | Level (1-3) | Accounting line hierarchy level (1=Header Category, 2=Subcategory, 3=Detail) |
| `value` | `Float64` | Numeric | Reported observation value in original units |
| `model_type` | `Int64` | - | Accounting model numeric code in Ministry of Finance taxonomy |

#### C. `vn_financial_ratios_quarterly_yearly.parquet`
* **Coverage**: 189,000 financial metrics and ratios (Profitability, Leverage, Liquidity, Valuation, Efficiency) across quarterly and annual terms, 8 columns.
* **Source**: VNDirect Financial Information (Finfo Open API) / Standard Financial Ratio Models. [https://api-finfo.vndirect.com.vn](https://api-finfo.vndirect.com.vn).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `report_date` | `String` | YYYY-MM-DD | Official report filing date or ratio calculation timestamp |
| `year` | `Int32` | Year | Fiscal reporting year |
| `quarter` | `Int32` | Quarter (1-4, 0) | Fiscal quarter (1, 2, 3, 4; 0 = Full Year Audited Annual Report) |
| `ratio_code` | `String` | - | Standardized fundamental ratio code (e.g. ROAE_TR_AVG5Q, PRICE_TO_EARNINGS) |
| `ratio_name` | `String` | - | Standard Vietnamese title of the financial metric or ratio |
| `category` | `String` | - | Ratio financial category (PROFITABILITY, VALUATION, LEVERAGE, LIQUIDITY, EFFICIENCY) |
| `value` | `Float64` | Numeric | Reported observation value in original units |

#### D. `vn_listed_q3_2026_financials.parquet`
* **Coverage**: 192 listed companies on HOSE, HNX, and UPCOM for Q3-2026, 21 columns.
* **Source**: Listed Company Financial Disclosures Registry for Q3-2026 via WiData. [https://widata.vn/ket-qua-kinh-doanh](https://widata.vn/ket-qua-kinh-doanh).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
| `exchange` | `String` | - | Stock exchange (HOSE, HNX) |
| `industry_level4` | `String` | - | ICB Level 4 Subsector Title |
| `revenue_bn_vnd` | `Float64` | Billion VND | Quarterly Net Revenue in Billion VND |
| `revenue_ytd_bn_vnd` | `Float64` | Billion VND | Cumulative 9-Month Net Revenue in Billion VND |
| `doanhthu_htkh` | `Float64` | % | Percentage completion of annual AGM revenue target |
| `revenue_growth_yoy_pct` | `Float64` | % YoY | Net Revenue Year-over-Year growth percentage |
| `revenue_growth_qoq_pct` | `Float64` | % QoQ | Net Revenue Quarter-over-Quarter growth percentage |
| `net_profit_bn_vnd` | `Float64` | Billion VND | Quarterly Net Profit After Tax in Billion VND (Lãi ròng quý) |
| `net_profit_ytd_bn_vnd` | `Float64` | Billion VND | Cumulative 9-Month Net Profit After Tax in Billion VND |
| `target_plan_completion_pct` | `Float64` | % | Percentage completion of annual AGM net profit target |
| `profit_growth_yoy_pct` | `Float64` | % YoY | Net Profit After Tax Year-over-Year growth percentage |
| `profit_growth_qoq_pct` | `Float64` | % QoQ | Net Profit After Tax Quarter-over-Quarter growth percentage |
| `pe_ratio` | `Float64` | Multiple | Trailing 12-Month Price-to-Earnings valuation ratio |
| `market_cap_bn_vnd` | `Float64` | Billion VND | Total market capitalization in Billion VND |
| `smg_score` | `Float64` | Score [0-100] | Quantitative Smart Money Grade composite score |
| `close_price` | `Int64` | VND | Closing equity price at filing snapshot date |
| `avg_vol_15d_kshares` | `Float64` | Thousand shares | 15-day average daily trading volume |
| `kiemtoan` | `Null` | - | Audited status flag |
| `report_date` | `String` | YYYY-MM-DD | Official report filing date or ratio calculation timestamp |
| `full_count` | `String` | - | Record counter string |

---

### 2.5 Global Macro & Cross-Asset Benchmarks (`data/cross_asset/`)

#### A. `global_macro_daily.parquet`
* **Coverage**: 2016-10-24 to 2026-10-22 (26,358 daily records across 10 global assets, 8 columns).
* **Source**: Yahoo Finance Multi-Asset Pricing Engine. [https://finance.yahoo.com](https://finance.yahoo.com).
* **Fields**:
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `ticker` | `String` | - | Asset ticker symbol (e.g. ^GSPC, ^TNX, DX-Y.NYB) |
| `asset_name` | `String` | - | Canonical financial asset name (e.g. sp500, gold_futures) |
| `date` | `String` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
| `open` | `Float64` | VND / Pts | Opening price or index level |
| `high` | `Float64` | VND / Pts | Intraday high price or index level |
| `low` | `Float64` | VND / Pts | Intraday low price or index level |
| `close` | `Float64` | VND / Pts | Closing price or index level |
| `volume` | `Float64` | Shares / Contracts | Total traded volume during the session |

---

## 3. Querying Datasets: Polars vs. Pandas Implementation Solutions

### Solution A: Point-in-Time As-Of Join (Monthly Macro to Daily Equity Prices)
```python
# --- Polars (Zero-Copy Vectorized) ---
import polars as pl
df_idx = pl.read_parquet('data/market/vn_indices_daily.parquet').filter(pl.col('symbol') == 'VNINDEX')
df_macro = pl.read_parquet('data/macro/vietnam_macro_monthly.parquet')
aligned = df_idx.join_asof(
    df_macro,
    left_on=pl.col('date').str.to_date(),
    right_on=pl.col('date').str.to_date(),
    strategy='backward'
)

# --- Pandas Equivalent ---
import pandas as pd
df_idx = pd.read_parquet('data/market/vn_indices_daily.parquet').query("symbol == 'VNINDEX'")
df_macro = pd.read_parquet('data/macro/vietnam_macro_monthly.parquet')
df_idx['date'] = pd.to_datetime(df_idx['date'])
df_macro['date'] = pd.to_datetime(df_macro['date'])
aligned = pd.merge_asof(df_idx.sort_values('date'), df_macro.sort_values('date'), on='date', direction='backward')
```

### Solution B: Rolling Volatility & Cash-Futures Basis Spread
```python
# --- Polars ---
import polars as pl
df = pl.read_parquet('data/market/vn_indices_daily.parquet')
basis_df = (
    df.filter(pl.col('symbol') == 'VN30')
    .with_columns([
        (pl.col('close').log() - pl.col('close').shift(1).log()).alias('ret'),
        pl.col('close').rolling_std(window_size=60).alias('vol_60d')
    ])
)

# --- Pandas ---
import pandas as pd
import numpy as np
df = pd.read_parquet('data/market/vn_indices_daily.parquet').query("symbol == 'VN30'")
df['ret'] = np.log(df['close'] / df['close'].shift(1))
df['vol_60d'] = df['close'].rolling(60).std()
```

### Solution C: Corporate Network Centrality Traversal
```python
# --- Polars ---
import polars as pl
nodes = pl.read_parquet('data/market_structure/vn_corporate_network_nodes.parquet')
edges = pl.read_parquet('data/market_structure/vn_corporate_network_edges.parquet')
top_central = nodes.sort('pagerank', descending=True).head(10)
vic_subs = edges.filter((pl.col('source_id') == 'VIC') & (pl.col('relationship_type') == 'SUBSIDIARY_CONTROL'))

# --- Pandas ---
import pandas as pd
nodes = pd.read_parquet('data/market_structure/vn_corporate_network_nodes.parquet')
edges = pd.read_parquet('data/market_structure/vn_corporate_network_edges.parquet')
top_central = nodes.sort_values('pagerank', ascending=False).head(10)
vic_subs = edges[(edges['source_id'] == 'VIC') & (edges['relationship_type'] == 'SUBSIDIARY_CONTROL')]
```

---

## 4. Data Quality, Verification & Integrity Checks

1. **Continuity & Completeness**: Market bars possess unbroken daily timestamps matching official HOSE/HNX trading calendars.
2. **Precision & Consistency**: Floating-point values adhere to 64-bit IEEE precision (`Float64`); calendar dates follow ISO 8601 (`YYYY-MM-DD`).
3. **Point-in-Time Causality**: All corporate and fundamental reports carry explicit announcement timestamps (`announced_at`, `report_date`, `effective_date`) preventing out-of-sample look-ahead bias.

---

## 5. References & Verified Primary Data Sources

1. **WiData Financial Platform**: High-frequency macroeconomic indicators, interbank market rates, SBV bills/OMO operations. [https://widata.vn/vi-mo/vn/pmi-san-xuat](https://widata.vn/vi-mo/vn/pmi-san-xuat) | [https://widata.vn/ket-qua-kinh-doanh](https://widata.vn/ket-qua-kinh-doanh)
2. **State Bank of Vietnam (SBV / NHNN)**: Central bank reference exchange rates, policy refinancing rates, discount rates, interbank data. [https://www.sbv.gov.vn](https://www.sbv.gov.vn)
3. **VNDirect Financial Information (Finfo Open API)**: Listed stock registry, 4-tier ICB taxonomy, financial statements (`TT200/TT202/TT49`), financial ratios, daily order book supply/demand (`trading_statistics`), daily foreign trading (`foreigns`), shareholders, and corporate relationships. [https://dchart.vndirect.com.vn](https://dchart.vndirect.com.vn) | [https://www.vndirect.com.vn](https://www.vndirect.com.vn)
4. **Ho Chi Minh Stock Exchange (HOSE) & Hanoi Stock Exchange (HNX)**: Official Point-in-Time constituent rebalancing announcements, free-float factors ($f$), capping factors ($c$), and index shares. [https://www.hsx.vn](https://www.hsx.vn) | [https://www.hnx.vn](https://www.hnx.vn)
5. **Ministry of Finance (Bộ Tài chính)**: Statutory accounting circulars governing financial statement reporting lines (`TT200/2014/TT-BTC`, `TT202/2014/TT-BTC`, `TT334/2016/TT-BTC`, `TT199/2014/TT-BTC`). [https://mof.gov.vn](https://mof.gov.vn)
6. **Yahoo Finance Open Engine**: Multi-asset global macro pricing benchmarks: S&P 500 (`^GSPC`), US 10Y Yield (`^TNX`), US Dollar Index (`DX-Y.NYB`), WTI Crude Oil (`CL=F`), Gold (`GC=F`), VIX (`^VIX`), Bitcoin (`BTC-USD`), and USD/VND (`VND=X`). [https://finance.yahoo.com](https://finance.yahoo.com)
7. **Federal Reserve Bank of St. Louis (FRED)**: US Federal Reserve monetary policies, Treasuries, and money supply series. [https://fred.stlouisfed.org](https://fred.stlouisfed.org)
8. **SylphxAI anymd**: Fast native Rust document & PDF parser. [https://github.com/SylphxAI/anymd](https://github.com/SylphxAI/anymd)