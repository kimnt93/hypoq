# Macroeconomic & Financial Market Dataset Documentation (DATA.md)

This document provides a comprehensive specification of all macroeconomic, monetary, equity market, derivative, and corporate financial datasets collected and serialized into Apache Parquet format. All data files are stored under the `data/` directory and are optimized for high-performance vectorized analysis using **Polars** or **PyArrow/Pandas**.

---

## 1. Directory Structure & Inventory

```text
paper-hub/data/
├── macro/
│   ├── widata_macro_all_series.parquet      # 204,426 rows | 107 macro series unpivoted (1970–2026)
│   ├── vietnam_monetary_daily.parquet       # 5,471 rows x 16 cols | Interbank rates, OMO, SBV rates, FX, Yields
│   ├── vietnam_macro_monthly.parquet        # 298 rows x 13 cols | PMI, CPI components, IIP, Retail sales, FDI
│   ├── vietnam_market_pe_daily.parquet      # 2,877 rows x 3 cols | Daily VN-Index P/E valuation (2015–2026)
│   └── us_macro_daily.parquet               # 13,821 rows x 10 cols | Fed Funds DFF, 3M/2Y/10Y Yields, M2, WALCL
├── market/
│   ├── vn_indices_daily.parquet             # 4,570 rows x 8 cols | VNINDEX & VN30 daily OHLCV (2015–2026)
│   ├── vn30f1m_derivatives_daily.parquet    # 2,296 rows x 8 cols | VN30F1M Futures daily OHLCV (2017–2026)
│   ├── vn_top_equities_daily.parquet        # 68,871 rows x 8 cols | 26 constituent stocks across 7 key sectors
│   ├── vn_stock_industry_classification.parquet # 1,971 rows x 22 cols | All listed stocks mapped to 4-tier ICB sectors
│   ├── vn_industry_sectors_icb.parquet      # 185 rows x 7 cols | 185 ICB sector definitions & ticker constituent lists
│   ├── vn_sector_indices_daily.parquet      # 26,113 rows x 8 cols | 12 official sector indices daily OHLCV (2017–2026)
│   ├── vn_foreign_trading_daily.parquet     # 130,837 rows x 12 cols | Daily Foreign Trading (GD NĐTNN: Buy, Sell, Net, Room)
│   └── vn_order_supply_demand_daily.parquet # 83,731 rows x 11 cols | Daily Order Book Supply/Demand (Cung cầu: Bid/Offer Orders, Vol)
├── market_structure/
│   ├── vn_corporate_network_edges.parquet   # 4,803 rows x 12 cols | Directed Network Graph Edges (Loans, Subs, Owners)
│   ├── vn_corporate_network_nodes.parquet   # 3,473 rows x 8 cols | Network Nodes & Centrality (PageRank, Degree, Sector)
│   ├── vn_shareholders_history.parquet      # 3,115 rows x 9 cols | Major Shareholders & Ownership History (Cổ đông lớn)
│   ├── vn_subsidiaries_affiliates.parquet   # 1,688 rows x 10 cols | Subsidiaries & Affiliates (Công ty con & liên kết)
│   ├── vn_charter_capital_history.parquet   # 4,391 rows x 13 cols | Historical Capital Changes & Issuances (Vốn điều lệ)
│   ├── free_float.parquet                   # 43,377 rows x 18 cols | Historical free-float factor & capping factors (2017–2026)
│   ├── vn30_free_float_history.parquet      # 690 rows x 18 cols | VN30-specific free float & index shares history
│   ├── free_float_changes_observed.parquet  # 10,178 rows x 22 cols | Tracking every change in free-float parameters
│   ├── index_constituents.parquet           # 43,377 rows x 18 cols | Historical basket constituents across all VN indices
│   └── basket_history.parquet               # 261 rows x 10 cols | Basket rebalancing events and effective dates
├── financials/
│   ├── vn_financial_models_taxonomy.parquet # 3,370 rows x 12 cols | Full accounting line item definitions (TT202, TT49, TT334)
│   ├── vn_financial_statements_quarterly_yearly.parquet # Balance Sheet, Income, Cash Flow line items (Quarters & Years)
│   ├── vn_financial_ratios_quarterly_yearly.parquet     # Financial Ratios (P/E, P/B, ROE, ROA, Margins, D/E)
│   └── vn_listed_q3_2026_financials.parquet # 192 rows x 21 cols | Q3-2026 corporate results (Revenue, Profit, P/E)
└── cross_asset/
    └── global_macro_daily.parquet           # 26,358 rows x 8 cols | S&P500, US10Y, DXY, Gold, WTI, VIX, BTC, USDVND
```

---

## 2. Dataset Specifications & Schemas

### 2.1 Macroeconomic Data (`data/macro/`)

#### A. `vietnam_monetary_daily.parquet`
* **Coverage**: 2011-01-03 to 2026-10-21 (5,471 daily observations).
* **Source**: WiData (WiGroup), State Bank of Vietnam (SBV), Commercial Banks.
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `String` / `Date` | YYYY-MM-DD | Trading calendar date |
  | `interbank_rate_on` | `Float64` | % / year | Overnight (ON) interbank offer rate |
  | `sbv_refinancing_rate` | `Float64` | % / year | SBV policy refinancing interest rate (Lãi suất tái cấp vốn) |
  | `sbv_discount_rate` | `Float64` | % / year | SBV policy rediscount interest rate (Lãi suất tái chiết khấu) |
  | `usdvnd_commercial_sell` | `Float64` | VND | USD/VND ask/selling rate at commercial banks |
  | `usdvnd_commercial_buy` | `Float64` | VND | USD/VND bid/buying rate at commercial banks |
  | `usdvnd_central_rate` | `Float64` | VND | SBV official central reference exchange rate |
  | `usdvnd_freemarket_sell`| `Float64` | VND | Informal / free market USD/VND selling rate |
  | `vn_govbond_yield_1y` | `Float64` | % / year | Vietnam 1-Year Government Benchmark Bond Yield |
  | `vn_govbond_yield_10y`| `Float64` | % / year | Vietnam 10-Year Government Benchmark Bond Yield |
  | `bank_deposit_rate_12m` | `Float64` | % / year | 12-Month deposit rate proxy (MBB, ACB, TCB average) |
  | `sbv_net_liquidity_injection` | `Float64` | Billion VND | Net daily liquidity injected (+) or absorbed (-) via OMO & T-Bills |
  | `omo_outstanding_balance` | `Float64` | Billion VND | Cumulative outstanding OMO repo balance in circulation |
  | `sbv_bills_outstanding_balance` | `Float64` | Billion VND | Cumulative outstanding SBV bills (Tín phiếu NHNN) balance |
  | `gold_sjc_domestic_vnd`| `Float64` | Million VND/tael | Domestic SJC physical gold buying price |
  | `gold_world_usd_adjusted` | `Float64` | USD/oz | International spot gold price adjusted to domestic basis |

#### B. `vietnam_macro_monthly.parquet`
* **Coverage**: 2001-12-31 to 2026-09-30 (298 monthly observations).
* **Source**: General Statistics Office (GSO), S&P Global / WiData.
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `String` / `Date` | YYYY-MM-DD | End of month date |
  | `pmi_manufacturing` | `Float64` | Index (50 = Neutral) | Vietnam Manufacturing Purchasing Managers' Index (PMI) |
  | `cpi_index` | `Float64` | Base Index | Consumer Price Index (headline level) |
  | `cpi_yoy` | `Float64` | % YoY | Year-over-Year Inflation Rate |
  | `cpi_food` | `Float64` | Sub-Index | CPI Food & Catering services component |
  | `cpi_transport` | `Float64` | Sub-Index | CPI Transport component (drives energy pass-through) |
  | `cpi_housing` | `Float64` | Sub-Index | CPI Housing, Water, Electricity & Building Materials |
  | `iip_industrial_production_yoy` | `Float64` | % YoY | Industrial Production Index (toàn ngành công nghiệp) YoY |
  | `iip_mining_yoy` | `Float64` | % YoY | Mining & Quarrying sub-sector IIP YoY |
  | `retail_sales_yoy` | `Float64` | % YoY | Total retail sales of consumer goods and services YoY |
  | `fdi_disbursed_ttm` | `Float64` | Million USD | Trailing 12-month implemented Foreign Direct Investment |
  | `fdi_registered_ttm`| `Float64` | Million USD | Trailing 12-month registered Foreign Direct Investment |

#### C. `us_macro_daily.parquet`
* **Coverage**: 1986 to 2026 (13,821 observations across daily/weekly series).
* **Source**: Federal Reserve Bank of St. Louis (FRED) via WiData.
* **Fields**:
  | Column Name | Type | Description |
  | :--- | :--- | :--- |
  | `us_fed_funds_rate_dff` | `Float64` | Effective Federal Funds Rate (daily, %) |
  | `us_fed_target_upper` | `Float64` | Federal Funds Target Rate - Upper Limit (%) |
  | `us_treasury_3m` | `Float64` | 3-Month Treasury Constant Maturity Rate (%) |
  | `us_treasury_2y` | `Float64` | 2-Year Treasury Constant Maturity Rate (%) |
  | `us_treasury_10y` | `Float64` | 10-Year Treasury Constant Maturity Rate (%) |
  | `us_sofr_volume` | `Float64` | Secured Overnight Financing Rate (SOFR) daily transaction volume |
  | `us_effr_volume` | `Float64` | Effective Federal Funds Rate daily transaction volume |
  | `us_fed_total_assets_walcl`| `Float64` | Fed Total Assets: Total Assets (Less Eliminations from Consolidation) ($M) |
  | `us_m2_money_supply_yoy` | `Float64` | US M2 Money Stock YoY Growth (%) |

#### D. `vietnam_market_pe_daily.parquet`
* **Coverage**: 2015-05-04 to 2026-10-21 (2,877 daily observations).
* **Fields**: `date` (`String`), `symbol` (`String`, `VNINDEX`), `pe` (`Float64`, Trailing P/E ratio).
* **Application**: Used to compute the Equity Risk Premium (ERP):
  $$\text{ERP}_t = \frac{1}{\text{PE}_t} - r_{\text{deposit}, t}^{12\text{M}}$$

---

### 2.2 Market & Derivatives Data (`data/market/`)

#### A. `vn_indices_daily.parquet`
* **Symbols**: `VNINDEX`, `VN30`.
* **Observations**: 4,570 rows (2,285 daily trading sessions per index from 2015 to 2026).
* **Fields**: `symbol`, `date`, `timestamp`, `open`, `high`, `low`, `close`, `volume`.

#### B. `vn30f1m_derivatives_daily.parquet`
* **Symbol**: `VN30F1M` (Front-month VN30 Index Futures contract).
* **Observations**: 2,296 daily trading sessions (from inception in August 2017 to October 2026).
* **Fields**: `symbol`, `date`, `timestamp`, `open`, `high`, `low`, `close`, `volume`.
* **Application**: Basis spread analysis ($\text{Basis}_t = F_t - S_t$), volatility spillover, high-frequency tail risk modeling.

#### C. `vn_top_equities_daily.parquet`
* **Symbols (26 Tickers across 7 Sectors)**:
  * **Banking**: VCB, BID, CTG, TCB, MBB, VPB, ACB
  * **Steel & Manufacturing**: HPG, HSG
  * **Technology & Telecommunications**: FPT
  * **Retail & Consumer Goods**: MWG, MSN, VNM, DGW
  * **Securities / Brokerages**: SSI, VND, VCI, HCM
  * **Real Estate**: VHM, VIC, KDH, NVL
  * **Energy & Infrastructure**: GAS, PVD, PVS, REE
* **Observations**: 68,871 daily bars from 2015 to 2026.
* **Fields**: `symbol`, `date`, `timestamp`, `open`, `high`, `low`, `close`, `volume`.

#### D. `vn_stock_industry_classification.parquet`
* **Coverage**: 1,971 actively listed stocks on HOSE (441), HNX (359), and UPCOM (1,171).
* **Source**: VNDirect Financial Information Engine (Finfo Open API) / Official Exchange ICB Registry.
* **Taxonomy**: Mapped to the complete 4-tier **Industry Classification Benchmark (ICB)**:
  * **Level 1 (Industry)**: 10 macro sectors (Financials, Industrials, Consumer Goods, Basic Materials, etc.)
  * **Level 2 (Supersector)**: 19 supersectors (Banks, Real Estate, Financial Services, Utilities, etc.)
  * **Level 3 (Sector)**: 42 sectors
  * **Level 4 (Subsector)**: 114 granular subsectors
* **Fields**:
  | Column Name | Type | Description |
  | :--- | :--- | :--- |
  | `symbol` | `String` | Official stock ticker symbol |
  | `company_name` | `String` | Official Vietnamese corporate name |
  | `company_name_eng` | `String` | Official English corporate name |
  | `short_name` | `String` | Common corporate abbreviation |
  | `floor` | `String` | Trading exchange (`HOSE`, `HNX`, `UPCOM`) |
  | `isin` | `String` | International Securities Identification Number |
  | `status` | `String` | Listing status (`listed`) |
  | `listed_date` | `String` | Official date of initial public listing |
  | `icb_level1_code` | `String` | ICB Level 1 code (e.g. `8000`) |
  | `icb_level1_en` / `vi` | `String` | ICB Level 1 English and Vietnamese industry title |
  | `icb_level2_code` | `String` | ICB Level 2 code (e.g. `8300` Banks, `8600` Real Estate) |
  | `icb_level2_en` / `vi` | `String` | ICB Level 2 English and Vietnamese supersector title |
  | `icb_level3_code` | `String` | ICB Level 3 code (e.g. `8350`) |
  | `icb_level3_en` / `vi` | `String` | ICB Level 3 English and Vietnamese sector title |
  | `icb_level4_code` | `String` | ICB Level 4 code (e.g. `8355`) |
  | `icb_level4_en` / `vi` | `String` | ICB Level 4 English and Vietnamese subsector title |
  | `tax_code` | `String` | Corporate tax identification number |
  | `company_id` | `String` | Internal financial database corporate identifier |

#### E. `vn_industry_sectors_icb.parquet`
* **Coverage**: 185 distinct ICB classification codes across Level 1 (10), Level 2 (19), Level 3 (42), and Level 4 (114).
* **Fields**: `industry_code`, `industry_level`, `higher_level_code`, `english_name`, `vietnamese_name`, `total_count`, `code_list` (comma-separated list of all constituent stock symbols in this industry).

#### F. `vn_sector_indices_daily.parquet`
* **Coverage**: 2017-08-24 to 2026-10-22 (26,113 daily OHLCV rows across 12 official exchange sector indices).
* **Sector Indices Included**:
  * `VNREAL`: Real Estate Sector Index (Bất động sản)
  * `VNFIN`: Financials Sector Index (Tài chính - Ngân hàng, Chứng khoán)
  * `VNMAT`: Materials Sector Index (Vật liệu cơ bản - Thép, Hóa chất)
  * `VNIND`: Industrials Sector Index (Công nghiệp - Xây dựng, Vận tải)
  * `VNIT`: Information Technology Sector Index (Công nghệ thông tin)
  * `VNENE`: Energy Sector Index (Năng lượng - Dầu khí)
  * `VNUTI`: Utilities Sector Index (Dịch vụ tiện ích - Điện, Nước)
  * `VNCONS`: Consumer Sector Index (Hàng tiêu dùng)
  * `VNHEAL`: Health Care Sector Index (Y tế - Dược phẩm)
  * `VNFINLEAD`: Financial Lead Index
  * `VNDIAMOND`: Vietnam Diamond Index (Leading FOL companies)
  * `VNSI`: Vietnam Sustainability Index
* **Fields**: `symbol`, `date`, `timestamp`, `open`, `high`, `low`, `close`, `volume`.

#### G. `vn_foreign_trading_daily.parquet` (Giao dịch Nhà đầu tư nước ngoài - GD NĐTNN)
* **Coverage**: 2018-08-30 to 2026-10-22 (130,837 daily rows across 65 leading equities).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `symbol` | `String` | - | Stock ticker symbol |
  | `date` | `String` | YYYY-MM-DD | Trading date |
  | `floor` | `String` | - | Trading exchange (`HOSE` / `HNX` / `UPCOM`) |
  | `buy_val` | `Float64` | VND | Foreign buy value (Giá trị mua nước ngoài) |
  | `sell_val` | `Float64` | VND | Foreign sell value (Giá trị bán nước ngoài) |
  | `net_val` | `Float64` | VND | Foreign net value (Giá trị mua ròng: `buy_val - sell_val`) |
  | `buy_vol` | `Float64` | Shares | Foreign buy volume (Khối lượng mua nước ngoài) |
  | `sell_vol` | `Float64` | Shares | Foreign sell volume (Khối lượng bán nước ngoài) |
  | `net_vol` | `Float64` | Shares | Foreign net volume (Khối lượng mua ròng: `buy_vol - sell_vol`) |
  | `total_room` | `Float64` | Shares | Maximum permitted foreign room limit |
  | `current_room` | `Float64` | Shares | Available remaining foreign room |
  | `room_pct` | `Float64` | % | Remaining foreign room percentage (`current_room / total_room * 100`) |

#### H. `vn_order_supply_demand_daily.parquet` (Cung cầu đặt lệnh Mua/Bán)
* **Coverage**: 2021-07-09 to 2026-10-22 (83,731 daily rows across 64 leading equities).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `symbol` | `String` | - | Stock ticker symbol |
  | `date` | `String` | YYYY-MM-DD | Trading date |
  | `floor` | `String` | - | Trading exchange (`HOSE` / `HNX` / `UPCOM`) |
  | `bid_order` | `Float64` | Orders | Total number of buy orders placed (Số lượng lệnh đặt mua) |
  | `bid_vol` | `Float64` | Shares | Total buy order volume placed (Khối lượng đặt mua) |
  | `offer_order` | `Float64` | Orders | Total number of sell orders placed (Số lượng lệnh đặt bán) |
  | `offer_vol` | `Float64` | Shares | Total sell order volume placed (Khối lượng đặt bán) |
  | `avg_bid_vol` | `Float64` | Shares/order | Average volume per buy order (`bid_vol / bid_order`) |
  | `avg_offer_vol` | `Float64` | Shares/order | Average volume per sell order (`offer_vol / offer_order`) |
  | `net_vol` | `Float64` | Shares | Net order book volume imbalance (`bid_vol - offer_vol`) |
  | `order_imbalance` | `Float64` | [-1, +1] | Standardized order imbalance: $\frac{\text{bid\_vol} - \text{offer\_vol}}{\text{bid\_vol} + \text{offer\_vol}}$ |

---

### 2.3 Global Macro & Cross-Asset Data (`data/cross_asset/`)

#### `global_macro_daily.parquet`
* **Observations**: 26,358 daily records across 10 global assets.
* **Assets**:
  * `^GSPC` (`sp500`): S&P 500 Index
  * `^DJI` (`dow_jones`): Dow Jones Industrial Average
  * `^IXIC` (`nasdaq`): Nasdaq Composite
  * `^TNX` (`us_treasury_10y_yield`): US 10-Year Treasury Constant Maturity Yield
  * `DX-Y.NYB` (`dollar_index_dxy`): US Dollar Currency Index
  * `CL=F` (`crude_oil_wti`): Light Sweet Crude Oil Futures
  * `GC=F` (`gold_futures`): Gold Futures
  * `^VIX` (`cboe_vix`): CBOE Market Volatility Index
  * `BTC-USD` (`bitcoin_usd`): Bitcoin / USD
  * `VND=X` (`usdvnd_fx_rate`): USD/VND Foreign Exchange Rate
* **Fields**: `ticker`, `asset_name`, `date`, `open`, `high`, `low`, `close`, `volume`.

---

### 2.4 Corporate Financials (`data/financials/`)

#### `vn_listed_q3_2026_financials.parquet`
* **Coverage**: 192 listed companies on HOSE, HNX, and UPCOM for Q3-2026.
* **Fields**:
  * `symbol`: Stock ticker (e.g., BSI, DHG, DVP, FTS, HTG)
  * `exchange`: Trading venue (HOSE / HNX / UPCOM)
  * `industry_level4`: ICB Level 4 Sector classification
  * `revenue_bn_vnd`: Q3 Net Revenue (Billion VND)
  * `revenue_ytd_bn_vnd`: 9-Month Cumulative Net Revenue
  * `revenue_growth_yoy_pct`: Revenue Growth YoY (%)
  * `revenue_growth_qoq_pct`: Revenue Growth QoQ (%)
  * `net_profit_bn_vnd`: Q3 Net Profit After Tax (Lãi ròng)
  * `net_profit_ytd_bn_vnd`: 9-Month Cumulative Net Profit
  * `profit_growth_yoy_pct`: Net Profit Growth YoY (%)
  * `profit_growth_qoq_pct`: Net Profit Growth QoQ (%)
  * `target_plan_completion_pct`: Percentage completion of annual AGM profit target (%)
  * `pe_ratio`: Trailing P/E multiple
  * `market_cap_bn_vnd`: Market Capitalization (Billion VND)
  * `close_price`: Closing price (VND)
  * `smg_score`: Quantitative Smart Money Grade score
  * `avg_vol_15d_kshares`: 15-day average trading volume (thousand shares)
  * `report_date`: Filing date


#### A. `vn_financial_models_taxonomy.parquet` (Bảng Danh mục Chỉ tiêu Báo cáo Tài chính)
* **Coverage**: Complete Vietnamese accounting taxonomy comprising **3,370 line items** across **51 official accounting models**.
* **Legal Standards**: Officially mapped to the Ministry of Finance and State Bank of Vietnam accounting circulars:
  * **Non-Financial Corporates**: `TT200/2014/TT-BTC` & `TT202/2014/TT-BTC` (Chế độ Kế toán Doanh nghiệp & BCTC Hợp nhất)
  * **Commercial Banks**: `TT49/2014/TT-NHNN` (Chế độ BCTC Ngân hàng Thương mại)
  * **Securities Companies**: `TT334/2016/TT-BTC` (Chế độ BCTC Công ty Chứng khoán)
  * **Insurance Companies**: `TT199/2014/TT-BTC` (Chế độ BCTC Doanh nghiệp Bảo hiểm)
* **Fields**:
  | Column Name | Type | Description |
  | :--- | :--- | :--- |
  | `model_type` | `Int64` | Accounting model numeric identifier |
  | `model_type_name` | `String` | Statement category (`BALANCESHEET`, `INCOME`, `CASHFLOW`, `EXPLAINATION`, `HIGHLIGHT`) |
  | `model_vn_desc` | `String` | Model description in Vietnamese (e.g. `Bảng Cân đối kế toán - Chung`) |
  | `model_en_desc` | `String` | Model description in English |
  | `company_form` | `String` | Legal enterprise form (`NON_FINANCE`, `BANK`, `SECURITIES`, `INSURANCE`) |
  | `accounting_circular` | `String` | Official governing circular (e.g. `TT202/2014/TT-BTC`) |
  | `item_code` | `Int64` | Standardized accounting line item code (e.g. `11000`, `21000`, `23800`) |
  | `item_vn_name` | `String` | Line item accounting title in Vietnamese (e.g. `Tài sản ngắn hạn`, `Lợi nhuận trước thuế`) |
  | `item_en_name` | `String` | Line item accounting title in English (e.g. `Current assets`, `Profit before tax`) |
  | `display_order` | `Int64` | Official regulatory display order on published statements |
  | `display_level` | `Int64` | Hierarchical indentation level ($1 = \text{Header Category}, 2 = \text{Subcategory}, 3 = \text{Detail}$) |
  | `form_type` | `String` | Web vs. Full Audit report format (`WEB`, `ALL`) |

#### B. `vn_financial_statements_quarterly_yearly.parquet` (Báo cáo Tài chính Chi tiết Hàng quý & Hàng năm)
* **Coverage**: Point-in-time financial statement line items (Balance Sheet, Income Statement, Cash Flow) for listed equities across all quarterly (`QUARTER1`, `QUARTER2`, `QUARTER3`, `QUARTER4`) and annual (`YEAR`) reporting periods.
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `symbol` | `String` | - | Stock ticker symbol |
  | `fiscal_date` | `String` | YYYY-MM-DD | Accounting period closing date |
  | `year` | `Int32` | Year | Fiscal reporting year |
  | `quarter` | `Int32` | 1-4, 0 | Reporting quarter ($1, 2, 3, 4$; $0 = \text{Full Year Audited Annual Report}$) |
  | `report_type` | `String` | - | Regulatory term (`QUARTER1`, `QUARTER2`, `QUARTER3`, `QUARTER4`, `YEAR`) |
  | `statement_type`| `String` | - | Financial statement (`BALANCESHEET`, `INCOME`, `CASHFLOW`, `EXPLAINATION`) |
  | `item_code` | `Int64` | - | Accounting line item code |
  | `item_name_vn` | `String` | - | Line item accounting description in Vietnamese |
  | `item_name_en` | `String` | - | Line item accounting description in English |
  | `display_level` | `Int32` | - | Accounting hierarchy level |
  | `value` | `Float64` | VND | Reported line item monetary value |
  | `model_type` | `Int64` | - | Corresponding accounting model identifier |

#### C. `vn_financial_ratios_quarterly_yearly.parquet` (Chỉ số Tài chính Chuẩn hóa Hàng quý & Hàng năm)
* **Coverage**: Point-in-time fundamental financial metrics and ratios calculated from official financial statements across quarterly and annual intervals:
  * **Profitability (Sinh lời)**: `ROAE_TR_AVG5Q` (ROE), `ROAA_TR_AVG5Q` (ROA), `GROSS_MARGIN_QR` (Biên lợi nhuận gộp), `NET_MARGIN_QR` (Biên lợi nhuận ròng), `OPERATING_EBIT_MARGIN_QR`, `OPERATING_EBITDA_MARGIN_QR`.
  * **Leverage & Solvency (Đòn bẩy & Nợ)**: `DEBT_TO_EQUITY_AQ` (D/E), `EQUITY_TO_ASSET_AQ`, `DEBT_EBITDA`.
  * **Liquidity (Thanh khoản)**: `CURRENT_RATIO_AQ` (Thanh toán hiện hành), `QUICK_RATIO_AQ` (Thanh toán nhanh).
  * **Valuation Multiples (Định giá)**: `PRICE_TO_EARNINGS` (P/E), `PRICE_TO_BOOK` (P/B), `PRICE_TO_SALES` (P/S), `DIVIDEND_YIELD`, `MARKETCAP`.
  * **Per-Share Metrics (Chỉ tiêu cổ phần)**: `EPS_TR` (EPS 4 quý liền kề), `BVPS_CR` (Giá trị sổ sách/CP).
  * **Operating Efficiency (Hiệu quả hoạt động)**: `INVENTORY_TURNOVER_TR_AVG5Q`, `RECEIVABLE_TURNOVER_TR_AVG5Q`, `PAYABLE_TURNOVER_TR_AVG5Q`.
  * **Scale & Growth (Quy mô & Tăng trưởng)**: `NET_SALES_QR`, `TOTAL_ASSETS_AQ`, `NET_SALES_TR_GRYOY`, `TOTAL_ASSETS_AQ_GRYOY`.
* **Fields**: `symbol`, `report_date`, `year`, `quarter`, `ratio_code`, `ratio_name`, `category`, `value`.
### 2.5 Market Structure & Free-Float Data (`data/market_structure/`)
* **Provenance**: Point-in-time exchange notices and index committee announcements from HOSE and HNX (sourced from `lazy-researcher/datasets/vn_market_structure/`).

#### A. `free_float.parquet` & `vn30_free_float_history.parquet`
* **Coverage**: 2017-07-24 to 2026-08-03 (43,377 point-in-time rows across 1,450 tickers and 14 indices; 690 rows specific to VN30).
* **Indices**: `VN30`, `VN100`, `VNALLSHARE`, `VNMIDCAP`, `VNSMALLCAP`, `VNSI`, `VNX50`, `VNXALLSHARE`, `HNX30`, `UPXLARGE`, etc.
* **Key Fields**:
  | Column Name | Type | Description |
  | :--- | :--- | :--- |
  | `time` | `String` (ISO UTC) | Timestamp of publication announcement (causal Point-in-Time availability) |
  | `symbol` | `String` | Stock ticker |
  | `index` | `String` | Target index basket (e.g., `VN30`) |
  | `exchange` | `String` | Trading exchange (`HOSE` / `HNX`) |
  | `effective_date` | `String` | Official date from which the parameters take effect |
  | `index_shares` | `Int64` | Total outstanding shares considered for index calculation |
  | `free_float_factor` | `Float64` | Free-float ratio ($f_i \in (0, 1]$), rounded to HOSE standard intervals |
  | `capping_factor` | `Float64` | Single-stock weight capping factor ($c_i \in (0, 1]$) |
  | `source_url` | `String` | Official exchange publication URL |
  | `document_url` | `String` | Direct link to official PDF disclosure document |
  | `source_sha256` | `String` | Cryptographic SHA-256 hash of the announcement document |

* **Mathematical Weight Formula in Index Construction**:
  $$w_{i, t} = \frac{P_{i, t} \times Q_{i, t} \times f_{i, t} \times c_{i, t}}{\sum_{j=1}^N P_{j, t} \times Q_{j, t} \times f_{j, t} \times c_{j, t}}$$
  where $P_{i,t}$ is market price, $Q_{i,t}$ is `index_shares`, $f_{i,t}$ is `free_float_factor`, and $c_{i,t}$ is `capping_factor`.

#### B. `free_float_changes_observed.parquet`
* **Coverage**: 10,178 observations capturing every discrete adjustment event in free-float factors or capping parameters over time.

#### C. `index_constituents.parquet` & `basket_history.parquet`
* **Coverage**: Full quarterly rebalancing history, entry/exit constituents, and effective rebalancing schedules across all major Vietnamese index baskets.


#### D. `vn_shareholders_history.parquet` (Cổ đông lớn & Lịch sử sở hữu theo báo cáo)
* **Coverage**: 3,115 historical shareholder records across 65 leading equities.
* **Fields**:
  | Column Name | Type | Description |
  | :--- | :--- | :--- |
  | `symbol` | `String` | Stock ticker symbol |
  | `shareholder_type` | `String` | Entity type: `PERSON` (Cá nhân) or `COMPANY` (Tổ chức) |
  | `shareholder_name` | `String` | Full name of shareholder (State corporation, fund, insider executive) |
  | `role_type` | `String` | Corporate governance role in Vietnamese (e.g. `Cổ đông lớn`, `Chủ tịch HĐQT`) |
  | `en_role_type` | `String` | Corporate governance role in English (`Major Shareholder`, `Chairman`, etc.) |
  | `number_of_shares`| `Float64` | Total number of shares held at disclosure date |
  | `ownership_pct` | `Float64` | Direct ownership percentage (%) |
  | `group_pct` | `Float64` | Group affiliated ownership percentage (%) |
  | `effective_date` | `String` | Point-in-time filing / effective announcement date (YYYY-MM-DD) |

#### E. `vn_subsidiaries_affiliates.parquet` (Công ty con & Công ty liên kết)
* **Coverage**: 1,688 corporate relationships across 65 leading listed groups.
* **Fields**:
  | Column Name | Type | Description |
  | :--- | :--- | :--- |
  | `symbol` | `String` | Parent stock ticker symbol |
  | `relation_type` | `String` | Relationship classification: `SUBSIDIARY` (>50%), `ASSOCIATE` (20-50%), `PARENT`, `AUDITOR` |
  | `relation_name_vn` | `String` | Entity name in Vietnamese |
  | `relation_name_en` | `String` | Entity name in English |
  | `relation_id` | `Int64` | Unique corporate entity identifier |
  | `effective_date` | `String` | Financial report effective date |
  | `year` | `String` | Reporting fiscal year |
  | `ownership_pct` | `Float64` | Parent group equity ownership percentage (%) |
  | `investment_capital` | `Float64` | Contributed capital value (VND) |
  | `unit` | `String` | Currency unit (`VND`) |

#### F. `vn_charter_capital_history.parquet` (Lịch sử tăng vốn điều lệ & phát hành)
* **Coverage**: 4,391 capital adjustment events across 65 leading equities.
* **Fields**:
  | Column Name | Type | Description |
  | :--- | :--- | :--- |
  | `symbol` | `String` | Stock ticker symbol |
  | `event_id` | `String` | Unique event identifier |
  | `event_group` | `String` | Event category (`investorRight`) |
  | `event_type` | `String` | Event action: `ISSUE` (Phát hành thêm), `LISTED` (Niêm yết/GD bổ sung), `DIVIDEND` |
  | `type_desc` | `String` | Description of issuance or rights entitlement |
  | `note` | `String` | Detailed operational terms (e.g. "Phát hành 900,000,000 cp cho CĐHH, giá 10000 đ/cp") |
  | `price` | `Float64` | Issuance execution price (VND/share) |
  | `ratio` | `Float64` | Issuance / entitlement ratio (%) |
  | `number_of_shares` | `Float64` | Additional number of shares issued |
  | `disclosure_date` | `String` | Date of public disclosure |
  | `effective_date` | `String` | Ex-rights / effective date |
  | `trading_start_date`| `String` | Date when new shares begin secondary trading |
  | `actual_date` | `String` | Date of final distribution |

#### G. `vn_corporate_network_edges.parquet` (Cấu trúc Đồ thị Mạng lưới Doanh nghiệp & Liên kết Sở hữu)
* **Coverage**: 4,803 directed relationship edges connecting Vietnamese listed companies, conglomerates, subsidiaries, major shareholders, state holding corporations, and auditors.
* **Fields**:
  | Column Name | Type | Description |
  | :--- | :--- | :--- |
  | `source_id` | `String` | Source entity identifier (e.g. `VIC`, `SCIC`, `VCB`, `KPMG`) |
  | `source_name` | `String` | Source entity clean title |
  | `source_type` | `String` | Entity category (`LISTED_EQUITY`, `STATE_OWNED_ENTITY`, `INSTITUTIONAL_INVESTOR`, `INDIVIDUAL_INSIDER`) |
  | `target_id` | `String` | Target entity identifier (e.g. `VHM`, `HVN`, `CTG`) |
  | `target_name` | `String` | Target entity clean title |
  | `target_type` | `String` | Target category (`LISTED_EQUITY`, `SUBSIDIARY`, `ASSOCIATE`, `AUDITOR`, `PARENT_GROUP`) |
  | `relationship_type` | `String` | Edge classification: `SUBSIDIARY_CONTROL` (>50%), `ASSOCIATE_AFFILIATE` (20-50%), `STATE_EQUITY_CONTROL`, `INSTITUTIONAL_BLOCK_HOLDING`, `INSIDER_MANAGEMENT_HOLDING`, `AUDITOR_SERVICE` |
  | `ownership_pct` | `Float64` | Equity holding percentage (%) if reported |
  | `capital_value_vnd`| `Float64` | Contributed capital value in VND if reported |
  | `effective_date` | `String` | Point-in-time disclosure date (YYYY-MM-DD) |
  | `weight` | `Float64` | Normalized connection weight $\in [0.05, 1.0]$ |
  | `is_directed` | `Boolean` | True (Directed graph edge) |

#### H. `vn_corporate_network_nodes.parquet` (Nút Mạng lưới & Chỉ số Trung tâm Graph Centrality)
* **Coverage**: 3,473 unique corporate and institutional entities in the Vietnamese financial network.
* **Network Centrality Metrics**:
  * **PageRank Centrality ($PR$)**:
    $$PR(u) = \frac{1 - d}{N} + d \sum_{v \in B_u} \frac{PR(v)}{L(v)}, \quad d = 0.85$$
  * **Degree Centrality**: Incoming edges ($k_{\text{in}}$), Outgoing edges ($k_{\text{out}}$), and Total Connectivity ($k_{\text{tot}}$).
* **Fields**:
  | Column Name | Type | Description |
  | :--- | :--- | :--- |
  | `node_id` | `String` | Canonical node identifier |
  | `node_name` | `String` | Entity official name |
  | `node_type` | `String` | Entity type (`LISTED_EQUITY`, `SUBSIDIARY`, `ASSOCIATE`, `STATE_OWNED_ENTITY`, `AUDIT_FIRM`, `CORPORATE_AFFILIATE`, `INDIVIDUAL_OR_OTHER`) |
  | `sector` | `String` | Economic sector or institutional domain |
  | `in_degree` | `Int64` | Total incoming directed edges |
  | `out_degree` | `Int64` | Total outgoing directed edges |
  | `total_degree` | `Int64` | Total degree centrality ($k_{\text{in}} + k_{\text{out}}$) |
  | `pagerank` | `Float64` | Google PageRank centrality score |
---

## 3. Data Loading Examples (Python & Polars)

### Example 1: Loading and Joining Macro Drivers with VN-Index Returns

```python
import polars as pl

# 1. Read VNINDEX daily prices
df_idx = pl.read_parquet("data/market/vn_indices_daily.parquet")
vnindex = (
    df_idx.filter(pl.col("symbol") == "VNINDEX")
    .sort("date")
    .with_columns([
        pl.col("date").cast(pl.Date),
        ((pl.col("close") / pl.col("close").shift(1)) - 1.0).alias("log_return")
    ])
)

# 2. Read Daily Monetary Conditions
df_mon = (
    pl.read_parquet("data/macro/vietnam_monetary_daily.parquet")
    .with_columns(pl.col("date").cast(pl.Date))
)

# 3. Read Market PE
df_pe = (
    pl.read_parquet("data/macro/vietnam_market_pe_daily.parquet")
    .with_columns(pl.col("date").cast(pl.Date))
    .select(["date", pl.col("pe").alias("market_pe")])
)

# 4. Join datasets
merged = (
    vnindex
    .join(df_mon, on="date", how="left")
    .join(df_pe, on="date", how="left")
    .with_columns([
        # Yield curve slope: 10Y - 1Y
        (pl.col("vn_govbond_yield_10y") - pl.col("vn_govbond_yield_1y")).alias("yield_curve_slope"),
        # Equity Risk Premium: (1 / PE) - 12M Deposit rate
        ((1.0 / pl.col("market_pe")) * 100.0 - pl.col("bank_deposit_rate_12m")).alias("equity_risk_premium")
    ])
)

print(merged.select(["date", "close", "log_return", "interbank_rate_on", "yield_curve_slope", "equity_risk_premium"]).tail(5))
```

### Example 2: Handling Mixed-Frequency (Monthly PMI Forward-Fill or MIDAS)

```python
import polars as pl

# Read monthly macro indicators
df_macro_m = (
    pl.read_parquet("data/macro/vietnam_macro_monthly.parquet")
    .with_columns(pl.col("date").cast(pl.Date))
    .select(["date", "pmi_manufacturing", "cpi_yoy", "iip_industrial_production_yoy"])
)

# Read daily trading calendar
daily_dates = vnindex.select("date")

# Mixed-frequency backward-asof join (avoids look-ahead bias by matching on release date)
daily_macro_aligned = daily_dates.join_asof(
    df_macro_m.sort("date"),
    on="date",
    strategy="backward"
)
```

---

## 4. Data Quality, Verification & Integrity Checks

1. **Continuity & Completeness**:
   * Equities & Indices have unbroken daily business day timestamps corresponding to HOSE/HNX trading calendars from 2015 to 2026.
   * Interest rates and exchange rates cover interbank and official central bank series without calendar gaps.
2. **Precision & Consistency**:
   * Prices and values are stored in 64-bit IEEE floating-point precision (`Float64`).
   * Dates adhere strictly to ISO 8601 format (`YYYY-MM-DD`).
3. **No Look-Ahead Leakage**:
   * Corporate quarterly results contain explicit `report_date` tags, enabling strict out-of-sample backtesting without look-ahead bias.

---

## 5. References & Verified Data Sources

1. **WiData / WiChart Financial Platform**: High-frequency macroeconomic series (Manufacturing PMI, CPI components, IIP), interbank rates, SBV bills/OMO operations, and market valuation multiples. [https://widata.vn/vi-mo/vn/pmi-san-xuat](https://widata.vn/vi-mo/vn/pmi-san-xuat) | [https://widata.vn/ket-qua-kinh-doanh](https://widata.vn/ket-qua-kinh-doanh)
2. **State Bank of Vietnam (SBV / NHNN)**: Central bank reference exchange rates, policy refinancing rates, discount rates, and interbank transaction benchmarks. [https://www.sbv.gov.vn](https://www.sbv.gov.vn)
3. **VNDirect Securities Financial Information Engine (Finfo Open API)**: Listed company registry, 4-tier ICB taxonomy, financial statements (TT200/TT202/TT49), financial ratios, daily order book supply/demand (`trading_statistics`), daily foreign trading (`foreigns`), shareholders, and corporate relationships. [https://dchart.vndirect.com.vn](https://dchart.vndirect.com.vn) | [https://www.vndirect.com.vn](https://www.vndirect.com.vn)
4. **Ho Chi Minh Stock Exchange (HOSE) & Hanoi Stock Exchange (HNX)**: Official Point-in-Time constituent rebalancing announcements, free-float factors ($f$), capping factors ($c$), and outstanding index shares. [https://www.hsx.vn](https://www.hsx.vn) | [https://www.hnx.vn](https://www.hnx.vn)
5. **Ministry of Finance (Bộ Tài chính)**: Statutory accounting circulars governing financial statement reporting lines (`TT200/2014/TT-BTC`, `TT202/2014/TT-BTC`, `TT334/2016/TT-BTC`, `TT199/2014/TT-BTC`). [https://mof.gov.vn](https://mof.gov.vn)
6. **Federal Reserve Bank of St. Louis (FRED)**: Economic Data Repository for US Federal Reserve monetary policies, Treasuries, and money supply series. [https://fred.stlouisfed.org](https://fred.stlouisfed.org)
7. **Yahoo Finance Open Financials Engine**: Global multi-asset pricing benchmarks: S&P 500 (`^GSPC`), US 10Y Yield (`^TNX`), US Dollar Index (`DX-Y.NYB`), WTI Crude Oil (`CL=F`), Gold (`GC=F`), VIX (`^VIX`), Bitcoin (`BTC-USD`), and USD/VND (`VND=X`). [https://finance.yahoo.com](https://finance.yahoo.com)
8. **SylphxAI anymd**: High-speed native Rust document and PDF extraction engine for financial filings and regulatory disclosures. [https://github.com/SylphxAI/anymd](https://github.com/SylphxAI/anymd)
