# Global Macroeconomic, Financial, Crypto & Multi-Asset Dataset Documentation (DATA.md)

This document provides a comprehensive, standardized specification of all 70 empirical datasets serialized into Apache Parquet format. All datasets are stored under `data/` and optimized for zero-copy vectorized analytics using **Polars** or **PyArrow/Pandas**.

---

## 1. Directory Structure & Inventory (70 Parquet Files, 3,106,907 Rows)

```text
data/
├── macro/
│   ├── asia_macro_daily.parquet                     # 20,822 rows x 11 cols | Asian FX Rates (USD/JPY, USD/CNY, USD/KRW, USD/INR)
│   ├── cn_macro_daily.parquet                       # 2,602 rows x 11 cols | China FX & Currency Series (USD/CNY)
│   ├── eu_macro_daily.parquet                       # 13,017 rows x 11 cols | European FX Rates (EUR/USD, GBP/USD, EUR Crosses)
│   ├── global_commodities_daily.parquet             # 35,195 rows x 11 cols | Energy, Precious & Industrial Metals, Agriculture
│   ├── global_credit_spreads_daily.parquet          # 19,758 rows x 11 cols | IG (LQD), HY (HYG), EMB, TIPS, Munis & Freight (BDRY)
│   ├── global_cross_asset_volatility_daily.parquet  # 2,516 rows x 5 cols | VIX Equity, VVIX Vol of Vol, OVX Oil, GVZ Gold
│   ├── global_macro_indicators_monthly.parquet      # 13,540 rows x 9 cols | FRED Inflation, Unemployment, NFCI, Breakeven, M2
│   ├── hk_macro_daily.parquet                       # 2,602 rows x 11 cols | Hong Kong FX & Peg Series (USD/HKD)
│   ├── jp_macro_daily.parquet                       # 5,206 rows x 11 cols | Japan FX & Cross Series (USD/JPY, EUR/JPY)
│   ├── kr_macro_daily.parquet                       # 2,603 rows x 11 cols | South Korea FX Series (USD/KRW)
│   ├── us_fed_balance_sheet_weekly.parquet          # 1,245 rows x 7 cols | Federal Reserve Assets, Loans, MBS & RRP
│   ├── us_fed_fomc_rate_decisions_pit.parquet       # 88 rows x 10 cols | Point-in-time FOMC decisions: announcement time, target range before/after, change in bp
│   ├── us_macro_daily.parquet                       # 26,414 rows x 10 cols | Fed Funds DFF, 3M/2Y/10Y Yields, M2, WALCL
│   ├── us_yield_curve_daily.parquet                 # 2,516 rows x 8 cols | US Treasury Curve (3M, 5Y, 10Y, 30Y), Slopes & VIX
│   ├── vn_cpi_inflation_annual.parquet              # 30 rows x 3 cols | World Bank annual CPI inflation for Vietnam (reference-period dated)
│   ├── vn_fdi_sectors_monthly.parquet               # 173 rows x 5 cols | Realized FDI by Sector (Manufacturing, RE)
│   ├── vn_macro_monthly.parquet                     # 298 rows x 13 cols | PMI, CPI components, IIP, Retail sales, FDI
│   ├── vn_market_pe_daily.parquet                   # 2,877 rows x 3 cols | Daily VN-Index P/E valuation (2015–2026)
│   ├── vn_monetary_daily.parquet                    # 5,471 rows x 16 cols | Interbank rates, OMO, SBV rates, FX, Yields
│   ├── vn_sbv_policy_rates_daily.parquet            # 4,429 rows x 6 cols | SBV Policy Rates & Required Reserves
│   └── vn_trade_exports_monthly.parquet             # 213 rows x 8 cols | Merchandise Exports (Steel, Coffee, Total)
├── market/
│   ├── asia_equities_daily.parquet                  # 22,457 rows x 12 cols | Asian Titans (TSMC, Samsung, Alibaba, Tencent, Toyota)
│   ├── asia_indices_daily.parquet                   # 22,182 rows x 11 cols | Nikkei 225, Hang Seng, SSE Comp, CSI 300, KOSPI, Nifty
│   ├── cn_equities_daily.parquet                    # 41,241 rows x 12 cols | 18 Chinese Leaders (Moutai, ICBC, CATL, Alibaba, PetroChina)
│   ├── cn_indices_daily.parquet                     # 2,427 rows x 11 cols | China Domestic Benchmark Indices (SSE Comp, CSI 300)
│   ├── eu_equities_daily.parquet                    # 50,599 rows x 12 cols | 21 European Blue-Chips (ASML, SAP, Shell, LVMH, Allianz)
│   ├── eu_indices_daily.parquet                     # 20,296 rows x 11 cols | Euro Stoxx 50, DAX 40, FTSE 100, CAC 40, IBEX 35
│   ├── global_etfs_daily.parquet                    # 42,738 rows x 11 cols | Asset Class, Regional & Factor ETFs (SPY, QQQ, TLT)
│   ├── global_factors_daily.parquet                 # 20,112 rows x 11 cols | Smart Beta Factors (MTUM, QUAL, USMV, VLUE, VUG, IJR)
│   ├── global_sectors_daily.parquet                 # 27,239 rows x 11 cols | 11 Official GICS Sector SPDRs (Tech, Financials, Energy)
│   ├── hk_equities_daily.parquet                    # 34,436 rows x 12 cols | 16 Hong Kong Blue-Chips (Tencent, Alibaba HK, HSBC, AIA, BYD)
│   ├── hk_indices_daily.parquet                     # 2,461 rows x 11 cols | Hong Kong Benchmark Indices (Hang Seng Index)
│   ├── jp_equities_daily.parquet                    # 44,334 rows x 12 cols | 18 Japanese Titans (Toyota, Sony, Keyence, SoftBank, Mitsubishi)
│   ├── jp_indices_daily.parquet                     # 2,442 rows x 11 cols | Japan Benchmark Indices (Nikkei 225 Index)
│   ├── kr_equities_daily.parquet                    # 37,845 rows x 12 cols | 16 South Korean Leaders (Samsung, SK Hynix, Hyundai, Kia, Celltrion)
│   ├── kr_indices_daily.parquet                     # 4,896 rows x 11 cols | South Korea Benchmark Indices (KOSPI, KOSDAQ)
│   ├── us_equities_daily.parquet                    # 87,990 rows x 12 cols | 35 US Mega-Caps across Tech, Financials, Health, Energy
│   ├── us_indices_daily.parquet                     # 15,086 rows x 11 cols | S&P 500, Nasdaq 100, Dow Jones, Russell 2000, MidCap
│   ├── vn_etf_market_daily.parquet                  # 22,391 rows x 12 cols | 15 Listed Vietnamese ETFs Daily OHLCV (2014–2026)
│   ├── vn_foreign_trading_daily.parquet             # 130,902 rows x 12 cols | Daily Foreign Trading (GD NĐTNN: Buy, Sell, Room)
│   ├── vn_indices_daily.parquet                     # 4,574 rows x 8 cols | VNINDEX & VN30 daily OHLCV (2015–2026)
│   ├── vn_industry_sectors_icb.parquet              # 185 rows x 7 cols | 185 ICB sector definitions & ticker lists
│   ├── vn_order_supply_demand_daily.parquet         # 85,173 rows x 11 cols | Daily Order Book Supply/Demand (Cung cầu: Orders, Vol)
│   ├── vn_sector_indices_daily.parquet              # 26,124 rows x 8 cols | 12 official sector indices daily OHLCV
│   ├── vn_stock_industry_classification.parquet     # 1,971 rows x 22 cols | All listed stocks mapped to 4-tier ICB
│   ├── vn_top_equities_daily.parquet                # 177,752 rows x 8 cols | 71 liquid stocks daily OHLCV
│   └── vn_vn30f1m_derivatives_daily.parquet         # 2,298 rows x 8 cols | VN30F1M Futures daily OHLCV (2017–2026)
├── community/
│   ├── global_hackernews_finance_posts.parquet      # 5,780 rows x 10 cols | Hacker News stories on macro, central banks, tariffs, crypto, markets (points, comments)
│   ├── global_mastodon_finance_posts.parquet        # 9,653 rows x 12 cols | Public Mastodon posts on markets, macro and crypto hashtags (raw text, engagement)
│   ├── global_stackexchange_finance_questions.parquet # 4,500 rows x 11 cols | Top-voted Stack Exchange money / quant / economics questions (raw text, tags, score)
│   ├── global_vip_social_posts_raw.parquet          # 37,251 rows x 11 cols | Raw Truth Social posts by Donald Trump: text, UTC timestamp, replies/reposts/likes
│   ├── us_fed_communications.parquet                # 1,214 rows x 8 cols | FOMC statements, minutes and Fed speeches/testimony, full text, UTC release times
│   └── us_sec_8k_filings.parquet                    # 5,664 rows x 10 cols | SEC EDGAR 8-K filings of the US mega-caps: acceptance time, item codes, document link
├── crypto/
│   ├── crypto_market_daily.parquet                  # 3,273 rows x 9 cols | Macro Crypto Indicators, ETH/BTC, Volatility & Corrs
│   └── crypto_top_assets_daily.parquet              # 32,369 rows x 12 cols | Top Cryptos (BTC, ETH, SOL, BNB, XRP, ADA, DOGE)
├── market_structure/
│   ├── vn_basket_history.parquet                    # 261 rows x 10 cols | Basket rebalancing events and effective dates
│   ├── vn_charter_capital_history.parquet           # 4,391 rows x 13 cols | Historical Capital Changes & Issuances (Vốn điều lệ)
│   ├── vn_corporate_network_edges.parquet           # 4,803 rows x 12 cols | Corporate Network Directed Edges (Loans, Subs, Owners)
│   ├── vn_corporate_network_nodes.parquet           # 3,473 rows x 8 cols | Corporate Network Nodes & Centrality (PageRank, Degree)
│   ├── vn_free_float.parquet                        # 43,377 rows x 18 cols | Historical free-float factor & capping factors
│   ├── vn_free_float_changes_observed.parquet       # 10,178 rows x 22 cols | Tracking discrete parameter changes
│   ├── vn_shareholders_history.parquet              # 3,115 rows x 9 cols | Major Shareholders & Ownership History (Cổ đông lớn)
│   ├── vn_subsidiaries_affiliates.parquet           # 1,688 rows x 10 cols | Subsidiaries & Affiliates (Công ty con & liên kết)
│   └── vn_vn30_free_float_history.parquet           # 690 rows x 18 cols | VN30-specific free float & index shares history
├── financials/
│   ├── global_corporate_fundamentals.parquet        # 50 rows x 20 cols | 50 Global Mega-Caps Valuation, Margins, Debt & Cash Flow
│   ├── vn_financial_models_taxonomy.parquet         # 3,370 rows x 12 cols | Full accounting line item definitions (TT202, TT49)
│   ├── vn_financial_ratios_quarterly_yearly.parquet # 805,352 rows x 8 cols | 
│   ├── vn_financial_statements_quarterly_yearly.parquet # 1,008,139 rows x 12 cols | Balance Sheet, Income, Cash Flow line items
│   └── vn_listed_q3_2026_financials.parquet         # 192 rows x 20 cols | Q3-2026 corporate results (Revenue, Profit, P/E)
└── cross_asset/
    └── global_macro_daily.parquet                   # 26,358 rows x 8 cols | S&P500, US10Y, DXY, Gold, WTI, VIX, BTC, USDVND
```

---

## 1.1 Dataset Standardized Jurisdiction Naming Standard

Every empirical dataset archived in `data/` must strictly use an ISO country or geographic scope prefix:
* **`cn_`**: China mainland equity indices, corporate leaders, and USD/CNY exchange rate.
* **`hk_`**: Hong Kong equity benchmarks, blue-chips, and USD/HKD peg series.
* **`jp_`**: Japan Nikkei indices, corporate titans, and JPY macro rates.
* **`kr_`**: South Korea KOSPI/KOSDAQ indices, chaebol equities, and USD/KRW currency series.
* **`asia_`**: Asian regional aggregate indices, multi-country equities, and currency benchmarks.
* **`us_`**: United States Federal Reserve, Treasury term structure, equity benchmarks, and mega-caps.
* **`eu_`**: European Central Bank, sovereign yields, European benchmark indices, and foreign exchange rates.
* **`global_`**: International multi-asset cross-border benchmarks, 11 GICS sectors, factors, commodities, credit, and fundamentals.
* **`crypto_`**: Decentralized digital assets, Layer 1 benchmarks, and crypto-macro aggregate indicators.
* **`community/`** (folder): raw public text and announcements: Truth Social, Hacker News, Stack Exchange, Mastodon (`global_*`), Fed statements and speeches, SEC 8-K filings (`us_*`).
* **`vn_`**: Vietnam domestic financial, corporate, and macroeconomic series.

## 1.2 Provenance, Refresh & Known Caveats

**Policy**: every row in `data/` is real and comes from a free, key-less public source (or a documented vendor snapshot that was cross-checked). Nothing is simulated,
hand-typed, interpolated or zero-filled: missing values are `null`. Paid or login-only sources (Bloomberg-style feeds, Reddit, X) are not used. The pipeline
(`make download`, `scripts/download_all.py`) refreshes the LIVE groups; `make verify` audits the result (duplicate keys, all-null columns, OHLC sanity, staleness, short series).

| Group | Files | Source | Refresh | Caveats |
| :--- | :--- | :--- | :---: | :--- |
| Global equities, indices, ETFs, sectors, factors, commodities, credit, FX, crypto | `us_/eu_/asia_/cn_/hk_/jp_/kr_*`, `global_etfs`, `global_sectors`, `global_factors`, `global_commodities`, `global_credit_spreads`, `global_cross_asset_volatility`, `*_macro_daily`, `us_yield_curve_daily`, `crypto_*`, `global_macro_daily` | Yahoo Finance (`scripts/yf_common.py`) | LIVE | `close` is split-adjusted, `adj_close` also dividend-adjusted; `returns` is the simple *price* return of consecutive bars, first bar `null`. FX/commodity bars can have `close` outside `[low, high]` (Yahoo quirk). WTI `CL=F` is genuinely negative on 2020-04-20/21. Equity universes are today's large caps: **survivorship-biased**. `global_corporate_fundamentals` (Yahoo `info`) is a snapshot as of the download date, not point-in-time. |
| US macro and Fed balance sheet | `us_macro_daily`, `us_fed_balance_sheet_weekly`, `global_macro_indicators_monthly` | FRED | LIVE | Balance-sheet items in USD billions. `us_m2_money_supply_yoy` is dated at the end of its reference month but published weeks later (lag it). FRED serves revised vintages, not first releases. |
| Annual CPI (World Bank) | `vn_cpi_inflation_annual` | World Bank API | LIVE | Annual, reference-period dated. |
| FOMC decisions (derived) | `us_fed_fomc_rate_decisions_pit` | Fed statements + FRED target range | LIVE | Announcement time is the official release time in UTC. |
| Vietnam prices, flows, fundamentals | `vn_indices_daily`, `vn_vn30f1m_derivatives_daily`, `vn_top_equities_daily`, `vn_sector_indices_daily`, `vn_foreign_trading_daily`, `vn_order_supply_demand_daily`, `vn_etf_market_daily`, `vn_shareholders_history`, `vn_subsidiaries_affiliates`, `vn_charter_capital_history`, `vn_financial_*`, `vn_stock_industry_classification`, `vn_industry_sectors_icb` | VNDirect open API | LIVE | `vn_financial_statements` mixes `QUARTER`, `QUARTER2`, `ANNUAL`, `ANNUAL2` and analyst `ESTIMATION` rows and has **no publication date**: keep one `report_type`, drop `ESTIMATION`, and lag quarterly data by at least 45 days (annual 90). Sessions without a bar are absent, not zero. |
| Vietnam network (derived) | `vn_corporate_network_edges`, `vn_corporate_network_nodes` | built from the shareholder/subsidiary tables | LIVE | Snapshot of current ownership, not point-in-time. |
| Vietnam index structure | `vn_free_float`, `vn_vn30_free_float_history`, `vn_free_float_changes_observed`, `vn_basket_history` | Public HOSE/HNX index notices, collected by the owner's `lazy-researcher` project (`HYPOQ_VN_STRUCTURE_DIR`) | SNAPSHOT 2026-10-12 | **Partial history, not certified for index replication**: see that project's `coverage.json`. Factors are published and rounded; unobserved intervals are not filled. `announced_at` is empty when the notice time was unknown. |
| Vietnam macro | `vn_monetary_daily`, `vn_macro_monthly`, `vn_sbv_policy_rates_daily`, `vn_trade_exports_monthly`, `vn_fdi_sectors_monthly`, `vn_market_pe_daily` | WiData (public `widata.vn` pages that aggregate SBV, GSO and exchange releases) | SNAPSHOT 2026-10-21 | Verified: CPI y/y matches World Bank annual averages within 0.01 pp for 2019-2025, annual exports and disbursed FDI match official totals, SBV rates match every policy decision checked, USD/VND tracks Yahoo. Monthly series are dated at period end (published later: lag them). `gold_world_usd_adjusted` is the world gold price converted to thousand VND per lượng, not USD/oz; `gold_sjc_domestic_vnd` is thousand VND per lượng. |
| Vietnam Q3-2026 results | `vn_listed_q3_2026_financials` | WiData snapshot | SNAPSHOT 2026-10-21 | One quarter only (192 companies). |
| Raw discussion and announcements | `global_vip_social_posts_raw`, `global_hackernews_finance_posts`, `global_stackexchange_finance_questions`, `global_mastodon_finance_posts`, `us_fed_communications`, `us_sec_8k_filings` | Truth Social archive, Algolia, Stack Exchange, Mastodon, federalreserve.gov, SEC EDGAR | LIVE | Raw text and counts only; no sentiment labels. Hashtag and keyword selection is non-random. Mastodon exposes only the most recent ~2.5 years and the sample is small. Set `HYPOQ_CONTACT` to your own email for SEC fair access. |

**Excluded on purpose**: Reddit and X (an account is required; `www.reddit.com` is unreachable anonymously), paid fundamentals, consensus estimates, ETF NAV/AUM and creation baskets (no free real source),
Vietnamese commodity prices (the earlier series was model-constructed, not observed).

---

## 2. Standardized Dataset Specifications (Single-Table Format)

Every dataset is documented in an identical, single-table format with standardized schema columns:
`| Column Name | Type | Unit | Description |`

### 2.1 Macroeconomic & Commodity Data (`data/macro/`)

#### A. `vn_monetary_daily.parquet`
* **Coverage**: 2007-07-03 to 2026-10-21 (5,471 daily observations, 16 columns).
* **Source**: WiData (WiGroup), State Bank of Vietnam (SBV), Commercial Banks. [https://widata.vn/vi-mo/vn/pmi-san-xuat](https://widata.vn/vi-mo/vn/pmi-san-xuat) | [https://www.sbv.gov.vn](https://www.sbv.gov.vn).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `gold_world_usd_adjusted` | `Float64` | '000 VND/lượng | World gold price converted to the domestic unit and USD/VND (name kept for compatibility) |
  | `gold_sjc_domestic_vnd` | `Float64` | '000 VND/lượng | Domestic SJC gold buying price per lượng (tael) |
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

#### B. `vn_macro_monthly.parquet`
* **Coverage**: 2001-12-31 to 2026-09-30 (298 monthly observations, 13 columns).
* **Source**: General Statistics Office (GSO), S&P Global / WiData. [https://widata.vn/vi-mo/vn/pmi-san-xuat](https://widata.vn/vi-mo/vn/pmi-san-xuat).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
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

#### D. `vn_trade_exports_monthly.parquet`
* **Coverage**: 2009-01-31 to 2026-09-30 (213 monthly observations across steel, coffee, and total exports, 8 columns).
* **Source**: General Statistics Office & Ministry of Industry and Trade via WiData. [https://widata.vn](https://widata.vn).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `coffee_export_value_usd` | `Float64` | Billion USD | Coffee export turnover value (Cà phê) |
  | `total_export_value_usd` | `Float64` | Billion USD | Total merchandise export turnover value of Vietnam |
  | `steel_export_value_usd` | `Float64` | Billion USD | Iron and steel export turnover value (Sắt thép) |
  | `steel_products_export_value_usd` | `Float64` | Billion USD | Products from iron and steel export turnover value (Sản phẩm từ sắt thép) |
  | `total_export_growth_yoy` | `Float64` | % YoY | Total merchandise export value YoY growth percentage |
  | `steel_export_growth_yoy` | `Float64` | % YoY | Iron and steel export value YoY growth percentage |
  | `steel_products_export_growth_yoy` | `Float64` | % YoY | Products from iron and steel export value YoY growth percentage |

#### E. `vn_fdi_sectors_monthly.parquet`
* **Coverage**: 2011-12-31 to 2026-08-31 (173 monthly observations across 4 economic sectors, 5 columns).
* **Source**: Ministry of Planning and Investment (MPI) & General Statistics Office via WiData. [https://widata.vn](https://widata.vn).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `fdi_real_estate_ttm_usd` | `Float64` | Billion USD | Trailing 12-month realized FDI in Real Estate Activities |
  | `fdi_manufacturing_ttm_usd` | `Float64` | Billion USD | Trailing 12-month realized FDI in Manufacturing & Processing |
  | `fdi_utilities_energy_ttm_usd` | `Float64` | Billion USD | Trailing 12-month realized FDI in Electricity, Gas & Water |
  | `fdi_science_technology_ttm_usd` | `Float64` | Billion USD | Trailing 12-month realized FDI in Professional Science & Technology |

#### F. `vn_sbv_policy_rates_daily.parquet`
* **Coverage**: 2011-01-03 to 2026-10-21 (4,429 daily policy rate and required reserve observations, 6 columns).
* **Source**: State Bank of Vietnam (SBV / NHNN) Policy Rate Bulletins. [https://www.sbv.gov.vn](https://www.sbv.gov.vn).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `discount_rate` | `Float64` | - | Reported field `discount_rate` |
  | `refinancing_rate` | `Float64` | - | Reported field `refinancing_rate` |
  | `capital_shortfall_lending_rate` | `Float64` | % / year | SBV lending rate for capital shortfall in interbank clearing |
  | `required_reserve_rate_fx` | `Float64` | % / year | SBV interest rate on required reserve deposits in Foreign Currency |
  | `required_reserve_rate_vnd` | `Float64` | % / year | SBV interest rate on required reserve deposits in VND |

#### G. `us_macro_daily.parquet`
* **Coverage**: 1954-07-01 to 2026-10-24 (26,414 observations, 10 columns).
* **Source**: Federal Reserve Bank of St. Louis (FRED), US Department of the Treasury. [https://fred.stlouisfed.org](https://fred.stlouisfed.org).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `us_fed_total_assets_walcl` | `Float64` | Million USD | Federal Reserve Total Assets (Less Eliminations from Consolidation) |
  | `us_fed_funds_rate_dff` | `Float64` | % / year | Effective Federal Funds Rate (DFF) |
  | `us_sofr_volume` | `Float64` | Billion USD | Secured Overnight Financing Rate (SOFR) daily transaction volume |
  | `us_effr_volume` | `Float64` | Billion USD | Effective Federal Funds Rate daily transaction volume |
  | `us_fed_target_upper` | `Float64` | % / year | Federal Open Market Committee target policy rate upper bound |
  | `us_treasury_2y` | `Float64` | % / year | US 2-Year Treasury Constant Maturity Yield |
  | `us_treasury_10y` | `Float64` | % / year | US 10-Year Treasury Constant Maturity Yield |
  | `us_treasury_3m` | `Float64` | % / year | US 3-Month Treasury Bill Secondary Market Rate |
  | `us_m2_money_supply_yoy` | `Float64` | % YoY | US M2 Money Stock YoY Growth Rate |

#### H. `us_fed_balance_sheet_weekly.parquet`
* **Coverage**: 2002-12-18 to 2026-10-21 (1,245 weekly observations of Federal Reserve assets, loans, and repo, 7 columns).
* **Source**: Federal Reserve Board H.4.1 Factors Affecting Reserve Balances via FRED. [https://fred.stlouisfed.org](https://fred.stlouisfed.org).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `fed_total_assets_walcl` | `Float64` | - | Reported field `fed_total_assets_walcl` |
  | `fed_treasury_securities_wshotsl` | `Float64` | Million USD | Treasury Securities Held Outright by the Federal Reserve |
  | `fed_mbs_securities_wshomcb` | `Float64` | Million USD | Mortgage-Backed Securities (MBS) Held Outright by the Federal Reserve |
  | `fed_loans_total_wdtgal` | `Float64` | Million USD | Total Loans Extended by the Federal Reserve (Discount Window, Credit Facilities) |
  | `fed_reverse_repo_wlrral` | `Float64` | Million USD | Reverse Repurchase Agreements (RRP) Liabilities |
  | `fed_commercial_bank_assets_wcblsa` | `Float64` | Billion USD | Total Commercial Bank Assets in the United States |

#### I. `vn_market_pe_daily.parquet`
* **Coverage**: 2015-05-04 to 2026-10-21 (2,877 daily observations, 3 columns).
* **Source**: WiData / Ho Chi Minh Stock Exchange (HOSE). [https://widata.vn](https://widata.vn).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
  | `pe` | `Float64` | Multiple | Trailing 12-month Price-to-Earnings valuation multiple |

---

#### J. `us_fed_fomc_rate_decisions_pit.parquet`
* **Coverage**: 2016-01-27 to 2026-09-16 (88 FOMC statements, 10 columns).
* **Source**: derived by `scripts/build_fomc_decisions.py` from `us_fed_communications` (release time) and FRED `DFEDTARU` / `DFEDTARL` (target range).
* **Fields**: `meeting_date` (`Date`, statement release date), `announced_at` (`Datetime[ms]`, UTC, point in time), `effective_date` (next calendar day), `action` (`HIKE`, `CUT`, `HOLD`), `rate_change_bps` (change of the upper bound), `target_range_high_before`, `target_range_low`, `target_range_high` (percent, after the decision), `title`, `url`.

#### K. `vn_cpi_inflation_annual.parquet`
* **Coverage**: 1996-12-31 to 2025-12-31 (30 annual observations, 3 columns).
* **Source**: World Bank API, indicator `FP.CPI.TOTL.ZG`, refreshed by `scripts/ingest_vn_cpi_inflation_annual.py`.
* **Fields**: `year` (`Int32`), `cpi_inflation_pct` (`Float64`, % annual CPI inflation), `date` (`Date`, reference-period end: published months later, lag it).

### 2.2 Market & Derivatives Data (`data/market/`)

#### A. `vn_indices_daily.parquet`
* **Coverage**: 2017-08-24 to 2026-10-23 (4,574 daily OHLCV bars across VNINDEX and VN30, 8 columns).
* **Source**: VNDirect Open Market DChart API / HOSE. [https://dchart.vndirect.com.vn](https://dchart.vndirect.com.vn).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `timestamp` | `Datetime[ms]` | Unix Epoch (s) | POSIX epoch timestamp in seconds |
  | `open` | `Float64` | VND / Pts | Opening price or index level |
  | `high` | `Float64` | VND / Pts | Intraday high price or index level |
  | `low` | `Float64` | VND / Pts | Intraday low price or index level |
  | `close` | `Float64` | VND / Pts | Closing price or index level |
  | `volume` | `Float64` | Shares / Contracts | Total traded volume during the session |

#### B. `vn_vn30f1m_derivatives_daily.parquet`
* **Coverage**: 2017-08-10 to 2026-10-23 (2,298 daily OHLCV bars of VN30F1M Front-Month Futures, 8 columns).
* **Source**: Hanoi Stock Exchange (HNX) Derivatives Market via VNDirect. [https://dchart.vndirect.com.vn](https://dchart.vndirect.com.vn).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `timestamp` | `Datetime[ms]` | Unix Epoch (s) | POSIX epoch timestamp in seconds |
  | `open` | `Float64` | VND / Pts | Opening price or index level |
  | `high` | `Float64` | VND / Pts | Intraday high price or index level |
  | `low` | `Float64` | VND / Pts | Intraday low price or index level |
  | `close` | `Float64` | VND / Pts | Closing price or index level |
  | `volume` | `Float64` | Shares / Contracts | Total traded volume during the session |

#### C. `vn_top_equities_daily.parquet`
* **Coverage**: 2015-10-28 to 2026-10-23 (177,752 daily OHLCV bars across 71 liquid VN equities, 8 columns).
* **Source**: VNDirect Open Market DChart API. [https://dchart.vndirect.com.vn](https://dchart.vndirect.com.vn).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `timestamp` | `Datetime[ms]` | Unix Epoch (s) | POSIX epoch timestamp in seconds |
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
  | `listed_date` | `Date` | YYYY-MM-DD | Initial official public listing date |
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
* **Coverage**: 2017-08-24 to 2026-10-23 (26,124 daily OHLCV rows across 12 sector indices, 8 columns).
* **Source**: Ho Chi Minh Stock Exchange (HOSE) Official Sector Indices via VNDirect. [https://dchart.vndirect.com.vn](https://dchart.vndirect.com.vn).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `timestamp` | `Datetime[ms]` | Unix Epoch (s) | POSIX epoch timestamp in seconds |
  | `open` | `Float64` | VND / Pts | Opening price or index level |
  | `high` | `Float64` | VND / Pts | Intraday high price or index level |
  | `low` | `Float64` | VND / Pts | Intraday low price or index level |
  | `close` | `Float64` | VND / Pts | Closing price or index level |
  | `volume` | `Float64` | Shares / Contracts | Total traded volume during the session |

#### H. `vn_foreign_trading_daily.parquet`
* **Coverage**: 2018-08-30 to 2026-10-23 (130,902 daily rows across 65 leading equities, 12 columns).
* **Source**: VNDirect Financial Information (Finfo Open API) / Official Exchange Trades. [https://api-finfo.vndirect.com.vn](https://api-finfo.vndirect.com.vn).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
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
* **Coverage**: 2021-07-09 to 2026-10-23 (85,173 daily rows across 64 leading equities, 11 columns).
* **Source**: VNDirect Financial Information (Finfo Open API) / Trading Order Statistics. [https://api-finfo.vndirect.com.vn](https://api-finfo.vndirect.com.vn).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `floor` | `String` | - | Trading venue / stock exchange (HOSE, HNX, UPCOM) |
  | `bid_order` | `Float64` | Orders | Total number of buy orders placed during trading session |
  | `bid_vol` | `Float64` | Shares | Total buy order volume placed in the order book |
  | `offer_order` | `Float64` | Orders | Total number of sell orders placed during trading session |
  | `offer_vol` | `Float64` | Shares | Total sell order volume placed in the order book |
  | `avg_bid_vol` | `Float64` | Shares/order | Average volume per placed buy order (bid_vol / bid_order) |
  | `avg_offer_vol` | `Float64` | Shares/order | Average volume per placed sell order (offer_vol / offer_order) |
  | `net_vol` | `Float64` | Shares | Net order book volume imbalance: bid_vol - offer_vol (Dư mua/bán ròng) |
  | `order_imbalance` | `Float64` | [-1, +1] | Standardized order book imbalance: (bid_vol - offer_vol) / (bid_vol + offer_vol) |

#### J. `vn_etf_market_daily.parquet`
* **Coverage**: 2016-10-24 to 2026-10-23 (22,391 daily OHLCV bars across 15 domestic and global Vietnamese ETFs, 12 columns).
* **Source**: VNDirect Open Market DChart API, NYSE Arca & Taiwan Stock Exchange via Yahoo Finance. [https://dchart.vndirect.com.vn](https://dchart.vndirect.com.vn) | [https://finance.yahoo.com](https://finance.yahoo.com).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
  | `etf_name` | `String` | - | Official fund legal name in Vietnamese |
  | `benchmark_index` | `String` | - | Target tracking benchmark index (e.g. VN30, VN DIAMOND, VNFIN LEAD) |
  | `exchange` | `String` | - | Stock exchange (HOSE, HNX) |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `timestamp` | `Datetime[ms]` | Unix Epoch (s) | POSIX epoch timestamp in seconds |
  | `open` | `Float64` | VND / Pts | Opening price or index level |
  | `high` | `Float64` | VND / Pts | Intraday high price or index level |
  | `low` | `Float64` | VND / Pts | Intraday low price or index level |
  | `close` | `Float64` | VND / Pts | Closing price or index level |
  | `volume` | `Float64` | Shares / Contracts | Total traded volume during the session |
  | `value_vnd` | `Float64` | VND | Daily trading turnover value in VND |

---

### 2.3 Market Structure & Corporate Governance (`data/market_structure/`)

#### B. `vn_corporate_network_edges.parquet`
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
  | `effective_date` | `Date` | YYYY-MM-DD | Official effective date from which parameters/records take legal effect |
  | `weight` | `Float64` | [0, 1] | Normalized connection strength based on ownership percentage or governance authority |
  | `is_directed` | `Boolean` | Boolean | True indicating directed causal edge from owner/parent to entity |

#### C. `vn_corporate_network_nodes.parquet`
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

#### D. `vn_shareholders_history.parquet`
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
  | `effective_date` | `Date` | YYYY-MM-DD | Official effective date from which parameters/records take legal effect |

#### E. `vn_subsidiaries_affiliates.parquet`
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
  | `effective_date` | `Date` | YYYY-MM-DD | Official effective date from which parameters/records take legal effect |
  | `year` | `String` | Year | Fiscal reporting year |
  | `ownership_pct` | `Float64` | % | Parent group equity ownership percentage (%) in corporate entity |
  | `investment_capital` | `Float64` | VND | Contributed investment capital value |
  | `unit` | `String` | - | Currency denomination unit (VND) |

#### F. `vn_charter_capital_history.parquet`
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
  | `disclosure_date` | `Date` | YYYY-MM-DD | Official public disclosure announcement date |
  | `effective_date` | `Date` | YYYY-MM-DD | Official effective date from which parameters/records take legal effect |
  | `trading_start_date` | `Date` | YYYY-MM-DD | Official date when additional shares commence trading on the exchange |
  | `actual_date` | `Date` | YYYY-MM-DD | Date of completed execution or distribution |

#### G. `vn_free_float.parquet`
* **Coverage**: 2017-07-24 to 2026-08-03 (43,377 point-in-time rows across 1,450 tickers and 14 indices, 18 columns).
* **Source**: HOSE & HNX Point-in-Time Index Committee Official PDF Notices (Sourced via lazy-researcher). [https://www.hsx.vn](https://www.hsx.vn).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `time` | `Datetime[ms]` | ISO UTC | Point-in-time timestamp of announcement or disclosure |
  | `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
  | `index` | `String` | - | Index basket code (e.g. VN30, VN100, VNALLSHARE) |
  | `exchange` | `String` | - | Stock exchange (HOSE, HNX) |
  | `company_name` | `String` | - | Full corporate legal name in Vietnamese |
  | `effective_date` | `Date` | YYYY-MM-DD | Official effective date from which parameters/records take legal effect |
  | `index_shares` | `Int64` | Shares | Total outstanding shares considered for index calculation (Q) |
  | `free_float_factor` | `Float64` | Ratio (0-1] | Exchange-calculated free-float ratio (f) rounded to regulatory bands |
  | `capping_factor` | `Float64` | Ratio (0-1] | Single-stock capping factor (c) ensuring compliance with concentration limits |
  | `source_page` | `Int64` | - | Page number in official PDF announcement |
  | `source_url` | `String` | URL | Official exchange notice web URL |
  | `document_url` | `String` | URL | Direct download URL for official exchange PDF disclosure |
  | `source_sha256` | `String` | SHA-256 | Cryptographic hash of the source PDF document for audit reproducibility |
  | `announced_at` | `Datetime[ms]` | ISO UTC | Official public announcement timestamp |
  | `retrieved_at` | `Datetime[ms]` | ISO UTC | System retrieval and ingestion timestamp |
  | `availability_basis` | `String` | - | Point-in-Time availability justification (e.g. scheduled_notice) |
  | `quality` | `String` | - | Data quality verification level (e.g. official_complete) |
  | `publication_evidence_url` | `String` | URL | Archival verification URL for exchange publication |

#### H. `vn_vn30_free_float_history.parquet`
* **Coverage**: 2017-07-24 to 2026-08-03 (690 point-in-time constituent review rows for VN30, 18 columns).
* **Source**: HOSE Official Point-in-Time Review Notices. [https://www.hsx.vn](https://www.hsx.vn).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `time` | `Datetime[ms]` | ISO UTC | Point-in-time timestamp of announcement or disclosure |
  | `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
  | `index` | `String` | - | Index basket code (e.g. VN30, VN100, VNALLSHARE) |
  | `exchange` | `String` | - | Stock exchange (HOSE, HNX) |
  | `company_name` | `String` | - | Full corporate legal name in Vietnamese |
  | `effective_date` | `Date` | YYYY-MM-DD | Official effective date from which parameters/records take legal effect |
  | `index_shares` | `Int64` | Shares | Total outstanding shares considered for index calculation (Q) |
  | `free_float_factor` | `Float64` | Ratio (0-1] | Exchange-calculated free-float ratio (f) rounded to regulatory bands |
  | `capping_factor` | `Float64` | Ratio (0-1] | Single-stock capping factor (c) ensuring compliance with concentration limits |
  | `source_page` | `Int64` | - | Page number in official PDF announcement |
  | `source_url` | `String` | URL | Official exchange notice web URL |
  | `document_url` | `String` | URL | Direct download URL for official exchange PDF disclosure |
  | `source_sha256` | `String` | SHA-256 | Cryptographic hash of the source PDF document for audit reproducibility |
  | `announced_at` | `Datetime[ms]` | ISO UTC | Official public announcement timestamp |
  | `retrieved_at` | `Datetime[ms]` | ISO UTC | System retrieval and ingestion timestamp |
  | `availability_basis` | `String` | - | Point-in-Time availability justification (e.g. scheduled_notice) |
  | `quality` | `String` | - | Data quality verification level (e.g. official_complete) |
  | `publication_evidence_url` | `String` | URL | Archival verification URL for exchange publication |

#### I. `vn_free_float_changes_observed.parquet`
* **Coverage**: 10,178 discrete parameter adjustment events observed across index rebalancing cycles, 22 columns.
* **Source**: Point-in-Time Discrete Free-Float and Capping Adjustment Tracker. [https://www.hsx.vn](https://www.hsx.vn).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `time` | `Datetime[ms]` | ISO UTC | Point-in-time timestamp of announcement or disclosure |
  | `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
  | `index` | `String` | - | Index basket code (e.g. VN30, VN100, VNALLSHARE) |
  | `exchange` | `String` | - | Stock exchange (HOSE, HNX) |
  | `company_name` | `String` | - | Full corporate legal name in Vietnamese |
  | `effective_date` | `Date` | YYYY-MM-DD | Official effective date from which parameters/records take legal effect |
  | `index_shares` | `Int64` | Shares | Total outstanding shares considered for index calculation (Q) |
  | `free_float_factor` | `Float64` | Ratio (0-1] | Exchange-calculated free-float ratio (f) rounded to regulatory bands |
  | `capping_factor` | `Float64` | Ratio (0-1] | Single-stock capping factor (c) ensuring compliance with concentration limits |
  | `source_page` | `Int64` | - | Page number in official PDF announcement |
  | `source_url` | `String` | URL | Official exchange notice web URL |
  | `document_url` | `String` | URL | Direct download URL for official exchange PDF disclosure |
  | `source_sha256` | `String` | SHA-256 | Cryptographic hash of the source PDF document for audit reproducibility |
  | `announced_at` | `Datetime[ms]` | ISO UTC | Official public announcement timestamp |
  | `retrieved_at` | `Datetime[ms]` | ISO UTC | System retrieval and ingestion timestamp |
  | `availability_basis` | `String` | - | Point-in-Time availability justification (e.g. scheduled_notice) |
  | `quality` | `String` | - | Data quality verification level (e.g. official_complete) |
  | `publication_evidence_url` | `String` | URL | Archival verification URL for exchange publication |
  | `previous_factor` | `Float64` | Ratio (0-1] | Preceding free-float factor prior to revision |
  | `previous_snapshot_date` | `Date` | YYYY-MM-DD | Date of previous parameter snapshot |
  | `previous_source_url` | `String` | URL | URL of previous official announcement |
  | `change_date_precision` | `String` | - | Resolution precision of effective date (e.g. day) |

#### K. `vn_basket_history.parquet`
* **Coverage**: 261 basket rebalancing events and official effective schedules, 10 columns.
* **Source**: HOSE Quarterly and Semi-Annual Basket Rebalancing Schedule. [https://www.hsx.vn](https://www.hsx.vn).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `time` | `Datetime[ms]` | ISO UTC | Point-in-time timestamp of announcement or disclosure |
  | `symbol` | `String` | - | Official stock ticker or market benchmark symbol (e.g. VCB, VNINDEX, VN30F1M) |
  | `basket` | `String` | - | Basket identifier |
  | `effective_date` | `Date` | YYYY-MM-DD | Official effective date from which parameters/records take legal effect |
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
  | `fiscal_date` | `Date` | YYYY-MM-DD | Accounting period closing balance sheet date |
  | `year` | `Int32` | Year | Fiscal reporting year |
  | `quarter` | `Int32` | Quarter (1-4, 0) | Fiscal quarter (1, 2, 3, 4; 0 = Full Year Audited Annual Report) |
  | `report_type` | `String` | - | Regulatory period term (`QUARTER`, `QUARTER2` alternate/restated quarter, `ANNUAL`, `ANNUAL2`, `ESTIMATION` analyst forecast: never mix them) |
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
  | `report_date` | `Date` | YYYY-MM-DD | Official report filing date or ratio calculation timestamp |
  | `year` | `Int32` | Year | Fiscal reporting year |
  | `quarter` | `Int32` | Quarter (1-4, 0) | Fiscal quarter (1, 2, 3, 4; 0 = Full Year Audited Annual Report) |
  | `ratio_code` | `String` | - | Standardized fundamental ratio code (e.g. ROAE_TR_AVG5Q, PRICE_TO_EARNINGS) |
  | `ratio_name` | `String` | - | Standard Vietnamese title of the financial metric or ratio |
  | `category` | `String` | - | Ratio financial category (PROFITABILITY, VALUATION, LEVERAGE, LIQUIDITY, EFFICIENCY) |
  | `value` | `Float64` | Numeric | Reported observation value in original units |

#### D. `vn_listed_q3_2026_financials.parquet`
* **Coverage**: 192 listed companies on HOSE, HNX, and UPCOM for Q3-2026, 20 columns.
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
  | `report_date` | `Date` | YYYY-MM-DD | Official report filing date or ratio calculation timestamp |
  | `full_count` | `String` | - | Record counter string |

---

### 2.5 Global Macro & Cross-Asset Benchmarks (`data/cross_asset/`)

#### A. `global_macro_daily.parquet`
* **Coverage**: 2016-10-20 to 2026-10-21 (26,358 daily records across 10 global assets, 8 columns).
* **Source**: Yahoo Finance Multi-Asset Pricing Engine. [https://finance.yahoo.com](https://finance.yahoo.com).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `ticker` | `String` | - | Asset ticker symbol (e.g. ^GSPC, ^TNX, DX-Y.NYB) |
  | `asset_name` | `String` | - | Canonical financial asset name (e.g. sp500, gold_futures) |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `open` | `Float64` | VND / Pts | Opening price or index level |
  | `high` | `Float64` | VND / Pts | Intraday high price or index level |
  | `low` | `Float64` | VND / Pts | Intraday low price or index level |
  | `close` | `Float64` | VND / Pts | Closing price or index level |
  | `volume` | `Float64` | Shares / Contracts | Total traded volume during the session |


### 2.6 Global Multi-Asset, Regional Equities, Macro & Crypto Datasets

#### A. `us_indices_daily.parquet` (`data/market/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (15,086 daily records across S&P 500, Nasdaq 100, Dow Jones, Russell 2000, MidCap, and VIX, 11 columns).
* **Source**: Yahoo Finance Benchmark Index Engine.
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `symbol` | `String` | - | Benchmark ticker symbol (`^GSPC`, `^NDX`, `^DJI`, `^RUT`, `^MID`, `^VIX`) |
  | `open` | `Float64` | Index Points | Trading session opening price |
  | `high` | `Float64` | Index Points | Daily session high price |
  | `low` | `Float64` | Index Points | Daily session low price |
  | `close` | `Float64` | Index Points | Daily session closing level |
  | `adj_close` | `Float64` | Index Points | Dividend- and split-adjusted closing level |
  | `volume` | `Float64` | Shares | Total session trading volume |
  | `returns` | `Float64` | Ratio | Daily percentage return |
  | `index_name` | `String` | - | Full canonical index name |
  | `country` | `String` | - | Geographic jurisdiction (`US`) |

#### B. `us_equities_daily.parquet` (`data/market/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (87,990 daily records across 15 US mega-cap sector leaders, 12 columns).
* **Source**: Yahoo Finance Listed Equity Data.
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `symbol` | `String` | - | US stock ticker (e.g. `AAPL`, `MSFT`, `NVDA`, `AMZN`, `JPM`) |
  | `open` | `Float64` | USD | Opening share price |
  | `high` | `Float64` | USD | Intraday high share price |
  | `low` | `Float64` | USD | Intraday low share price |
  | `close` | `Float64` | USD | Closing share price |
  | `adj_close` | `Float64` | USD | Split- and dividend-adjusted closing price |
  | `volume` | `Float64` | Shares | Daily volume of shares traded |
  | `returns` | `Float64` | Ratio | Daily discrete return |
  | `company_name` | `String` | - | Registered corporate entity name |
  | `sector` | `String` | - | GICS economic sector classification |
  | `country` | `String` | - | Domicile country (`US`) |

#### C. `eu_indices_daily.parquet` (`data/market/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (20,296 daily records across 8 European benchmark indices, 11 columns).
* **Source**: Euronext, Deutsche Börse, London Stock Exchange, BME, SIX Swiss Exchange via Yahoo Finance.
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `symbol` | `String` | - | Index ticker (`^STOXX50E`, `^GDAXI`, `^FTSE`, `^FCHI`, `^IBEX`, `FTSEMIB.MI`, `^SSMI`, `^AEX`) |
  | `open` | `Float64` | Index Points | Opening index level |
  | `high` | `Float64` | Index Points | Intraday high level |
  | `low` | `Float64` | Index Points | Intraday low level |
  | `close` | `Float64` | Index Points | Closing index level |
  | `adj_close` | `Float64` | Index Points | Adjusted closing index level |
  | `volume` | `Float64` | Shares | Traded index constituents volume |
  | `returns` | `Float64` | Ratio | Daily percentage return |
  | `index_name` | `String` | - | Canonical European index name |
  | `country` | `String` | - | Country or region of origin |

#### D. `eu_equities_daily.parquet` (`data/market/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (50,599 daily records across 11 European blue-chips, 12 columns).
* **Source**: Yahoo Finance European Exchange Listings.
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `symbol` | `String` | - | Equity ticker (e.g. `ASML`, `SAP`, `SHEL`, `NVO`, `AZN`, `MC.PA`) |
  | `open` | `Float64` | EUR / GBP / CHF | Opening equity price |
  | `high` | `Float64` | EUR / GBP / CHF | Session high price |
  | `low` | `Float64` | EUR / GBP / CHF | Session low price |
  | `close` | `Float64` | EUR / GBP / CHF | Session closing price |
  | `adj_close` | `Float64` | EUR / GBP / CHF | Split- and dividend-adjusted price |
  | `volume` | `Float64` | Shares | Number of shares traded |
  | `returns` | `Float64` | Ratio | Daily return |
  | `company_name` | `String` | - | Corporate name |
  | `sector` | `String` | - | Industry sector |
  | `country` | `String` | - | Domicile country |

#### E. `asia_indices_daily.parquet` (`data/market/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (22,182 daily records across 10 major Asian benchmarks, 11 columns).
* **Source**: Tokyo Stock Exchange, HKEX, Shanghai, Shenzhen, KRX, TWSE, NSE, SGX, ASX via Yahoo Finance.
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `symbol` | `String` | - | Benchmark ticker (`^N225`, `^HSI`, `000001.SS`, `000300.SS`, `^KS11`, `^TWII`, `^NSEI`, `^BSESN`, `^STI`, `^AXJO`) |
  | `open` | `Float64` | Index Points | Opening index level |
  | `high` | `Float64` | Index Points | Session high level |
  | `low` | `Float64` | Index Points | Session low level |
  | `close` | `Float64` | Index Points | Closing index level |
  | `adj_close` | `Float64` | Index Points | Adjusted closing index level |
  | `volume` | `Float64` | Shares | Market constituent volume |
  | `returns` | `Float64` | Ratio | Daily percentage return |
  | `index_name` | `String` | - | Index name |
  | `country` | `String` | - | Asian country / jurisdiction |

#### F. `asia_equities_daily.parquet` (`data/market/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (22,457 daily records across 9 Asian corporate titans, 12 columns).
* **Source**: Yahoo Finance Asian Listings and ADRs.
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `symbol` | `String` | - | Ticker (e.g. `TSM`, `BABA`, `TCEHY`, `005930.KS`, `7203.T`, `HDB`, `INFY`, `SONY`, `9984.T`) |
  | `open` | `Float64` | Local Currency | Opening price |
  | `high` | `Float64` | Local Currency | Daily high price |
  | `low` | `Float64` | Local Currency | Daily low price |
  | `close` | `Float64` | Local Currency | Closing price |
  | `adj_close` | `Float64` | Local Currency | Adjusted closing price |
  | `volume` | `Float64` | Shares | Total volume |
  | `returns` | `Float64` | Ratio | Daily percentage return |
  | `company_name` | `String` | - | Corporate name |
  | `sector` | `String` | - | Economic sector |
  | `country` | `String` | - | Headquarters jurisdiction |

#### G. `global_etfs_daily.parquet` (`data/market/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (42,738 daily records across 17 cross-asset and regional ETFs, 11 columns).
* **Source**: Yahoo Finance ETF Database.
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `symbol` | `String` | - | ETF ticker (`SPY`, `QQQ`, `IWM`, `EFA`, `EEM`, `ACWI`, `FXI`, `EWJ`, `EWG`, `EWU`, `INDA`, `TLT`, `IEF`, `LQD`, `HYG`, `GLD`, `VNQ`) |
  | `open` | `Float64` | USD | Opening ETF price |
  | `high` | `Float64` | USD | Session high price |
  | `low` | `Float64` | USD | Session low price |
  | `close` | `Float64` | USD | Session close price |
  | `adj_close` | `Float64` | USD | Distribution-adjusted close |
  | `volume` | `Float64` | Shares | ETF volume traded |
  | `returns` | `Float64` | Ratio | Daily percentage return |
  | `etf_name` | `String` | - | Full fund name |
  | `asset_class` | `String` | - | Target asset class / exposure |

#### H. `global_commodities_daily.parquet` (`data/macro/`)
* **Coverage**: 2016-10-25 to 2026-10-23 (35,195 daily records across 14 global energy, metals, and agricultural futures, 11 columns).
* **Source**: NYMEX, COMEX, CBOT, ICE via Yahoo Finance.
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `symbol` | `String` | - | Futures contract symbol (`CL=F`, `BZ=F`, `NG=F`, `RB=F`, `GC=F`, `SI=F`, `HG=F`, `PL=F`, `ZC=F`, `ZW=F`, `ZS=F`, `KC=F`, `SB=F`, `CT=F`) |
  | `open` | `Float64` | USD / contract unit | Opening contract price |
  | `high` | `Float64` | USD / contract unit | Intraday high |
  | `low` | `Float64` | USD / contract unit | Intraday low |
  | `close` | `Float64` | USD / contract unit | Settle / closing price |
  | `adj_close` | `Float64` | USD / contract unit | Adjusted settlement price |
  | `volume` | `Float64` | Contracts | Volume of futures contracts traded |
  | `returns` | `Float64` | Ratio | Daily return |
  | `commodity_name` | `String` | - | Canonical commodity name |
  | `sector` | `String` | - | Category (`Energy`, `Precious Metals`, `Industrial Metals`, `Agriculture`, `Soft Commodities`) |

#### I. `us_yield_curve_daily.parquet` (`data/macro/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (2,516 daily records, 8 columns).
* **Source**: US Department of the Treasury / CBOE via Yahoo Finance.
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `tbill_3m` | `Float64` | % / year | 13-Week / 3-Month US Treasury Bill Yield (`^IRX`) |
  | `tnote_5y` | `Float64` | % / year | 5-Year US Treasury Note Benchmark Yield (`^FVX`) |
  | `tnote_10y` | `Float64` | % / year | 10-Year US Treasury Benchmark Yield (`^TNX`) |
  | `tbond_30y` | `Float64` | % / year | 30-Year US Treasury Bond Benchmark Yield (`^TYX`) |
  | `yield_spread_10y_3m` | `Float64` | % points | Term spread slope ($y_{10Y} - y_{3M}$) |
  | `yield_spread_10y_5y` | `Float64` | % points | Term spread slope ($y_{10Y} - y_{5Y}$) |
  | `cboe_vix` | `Float64` | Index | CBOE S&P 500 Volatility Index level (`^VIX`) |

#### J. `eu_macro_daily.parquet` (`data/macro/`)
* **Coverage**: 2016-10-24 to 2026-10-25 (13,017 daily records across 5 European foreign exchange pairs, 11 columns).
* **Source**: European Central Bank / Interbank FX via Yahoo Finance.
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `symbol` | `String` | - | FX ticker (`EURUSD=X`, `GBPUSD=X`, `EURGBP=X`, `EURCHF=X`, `EURJPY=X`) |
  | `open` | `Float64` | Quote Currency | Opening exchange rate |
  | `high` | `Float64` | Quote Currency | Intraday high |
  | `low` | `Float64` | Quote Currency | Intraday low |
  | `close` | `Float64` | Quote Currency | Closing exchange rate |
  | `adj_close` | `Float64` | Quote Currency | Adjusted closing exchange rate |
  | `volume` | `Float64` | Units | Quoted volume |
  | `returns` | `Float64` | Ratio | Daily percentage change |
  | `currency_pair` | `String` | - | Canonical currency pair name |
  | `region` | `String` | - | Regional classification |

#### K. `asia_macro_daily.parquet` (`data/macro/`)
* **Coverage**: 2016-10-24 to 2026-10-25 (20,822 daily records across 8 Asian currencies vs USD, 11 columns).
* **Source**: Asian Central Banks & Interbank Foreign Exchange.
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `symbol` | `String` | - | FX ticker (`USDJPY=X`, `USDCNY=X`, `USDKRW=X`, `USDINR=X`, `USDSGD=X`, `USDHKD=X`, `AUDUSD=X`, `USDVND=X`) |
  | `open` | `Float64` | Local Currency / USD | Opening exchange rate |
  | `high` | `Float64` | Local Currency / USD | Session high rate |
  | `low` | `Float64` | Local Currency / USD | Session low rate |
  | `close` | `Float64` | Local Currency / USD | Closing exchange rate |
  | `adj_close` | `Float64` | Local Currency / USD | Adjusted closing exchange rate |
  | `volume` | `Float64` | Units | Daily volume |
  | `returns` | `Float64` | Ratio | Daily percentage change |
  | `currency_pair` | `String` | - | Currency pair identifier |
  | `country` | `String` | - | Sovereign jurisdiction |

#### L. `crypto_top_assets_daily.parquet` (`data/crypto/`)
* **Coverage**: 2016-10-25 to 2026-10-25 (32,369 daily records across 11 major cryptocurrencies, 12 columns).
* **Source**: CoinMarketCap / Binance / Coinbase aggregate via Yahoo Finance.
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | 24/7 calendar date (ISO 8601) |
  | `symbol` | `String` | - | Pair ticker (`BTC-USD`, `ETH-USD`, `SOL-USD`, `BNB-USD`, `XRP-USD`, `ADA-USD`, `DOGE-USD`, `AVAX-USD`, `LINK-USD`, `DOT-USD`, `NEAR-USD`) |
  | `open` | `Float64` | USD | Opening price at 00:00 UTC |
  | `high` | `Float64` | USD | 24h session high |
  | `low` | `Float64` | USD | 24h session low |
  | `close` | `Float64` | USD | 23:59:59 UTC closing price |
  | `adj_close` | `Float64` | USD | Adjusted price |
  | `volume` | `Float64` | USD | Total 24h traded volume |
  | `returns` | `Float64` | Ratio | 24h discrete percentage return |
  | `asset_name` | `String` | - | Asset name (e.g. Bitcoin, Ethereum) |
  | `category` | `String` | - | Architectural role (e.g. Smart Contract L1, Store of Value) |
  | `volatility_30d` | `Float64` | % / year | 30-day annualized realized volatility |

#### M. `crypto_market_daily.parquet` (`data/crypto/`)
* **Coverage**: 2017-11-09 to 2026-10-25 (3,273 daily records, 9 columns).
* **Source**: Cross-Asset Derived Market Metrics.
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Calendar date (ISO 8601) |
  | `btc_close` | `Float64` | USD | Bitcoin USD closing price |
  | `eth_close` | `Float64` | USD | Ethereum USD closing price |
  | `eth_btc_ratio` | `Float64` | Ratio | Relative strength ratio ($P_{\text{ETH}} / P_{\text{BTC}}$) |
  | `crypto_top_volume_sum` | `Float64` | USD | Combined BTC + ETH daily trading volume proxy |
  | `btc_vol_30d` | `Float64` | % / year | Bitcoin 30-day annualized realized volatility |
  | `eth_vol_30d` | `Float64` | % / year | Ethereum 30-day annualized realized volatility |
  | `btc_sp500_corr_30d` | `Float64` | Correlation (-1 to 1) | 30-day rolling correlation between BTC and S&P 500 |
  | `btc_gold_corr_30d` | `Float64` | Correlation (-1 to 1) | 30-day rolling correlation between BTC and Gold futures |

### 2.7 Country-Specific Financial Markets (China, Hong Kong, Japan, South Korea)

#### A. `jp_indices_daily.parquet` (`data/market/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (2,442 daily records, 11 columns).
* **Source**: Tokyo Stock Exchange via Yahoo Finance.
* **Fields**: `date` (`Date`), `symbol` (`String`), `open`, `high`, `low`, `close`, `adj_close`, `volume`, `returns` (`Float64`), `index_name` (`String`), `country` (`String`).

#### B. `jp_equities_daily.parquet` (`data/market/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (44,334 daily records across 10 Japanese corporate titans, 12 columns).
* **Source**: Tokyo Stock Exchange.
* **Fields**: `date` (`Date`), `symbol` (`String`), `open`, `high`, `low`, `close`, `adj_close`, `volume`, `returns` (`Float64`), `company_name` (`String`), `sector` (`String`), `country` (`String`).

#### C. `jp_macro_daily.parquet` (`data/macro/`)
* **Coverage**: 2016-10-24 to 2026-10-25 (5,206 daily records across USD/JPY and EUR/JPY, 11 columns).
* **Source**: Bank of Japan & Interbank FX.
* **Fields**: `date` (`Date`), `symbol` (`String`), `open`, `high`, `low`, `close`, `adj_close`, `volume`, `returns` (`Float64`), `currency_pair` (`String`), `country` (`String`).

#### D. `kr_indices_daily.parquet` (`data/market/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (4,896 daily records across KOSPI and KOSDAQ, 11 columns).
* **Source**: Korea Exchange (KRX) via Yahoo Finance.
* **Fields**: `date` (`Date`), `symbol` (`String`), `open`, `high`, `low`, `close`, `adj_close`, `volume`, `returns` (`Float64`), `index_name` (`String`), `country` (`String`).

#### E. `kr_equities_daily.parquet` (`data/market/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (37,845 daily records across 9 South Korean chaebol leaders, 12 columns).
* **Source**: Korea Exchange (KRX).
* **Fields**: `date` (`Date`), `symbol` (`String`), `open`, `high`, `low`, `close`, `adj_close`, `volume`, `returns` (`Float64`), `company_name` (`String`), `sector` (`String`), `country` (`String`).

#### F. `kr_macro_daily.parquet` (`data/macro/`)
* **Coverage**: 2016-10-24 to 2026-10-25 (2,603 daily records, 11 columns).
* **Source**: Bank of Korea & Interbank FX.
* **Fields**: `date` (`Date`), `symbol` (`String`), `open`, `high`, `low`, `close`, `adj_close`, `volume`, `returns` (`Float64`), `currency_pair` (`String`), `country` (`String`).

#### G. `hk_indices_daily.parquet` (`data/market/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (2,461 daily records, 11 columns).
* **Source**: Hong Kong Exchanges and Clearing (HKEX).
* **Fields**: `date` (`Date`), `symbol` (`String`), `open`, `high`, `low`, `close`, `adj_close`, `volume`, `returns` (`Float64`), `index_name` (`String`), `country` (`String`).

#### H. `hk_equities_daily.parquet` (`data/market/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (34,436 daily records across 9 Hong Kong blue-chips, 12 columns).
* **Source**: Hong Kong Exchanges and Clearing (HKEX).
* **Fields**: `date` (`Date`), `symbol` (`String`), `open`, `high`, `low`, `close`, `adj_close`, `volume`, `returns` (`Float64`), `company_name` (`String`), `sector` (`String`), `country` (`String`).

#### I. `hk_macro_daily.parquet` (`data/macro/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (2,602 daily records, 11 columns).
* **Source**: Hong Kong Monetary Authority (HKMA) Linked Exchange Rate System.
* **Fields**: `date` (`Date`), `symbol` (`String`), `open`, `high`, `low`, `close`, `adj_close`, `volume`, `returns` (`Float64`), `currency_pair` (`String`), `country` (`String`).

#### J. `cn_indices_daily.parquet` (`data/market/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (2,427 daily records across SSE Composite and CSI 300, 11 columns).
* **Source**: Shanghai and Shenzhen Stock Exchanges.
* **Fields**: `date` (`Date`), `symbol` (`String`), `open`, `high`, `low`, `close`, `adj_close`, `volume`, `returns` (`Float64`), `index_name` (`String`), `country` (`String`).

#### K. `cn_equities_daily.parquet` (`data/market/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (41,241 daily records across 10 Chinese corporate leaders, 12 columns).
* **Source**: SSE, SZSE, and US ADR listings.
* **Fields**: `date` (`Date`), `symbol` (`String`), `open`, `high`, `low`, `close`, `adj_close`, `volume`, `returns` (`Float64`), `company_name` (`String`), `sector` (`String`), `country` (`String`).

#### L. `cn_macro_daily.parquet` (`data/macro/`)
* **Coverage**: 2016-10-24 to 2026-10-25 (2,602 daily records, 11 columns).
* **Source**: People's Bank of China (PBoC) CFETS Reference Rates.
* **Fields**: `date` (`Date`), `symbol` (`String`), `open`, `high`, `low`, `close`, `adj_close`, `volume`, `returns` (`Float64`), `currency_pair` (`String`), `country` (`String`).

---

### 2.8 Raw Community & Social Discussions (`data/community/`)

All files are downloaded from free, key-less public sources by `scripts/ingest_community_and_social.py` and contain raw text, timestamps and engagement counts only. No sentiment labels, polarity scores or return reactions are stored: agents derive those themselves. Re-running the script refreshes every file. Reddit and X block anonymous access and are therefore not included.

#### A. `global_vip_social_posts_raw.parquet`
* **Coverage**: 2022-02-14 onwards (37,251 posts, 11 columns).
* **Source**: Public Truth Social archive of Donald Trump posts (CNN open data mirror).
* **Fields**: `datetime_utc` (`Datetime[ms]`, UTC), `author`, `platform`, `post_id`, `text`, `url`, `has_media` (`Boolean`), `replies_count`, `reposts_count`, `likes_count` (`Int64`), `date` (`Date`).

#### B. `global_hackernews_finance_posts.parquet`
* **Coverage**: 2015-01-01 onwards (5,780 stories with at least 10 points, 10 columns).
* **Source**: Hacker News via the Algolia search API; queries: federal reserve, inflation, interest rates, recession, tariffs, bitcoin, stock market, treasury yields, oil price, bank run.
* **Fields**: `datetime_utc`, `post_id`, `author`, `title`, `body_text`, `url`, `points`, `num_comments`, `matched_queries` (comma-separated), `date`.

#### C. `global_stackexchange_finance_questions.parquet`
* **Coverage**: 2009-2026 (4,500 top-voted questions from `money`, `quant`, `economics`, 11 columns).
* **Source**: Stack Exchange public API v2.3.
* **Fields**: `datetime_utc`, `site`, `post_id`, `title`, `body_text` (HTML stripped), `tags` (comma-separated), `score`, `view_count`, `answer_count`, `url`, `date`.

#### D. `global_mastodon_finance_posts.parquet`
* **Coverage**: 2024-05-01 to 2026-10-25 (9,653 posts from the hashtags `stocks, economy, inflation, bitcoin, crypto, markets, fed, trading, investing` on mastodon.social, 12 columns).
* **Source**: Mastodon public hashtag timeline API (no key).
* **Fields**: `datetime_utc`, `post_id`, `author`, `language`, `text` (HTML stripped), `url`, `replies_count`, `reposts_count`, `likes_count`, `tags_matched` (comma-separated queried tags), `hashtags` (all tags on the post), `date`.

#### E. `us_fed_communications.parquet`
* **Coverage**: 2016-01-03 to 2026-10-15 (1,214 documents, 8 columns).
* **Source**: federalreserve.gov JSON feeds (`ne-press.json` filtered to Monetary Policy, `ne-speeches.json`) and the document pages (full text).
* **Fields**: `datetime_utc` (`Datetime[ms]`, official release time converted from US/Eastern to UTC), `doc_type` (`press_release` for FOMC statements, minutes and implementation notes; `speech` for speeches and testimony), `title`, `speaker`, `venue` (speeches only), `url`, `text`, `date`.

#### F. `us_sec_8k_filings.parquet`
* **Coverage**: 2016-01-04 to 2026-10-23 (5,664 filings by 35 US mega-caps, 10 columns).
* **Source**: SEC EDGAR submissions API (`data.sec.gov`).
* **Fields**: `symbol`, `cik`, `form` (`8-K` or `8-K/A`), `accession_number`, `report_date` (event date stated in the filing, may be null), `items` (comma-separated 8-K item codes, e.g. `2.02,9.01` = results of operations), `document_description`, `url`, `datetime_utc` (EDGAR acceptance time in UTC: the moment the filing became public), `date`.

### 2.9 World Financial Data Taxonomy: Macro, GICS Sectors, Factors, Credit & Fundamentals

#### A. `global_macro_indicators_monthly.parquet` (`data/macro/`)
* **Coverage**: 1948-01-01 to 2026-10-23 (13,540 continuous daily/monthly aligned records, 9 columns).
* **Source**: Federal Reserve Bank of St. Louis (FRED Public Open Data).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Calendar observation date (ISO 8601) |
  | `pce_headline_inflation_index` | `Float64` | Index | US Headline Personal Consumption Expenditures Price Index (`PCEPI`) |
  | `unemployment_rate_pct` | `Float64` | % | US Civilian Unemployment Rate (`UNRATE`) |
  | `chicago_fed_financial_conditions_index` | `Float64` | Index | National Financial Conditions Index (`NFCI`), 0=Neutral |
  | `treasury_10y_breakeven_inflation` | `Float64` | % / year | 10-Year Breakeven Inflation Rate (`T10YIE`) |
  | `term_spread_10y_2y` | `Float64` | % points | 10-Year Minus 2-Year Treasury Constant Maturity Spread (`T10Y2Y`) |
  | `m2_money_supply_billions` | `Float64` | Billion USD | US M2 Seasonally Adjusted Money Supply (`M2SL`) |
  | `public_debt_pct_gdp` | `Float64` | % of GDP | Federal Debt: Total Public Debt as Percent of Gross Domestic Product |
  | `high_yield_option_adjusted_spread_bps` | `Float64` | bps | ICE BofA US High Yield Index Option-Adjusted Spread (`BAMLH0A0HYM2`) |

#### B. `global_sectors_daily.parquet` (`data/market/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (27,239 daily records across all 11 official GICS sectors, 11 columns).
* **Source**: Select Sector SPDR Trust via Yahoo Finance.
* **Fields**: `date` (`Date`), `symbol` (`String` e.g. `XLK`, `XLF`, `XLV`, `XLE`, `XLY`, `XLP`, `XLI`, `XLB`, `XLU`, `XLRE`, `XLC`), `open`, `high`, `low`, `close`, `adj_close`, `volume`, `returns` (`Float64`), `sector_name` (`String`), `sector_code` (`String`).

#### C. `global_factors_daily.parquet` (`data/market/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (20,112 daily records across 8 smart beta and style factor benchmarks, 11 columns).
* **Source**: MSCI / iShares / Vanguard / ProShares via Yahoo Finance.
* **Fields**: `date` (`Date`), `symbol` (`String` e.g. `MTUM`, `QUAL`, `USMV`, `VLUE`, `VUG`, `IJR`, `IJH`, `NOBL`), `open`, `high`, `low`, `close`, `adj_close`, `volume`, `returns` (`Float64`), `factor_name` (`String`), `style_category` (`String`: Momentum, Quality, Low Vol, Value, Growth, Size, Dividend).

#### D. `global_credit_spreads_daily.parquet` (`data/macro/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (19,758 daily records across fixed income credit classes and maritime freight, 11 columns).
* **Source**: iShares / Breakwave Dry Bulk via Yahoo Finance.
* **Fields**: `date` (`Date`), `symbol` (`String` e.g. `LQD`, `HYG`, `EMB`, `TIP`, `SHY`, `TLT`, `MUB`, `BDRY`), `open`, `high`, `low`, `close`, `adj_close`, `volume`, `returns` (`Float64`), `asset_name` (`String`), `asset_category` (`String`).

#### E. `global_cross_asset_volatility_daily.parquet` (`data/macro/`)
* **Coverage**: 2016-10-24 to 2026-10-23 (2,516 daily records, 5 columns).
* **Source**: Chicago Board Options Exchange (CBOE).
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `date` | `Date` | YYYY-MM-DD | Trading calendar date (ISO 8601) |
  | `vix_equity` | `Float64` | Index | CBOE S&P 500 Volatility Index level (`^VIX`) |
  | `vvix_vol_of_vol` | `Float64` | Index | CBOE VIX of VIX Volatility Index (`^VVIX`) |
  | `ovx_crude_oil` | `Float64` | Index | CBOE Crude Oil ETF Volatility Index (`^OVX`) |
  | `gvz_gold` | `Float64` | Index | CBOE Gold ETF Volatility Index (`^GVZ`) |

#### F. `global_corporate_fundamentals.parquet` (`data/financials/`)
* **Coverage**: Comprehensive cross-sectional profiles across 50 global mega-cap corporations in the US, Europe, Japan, South Korea, China, and Hong Kong (50 rows, 20 columns).
* **Source**: S&P / SEC / Annual & Quarterly Corporate Disclosures via Yahoo Finance API.
* **Fields**:
  | Column Name | Type | Unit | Description |
  | :--- | :--- | :--- | :--- |
  | `symbol` | `String` | - | Primary equity ticker (e.g. `AAPL`, `NVDA`, `ASML`, `7203.T`, `005930.KS`, `0700.HK`) |
  | `company_name` | `String` | - | Full legal or short corporate name |
  | `country` | `String` | - | Sovereign jurisdiction of incorporation / headquarters |
  | `sector` | `String` | - | GICS economic sector classification |
  | `industry` | `String` | - | Specific industry group |
  | `market_cap` | `Float64` | Currency | Total market capitalization |
  | `enterprise_value` | `Float64` | Currency | Total enterprise value (Market Cap + Debt - Cash) |
  | `pe_trailing` | `Float64` | Multiple | Trailing 12-month Price-to-Earnings ratio |
  | `pe_forward` | `Float64` | Multiple | Forward 12-month consensus Price-to-Earnings multiple |
  | `pb_ratio` | `Float64` | Multiple | Price-to-Book value ratio |
  | `ps_ratio` | `Float64` | Multiple | Price-to-Sales trailing multiple |
  | `ev_to_ebitda` | `Float64` | Multiple | Enterprise Value to EBITDA ratio |
  | `profit_margin` | `Float64` | Ratio | Net profit margin |
  | `operating_margin` | `Float64` | Ratio | Operating operating income margin |
  | `roe` | `Float64` | Ratio | Return on Common Equity |
  | `roa` | `Float64` | Ratio | Return on Total Assets |
  | `debt_to_equity` | `Float64` | % | Total Debt to Shareholder Equity ratio |
  | `dividend_yield` | `Float64` | Ratio | Annual dividend yield |
  | `free_cashflow` | `Float64` | Currency | Trailing free cash flow |
  | `beta` | `Float64` | Coefficient | 5-year monthly market beta |
---

## 3. Querying Datasets: Polars vs. Pandas Implementation Solutions

### Solution A: Point-in-Time As-Of Join (Monthly Macro to Daily Equity Prices)
```python
# --- Polars (Zero-Copy Vectorized) ---
import polars as pl
df_idx = pl.read_parquet('data/market/vn_indices_daily.parquet').filter(pl.col('symbol') == 'VNINDEX')
df_macro = pl.read_parquet('data/macro/vn_macro_monthly.parquet')
aligned = df_idx.join_asof(
    df_macro,
    left_on=pl.col('date').str.to_date(),
    right_on=pl.col('date').str.to_date(),
    strategy='backward'
)

# --- Pandas Equivalent ---
import pandas as pd
df_idx = pd.read_parquet('data/market/vn_indices_daily.parquet').query("symbol == 'VNINDEX'")
df_macro = pd.read_parquet('data/macro/vn_macro_monthly.parquet')
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