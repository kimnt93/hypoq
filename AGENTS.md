# AGENTS.md — Research & Dataset Instructions for paper-hub & vnresearch

Welcome to **paper-hub**, the quantitative research and academic paper engineering workspace, and **vnresearch** ([`https://github.com/kimnt93/vnresearch.git`](https://github.com/kimnt93/vnresearch.git)), the autonomous quantitative data lake. This workspace hosts empirical datasets, modeling code, and LaTeX manuscripts targeting top peer-reviewed journals in Financial Data Science, Quantitative Economics, and Machine Learning.

---

## 1. Autonomous Prompt-Driven Agent Protocol

Whenever an agent receives the following user command:
```text
@AGENTS.md i want to download complete dataset upto now
```
**The agent MUST perform the following automated protocol without requiring manual intervention**:
1. Navigate to the repository root or `vnresearch/`.
2. Execute the master data download pipeline:
   ```bash
   make download
   ```
   *This command autonomously triggers WiData macro decryption, corporate financial report scraping, VNDirect market OHLCV updates, Yahoo Finance cross-asset sync, free-float factor updates, 4-tier ICB classifications, shareholder/subsidiary ingestion, financial statement extraction, order book microstructure, and corporate network graph generation.*
3. Execute the automated verification check:
   ```bash
   make verify
   ```
4. Output a concise delta report summarizing:
   - Total datasets audited (36 Parquet files)
   - Total observation rows (1,254,222+ rows)
   - Latest timestamp per market asset (e.g. `VNINDEX`, `VN30F1M`, `S&P 500`)
   - Any newly detected corporate events or rebalancing disclosures.

---

## 2. Complete Dataset Architecture & Inventory (36 Parquet Datasets)

All datasets are serialized in **Apache Parquet** format under `data/` (physically stored in `vnresearch/data/`):

```text
data/
├── macro/
│   ├── vietnam_monetary_daily.parquet           # 5,471 rows x 16 cols | Interbank rates, OMO, SBV rates, FX, Yields
│   ├── vietnam_macro_monthly.parquet            # 298 rows x 13 cols | PMI, CPI components, IIP, Retail sales, FDI
│   ├── vietnam_commodities_daily.parquet        # 2,285 rows x 17 cols | Construction Steel, Pork, Fuels, Coffee, Rice
│   ├── vietnam_trade_exports_monthly.parquet    # 213 rows x 8 cols | Merchandise Exports (Steel, Coffee, Total)
│   ├── vietnam_fdi_sectors_monthly.parquet      # 173 rows x 5 cols | Realized FDI by Sector (Manufacturing, RE)
│   ├── vietnam_sbv_policy_rates_daily.parquet   # 4,429 rows x 6 cols | SBV Policy Rates & Required Reserves
│   ├── us_macro_daily.parquet                   # 13,821 rows x 10 cols | Fed Funds DFF, 3M/2Y/10Y Yields, M2, WALCL
│   ├── us_fed_balance_sheet_weekly.parquet      # 2,128 rows x 7 cols | Federal Reserve Assets, Loans, MBS & RRP
│   └── vietnam_market_pe_daily.parquet          # 2,877 rows x 3 cols | Daily VN-Index P/E valuation (2015–2026)
├── market/
│   ├── vn_indices_daily.parquet                 # 4,570 rows x 8 cols | VNINDEX & VN30 daily OHLCV (2015–2026)
│   ├── vn30f1m_derivatives_daily.parquet        # 2,296 rows x 8 cols | VN30F1M Futures daily OHLCV (2017–2026)
│   ├── vn_top_equities_daily.parquet            # 136,302 rows x 8 cols | 55 constituent stocks daily OHLCV
│   ├── vn_stock_industry_classification.parquet # 1,971 rows x 22 cols | All listed stocks mapped to 4-tier ICB
│   ├── vn_industry_sectors_icb.parquet          # 185 rows x 7 cols | 185 ICB sector definitions & ticker lists
│   ├── vn_sector_indices_daily.parquet          # 26,113 rows x 8 cols | 12 official sector indices daily OHLCV
│   ├── vn_foreign_trading_daily.parquet         # 130,837 rows x 12 cols | Daily Foreign Trading (GD NĐTNN: Buy, Sell, Room)
│   ├── vn_order_supply_demand_daily.parquet     # 83,731 rows x 11 cols | Daily Order Book Supply/Demand (Cung cầu: Orders, Vol)
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
│   ├── vn_financial_models_taxonomy.parquet     # 3,370 rows x 12 cols | Accounting line item definitions (TT202, TT49, TT334)
│   ├── vn_financial_statements_quarterly_yearly.parquet # 325,000 rows x 12 cols | Balance Sheet, Income, Cash Flow line items
│   ├── vn_financial_ratios_quarterly_yearly.parquet     # 189,000 rows x 8 cols  | Standardized financial ratios (P/E, ROE, Margins)
│   └── vn_listed_q3_2026_financials.parquet     # 192 rows x 21 cols | Q3-2026 corporate results (Revenue, Profit, P/E)
└── cross_asset/
    └── global_macro_daily.parquet               # 26,358 rows x 8 cols | S&P500, US10Y, DXY, Gold, WTI, VIX, BTC, USDVND
```

---

## 3. Verified Data Sources & Reference Links

Every dataset in this repository is collected from primary institutional and regulatory disclosure sources:

1. **WiData Financial & Macro Engine**: High-frequency macroeconomic series, interbank market rates, SBV bills/OMO operations, and market valuation multiples.  
   *Reference*: [https://widata.vn/vi-mo/vn/pmi-san-xuat](https://widata.vn/vi-mo/vn/pmi-san-xuat) | [https://widata.vn/ket-qua-kinh-doanh](https://widata.vn/ket-qua-kinh-doanh)
2. **State Bank of Vietnam (SBV / NHNN)**: Central exchange rate reference, refinancing rates, discount rates, and interbank transaction benchmarks.  
   *Reference*: [https://www.sbv.gov.vn](https://www.sbv.gov.vn)
3. **VNDirect Open Financial Information (Finfo) Engine**: Listed company registry, 4-tier ICB taxonomy, financial statements (TT200/TT202/TT49), financial ratios, daily order book supply/demand (`trading_statistics`), daily foreign trading (`foreigns`), shareholders, and subsidiaries.  
   *Reference*: [https://dchart.vndirect.com.vn](https://dchart.vndirect.com.vn) | [https://www.vndirect.com.vn](https://www.vndirect.com.vn)
4. **Ho Chi Minh Stock Exchange (HOSE) & Hanoi Stock Exchange (HNX)**: Official Point-in-Time constituent rebalancing announcements, free-float factors ($f$), capping factors ($c$), and outstanding index shares.  
   *Reference*: [https://www.hsx.vn](https://www.hsx.vn) | [https://www.hnx.vn](https://www.hnx.vn)
5. **Ministry of Finance (Bộ Tài chính)**: Statutory accounting circulars governing financial statement reporting lines (`TT200/2014/TT-BTC`, `TT202/2014/TT-BTC`, `TT334/2016/TT-BTC`, `TT199/2014/TT-BTC`).  
   *Reference*: [https://mof.gov.vn](https://mof.gov.vn)
6. **Yahoo Finance Open Engine**: Global macroeconomic cross-asset benchmarks: S&P 500 (`^GSPC`), US 10Y Yield (`^TNX`), US Dollar Index (`DX-Y.NYB`), WTI Crude Oil (`CL=F`), Gold (`GC=F`), VIX (`^VIX`), Bitcoin (`BTC-USD`), and USD/VND (`VND=X`).  
   *Reference*: [https://finance.yahoo.com](https://finance.yahoo.com)
7. **Federal Reserve Bank of St. Louis (FRED)**: US monetary and debt time series: Effective Federal Funds Rate (DFF), Treasury yields (3M, 2Y, 10Y), M2 money stock, and Fed balance sheet assets (WALCL).  
   *Reference*: [https://fred.stlouisfed.org](https://fred.stlouisfed.org)

---

## 4. Polars vs. Pandas Implementation Solutions

To ensure high-performance execution across research scripts, agents should utilize the following side-by-side patterns:

### Solution A: Point-in-Time Alignment (Macro to Equity Returns)
*When merging monthly macroeconomic data (PMI/CPI) with daily equity prices without forward look-ahead bias.*

* **Polars (Recommended — Zero-Copy Vectorized)**:
  ```python
  import polars as pl
  
  df_idx = pl.read_parquet("data/market/vn_indices_daily.parquet").filter(pl.col("symbol") == "VNINDEX")
  df_macro = pl.read_parquet("data/macro/vietnam_macro_monthly.parquet")
  
  # Point-in-time backward as-of join
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
  
  aligned = pd.merge_asof(
      df_idx.sort_values("date"),
      df_macro.sort_values("date"),
      on="date",
      direction="backward"
  )
  ```

---

### Solution B: Rolling Volatility & Basis Spread Calculation
*Calculating 60-day rolling annualized volatility and cash-futures basis.*

* **Polars**:
  ```python
  import polars as pl
  
  df = pl.read_parquet("data/market/vn_indices_daily.parquet")
  basis_df = (
      df.filter(pl.col("symbol") == "VN30")
      .with_columns([
          (pl.col("close").log() - pl.col("close").shift(1).log()).alias("ret"),
          pl.col("close").rolling_std(window_size=60).alias("vol_60d")
      ])
  )
  ```
* **Pandas**:
  ```python
  import pandas as pd
  import numpy as np
  
  df = pd.read_parquet("data/market/vn_indices_daily.parquet").query("symbol == 'VN30'")
  df["ret"] = np.log(df["close"] / df["close"].shift(1))
  df["vol_60d"] = df["close"].rolling(60).std()
  ```

---

### Solution C: Fundamental Financial Ratio Screening
*Screening equities with $P/E < 15$, $ROE > 15\%$, and $\text{Debt/Equity} < 1.0$.*

* **Polars**:
  ```python
  import polars as pl
  
  ratios = pl.read_parquet("data/financials/vn_financial_ratios_quarterly_yearly.parquet")
  screened = (
      ratios.filter(
          ((pl.col("ratio_code") == "PRICE_TO_EARNINGS") & (pl.col("value") < 15.0) & (pl.col("value") > 0)) |
          ((pl.col("ratio_code") == "ROAE_TR_AVG5Q") & (pl.col("value") > 15.0)) |
          ((pl.col("ratio_code") == "DEBT_TO_EQUITY_AQ") & (pl.col("value") < 1.0))
      )
  )
  ```
* **Pandas**:
  ```python
  import pandas as pd
  
  ratios = pd.read_parquet("data/financials/vn_financial_ratios_quarterly_yearly.parquet")
  pe_mask = (ratios["ratio_code"] == "PRICE_TO_EARNINGS") & (ratios["value"] < 15.0) & (ratios["value"] > 0)
  roe_mask = (ratios["ratio_code"] == "ROAE_TR_AVG5Q") & (ratios["value"] > 15.0)
  de_mask = (ratios["ratio_code"] == "DEBT_TO_EQUITY_AQ") & (ratios["value"] < 1.0)
  screened = ratios[pe_mask | roe_mask | de_mask]
  ```

---

### Solution D: Corporate Network Centrality & Traversal
*Extracting key conglomerates and their subsidiary network.*

* **Polars**:
  ```python
  import polars as pl
  
  edges = pl.read_parquet("data/market_structure/vn_corporate_network_edges.parquet")
  nodes = pl.read_parquet("data/market_structure/vn_corporate_network_nodes.parquet")
  
  # Top 10 systemically influential entities by PageRank
  top_central = nodes.sort("pagerank", descending=True).head(10)
  
  # Extract Vingroup direct subsidiaries
  vic_subs = edges.filter((pl.col("source_id") == "VIC") & (pl.col("relationship_type") == "SUBSIDIARY_CONTROL"))
  ```
* **Pandas**:
  ```python
  import pandas as pd
  
  edges = pd.read_parquet("data/market_structure/vn_corporate_network_edges.parquet")
  nodes = pd.read_parquet("data/market_structure/vn_corporate_network_nodes.parquet")
  
  top_central = nodes.sort_values("pagerank", ascending=False).head(10)
  vic_subs = edges[(edges["source_id"] == "VIC") & (edges["relationship_type"] == "SUBSIDIARY_CONTROL")]
  ```

---

## 5. Research Guidelines & Non-Negotiables for Agents

1. **Strict Causality & Point-in-Time (PIT) Integrity**:
   * **Never** use data before its official release timestamp (`time`, `announced_at`, or `report_date`).
   * When aligning monthly macro data with daily equity prices, use **backward matching** (`strategy="backward"` in Polars `join_asof`). **Never** forward-fill future announcements into the past.
2. **Free-Float & Capping Factor Application**:
   * For index replication, portfolio weighting, or basis trading ($\text{Basis}_t = F_t - S_t$), stock weights **must** use the official free-float and capping factors from `data/market_structure/`:
     $$w_{i, t} = \frac{P_{i, t} \times Q_{i, t} \times f_{i, t} \times c_{i, t}}{\sum_{j=1}^N P_{j, t} \times Q_{j, t} \times f_{j, t} \times c_{j, t}}$$
3. **Dedicated Research Repositories**:
   * `vnresearch/`: Standalone quantitative data lake, ingestion pipelines, and verification tools (no PDF reports, data/scripts/make/readmes only, hosted at [`https://github.com/kimnt93/vnresearch.git`](https://github.com/kimnt93/vnresearch.git)) feeding empirical datasets to Papers 001 through 015.
   * `001_macro_volatility_garch_midas/` through `015_tightly_held_shares_basis_dynamics/`: Independent empirical research papers with self-contained pipelines and LaTeX manuscripts.
4. **PDF Compilation Protocol**:
   * Compile research paper manuscripts using Tectonic:
     ```bash
     cd <paper_directory> && /home/kim/.local/bin/tectonic main.tex
     ```
5. **Visualizations**:
   * All charts must be rendered using **Polars** for data processing and **Seaborn** for styling (`scripts/render_*.py`). In `vnresearch/`, output **ONLY PNG** files (zero PDF reports).

---

## 6. AutoResearch Harness & Multi-Agent Architecture

This repository functions as an autonomous quantitative research harness implementing the Karpathy AutoResearch cycle:
$$\text{Idea} \longrightarrow \text{Plan} \longrightarrow \text{Code} \longrightarrow \text{Experiment} \longrightarrow \text{Analyse} \longrightarrow \text{Results} \longrightarrow \text{Conclusions} \longrightarrow \text{Database Sync}$$

### 6.1 Research Scope & Directives
All agents must strictly consult **`RESEARCH.md`** for:
* The quantitative research taxonomy across 6 core domains (Microstructure, Derivatives & Basis, Corporate Networks & Contagion, Macro Transmission, Fundamental Factors & SUE, Free-Float Wedges).
* Baseline empirical results and methodologies established in Papers 001 through 015.
* Hypothesis formulation rules ($H_0$ vs $H_1$, Newey--West HAC inference, placebo bootstrap tests).
* Active and frontier research questions.

### 6.2 Agent Skills Framework (`.agents/skills/`)
Modular agent skills are located under `.agents/skills/` following the [agentskills.io](https://agentskills.io) open standard (`SKILL.md` with YAML frontmatter):

1. **`data-discovery`**: Discovering, auditing schemas, and querying the 36 Parquet datasets in `data/`.
2. **`polars-fast-data`**: High-performance zero-copy Polars routines, backward `join_asof` alignment, rolling metrics.
3. **`hypothesis-testing-stats`**: Rigorous econometric hypothesis testing (ADF, HAC Newey-West, event studies, placebo tests).
4. **`charts-visualization`**: Publication-grade Polars + Seaborn chart rendering to high-DPI PNGs.
5. **`latex-report-generation`**: Compiling concise Research Future LaTeX reports into `main.pdf` via Tectonic.
6. **`arxiv-paper-search`**: Searching arXiv and academic engines for prior literature and mathematical models.
7. **`paper-pdf-to-md`**: Extracting equations, tables, and methodologies from research PDFs into structured Markdown.
8. **`autoresearch-loop`**: Autonomous end-to-end execution of the Karpathy research loop.
9. **`dedup-hypothesis-search`**: Querying `autoresearch.db` with SQLite FTS5 BM25 search to eliminate duplicate ideas.

### 6.3 Multi-Agent CLI Execution (`run.py`)
Agents can be awakened individually or concurrently using the unified CLI runner:
```bash
# Launch concurrent agent sessions
python run.py --agents=codex,claude,hermes

# Launch single agent with specific research topic
python run.py --agent=claude --topic="Foreign trading flow price impact in VN30"

# Search database to check for duplicate hypotheses before running
python run.py --search="order flow imbalance return predictability"

# List all recorded hypotheses and empirical decisions
python run.py --list

# Start the Web Research Dashboard and Dataset Explorer
python run.py --web --port=8080
```

Supported agent engines: `claude`, `codex`, `pi`, `omp`, `hermes`. Any agent session launched in this directory reads `AGENTS.md`, loads the relevant skills from `.agents/skills/`, and checks `autoresearch.db` before writing code.

### 6.4 SQLite Tracking Database (`autoresearch.db`)
All research activities, hypotheses, experimental parameters, and decisions are persisted in `autoresearch.db`:
* **`sessions`**: Tracks agent session ID, agent list, research domain, and execution timestamp.
* **`hypotheses`**: Tracks hypothesis title, formal $H_0$ and $H_1$, metric names ($t$-stat, $p$-val, Sharpe, $R^2$), empirical decision (`ACCEPTED`, `REJECTED`, `INCONCLUSIVE`), summary findings, and paper PDF path.
* **`hypotheses_fts`**: FTS5 virtual table indexing title, statement, and findings for fast BM25 deduplication ranking.
* **`experiments`**: Records execution scripts, duration, and raw JSON results.

### 6.5 Interactive Web Dashboard
The web dashboard provides an interactive UI to:
1. Inspect hypotheses and their empirical outcomes (`ACCEPTED` vs `REJECTED`).
2. Explore and preview all 36 Parquet datasets in `data/`.
3. View generated research PDF manuscripts inline.
4. Run real-time deduplication searches on research ideas.
