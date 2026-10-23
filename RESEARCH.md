# RESEARCH.md — Vietnam Quantitative Financial Economics Research Scope & Taxonomy

This document defines the research mandate, empirical taxonomy, econometric standards, and autonomous research agenda for **vnresearch** ([https://github.com/kimnt93/vnresearch.git](https://github.com/kimnt93/vnresearch.git)), focused on the **Vietnamese Financial Market System** across equities, derivatives, monetary policy, and corporate structures.

---

## 1. Executive Research Mandate

The primary mission is the autonomous generation, empirical econometric testing, and peer-reviewed documentation of quantitative hypotheses within the Vietnamese financial market ecosystem and its international macro linkages.

Every empirical investigation follows the **Karpathy AutoResearch Cycle**:
$$\text{Idea} \longrightarrow \text{Plan} \longrightarrow \text{Code} \longrightarrow \text{Experiment} \longrightarrow \text{Analyse} \longrightarrow \text{Results} \longrightarrow \text{Conclusion} \longrightarrow \text{DB Sync}$$

### Core Invariants:
1. **Hypothesis-Driven First**: No ex-post curve-fitting. Every experiment begins with an explicit Null Hypothesis ($H_0$) and Alternative Hypothesis ($H_1$).
2. **Strict Point-in-Time (PIT) Causality**: Zero look-ahead bias ($\tau_{\text{info}} \le \tau_{\text{market}}$).
3. **Econometric Rigor**: Newey--West HAC robust covariance, stationarity verification (ADF), placebo/permutation bootstrap tests, and realistic transaction cost hurdles.
4. **Python-Only Empirical Implementation**: All estimation pipelines and data wrangling must be executed in Python using Polars and Statsmodels (`code/run_pipeline.py`).
5. **Autonomous Serialization**: Every study compiles to a concise, high-density LaTeX report (`main.pdf`) using the **Research Future Template** and registers metrics in the SQLite research database (`autoresearch.db`).

---

## 2. Research Taxonomy: Six Core Vietnamese Market Domains

Our quantitative data lake (36 curated Parquet datasets under `data/`) covers six wide-scope empirical financial domains:

```text
                                 ┌──────────────────────────────────────────────┐
                                 │     Vietnamese Data Lake (36 Parquets)       │
                                 └──────────────────────┬───────────────────────┘
                                                        │
          ┌─────────────────────┬───────────────────────┼───────────────────────┬─────────────────────┐
          │                     │                       │                       │                     │
          ▼                     ▼                       ▼                       ▼                     ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│ 1. Microstructure │ │ 2. Derivatives &  │ │  3. Corporate     │ │ 4. Monetary Policy│ │  5. Fundamental   │
│   & Foreign Flow  │ │  Basis Dynamics   │ │  Network Topology │ │  & Macro Pass-Thru│ │ Factors & Earnings│
└───────────────────┘ └───────────────────┘ └───────────────────┘ └───────────────────┘ └───────────────────┘
          │                     │                       │                       │                     │
          └─────────────────────┴───────────────────────┼───────────────────────┴─────────────────────┘
                                                        │
                                                        ▼
                                       ┌──────────────────────────────────┐
                                       │ 6. Free-Float Wedges, Index ETFs │
                                       │    & Rebalancing Frictions       │
                                       └──────────────────────────────────┘
```

### Domain 1: Market Microstructure, Order Flow & Foreign Institutional Capital
- **Scope**: Limit order book supply/demand dynamics, retail trade clustering, Kyle's lambda price impact, Amihud illiquidity, foreign investor quota/room (`GD NĐTNN`), and auction clearing mechanisms (ATO/ATC).
- **Key Datasets**: `data/market/vn_order_supply_demand_daily.parquet`, `data/market/vn_foreign_trading_daily.parquet`, `data/market/vn_top_equities_daily.parquet`.
- **Core Questions**: Does aggregate order flow imbalance exhibit persistent short-term return predictability? Do foreign institutional transactions produce permanent vs temporary price impact in a retail-dominated exchange?

### Domain 2: Derivatives Pricing, Futures-Cash Basis & Arbitrage Frictions
- **Scope**: VN30F1M index futures valuation, cost-of-carry arbitrage bounds, reverse cash-and-carry limits under short-sale constraints, contract roll dynamics, and Hasbrouck price discovery.
- **Key Datasets**: `data/market/vn30f1m_derivatives_daily.parquet`, `data/market/vn_indices_daily.parquet`, `data/macro/vietnam_monetary_daily.parquet`.
- **Core Questions**: What institutional frictions sustain persistent negative futures basis discounts? Does the derivatives market lead or lag underlying cash basket price discovery?

### Domain 3: Corporate Ownership Networks & Systemic Risk Contagion
- **Scope**: Directed corporate network topologies (state ownership, parent-subsidiary relationships, bank-conglomerate lending exposures, cross-shareholdings), debt-to-equity leverage propagation, and systemic banking fragility ($\Delta\text{CoVaR}$, MES).
- **Key Datasets**: `data/market_structure/vn_corporate_network_edges.parquet`, `data/market_structure/vn_corporate_network_nodes.parquet`, `data/market_structure/vn_shareholders_history.parquet`, `data/market_structure/vn_subsidiaries_affiliates.parquet`.
- **Core Questions**: Does network centrality (PageRank, Out-Degree) predict financial distress transmission or credit risk spreads among multi-tier conglomerates?

### Domain 4: State Bank of Vietnam (SBV) Monetary Transmission & Global Spillovers
- **Scope**: SBV monetary policy tools (refinancing rate, discount rate, OMO auctions, SBV treasury bill issuance), interbank interest rate corridor, USD/VND central rate adjustments, US Federal Reserve policy rate transmission (FOMC PIT releases), and commodity price pass-through (steel, fuel, pork, coffee).
- **Key Datasets**: `data/macro/vietnam_monetary_daily.parquet`, `data/macro/vietnam_macro_monthly.parquet`, `data/macro/vietnam_sbv_policy_rates_daily.parquet`, `data/macro/us_macro_daily.parquet`, `data/macro/vietnam_commodities_daily.parquet`, `data/cross_asset/global_macro_daily.parquet`.
- **Core Questions**: How rapidly do interbank liquidity shocks transmit to equity sector volatility? Does US Federal Reserve tightening transmit through currency depreciation or capital outflows?

### Domain 5: Fundamental Factors, Accounting circulars & Earnings Momentum
- **Scope**: Vietnamese statutory accounting circulars (`TT200/2014/TT-BTC`, `TT202/2014/TT-BTC`, `TT334/2016/TT-BTC`, `TT49/2014/TT-NHNN`), Standardized Unexpected Earnings (SUE), Post-Earnings Announcement Drift (PEAD), quality, value, and leverage anomalies.
- **Key Datasets**: `data/financials/vn_financial_statements_quarterly_yearly.parquet`, `data/financials/vn_financial_ratios_quarterly_yearly.parquet`, `data/financials/vn_financial_models_taxonomy.parquet`.
- **Core Questions**: Does standardized unexpected earnings (SUE) generate post-announcement drift in Vietnam? Which accounting line items reliably predict subsequent credit distress?

### Domain 6: Free-Float Wedges, Index ETFs & Rebalancing Dynamics
- **Scope**: Official HOSE index calculation rules, tightly held shares ($f \le 0.35$), market capitalization wedges ($\Omega_t$), capping factors ($c$), ETF arbitrage bands (NAV premium/discount), and semi-annual constituent rebalancing price pressure.
- **Key Datasets**: `data/market_structure/free_float.parquet`, `data/market_structure/vn30_free_float_history.parquet`, `data/market_structure/index_constituents.parquet`, `data/market_structure/basket_history.parquet`, `data/market/vn_etf_market_daily.parquet`, `data/market/vn_etf_nav_aum_daily.parquet`.
- **Core Questions**: How does the capitalization wedge between full-capitalization and float-adjusted baskets create tracking error between VN-Index and VN30? How rapidly do ETF tracking discounts mean-revert?

---

## 3. Empirical Standards & Non-Negotiables

1. **Strict Point-in-Time (PIT) Alignment**:
   * Low-frequency announcements (macro releases, corporate earnings, index review parameters) must be matched using backward as-of joins (`join_asof(..., strategy="backward")`). Never look forward.
2. **Robust Statistical Inference**:
   * Time series: Newey--West HAC standard errors with Bartlett kernel lag length.
   * Unit root testing: Augmented Dickey-Fuller (ADF) checks prior to regression.
   * Placebo controls: Randomized bootstrap shuffle tests (min 1,000 draws).
3. **Friction-Adjusted Economic Hurdle**:
   * Minimum 15 bps two-way cash equity transaction fee, 2 ticks on futures, plus financing borrow rates.
4. **Outcome Classification**:
   * **ACCEPTED**: Statistically significant ($p < 0.05, |t| \ge 1.96$), economically meaningful net of frictions.
   * **REJECTED**: Null hypothesis maintained ($p \ge 0.05$ or opposite sign).
   * **INCONCLUSIVE**: Boundary cases, extreme parameter instability, or regime dependency.
