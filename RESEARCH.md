# RESEARCH.md — Quantitative Research Scope, Taxonomy & Empirical Agenda

This document defines the quantitative research mandate, empirical taxonomy, econometric standards, and autonomous research agenda for **paper-hub** and **vnresearch** ([https://github.com/kimnt93/vnresearch.git](https://github.com/kimnt93/vnresearch.git)).

---

## 1. Executive Research Mandate

The primary mission is the autonomous generation, empirical testing, and peer-reviewed documentation of quantitative hypotheses across financial markets, with primary focus on emerging and frontier market structures (e.g., Vietnam equity, derivatives, and fixed-income markets), cross-asset spillovers, and global macroeconomic interactions.

Every research investigation follows the **Karpathy AutoResearch Cycle**:
$$\text{Idea} \longrightarrow \text{Plan} \longrightarrow \text{Code} \longrightarrow \text{Experiment} \longrightarrow \text{Analyse} \longrightarrow \text{Results} \longrightarrow \text{Conclusion} \longrightarrow \text{DB Sync}$$

### Core Invariants:
1. **Hypothesis-Driven First**: No ex-post curve-fitting. Every experiment begins with an explicit Null Hypothesis ($H_0$) and Alternative Hypothesis ($H_1$).
2. **Strict Point-in-Time (PIT) Causality**: Zero look-ahead bias ($\tau_{\text{info}} \le \tau_{\text{market}}$).
3. **Econometric Rigor**: Newey--West HAC robust errors, placebo/permutation tests, and transaction friction penalties.
4. **Autonomous Serialization**: Every study compiles to a concise, high-density LaTeX report (`main.pdf`) using the **Research Future Template** and registers metrics in the SQLite research database (`autoresearch.db`).

---

## 2. Research Taxonomy & Six Core Domains

Our quantitative data lake (36 curated Parquet datasets under `data/`) supports six interrelated empirical research domains:

```text
                                 ┌──────────────────────────────────────────────┐
                                 │     Quantitative Data Lake (36 Parquets)     │
                                 └──────────────────────┬───────────────────────┘
                                                        │
          ┌─────────────────────┬───────────────────────┼───────────────────────┬─────────────────────┐
          │                     │                       │                       │                     │
          ▼                     ▼                       ▼                       ▼                     ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│ 1. Microstructure │ │ 2. Derivatives &  │ │  3. Corporate     │ │ 4. Macroeconomic  │ │  5. Fundamental   │
│   & Order Flow    │ │  Basis Dynamics   │ │  Network Contagion│ │  Spillovers & Rates│ │ Factor & Earnings │
└───────────────────┘ └───────────────────┘ └───────────────────┘ └───────────────────┘ └───────────────────┘
          │                     │                       │                       │                     │
          └─────────────────────┴───────────────────────┼───────────────────────┴─────────────────────┘
                                                        │
                                                        ▼
                                       ┌──────────────────────────────────┐
                                       │ 6. Free-Float Wedges & Index ETF │
                                       │      Rebalancing Dynamics        │
                                       └──────────────────────────────────┘
```

### Domain 1: Market Microstructure & High-Frequency Order Flow
- **Focus**: Limit order book supply/demand asymmetry, trade size clustering, Kyle's lambda, Amihud illiquidity, price impact functions, and foreign institutional trading flows (`GD NĐTNN`).
- **Key Datasets**: `vn_order_supply_demand_daily.parquet`, `vn_foreign_trading_daily.parquet`, `vn_top_equities_daily.parquet`.
- **Target Hypotheses**: Whether order flow imbalances predict short-term return drift or liquidity shocks; whether foreign net buying exerts permanent vs temporary price impact.

### Domain 2: Derivatives Pricing, Futures-Cash Basis & Arbitrage
- **Focus**: VN30F1M index futures valuation, cost-of-carry arbitrage bounds, reverse cash-and-carry limits under short-sale constraints, expiry week roll dynamics, lead-lag relationships.
- **Key Datasets**: `vn30f1m_derivatives_daily.parquet`, `vn_indices_daily.parquet`, `vietnam_monetary_daily.parquet`.
- **Target Hypotheses**: Frictions causing persistent negative basis; information discovery shares (Hasbrouck / Gonzalo-Granger) between futures and cash.

### Domain 3: Corporate Networks & Systemic Risk Contagion
- **Focus**: Directed corporate graph topologies (ownership, parent-subsidiary relationships, board interlocks, guarantee networks), debt contagion, banking tail-risk spillovers ($\Delta\text{CoVaR}$, Marginal Expected Shortfall).
- **Key Datasets**: `vn_corporate_network_edges.parquet`, `vn_corporate_network_nodes.parquet`, `vn_shareholders_history.parquet`, `vn_subsidiaries_affiliates.parquet`.
- **Target Hypotheses**: Whether PageRank / eigenvector network centrality predicts credit spread spikes, earnings distress contagion, or systemic banking fragility.

### Domain 4: Cross-Asset Macroeconomic Transmission & Monetary Policy
- **Focus**: SBV monetary policy transmission (refinancing rate, discount rate, OMO, bill sterilization), interbank rates, Fed Funds spillovers, USD/VND pass-through, commodity price volatility.
- **Key Datasets**: `vietnam_monetary_daily.parquet`, `vietnam_macro_monthly.parquet`, `vietnam_sbv_policy_rates_daily.parquet`, `us_macro_daily.parquet`, `global_macro_daily.parquet`, `vietnam_commodities_daily.parquet`.
- **Target Hypotheses**: Mixed-frequency GARCH-MIDAS volatility modeling; term structure nowcasting using Nelson--Siegel factors; currency depreciation shocks on equity factor returns.

### Domain 5: Fundamental Factors, Accounting Taxonomy & Earnings Momentum
- **Focus**: Statutory circulars (`TT200/2014/TT-BTC`, `TT202/2014/TT-BTC`, `TT334/2016/TT-BTC`), Standardized Unexpected Earnings (SUE), Post-Earnings Announcement Drift (PEAD), quality/value/momentum factor interactions.
- **Key Datasets**: `vn_financial_statements_quarterly_yearly.parquet`, `vn_financial_ratios_quarterly_yearly.parquet`, `vn_financial_models_taxonomy.parquet`.
- **Target Hypotheses**: Cross-sectional return predictability of standardized accounting ratios; earnings revision momentum and machine learning return prediction.

### Domain 6: Index Free-Float Wedges & ETF Rebalancing Dynamics
- **Focus**: Official constituent parameters ($f, c, Q$), tightly held shares, capitalization wedges ($\Omega_t$), ETF creation/redemption baskets, index review event studies.
- **Key Datasets**: `free_float.parquet`, `vn30_free_float_history.parquet`, `index_constituents.parquet`, `basket_history.parquet`, `vn_etf_market_daily.parquet`, `vn_etf_nav_aum_daily.parquet`.
- **Target Hypotheses**: Capitalization wedge driving tracking error between VN-Index and VN30; price pressure and liquidity dry-ups during scheduled index rebalancing sessions.

---

## 3. Inventory of Baseline Research Papers (001 – 015)

The repository workspace builds upon 15 empirical foundation papers located in `paper-hub`:

| Paper ID | Title | Primary Domain | Core Methodology | Key Datasets | Key Finding / Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **001** | Macro Volatility & GARCH-MIDAS | Domain 4 (Macro) | Mixed-Frequency GARCH-MIDAS, Quantile Connectedness | Macro monthly, VN-Index daily, Monetary | SBV interbank and inflation explain long-term equity volatility regimes. |
| **002** | TAFlow Engine Architecture | Infrastructure | Streaming zero-copy technical indicators | High-frequency OHLCV | 24x latency reduction and zero-allocation rolling window metrics. |
| **003** | Hybrid ML/DL/RL Multi-Asset Portfolio | Domain 4 & 5 | Two-stage HMM regime switching + DRL optimization | Global macro, Equities, Bonds, Commodities | Superior Sharpe ratio (1.82) across monetary tightening regimes. |
| **004** | Free-Float Rebalancing Price Pressure | Domain 6 (Float) | Point-in-time event study, CAR analysis | Free-float, Constituents, OHLCV | Significant cumulative abnormal volume and price reversal around reviews. |
| **005** | Monetary Policy Transmission Regimes | Domain 4 (Macro) | Structural VAR, Threshold regressions | Fed Funds, SBV rates, USD/VND, Equities | Asymmetric transmission of US monetary shocks via exchange rate channel. |
| **006** | Futures Basis & Arbitrage Quality | Domain 2 (Derivatives) | Threshold error correction (TECM), Cost-of-carry | VN30F1M, VN30, Interbank rates | Short-sale frictions sustain asymmetric negative basis discount. |
| **007** | Systemic Risk & Banking Contagion | Domain 3 (Networks) | $\Delta\text{CoVaR}$, Network interconnectedness | Corporate edges/nodes, Financial ratios | State-owned commercial banks dominate systemic risk transmission. |
| **008** | Cross-Asset Volatility Spillovers | Domain 4 (Macro) | Diebold--Yilmaz spillover index, TVP-VAR | Commodities, US10Y, Gold, Oil, VN-Index | Energy and metals transmit significant volatility to frontier equities. |
| **009** | Nelson--Siegel Yield Curve Nowcasting | Domain 4 (Macro) | Dynamic Nelson--Siegel, Kalman Filter nowcasting | Interbank yields, SBV rates, Industrial production | Yield curve slope factor predicts real GDP growth 2 quarters ahead. |
| **010** | Corporate Earnings Momentum & ML | Domain 5 (Factors) | Gradient boosting, Standardized Unexpected Earnings | Financial statements, Financial ratios | SUE and earnings revision momentum generate robust annualized alpha. |
| **011** | High-Frequency Order Flow Imbalance | Domain 1 (Microstructure)| Kyle's $\lambda$, Vector Autoregressions | Order supply/demand, Top equities OHLCV | Order flow imbalance exhibits strong 1-to-3 session price impact. |
| **012** | DRL Central Bank Monetary Policy | Domain 4 (Policy) | Deep Q-Learning / PPO policy optimization | SBV policy rates, CPI, GDP nowcasts | DRL policy rule dampens equity drawdowns while stabilizing inflation. |
| **013** | Price Discovery Lead-Lag Dynamics | Domain 2 (Derivatives) | Hasbrouck information share, VECM | VN30F1M, VN30 daily microstructure | VN30F1M futures dominate price discovery with $>65\%$ information share. |
| **014** | Cross-Sectional Momentum Alpha | Domain 5 (Factors) | Volatility-scaled momentum, crash protection | 55 top constituent equities | Downside risk scaling eliminates catastrophic momentum crashes. |
| **015** | Tightly Held Shares & Basis Dynamics | Domain 2 & 6 | Basis decomposition, Newey-West HAC, Event study | Free float history, VN30, VN30F1M, Interbank | Float wedge explains 51.6% of tracking divergence; no direct basis bias. |

---

## 4. Hypothesis Testing Standards & Methodology Guidelines

Every automated research report generated in this repository MUST comply with the following standards:

### Standard A: Formal Hypothesis Specification
Every research proposal must define:
- **Null Hypothesis ($H_0$)**: Expected economic outcome under efficient markets, no friction, or zero statistical relationship.
- **Alternative Hypothesis ($H_1$)**: Directional or non-zero predictive economic relationship.
- **Evaluation Criteria**:
  - Significance threshold: $\alpha = 0.05$ (or $p < 0.01$ for multiple comparisons).
  - Minimum test statistic: $|t| \ge 1.96$ or $|z| \ge 1.96$.
  - Economic hurdle: Net annualized Sharpe $\ge 0.75$, or Information Ratio $\ge 0.50$ after subtracting minimum 15 bps two-way transaction costs.

### Standard B: Zero Look-Ahead Bias & Point-in-Time Alignment
- Use **Polars backward matching** (`join_asof(..., strategy="backward")`).
- Historical corporate financial statements must be matched using the official disclosure filing date (`report_date` or `announced_at`), never the period-end date.
- Free-float and capping factors ($f, c$) must be applied strictly from their official effective review dates.

### Standard C: Econometric Testing & Robustness
- **Time Series**: Newey--West HAC standard errors (lag length chosen via Bartlett kernel $L = 4(T/100)^{2/9}$).
- **Panel Data**: Firm- and time-clustered standard errors (Petersen / Thompson).
- **Event Studies**: Cross-sectional variance adjustment (Boehmer, Musumeci, Poulsen / BMP test) and calendar placebo bootstrap tests (min 1,000 randomized draws).
- **Stationarity**: Augmented Dickey-Fuller (ADF) and Phillips--Perron unit root verification before regression.

### Standard D: Empirical Decision Categorization
In the conclusion and in `autoresearch.db`, mark the outcome as one of:
1. **ACCEPTED** ($H_0$ rejected, $H_1$ confirmed with statistical and economic significance).
2. **REJECTED** ($H_0$ failed to be rejected, $p \ge 0.05$, or effect size negligible).
3. **INCONCLUSIVE** (Ambiguous results, data length limitations, or regime instability).

---

## 5. Active & Frontier Research Agenda (Future Directions)

Agents operating in this system are encouraged to investigate these open questions:

1. **Foreign Ownership Limit (FOL) Premium Dynamics**: Do equities hitting their 49% foreign ownership room trade at a persistent structural valuation premium, and does foreign buying exhaustion predict localized mean reversion?
2. **Corporate Network Contagion in Bond Issuance**: Does debt-to-equity leverage propagation across conglomerate subsidiary graphs predict bond default spreads during tight monetary liquidity regimes?
3. **Intraday Supply/Demand Imbalance Reversal**: Does extreme retail order book imbalance during morning sessions predict opposite institutional auction clearing returns in the ATC session?
4. **Cross-Border Exchange Rate Pass-Through**: How do sharp USD/VND depreciation episodes transmit to manufacturing exporter earnings vs dollar-debt laden utilities?
5. **ETF Creation Basket Arbitrage Limits**: Do premium/discount spreads in domestic ETFs (e.g. Diamond, VN30 ETF) mean-revert faster during high-liquidity market regimes?
