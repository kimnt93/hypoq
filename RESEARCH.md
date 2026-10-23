# RESEARCH.md — Vietnam Quantitative Financial Economics Research Agenda (hypoq: Hypothesis × Quant)

---

## About & Metadata

* **Author**: AI Quantitative Research Network
* **Cooperation**: Autonomous Financial Machine Learning & Empirical Econometrics Lab
* **Focus**: Vietnamese Financial Market System (Cash Equities, Index Derivatives, Monetary Policy, Corporate Networks, Sector Indices & ETFs)
* **Execution Standard**: Strict hypothesis-driven econometrics, zero look-ahead bias, and Python-only pipeline implementation.

---

## 1. Executive Research Mandate

The primary mission is the autonomous generation, empirical econometric testing, and peer-reviewed documentation of quantitative hypotheses within the Vietnamese financial market ecosystem and its international macro linkages.
Every empirical investigation follows the **AutoResearch Cycle**:
$$\text{Idea} \longrightarrow \text{Plan} \longrightarrow \text{Code} \longrightarrow \text{Experiment} \longrightarrow \text{Analyse} \longrightarrow \text{Results} \longrightarrow \text{Conclusion} \longrightarrow \text{DB Sync}$$

### Core Invariants:
1. **Hypothesis-Driven First**: No ex-post curve-fitting. Every experiment begins with an explicit Null Hypothesis ($H_0$) and Alternative Hypothesis ($H_1$).
2. **Strict Point-in-Time (PIT) Causality**: Zero look-ahead bias ($\tau_{\text{info}} \le \tau_{\text{market}}$).
3. **Econometric & Statistical Rigor**: Newey--West HAC robust covariance, stationarity verification (ADF), placebo bootstrap tests, and transaction friction hurdles.
4. **Python-Only Empirical & ML/DL/RL Implementation**: All estimation pipelines, machine learning (Scikit-Learn), deep learning (PyTorch), and reinforcement learning (Stable-Baselines3) models must be executed in Python using Polars, Scikit-Learn, PyTorch, Stable-Baselines3, and Statsmodels (`code/run_pipeline.py`).
5. **Autonomous Serialization**: Every study compiles to a concise, high-density LaTeX report (`main.pdf`) using the standard research template and registers metrics in the SQLite research database (`autoresearch.db`).

---

## 2. Research Domains & Keywords

Each domain represents an active empirical research agenda within the Vietnamese financial market system:

* **Market Microstructure & Order Book Queue Imbalances**: Limit order book supply-demand imbalance, bid-ask queue volume asymmetry, order imbalance ratio (OIR), trade size clustering, Kyle's lambda price impact, Amihud illiquidity, ATO/ATC auction clearing dynamics, econometric, quantitative vietnam market.
* **Foreign Institutional Capital & Ownership Limit (FOL) Dynamics**: Foreign trading flows (`GD NĐTNN`), net institutional buying pressure, foreign ownership limit room exhaustion, foreign premium dynamics, foreign capital reallocation, cross-border equity shocks, quantitative vietnam market.
* **Derivatives Pricing, Futures-Cash Basis & Limits of Arbitrage**: VN30F1M index futures, cost-of-carry fair valuation, cash-futures basis spread, structural discount regimes, short-sale borrowing frictions, reverse cash-and-carry arbitrage bands, contract roll dynamics, third-Thursday expiry effects, VECM lead-lag price discovery, econometric.
* **Exchange-Traded Funds (ETFs) & NAV Parity Arbitrage**: ETF, E1VFVN30, FUEVFVND Diamond ETF, FUESSVFL, FUEVN100, Net Asset Value (NAV) parity, premium-discount spread, creation-redemption baskets, authorized participants (AP), tracking error dispersion, error-correction mean reversion half-life, arbitrage efficiency, quantitative vietnam market.
* **State Bank of Vietnam (SBV) Monetary Transmission & Liquidity Corridor**: SBV policy rates, refinancing rate, discount rate, Open Market Operations (OMO), SBV treasury bills, overnight interbank borrowing rate corridor, daily monetary liquidity shocks, interest rate pass-through, commercial bank deposit benchmarks, exchange rate central parity adjustments, macro transmission.
* **Corporate Ownership Network Topologies & Systemic Contagion**: Directed corporate network graphs, conglomerate parent-subsidiary control, intercompany guarantee loans, major shareholdings, state ownership stakes, PageRank network centrality, degree centrality, debt-to-equity leverage transmission, credit risk contagion, systemic risk, Delta-CoVaR, econometric.
* **Statutory Accounting Taxonomy, Earnings Surprises & Quality Anomalies**: Statutory circulars `TT200/2014/TT-BTC`, `TT202/2014/TT-BTC`, `TT334/2016/TT-BTC`, `TT49/2014/TT-NHNN`, Standardized Unexpected Earnings (SUE), Post-Earnings Announcement Drift (PEAD), balance sheet health, corporate debt coverage, accrual quality, valuation multiples (P/E, P/B, EV/EBITDA).
* **Index Free-Float Wedges, Capping Factors & Review Price Pressure**: Free-float factor $f$, capping factor $c$, outstanding index shares $Q$, tightly held equities, state-owned enterprise (SOE) float restrictions, capitalization wedge $\Omega_t$, VNINDEX vs VN30 tracking divergence, semi-annual HOSE index reviews, rebalancing price pressure event studies, econometric.
* **Commodity Price Shocks & Cross-Asset Inflation Pass-Through**: Domestic construction steel (Hoa Phat CB240/D10), retail fuels (RON95, E5, Diesel), agricultural commodities (Robusta coffee, IR50404 fresh rice, live hog pork), commodity price shocks, materials sector index VNMAT, energy sector index VNENE, domestic inflation pass-through, TVP-VAR connectedness.
* **Global Macroeconomic Linkages & Currency Pass-Through**: US Federal Reserve FOMC interest rate decisions (PIT announcements), effective Fed Funds (DFF), US Treasury yields (3M/2Y/10Y), US Dollar Index (DXY), USD/VND commercial exchange rate, global cross-asset volatility spillovers (S&P 500, Gold, WTI crude, VIX, Bitcoin), emerging market capital flows.

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
