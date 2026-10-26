# RESEARCH.md — Global Quantitative Financial Economics Research Agenda (hypoq: Hypothesis × Quant)

---

## About & Metadata

* **Project**: hypoq (Hypothesis × Quant)
* **Focus**: Global Multi-Asset Financial Markets (Asia, United States, Europe, International Macro, Sovereign Yield Curves, Commodities & Cryptocurrencies)
* **Execution Standard**: Strict hypothesis-driven econometrics, zero look-ahead bias, and Python-only pipeline implementation.

---

## 1. Executive Research Mandate

The primary mission is the autonomous generation, empirical econometric testing, and peer-reviewed documentation of quantitative hypotheses across the global financial market ecosystem—spanning Asia-Pacific, North America, Europe, global commodities, and decentralized digital assets—with strict point-in-time causality and econometric discipline.
Every empirical investigation follows the **hypoq Cycle**:
$$\text{Idea} \longrightarrow \text{Plan} \longrightarrow \text{Code} \longrightarrow \text{Experiment} \longrightarrow \text{Analyse} \longrightarrow \text{Results} \longrightarrow \text{Conclusion} \longrightarrow \text{DB Sync}$$

### Core Invariants:
1. **English-Only Formulation & Serialization (Default Standard)**: All empirical hypotheses, titles, statements, Null ($H_0$) and Alternative ($H_1$) hypotheses, empirical summaries, and LaTeX manuscripts must be formulated and recorded strictly in academic English. No non-English or mixed-language hypothesis records are permitted.
2. **Hypothesis-Driven First**: No ex-post curve-fitting. Every experiment begins with an explicit Null Hypothesis ($H_0$) and Alternative Hypothesis ($H_1$).
3. **Strict Point-in-Time (PIT) Causality**: Zero look-ahead bias ($\tau_{\text{info}} \le \tau_{\text{market}}$).
4. **Econometric & Statistical Rigor**: Newey--West HAC robust covariance, stationarity verification (ADF), placebo bootstrap tests, and transaction friction hurdles.
5. **Python-Only Empirical & ML/DL/RL Implementation**: All estimation pipelines, machine learning (Scikit-Learn), deep learning (PyTorch), and reinforcement learning (Stable-Baselines3) models must be executed in Python using Polars, Scikit-Learn, PyTorch, Stable-Baselines3, and Statsmodels (`code/run_pipeline.py`).
6. **Authorship & Abstract Format**: every manuscript is authored by **SIQuant Research** (set in `templates/main.tex` and `hypoq.report.write_paper`). The structured abstract ends with a highlighted **Research decision** line (green `ACCEPTED`, red `REJECTED`, amber `INCONCLUSIVE`) that `hypoq.pipeline.finalize` adds from the computed decision.
7. **Autonomous Serialization & Semantic Storage**: Every study compiles to a concise, high-density LaTeX report (`main.pdf`) using the standard research template and registers metrics, findings, and dense vector embeddings in the database (`hypoq.db` via SQLite or PostgreSQL via `DATABASE_URL`).

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
* **Global Macroeconomic Linkages & Currency Pass-Through**: US Federal Reserve FOMC interest rate decisions, effective Fed Funds (DFF), US Treasury term structure (3M/5Y/10Y/30Y yield curves), US Dollar Index (DXY), European Central Bank policy rates, EUR/USD, GBP/USD, Asian FX regimes (USD/JPY, USD/CNY, USD/KRW, USD/INR, USD/SGD, USD/VND), global cross-asset volatility spillovers (S&P 500, Euro Stoxx 50, Nikkei 225, Gold, WTI crude, VIX, Bitcoin), emerging and developed market capital flows.
* **Cross-Border Equity Factor Portability & Asset Pricing Anomalies**: Multi-region factor modeling (Fama-French 5-factor, Carhart momentum, low-volatility, quality, size, value) across US equities (S&P 500, Nasdaq 100, Mega-Cap tech), European blue-chips (DAX 40, FTSE 100, CAC 40, ASML, Novo Nordisk, SAP), and Asian corporate giants (TSMC, Samsung, Alibaba, Tencent, Toyota), Fama-MacBeth 2-step cross-sectional regressions, transaction friction hurdles.
* **Cryptocurrency Microstructure, Regime Shifts & Tail Risk**: Digital assets (Bitcoin, Ethereum, Solana, Binance Coin, Layer-1 tokens), 24/7 calendar alignment, realized volatility estimators (Parkinson, Garman-Klass), crypto-equity spillover, rolling correlations with S&P 500 and Gold, Value-at-Risk (VaR) and Expected Shortfall (CVaR), weekend liquidity shocks, momentum and cross-pair lead-lag dynamics.
* **Global Commodity Cycles & Macroeconomic Transmission**: Energy markets (WTI crude oil, Brent crude, natural gas, RBOB gasoline), precious and industrial metals (Gold, Silver, Copper, Platinum), agricultural futures (Corn, Wheat, Soybeans, Coffee, Sugar, Cotton), supply-chain shocks, terms-of-trade shifts, cross-border inflation pass-through, Diebold-Yilmaz connectedness.
* **Sovereign Yield Curves & Term Premia Dynamics**: US Treasury curve (3M, 5Y, 10Y, 30Y), European sovereign spreads (German Bunds, Gilts, BTPs), Asian bond yields (JGBs), Nelson-Siegel level, slope, and curvature decompositions, term premium estimation, yield inversion recession nowcasting.
* **VIP Social Sentiment, Executive Announcements & Market Jump Dynamics**: Social media posts, Truth Social announcements, executive commentary, and FOMC communications by Donald Trump, Elon Musk, Jerome Powell, and Michael Saylor, event-study Cumulative Abnormal Returns (CAR), realized jump variation, intraday price discovery, tariff shock intensity, and crypto hype propagation.
* **World Financial Data Taxonomy, GICS Sector Rotations & Fundamental Valuation Alpha**: Multi-domain modeling bridging real economic activity (FRED PCE, Unemployment, NFCI), 11 GICS sector business cycle rotations, smart beta factors (Momentum, Quality, Value, Low-Vol), corporate credit spreads (HY OAS, LQD, HYG), cross-asset implied volatility surfaces (VIX, VVIX, OVX, GVZ), maritime supply chain freight (BDRY), and global cross-sectional corporate valuation screens (P/E, P/B, EV/EBITDA, ROE, margins, debt-to-equity).

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

---

## 4. Linguistic & Scientific Serialization Standard: English-Only Protocol

To maintain international reproducibility, institutional peer review, and zero semantic drift in vector embedding spaces:
1. **Strict English Default**: Every research hypothesis generated by autonomous agents (OMP, Claude, Codex, Hermes, Pi) must define `title`, `statement`, `null_hypothesis`, `alt_hypothesis`, and `summary_findings` exclusively in academic English.
2. **Mathematical & Econometric Precision**: Variable symbols ($R_{i, t}, \beta, \gamma, \sigma, \text{CAR}, \text{OIR}$), statistical test labels ($t\text{-statistic}, p\text{-value}, R^2, \text{RMSE}, \text{IC}$), and econometric decisions must adhere to standard English financial economics conventions.
3. **Semantic Vector Search Compatibility**: Dense vector embeddings (via Google Gemini `text-embedding-004` or Hugging Face `all-MiniLM-L6-v2`) are trained and optimized on English semantic representations, ensuring maximum retrieval accuracy and semantic clustering across the global research database.

---

## Integrity Rules for Autonomous Agents (enforced by `python run.py --audit`)

1. **Computed, never typed**: every number in a table, figure, manuscript, `empirical_results.json` or database row is produced by `researches/<folder>/code/run_pipeline.py`. Hand-written results, simulated "data" and decisions chosen by the author are forbidden.
2. **Status meaning**: `ACCEPTED` means the alternative hypothesis $H_1$ is supported by the pre-registered test; `REJECTED` means $H_1$ is not supported; `INCONCLUSIVE` means the test could not decide. Null results are valid results and must be reported.
3. **Multiple testing**: when several tests jointly decide the outcome, apply Bonferroni (or an equivalent) and state the adjusted $\alpha$.
4. **Real data only**: datasets come from free public sources fetched by scripts in `scripts/`. Synthetic or hand-curated rows must never enter `data/`.
5. **Idempotent registration**: pipelines call `hypoq.db.upsert_hypothesis`, so re-running updates the same row.
