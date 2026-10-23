# Research 003: Net Asset Value Premium-Discount Mean Reversion in Vietnamese Exchange-Traded Funds

* **Author**: AI Quantitative Research Network
* **Format**: Pure English LaTeX (Research Future Overleaf Template Style).
* **Compiler**: Tectonic (`/home/kim/.local/bin/tectonic main.tex`).
* **JEL Codes**: `G11`, `G14`, `G23`, `C22`.
* **Keywords**: Arbitrage efficiency; Authorized participants; Creation and redemption; Exchange-traded funds; Net Asset Value (NAV); Premium-discount mean reversion.

---

## 1. Directory Structure

```text
researches/res_003_etf_arbitrage_discount_reversal/
├── main.tex                    # Manuscript
├── references.bib              # BibTeX literature citations
├── main.pdf                    # Compiled manuscript (Tectonic)
├── README.md                   # Documentation & replication
├── code/
│   └── run_pipeline.py         # Polars + Statsmodels pipeline
├── results/
│   ├── empirical_results.json  # Raw metrics
│   └── table1_etf_arbitrage_reversion.tex # Formatted LaTeX table
└── figures/
    └── fig1_etf_premium_reversion.png # High-DPI publication chart
```

---

## 2. Hypothesis Formulation & Decision

* **Hypothesis**: ETF NAV premium/discount tracking spreads follow a unit-root random walk ($H_0: \beta = 0$) versus displaying rapid error-correction mean reversion ($H_1: \beta < -0.50, t < -10.0$).
* **Empirical Research Decision**: **ACCEPTED (Arbitrage Efficiency Confirmed)**.
  * Mean reversion speed: $\beta = -0.9710$ ($t = -50.31, p < 10^{-30}$).
  * Half-life: 0.20 trading sessions. Over 97% of tracking mispricing is eliminated within 1 trading day by AP arbitrageurs.
