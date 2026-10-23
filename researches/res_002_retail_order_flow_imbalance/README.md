# Research 002: Intraday Order Book Imbalance and Price Impact Dynamics in Vietnamese Equities

* **Author**: AutoResearch Agent (Codex), Quantitative Research Network, Ho Chi Minh City, Vietnam.
* **Format**: Pure English LaTeX (Research Future Overleaf Template Style).
* **Compiler**: Tectonic (`/home/kim/.local/bin/tectonic main.tex`).
* **JEL Codes**: `G14`, `G12`, `C22`, `D47`.
* **Keywords**: Limit order book; Market microstructure; Order flow imbalance; Price impact; Vietnamese equities; Newey--West HAC.

---

## 1. Directory Structure

```text
researches/res_002_retail_order_flow_imbalance/
├── main.tex                    # Manuscript
├── references.bib              # BibTeX literature citations
├── main.pdf                    # Compiled manuscript (Tectonic)
├── README.md                   # Documentation & replication
├── code/
│   └── run_pipeline.py         # Polars + Statsmodels pipeline
├── results/
│   ├── empirical_results.json  # Raw metrics
│   └── table1_order_flow_regressions.tex # Formatted LaTeX table
└── figures/
    └── fig1_order_imbalance_impact.png # High-DPI publication chart
```

---

## 2. Hypothesis Formulation & Decision

* **Hypothesis 1 (Contemporaneous Price Impact)**: Order flow imbalance positively and significantly determines contemporaneous index returns ($H_0: \beta = 0$ vs $H_1: \beta > 0$).
  * **Result**: **ACCEPTED** ($\beta = 10.1043$, HAC $t = 11.4579$, $p < 10^{-29}$, $R^2 = 20.58\%$).
* **Hypothesis 2 (Predictive Overnight Drift)**: Order flow imbalance generates forward next-session return drift ($H_0: \beta_{\text{next}} = 0$).
  * **Result**: **REJECTED** ($t = 1.0926$, $p = 0.2746$). Order imbalances are contemporaneously incorporated into closing auction prices.
