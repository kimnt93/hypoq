# Research 005: Futures-Cash Basis Mispricing, Short-Sale Constraints, and Arbitrage Bands in VN30 Derivatives

* **Author**: AutoResearch Agent (Codex), Quantitative Research Network, Ho Chi Minh City, Vietnam.
* **Format**: Pure English LaTeX (Research Future Overleaf Template Style).
* **Compiler**: Tectonic (`/home/kim/.local/bin/tectonic main.tex`).
* **JEL Codes**: `G13`, `G14`, `G11`, `C32`.
* **Keywords**: Arbitrage bands; Cost of carry; Derivatives pricing; Futures-cash basis; Short-sale constraints; VN30 index futures; Newey--West HAC.

---

## 1. Directory Structure

```text
researches/res_005_futures_basis_arbitrage_bands/
├── main.tex                    # Manuscript
├── references.bib              # BibTeX literature citations
├── main.pdf                    # Compiled manuscript (Tectonic)
├── README.md                   # Documentation & replication
├── code/
│   └── run_pipeline.py         # Polars + Statsmodels pipeline
├── results/
│   ├── empirical_results.json  # Raw metrics
│   └── table1_futures_basis_estimates.tex # Formatted LaTeX table
└── figures/
    └── fig1_futures_basis_dynamics.png # High-DPI publication chart
```

---

## 2. Hypothesis Formulation & Decision

* **Hypothesis 1 (Structural Discount)**: Average futures basis is significantly negative due to short-sale restrictions ($H_0: \mu = 0$ versus $H_1: \mu < 0$).
  * **Result**: **ACCEPTED** ($\mu = -20.07$ bps, HAC $t = -5.4823, p = 4.20 \times 10^{-8}$).
* **Hypothesis 2 (Arbitrage Mean Reversion)**: Basis deviations exhibit rapid error-correction mean reversion ($H_0: \beta_{\text{rev}} = 0$ versus $H_1: \beta_{\text{rev}} < -0.20$).
  * **Result**: **ACCEPTED** ($\beta_{\text{rev}} = -0.2937$, HAC $t = -5.2039, p = 1.95 \times 10^{-7}$). Half-life of basis deviations is 1.99 trading sessions.
