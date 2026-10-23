# Research 004: Interbank Monetary Liquidity Shocks and Equity Return Transmission in Vietnam

* **Author**: AI Quantitative Research Network
* **Format**: Pure English LaTeX (Research Future Overleaf Template Style).
* **Compiler**: Tectonic (`/home/kim/.local/bin/tectonic main.tex`).
* **JEL Codes**: `E52`, `E44`, `G12`, `G15`.
* **Keywords**: Asset pricing; Interbank lending rate; Monetary transmission; State Bank of Vietnam (SBV); Stock returns; Newey--West HAC.

---

## 1. Directory Structure

```text
researches/res_004_interbank_rate_equity_volatility/
├── main.tex                    # Manuscript
├── references.bib              # BibTeX literature citations
├── main.pdf                    # Compiled manuscript (Tectonic)
├── README.md                   # Documentation & replication
├── code/
│   └── run_pipeline.py         # Polars + Statsmodels pipeline
├── results/
│   ├── empirical_results.json  # Raw metrics
│   └── table1_monetary_regressions.tex # Formatted LaTeX table
└── figures/
    └── fig1_monetary_transmission.png # High-DPI publication chart
```

---

## 2. Hypothesis Formulation & Decision

* **Hypothesis**: Daily unanticipated overnight interbank rate fluctuations depress contemporaneous equity market returns ($H_0: \beta_{\text{ON}} = 0$ versus $H_1: \beta_{\text{ON}} < 0$).
* **Empirical Research Decision**: **ACCEPTED (Monetary Contraction Confirmed)**.
  * Point estimate: $\beta = -0.0782$\% return per 100 bps rate increase ($t = -2.5937, p = 0.0095$).
  * Every 100 bps interbank liquidity tightening contracts the VN-Index by $-7.82$ bps on the same session.
