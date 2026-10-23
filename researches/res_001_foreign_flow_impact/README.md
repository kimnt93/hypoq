# Research 001: Foreign Institutional Trading Flow and Contemporaneous Price Impact in Frontier Equities

* **Author**: AI Quantitative Research Network
* **Format**: Pure English LaTeX (Research Future Overleaf Template Style).
* **Compiler**: Tectonic (`/home/kim/.local/bin/tectonic main.tex`).
* **JEL Codes**: `G14`, `G15`, `G11`, `C22`, `C58`.
* **Keywords**: Contemporaneous price impact; Foreign institutional flow; Frontier markets; Market microstructure; Newey--West HAC; Order flow.

---

## 1. Directory Structure

```text
researches/res_001_foreign_flow_impact/
├── main.tex                    # Manuscript (compiled to main.pdf)
├── references.bib              # BibTeX literature citations
├── main.pdf                    # Compiled manuscript (Tectonic)
├── README.md                   # Study documentation & findings
├── code/
│   └── run_pipeline.py         # End-to-end Polars + Statsmodels pipeline
├── results/
│   ├── empirical_results.json  # Raw metrics (test stat, p-val, R2)
│   └── table1_regression.tex   # Formatted LaTeX table (\input into main.tex)
└── figures/
    └── fig1_foreign_flow_impact.png # High-DPI publication chart
```

---

## 2. Hypothesis Formulation & Decision

* **Null Hypothesis ($H_0$)**: Foreign institutional net buying order flow exerts zero contemporaneous price impact on headline equity index returns ($\beta_{\text{foreign}} = 0$).
* **Alternative Hypothesis ($H_1$)**: Foreign institutional net buying order flow exerts a positive and statistically significant contemporaneous price impact ($\beta_{\text{foreign}} > 0, t \ge 1.96, p < 0.05$).
* **Empirical Research Decision**: **REJECTED (Null Hypothesis $H_0$ Maintained)**.

---

## 3. Empirical Findings (2,031 Trading Sessions, 2018-08-31 to 2026-10-21)

1. **Stationarity**: Both VN-Index returns ($ADF = -19.04, p < 0.001$) and standardized net foreign flow ($ADF = -9.51, p < 0.001$) are stationary.
2. **Point Estimate**: Foreign net flow displays a positive point coefficient $\beta = 0.1001$\% per standard deviation shock.
3. **Statistical Inference**: With Newey--West HAC standard errors (Bartlett kernel $L = 8$), the $t$-statistic is $1.6711$ ($p = 0.0947$), which fails to achieve the standard 5\% significance threshold.
4. **Controlled Model**: Controlling for lagged index return $r_{t-1}$ yields an identical conclusion ($\beta = 0.0998, t = 1.6702, p = 0.0949$).
5. **Economic Interpretation**: In frontier equity markets dominated by domestic retail liquidity ($>85\%$ of daily turnover), foreign institutional order execution is readily absorbed by retail limit order books without generating persistent index-level displacement.

---

## 4. Replication

```bash
# Run pipeline to estimate models, generate figure, and record in autoresearch.db
/home/kim/agents/paper-hub/.venv/bin/python3 researches/res_001_foreign_flow_impact/code/run_pipeline.py

# Compile manuscript to PDF
cd researches/res_001_foreign_flow_impact && /home/kim/.local/bin/tectonic main.tex
```
