# Research Future LaTeX Template for Quantitative Financial Economics

This LaTeX template is adapted from the Overleaf **Research Future Template** (`https://www.overleaf.com/latex/templates/research-future-template/dqwvghptfcmj`) published by the Science Partner Journal Program, customized specifically for **Autonomous Financial Research Reports** within `vnresearch` and `paper-hub`.

---

## 1. Core Principles

1. **Hypothesis-Driven Structure**:
   - The report starts with a concrete, falsifiable research hypothesis ($H_0$ vs $H_1$).
   - Abstract explicitly states: Hypothesis Formulation, Data & Setting, Econometric Methodology, and Empirical Summary Decision.
   - Section 1 defines formal null and alternative hypotheses.
   - Conclusion explicitly renders the research decision: **REJECT $H_0$ (SUPPORTED)**, **FAIL TO REJECT $H_0$**, or **INCONCLUSIVE**.

2. **Concise & High Information Density**:
   - Short reports (3 to 8 pages) emphasizing rigorous statistical tests ($t$-statistics, HAC standard errors, $p$-values, effect sizes) rather than lengthy prose.
   - Complete empirical replication via `code/run_pipeline.py`.
   - Figures rendered via Polars + Seaborn/Matplotlib into `figures/`.
   - Key numeric metrics saved to `results/empirical_results.json`.

3. **Autonomous Compilation**:
   - Compiled with Tectonic (zero-configuration standalone TeX/BibTeX engine):
     ```bash
     /home/kim/.local/bin/tectonic main.tex
     ```

---

## 2. Directory Layout

```text
research_folder/
├── main.tex                    # LaTeX manuscript
├── references.bib              # BibTeX literature citations
├── main.pdf                    # Compiled output
├── README.md                   # Research synopsis & reproduction guide
├── code/
│   └── run_pipeline.py         # Polars + Statsmodels empirical pipeline
├── results/
│   ├── empirical_results.json  # Raw metrics (test stat, p-val, Sharpe)
│   ├── table1_descriptives.tex # Generated table (optional \input)
│   └── table2_hypothesis.tex   # Generated table (optional \input)
└── figures/
    └── fig1_empirical_findings.png # Polars + Seaborn visualization
```
