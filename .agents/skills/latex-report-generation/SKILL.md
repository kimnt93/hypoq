---
name: latex-report-generation
description: Format, validate, and compile concise quantitative research reports into publication-ready PDFs using the Research Future LaTeX template and Tectonic.
version: 1.0.0
author: hypoq
tags: [latex, tectonic, report, compilation, research-future, pdf]
---

# LaTeX Report Generation Skill

## Overview
This skill provides the automated workflow for creating and compiling quantitative research reports using the **Research Template** (`templates/main.tex`) and the standalone TeX engine **Tectonic**.

## When to Use
- Compiling the final empirical research report `main.tex` to `main.pdf`.
- Generating publication-quality LaTeX tables (`results/table*.tex`) from Polars and Statsmodels outputs.
- Linking bibtex citations (`references.bib`) and high-resolution figures (`figures/*.png`).

## Required Report Layout
A compliant research report is short (typically 3 to 8 pages), highly focused, and starts immediately with the hypothesis and testing results:
1. **Title**: Concise, informative, $\le 20$ words.
2. **Abstract**: Four mandatory labeled components:
   - `\noindent \textbf{Hypothesis Formulation:}` Null ($H_0$) and Alternative ($H_1$) hypotheses.
   - `\textbf{Data \& Empirical Setting:}` Point-in-time sample details.
   - `\textbf{Econometric Methodology:}` HAC errors, zero-lookahead alignment, event study or model structure.
   - `\textbf{Empirical Results \& Summary:}` Key test statistics, $p$-values, effect size, and explicit research decision (**ACCEPTED**, **REJECTED**, or **INCONCLUSIVE**).
3. **Keywords & JEL Codes**.
4. **Section 1: Introduction & Hypothesis Formulation**.
5. **Section 2: Data Architecture & Methodology**.
6. **Section 3: Empirical Results & Hypothesis Testing** (with tables and figures).
7. **Section 4: Discussion & Frictions / Robustness**.
8. **Section 5: Conclusion & Research Decision**.
9. **References**: APA style via `\bibliographystyle{apalike}`.

## Automated Compilation via Tectonic
Compile with zero manual configuration using Tectonic:
```bash
/home/kim/.local/bin/tectonic main.tex
```

## Python Helper: Exporting Polars DataFrames to LaTeX Tables
```python
import polars as pl

def df_to_latex_table(df: pl.DataFrame, caption: str, label: str, out_path: str):
    """
    Renders a Polars DataFrame into a clean booktabs LaTeX table.
    """
    cols = df.columns
    header_str = " & ".join([f"\\textbf{{{c}}}" for c in cols]) + " \\\\"
    
    rows = []
    for row in df.iter_rows():
        formatted_cells = []
        for cell in row:
            if isinstance(cell, float):
                formatted_cells.append(f"{cell:.4f}")
            else:
                formatted_cells.append(str(cell))
        rows.append(" & ".join(formatted_cells) + " \\\\")
    
    body_str = "\n".join(rows)
    align_str = "l" + "r" * (len(cols) - 1)
    
    latex_code = f"""\\begin{{table}}[htbp]
\\centering
\\small
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{align_str}}}
\\toprule
{header_str}
\\midrule
{body_str}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(out_path, "w") as f:
        f.write(latex_code)
```
