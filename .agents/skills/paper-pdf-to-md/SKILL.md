---
name: paper-pdf-to-md
description: Extract, convert, and summarize research paper PDFs into structured Markdown using anymd and Python, extracting hypotheses, mathematical models, and empirical results.
version: 1.0.0
author: hypoq
tags: [pdf, markdown, anymd, parsing, literature, extraction]
---

# Paper PDF to Markdown Skill

## Overview
This skill provides automated conversion of academic research papers from PDF format into structured Markdown (`paper.md`), extracting equations, tables, experimental setups, and empirical findings for agent reasoning.

## When to Use
- Reading previous research papers (any PDF, including the compiled `researches/*/main.pdf`) to understand prior methodology.
- Ingesting newly downloaded arXiv or journal PDFs.
- Creating readable text summaries for the web dashboard or autonomous cross-study comparisons.

## Command-Line Execution via `anymd`
The system utilizes `anymd` (`/home/kim/.bun/bin/anymd`) for fast, clean PDF conversion:
```bash
# Convert a full PDF paper into Markdown
/home/kim/.bun/bin/anymd paper.pdf -o paper.md

# Convert specific pages (e.g., Abstract and Methodology)
/home/kim/.bun/bin/anymd paper.pdf -p 1-5 -o methodology.md
```

## Python Integration
Use `scripts/anymd_parser.py`:
```python
from scripts.anymd_parser import parse_to_markdown

# Extract paper text
markdown_text = parse_to_markdown("015_tightly_held_shares_basis_dynamics/main.pdf")

# Extract specific sections (e.g. first 4 pages)
summary_text = parse_to_markdown("015_tightly_held_shares_basis_dynamics/main.pdf", pages="1-4")
```

## Structured Extraction Checklist
When an agent reads a converted `paper.md`, it must extract and record:
1. **Core Research Question & Hypothesis ($H_0$ vs $H_1$)**.
2. **Identification Strategy**: Regression equation, event window, or machine learning model.
3. **Data Sources & Observation Frequency**: Equities, derivatives, macro indicators, and sample date range.
4. **Primary Empirical Findings**: Test statistics, effect sizes, $R^2$, or Sharpe ratios.
5. **Reported Frictions & Limitations**: Transaction costs, short-sale bounds, liquidity constraints.
