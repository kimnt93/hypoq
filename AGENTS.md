# AGENTS.md — Autonomous Agent Operating Protocol for hypoq (Hypothesis × Quant)

Welcome to **hypoq** (**Hypothesis $\times$ Quant**), the autonomous AI quantitative research lab and empirical modeling platform developed by the **AI Quantitative Research Network**.

---

## 1. Core Mandate for AI Agents

Whenever an AI agent (Claude, Codex, Hermes, Pi, OMP) is awakened in this repository, it must strictly orient itself by reading the following foundational files:

1. **[`RESEARCH.md`](RESEARCH.md)**: Defines the research taxonomy across active quantitative domains, required modeling paradigms (Machine Learning with `scikit-learn`, Deep Learning with `pytorch`, Reinforcement Learning with `stable-baselines3`, and Econometrics with `statsmodels`), and strict causality invariants.
2. **[`DATA.md`](DATA.md)**: Details the quantitative data lake (80 Parquet files under `data/`, 2.0M+ rows, strictly < 100MB per file for GitHub safety), schema dictionaries, and the standardized jurisdiction prefix standard (`cn_`, `hk_`, `jp_`, `kr_`, `asia_`, `us_`, `eu_`, `global_`, `crypto_`, `sentiment/`, `vn_`).
3. **[`.agents/skills/`](.agents/skills/)**: Contains 18 self-contained skills adhering to the [AgentSkills.io](https://agentskills.io) standard:
   * `chunked-data-exploration`: Exploring, scanning, filtering, and streaming partitioned / chunked Parquet files with zero RAM bloat.
   * `world-data-taxonomy-modeling`: Multi-domain modeling across FRED macro, 11 GICS sectors, factors, credit spreads, and global valuation ratios.
   * `vip-sentiment-event-studies`: Quantifying market reactions, CAR, price jumps, and volatility surges around Trump, Musk, and Powell announcements.
   * `global-macro-cross-asset`: Cross-border macroeconomic transmission, sovereign yield curves, FX pass-through, and commodity shock propagation.
   * `crypto-quantitative-modeling`: Cryptocurrency microstructure, high-volatility regime modeling, crypto-macro spillover, and tail risk (VaR/CVaR).
   * `global-equity-factor-modeling`: Multi-region equity factor modeling (Fama-French, momentum, quality, low-vol) across US, EU, and Asia.
   * `ml-dl-rl-modeling`: Scikit-Learn, PyTorch, and Stable-Baselines3 implementations.
   * `hypothesis-testing-stats`: Econometric inference (Newey--West HAC, ADF, placebo bootstrap).
   * `polars-fast-data`: High-performance zero-copy and Point-in-Time backward alignment.
   * `data-discovery`: Data lake auditing, schema inspection, and continuous ingestion.
   * `data-explore-pit-builder`: Point-in-Time event dataset construction (Fed FOMC, central bank rates).
   * `data-exploration-js`: JavaScript / Arquero / `node:sqlite` data manipulation.
   * `charts-visualization`: High-DPI publication chart rendering with Seaborn.
   * `latex-report-generation`: Compiling `templates/main.tex` to `main.pdf` via Tectonic.
   * `arxiv-paper-search`: Literature exploration and BibTeX extraction.
   * `paper-pdf-to-md`: PDF-to-Markdown document parsing using `anymd`.
   * `autoresearch-loop`: Execution protocol for the 8-stage research cycle.
   * `dedup-hypothesis-search`: FTS5 BM25 deduplication ranking in `autoresearch.db`.
---

## 2. Autonomous Research Execution Protocol

All empirical research investigations must execute the **AutoResearch Cycle**:
$$\text{Idea} \longrightarrow \text{Plan} \longrightarrow \text{Code} \longrightarrow \text{Experiment} \longrightarrow \text{Analyse} \longrightarrow \text{Results} \longrightarrow \text{Conclusion} \longrightarrow \text{DB Sync}$$

### Non-Negotiables:
1. **English-Only by Default**: All hypothesis statements, titles, null/alt formulations, and findings must be written exclusively in academic English.
2. **Hypothesis-First**: Formulate explicit Null ($H_0$) and Alternative ($H_1$) hypotheses.
3. **Deduplication Check**: Always query the semantic vector search engine (`python run.py --search="..."`) to confirm research novelty via cosine similarity before writing code.
4. **Python-Only Research Pipelines**: All data transformations, ML/DL/RL training, and econometric regressions must be coded in Python using Polars, Scikit-Learn, PyTorch, Stable-Baselines3, and Statsmodels (`code/run_pipeline.py`).
5. **Strict Point-in-Time Integrity**: Zero look-ahead bias. Low-frequency macro and accounting disclosures must be backward-matched (`join_asof(..., strategy="backward")`).
6. **Manuscript Compilation**: Every study must be serialized into `main.tex` using the standard template in `templates/` and compiled into `main.pdf` using Tectonic:
   ```bash
   cd researches/<research_folder> && /home/kim/.local/bin/tectonic main.tex
   ```
7. **Dual-Backend Persistence**: Register empirical metrics, $p$-values, effect sizes, and dense vector embeddings in SQLite (`autoresearch.db`) or PostgreSQL (via `DATABASE_URL`).

---

## 3. Operational CLI & System Commands

```bash
# Environment Setup (Host Machine via uv)
make install   # Creates .venv and installs Polars, PyTorch, Stable-Baselines3, Scikit-Learn, Statsmodels

# Data Ingestion & Integrity Verification
make download  # Triggers master WiData, regulatory filing, and macro ingestion
make verify    # Audits row counts, schemas, and nullness across all 80 Parquet datasets

# Web UI Display Stack (Docker Compose on Port 3030)
# Note: Docker purely displays the web app. Quantitative research is executed outside Docker on host mounts.
make start     # Launches Next.js TypeScript container with live volume mounts
make stop      # Terminates container stack

# Multi-Agent Harness CLI
python run.py --agents=codex,claude,hermes --topic="VN30 futures basis discount"
python run.py --search="interest rate shock transmission to ETFs"  # Dense semantic vector search
DATABASE_URL="postgresql://user:pass@localhost:5432/hypoq" python run.py --list
```
