# AGENTS.md — Autonomous Agent Operating Protocol for hypoq (Hypothesis × Quant)

Welcome to **hypoq** (**Hypothesis $\times$ Quant**), the autonomous AI quantitative research lab and empirical modeling platform developed by the **AI Quantitative Research Network**.

---

## 1. Core Mandate for AI Agents

Whenever an AI agent (Claude, Codex, Hermes, Pi, OMP) is awakened in this repository, it must strictly orient itself by reading the following foundational files:

1. **[`RESEARCH.md`](RESEARCH.md)**: Defines the research taxonomy across active quantitative domains, required modeling paradigms (Machine Learning with `scikit-learn`, Deep Learning with `pytorch`, Reinforcement Learning with `stable-baselines3`, and Econometrics with `statsmodels`), and strict causality invariants.
2. **[`DATA.md`](DATA.md)**: Details the quantitative data lake (36 Parquet files under `data/`), schema dictionaries, and the ISO jurisdiction prefix standard (`vn_`, `us_`, `eu_`, `global_`).
3. **[`.agents/skills/`](.agents/skills/)**: Contains 12 self-contained skills adhering to the [AgentSkills.io](https://agentskills.io) standard:
   * `ml-dl-rl-modeling`: Scikit-Learn, PyTorch, and Stable-Baselines3 implementations.
   * `hypothesis-testing-stats`: Econometric inference (Newey--West HAC, ADF, placebo bootstrap).
   * `polars-fast-data`: High-performance zero-copy and Point-in-Time backward alignment.
   * `data-discovery`: Data lake auditing and schema inspection.
   * `data-explore-pit-builder`: Point-in-Time event dataset construction (Fed FOMC, SBV rates).
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
1. **Hypothesis-First**: Formulate explicit Null ($H_0$) and Alternative ($H_1$) hypotheses.
2. **Deduplication Check**: Always query `autoresearch.db` (`python run.py --search="..."`) to confirm research novelty before writing code.
3. **Python-Only Research Pipelines**: All data transformations, ML/DL/RL training, and econometric regressions must be coded in Python using Polars, Scikit-Learn, PyTorch, Stable-Baselines3, and Statsmodels (`code/run_pipeline.py`).
4. **Strict Point-in-Time Integrity**: Zero look-ahead bias. Low-frequency macro and accounting disclosures must be backward-matched (`join_asof(..., strategy="backward")`).
5. **Manuscript Compilation**: Every study must be serialized into `main.tex` using the standard template in `templates/` and compiled into `main.pdf` using Tectonic:
   ```bash
   cd researches/<research_folder> && /home/kim/.local/bin/tectonic main.tex
   ```
6. **Database Persistence**: Register empirical metrics, $p$-values, effect sizes, and final decisions (**ACCEPTED** or **REJECTED**) in `autoresearch.db`.

---

## 3. Operational CLI & System Commands

```bash
# Environment Setup (Host Machine via uv)
make install   # Creates .venv and installs Polars, PyTorch, Stable-Baselines3, Scikit-Learn, Statsmodels

# Data Ingestion & Integrity Verification
make download  # Triggers master WiData, regulatory filing, and macro ingestion
make verify    # Audits row counts, schemas, and nullness across all 36 Parquet datasets

# Web UI Display Stack (Docker Compose on Port 3030)
# Note: Docker purely displays the web app. Quantitative research is executed outside Docker on host mounts.
make start     # Launches Next.js TypeScript container with live volume mounts
make stop      # Terminates container stack
```
# Multi-Agent Harness CLI
python run.py --agents=codex,claude,hermes --topic="VN30 futures basis discount"
python run.py --search="order flow imbalance"
python run.py --list [--status=ACCEPTED]
```
