# hypoq (Hypothesis × Quant) — Autonomous AI Quantitative Research Lab & Empirical Modeling Harness

> **hypoq** ($\text{Hypothesis} \times \text{Quant}$) — Where falsifiable scientific hypotheses meet institutional-grade quantitative modeling.

[![GitHub repository](https://img.shields.io/badge/GitHub-kimnt93%2Fhypoq-blue.svg?logo=github)](https://github.com/kimnt93/hypoq.git)
[![ML / DL / RL Stack](https://img.shields.io/badge/ML%2FDL%2FRL-PyTorch%20%7C%20Stable--Baselines3%20%7C%20Scikit--Learn-ee4c2c.svg)](https://pytorch.org/)
[![Econometrics](https://img.shields.io/badge/Stats-Statsmodels%20%7C%20Scipy%20%7C%20Polars-2ca02c.svg)](https://www.statsmodels.org/)
[![Host Package Manager](https://img.shields.io/badge/uv-Fast%20Python%20Env-de5fe9.svg)](https://astral.sh/uv)
[![Web Stack](https://img.shields.io/badge/Web-Next.js%2016%20%7C%20TypeScript%20%7C%20Arquero-black.svg)](https://nextjs.org/)
[![Deployment](https://img.shields.io/badge/Deploy-Docker%20Compose-2496ed.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LaTeX Compiler](https://img.shields.io/badge/LaTeX-Tectonic%20Engine-58a6ff.svg)](https://tectonic-typesetting.github.io/)

**hypoq** (**Hypothesis $\times$ Quant**) is an institutional-grade autonomous AI quantitative research lab, empirical modeling harness, and financial data lake. Designed for global financial economics, cross-border equity factor modeling, sovereign yield curve term structure, macroeconomic spillovers, cryptocurrency microstructure, and multi-asset derivatives across **Asia**, the **United States**, **Europe**, and decentralized digital asset markets. **hypoq** operationalizes an autonomous agent research loop inspired by the AutoResearch paradigm (with acknowledgments to Andrej Karpathy for early concepts) and EvoMap architecture:

$$\text{Idea} \longrightarrow \text{Plan} \longrightarrow \text{Code} \longrightarrow \text{Experiment} \longrightarrow \text{Analyse} \longrightarrow \text{Results} \longrightarrow \text{Conclusion} \longrightarrow \text{DB Sync}$$

---

## 🖥️ Platform Dashboard

![hypoq Platform Interface](assets/hypoq_ui.png)

---

## ⚡ System Architecture: Host Modeling & Docker Web Display

**hypoq** enforces a strict architectural boundary between heavy quantitative computation and lightweight web visualization:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ HOST MACHINE (Native Quantitative Research Engine)                     │
│ • Environment: Python 3.12 managed via uv (make install)               │
│ • Modeling: Polars, Statsmodels, Scikit-Learn, PyTorch, SB3            │
│ • Ingestion: scripts/download_all.py -> data/ (65 Parquet files)       │
│ • Pipeline: researches/res_*/code/run_pipeline.py                      │
│ • Typesetting: Tectonic engine -> main.tex -> main.pdf                 │
│ • Persistence: SQLite / PostgreSQL (Dense Semantic Vector Search via Gemini/HF) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Live Host Volume Mounts
                                    │ (./data, ./hypoq.db, ./researches)
┌───────────────────────────────────▼────────────────────────────────────┐
│ DOCKER CONTAINER (Web UI & Audit Display on Port 3030)                 │
│ • Service: hypoq-web (Next.js 16 + TypeScript + Tailwind CSS)          │
│ • Launch: make start / make stop                                       │
│ • Real-time display of hypothesis cards, metric audits, & PDF reader   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Core Modeling Paradigms

Every research investigation adheres to strict empirical non-negotiables:

1. **Hypothesis-First**: Formulate explicit Null ($H_0$) and Alternative ($H_1$) hypotheses prior to modeling.
2. **Point-in-Time (PIT) Causality**: Zero look-ahead bias ($\tau_{\text{info}} \le \tau_{\text{market}}$) enforced via zero-copy Polars backward joins (`join_asof`).
3. **Four Quantitative Paradigms**:
   * **Machine Learning**: Supervised return forecasting, cross-validation, and feature attribution using `scikit-learn`.
   * **Deep Learning**: Non-linear volatility nowcasting and neural sequence modeling with `pytorch`.
   * **Reinforcement Learning**: Continuous asset allocation and order execution agents using `stable-baselines3` under realistic transaction friction hurdles.
   * **Econometrics & Statistics**: Newey--West HAC robust covariance, Augmented Dickey-Fuller (ADF) unit roots, event-study Cumulative Abnormal Returns (CAR), and error-correction mean-reversion with `statsmodels`.
4. **Autonomous Serialization**: Every study compiles into a publication-ready LaTeX paper (`main.pdf`) using Tectonic and registers empirical metrics into `hypoq.db`.

---

## 📊 Empirical Research Study Examples

> **Note**: The studies below are reproducible reference examples. Every number in their tables, figures, papers and database rows is computed by `researches/<id>/code/run_pipeline.py`, and decisions come from the test, not from the author. Null results are published as such. `python run.py --audit` enforces this for every folder.

| ID | Domain | Paradigm | Status | Key Metric | Title & Focus |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **001** | Macro transmission | Econometrics | **REJECTED** | Nikkei $t = 7.70$, Euro Stoxx $t = 1.93$ | Cross-border return transmission from the US session to European and Japanese equities (the Japan leg is significant, the Europe leg fails the Bonferroni threshold) |
| **002** | Sentiment events | Econometrics | **REJECTED** | placebo $p \ge 0.07$ | Tariff-related Truth Social posts and next-day S&P 500 / VIX reactions |

All quantitative analysis, ML/DL/RL training, and LaTeX paper compilations execute directly on the host using **`uv`**:

```bash
# Install all quantitative dependencies into .venv
make install
```

### 2. Web UI Display via Docker Compose (Port 3030)

The Docker container displays the Next.js web application and dashboard, with live volume mounts to the host's `data/`, `hypoq.db`, and `researches/` directories:

```bash
# Start Next.js web display container
make start

# View container status
docker compose ps

# Stop container
make stop
```

Open **`http://localhost:3030`** (or `http://<server-ip>:3030`) in your browser.

### 3. Multi-Agent CLI Harness (`run.py`)

```bash
# Write the protocol prompt for each agent under runs/<session>/ (dry run)
python run.py --agents=claude,codex --topic="Euro area credit spread shocks and bank equity returns"

# Actually launch the agent CLIs in parallel, then audit what they produced
python run.py --agents=claude,codex --topic="..." --execute

# Verify every researches/* folder is reproducible and consistent with the database
python run.py --audit

# Dense semantic search over past hypotheses to prevent duplicate research
python run.py --search="order flow imbalance"

# Start the JSON API + dashboard (api.py)
python run.py --web --port 8080

# List all recorded hypotheses and empirical decisions
python run.py --list
python run.py --list --status=ACCEPTED
```

---

## 🛠️ Makefile Command Reference

```bash
make install          # Install host Python research stack (.venv) & start default SQLite stack
make download         # Ingest regulatory filings, WiData macro, and build Parquet datasets
make verify           # Audit schema integrity and row counts across all 65 Parquet files
make install-pg       # Launch PostgreSQL via Docker Compose & initialize semantic DB
make install-sqlite   # Launch SQLite & Next.js web container via Docker Compose
make start            # Launch Next.js web display container on Docker (port 3030)
make stop             # Terminate Next.js web display container
make parse-doc        # Parse document/PDF to Markdown (FILE=<path> [PAGES=1-5])
make clean            # Remove temporary download caches and artifacts
```

---

## ⚖️ License & Attribution

Developed by the **AI Quantitative Research Network**. Released under the MIT License.
