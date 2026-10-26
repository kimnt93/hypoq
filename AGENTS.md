# AGENTS.md — Autonomous Agent Operating Protocol for hypoq (Hypothesis × Quant)

Welcome to **hypoq** (**Hypothesis $\times$ Quant**), a harness for autonomous AI quantitative research on a free, real data lake.

---

## 1. Core Mandate for AI Agents

Whenever an AI agent (Claude, Codex, Hermes, Pi, OMP) is awakened in this repository, it must strictly orient itself by reading the following foundational files:

1. **[`RESEARCH.md`](RESEARCH.md)**: Defines the research taxonomy across active quantitative domains, required modeling paradigms (Machine Learning with `scikit-learn`, Deep Learning with `pytorch`, Reinforcement Learning with `stable-baselines3`, and Econometrics with `statsmodels`), and strict causality invariants.
2. **[`DATA.md`](DATA.md)**: Details the quantitative data lake (70 Parquet files under `data/`, 3.1M rows, strictly < 100MB per file for GitHub safety), schema dictionaries, and the standardized jurisdiction prefix standard (`cn_`, `hk_`, `jp_`, `kr_`, `asia_`, `us_`, `eu_`, `global_`, `crypto_`, `vn_`).
3. **[`.agents/skills/`](.agents/skills/)**: Contains 18 self-contained skills adhering to the [AgentSkills.io](https://agentskills.io) standard:
   * `chunked-data-exploration`: Exploring, scanning, filtering, and streaming partitioned / chunked Parquet files with zero RAM bloat.
   * `world-data-taxonomy-modeling`: Multi-domain modeling across FRED macro, 11 GICS sectors, factors, credit spreads, and global valuation ratios.
   * `vip-sentiment-event-studies`: Quantifying market reactions, CAR, price jumps, and volatility surges around Trump (Truth Social) and other executive announcements.
   * `global-macro-cross-asset`: Cross-border macroeconomic transmission, sovereign yield curves, FX pass-through, and commodity shock propagation.
   * `crypto-quantitative-modeling`: Cryptocurrency microstructure, high-volatility regime modeling, crypto-macro spillover, and tail risk (VaR/CVaR).
   * `global-equity-factor-modeling`: Multi-region equity factor modeling (Fama-French, momentum, quality, low-vol) across US, EU, and Asia.
   * `ml-dl-rl-modeling`: Scikit-Learn, PyTorch, and Stable-Baselines3 implementations.
   * `hypothesis-testing-stats`: Metrics and inference catalogue (performance, risk, forecast accuracy, IC, HAC, bootstrap/placebo, Diebold-Mariano, Deflated Sharpe, multiple testing, event studies, walk-forward splits) with a tested reference library to copy from, and rules for writing a missing metric correctly in Polars/NumPy.
   * `polars-fast-data`: High-performance zero-copy and Point-in-Time backward alignment.
   * `data-discovery`: Data lake auditing, schema inspection, and continuous ingestion.
   * `data-explore-pit-builder`: Point-in-Time event dataset construction (Fed FOMC, central bank rates).
   * `data-exploration-js`: JavaScript / Arquero / `node:sqlite` data manipulation.
   * `charts-visualization`: High-DPI publication chart rendering with Seaborn.
   * `latex-report-generation`: Compiling `templates/main.tex` to `main.pdf` via Tectonic.
   * `arxiv-paper-search`: Literature exploration and BibTeX extraction.
   * `paper-pdf-to-md`: PDF-to-Markdown document parsing using `anymd`.
   * `hypoq-loop`: Execution protocol for the 8-stage research cycle.
   * `dedup-hypothesis-search`: Dense semantic vector deduplication in `hypoq.db`.
---

## 2. Autonomous Research Execution Protocol

All empirical research investigations must execute the **hypoq Cycle**:
$$\text{Idea} \longrightarrow \text{Plan} \longrightarrow \text{Code} \longrightarrow \text{Experiment} \longrightarrow \text{Analyse} \longrightarrow \text{Results} \longrightarrow \text{Conclusion} \longrightarrow \text{DB Sync}$$

### Non-Negotiables:
1. **Computed, never typed (enforced by `python run.py --audit`)**: every number in a table, figure, manuscript, `results/empirical_results.json` or database row must be produced by `researches/<folder>/code/run_pipeline.py`. Never hand-type results, simulate or interpolate data, zero-fill missing values, or choose the decision by hand. The decision (`ACCEPTED` = $H_1$ supported, `REJECTED` = not supported, `INCONCLUSIVE`) is derived from the pre-specified test. Null results are valid results.
2. **Real, free data only**: use datasets in `data/` or add a new source through a script in `scripts/` that downloads it from a free public endpoint (no synthetic rows, no paid feeds). Read `DATA.md` §1.2 for provenance and point-in-time caveats before using a dataset.
3. **English-Only by Default**: all hypothesis statements, titles, null/alt formulations and findings are written in academic English.
4. **Hypothesis-First**: formulate explicit Null ($H_0$) and Alternative ($H_1$) hypotheses, the test and the multiple-testing correction *before* looking at results.
5. **Deduplication Check**: run `python run.py --search="..."` before writing code; do not repeat a `REJECTED` idea without a new angle.
6. **Python-Only Pipelines**: Polars, Scikit-Learn, PyTorch, Stable-Baselines3 and Statsmodels in `code/run_pipeline.py`.
7. **Strict Point-in-Time Integrity**: zero look-ahead. Features at day $t$ use information published by $t$; low-frequency macro and accounting data are matched with `join_asof(..., strategy="backward")` after lagging by their publication delay; train/test splits are chronological with an embargo.
8. **One finishing call**: end every pipeline with `hypoq.pipeline.finalize(...)`: it writes the results JSON, the raw record `results/hypothesis_record.json` (the versioned form of the hypothesis; the database file itself is git-ignored and rebuilt by `make db-import`) and the paper (`templates/main.tex` style, `hypoq.report.write_paper`), compiles `main.pdf` with Tectonic and registers the hypothesis and experiment idempotently (`hypoq.db.upsert_hypothesis`) in SQLite (`hypoq.db`) or PostgreSQL (`DATABASE_URL`).
9. **The `hypoq/` package, `scripts/` and `api.py` are frozen during research.** `hypoq` only discovers, saves and loads hypotheses. All study logic (data prep, models, metrics, tests, figures) goes in `researches/<id>/code/run_pipeline.py`; if a metric or test is missing, copy it from `.agents/skills/hypothesis-testing-stats/scripts/quant_metrics.py` or write it in your pipeline following that skill. New *data* sources are the one exception: add a script under `scripts/` and document it in `DATA.md`.
10. **Definition of done**: `python run.py --audit` passes for your folder and `python run.py --list` shows your hypothesis with the computed status. Run `make test` if you touched shared code.

---

## 3. Operational CLI & System Commands

```bash
make install     # uv sync + verification of the research stack, starts the default SQLite/Docker stack
make download    # refresh every free data source (about 10 minutes); then: python scripts/update_data_docs.py
make verify      # data-quality audit: duplicates, all-null columns, OHLC sanity, staleness, short series
make audit       # research integrity audit (code, JSON, PDF and database must agree)
make seed        # re-run every researches/*/code/run_pipeline.py and refresh the database
make db-import   # load researches/*/results/hypothesis_record.json into the git-ignored database and embed them (part of make install)
make db-migrate  # re-embed after changing EMBEDDING_PROVIDER / EMBEDDING_MODEL (FORCE=1 redoes all)
make test        # pytest suite
make api         # JSON API + dashboard on http://127.0.0.1:8080
make install-pg / install-sqlite / start / stop   # Docker Compose stack (Next.js dashboard, optional PostgreSQL)

python run.py --agents=claude,hermes --topic="..."              # write one protocol prompt per agent (dry run)
python run.py --agents=claude,hermes --topic="..." --execute    # launch the agent CLIs, then audit new research folders
python run.py --search="interest rate shock transmission to ETFs"   # dense semantic search of past hypotheses
python run.py --list [--status=ACCEPTED]
python run.py --audit
```

Reference implementations to copy: `researches/res_003_us_megacap_ml_return_ranking` (ML), `res_004_volatility_nowcasting_lstm_vs_har` (DL), `res_005_ppo_multi_asset_allocation` (RL), `res_001`, `res_002`, `res_006`, `res_007` (econometrics and event studies), `res_008`, `res_010`, `res_012`, `res_013`, `res_015`, `res_019`, `res_021` (predictive regressions with HAC and circular-shift placebo), `res_009`, `res_011`, `res_018` (strategy and portfolio-sort tests with stationary bootstrap), `res_014`, `res_016`, `res_017`, `res_020` (safe-haven, calendar and return-decomposition tests), `res_022` (walk-forward ML classification with a permutation test).

Configuration: copy `.env.example` to `.env` (database, embeddings, ports, SEC contact); `make` and `docker compose` read it automatically.
