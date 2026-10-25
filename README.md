# hypoq (Hypothesis × Quant) — A Harness for Autonomous AI Quantitative Research

> **hypoq** is a harness that lets AI coding agents (Claude, Codex, Hermes, Pi, OMP) run falsifiable quantitative research on a **free, real, point-in-time-aware data lake**, and then *proves* that what they report is what they computed.

[![GitHub repository](https://img.shields.io/badge/GitHub-kimnt93%2Fhypoq-blue.svg?logo=github)](https://github.com/kimnt93/hypoq.git)
[![ML / DL / RL Stack](https://img.shields.io/badge/ML%2FDL%2FRL-PyTorch%20%7C%20Stable--Baselines3%20%7C%20Scikit--Learn-ee4c2c.svg)](https://pytorch.org/)
[![Econometrics](https://img.shields.io/badge/Stats-Statsmodels%20%7C%20Scipy%20%7C%20Polars-2ca02c.svg)](https://www.statsmodels.org/)
[![Deployment](https://img.shields.io/badge/Deploy-Docker%20Compose-2496ed.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LaTeX Compiler](https://img.shields.io/badge/LaTeX-Tectonic%20Engine-58a6ff.svg)](https://tectonic-typesetting.github.io/)

![hypoq dashboard](assets/hypoq_ui.png)

---

## Why this exists

Autonomous research agents are fast and fluent, which makes it easy for them to produce convincing but unfounded results (typed-in numbers, simulated data, decisions chosen to look good).
hypoq makes that structurally hard:

1. **Real data only.** `data/` holds 70 Parquet files (3.1M rows) downloaded from free, key-less public sources. Nothing is simulated, hand-typed, interpolated or zero-filled. Paid or login-only sources are excluded.
2. **Numbers are computed, never typed.** Each study is a script (`researches/<id>/code/run_pipeline.py`) that writes its tables, figures, paper and database row from the same results dictionary (`hypoq.pipeline.finalize`).
3. **Decisions come from the test.** `ACCEPTED` means the pre-registered test supports $H_1$, `REJECTED` that it does not. Null results are published as such.
4. **An audit gate.** `python run.py --audit` fails any study without code, without a PDF, or whose JSON, paper and database status disagree.
5. **A deduplication memory.** Every hypothesis is stored with a dense embedding so agents see what was already tested before they start.

```text
Idea ─► Dedup ─► Plan ─► Code ─► Experiment ─► Analyse ─► Report (LaTeX/PDF) ─► DB sync ─► Audit
```

---

## Quick start

```bash
make install                       # uv sync + verifies the stack (PyTorch, SB3, statsmodels, polars)
python run.py --list               # what has been tested so far
python run.py --search "tariff announcements and equity volatility"   # semantic dedup
make verify                        # data-quality audit of every Parquet file
make audit                         # research integrity audit
make test                          # unit tests
make api                           # JSON API + dashboard on http://127.0.0.1:8080
```

### Launch agents

```bash
# 1) Dry run: writes one protocol prompt per agent under runs/<session>/
python run.py --agents=claude,hermes --topic="Euro area credit spread shocks and bank equity returns"

# 2) Real run: launches the agent CLIs in parallel in this checkout, logs them, then audits what they produced
python run.py --agents=claude,hermes --topic="..." --execute
```

The prompt tells each agent the protocol, the integrity rules, the overlapping prior hypotheses (with status, metric and similarity) and every available dataset.
Agent commands are defined in `hypoq/harness.py` (`DEFAULT_AGENT_COMMANDS`) and can be overridden per agent with `HYPOQ_AGENT_CMD_<NAME>` (placeholders `{prompt_file}` and `{root}`).

| Agent | Default command | Status |
| :--- | :--- | :--- |
| `claude` | `claude -p … --permission-mode acceptEdits --allowedTools "Bash Edit Write Read Glob Grep"` | verified end to end |
| `hermes` | `hermes -z … --yolo` | verified end to end |
| `codex` | `codex exec -s workspace-write -C <repo> …` | flags match `--help`; needs a valid login |
| `omp` | `omp -p … --approval-mode=yolo` | flags match `--help`; needs a valid login |
| `pi` | `pi -p …` | flags match `--help`; needs a valid login |

> **Security**: agents run unsandboxed on the host with shell access. Use the harness only in a trusted checkout (or a container/VM).

---

## Studies produced with the harness

Every number below is computed by the study's own `run_pipeline.py`; re-run them all with `make seed`.

| ID | Domain | Paradigm | Status | Key result | Title |
| :--- | :--- | :--- | :---: | :--- | :--- |
| [`001`](researches/res_001_global_cross_asset_volatility_spillover) | Macro transmission | Econometrics | **REJECTED** | Nikkei $t=7.70$; Euro Stoxx $t=1.93$ | Cross-Border Return Transmission from the US Session to European and Japanese Equity Markets |
| [`002`](researches/res_002_vip_tariff_posts_event_study) | Sentiment events | Econometrics / event study | **REJECTED** | placebo $p\ge$ 0.07 | Tariff-Related Executive Social Media Posts and Next-Day US Equity and Implied Volatility Reactions |
| [`003`](researches/res_003_us_megacap_ml_return_ranking) | Factor / return ranking | Machine learning | **ACCEPTED** | GBM IC 0.0133, $p=0.036$ | Gradient-Boosted Ranking of Next-Week Returns in US Mega-Cap Equities: A Walk-Forward Test |
| [`004`](researches/res_004_volatility_nowcasting_lstm_vs_har) | Volatility forecasting | Deep learning | **INCONCLUSIVE** | S&P 500: DM $p=0.742$; Bitcoin: DM $p<0.001$ | Recurrent Neural Networks versus HAR for Five-Day-Ahead Volatility Forecasts of Equities and Bitcoin |
| [`005`](researches/res_005_ppo_multi_asset_allocation) | Asset allocation | Reinforcement learning | **REJECTED** | Sharpe PPO 0.18 vs EW 1.17 | Proximal Policy Optimisation for Multi-Asset ETF Allocation under Transaction Costs |
| [`006`](researches/res_006_earnings_8k_announcement_reaction) | Earnings events | Event study | **ACCEPTED** | mean $|$CAR$|$ 4.59% vs placebo 1.70% | Earnings Filings and Price Reactions: An Event Study of SEC 8-K Item 2.02 Announcements for US Mega-Caps |

`ACCEPTED` = the pre-registered test supports $H_1$; `REJECTED` = it does not; `INCONCLUSIVE` = the evidence is mixed across the pre-specified tests. Study 001 is rejected because its Europe leg fails the Bonferroni threshold, and study 004 is inconclusive because the LSTM beats HAR for Bitcoin but not for the S&P 500.

---

## Data lake

70 Parquet files across Asia, the United States, Europe, crypto, Vietnam and raw community text. Full schemas, provenance, refresh mode and caveats are in **[`DATA.md`](DATA.md)**.

| Folder | Content | Source |
| :--- | :--- | :--- |
| `data/market/` | Indices, mega-cap equities and ETFs for US, EU, JP, KR, HK, CN, Asia aggregates; sector, factor and Vietnamese market data | Yahoo Finance, VNDirect |
| `data/macro/` | Yield curve, Fed balance sheet, FOMC decisions, credit spreads, volatility indices, commodities, FX, Vietnamese macro | FRED, Yahoo Finance, FOMC statements, WiData snapshot |
| `data/crypto/` | Top crypto assets and BTC/ETH aggregate indicators | Yahoo Finance |
| `data/community/` | Truth Social posts, Fed statements and speeches, SEC 8-K filings, Hacker News, Stack Exchange, Mastodon (raw text, no sentiment) | public APIs |
| `data/financials/` | Global fundamentals snapshot, Vietnamese statements and ratios | Yahoo Finance, VNDirect |
| `data/market_structure/` | Vietnamese free float, baskets, ownership and corporate networks | exchange notices (snapshot), VNDirect |
| `data/cross_asset/` | Cross-asset daily panel | Yahoo Finance |

Refresh everything live with `make download` (about ten minutes), then `python scripts/update_data_docs.py` to resync the numbers in `DATA.md`.
Two environment variables matter: `HYPOQ_CONTACT` (your own email, required politely by SEC EDGAR's fair-access policy) and `HYPOQ_VN_STRUCTURE_DIR` (location of the Vietnamese index-structure CSV snapshot).

**Deliberately excluded**: Reddit and X (an account is required), paid fundamentals and consensus estimates, ETF NAV/AUM, and any synthetic series.

**Point-in-time warnings** that agents must respect: Yahoo equity universes are today's large caps (survivorship bias); company fundamentals snapshots are as of the download date; Vietnamese statements carry no publication date and mix actuals with analyst estimates; many macro series are dated at the end of their reference period and published later. See `DATA.md` §1.2.

---

## Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│ HOST (native research engine)                                          │
│ • scripts/download_all.py  -> data/*.parquet (free public sources)     │
│ • run.py / hypoq/harness.py -> prompts, agent launch, audit            │
│ • researches/*/code/run_pipeline.py -> results, figures, main.pdf      │
│ • hypoq/db.py -> SQLite (hypoq.db) or PostgreSQL (DATABASE_URL)        │
│ • api.py -> JSON API + dashboard (read-only, localhost by default)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ volume mounts (./data, ./hypoq.db, ./researches)
┌───────────────────────────────────▼────────────────────────────────────┐
│ DOCKER (Next.js dashboard, port 3030): hypothesis cards, PDFs, datasets│
└────────────────────────────────────────────────────────────────────────┘
```

| Path | Purpose |
| :--- | :--- |
| `hypoq/` | `db.py` (dual backend, embeddings, `upsert_hypothesis`), `harness.py` (sessions, prompts, `audit_research`), `pipeline.py` (`finalize`), `report.py` (LaTeX), `embeddings.py` |
| `scripts/` | Ingestion (`ingest_*.py`, `yf_common.py`, `vnd_common.py`), `verify_datasets.py`, `update_data_docs.py`, `seed_database.py` |
| `researches/` | One folder per study: `code/`, `results/`, `figures/`, `main.tex`, `main.pdf`, `references.bib` |
| `.agents/skills/` | 18 AgentSkills (data exploration, event studies, ML/DL/RL, PIT construction, LaTeX, dedup, the hypoq loop) |
| `templates/` | LaTeX template |
| `web/`, `Dockerfile`, `docker-compose.yml` | Next.js dashboard and PostgreSQL option |
| `tests/` | pytest suite (database, audit, harness, API security, data quality) |

### Dashboard / API

```bash
make api                 # python run.py --web  -> http://127.0.0.1:8080  (set HYPOQ_HOST to expose it)
make install-sqlite      # Next.js dashboard in Docker on http://localhost:3030
make install-pg          # same, with PostgreSQL + pgvector
```

Endpoints (`api.py`): `/api/hypotheses?q=&status=`, `/api/datasets`, `/api/dataset_preview?path=`, `/api/reports`, `/pdf?path=`. File endpoints are restricted to `data/*.parquet` and `researches/**/*.pdf`.

---

## Writing a new study by hand

```bash
mkdir -p researches/res_NNN_<slug>/{code,figures,results}
# copy researches/res_003_us_megacap_ml_return_ranking/code/run_pipeline.py as a starting point,
# change the data, the test and the decision rule, then:
python researches/res_NNN_<slug>/code/run_pipeline.py
python run.py --audit
```

`finalize(...)` writes `results/empirical_results.json`, `main.tex`, `references.bib`, compiles `main.pdf` with Tectonic and registers the hypothesis and experiment idempotently.

---

## Command reference

```bash
make install   make download   make verify   make audit   make seed   make test   make api
make install-pg   make install-sqlite   make start   make stop   make parse-doc FILE=<path>   make clean
python run.py --agents=<a,b> --topic="…" [--execute]   python run.py --audit   python run.py --search "…"   python run.py --list [--status=ACCEPTED]
```

## Limitations

- The studies are small reference implementations, not investment advice: universes are survivorship-biased, test windows are short, transaction costs are crude.
- Agent CLIs other than `claude` and `hermes` were not verified end to end on the author's machine (expired logins).
- Vietnamese macro and index-structure tables are dated snapshots (see `DATA.md` §1.2), not live downloads.
- The harness cannot stop an agent from writing a dishonest *pipeline*; the audit checks structure and consistency, so review the code of studies you rely on.

## License

MIT, see [`LICENSE`](LICENSE).
