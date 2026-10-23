---
name: autoresearch-loop
description: Orchestrate the end-to-end Karpathy AutoResearch cycle from idea inception, planning, and coding to experimentation, econometric analysis, and SQLite synchronization.
version: 1.0.0
author: kim / vnresearch
tags: [autoresearch, loop, workflow, karpathy, evomap, lifecycle]
---

# AutoResearch Loop Skill

## Overview
This skill operationalizes the autonomous quantitative research lifecycle inspired by the Karpathy AutoResearch pattern and EvoMap architecture:
$$\text{Idea} \longrightarrow \text{Plan} \longrightarrow \text{Code} \longrightarrow \text{Experiment} \longrightarrow \text{Analyse} \longrightarrow \text{Results} \longrightarrow \text{Conclusions} \longrightarrow \text{Database Sync}$$

## The 8-Stage Lifecycle

```text
 1. IDEA        ───>  Formulate falsifiable hypothesis (H0 vs H1) across 6 research domains.
 2. DEDUP       ───>  Query SQLite FTS5 database to ensure idea novelty. If duplicate, refine.
 3. PLAN        ───>  Identify datasets in data/, define econometric equation, set p-value hurdle.
 4. CODE        ───>  Write self-contained script under research folder: code/run_pipeline.py.
 5. EXPERIMENT  ───>  Execute pipeline using Polars & Statsmodels. Log execution duration and metrics.
 6. ANALYSE     ───>  Verify stationarity (ADF), HAC errors, placebo robustness. Render PNG figures.
 7. REPORT      ───>  Populate Research Future LaTeX template and compile to main.pdf via Tectonic.
 8. DB SYNC     ───>  Save hypothesis status (ACCEPTED / REJECTED) and metrics to autoresearch.db.
```

## Step-by-Step Autonomous Execution

### Step 1: Idea Inception & Deduplication
Check `autoresearch.db` to prevent duplicating existing or rejected hypotheses:
```python
from autoresearch.db import search_hypotheses

matches = search_hypotheses("order flow imbalance return predictability")
if matches and matches[0]["score"] > 0.8:
    print(f"Idea closely related to existing test #{matches[0]['id']}: {matches[0]['title']}. Refining angle...")
```

### Step 2: Planning & Folder Scaffold
Create dedicated research folder under `researches/`:
```bash
mkdir -p researches/<research_id>/{code,figures,results}
```

### Step 3: Implement Pipeline (`code/run_pipeline.py`)
Requirements for the empirical script:
- Load data using Polars with strict PIT causality.
- Compute test statistics ($t$, $p$-value, HAC errors).
- Generate figures into `figures/*.png`.
- Output raw metrics to `results/empirical_results.json`:
  ```json
  {
    "hypothesis": "Order flow imbalance predicts 1-day VN30 returns",
    "decision": "ACCEPTED",
    "t_stat": 3.42,
    "p_value": 0.0006,
    "r_squared": 0.048,
    "sharpe_net": 1.25
  }
  ```

### Step 4: Compile LaTeX Manuscript
Compile `main.tex` into `main.pdf` using Tectonic:
```bash
cd researches/<research_id> && /home/kim/.local/bin/tectonic main.tex
```

### Step 5: Database Synchronization
Log the hypothesis outcome and report path into `autoresearch.db`:
```python
from autoresearch.db import record_hypothesis

record_hypothesis(
    session_id=session_id,
    agent_name=agent_name,
    title=title,
    statement=statement,
    null_h=null_h,
    alt_h=alt_h,
    domain=domain,
    status="ACCEPTED",  # or REJECTED / INCONCLUSIVE
    metric_name="HAC t-stat",
    metric_value=3.42,
    p_value=0.0006,
    summary_findings=summary,
    report_dir=f"researches/{research_id}",
    pdf_path=f"researches/{research_id}/main.pdf"
)
```
