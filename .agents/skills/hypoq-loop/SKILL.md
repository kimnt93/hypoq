---
name: hypoq-loop
description: Orchestrate the end-to-end hypoq cycle from idea inception, deduplication, planning and coding to computed experiments, reporting, an integrity audit and database synchronization.
version: 1.0.0
author: hypoq
tags: [hypoq, loop, workflow, evomap, lifecycle]
---

# hypoq Loop Skill

## Overview
This skill operationalizes the autonomous quantitative research lifecycle inspired by the AutoResearch paradigm and EvoMap architecture:

## The 8-Stage Lifecycle

```text
 1. IDEA        ───>  Formulate falsifiable hypothesis (H0 vs H1) in English.
 2. DEDUP       ───>  Dense semantic search over hypoq.db to ensure novelty. If duplicate, refine the angle.
 3. PLAN        ───>  Identify datasets in data/, define the test, the multiple-testing correction and the decision rule BEFORE looking at results.
 4. CODE        ───>  Write researches/<id>/code/run_pipeline.py (the only place numbers are produced).
 5. EXPERIMENT  ───>  Run it with Polars / Statsmodels / scikit-learn / PyTorch / Stable-Baselines3.
 6. ANALYSE     ───>  Placebo / bootstrap robustness, HAC errors, benchmarks; render PNG figures.
 7. REPORT      ───>  Format every number from the results dict into main.tex and compile main.pdf with Tectonic.
 8. DB SYNC     ───>  Register the hypothesis and experiment in hypoq.db (idempotent).
```

Status vocabulary: `ACCEPTED` = H1 is supported by the pre-registered test, `REJECTED` = H1 is not supported, `INCONCLUSIVE` = the test could not decide.
The decision is computed from the test statistic in code; null results are valid and must be published as such.

## Step-by-Step Autonomous Execution

### Step 1: Idea Inception & Deduplication
```python
from hypoq.db import search_hypotheses

matches = search_hypotheses("order flow imbalance return predictability", top_k=5)
if matches and matches[0]["similarity_score"] > 0.8:
    print(f"Closely related to #{matches[0]['id']} ({matches[0]['status']}): {matches[0]['title']}. Refine the angle.")
```
or `python run.py --search "order flow imbalance return predictability"`.

### Step 2: Planning & Folder Scaffold
```bash
mkdir -p researches/res_NNN_<slug>/{code,figures,results}
```

### Step 3: Implement the Pipeline (`code/run_pipeline.py`)
- Load data with Polars under strict point-in-time causality (features at day t use information up to t only).
- Compute test statistics (t, p, HAC errors, placebo p) and a computed decision. Correct for multiple tests.
- Save figures to `figures/*.png`.
- Finish by calling `hypoq.pipeline.finalize(...)`, which writes `results/empirical_results.json`, `main.tex`, `references.bib`,
  compiles `main.pdf`, and registers the hypothesis and experiment (see `researches/res_003_us_megacap_ml_return_ranking/code/run_pipeline.py`).

### Step 4: Audit
```bash
python run.py --audit      # fails for any study without code, without a PDF, or whose JSON and database status disagree
python run.py --list
```

### Manual registration (only when not using `finalize`)
```python
from hypoq.db import upsert_hypothesis

upsert_hypothesis(
    title=title, statement=statement, null_hypothesis=null_h, alt_hypothesis=alt_h, domain=domain,
    session_id=session_id, agent_name=agent_name, status=decision,        # computed, never typed
    metric_name="HAC t-stat", metric_value=t_stat, p_value=p_value, summary_findings=summary,
    report_dir=f"researches/{research_id}", pdf_path=f"researches/{research_id}/main.pdf",
)
```
