"""
Shared finalisation step for research pipelines.

A pipeline computes its numbers, then calls `finalize(...)` which, in this order:
1. writes results/empirical_results.json from the computed `results` dict
2. writes main.tex / references.bib (every number must be formatted from `results` by the caller)
3. compiles main.pdf with Tectonic
4. registers the hypothesis idempotently and logs the experiment run in the database
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from hypoq.db import record_experiment, upsert_hypothesis
from hypoq.report import compile_pdf, write_paper

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VALID_DECISIONS = ("ACCEPTED", "REJECTED", "INCONCLUSIVE")


def finalize(
    research_dir: str,
    started_at: float,
    title: str,
    decision: str,
    results: Dict[str, Any],
    abstract: Dict[str, str],
    body: str,
    bib: str,
    statement: str,
    null_hypothesis: str,
    alt_hypothesis: str,
    domain: str,
    datasets: List[str],
    methodology: str,
    metric_name: str,
    metric_value: float,
    p_value: Optional[float],
    summary_findings: str,
    session_id: str = "sess_hypoq_examples",
    agent_name: str = "hypoq-pipeline",
) -> int:
    if decision not in VALID_DECISIONS:
        raise ValueError(f"decision must be one of {VALID_DECISIONS}, got {decision!r}")
    results = {"title": title, "decision": decision, **results}
    os.makedirs(os.path.join(research_dir, "results"), exist_ok=True)
    with open(os.path.join(research_dir, "results", "empirical_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    write_paper(research_dir, title, abstract, body, bib)
    pdf = compile_pdf(research_dir)
    if pdf is None:
        raise RuntimeError(f"main.tex in {research_dir} failed to compile; fix the LaTeX (escape &, %, _ with hypoq.report.latex_escape)")
    rel = os.path.relpath(research_dir, PROJECT_ROOT)
    hid = upsert_hypothesis(
        title=title,
        statement=statement,
        null_hypothesis=null_hypothesis,
        alt_hypothesis=alt_hypothesis,
        domain=domain,
        session_id=session_id,
        agent_name=agent_name,
        dataset_dependencies=datasets,
        methodology=methodology,
        metric_name=metric_name,
        metric_value=metric_value,
        p_value=p_value,
        status=decision,
        summary_findings=summary_findings,
        report_dir=rel,
        pdf_path=f"{rel}/main.pdf" if pdf else None,
    )
    record_experiment(
        hypothesis_id=hid,
        session_id=session_id,
        agent_name=agent_name,
        code_path=f"{rel}/code/run_pipeline.py",
        duration_seconds=time.time() - started_at,
        metrics=results,
    )
    print(f">>> {decision}: hypothesis #{hid} ({metric_name} = {metric_value:.4g}, p = {p_value})")
    return hid
