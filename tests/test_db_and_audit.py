import json
import os

from hypoq.db import list_hypotheses, upsert_hypothesis
from hypoq.harness import REPO_ROOT, audit_research


def _register(db_path, title="T", status="REJECTED", report_dir="researches/x"):
    return upsert_hypothesis(
        title=title, statement="s", null_hypothesis="n", alt_hypothesis="a", domain="d",
        status=status, metric_name="m", metric_value=1.0, p_value=0.5, summary_findings="f",
        report_dir=report_dir, db_path=db_path,
    )


def test_upsert_is_idempotent(db_path):
    first = _register(db_path, status="PENDING")
    second = _register(db_path, status="REJECTED")
    rows = list_hypotheses(db_path=db_path)
    assert first == second and len(rows) == 1 and rows[0]["status"] == "REJECTED"


def _make_study(tmp_path, with_code=True, decision="REJECTED"):
    d = tmp_path / "res_test"
    for sub in ("code", "results", "figures"):
        (d / sub).mkdir(parents=True)
    if with_code:
        (d / "code" / "run_pipeline.py").write_text("# writes empirical_results.json\nupsert_hypothesis()\n")
    (d / "results" / "empirical_results.json").write_text(json.dumps({"decision": decision}))
    (d / "results" / "hypothesis_record.json").write_text(json.dumps({"schema": "hypoq.record/1", "hypothesis": {"status": decision}}))
    for f in ("main.tex", "main.pdf", "references.bib", "figures/f.png"):
        (d / f).write_text("x")
    return str(d)


def test_audit_passes_consistent_study(tmp_path, db_path):
    d = _make_study(tmp_path)
    _register(db_path, report_dir=os.path.relpath(d, REPO_ROOT), status="REJECTED")
    report = audit_research(d, db_path)
    assert report["ok"], report["problems"]


def test_audit_flags_study_without_code(tmp_path, db_path):
    d = _make_study(tmp_path, with_code=False)
    _register(db_path, report_dir=os.path.relpath(d, REPO_ROOT))
    report = audit_research(d, db_path)
    assert not report["ok"] and any("run_pipeline.py" in p for p in report["problems"])


def test_audit_flags_status_mismatch(tmp_path, db_path):
    d = _make_study(tmp_path, decision="ACCEPTED")
    _register(db_path, report_dir=os.path.relpath(d, REPO_ROOT), status="REJECTED")
    report = audit_research(d, db_path)
    assert not report["ok"] and any("!=" in p for p in report["problems"])


def test_audit_flags_missing_database_row(tmp_path, db_path):
    d = _make_study(tmp_path)
    report = audit_research(d, db_path)
    assert not report["ok"] and any("database" in p for p in report["problems"])


def test_audit_accepts_pipelines_that_use_finalize(tmp_path, db_path):
    d = _make_study(tmp_path)
    open(os.path.join(d, "code", "run_pipeline.py"), "w").write("from hypoq.pipeline import finalize\nfinalize()\n")
    _register(db_path, report_dir=os.path.relpath(d, REPO_ROOT))
    assert audit_research(d, db_path)["ok"]


def test_audit_flags_missing_record(tmp_path, db_path):
    d = _make_study(tmp_path)
    os.remove(os.path.join(d, "results", "hypothesis_record.json"))
    _register(db_path=db_path, report_dir=os.path.relpath(d, REPO_ROOT))
    report = audit_research(d, db_path)
    assert not report["ok"] and any("hypothesis_record.json" in p for p in report["problems"])
