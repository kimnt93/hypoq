import json
import sqlite3

import numpy as np

import hypoq.db as db
from hypoq.db import build_record, import_record, list_hypotheses, reembed_all, search_hypotheses

HYPO = dict(title="Record study", statement="s", null_hypothesis="n", alt_hypothesis="a", domain="d", dataset_dependencies=["data/x.parquet"],
            methodology="m", metric_name="t", metric_value=1.5, p_value=0.02, confidence_level=0.95, status="ACCEPTED", summary_findings="found",
            report_dir="researches/res_x", pdf_path="researches/res_x/main.pdf", agent_name="hypoq-pipeline", session_id="sess_test")


def _record(**over):
    return build_record({**HYPO, **over}, {"decision": "ACCEPTED", "x": 1.0}, "researches/res_x/code/run_pipeline.py")


def _model_ids(path):
    return {r[0]: r[1] for r in sqlite3.connect(path).execute("SELECT model_name, COUNT(*) FROM hypothesis_embeddings GROUP BY 1")}


def test_record_has_no_vectors_ids_or_timestamps():
    rec = _record()
    text = json.dumps(rec)
    assert rec["schema"] == "hypoq.record/1" and "vector" not in text and "created_at" not in text and "embedding" not in text


def test_import_is_idempotent_and_creates_embedding(db_path):
    a = import_record(_record(), db_path)
    b = import_record(_record(), db_path)
    rows = list_hypotheses(db_path=db_path)
    assert a == b and len(rows) == 1 and rows[0]["status"] == "ACCEPTED"
    assert _model_ids(db_path) == {"test:fake-16": 1}
    assert sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1


def test_import_updates_changed_record_and_refreshes_vector(db_path):
    import_record(_record(), db_path)
    before = sqlite3.connect(db_path).execute("SELECT vector FROM hypothesis_embeddings").fetchone()[0]
    import_record(_record(statement="a different statement", status="REJECTED"), db_path)
    row = list_hypotheses(db_path=db_path)[0]
    after = sqlite3.connect(db_path).execute("SELECT vector FROM hypothesis_embeddings").fetchone()[0]
    assert row["statement"] == "a different statement" and row["status"] == "REJECTED" and before != after


def test_changing_embedding_method_is_detected_and_migrated(db_path, monkeypatch):
    import_record(_record(), db_path)
    import_record(_record(title="Second study"), db_path)
    assert reembed_all(db_path)["re_embedded"] == 0                      # nothing stale yet
    monkeypatch.setattr(db, "get_model_id", lambda: "test:other-32")
    monkeypatch.setattr(db, "get_embedding", lambda text: np.ones(32, dtype="float32") / np.sqrt(32))
    stats = reembed_all(db_path)
    assert stats["re_embedded"] == 2 and stats["previous_models"] == {"test:fake-16": 2}
    assert _model_ids(db_path) == {"test:other-32": 2}
    assert reembed_all(db_path)["re_embedded"] == 0                      # idempotent
    assert reembed_all(db_path, force=True)["re_embedded"] == 2


def test_search_migrates_on_demand_instead_of_recursing(db_path, monkeypatch):
    import_record(_record(), db_path)
    monkeypatch.setattr(db, "get_model_id", lambda: "test:other-32")
    monkeypatch.setattr(db, "get_embedding", lambda text: np.ones(32, dtype="float32") / np.sqrt(32))
    hits = search_hypotheses("anything", db_path=db_path)
    assert len(hits) == 1 and _model_ids(db_path) == {"test:other-32": 1}


def test_search_returns_empty_when_embedding_method_is_unavailable(db_path, monkeypatch):
    import_record(_record(), db_path)
    monkeypatch.setattr(db, "get_model_id", lambda: "test:broken")

    def broken(text):
        raise RuntimeError("model not available")

    monkeypatch.setattr(db, "get_embedding", broken)
    assert search_hypotheses("anything", db_path=db_path) == []


def test_embedding_failure_never_fabricates_a_vector(monkeypatch):
    import hypoq.embeddings as emb
    monkeypatch.setattr(emb, "_get_hf_model", lambda: False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    try:
        emb.get_embedding("text")
    except RuntimeError as exc:
        assert "could not be loaded" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
