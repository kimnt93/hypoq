"""
SQLite Database & FTS5 Search Tracking Engine for AutoResearch.
Manages research sessions, hypotheses, experiments, and deduplication search.
"""

import sqlite3
import json
import os
import re
from datetime import datetime
from typing import List, Dict, Optional, Any

DEFAULT_DB_PATH = os.environ.get("AUTORESEARCH_DB", "autoresearch.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    agents TEXT NOT NULL,
    topic TEXT,
    status TEXT DEFAULT 'ACTIVE',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    agent_name TEXT,
    title TEXT NOT NULL,
    statement TEXT NOT NULL,
    null_hypothesis TEXT NOT NULL,
    alt_hypothesis TEXT NOT NULL,
    domain TEXT NOT NULL,
    dataset_dependencies TEXT,
    methodology TEXT,
    metric_name TEXT,
    metric_value REAL,
    p_value REAL,
    confidence_level REAL DEFAULT 0.95,
    status TEXT NOT NULL DEFAULT 'PENDING',
    summary_findings TEXT,
    report_dir TEXT,
    pdf_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id INTEGER,
    session_id TEXT,
    agent_name TEXT,
    code_path TEXT,
    start_time TEXT,
    end_time TEXT,
    duration_seconds REAL,
    metrics_json TEXT,
    status TEXT DEFAULT 'SUCCESS',
    log_output TEXT,
    FOREIGN KEY(hypothesis_id) REFERENCES hypotheses(id),
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

-- Content-backed FTS5 virtual table for full-text deduplication search
CREATE VIRTUAL TABLE IF NOT EXISTS hypotheses_fts USING fts5(
    title,
    statement,
    null_hypothesis,
    alt_hypothesis,
    summary_findings,
    domain,
    content='hypotheses',
    content_rowid='id'
);

-- Triggers to automatically keep FTS5 synchronized with hypotheses table
CREATE TRIGGER IF NOT EXISTS hypotheses_ai AFTER INSERT ON hypotheses BEGIN
    INSERT INTO hypotheses_fts(rowid, title, statement, null_hypothesis, alt_hypothesis, summary_findings, domain)
    VALUES (new.id, new.title, new.statement, new.null_hypothesis, new.alt_hypothesis, new.summary_findings, new.domain);
END;

CREATE TRIGGER IF NOT EXISTS hypotheses_ad AFTER DELETE ON hypotheses BEGIN
    INSERT INTO hypotheses_fts(hypotheses_fts, rowid, title, statement, null_hypothesis, alt_hypothesis, summary_findings, domain)
    VALUES ('delete', old.id, old.title, old.statement, old.null_hypothesis, old.alt_hypothesis, old.summary_findings, old.domain);
END;

CREATE TRIGGER IF NOT EXISTS hypotheses_au AFTER UPDATE ON hypotheses BEGIN
    INSERT INTO hypotheses_fts(hypotheses_fts, rowid, title, statement, null_hypothesis, alt_hypothesis, summary_findings, domain)
    VALUES ('delete', old.id, old.title, old.statement, old.null_hypothesis, old.alt_hypothesis, old.summary_findings, old.domain);
    INSERT INTO hypotheses_fts(rowid, title, statement, null_hypothesis, alt_hypothesis, summary_findings, domain)
    VALUES (new.id, new.title, new.statement, new.null_hypothesis, new.alt_hypothesis, new.summary_findings, new.domain);
END;
"""


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes the SQLite database schema and FTS5 tables."""
    conn = get_connection(db_path)
    with conn:
        conn.executescript(SCHEMA_SQL)
    conn.close()


def create_session(
    session_id: str,
    agents: List[str],
    topic: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Registers a new research session."""
    init_db(db_path)
    conn = get_connection(db_path)
    now = datetime.utcnow().isoformat() + "Z"
    agents_str = json.dumps(agents) if isinstance(agents, list) else str(agents)
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions (session_id, created_at, agents, topic, status, notes)
            VALUES (?, ?, ?, ?, 'ACTIVE', ?)
            """,
            (session_id, now, agents_str, topic, notes),
        )
    conn.close()
    return session_id


def record_hypothesis(
    title: str,
    statement: str,
    null_hypothesis: str,
    alt_hypothesis: str,
    domain: str,
    session_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    dataset_dependencies: Optional[List[str]] = None,
    methodology: Optional[str] = None,
    metric_name: Optional[str] = None,
    metric_value: Optional[float] = None,
    p_value: Optional[float] = None,
    confidence_level: float = 0.95,
    status: str = "PENDING",
    summary_findings: Optional[str] = None,
    report_dir: Optional[str] = None,
    pdf_path: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Inserts a new hypothesis and returns its ID."""
    init_db(db_path)
    conn = get_connection(db_path)
    now = datetime.utcnow().isoformat() + "Z"
    if session_id:
        with conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions (session_id, created_at, agents, topic, status)
                VALUES (?, ?, ?, ?, 'ACTIVE')
                """,
                (session_id, now, json.dumps([agent_name or "harness"]), title),
            )
    datasets_str = json.dumps(dataset_dependencies) if dataset_dependencies else "[]"
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO hypotheses (
                session_id, agent_name, title, statement, null_hypothesis, alt_hypothesis,
                domain, dataset_dependencies, methodology, metric_name, metric_value,
                p_value, confidence_level, status, summary_findings, report_dir, pdf_path,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                agent_name,
                title,
                statement,
                null_hypothesis,
                alt_hypothesis,
                domain,
                datasets_str,
                methodology,
                metric_name,
                metric_value,
                p_value,
                confidence_level,
                status.upper(),
                summary_findings,
                report_dir,
                pdf_path,
                now,
                now,
            ),
        )
        hypo_id = cursor.lastrowid
    conn.close()
    return hypo_id


def update_hypothesis_status(
    hypothesis_id: int,
    status: str,
    metric_name: Optional[str] = None,
    metric_value: Optional[float] = None,
    p_value: Optional[float] = None,
    summary_findings: Optional[str] = None,
    report_dir: Optional[str] = None,
    pdf_path: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Updates the status and findings of an existing hypothesis."""
    init_db(db_path)
    conn = get_connection(db_path)
    now = datetime.utcnow().isoformat() + "Z"

    fields = ["status = ?", "updated_at = ?"]
    params: List[Any] = [status.upper(), now]

    if metric_name is not None:
        fields.append("metric_name = ?")
        params.append(metric_name)
    if metric_value is not None:
        fields.append("metric_value = ?")
        params.append(metric_value)
    if p_value is not None:
        fields.append("p_value = ?")
        params.append(p_value)
    if summary_findings is not None:
        fields.append("summary_findings = ?")
        params.append(summary_findings)
    if report_dir is not None:
        fields.append("report_dir = ?")
        params.append(report_dir)
    if pdf_path is not None:
        fields.append("pdf_path = ?")
        params.append(pdf_path)

    params.append(hypothesis_id)
    sql = f"UPDATE hypotheses SET {', '.join(fields)} WHERE id = ?"

    with conn:
        cursor = conn.execute(sql, params)
        updated = cursor.rowcount > 0
    conn.close()
    return updated


def get_hypothesis(hypothesis_id: int, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves a single hypothesis by ID."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.execute("SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_hypotheses(
    status: Optional[str] = None,
    domain: Optional[str] = None,
    agent: Optional[str] = None,
    limit: int = 100,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """Lists hypotheses filtered by optional status, domain, or agent."""
    init_db(db_path)
    conn = get_connection(db_path)
    sql = "SELECT * FROM hypotheses WHERE 1=1"
    params: List[Any] = []

    if status:
        sql += " AND UPPER(status) = ?"
        params.append(status.upper())
    if domain:
        sql += " AND domain = ?"
        params.append(domain)
    if agent:
        sql += " AND agent_name = ?"
        params.append(agent)

    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cursor = conn.execute(sql, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def search_hypotheses(
    query: str,
    top_k: int = 5,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Performs BM25 ranked full-text search across hypotheses using SQLite FTS5.
    Returns matched hypotheses with their relevance score.
    """
    init_db(db_path)
    conn = get_connection(db_path)

    # Tokenize and sanitize search terms for FTS5 syntax
    raw_tokens = re.findall(r"[A-Za-z0-9_]{3,}", query)
    if not raw_tokens:
        conn.close()
        return []

    fts_query = " OR ".join(f'"{t}"' for t in raw_tokens)

    sql = """
    SELECT h.*, bm25(hypotheses_fts) AS rank_score
    FROM hypotheses_fts
    JOIN hypotheses h ON hypotheses_fts.rowid = h.id
    WHERE hypotheses_fts MATCH ?
    ORDER BY rank_score ASC
    LIMIT ?;
    """
    try:
        cursor = conn.execute(sql, (fts_query, top_k))
        results = []
        for r in cursor.fetchall():
            d = dict(r)
            # Invert BM25 negative score to a positive relevance normalized score
            # SQLite bm25() returns lower (more negative) values for better matches
            bm25_val = d.pop("rank_score", 0.0)
            d["relevance_score"] = round(abs(bm25_val), 4)
            results.append(d)
    except Exception as e:
        print(f"FTS5 search error: {e}")
        results = []
    finally:
        conn.close()

    return results


def record_experiment(
    hypothesis_id: int,
    session_id: Optional[str],
    agent_name: str,
    code_path: str,
    duration_seconds: float,
    metrics: Dict[str, Any],
    status: str = "SUCCESS",
    log_output: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Logs an experimental execution run."""
    init_db(db_path)
    conn = get_connection(db_path)
    now = datetime.utcnow().isoformat() + "Z"
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO experiments (
                hypothesis_id, session_id, agent_name, code_path, start_time,
                end_time, duration_seconds, metrics_json, status, log_output
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hypothesis_id,
                session_id,
                agent_name,
                code_path,
                now,
                now,
                duration_seconds,
                json.dumps(metrics),
                status,
                log_output,
            ),
        )
        exp_id = cursor.lastrowid
    conn.close()
    return exp_id
