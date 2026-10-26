"""
Database & Semantic Search Engine for hypoq.
Supports dual backends:
1. SQLite (default, single file `hypoq.db` or via `sqlite:///path/to/db`)
2. PostgreSQL (via connection string `postgresql://user:pass@host:port/dbname`)

Features:
- Full ACID relational integrity across sessions, hypotheses, experiments, and embeddings.
- Dense semantic vector search powered by Google Gemini (free tier) or open-source Hugging Face models (< 8GB RAM).
- Automatic vector embedding generation and synchronization upon hypothesis registration.
- Backward-compatible FTS5 index on SQLite for hybrid fallback.
"""

import os
import sys
import re
import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

from hypoq.embeddings import (
    get_embedding,
    get_active_provider,
    get_model_id,
    vector_to_blob,
    blob_to_vector,
    cosine_similarity,
)

DEFAULT_DB_URI = os.environ.get("DATABASE_URL") or os.environ.get("HYPOQ_DB") or "hypoq.db"
DEFAULT_DB_PATH = DEFAULT_DB_URI

# -----------------------------------------------------------------------------
# SQL Schemas
# -----------------------------------------------------------------------------

SQLITE_SCHEMA = """
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
    FOREIGN KEY(hypothesis_id) REFERENCES hypotheses(id) ON DELETE CASCADE,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS hypothesis_embeddings (
    hypothesis_id INTEGER PRIMARY KEY,
    model_name TEXT NOT NULL,
    vector BLOB NOT NULL,
    dimensions INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(hypothesis_id) REFERENCES hypotheses(id) ON DELETE CASCADE
);

-- Content-backed FTS5 virtual table for keyword search
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

-- Triggers for FTS5 sync
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

POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    agents TEXT NOT NULL,
    topic TEXT,
    status TEXT DEFAULT 'ACTIVE',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id SERIAL PRIMARY KEY,
    session_id TEXT REFERENCES sessions(session_id),
    agent_name TEXT,
    title TEXT NOT NULL,
    statement TEXT NOT NULL,
    null_hypothesis TEXT NOT NULL,
    alt_hypothesis TEXT NOT NULL,
    domain TEXT NOT NULL,
    dataset_dependencies TEXT,
    methodology TEXT,
    metric_name TEXT,
    metric_value DOUBLE PRECISION,
    p_value DOUBLE PRECISION,
    confidence_level DOUBLE PRECISION DEFAULT 0.95,
    status TEXT NOT NULL DEFAULT 'PENDING',
    summary_findings TEXT,
    report_dir TEXT,
    pdf_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    hypothesis_id INTEGER REFERENCES hypotheses(id) ON DELETE CASCADE,
    session_id TEXT REFERENCES sessions(session_id),
    agent_name TEXT,
    code_path TEXT,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    duration_seconds DOUBLE PRECISION,
    metrics_json TEXT,
    status TEXT DEFAULT 'SUCCESS',
    log_output TEXT
);

CREATE TABLE IF NOT EXISTS hypothesis_embeddings (
    hypothesis_id INTEGER PRIMARY KEY REFERENCES hypotheses(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    vector BYTEA NOT NULL,
    dimensions INTEGER NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# -----------------------------------------------------------------------------
# Database Connection Management
# -----------------------------------------------------------------------------

def parse_db_uri(db_uri: Optional[str] = None) -> Tuple[str, str]:
    """
    Parses connection string and returns (backend_type, cleaned_conn_string).
    backend_type is either 'postgres' or 'sqlite'.
    """
    raw_uri = db_uri or os.environ.get("DATABASE_URL") or os.environ.get("HYPOQ_DB") or DEFAULT_DB_URI
    raw_uri = raw_uri.strip()

    if raw_uri.startswith("postgresql://") or raw_uri.startswith("postgres://"):
        return "postgres", raw_uri
    elif raw_uri.startswith("sqlite:///"):
        return "sqlite", raw_uri[10:]
    elif raw_uri.startswith("sqlite://"):
        return "sqlite", raw_uri[9:]
    return "sqlite", raw_uri


def get_connection(db_uri: Optional[str] = None):
    """Returns database connection based on configured URI."""
    backend, conn_str = parse_db_uri(db_uri)
    if backend == "postgres":
        import psycopg
        from psycopg.rows import dict_row
        conn = psycopg.connect(conn_str, row_factory=dict_row)
        return conn
    else:
        conn = sqlite3.connect(conn_str)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn


def init_db(db_uri: Optional[str] = None) -> None:
    """Initializes tables, relations, and embeddings storage."""
    backend, conn_str = parse_db_uri(db_uri)
    conn = get_connection(db_uri)
    try:
        if backend == "postgres":
            with conn.cursor() as cur:
                cur.execute(POSTGRES_SCHEMA)
            conn.commit()
        else:
            with conn:
                conn.executescript(SQLITE_SCHEMA)
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Embeddings Synchronization & Semantic Representation
# -----------------------------------------------------------------------------

def build_hypothesis_text_for_embedding(hypo: Dict[str, Any]) -> str:
    """Constructs structured representation in English for semantic vector embedding."""
    title = hypo.get("title", "")
    statement = hypo.get("statement", "")
    null_h = hypo.get("null_hypothesis", "")
    alt_h = hypo.get("alt_hypothesis", "")
    domain = hypo.get("domain", "")
    findings = hypo.get("summary_findings") or ""
    return (
        f"Domain: {domain}. Title: {title}. "
        f"Statement: {statement}. "
        f"Null Hypothesis (H0): {null_h}. "
        f"Alternative Hypothesis (H1): {alt_h}. "
        f"Empirical Findings: {findings}"
    ).strip()


def _store_embedding(hypothesis_id: int, text: str, db_uri: Optional[str] = None) -> None:
    """Embeds `text` with the active method and upserts the vector together with the model id that produced it."""
    vec = get_embedding(text)
    blob, model_id = vector_to_blob(vec), get_model_id()
    backend, _ = parse_db_uri(db_uri)
    conn = get_connection(db_uri)
    try:
        if backend == "postgres":
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO hypothesis_embeddings (hypothesis_id, model_name, vector, dimensions, updated_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (hypothesis_id) DO UPDATE SET model_name = EXCLUDED.model_name, vector = EXCLUDED.vector,
                        dimensions = EXCLUDED.dimensions, updated_at = NOW()
                """, (hypothesis_id, model_id, blob, len(vec)))
            conn.commit()
        else:
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO hypothesis_embeddings (hypothesis_id, model_name, vector, dimensions, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (hypothesis_id, model_id, blob, len(vec), datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"))
    finally:
        conn.close()


def reembed_all(db_uri: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    """
    Embedding migration. Recomputes the vector of every hypothesis that has no vector or whose stored model id differs from the
    active embedding method (`hypoq.embeddings.get_model_id()`); `force=True` recomputes all. Run it after changing
    EMBEDDING_PROVIDER / EMBEDDING_MODEL / GEMINI_API_KEY (`make db-migrate`).
    Returns {"model", "total", "re_embedded", "previous_models"}. Raises if the active method cannot produce vectors.
    """
    init_db(db_uri)
    backend, _ = parse_db_uri(db_uri)
    model_id = get_model_id()
    sql = ("SELECT h.*, e.model_name AS emb_model FROM hypotheses h "
           "LEFT JOIN hypothesis_embeddings e ON h.id = e.hypothesis_id ORDER BY h.id")
    conn = get_connection(db_uri)
    try:
        if backend == "postgres":
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = [dict(r) for r in cur.fetchall()]
        else:
            rows = [dict(r) for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()
    previous: Dict[str, int] = {}
    todo = []
    for r in rows:
        previous[r["emb_model"] or "none"] = previous.get(r["emb_model"] or "none", 0) + 1
        if force or r["emb_model"] != model_id:
            todo.append(r)
    for r in todo:
        _store_embedding(r["id"], build_hypothesis_text_for_embedding(r), db_uri)
    return {"model": model_id, "total": len(rows), "re_embedded": len(todo), "previous_models": previous}


def sync_all_embeddings(db_uri: Optional[str] = None) -> int:
    """Backfills missing vectors and re-embeds stale ones (see `reembed_all`). Returns the number of vectors written."""
    return reembed_all(db_uri)["re_embedded"]


def refresh_embedding(hypothesis_id: int, db_uri: Optional[str] = None) -> None:
    """Recomputes the vector of one hypothesis after its text changed."""
    hypo = get_hypothesis(hypothesis_id, db_uri)
    if hypo:
        _store_embedding(hypothesis_id, build_hypothesis_text_for_embedding(hypo), db_uri)


# -----------------------------------------------------------------------------
# CRUD Operations
# -----------------------------------------------------------------------------

def create_session(
    session_id: str,
    agents: List[str],
    topic: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: Optional[str] = None,
) -> str:
    """Registers a new research session."""
    init_db(db_path)
    backend, _ = parse_db_uri(db_path)
    conn = get_connection(db_path)
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"
    agents_str = json.dumps(agents) if isinstance(agents, list) else str(agents)

    try:
        if backend == "postgres":
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO sessions (session_id, created_at, agents, topic, status, notes)
                    VALUES (%s, NOW(), %s, %s, 'ACTIVE', %s)
                    ON CONFLICT (session_id) DO NOTHING
                """, (session_id, agents_str, topic, notes))
            conn.commit()
        else:
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO sessions (session_id, created_at, agents, topic, status, notes)
                    VALUES (?, ?, ?, ?, 'ACTIVE', ?)
                """, (session_id, now, agents_str, topic, notes))
    finally:
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
    db_path: Optional[str] = None,
) -> int:
    """Inserts hypothesis and automatically generates semantic vector embedding."""
    init_db(db_path)
    backend, _ = parse_db_uri(db_path)
    conn = get_connection(db_path)
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"
    datasets_str = json.dumps(dataset_dependencies) if dataset_dependencies else "[]"

    try:
        if session_id:
            if backend == "postgres":
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO sessions (session_id, created_at, agents, topic, status)
                        VALUES (%s, NOW(), %s, %s, 'ACTIVE')
                        ON CONFLICT (session_id) DO NOTHING
                    """, (session_id, json.dumps([agent_name or "harness"]), title))
                conn.commit()
            else:
                with conn:
                    conn.execute("""
                        INSERT OR IGNORE INTO sessions (session_id, created_at, agents, topic, status)
                        VALUES (?, ?, ?, ?, 'ACTIVE')
                    """, (session_id, now, json.dumps([agent_name or "harness"]), title))

        if backend == "postgres":
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO hypotheses (
                        session_id, agent_name, title, statement, null_hypothesis, alt_hypothesis,
                        domain, dataset_dependencies, methodology, metric_name, metric_value,
                        p_value, confidence_level, status, summary_findings, report_dir, pdf_path,
                        created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                    RETURNING id
                """, (
                    session_id, agent_name, title, statement, null_hypothesis, alt_hypothesis,
                    domain, datasets_str, methodology, metric_name, metric_value,
                    p_value, confidence_level, status.upper(), summary_findings, report_dir, pdf_path
                ))
                hypo_id = cur.fetchone()["id"]
            conn.commit()
        else:
            with conn:
                cursor = conn.execute("""
                    INSERT INTO hypotheses (
                        session_id, agent_name, title, statement, null_hypothesis, alt_hypothesis,
                        domain, dataset_dependencies, methodology, metric_name, metric_value,
                        p_value, confidence_level, status, summary_findings, report_dir, pdf_path,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, agent_name, title, statement, null_hypothesis, alt_hypothesis,
                    domain, datasets_str, methodology, metric_name, metric_value,
                    p_value, confidence_level, status.upper(), summary_findings, report_dir, pdf_path,
                    now, now
                ))
                hypo_id = cursor.lastrowid
    finally:
        conn.close()

    # Automatically generate the vector embedding for semantic search
    try:
        refresh_embedding(hypo_id, db_path)
    except Exception as err:
        print(f"[Warning] Failed to generate semantic embedding for hypothesis {hypo_id}: {err} (run `make db-migrate` later)")

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
    db_path: Optional[str] = None,
) -> bool:
    """Updates status and findings, refreshing semantic vector if findings change."""
    init_db(db_path)
    backend, _ = parse_db_uri(db_path)
    conn = get_connection(db_path)
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"

    fields = ["status = ?", "updated_at = ?"] if backend == "sqlite" else ["status = %s", "updated_at = NOW()"]
    params: List[Any] = [status.upper()]
    if backend == "sqlite":
        params.append(now)

    def add_param(col_name: str, val: Any):
        fields.append(f"{col_name} = ?" if backend == "sqlite" else f"{col_name} = %s")
        params.append(val)

    if metric_name is not None: add_param("metric_name", metric_name)
    if metric_value is not None: add_param("metric_value", metric_value)
    if p_value is not None: add_param("p_value", p_value)
    if summary_findings is not None: add_param("summary_findings", summary_findings)
    if report_dir is not None: add_param("report_dir", report_dir)
    if pdf_path is not None: add_param("pdf_path", pdf_path)

    params.append(hypothesis_id)
    where_clause = "WHERE id = ?" if backend == "sqlite" else "WHERE id = %s"
    sql = f"UPDATE hypotheses SET {', '.join(fields)} {where_clause}"

    updated = False
    try:
        if backend == "postgres":
            with conn.cursor() as cur:
                cur.execute(sql, params)
                updated = cur.rowcount > 0
            conn.commit()
        else:
            with conn:
                cursor = conn.execute(sql, params)
                updated = cursor.rowcount > 0
    finally:
        conn.close()

    # If findings were updated, refresh the semantic vector embedding
    if updated and summary_findings:
        try:
            refresh_embedding(hypothesis_id, db_path)
        except Exception as err:
            print(f"[Warning] Failed to refresh embedding for hypothesis {hypothesis_id}: {err} (run `make db-migrate` later)")

    return updated


def get_hypothesis(hypothesis_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single hypothesis by ID."""
    init_db(db_path)
    backend, _ = parse_db_uri(db_path)
    conn = get_connection(db_path)
    try:
        if backend == "postgres":
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM hypotheses WHERE id = %s", (hypothesis_id,))
                row = cur.fetchone()
                return dict(row) if row else None
        else:
            cursor = conn.execute("SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def list_hypotheses(
    status: Optional[str] = None,
    domain: Optional[str] = None,
    agent: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Lists hypotheses filtered by optional criteria."""
    init_db(db_path)
    backend, _ = parse_db_uri(db_path)
    conn = get_connection(db_path)

    sql = "SELECT * FROM hypotheses WHERE 1=1"
    params: List[Any] = []

    def append_cond(col: str, val: Any):
        nonlocal sql
        sql += f" AND UPPER({col}) = ?" if backend == "sqlite" else f" AND UPPER({col}) = %s"
        params.append(val.upper())

    if status: append_cond("status", status)
    if domain: append_cond("domain", domain)
    if agent: append_cond("agent_name", agent)

    sql += " ORDER BY id DESC LIMIT ?" if backend == "sqlite" else " ORDER BY id DESC LIMIT %s"
    params.append(limit)

    try:
        if backend == "postgres":
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]
        else:
            cursor = conn.execute(sql, params)
            return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Semantic Vector Search Engine
# -----------------------------------------------------------------------------

def search_hypotheses(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Performs dense semantic vector search across hypotheses.
    Returns matched hypotheses ranked by cosine similarity score (0.0 to 1.0).
    """
    init_db(db_path)
    backend, _ = parse_db_uri(db_path)

    # 1. Make sure every vector was produced by the active embedding method (migrates on demand)
    try:
        stats = reembed_all(db_path)
        if stats["re_embedded"]:
            print(f"[hypoq] re-embedded {stats['re_embedded']} hypotheses with {stats['model']} (previous: {stats['previous_models']})")
    except Exception as e:
        print(f"[Semantic Search] Embedding method unavailable: {e}")
        return []

    # 2. Generate embedding for user query
    clean_query = query.strip()
    if not clean_query:
        return []

    try:
        query_vec = get_embedding(clean_query)
    except Exception as e:
        print(f"[Semantic Search] Query embedding failed: {e}")
        return []

    # 3. Retrieve all hypothesis vectors and metadata
    conn = get_connection(db_path)
    try:
        query_sql = """
            SELECT h.*, e.vector, e.dimensions, e.model_name
            FROM hypotheses h
            JOIN hypothesis_embeddings e ON h.id = e.hypothesis_id
        """
        if backend == "postgres":
            with conn.cursor() as cur:
                cur.execute(query_sql)
                rows = cur.fetchall()
        else:
            cur = conn.execute(query_sql)
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    if not rows:
        return []

    # 4. Compute cosine similarities
    doc_vectors = []
    valid_rows = []
    expected_dim = len(query_vec)

    for r in rows:
        raw_blob = r.pop("vector")
        vec = blob_to_vector(raw_blob)
        if len(vec) == expected_dim:
            doc_vectors.append(vec)
            valid_rows.append(r)

    if not doc_vectors:
        return []

    matrix = np.vstack(doc_vectors)
    similarities = cosine_similarity(query_vec, matrix)

    # 5. Sort by similarity score descending
    scored_results = []
    for idx, sim in enumerate(similarities):
        score = float(sim)
        if score >= min_score:
            item = dict(valid_rows[idx])
            item["similarity_score"] = round(score, 4)
            scored_results.append(item)

    scored_results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return scored_results[:top_k]


def record_experiment(
    hypothesis_id: int,
    session_id: Optional[str],
    agent_name: str,
    code_path: str,
    duration_seconds: float,
    metrics: Dict[str, Any],
    status: str = "SUCCESS",
    log_output: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """Logs an experimental execution run."""
    init_db(db_path)
    backend, _ = parse_db_uri(db_path)
    conn = get_connection(db_path)
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"

    try:
        if backend == "postgres":
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO experiments (
                        hypothesis_id, session_id, agent_name, code_path, start_time,
                        end_time, duration_seconds, metrics_json, status, log_output
                    ) VALUES (%s, %s, %s, %s, NOW(), NOW(), %s, %s, %s, %s)
                    RETURNING id
                """, (
                    hypothesis_id, session_id, agent_name, code_path,
                    duration_seconds, json.dumps(metrics), status, log_output
                ))
                exp_id = cur.fetchone()["id"]
            conn.commit()
        else:
            with conn:
                cursor = conn.execute("""
                    INSERT INTO experiments (
                        hypothesis_id, session_id, agent_name, code_path, start_time,
                        end_time, duration_seconds, metrics_json, status, log_output
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    hypothesis_id, session_id, agent_name, code_path,
                    now, now, duration_seconds, json.dumps(metrics), status, log_output
                ))
                exp_id = cursor.lastrowid
    finally:
        conn.close()
    return exp_id


def find_hypothesis_id_by_title(title: str, db_path: Optional[str] = None) -> Optional[int]:
    """Returns the id of the hypothesis with this exact title, if any."""
    init_db(db_path)
    backend, _ = parse_db_uri(db_path)
    conn = get_connection(db_path)
    try:
        if backend == "postgres":
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM hypotheses WHERE title = %s ORDER BY id DESC LIMIT 1", (title,))
                row = cur.fetchone()
            return row["id"] if row else None
        row = conn.execute(
            "SELECT id FROM hypotheses WHERE title = ? ORDER BY id DESC LIMIT 1", (title,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


_UPDATABLE = ("agent_name", "statement", "null_hypothesis", "alt_hypothesis", "domain", "dataset_dependencies", "methodology", "metric_name",
              "metric_value", "p_value", "confidence_level", "status", "summary_findings", "report_dir", "pdf_path")


def update_hypothesis_fields(hypothesis_id: int, db_path: Optional[str] = None, **fields: Any) -> bool:
    """Updates any of the columns in `_UPDATABLE` (None values are skipped); the embedding is refreshed afterwards."""
    cols = {k: v for k, v in fields.items() if k in _UPDATABLE and v is not None}
    if "dataset_dependencies" in cols and not isinstance(cols["dataset_dependencies"], str):
        cols["dataset_dependencies"] = json.dumps(cols["dataset_dependencies"])
    if "status" in cols:
        cols["status"] = cols["status"].upper()
    if not cols:
        return False
    init_db(db_path)
    backend, _ = parse_db_uri(db_path)
    ph = "?" if backend == "sqlite" else "%s"
    sets = [f"{k} = {ph}" for k in cols]
    params: List[Any] = list(cols.values())
    if backend == "sqlite":
        sets.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z")
    else:
        sets.append("updated_at = NOW()")
    params.append(hypothesis_id)
    conn = get_connection(db_path)
    try:
        sql = f"UPDATE hypotheses SET {', '.join(sets)} WHERE id = {ph}"
        if backend == "postgres":
            with conn.cursor() as cur:
                cur.execute(sql, params)
                ok = cur.rowcount > 0
            conn.commit()
        else:
            with conn:
                ok = conn.execute(sql, params).rowcount > 0
    finally:
        conn.close()
    if ok:
        try:
            refresh_embedding(hypothesis_id, db_path)
        except Exception as err:
            print(f"[Warning] Failed to refresh embedding for hypothesis {hypothesis_id}: {err} (run `make db-migrate` later)")
    return ok


def upsert_hypothesis(title: str, db_path: Optional[str] = None, **fields: Any) -> int:
    """Idempotent registration: re-running a pipeline (or re-importing a record) updates its row instead of duplicating it."""
    existing = find_hypothesis_id_by_title(title, db_path)
    if existing is None:
        return record_hypothesis(title=title, db_path=db_path, **fields)
    update_hypothesis_fields(existing, db_path, **fields)
    return existing


# -----------------------------------------------------------------------------
# Raw records: the committed, embedding-free form of a hypothesis (researches/<id>/results/hypothesis_record.json)
# -----------------------------------------------------------------------------

RECORD_SCHEMA = "hypoq.record/1"
RECORD_FIELDS = ("title", "statement", "null_hypothesis", "alt_hypothesis", "domain", "dataset_dependencies", "methodology", "metric_name",
                 "metric_value", "p_value", "confidence_level", "status", "summary_findings", "report_dir", "pdf_path", "agent_name", "session_id")


def build_record(hypothesis: Dict[str, Any], results: Dict[str, Any], code_path: str) -> Dict[str, Any]:
    """
    Raw record of one study: hypothesis metadata + the computed results. No vectors, ids or timestamps, so it is stable under
    version control and independent of the embedding method; vectors are created when the record is imported.
    """
    h = {k: hypothesis.get(k) for k in RECORD_FIELDS}
    if isinstance(h["dataset_dependencies"], str):
        h["dataset_dependencies"] = json.loads(h["dataset_dependencies"] or "[]")
    return {"schema": RECORD_SCHEMA, "hypothesis": h, "experiment": {"code_path": code_path, "status": "SUCCESS", "metrics": results}}


def replace_experiments(hypothesis_id: int, session_id: Optional[str], agent_name: str, code_path: str, metrics: Dict[str, Any],
                        status: str = "SUCCESS", duration_seconds: float = 0.0, db_path: Optional[str] = None) -> int:
    """Makes `metrics` the single experiment of a hypothesis (idempotent import; `record_experiment` appends run history instead)."""
    backend, _ = parse_db_uri(db_path)
    conn = get_connection(db_path)
    try:
        if backend == "postgres":
            with conn.cursor() as cur:
                cur.execute("DELETE FROM experiments WHERE hypothesis_id = %s", (hypothesis_id,))
            conn.commit()
        else:
            with conn:
                conn.execute("DELETE FROM experiments WHERE hypothesis_id = ?", (hypothesis_id,))
    finally:
        conn.close()
    return record_experiment(hypothesis_id, session_id, agent_name, code_path, duration_seconds, metrics, status=status, db_path=db_path)


def import_record(record: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """Loads one raw record into the database (insert or update by title), creating its embedding. Returns the hypothesis id."""
    if record.get("schema") != RECORD_SCHEMA:
        raise ValueError(f"unsupported record schema {record.get('schema')!r}; expected {RECORD_SCHEMA!r}")
    h = {k: v for k, v in record["hypothesis"].items() if k in RECORD_FIELDS}
    title = h.pop("title")
    hid = upsert_hypothesis(title, db_path=db_path, **h)
    exp = record.get("experiment")
    if exp:
        replace_experiments(hid, h.get("session_id"), h.get("agent_name") or "hypoq-pipeline", exp.get("code_path", ""), exp.get("metrics", {}),
                            status=exp.get("status", "SUCCESS"), db_path=db_path)
    return hid
