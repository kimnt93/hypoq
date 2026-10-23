---
name: dedup-hypothesis-search
description: Search existing research hypotheses in the SQLite research database using FTS5 BM25 text ranking and keyword similarity to eliminate duplicate experiments.
version: 1.0.0
author: kim / vnresearch
tags: [search, deduplication, sqlite, fts5, hypothesis, bm25]
---

# Deduplication & Hypothesis Search Skill

## Overview
To prevent wasted compute and redundant empirical exploration, this skill provides fast querying over historical hypotheses stored in `autoresearch.db`. It utilizes SQLite FTS5 (Full-Text Search) with BM25 relevance ranking and token overlap scoring.

## When to Use
- Before launching a new experimental research run.
- When an agent is prompted with a research question to see if previous experiments accepted, rejected, or found inconclusive evidence on the same topic.
- To discover prior test statistics and refine an alternative angle or model specification.

## CLI Usage
Query the database directly from the command line:
```bash
# Search by keywords
python run.py --search="tightly held shares basis"

# Search by topic or factor
python run.py --search="foreign flow trading pressure"
```

## Python Integration
```python
import sqlite3

def query_dedup_search(query_str: str, db_path: str = "autoresearch.db", top_k: int = 5):
    """
    Executes BM25 full-text search against the hypotheses_fts table.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Clean query tokens for FTS5 syntax
    tokens = [t.replace('"', '') for t in query_str.split() if len(t) > 2]
    if not tokens:
        return []
    
    fts_query = " OR ".join(tokens)
    
    sql = """
    SELECT h.id, h.title, h.status, h.metric_name, h.metric_value, h.p_value,
           h.summary_findings, h.report_dir, h.pdf_path,
           bm25(hypotheses_fts) AS rank_score
    FROM hypotheses_fts
    JOIN hypotheses h ON hypotheses_fts.rowid = h.id
    WHERE hypotheses_fts MATCH ?
    ORDER BY rank_score ASC
    LIMIT ?;
    """
    try:
        cursor.execute(sql, (fts_query, top_k))
        results = [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        print(f"Search warning: {e}")
        results = []
    finally:
        conn.close()
        
    return results
```

## Deduplication Strategy Guidelines
- If a query returns a match with high relevance score and status **`ACCEPTED`**:
  Do NOT rerun the exact same specification. Instead, test an extension (e.g., higher frequency, sub-sample stability, interaction with macro regimes).
- If a query returns a match with status **`REJECTED`**:
  Examine the failure reason (e.g. transaction cost erosion, lack of statistical power). Do not repeat the failed pipeline unless introducing a fundamentally different conditioning mechanism or data source.
