#!/usr/bin/env python3
"""
Embedding migration. Run after changing the embedding method (EMBEDDING_PROVIDER, EMBEDDING_MODEL, GEMINI_API_KEY).

Every vector is stored with the id of the method that produced it (e.g. "huggingface:sentence-transformers/all-MiniLM-L6-v2",
"gemini:text-embedding-004"). This script recomputes the vectors of all hypotheses whose stored id differs from the active method
(or all of them with --force) from the hypothesis text already in the database, so no study has to be re-run.

  python scripts/db_migrate_embeddings.py [--db PATH_OR_URL] [--force] [--check]      (make db-migrate)
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from hypoq.db import DEFAULT_DB_PATH, get_connection, init_db, parse_db_uri, reembed_all  # noqa: E402
from hypoq.embeddings import get_model_id  # noqa: E402


def stored_models(db):
    init_db(db)
    backend, _ = parse_db_uri(db)
    conn = get_connection(db)
    try:
        sql = ("SELECT COALESCE(e.model_name, 'none') AS m, COUNT(*) AS n FROM hypotheses h "
               "LEFT JOIN hypothesis_embeddings e ON h.id = e.hypothesis_id GROUP BY 1")
        if backend == "postgres":
            with conn.cursor() as cur:
                cur.execute(sql)
                return {r["m"]: r["n"] for r in cur.fetchall()}
        return {r[0]: r[1] for r in conn.execute(sql).fetchall()}
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=DEFAULT_DB_PATH)
    ap.add_argument("--force", action="store_true", help="recompute every vector even if the model id matches")
    ap.add_argument("--check", action="store_true", help="only report; exit 1 if a migration is needed")
    args = ap.parse_args()
    active = get_model_id()
    before = stored_models(args.db)
    print(f"Active embedding method : {active}")
    print(f"Stored vectors          : {before or '(no hypotheses)'}")
    stale = sum(n for m, n in before.items() if m != active)
    if args.check:
        print("migration needed" if stale else "up to date")
        return 1 if stale else 0
    if not stale and not args.force:
        print("Nothing to migrate.")
        return 0
    stats = reembed_all(args.db, force=args.force)
    print(f">>> re-embedded {stats['re_embedded']} of {stats['total']} hypotheses with {stats['model']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
