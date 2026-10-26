#!/usr/bin/env python3
"""
Load every raw study record (researches/*/results/hypothesis_record.json) into the database and create the embeddings.

The database file is not versioned: it is rebuilt from the committed records. Records hold only metadata and computed results;
the vectors are produced here with the active embedding method (EMBEDDING_PROVIDER / EMBEDDING_MODEL, see .env.example).
Idempotent: a record is matched by title, so re-running updates rows instead of duplicating them.

  python scripts/db_import.py [--db PATH_OR_URL] [--records-dir researches]      (make db-import)
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from hypoq.db import DEFAULT_DB_PATH, import_record, init_db, reembed_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite path or PostgreSQL URL (default: DATABASE_URL / HYPOQ_DB / hypoq.db)")
    ap.add_argument("--records-dir", default=os.path.join(ROOT, "researches"))
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.records_dir, "*", "results", "hypothesis_record.json")))
    if not paths:
        print(f"No records found under {args.records_dir}/*/results/hypothesis_record.json (run `python scripts/db_export_records.py` or a pipeline)")
        return 1
    init_db(args.db)
    failed = 0
    for path in paths:
        rel = os.path.relpath(path, ROOT)
        try:
            hid = import_record(json.load(open(path, encoding="utf-8")), args.db)
            print(f"  imported #{hid:<3} {rel}")
        except Exception as exc:
            failed += 1
            print(f"  FAILED       {rel}: {exc}")
    try:
        stats = reembed_all(args.db)          # records whose vector is missing or from another embedding method
        print(f">>> {len(paths) - failed}/{len(paths)} records in {args.db}; embedding model {stats['model']} "
              f"({stats['total']} hypotheses embedded)")
    except Exception as exc:
        print(f">>> records imported but embeddings are missing: {exc}\n    fix the embedding setup, then run `make db-migrate`")
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
