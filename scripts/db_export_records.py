#!/usr/bin/env python3
"""
Write results/hypothesis_record.json for every hypothesis in the database that has a research folder.
`hypoq.pipeline.finalize` already writes the record when a pipeline runs; use this to back-fill studies that were registered
before records existed, or to refresh records from a database you edited.

  python scripts/db_export_records.py [--db PATH_OR_URL]      (make db-export)
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from hypoq.db import DEFAULT_DB_PATH, build_record, list_hypotheses  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=DEFAULT_DB_PATH)
    args = ap.parse_args()
    n = 0
    for h in list_hypotheses(limit=100000, db_path=args.db):
        rd = h.get("report_dir") or ""
        results_path = os.path.join(ROOT, rd, "results", "empirical_results.json")
        if not rd.startswith("researches/") or not os.path.isfile(results_path):
            continue
        record = build_record(h, json.load(open(results_path)), f"{rd}/code/run_pipeline.py")
        with open(os.path.join(ROOT, rd, "results", "hypothesis_record.json"), "w") as f:
            json.dump(record, f, indent=2, default=str)
            f.write("\n")
        n += 1
        print(f"  wrote {rd}/results/hypothesis_record.json")
    print(f">>> {n} records written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
