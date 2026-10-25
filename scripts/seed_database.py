#!/usr/bin/env python3
"""
Rebuild the hypothesis database from the research pipelines.

The database holds only results computed by researches/*/code/run_pipeline.py. This script re-runs every
pipeline (each one registers itself idempotently through hypoq.db.upsert_hypothesis), then refreshes the
semantic embeddings. It never inserts hand-written hypotheses or metrics.
"""

import glob
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from hypoq.db import DEFAULT_DB_PATH, init_db, sync_all_embeddings


def seed():
    init_db(DEFAULT_DB_PATH)
    pipelines = sorted(glob.glob(os.path.join(ROOT, "researches", "*", "code", "run_pipeline.py")))
    failed = []
    for path in pipelines:
        print(f">>> Running {os.path.relpath(path, ROOT)}")
        if subprocess.run([sys.executable, path], cwd=ROOT).returncode != 0:
            failed.append(path)
    n = sync_all_embeddings(DEFAULT_DB_PATH)
    print(f"\nRan {len(pipelines) - len(failed)}/{len(pipelines)} pipelines; embedded {n} hypotheses in {DEFAULT_DB_PATH}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    seed()
