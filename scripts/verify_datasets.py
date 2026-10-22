#!/usr/bin/env python3
"""
Parquet Datasets Integrity and Schema Validator.
Validates all datasets under data/, reporting row counts, columns, and file sizes.
"""

import os
import glob
import polars as pl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def verify():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "**/*.parquet"), recursive=True))
    print(f"Found {len(files)} Parquet files across {DATA_DIR}:")
    print("-" * 90)
    print(f"{'Dataset Parquet Path':<56} | {'Rows':>10} | {'Cols':>4} | {'Size':>9}")
    print("-" * 90)
    total_rows = 0
    total_bytes = 0

    for f in files:
        rel_path = os.path.relpath(f, BASE_DIR)
        size_bytes = os.path.getsize(f)
        total_bytes += size_bytes
        size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes < 1024 * 1024 else f"{size_bytes / (1024 * 1024):.2f} MB"
        try:
            df = pl.read_parquet(f)
            n_rows = len(df)
            n_cols = len(df.columns)
            total_rows += n_rows
            print(f"{rel_path:<56} | {n_rows:>10,} | {n_cols:>4} | {size_str:>9}")
        except Exception as e:
            print(f"{rel_path:<56} | [ERROR: {e}]")

    print("-" * 90)
    total_mb = total_bytes / (1024 * 1024)
    print(f"TOTAL: {len(files)} Parquet files | {total_rows:,} total rows | {total_mb:.2f} MB on disk")
    print("-" * 90)

if __name__ == "__main__":
    verify()
