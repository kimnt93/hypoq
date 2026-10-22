#!/usr/bin/env python3
"""
Migrate and standardize all Date and Time columns to native Parquet Date & Datetime types.
Transforms:
1. Date strings ("YYYY-MM-DD") -> pl.Date
2. Integer epoch timestamps (Int64 seconds) -> pl.Datetime("ms")
3. ISO datetime strings ("YYYY-MM-DDTHH:MM:SSZ") -> pl.Datetime("ms")
Strictly eliminates raw String dates and raw Int64 timestamps across all 36 Parquet datasets.
"""

import os
import glob
import polars as pl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "vnresearch/data")

DATE_COLUMNS = [
    "date", "fiscal_date", "report_date", "effective_date", "listed_date",
    "disclosure_date", "trading_start_date", "actual_date", "previous_snapshot_date"
]

DATETIME_COLUMNS = [
    "time", "announced_at", "retrieved_at", "created_date", "modified_date"
]

def migrate_file(fpath):
    rel_path = os.path.relpath(fpath, DATA_DIR)
    df = pl.read_parquet(fpath)
    
    exprs = []
    converted_cols = []

    for col, dtype in df.schema.items():
        type_str = str(dtype)
        c_lower = col.lower()

        # 1. Date columns -> pl.Date
        if c_lower in DATE_COLUMNS and type_str != "Date":
            exprs.append(
                pl.col(col).cast(pl.Utf8).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False).alias(col)
            )
            converted_cols.append(f"{col} ({type_str} -> Date)")

        # 2. Integer epoch timestamp -> pl.Datetime('ms')
        elif c_lower == "timestamp" and ("Int" in type_str or type_str == "String"):
            if "Int" in type_str:
                exprs.append(
                    pl.from_epoch(pl.col(col), time_unit="s").cast(pl.Datetime("ms")).alias(col)
                )
            else:
                exprs.append(
                    pl.col(col).str.to_datetime(strict=False).cast(pl.Datetime("ms")).alias(col)
                )
            converted_cols.append(f"{col} ({type_str} -> Datetime[ms])")

        # 3. ISO / Datetime strings -> pl.Datetime('ms')
        elif c_lower in DATETIME_COLUMNS and "Datetime" not in type_str:
            exprs.append(
                pl.col(col).cast(pl.Utf8).str.to_datetime(strict=False).cast(pl.Datetime("ms")).alias(col)
            )
            converted_cols.append(f"{col} ({type_str} -> Datetime[ms])")

    if exprs:
        df_new = df.with_columns(exprs)
        df_new.write_parquet(fpath)
        print(f"  [CONVERTED] {rel_path:<50s} | {', '.join(converted_cols)}")
        return 1
    else:
        print(f"  [OK - CLEAN] {rel_path:<50s} | All date/time types already native")
        return 0

def run_migration():
    print("=" * 85)
    print("STARTING DATE & TIME NATIVE PARQUET DATATYPES MIGRATION")
    print("=" * 85)

    parquet_files = sorted(glob.glob(os.path.join(DATA_DIR, "**/*.parquet"), recursive=True))
    print(f"Scanning {len(parquet_files)} datasets in {DATA_DIR}...\n")

    converted_count = 0
    for f in parquet_files:
        converted_count += migrate_file(f)

    print("=" * 85)
    print(f"MIGRATION COMPLETE: Converted time types in {converted_count}/{len(parquet_files)} datasets.")
    print("=" * 85)

if __name__ == "__main__":
    run_migration()
