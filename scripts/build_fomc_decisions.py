#!/usr/bin/env python3
"""
Point-in-time FOMC rate-decision table, derived only from real downloaded data.

Inputs : data/community/us_fed_communications.parquet  (official FOMC statement release timestamps, UTC)
         FRED DFEDTARU / DFEDTARL                         (federal funds target range, upper / lower bound)
Output : data/macro/us_fed_fomc_rate_decisions_pit.parquet

A decision row is available to the market at `announced_at` (UTC). The target range before the decision is the last value
published before the announcement date; the range after is the value on the following calendar day (the effective date).
`rate_change_bps` is the change in the UPPER bound, so a cut of 25 bp reads -25.
"""
import os
import sys

import polars as pl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from scripts.ingest_fred_us_macro import fred_series  # noqa: E402


def build_fomc_decisions() -> pl.DataFrame:
    comm_path = os.path.join(BASE_DIR, "data", "community", "us_fed_communications.parquet")
    if not os.path.exists(comm_path):
        print("    ! us_fed_communications.parquet missing; run scripts/ingest_community_and_social.py first.")
        return pl.DataFrame()
    stm = (
        pl.read_parquet(comm_path)
        .filter((pl.col("doc_type") == "press_release") & pl.col("title").str.to_lowercase().str.contains("fomc statement"))
        .select(["datetime_utc", "date", "title", "url"])
        .sort("datetime_utc")
    )
    upper = fred_series("DFEDTARU").rename({"value": "upper"})
    lower = fred_series("DFEDTARL").rename({"value": "lower"})
    tgt = upper.join(lower, on="date", how="inner").sort("date")
    before = tgt.rename({"upper": "upper_before", "lower": "lower_before"}).with_columns(pl.col("date") + pl.duration(days=1)).rename({"date": "key_before"})
    after = tgt.rename({"upper": "target_range_high", "lower": "target_range_low"}).rename({"date": "key_after"})
    out = (
        stm.with_columns((pl.col("date") - pl.duration(days=1)).alias("key_before"), (pl.col("date") + pl.duration(days=1)).alias("key_after"))
        .sort("key_before").join_asof(before.sort("key_before"), on="key_before", strategy="backward")
        .sort("key_after").join_asof(after.sort("key_after"), on="key_after", strategy="forward")
        .with_columns(((pl.col("target_range_high") - pl.col("upper_before")) * 100).alias("rate_change_bps"))
        .with_columns(
            pl.when(pl.col("rate_change_bps") > 0).then(pl.lit("HIKE")).when(pl.col("rate_change_bps") < 0).then(pl.lit("CUT")).otherwise(pl.lit("HOLD")).alias("action"),
            (pl.col("date") + pl.duration(days=1)).alias("effective_date"),
        )
        .select([
            pl.col("date").alias("meeting_date"), pl.col("datetime_utc").alias("announced_at"), "effective_date", "action", "rate_change_bps",
            pl.col("upper_before").alias("target_range_high_before"), "target_range_low", "target_range_high", "title", "url",
        ])
        .sort("announced_at")
    )
    path = os.path.join(BASE_DIR, "data", "macro", "us_fed_fomc_rate_decisions_pit.parquet")
    out.write_parquet(path, compression="zstd")
    print(f"    => us_fed_fomc_rate_decisions_pit.parquet ({len(out)} statements: "
          f"{out.filter(pl.col('action') == 'CUT').height} cuts, {out.filter(pl.col('action') == 'HIKE').height} hikes, {out.filter(pl.col('action') == 'HOLD').height} holds)")
    return out


if __name__ == "__main__":
    build_fomc_decisions()
