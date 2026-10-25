---
name: data-explore-pit-builder
description: Explore, ingest, clean, and validate Point-in-Time (PIT) event datasets (such as US Federal Reserve FOMC interest rate announcements, SBV rate changes, or corporate disclosures) with strict timestamp causality.
version: 1.0.0
author: hypoq
tags: [data, point-in-time, pit, fed-funds, fomc, announcements, macro, ingestion]
---

# Point-in-Time (PIT) Dataset Builder Skill

## Overview
This skill defines the methodology for discovering, constructing, validating, and registering new **Point-in-Time (PIT)** financial and macroeconomic datasets into `data/`.

## When to Use
- When asked to ingest a new macro indicator or event series (e.g. US Federal Reserve FOMC interest rate announcements, SBV refinancing rate cuts, GDP nowcasts, corporate dividend announcements).
- When validating that historical time-series entries reflect the exact information available to market participants at that moment, without ex-post revisions or look-ahead bias.

## Point-in-Time Causality Principles
1. **Two Mandatory Temporal Columns**:
   * `announced_at` or `release_timestamp`: Exact date-time when information became publicly observable to market participants (e.g., `2026-03-18 14:00:00-04:00`).
   * `effective_date`: The calendar date on which the policy rate or corporate action legally takes effect.
2. **Never Forward-Fill Before Release**: Data values must strictly remain `null` or hold their previous value prior to `announced_at`.
3. **Merging Rule**: Always use Polars backward as-of join:
   ```python
   aligned = market_daily.join_asof(
       announcements.sort("announced_at"),
       left_on="timestamp",
       right_on="announced_at",
       strategy="backward"
   )
   ```

## Construction Recipe: US Federal Reserve Interest Rate Decision Dataset

The table `data/macro/us_fed_fomc_rate_decisions_pit.parquet` is **derived, not typed**: `scripts/build_fomc_decisions.py` takes the official FOMC
statement release timestamps from `data/community/us_fed_communications.parquet` (Federal Reserve JSON feed, converted from US/Eastern to UTC) and
the federal funds target range from FRED (`DFEDTARU`, `DFEDTARL`). The range before the decision is the last value published before the announcement date;
the range after is the value on the next calendar day.

```python
import polars as pl

d = pl.read_parquet("data/macro/us_fed_fomc_rate_decisions_pit.parquet")
# columns: meeting_date, announced_at (UTC), effective_date, action (HIKE/CUT/HOLD), rate_change_bps,
#          target_range_high_before, target_range_low, target_range_high, title, url

# Point-in-time join: each trading day sees only decisions announced before its close
# (14:00 US/Eastern announcements are available the same session; UTC naive timestamps).
days = pl.read_parquet("data/market/us_indices_daily.parquet").filter(pl.col("symbol") == "^GSPC").sort("date")
pit = days.join_asof(
    d.sort("meeting_date").select(["meeting_date", "action", "rate_change_bps", "target_range_high"]),
    left_on="date", right_on="meeting_date", strategy="backward",
)
```

To build a new point-in-time event table, follow the same pattern: (1) download the official event feed with real timestamps, (2) join the
state variable from a free series, (3) compute changes from the data, (4) write with `compression="zstd"`. Never type event rows by hand.

## Quality Control & Validation Checklist
Before saving any newly explored dataset into `data/`:
1. Check for duplicates across `(entity, announced_at)`.
2. Verify chronological sorting by timestamp.
3. Test backward join alignment against a daily index (e.g. `us_indices_daily.parquet` or `vn_indices_daily.parquet`).
4. Ensure Parquet compression is set to `zstd`.
