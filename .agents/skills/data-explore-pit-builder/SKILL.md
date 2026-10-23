---
name: data-explore-pit-builder
description: Explore, ingest, clean, and validate Point-in-Time (PIT) event datasets (such as US Federal Reserve FOMC interest rate announcements, SBV rate changes, or corporate disclosures) with strict timestamp causality.
version: 1.0.0
author: kim / vnresearch
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

Below is the standard Python/Polars pipeline to build and serialize a Point-in-Time Federal Reserve announcement series into `data/macro/us_fed_fomc_rate_decisions_pit.parquet`:

```python
import polars as pl
from datetime import datetime

def build_fed_fomc_pit_dataset(output_path: str = "data/macro/us_fed_fomc_rate_decisions_pit.parquet"):
    """
    Constructs a Point-in-Time FOMC rate announcement table with official release timestamps.
    """
    records = [
        # Sample historical FOMC decisions with exact announcement times
        {
            "meeting_date": "2024-09-18",
            "announced_at": "2024-09-18 14:00:00",
            "effective_date": "2024-09-19",
            "action": "CUT",
            "rate_change_bps": -50.0,
            "target_range_low": 4.75,
            "target_range_high": 5.00,
            "dissent_votes": 1,
            "statement_summary": "Committee decided to lower the target range for the federal funds rate by 50 bps."
        },
        {
            "meeting_date": "2024-11-07",
            "announced_at": "2024-11-07 14:00:00",
            "effective_date": "2024-11-08",
            "action": "CUT",
            "rate_change_bps": -25.0,
            "target_range_low": 4.50,
            "target_range_high": 4.75,
            "dissent_votes": 0,
            "statement_summary": "Committee decided to lower the target range by 25 bps."
        },
        {
            "meeting_date": "2024-12-18",
            "announced_at": "2024-12-18 14:00:00",
            "effective_date": "2024-12-19",
            "action": "CUT",
            "rate_change_bps": -25.0,
            "target_range_low": 4.25,
            "target_range_high": 4.50,
            "dissent_votes": 0,
            "statement_summary": "Committee lowered the target range by 25 bps."
        }
    ]

    df = (
        pl.DataFrame(records)
        .with_columns([
            pl.col("meeting_date").str.to_date(),
            pl.col("announced_at").str.to_datetime(),
            pl.col("effective_date").str.to_date(),
            pl.col("rate_change_bps").cast(pl.Float64),
            pl.col("target_range_low").cast(pl.Float64),
            pl.col("target_range_high").cast(pl.Float64),
            pl.col("dissent_votes").cast(pl.Int32),
        ])
        .sort("announced_at")
    )

    df.write_parquet(output_path, compression="zstd")
    print(f"Successfully generated Point-in-Time dataset: {output_path} ({len(df)} rows)")
    return df
```

## Quality Control & Validation Checklist
Before saving any newly explored dataset into `data/`:
1. Check for duplicates across `(entity, announced_at)`.
2. Verify chronological sorting by timestamp.
3. Test backward join alignment against `vn_indices_daily.parquet`.
4. Ensure Parquet compression is set to `zstd`.
