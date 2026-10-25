import datetime as dt

import polars as pl
import pytest

from scripts.verify_datasets import audit_file


def _write(tmp_path, df, name="us_test_daily.parquet"):
    p = tmp_path / name
    df.write_parquet(p)
    return str(p)


def _frame(**over):
    base = {
        "date": [dt.date(2026, 10, 20), dt.date(2026, 10, 21)], "symbol": ["A", "A"],
        "open": [10.0, 11.0], "high": [12.0, 12.0], "low": [9.0, 10.0], "close": [11.0, 11.5], "returns": [None, 0.04],
    }
    base.update(over)
    return pl.DataFrame(base)


def test_clean_frame_has_no_errors(tmp_path):
    _, errors, _, _, _ = audit_file(_write(tmp_path, _frame()))
    assert errors == []


def test_duplicates_are_errors(tmp_path):
    df = _frame(date=[dt.date(2026, 10, 20)] * 2)
    _, errors, _, _, _ = audit_file(_write(tmp_path, df))
    assert any("duplicate" in e for e in errors)


def test_high_below_low_and_nonpositive_prices_are_errors(tmp_path):
    df = _frame(high=[8.0, 12.0], close=[11.0, -1.0])
    _, errors, _, _, _ = audit_file(_write(tmp_path, df))
    assert any("high < low" in e for e in errors) and any("non-positive" in e for e in errors)


def test_missing_prefix_and_all_null_column_are_errors(tmp_path):
    df = _frame(returns=[None, None])
    _, errors, _, _, _ = audit_file(_write(tmp_path, df, name="bad_name.parquet"))
    assert any("prefix" in e for e in errors) and any("entirely null" in e for e in errors)


def test_repository_data_lake_passes_the_audit():
    from scripts.verify_datasets import verify

    assert verify() == 0
