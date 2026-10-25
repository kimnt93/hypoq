import polars as pl
import pytest

from scripts.migrate_time_datatypes import _iso_to_datetime


def test_iso_z_timestamps_survive_conversion():
    df = pl.DataFrame({"t": ["2017-10-16T08:59:00Z", "2026-10-12T03:55:52.991Z", None]})
    out = df.with_columns(_iso_to_datetime(pl.col("t")).alias("t"))
    assert out["t"].null_count() == 1
    assert str(out["t"][0]) == "2017-10-16 08:59:00"


def test_fred_series_parser_keeps_missing_as_null(monkeypatch):
    import scripts.ingest_fred_us_macro as fred

    class R:
        content = b"observation_date,X\n2026-01-01,1.5\n2026-01-02,.\n2026-01-03,2.5\n"
        def raise_for_status(self): pass

    monkeypatch.setattr(fred.requests, "get", lambda *a, **k: R())
    df = fred.fred_series("X")
    assert df.height == 2 and df["value"].to_list() == [1.5, 2.5]
