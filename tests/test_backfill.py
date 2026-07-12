"""Tests for the backfill's pure partitioning and resume logic."""

from __future__ import annotations

from datetime import date

from elt_btc.ingestion.backfill_1m import is_partition_complete, month_bounds_ms, month_range
from elt_btc.storage.parquet_io import write_partition

MIN_MS = 60_000


def test_month_range_single_month():
    assert month_range(date(2020, 5, 10), date(2020, 5, 20)) == [(2020, 5)]


def test_month_range_across_years():
    assert month_range(date(2019, 11, 15), date(2020, 2, 1)) == [
        (2019, 11),
        (2019, 12),
        (2020, 1),
        (2020, 2),
    ]


def test_month_bounds_ms_utc():
    start_ms, end_ms = month_bounds_ms(2020, 1)
    assert start_ms == 1_577_836_800_000  # 2020-01-01T00:00:00Z
    assert end_ms == 1_580_515_200_000  # 2020-02-01T00:00:00Z


def test_month_bounds_ms_december_rollover():
    _, end_ms = month_bounds_ms(2019, 12)
    start_ms, _ = month_bounds_ms(2020, 1)
    assert end_ms == start_ms


def test_is_partition_complete_missing_file(tmp_path):
    assert not is_partition_complete(tmp_path, 2020, 1, "1m")


def test_is_partition_complete_partial_month(tmp_path, ohlcv_factory):
    start_ms, _ = month_bounds_ms(2020, 1)
    df = ohlcv_factory([start_ms + i * MIN_MS for i in range(100)])  # stops mid-month
    write_partition(df, tmp_path, 2020, 1)
    assert not is_partition_complete(tmp_path, 2020, 1, "1m")


def test_is_partition_complete_full_month(tmp_path, ohlcv_factory):
    start_ms, end_ms = month_bounds_ms(2020, 1)
    last_open = end_ms - MIN_MS  # final 1m candle of the month
    df = ohlcv_factory([start_ms, last_open])
    write_partition(df, tmp_path, 2020, 1)
    assert is_partition_complete(tmp_path, 2020, 1, "1m")
