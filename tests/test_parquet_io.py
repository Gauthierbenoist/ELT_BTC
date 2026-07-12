"""Partitioned Parquet storage tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from elt_btc.storage.parquet_io import (
    existing_partitions,
    last_timestamp,
    partition_path,
    read_partition,
    write_partition,
)

T0 = 1_700_000_000_000
MIN_MS = 60_000


def test_partition_path_layout():
    path = partition_path(Path("data"), 2017, 8)
    assert path == Path("data") / "year=2017" / "month=08" / "data.parquet"


def test_write_read_roundtrip(tmp_path, ohlcv_factory):
    df = ohlcv_factory([T0 + i * MIN_MS for i in range(5)])
    write_partition(df, tmp_path, 2020, 1)
    out = read_partition(tmp_path, 2020, 1)
    pd.testing.assert_frame_equal(out, df)


def test_write_leaves_no_temp_file(tmp_path, ohlcv_factory):
    df = ohlcv_factory([T0])
    path = write_partition(df, tmp_path, 2020, 1)
    assert [f.name for f in path.parent.iterdir()] == ["data.parquet"]


def test_write_overwrites_existing_partition(tmp_path, ohlcv_factory):
    write_partition(ohlcv_factory([T0]), tmp_path, 2020, 1)
    bigger = ohlcv_factory([T0 + i * MIN_MS for i in range(3)])
    write_partition(bigger, tmp_path, 2020, 1)
    assert len(read_partition(tmp_path, 2020, 1)) == 3


def test_existing_partitions(tmp_path, ohlcv_factory):
    assert existing_partitions(tmp_path / "missing") == set()
    write_partition(ohlcv_factory([T0]), tmp_path, 2017, 8)
    write_partition(ohlcv_factory([T0]), tmp_path, 2018, 12)
    assert existing_partitions(tmp_path) == {(2017, 8), (2018, 12)}


def test_last_timestamp(tmp_path, ohlcv_factory):
    timestamps = [T0, T0 + MIN_MS, T0 + 2 * MIN_MS]
    path = write_partition(ohlcv_factory(timestamps), tmp_path, 2020, 1)
    assert last_timestamp(path) == timestamps[-1]
