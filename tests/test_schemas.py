"""Validation schema and gap detection tests."""

from __future__ import annotations

import pandas as pd
import pandera.errors
import pytest

from elt_btc.validation.schemas import find_gaps, gaps_to_ranges, validate_ohlcv

TF = "15m"
TF_MS = 900_000
T0 = 1_700_000_000_000


def grid(n: int, start: int = T0, step: int = TF_MS) -> list[int]:
    return [start + i * step for i in range(n)]


def test_valid_frame_passes(ohlcv_factory):
    df = ohlcv_factory(grid(5))
    assert len(validate_ohlcv(df)) == 5


def test_low_above_open_or_close_fails(ohlcv_factory):
    df = ohlcv_factory(grid(3))
    df.loc[1, "low"] = df.loc[1, "open"] + 10.0
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_ohlcv(df)


def test_high_below_open_or_close_fails(ohlcv_factory):
    df = ohlcv_factory(grid(3))
    df.loc[2, "high"] = df.loc[2, "close"] - 10.0
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_ohlcv(df)


def test_negative_volume_fails(ohlcv_factory):
    df = ohlcv_factory(grid(3))
    df.loc[0, "volume"] = -1.0
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_ohlcv(df)


def test_non_positive_price_fails(ohlcv_factory):
    df = ohlcv_factory(grid(3))
    df.loc[0, "open"] = 0.0
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_ohlcv(df)


def test_duplicate_timestamp_fails(ohlcv_factory):
    df = ohlcv_factory(grid(3))
    df = pd.concat([df, df.iloc[[1]]], ignore_index=True)
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_ohlcv(df)


def test_find_gaps_none(ohlcv_factory):
    df = ohlcv_factory(grid(10))
    assert find_gaps(df, TF) == []


def test_find_gaps_detects_missing_candles(ohlcv_factory):
    timestamps = grid(10)
    missing = [timestamps[3], timestamps[4], timestamps[7]]
    df = ohlcv_factory([ts for ts in timestamps if ts not in missing])
    assert find_gaps(df, TF) == missing


def test_find_gaps_short_frame(ohlcv_factory):
    assert find_gaps(ohlcv_factory([T0]), TF) == []
    assert find_gaps(ohlcv_factory([]), TF) == []


def test_gaps_to_ranges_compresses_consecutive():
    gaps = [T0, T0 + TF_MS, T0 + 5 * TF_MS]
    assert gaps_to_ranges(gaps, TF) == [(T0, T0 + TF_MS), (T0 + 5 * TF_MS, T0 + 5 * TF_MS)]


def test_gaps_to_ranges_empty():
    assert gaps_to_ranges([], TF) == []
