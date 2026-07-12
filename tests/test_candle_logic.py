"""Tests for pure candle transformations (dedup, closed-candle filter)."""

from __future__ import annotations

import pytest

from elt_btc.candles import (
    OHLCV_COLUMNS,
    deduplicate,
    filter_closed_candles,
    ohlcv_to_dataframe,
    timeframe_to_ms,
)

TF = "15m"
TF_MS = 900_000
T0 = 1_700_000_000_000


def test_timeframe_to_ms():
    assert timeframe_to_ms("1m") == 60_000
    assert timeframe_to_ms("15m") == 900_000
    assert timeframe_to_ms("4h") == 14_400_000
    assert timeframe_to_ms("1d") == 86_400_000


def test_timeframe_to_ms_invalid():
    with pytest.raises(ValueError):
        timeframe_to_ms("15x")


def test_ohlcv_to_dataframe_empty():
    df = ohlcv_to_dataframe([])
    assert list(df.columns) == OHLCV_COLUMNS
    assert df.empty
    assert str(df["timestamp"].dtype) == "int64"


def test_deduplicate_keeps_last_and_sorts():
    rows = [
        [2000, 1.0, 2.0, 0.5, 1.5, 10.0],
        [1000, 1.0, 2.0, 0.5, 1.5, 10.0],
        [2000, 9.0, 9.5, 8.0, 9.2, 5.0],  # duplicate open time, later fetch wins
    ]
    out = deduplicate(ohlcv_to_dataframe(rows))
    assert list(out["timestamp"]) == [1000, 2000]
    assert out.loc[1, "open"] == 9.0


def test_filter_closed_drops_in_progress_candle(ohlcv_factory):
    df = ohlcv_factory([T0, T0 + TF_MS, T0 + 2 * TF_MS])
    now_ms = T0 + 2 * TF_MS + 1  # third candle opened 1 ms ago
    out = filter_closed_candles(df, TF, now_ms)
    assert list(out["timestamp"]) == [T0, T0 + TF_MS]


def test_filter_closed_keeps_candle_closing_exactly_now(ohlcv_factory):
    df = ohlcv_factory([T0, T0 + TF_MS])
    now_ms = T0 + TF_MS  # first candle closes exactly now, second just opened
    out = filter_closed_candles(df, TF, now_ms)
    assert list(out["timestamp"]) == [T0]


def test_filter_closed_empty_frame(ohlcv_factory):
    out = filter_closed_candles(ohlcv_factory([]), TF, T0)
    assert out.empty
