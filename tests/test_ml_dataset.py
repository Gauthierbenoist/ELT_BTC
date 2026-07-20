"""Resampling, target alignment and gap handling for the ML dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elt_btc.ml.dataset import make_next_return, make_target, resample_to_bars

MIN_MS = 60_000
HOUR_MS = 3_600_000
T0 = 1_600_002_000_000  # exact hour boundary (multiple of 3_600_000)


def minute_frame(hours: int, minutes_per_hour: int = 60) -> pd.DataFrame:
    """1m candles with close = minute index, covering `hours` full hours."""
    rows = []
    for h in range(hours):
        for m in range(minutes_per_hour):
            ts = T0 + h * HOUR_MS + m * MIN_MS
            price = float(h * 100 + m + 1)
            rows.append(
                {
                    "timestamp": ts,
                    "open": price,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price,
                    "volume": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_resample_ohlc_aggregation():
    bars = resample_to_bars(minute_frame(3), "1h", min_minutes_per_bar=45)
    assert len(bars) == 3
    assert list(bars["timestamp"]) == [T0, T0 + HOUR_MS, T0 + 2 * HOUR_MS]
    first = bars.iloc[0]
    assert first["open"] == 1.0  # first minute of the hour
    assert first["close"] == 60.0  # last minute of the hour
    assert first["high"] == 60.5
    assert first["low"] == 0.5
    assert first["volume"] == 60.0  # sum of the hour's 1m volumes (1.0 each)


def test_resample_drops_gappy_bars():
    df = minute_frame(3)
    # Keep only 30 minutes of the second hour: below the 45-minute threshold.
    second_hour = (df["timestamp"] >= T0 + HOUR_MS) & (df["timestamp"] < T0 + 2 * HOUR_MS)
    df = df[~(second_hour & (df["timestamp"] >= T0 + HOUR_MS + 30 * MIN_MS))]
    bars = resample_to_bars(df, "1h", min_minutes_per_bar=45)
    assert list(bars["timestamp"]) == [T0, T0 + 2 * HOUR_MS]


def test_weekly_resample_anchored_on_monday():
    # 1m candles on Wednesday 2024-01-03 must land in the week opening
    # Monday 2024-01-01 00:00 UTC, not in an epoch-floored (Thursday) week.
    monday = 1_704_067_200_000
    wednesday = monday + 2 * 86_400_000
    df = minute_frame(1)
    df["timestamp"] = wednesday + np.arange(60) * MIN_MS
    bars = resample_to_bars(df, "1w", min_minutes_per_bar=30)
    assert list(bars["timestamp"]) == [monday]


def test_make_target_alignment():
    close = pd.Series([100.0, 101.0, 100.5, 100.5, 102.0])
    y = make_target(close)
    # y_t compares close_{t+1} to close_t; equality counts as "not up".
    assert list(y.iloc[:4]) == [1.0, 0.0, 0.0, 1.0]
    assert np.isnan(y.iloc[4])  # last label unknowable without the future


def test_next_return_consistent_with_target():
    close = pd.Series([100.0, 101.0, 100.5, 100.5, 102.0])
    ret_next = make_next_return(close)
    assert ret_next.iloc[0] == pytest.approx(0.01)
    assert np.isnan(ret_next.iloc[4])
    y = make_target(close)
    # The label is exactly the sign of the evaluation return.
    valid = ret_next.notna()
    assert list((ret_next[valid] > 0).astype(float)) == list(y[valid])
