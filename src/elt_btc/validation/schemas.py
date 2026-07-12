"""Pandera validation schema and temporal-grid gap detection for OHLCV frames.

Hard checks (raise): OHLC coherence, positive prices, non-negative volume,
unique timestamps. Soft check (caller logs a warning): missing candles on
the expected time grid — exchanges have legitimate gaps (maintenance
windows), so gaps must never block ingestion.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from elt_btc.candles import timeframe_to_ms

OHLCV_SCHEMA = pa.DataFrameSchema(
    columns={
        "timestamp": pa.Column(int, checks=pa.Check.gt(0), unique=True),
        "open": pa.Column(float, checks=pa.Check.gt(0)),
        "high": pa.Column(float, checks=pa.Check.gt(0)),
        "low": pa.Column(float, checks=pa.Check.gt(0)),
        "close": pa.Column(float, checks=pa.Check.gt(0)),
        "volume": pa.Column(float, checks=pa.Check.ge(0)),
    },
    checks=[
        pa.Check(
            lambda df: df["low"] <= df[["open", "close"]].min(axis=1),
            name="low_le_min_open_close",
            error="low must be <= min(open, close)",
        ),
        pa.Check(
            lambda df: df["high"] >= df[["open", "close"]].max(axis=1),
            name="high_ge_max_open_close",
            error="high must be >= max(open, close)",
        ),
    ],
    strict=True,
)


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an OHLCV frame against :data:`OHLCV_SCHEMA`.

    Uses lazy validation so all failures are reported at once.

    Raises:
        pandera.errors.SchemaErrors: If any hard check fails.
    """
    return OHLCV_SCHEMA.validate(df, lazy=True)


def find_gaps(df: pd.DataFrame, timeframe: str) -> list[int]:
    """Return the open times missing from the regular grid spanned by ``df``.

    The grid runs from the frame's min to max timestamp with one candle per
    ``timeframe`` step. An empty or single-row frame has no detectable gaps.
    """
    if len(df) < 2:
        return []
    tf_ms = timeframe_to_ms(timeframe)
    present = set(df["timestamp"].astype("int64"))
    first, last = min(present), max(present)
    return [ts for ts in range(first, last + 1, tf_ms) if ts not in present]


def gaps_to_ranges(gaps: list[int], timeframe: str) -> list[tuple[int, int]]:
    """Compress a sorted list of missing open times into (first, last) ranges.

    Consecutive missing candles collapse into a single range, which keeps
    warning logs readable when an exchange outage spans hours.
    """
    if not gaps:
        return []
    tf_ms = timeframe_to_ms(timeframe)
    ranges: list[tuple[int, int]] = []
    range_start = prev = gaps[0]
    for ts in gaps[1:]:
        if ts - prev > tf_ms:
            ranges.append((range_start, prev))
            range_start = ts
        prev = ts
    ranges.append((range_start, prev))
    return ranges
