"""Pure candle-level transformations shared by both ingestion flows.

Every timestamp is a candle *open time* in milliseconds since the Unix
epoch, UTC. These functions do no I/O so they are directly unit-testable.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

_OHLCV_DTYPES = {
    "timestamp": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}

_TIMEFRAME_RE = re.compile(r"^(\d+)([mhd])$")
_UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}


def timeframe_to_ms(timeframe: str) -> int:
    """Return the duration of one candle in milliseconds.

    Supports ccxt-style timeframes with minute/hour/day units, e.g. ``"1m"``,
    ``"15m"``, ``"4h"``, ``"1d"``.

    Raises:
        ValueError: If the timeframe string is not recognized.
    """
    match = _TIMEFRAME_RE.match(timeframe)
    if match is None:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    return int(match.group(1)) * _UNIT_MS[match.group(2)]


def ohlcv_to_dataframe(rows: Sequence[Sequence[float]]) -> pd.DataFrame:
    """Convert raw ccxt OHLCV rows into a typed DataFrame.

    ccxt returns ``[timestamp_ms, open, high, low, close, volume]`` per row.
    """
    return pd.DataFrame(rows, columns=OHLCV_COLUMNS).astype(_OHLCV_DTYPES)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate open times (keeping the last occurrence) and sort ascending."""
    return (
        df.drop_duplicates(subset="timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def filter_closed_candles(df: pd.DataFrame, timeframe: str, now_ms: int) -> pd.DataFrame:
    """Keep only candles whose period has fully elapsed.

    A candle opening at ``t`` is closed once ``t + timeframe <= now``; the
    in-progress candle must never be persisted.
    """
    tf_ms = timeframe_to_ms(timeframe)
    return df.loc[df["timestamp"] + tf_ms <= now_ms].reset_index(drop=True)
