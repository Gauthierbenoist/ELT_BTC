"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd
import pytest

from elt_btc.candles import OHLCV_COLUMNS


def _make_ohlcv(timestamps: Sequence[int], base_price: float = 100.0) -> pd.DataFrame:
    """Build a schema-valid OHLCV frame with one row per open time (epoch ms)."""
    rows = []
    for i, ts in enumerate(timestamps):
        open_ = base_price + i
        close = open_ + 0.5
        rows.append(
            {
                "timestamp": ts,
                "open": open_,
                "high": close + 1.0,
                "low": open_ - 1.0,
                "close": close,
                "volume": 10.0,
            }
        )
    df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    return df.astype(
        {
            "timestamp": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        }
    )


@pytest.fixture
def ohlcv_factory() -> Callable[..., pd.DataFrame]:
    """Factory fixture: ``ohlcv_factory([t0, t1, ...])`` -> valid OHLCV frame."""
    return _make_ohlcv
