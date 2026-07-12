"""Pagination and backoff tests against a fake in-memory exchange."""

from __future__ import annotations

import ccxt
import pytest

from elt_btc.config import ExchangeSettings
from elt_btc.ingestion.exchange import fetch_ohlcv_paginated, fetch_page_with_retry

TF = "1m"
TF_MS = 60_000
T0 = 1_600_000_000_000


class GridExchange:
    """Fake exchange serving one candle per minute over [data_start, data_end)."""

    def __init__(self, data_start: int, data_end: int, fail_times: int = 0) -> None:
        self.data_start = data_start
        self.data_end = data_end
        self.fail_times = fail_times
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ccxt.NetworkError("transient network failure")
        first = max(since, self.data_start)
        offset = (first - self.data_start) % TF_MS
        if offset:
            first += TF_MS - offset
        page = []
        ts = first
        while ts < self.data_end and len(page) < limit:
            page.append([float(ts), 1.0, 2.0, 0.5, 1.5, 10.0])
            ts += TF_MS
        return page


def settings(**overrides) -> ExchangeSettings:
    defaults = {
        "page_limit": 100,
        "max_retries": 3,
        "backoff_base_seconds": 0.01,
        "backoff_max_seconds": 0.05,
    }
    return ExchangeSettings(**{**defaults, **overrides})


def test_pagination_collects_full_range():
    n = 250  # 3 pages of 100
    exchange = GridExchange(T0, T0 + n * TF_MS)
    rows = fetch_ohlcv_paginated(exchange, "BTC/USDT", TF, T0, T0 + n * TF_MS, settings())
    timestamps = [int(r[0]) for r in rows]
    assert len(rows) == n
    assert timestamps == sorted(set(timestamps))  # no duplicates across page seams
    assert timestamps[0] == T0
    assert timestamps[-1] == T0 + (n - 1) * TF_MS


def test_until_is_exclusive():
    exchange = GridExchange(T0, T0 + 100 * TF_MS)
    until = T0 + 10 * TF_MS
    rows = fetch_ohlcv_paginated(exchange, "BTC/USDT", TF, T0, until, settings())
    assert len(rows) == 10
    assert int(rows[-1][0]) == until - TF_MS


def test_empty_pages_skip_dead_zone():
    # Data only starts 500 candles after `since` (e.g. pre-listing period).
    data_start = T0 + 500 * TF_MS
    exchange = GridExchange(data_start, data_start + 50 * TF_MS)
    rows = fetch_ohlcv_paginated(exchange, "BTC/USDT", TF, T0, data_start + 50 * TF_MS, settings())
    assert len(rows) == 50
    assert int(rows[0][0]) == data_start


def test_retry_then_success():
    exchange = GridExchange(T0, T0 + 10 * TF_MS, fail_times=2)
    sleeps: list[float] = []
    page = fetch_page_with_retry(
        exchange,
        "BTC/USDT",
        TF,
        T0,
        100,
        max_retries=5,
        backoff_base_seconds=0.01,
        backoff_max_seconds=0.05,
        sleep=sleeps.append,
    )
    assert len(page) == 10
    assert len(sleeps) == 2
    assert all(delay > 0 for delay in sleeps)


def test_retry_exhausted_raises():
    exchange = GridExchange(T0, T0 + 10 * TF_MS, fail_times=99)
    sleeps: list[float] = []
    with pytest.raises(ccxt.NetworkError):
        fetch_page_with_retry(
            exchange,
            "BTC/USDT",
            TF,
            T0,
            100,
            max_retries=3,
            backoff_base_seconds=0.01,
            backoff_max_seconds=0.05,
            sleep=sleeps.append,
        )
    assert len(sleeps) == 2  # no sleep after the final attempt
