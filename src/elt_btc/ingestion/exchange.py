"""ccxt exchange access: pagination and exponential backoff.

Rate limiting relies on ccxt's built-in throttler (``enableRateLimit``);
backoff only kicks in when the exchange still pushes back (network errors,
HTTP 429 / rate-limit responses).
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable

import ccxt

from elt_btc.candles import timeframe_to_ms
from elt_btc.config import ExchangeSettings

logger = logging.getLogger(__name__)

OhlcvRow = list[float]


def create_exchange(exchange_id: str) -> ccxt.Exchange:
    """Instantiate a ccxt exchange with built-in rate limiting enabled."""
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({"enableRateLimit": True})


def fetch_page_with_retry(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    limit: int,
    *,
    max_retries: int,
    backoff_base_seconds: float,
    backoff_max_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> list[OhlcvRow]:
    """One ``fetch_ohlcv`` call with exponential backoff and jitter.

    Retries on :class:`ccxt.NetworkError` (which covers rate-limit and DDoS
    protection errors); anything else — e.g. a bad symbol — is a programming
    or configuration error and propagates immediately.

    Args:
        sleep: Injected for testability; defaults to ``time.sleep``.

    Raises:
        ccxt.NetworkError: If all ``max_retries`` attempts fail.
    """
    for attempt in range(1, max_retries + 1):
        try:
            page: list[OhlcvRow] = exchange.fetch_ohlcv(
                symbol, timeframe, since=since_ms, limit=limit
            )
            return page
        except ccxt.NetworkError as exc:
            if attempt == max_retries:
                logger.error("fetch_ohlcv failed after %d attempts: %s", max_retries, exc)
                raise
            delay = min(backoff_max_seconds, backoff_base_seconds * 2 ** (attempt - 1))
            delay *= 0.5 + random.random()  # jitter in [0.5x, 1.5x)
            logger.warning(
                "fetch_ohlcv attempt %d/%d failed (%s), retrying in %.1fs",
                attempt,
                max_retries,
                exc,
                delay,
            )
            sleep(delay)
    raise AssertionError("unreachable")


def fetch_ohlcv_paginated(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    settings: ExchangeSettings,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> list[OhlcvRow]:
    """Fetch every candle whose open time falls in ``[since_ms, until_ms)``.

    Pages through ``fetch_ohlcv`` advancing the cursor past the last candle
    received. Empty pages (dead zones such as pre-listing periods) advance
    the cursor by a full page instead of stopping, so the loop always
    terminates at ``until_ms``.
    """
    tf_ms = timeframe_to_ms(timeframe)
    page_span_ms = settings.page_limit * tf_ms
    rows: list[OhlcvRow] = []
    cursor = since_ms
    while cursor < until_ms:
        page = fetch_page_with_retry(
            exchange,
            symbol,
            timeframe,
            cursor,
            settings.page_limit,
            max_retries=settings.max_retries,
            backoff_base_seconds=settings.backoff_base_seconds,
            backoff_max_seconds=settings.backoff_max_seconds,
            sleep=sleep,
        )
        if not page:
            cursor += page_span_ms
            continue
        rows.extend(row for row in page if cursor <= row[0] < until_ms)
        next_cursor = int(page[-1][0]) + tf_ms
        # Defensive: never let a misbehaving response stall the cursor.
        cursor = next_cursor if next_cursor > cursor else cursor + page_span_ms
    return rows
