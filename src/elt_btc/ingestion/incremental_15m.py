"""Incremental ingestion of 15m candles into the Neon Postgres table.

Fetches the last ``lookback_hours`` (48h by default) of closed 15m candles
and inserts them idempotently (``ON CONFLICT DO NOTHING``). Designed to run
daily from GitHub Actions; any unhandled exception exits non-zero and fails
the job.

Usage::

    DATABASE_URL=postgres://... uv run python -m elt_btc.ingestion.incremental_15m
    uv run python -m elt_btc.ingestion.incremental_15m --dry-run   # no DB needed
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from elt_btc.candles import deduplicate, filter_closed_candles, ohlcv_to_dataframe
from elt_btc.config import load_settings
from elt_btc.ingestion.exchange import create_exchange, fetch_ohlcv_paginated
from elt_btc.storage.postgres_io import insert_ohlcv
from elt_btc.utils.logging import setup_logging
from elt_btc.utils.tls import configure_tls
from elt_btc.validation.schemas import find_gaps, gaps_to_ranges, validate_ohlcv

logger = logging.getLogger(__name__)

_DATABASE_URL_ENV_VAR = "DATABASE_URL"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to settings.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and validate only; skip the database insert",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    configure_tls()
    settings = load_settings(args.config)
    cfg = settings.incremental

    now_ms = int(time.time() * 1000)
    since_ms = now_ms - cfg.lookback_hours * 3_600_000
    exchange = create_exchange(settings.exchange.id, settings.exchange.public_api_url)
    logger.info(
        "Fetching %s %s candles since %s (lookback %dh)",
        settings.exchange.symbol,
        cfg.timeframe,
        datetime.fromtimestamp(since_ms / 1000, tz=UTC).isoformat(),
        cfg.lookback_hours,
    )

    rows = fetch_ohlcv_paginated(
        exchange, settings.exchange.symbol, cfg.timeframe, since_ms, now_ms, settings.exchange
    )
    df = filter_closed_candles(deduplicate(ohlcv_to_dataframe(rows)), cfg.timeframe, now_ms)
    if df.empty:
        raise RuntimeError(
            f"No closed {cfg.timeframe} candle returned over the last {cfg.lookback_hours}h"
        )

    validate_ohlcv(df)
    gaps = find_gaps(df, cfg.timeframe)
    if gaps:
        logger.warning(
            "%d missing candle(s) in the fetched window: %s",
            len(gaps),
            gaps_to_ranges(gaps, cfg.timeframe),
        )

    first_ts = datetime.fromtimestamp(int(df["timestamp"].iloc[0]) / 1000, tz=UTC)
    last_ts = datetime.fromtimestamp(int(df["timestamp"].iloc[-1]) / 1000, tz=UTC)
    logger.info(
        "Fetched %d closed candles [%s -> %s]", len(df), first_ts.isoformat(), last_ts.isoformat()
    )

    if args.dry_run:
        logger.info("Dry run: skipping database insert")
        return 0

    database_url = os.environ.get(_DATABASE_URL_ENV_VAR)
    if not database_url:
        raise RuntimeError(f"{_DATABASE_URL_ENV_VAR} environment variable is not set")

    inserted = insert_ohlcv(database_url, df, table=cfg.table)
    logger.info(
        "Inserted %d new row(s) into %s (%d fetched, %d already present)",
        inserted,
        cfg.table,
        len(df),
        len(df) - inserted,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
