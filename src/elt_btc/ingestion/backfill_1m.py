"""Backfill Binance spot 1m OHLCV into a locally partitioned Parquet lake.

One calendar month = one partition (``data/raw/1m/year=YYYY/month=MM``).
The script is resumable: complete partitions are skipped, so it can be
interrupted and re-run until the whole history is on disk.

Usage::

    uv run python -m elt_btc.ingestion.backfill_1m [--end YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import UTC, date, datetime
from pathlib import Path

from elt_btc.candles import (
    deduplicate,
    filter_closed_candles,
    ohlcv_to_dataframe,
    timeframe_to_ms,
)
from elt_btc.config import Settings, load_settings
from elt_btc.ingestion.exchange import create_exchange, fetch_ohlcv_paginated
from elt_btc.storage.parquet_io import last_timestamp, partition_path, write_partition
from elt_btc.utils.logging import setup_logging
from elt_btc.utils.tls import configure_tls
from elt_btc.validation.schemas import find_gaps, gaps_to_ranges, validate_ohlcv

logger = logging.getLogger(__name__)


def date_to_epoch_ms(day: date) -> int:
    """Epoch milliseconds of a calendar day's midnight, UTC."""
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)


def month_range(start: date, end: date) -> list[tuple[int, int]]:
    """All (year, month) pairs from ``start``'s month through ``end``'s month."""
    months: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def month_bounds_ms(year: int, month: int) -> tuple[int, int]:
    """Half-open UTC bounds ``[month start, next month start)`` in epoch ms."""
    start = datetime(year, month, 1, tzinfo=UTC)
    next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
    end = datetime(next_year, next_month, 1, tzinfo=UTC)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def is_partition_complete(root: Path, year: int, month: int, timeframe: str) -> bool:
    """Whether a partition holds the month's final candle.

    Atomic writes guarantee a partition file is never truncated, but a
    partition written while its month was still in progress is partial; it
    is detected here (last stored open time < last open time of the month)
    and re-downloaded on the next run.
    """
    path = partition_path(root, year, month)
    if not path.exists():
        return False
    last = last_timestamp(path)
    if last is None:
        return False
    _, month_end_ms = month_bounds_ms(year, month)
    return last >= month_end_ms - timeframe_to_ms(timeframe)


def _backfill_month(
    settings: Settings,
    exchange: object,
    year: int,
    month: int,
    now_ms: int,
) -> int:
    """Download, validate and write one monthly partition; returns rows written."""
    cfg = settings.backfill
    timeframe = cfg.timeframe
    month_start_ms, month_end_ms = month_bounds_ms(year, month)
    since_ms = max(month_start_ms, date_to_epoch_ms(cfg.start))
    until_ms = min(month_end_ms, now_ms)

    rows = fetch_ohlcv_paginated(
        exchange, settings.exchange.symbol, timeframe, since_ms, until_ms, settings.exchange
    )
    df = filter_closed_candles(deduplicate(ohlcv_to_dataframe(rows)), timeframe, now_ms)
    if df.empty:
        logger.warning("year=%04d month=%02d: no closed candle returned, skipping", year, month)
        return 0

    validate_ohlcv(df)
    gaps = find_gaps(df, timeframe)
    if gaps:
        ranges = gaps_to_ranges(gaps, timeframe)
        logger.warning(
            "year=%04d month=%02d: %d missing candle(s) in %d gap(s): %s",
            year,
            month,
            len(gaps),
            len(ranges),
            [
                (
                    datetime.fromtimestamp(a / 1000, tz=UTC).isoformat(),
                    datetime.fromtimestamp(b / 1000, tz=UTC).isoformat(),
                )
                for a, b in ranges
            ],
        )

    path = write_partition(df, cfg.parquet_root, year, month, compression=cfg.compression)
    logger.info(
        "year=%04d month=%02d: wrote %d rows to %s [%s -> %s]",
        year,
        month,
        len(df),
        path,
        datetime.fromtimestamp(int(df["timestamp"].iloc[0]) / 1000, tz=UTC).isoformat(),
        datetime.fromtimestamp(int(df["timestamp"].iloc[-1]) / 1000, tz=UTC).isoformat(),
    )
    return len(df)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to settings.yaml")
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=None,
        help="Exclusive end date (UTC, YYYY-MM-DD); defaults to now",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    configure_tls()
    settings = load_settings(args.config)
    cfg = settings.backfill

    now_ms = int(time.time() * 1000)
    if args.end is not None:
        now_ms = min(now_ms, date_to_epoch_ms(args.end))
    end_date = datetime.fromtimestamp(now_ms / 1000, tz=UTC).date()

    exchange = create_exchange(settings.exchange.id, settings.exchange.public_api_url)
    logger.info(
        "Backfilling %s %s from %s to %s into %s",
        settings.exchange.symbol,
        cfg.timeframe,
        cfg.start.isoformat(),
        end_date.isoformat(),
        cfg.parquet_root,
    )

    total_rows = 0
    skipped = 0
    for year, month in month_range(cfg.start, end_date):
        if is_partition_complete(cfg.parquet_root, year, month, cfg.timeframe):
            logger.info("year=%04d month=%02d: partition complete, skipping", year, month)
            skipped += 1
            continue
        total_rows += _backfill_month(settings, exchange, year, month, now_ms)

    logger.info("Backfill done: %d rows written, %d partition(s) skipped", total_rows, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
