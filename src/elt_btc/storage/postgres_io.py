"""Postgres (Neon) writes for the 15m incremental flow."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import psycopg
from psycopg import sql


def insert_ohlcv(database_url: str, df: pd.DataFrame, table: str = "raw_ohlcv_15m") -> int:
    """Idempotently insert OHLCV rows and return how many were actually inserted.

    Timestamps (epoch ms UTC, candle open time) are stored as ``timestamptz``.
    Rows whose timestamp already exists are left untouched
    (``ON CONFLICT (timestamp) DO NOTHING``), so re-running over an
    overlapping window is safe.
    """
    if df.empty:
        return 0
    query = sql.SQL(
        "INSERT INTO {} (timestamp, open, high, low, close, volume) "
        "VALUES (%s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (timestamp) DO NOTHING"
    ).format(sql.Identifier(table))
    rows = [
        (datetime.fromtimestamp(ts / 1000, tz=UTC), op, hi, lo, cl, vol)
        for ts, op, hi, lo, cl, vol in zip(
            df["timestamp"].tolist(),
            df["open"].tolist(),
            df["high"].tolist(),
            df["low"].tolist(),
            df["close"].tolist(),
            df["volume"].tolist(),
            strict=True,
        )
    ]
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.executemany(query, rows)
        # psycopg 3 reports the cumulative affected-row count for executemany,
        # i.e. the number of rows that did not hit the conflict clause.
        return cur.rowcount
