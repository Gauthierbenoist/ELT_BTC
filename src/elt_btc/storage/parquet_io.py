"""Partitioned Parquet storage for the raw 1m OHLCV lake.

Layout: ``<root>/year=YYYY/month=MM/data.parquet``. Writes are atomic
(temp file + rename), so any partition present on disk was fully written.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pandas as pd

PARTITION_FILENAME = "data.parquet"


def partition_path(root: Path, year: int, month: int) -> Path:
    """Path of the Parquet file for one calendar-month partition."""
    return root / f"year={year:04d}" / f"month={month:02d}" / PARTITION_FILENAME


def existing_partitions(root: Path) -> set[tuple[int, int]]:
    """Return the (year, month) pairs that already have a partition file."""
    partitions: set[tuple[int, int]] = set()
    for file in root.glob(f"year=*/month=*/{PARTITION_FILENAME}"):
        try:
            year = int(file.parent.parent.name.removeprefix("year="))
            month = int(file.parent.name.removeprefix("month="))
        except ValueError:
            continue
        partitions.add((year, month))
    return partitions


def write_partition(
    df: pd.DataFrame,
    root: Path,
    year: int,
    month: int,
    compression: Literal["zstd", "snappy"] = "zstd",
) -> Path:
    """Atomically write one monthly partition and return its path.

    The frame is written to a temp file in the target directory, then moved
    into place with ``os.replace`` — a crash mid-write never leaves a
    truncated ``data.parquet`` behind.
    """
    path = partition_path(root, year, month)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / (path.name + ".tmp")
    df.to_parquet(tmp_path, compression=compression, index=False)
    os.replace(tmp_path, path)
    return path


def read_partition(root: Path, year: int, month: int) -> pd.DataFrame:
    """Read one monthly partition into a DataFrame."""
    return pd.read_parquet(partition_path(root, year, month))


def last_timestamp(path: Path) -> int | None:
    """Max open time (epoch ms) stored in a partition, or None if empty.

    Only the timestamp column is read, so this stays cheap even on large
    partitions — it is used to decide whether a partition is complete.
    """
    frame = pd.read_parquet(path, columns=["timestamp"])
    if frame.empty:
        return None
    return int(frame["timestamp"].max())
