"""Dataset assembly: 1m Parquet lake -> hourly bars -> (X, y, timestamps).

Target definition: ``y_t = 1{close_{t+1} > close_t}`` — the direction of the
*next* bar's close, the only place the future is referenced. Features at row
``t`` use information available at the close of bar ``t`` (see
:mod:`elt_btc.features.ohlc` for the causality contract).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from elt_btc.candles import OHLCV_COLUMNS, timeframe_to_ms
from elt_btc.features.ohlc import build_features
from elt_btc.features.volume import build_volume_features
from elt_btc.ml.config import BenchmarkSettings

logger = logging.getLogger(__name__)

_BAR_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class Dataset:
    """Aligned learning matrix: one row per decision time (bar open time in ms).

    ``ret_next`` is the next-bar simple return the label refers to. Like
    ``y`` it is an *outcome* column — used for evaluation/backtest only,
    never as a feature.
    """

    X: pd.DataFrame
    y: pd.Series
    timestamps: pd.Series
    ret_next: pd.Series


def load_1m_lake(root: Path) -> pd.DataFrame:
    """Load the partitioned 1m lake, sorted and deduplicated on timestamp."""
    df = pd.read_parquet(root)
    df = df[OHLCV_COLUMNS]  # drop hive partition columns (year, month)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    return df.reset_index(drop=True)


def resample_to_bars(df_1m: pd.DataFrame, timeframe: str, min_minutes_per_bar: int) -> pd.DataFrame:
    """Aggregate 1m candles into fixed-timeframe OHLC bars, causally.

    A bar with open time ``T`` uses only the 1m candles in ``[T, T + tf)``.
    Bars built from fewer than ``min_minutes_per_bar`` candles (exchange
    outages) are dropped: their OHLC would be distorted.
    """
    tf_ms = timeframe_to_ms(timeframe)
    bar_open = (df_1m["timestamp"] // tf_ms) * tf_ms
    bars = df_1m.groupby(bar_open).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        n_candles=("close", "size"),
    )
    bars.index.name = "timestamp"
    bars = bars.reset_index()
    complete = bars["n_candles"] >= min_minutes_per_bar
    dropped = int((~complete).sum())
    if dropped:
        logger.warning(
            "Dropped %d/%d %s bar(s) built from fewer than %d one-minute candles",
            dropped,
            len(bars),
            timeframe,
            min_minutes_per_bar,
        )
    return bars.loc[complete, _BAR_COLUMNS].reset_index(drop=True)


def make_target(close: pd.Series) -> pd.Series:
    """``1{close_{t+1} > close_t}``; NaN on the last row (label unknown)."""
    return pd.Series(np.where(close.shift(-1) > close, 1.0, 0.0), index=close.index).where(
        close.shift(-1).notna()
    )


def make_next_return(close: pd.Series) -> pd.Series:
    """Next-bar simple return ``close_{t+1}/close_t - 1``; NaN on the last row."""
    return close.shift(-1) / close - 1.0


def build_dataset(settings: BenchmarkSettings) -> Dataset:
    """Full pipeline: lake -> bars -> features + target, NaN rows dropped."""
    cfg = settings.dataset
    df_1m = load_1m_lake(cfg.parquet_root)
    logger.info("Loaded %d one-minute candles from %s", len(df_1m), cfg.parquet_root)

    bars = resample_to_bars(df_1m, cfg.timeframe, cfg.min_minutes_per_bar)
    logger.info("Resampled into %d %s bars", len(bars), cfg.timeframe)

    features = build_features(bars, settings.features)
    if settings.features.volume_windows:
        volume_features = build_volume_features(bars, settings.features.volume_windows)
        features = pd.concat([features, volume_features], axis=1)
    target = make_target(bars["close"])
    next_return = make_next_return(bars["close"])

    valid = features.notna().all(axis=1) & target.notna()
    X = features.loc[valid].reset_index(drop=True)
    y = target.loc[valid].astype("int64").reset_index(drop=True)
    timestamps = bars.loc[valid, "timestamp"].reset_index(drop=True)
    ret_next = next_return.loc[valid].reset_index(drop=True)
    logger.info(
        "Dataset ready: %d samples x %d features, up-rate %.4f",
        len(X),
        X.shape[1],
        float(y.mean()),
    )
    return Dataset(X=X, y=y, timestamps=timestamps, ret_next=ret_next)
