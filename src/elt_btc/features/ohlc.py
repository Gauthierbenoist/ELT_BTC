"""Causal OHLC features for hourly (or any fixed-timeframe) bars.

Causality contract — the reason this module is safe against look-ahead bias:
every feature at row ``t`` may only use rows ``<= t``. Allowed pandas
operations: ``rolling(w)`` (window ends at the current row), ``shift(k)``
with ``k >= 0``, ``diff``, ``ewm`` (past-weighted), and per-row arithmetic.
Forbidden: ``shift(-k)``, centered windows, and any global statistic
(whole-column mean/std). The contract is enforced by the prefix-invariance
test in ``tests/test_features_ohlc.py``: features computed on a truncated
frame must match those computed on the full frame, row for row.

Input frames follow the repo convention: ``timestamp`` is the bar open time
in epoch ms UTC, one row per bar, sorted ascending.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from elt_btc.ml.config import FeatureSettings

_HOUR_MS = 3_600_000
_DAY_MS = 86_400_000


def log_returns(close: pd.Series) -> pd.Series:
    """One-bar log return, NaN on the first row."""
    return pd.Series(np.log(close / close.shift(1)), index=close.index)


def rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI in [0, 100], computed causally via ewm smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return 100.0 * avg_gain / (avg_gain + avg_loss)


def parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Parkinson range volatility over a trailing window of bars."""
    log_hl = pd.Series(np.log(high / low), index=high.index)
    return pd.Series(
        np.sqrt((log_hl**2).rolling(window).mean() / (4.0 * math.log(2.0))), index=high.index
    )


def garman_klass_vol(bars: pd.DataFrame, window: int) -> pd.Series:
    """Garman-Klass OHLC volatility over a trailing window of bars."""
    log_hl = pd.Series(np.log(bars["high"] / bars["low"]), index=bars.index)
    log_co = pd.Series(np.log(bars["close"] / bars["open"]), index=bars.index)
    variance = (0.5 * log_hl**2 - (2.0 * math.log(2.0) - 1.0) * log_co**2).rolling(window).mean()
    return pd.Series(np.sqrt(variance.clip(lower=0.0)), index=bars.index)


def build_features(bars: pd.DataFrame, settings: FeatureSettings) -> pd.DataFrame:
    """Assemble the full causal feature matrix (same index as ``bars``).

    Warm-up rows (rolling windows not yet filled) contain NaN; the dataset
    assembly is responsible for dropping them.
    """
    open_, high = bars["open"], bars["high"]
    low, close = bars["low"], bars["close"]
    out: dict[str, pd.Series] = {}

    r = log_returns(close)
    out["ret_1"] = r
    for window in settings.momentum_windows:
        out[f"ret_{window}"] = r.rolling(window).sum()

    for window in settings.vol_windows:
        out[f"vol_{window}"] = r.rolling(window).std()
    w_short, w_long = min(settings.vol_windows), max(settings.vol_windows)
    out[f"vol_ratio_{w_short}_{w_long}"] = out[f"vol_{w_short}"] / out[f"vol_{w_long}"]

    for window in settings.range_vol_windows:
        out[f"parkinson_{window}"] = parkinson_vol(high, low, window)
        out[f"garman_klass_{window}"] = garman_klass_vol(bars, window)

    out["range_pct"] = (high - low) / close
    bar_range = high - low
    out["close_pos_bar"] = pd.Series(
        np.where(bar_range > 0, (close - low) / bar_range, 0.5), index=bars.index
    )
    for window in settings.channel_windows:
        roll_high = high.rolling(window).max()
        roll_low = low.rolling(window).min()
        channel = roll_high - roll_low
        out[f"close_pos_{window}"] = pd.Series(
            np.where(channel > 0, (close - roll_low) / channel, 0.5), index=bars.index
        )
        out[f"dist_high_{window}"] = close / roll_high - 1.0

    out["gap"] = open_ / close.shift(1) - 1.0
    out[f"rsi_{settings.rsi_period}"] = rsi(close, settings.rsi_period)

    # Calendar features: derived from the bar open time only (no external data).
    hour = (bars["timestamp"] // _HOUR_MS) % 24
    day_of_week = (bars["timestamp"] // _DAY_MS + 4) % 7  # epoch day 0 = Thursday
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)

    return pd.DataFrame(out, index=bars.index)
