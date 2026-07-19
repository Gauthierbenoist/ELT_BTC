"""Causality (anti-look-ahead) and correctness tests for the OHLC features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elt_btc.features.ohlc import build_features, log_returns, rsi
from elt_btc.ml.config import FeatureSettings

HOUR_MS = 3_600_000
T0 = 1_600_002_000_000  # exact hour boundary (multiple of 3_600_000)


def synthetic_bars(n: int, seed: int = 7) -> pd.DataFrame:
    """Deterministic random-walk hourly OHLC frame."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.normal(0, 0.005, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame(
        {
            "timestamp": T0 + HOUR_MS * np.arange(n, dtype="int64"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


SETTINGS = FeatureSettings(
    momentum_windows=[3, 12, 24],
    vol_windows=[12, 24],
    range_vol_windows=[12],
    channel_windows=[24],
    rsi_period=14,
)


def test_prefix_invariance_no_look_ahead():
    """THE causality guarantee: truncating the future must not change the past.

    If any feature at row t used information from rows > t, its value would
    differ between the full frame and a frame truncated right after t.
    """
    bars = synthetic_bars(500)
    full = build_features(bars, SETTINGS)
    for k in (60, 250, 499):
        prefix = build_features(bars.iloc[:k].reset_index(drop=True), SETTINGS)
        pd.testing.assert_series_equal(
            prefix.iloc[k - 1], full.iloc[k - 1], check_names=False, atol=1e-12, rtol=0
        )


def test_future_perturbation_leaves_past_untouched():
    """Complementary check: corrupting rows > k must not alter features <= k."""
    bars = synthetic_bars(300)
    reference = build_features(bars, SETTINGS)
    corrupted = bars.copy()
    corrupted.loc[201:, ["open", "high", "low", "close"]] *= 100.0
    perturbed = build_features(corrupted, SETTINGS)
    pd.testing.assert_frame_equal(perturbed.iloc[:201], reference.iloc[:201])


def test_log_returns_values():
    close = pd.Series([100.0, 110.0, 99.0])
    r = log_returns(close)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(np.log(1.10))
    assert r.iloc[2] == pytest.approx(np.log(99.0 / 110.0))


def test_rsi_bounds_and_extremes():
    n = 50
    rising = pd.Series(np.linspace(100, 200, n))
    assert rsi(rising, 14).iloc[-1] == pytest.approx(100.0)
    falling = pd.Series(np.linspace(200, 100, n))
    assert rsi(falling, 14).iloc[-1] == pytest.approx(0.0)
    bars = synthetic_bars(200)
    values = rsi(bars["close"], 14).dropna()
    assert ((values >= 0) & (values <= 100)).all()


def test_bounded_features_stay_bounded():
    bars = synthetic_bars(400)
    features = build_features(bars, SETTINGS)
    assert features["close_pos_bar"].dropna().between(0, 1).all()
    assert features["close_pos_24"].dropna().between(0, 1).all()
    assert (features["dist_high_24"].dropna() <= 0).all()


def test_warm_up_rows_are_nan():
    bars = synthetic_bars(100)
    features = build_features(bars, SETTINGS)
    assert features["ret_24"].iloc[:24].isna().all()
    assert not np.isnan(features["ret_24"].iloc[24])
