"""Pivot-distance features: detection, confirmation delay, causality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elt_btc.features.ohlc import pivot_distances

HOUR_MS = 3_600_000
T0 = 1_600_002_000_000


def bars_with_spike(n: int = 30, spike_at: int = 10, dip_at: int = 20) -> pd.DataFrame:
    """Flat ~100 bars with one clear pivot high (110) and one pivot low (90)."""
    high = np.full(n, 100.5)
    low = np.full(n, 99.5)
    close = np.full(n, 100.0)
    high[spike_at] = 110.0
    low[dip_at] = 90.0
    return pd.DataFrame(
        {
            "timestamp": T0 + HOUR_MS * np.arange(n, dtype="int64"),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(n),
        }
    )


def test_pivot_high_distance_after_confirmation():
    bars = bars_with_spike()
    features = pivot_distances(bars, window=5)
    # The pivot at bar 10 is only knowable at bar 15 (5 right-hand bars).
    assert features["pivot_high_dist_5"].iloc[15] == pytest.approx(110.0 / 100.0 - 1.0)  # +10%
    assert features["pivot_high_dist_5"].iloc[29] == pytest.approx(0.10)  # ffill persists
    # Sign convention: pivot low below the close -> negative distance.
    assert features["pivot_low_dist_5"].iloc[25] == pytest.approx(90.0 / 100.0 - 1.0)  # -10%


def test_pivot_not_visible_before_confirmation():
    bars = bars_with_spike()
    features = pivot_distances(bars, window=5)
    # Before bar 15 the spike at 10 must NOT appear in the feature.
    early = features["pivot_high_dist_5"].iloc[:15].dropna()
    assert not np.isclose(early, 0.10).any()


def test_prefix_invariance_no_look_ahead():
    rng = np.random.default_rng(21)
    n = 200
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    spread = np.abs(rng.normal(0, 0.004, n)) * close
    bars = pd.DataFrame(
        {
            "timestamp": T0 + HOUR_MS * np.arange(n, dtype="int64"),
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": np.ones(n),
        }
    )
    full = pivot_distances(bars, window=5)
    for k in (60, 120, 199):
        prefix = pivot_distances(bars.iloc[:k].reset_index(drop=True), window=5)
        pd.testing.assert_series_equal(
            prefix.iloc[k - 1], full.iloc[k - 1], check_names=False, atol=1e-12, rtol=0
        )


def test_nearest_in_time_pivot_wins():
    # Two pivot highs; after the second is confirmed, it replaces the first.
    bars = bars_with_spike(n=40, spike_at=8, dip_at=35)
    bars.loc[22, "high"] = 105.0  # second, smaller pivot high
    features = pivot_distances(bars, window=5)
    assert features["pivot_high_dist_5"].iloc[26] == pytest.approx(0.10)  # still the first
    assert features["pivot_high_dist_5"].iloc[27] == pytest.approx(0.05)  # 22 confirmed at 27
