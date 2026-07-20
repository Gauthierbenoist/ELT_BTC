"""Causality and correctness tests for the volume features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elt_btc.features.volume import build_volume_features

HOUR_MS = 3_600_000
T0 = 1_600_002_000_000


def synthetic_volume_bars(n: int, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = np.exp(rng.normal(3.0, 1.0, n))  # lognormal activity
    return pd.DataFrame(
        {
            "timestamp": T0 + HOUR_MS * np.arange(n, dtype="int64"),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
        }
    )


def test_prefix_invariance_no_look_ahead():
    bars = synthetic_volume_bars(300)
    full = build_volume_features(bars, windows=[12, 24])
    for k in (50, 150, 299):
        prefix = build_volume_features(bars.iloc[:k].reset_index(drop=True), windows=[12, 24])
        pd.testing.assert_series_equal(
            prefix.iloc[k - 1], full.iloc[k - 1], check_names=False, atol=1e-12, rtol=0
        )


def test_baseline_excludes_current_bar():
    bars = synthetic_volume_bars(30)
    features = build_volume_features(bars, windows=[20])
    # z at row 19 uses the mean/std of log1p(volume) over rows 0..18 only.
    log_v = np.log1p(bars["volume"])
    baseline = log_v.iloc[0:19]
    expected = (log_v.iloc[19] - baseline.mean()) / baseline.std()
    assert features.loc[19, "volume_z_20"] == pytest.approx(expected)


def test_volume_spike_detected():
    bars = synthetic_volume_bars(100, seed=1)
    bars.loc[80, "volume"] = bars["volume"].max() * 50
    features = build_volume_features(bars, windows=[20])
    assert features.loc[80, "volume_z_20"] > 2
    assert features["volume_z_20"].abs().dropna().idxmax() == 80


def test_warm_up_rows_are_nan():
    features = build_volume_features(synthetic_volume_bars(40), windows=[20])
    # Needs 19 prior bars, so rows 0..18 are NaN and row 19 onward is defined.
    assert features["volume_z_20"].iloc[:19].isna().all()
    assert not features["volume_z_20"].iloc[19:].isna().any()


def test_constant_series_has_no_defined_zscore():
    bars = synthetic_volume_bars(30)
    bars["volume"] = 7.5
    features = build_volume_features(bars, windows=[20])
    # Zero dispersion in the baseline -> z-score undefined, never a finite number.
    assert not np.isfinite(features["volume_z_20"].to_numpy()).any()


def test_window_below_three_is_rejected():
    with pytest.raises(ValueError):
        build_volume_features(synthetic_volume_bars(30), windows=[2])
