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


def test_relative_volume_of_constant_series_is_one():
    bars = synthetic_volume_bars(50)
    bars["volume"] = 7.5
    features = build_volume_features(bars, windows=[12])
    assert features["volume_rel_12"].dropna().to_numpy() == pytest.approx(np.ones(50 - 11))


def test_volume_spike_detected():
    bars = synthetic_volume_bars(100, seed=1)
    bars.loc[80, "volume"] = bars["volume"].max() * 50
    features = build_volume_features(bars, windows=[24])
    assert features.loc[80, "volume_rel_24"] > 10
    assert features.loc[80, "volume_z_24"] > 2
    assert features["volume_z_24"].abs().dropna().max() == features.loc[80, "volume_z_24"]


def test_warm_up_rows_are_nan():
    features = build_volume_features(synthetic_volume_bars(40), windows=[24])
    assert features["volume_z_24"].iloc[:23].isna().all()
    assert not features["volume_z_24"].iloc[24:].isna().any()
