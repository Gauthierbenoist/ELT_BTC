"""Hand-computed triple-barrier labeling cases and purge enforcement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from elt_btc.ml.config import BenchmarkSettings, SplitSettings, TargetSettings
from elt_btc.ml.labels import ewma_volatility, triple_barrier_labels


def bars_from_closes(closes, spread: float = 0.0) -> pd.DataFrame:
    """Bars whose high/low equal close +/- spread (path fully controlled)."""
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": np.arange(len(closes), dtype="int64") * 14_400_000,
            "open": closes,
            "high": closes + spread,
            "low": closes - spread,
            "close": closes,
            "volume": np.ones(len(closes)),
        }
    )


def constant_vol(n: int, value: float = 0.01) -> pd.Series:
    return pd.Series(np.full(n, value))


# With vol=1% and pt=sl=1, barriers sit at close_t * (1 +/- 0.01).


def test_upper_barrier_hit_first():
    bars = bars_from_closes([100.0, 100.5, 101.5, 101.0, 101.0, 101.0, 101.0])
    tb = triple_barrier_labels(
        bars,
        vol_span=2,
        pt_mult=1.0,
        sl_mult=1.0,
        max_holding=4,
        volatility=constant_vol(7),
    )
    # Upper barrier 101.0 touched at t+2 (high 101.5).
    assert tb.loc[0, "label"] == 1.0
    assert tb.loc[0, "ret_trade"] == pytest.approx(0.01)
    assert tb.loc[0, "holding_bars"] == 2


def test_lower_barrier_hit_first():
    bars = bars_from_closes([100.0, 99.5, 98.0, 99.0, 99.0, 99.0, 99.0])
    tb = triple_barrier_labels(
        bars,
        vol_span=2,
        pt_mult=1.0,
        sl_mult=1.0,
        max_holding=4,
        volatility=constant_vol(7),
    )
    # Lower barrier 99.0: not touched at t+1 (low 99.5), touched at t+2 (low 98.0).
    assert tb.loc[0, "label"] == 0.0
    assert tb.loc[0, "ret_trade"] == pytest.approx(-0.01)
    assert tb.loc[0, "holding_bars"] == 2


def test_both_barriers_same_bar_is_conservative_loss():
    # Bar t+1 spans both barriers (high 102, low 98): the order inside the
    # bar is unknowable, so the stop-loss wins by convention.
    bars = bars_from_closes([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], spread=2.0)
    tb = triple_barrier_labels(
        bars,
        vol_span=2,
        pt_mult=1.0,
        sl_mult=1.0,
        max_holding=3,
        volatility=constant_vol(6),
    )
    assert tb.loc[0, "label"] == 0.0
    assert tb.loc[0, "holding_bars"] == 1


def test_vertical_barrier_takes_return_sign():
    closes = [100.0, 100.2, 100.3, 100.2, 100.4, 100.0, 100.0, 100.0]
    bars = bars_from_closes(closes)
    tb = triple_barrier_labels(
        bars,
        vol_span=2,
        pt_mult=1.0,
        sl_mult=1.0,
        max_holding=4,
        volatility=constant_vol(8),
    )
    # No +/-1% move within 4 bars: vertical exit at close[4]=100.4 -> up.
    assert tb.loc[0, "label"] == 1.0
    assert tb.loc[0, "ret_trade"] == pytest.approx(0.004)
    assert tb.loc[0, "holding_bars"] == 4


def test_tail_rows_have_no_label():
    bars = bars_from_closes(np.full(10, 100.0))
    tb = triple_barrier_labels(
        bars,
        vol_span=2,
        pt_mult=1.0,
        sl_mult=1.0,
        max_holding=4,
        volatility=constant_vol(10),
    )
    # t + max_holding must stay inside the data: rows 6..9 are unknowable.
    assert tb["label"].iloc[6:].isna().all()
    assert tb["label"].iloc[:6].notna().all()


def test_ewma_volatility_is_causal_and_positive():
    rng = np.random.default_rng(5)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    vol = ewma_volatility(close, span=20)
    assert vol.iloc[:20].isna().all()
    assert (vol.iloc[20:] > 0).all()
    truncated = ewma_volatility(close.iloc[:100], span=20)
    assert truncated.iloc[99] == pytest.approx(vol.iloc[99])  # prefix invariance


def test_config_rejects_purge_smaller_than_label_horizon():
    with pytest.raises(ValidationError, match="purge"):
        BenchmarkSettings(
            target=TargetSettings(type="triple_barrier", max_holding=42),
            split=SplitSettings(purge=24),  # < 42: train labels overlap test
        )
    # Same purge is fine for the 1-bar next_bar target.
    BenchmarkSettings(target=TargetSettings(type="next_bar"), split=SplitSettings(purge=24))
