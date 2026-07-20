"""Hand-computed cases for the sequential trade-level backtest."""

from __future__ import annotations

import numpy as np
import pytest

from elt_btc.ml.trade_backtest import simulate_trades

BAR_MS = 14_400_000  # 4h


def ts(n: int) -> np.ndarray:
    return np.arange(n, dtype="int64") * BAR_MS


def test_signal_ignored_while_trade_open():
    # Entry at t=0 holds 3 bars: signals at t=1,2 are skipped; t=3 is free.
    p_up = np.array([0.9, 0.9, 0.9, 0.9])
    ret = np.array([0.02, 0.05, 0.05, 0.01])
    holding = np.array([3, 1, 1, 1])
    result = simulate_trades(ts(4), p_up, ret, holding, bar_ms=BAR_MS, fee_rate=0.0)
    assert len(result.trades) == 2
    assert list(result.trades["entry_ts"]) == [0, 3 * BAR_MS]
    assert list(result.trades["ret_gross"]) == [0.02, 0.01]


def test_short_mirrors_long_trade_return():
    p_up = np.array([0.1])  # confident short
    ret = np.array([-0.02])  # the long virtual trade loses 2%
    result = simulate_trades(ts(1), p_up, ret, np.array([1]), bar_ms=BAR_MS, fee_rate=0.0)
    assert result.trades.loc[0, "direction"] == -1
    assert result.trades.loc[0, "ret_gross"] == pytest.approx(0.02)  # short wins


def test_round_trip_fees():
    p_up = np.array([0.9])
    ret = np.array([0.02])
    result = simulate_trades(ts(1), p_up, ret, np.array([1]), bar_ms=BAR_MS, fee_rate=0.001)
    assert result.trades.loc[0, "ret_net"] == pytest.approx(0.02 - 0.002)


def test_neutral_band_skips_unconfident_signals():
    p_up = np.array([0.52, 0.48, 0.70])
    ret = np.array([0.01, 0.01, 0.01])
    result = simulate_trades(
        ts(3),
        p_up,
        ret,
        np.ones(3, dtype=int),
        bar_ms=BAR_MS,
        fee_rate=0.0,
        threshold_band=0.05,
    )
    assert len(result.trades) == 1
    assert result.trades.loc[0, "entry_ts"] == 2 * BAR_MS


def test_metrics_hand_computed():
    p_up = np.array([0.9, 0.9])
    ret = np.array([0.10, -0.05])
    holding = np.array([1, 1])
    result = simulate_trades(ts(2), p_up, ret, holding, bar_ms=BAR_MS, fee_rate=0.0)
    m = result.metrics
    assert m["n_trades"] == 2
    assert m["win_rate"] == pytest.approx(0.5)
    assert m["avg_ret_net"] == pytest.approx(0.025)
    # Equity 1.10 -> 1.045: drawdown = 0.95 - 1 = -5%.
    assert m["max_drawdown_net"] == pytest.approx(-0.05)
    assert m["exposure"] == pytest.approx(1.0)  # in a trade the whole window


def test_no_trades():
    result = simulate_trades(
        ts(5),
        np.full(5, 0.5),
        np.ones(5),
        np.ones(5, dtype=int),
        bar_ms=BAR_MS,
        fee_rate=0.001,
        threshold_band=0.05,
    )
    assert result.metrics["n_trades"] == 0
    assert result.metrics["exposure"] == 0.0
    assert np.isnan(result.metrics["win_rate"])


def test_unsorted_timestamps_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        simulate_trades(
            np.array([0, 2 * BAR_MS, BAR_MS]),
            np.full(3, 0.9),
            np.ones(3),
            np.ones(3, dtype=int),
            bar_ms=BAR_MS,
            fee_rate=0.0,
        )
