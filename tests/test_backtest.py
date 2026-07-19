"""Hand-computed cases for the evaluation backtest."""

from __future__ import annotations

import numpy as np
import pytest

from elt_btc.ml.backtest import (
    backtest_metrics,
    max_drawdown,
    positions_from_proba,
    sharpe_ratio,
    strategy_returns,
)

BARS_PER_YEAR = 24 * 365


def test_positions_threshold_band():
    p = np.array([0.7, 0.52, 0.5, 0.48, 0.3])
    assert list(positions_from_proba(p, threshold_band=0.0)) == [1, 1, 0, -1, -1]
    assert list(positions_from_proba(p, threshold_band=0.05)) == [1, 0, 0, 0, -1]


def test_sharpe_known_value():
    returns = np.array([0.01, -0.01, 0.01, -0.01, 0.01, 0.03])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(BARS_PER_YEAR)
    assert sharpe_ratio(returns, BARS_PER_YEAR) == pytest.approx(expected)
    assert np.isnan(sharpe_ratio(np.zeros(10), BARS_PER_YEAR))  # zero std
    assert np.isnan(sharpe_ratio(np.array([0.01]), BARS_PER_YEAR))  # too short


def test_max_drawdown_hand_computed():
    # Equity: 1.10 -> 0.88: drawdown = 0.88/1.10 - 1 = -20%.
    assert max_drawdown(np.array([0.10, -0.20])) == pytest.approx(-0.20)
    assert max_drawdown(np.array([0.05, 0.05])) == 0.0
    assert max_drawdown(np.array([])) == 0.0


def test_perfect_foresight_is_profitable():
    rng = np.random.default_rng(1)
    ret = rng.normal(0, 0.01, 500)
    p_up = np.where(ret > 0, 0.9, 0.1)  # oracle
    metrics = backtest_metrics(p_up, ret, fee_rate=0.0, bars_per_year=BARS_PER_YEAR)
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["sharpe_gross"] > 5
    assert metrics["exposure"] == pytest.approx(1.0)


def test_fees_reduce_performance():
    rng = np.random.default_rng(2)
    ret = rng.normal(0, 0.01, 500)
    p_up = np.where(ret > 0, 0.9, 0.1)  # oracle flips constantly -> high turnover
    gross = backtest_metrics(p_up, ret, fee_rate=0.0, bars_per_year=BARS_PER_YEAR)
    net = backtest_metrics(p_up, ret, fee_rate=0.001, bars_per_year=BARS_PER_YEAR)
    assert net["sharpe_net"] < gross["sharpe_net"]
    assert net["sharpe_gross"] == pytest.approx(gross["sharpe_gross"])  # fees only hit net


def test_constant_position_pays_fees_once():
    ret = np.array([0.01, 0.02, -0.01, 0.03])
    p_up = np.full(4, 0.9)  # always long
    metrics = backtest_metrics(p_up, ret, fee_rate=0.001, bars_per_year=BARS_PER_YEAR)
    assert metrics["turnover"] == pytest.approx(1 / 4)  # single entry, no flips
    expected_net_mean = ret.mean() - 0.001 / 4
    assert metrics["ann_return_net"] == pytest.approx(expected_net_mean * BARS_PER_YEAR)


def test_metrics_consistent_with_strategy_returns():
    rng = np.random.default_rng(3)
    ret = rng.normal(0, 0.01, 300)
    p_up = rng.random(300)
    gross, net, positions = strategy_returns(p_up, ret, fee_rate=0.001)
    metrics = backtest_metrics(p_up, ret, fee_rate=0.001, bars_per_year=BARS_PER_YEAR)
    assert metrics["sharpe_net"] == pytest.approx(sharpe_ratio(net, BARS_PER_YEAR))
    assert metrics["max_drawdown_net"] == pytest.approx(max_drawdown(net))
    assert metrics["ann_return_net"] == pytest.approx(net.mean() * BARS_PER_YEAR)
    assert metrics["ann_volatility_net"] == pytest.approx(net.std(ddof=1) * np.sqrt(BARS_PER_YEAR))
    assert metrics["exposure"] == pytest.approx(np.abs(positions).mean())


def test_n_trades_counts_position_changes():
    # long, long, short, flat-ish? -> use explicit probabilities
    p_up = np.array([0.9, 0.9, 0.1, 0.9])  # +1, +1, -1, +1
    ret = np.zeros(4)
    metrics = backtest_metrics(p_up, ret, fee_rate=0.0, bars_per_year=BARS_PER_YEAR)
    # Entry (1) + flip to short (2) + flip back to long (2) = 5 units of change.
    assert metrics["n_trades"] == pytest.approx(5.0)


def test_neutral_band_keeps_out_of_market():
    p_up = np.full(10, 0.5)
    metrics = backtest_metrics(p_up, np.full(10, 0.01), fee_rate=0.001, bars_per_year=BARS_PER_YEAR)
    assert metrics["exposure"] == 0.0
    assert np.isnan(metrics["hit_rate"])
    assert metrics["ann_return_net"] == 0.0
