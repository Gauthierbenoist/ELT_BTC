"""Hand-computed cases for the trailing-barrier (v2) trade simulation."""

from __future__ import annotations

import numpy as np
import pytest

from elt_btc.ml.trade_backtest import simulate_trades_trailing

BAR_MS = 14_400_000


def ts(n: int) -> np.ndarray:
    return np.arange(n, dtype="int64") * BAR_MS


def run(closes, highs, lows, p_up, side, sigma=0.01, **kwargs):
    n = len(closes)
    defaults = {
        "bar_ms": BAR_MS,
        "fee_rate": 0.0,
        "pt_mult": 2.0,
        "sl_mult": 1.0,
        "max_holding": 10,
        "threshold_band": 0.0,
    }
    return simulate_trades_trailing(
        ts(n),
        np.asarray(p_up, dtype=float),
        np.asarray(side, dtype=float),
        np.asarray(highs, dtype=float),
        np.asarray(lows, dtype=float),
        np.asarray(closes, dtype=float),
        np.full(n, sigma),
        **{**defaults, **kwargs},
    )


def flat(closes):
    return closes, closes, closes  # high = low = close (fully controlled path)


def test_no_resignal_behaves_like_static_barriers():
    # Entry at 100 (TP 102, SL 99); no re-signal (p<0.5 after entry);
    # bar 2 low touches 99 -> static stop loss.
    closes = [100.0, 100.5, 99.0, 100.0, 100.0]
    p = [0.9, 0.2, 0.2, 0.2, 0.2]
    result = run(closes, closes, closes, p, np.ones(5))
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "sl"
    assert trade["n_updates"] == 0
    assert trade["ret_gross"] == pytest.approx(-0.01)


def test_trailing_stop_locks_in_profit():
    # Long at 100 (TP 102, SL 99). Bar 1 closes 101.5 with re-signal:
    # candidate TP 103.53, SL 100.485 -> both ratchet up. Bar 2 drops to
    # 100.4: new SL 100.485 hit -> exit ABOVE entry (profit locked).
    closes = [100.0, 101.5, 100.4, 100.0, 100.0]
    p = [0.9, 0.9, 0.9, 0.2, 0.2]
    result = run(closes, closes, closes, p, np.ones(5))
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "sl"
    assert trade["n_updates"] == 1
    assert trade["exit_price"] == pytest.approx(101.5 * 0.99)  # 100.485
    assert trade["ret_gross"] == pytest.approx(101.5 * 0.99 / 100.0 - 1.0)
    assert trade["ret_gross"] > 0  # the point of the trailing stop


def test_barriers_never_ratchet_unfavorably():
    # Bar 1 re-signal from a LOWER close: candidates (TP 100.98, SL 98.01)
    # are both below the current barriers (102, 99) -> nothing moves.
    closes = [100.0, 99.2, 98.9, 100.0, 100.0]
    p = [0.9, 0.9, 0.9, 0.2, 0.2]
    result = run(closes, closes, closes, p, np.ones(5))
    trade = result.trades.iloc[0]
    # Original SL 99 hit at bar 2 (low 98.9): candidates never lowered it.
    assert trade["exit_reason"] == "sl"
    assert trade["exit_price"] == pytest.approx(99.0)
    assert trade["n_updates"] == 1  # update evaluated, but no barrier moved


def test_take_profit_ratchets_independently_of_stop():
    # sigma shrinks after entry: candidate TP can rise while candidate SL
    # also rises; with a big drop candidate SL below current stays put.
    # Here: entry 100, sigma 0.01 (TP 102 / SL 99). Bar 1 close 101,
    # sigma still 0.01 -> cand TP 103.02 (up), cand SL 99.99 (up).
    # Bar 2 high 103.1 -> trailing TP 103.02 hit.
    closes = [100.0, 101.0, 102.5, 100.0, 100.0]
    highs = [100.0, 101.0, 103.1, 100.0, 100.0]
    p = [0.9, 0.9, 0.9, 0.2, 0.2]
    result = run(closes, highs, closes, p, np.ones(5))
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "tp"
    assert trade["exit_price"] == pytest.approx(101.0 * 1.02)  # 103.02, ratcheted


def test_short_barriers_ratchet_down():
    # Short at 100 (TP 98, SL 101). Bar 1 closes 99 with short re-signal:
    # cand TP 97.02 (down: ratchet), cand SL 99.99 (down: ratchet).
    # Bar 2 rises to 100.2 -> new SL 99.99 hit: profit locked on a short.
    closes = [100.0, 99.0, 100.2, 100.0, 100.0]
    p = [0.9, 0.9, 0.9, 0.2, 0.2]
    result = run(closes, closes, closes, p, -np.ones(5))
    trade = result.trades.iloc[0]
    assert trade["direction"] == -1
    assert trade["exit_reason"] == "sl"
    assert trade["exit_price"] == pytest.approx(99.0 * 1.01)  # 99.99
    assert trade["ret_gross"] == pytest.approx(-(99.0 * 1.01 / 100.0 - 1.0))
    assert trade["ret_gross"] > 0


def test_vertical_barrier_still_anchored_at_entry():
    closes = np.full(8, 100.0)
    p = [0.9] + [0.2] * 7
    result = run(list(closes), list(closes), list(closes), p, np.ones(8), max_holding=4)
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "vertical"
    assert trade["exit_ts"] == 4 * BAR_MS


def test_unresolved_trade_is_discarded():
    closes = [100.0, 100.1, 100.2]
    p = [0.9, 0.2, 0.2]
    result = run(closes, closes, closes, p, np.ones(3), max_holding=10)
    assert len(result.trades) == 0  # never exited before the data ends


def test_fees_charged_per_round_trip():
    closes = [100.0, 103.0, 100.0, 100.0, 100.0]
    p = [0.9, 0.2, 0.2, 0.2, 0.2]
    result = run(closes, closes, closes, p, np.ones(5), fee_rate=0.001)
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "tp"
    assert trade["ret_net"] == pytest.approx(trade["ret_gross"] - 0.002)
