"""Meta-labeling: sided barriers, primary signal, meta policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elt_btc.ml.backtest import meta_effective_proba
from elt_btc.ml.labels import momentum_side, triple_barrier_labels
from elt_btc.ml.trade_backtest import simulate_trades

BAR_MS = 14_400_000


def bars_from_closes(closes, spread: float = 0.0) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": np.arange(len(closes), dtype="int64") * BAR_MS,
            "open": closes,
            "high": closes + spread,
            "low": closes - spread,
            "close": closes,
            "volume": np.ones(len(closes)),
        }
    )


def constant_vol(n: int, value: float = 0.01) -> pd.Series:
    return pd.Series(np.full(n, value))


# vol=1%, pt=3, sl=1: long TP at +3%, SL at -1%; short TP at -3%, SL at +1%.


def test_short_side_wins_on_drop():
    closes = [100.0, 98.0, 96.5, 97.0, 97.0, 97.0, 97.0]
    bars = bars_from_closes(closes)
    side = pd.Series(np.full(7, -1.0))
    tb = triple_barrier_labels(
        bars,
        vol_span=2,
        pt_mult=3.0,
        sl_mult=1.0,
        max_holding=4,
        volatility=constant_vol(7),
        side=side,
    )
    # Short TP at 97.0 touched at t+2 (low 96.5): the sided trade WINS +3%.
    assert tb.loc[0, "label"] == 1.0
    assert tb.loc[0, "ret_trade"] == pytest.approx(0.03)
    assert tb.loc[0, "holding_bars"] == 2


def test_short_side_stopped_on_rise():
    closes = [100.0, 101.5, 100.0, 100.0, 100.0, 100.0, 100.0]
    bars = bars_from_closes(closes)
    side = pd.Series(np.full(7, -1.0))
    tb = triple_barrier_labels(
        bars,
        vol_span=2,
        pt_mult=3.0,
        sl_mult=1.0,
        max_holding=4,
        volatility=constant_vol(7),
        side=side,
    )
    # Short SL at 101.0 touched at t+1: the sided trade LOSES -1%.
    assert tb.loc[0, "label"] == 0.0
    assert tb.loc[0, "ret_trade"] == pytest.approx(-0.01)


def test_short_double_touch_resolves_to_stop():
    bars = bars_from_closes(np.full(6, 100.0), spread=4.0)  # spans both barriers
    side = pd.Series(np.full(6, -1.0))
    tb = triple_barrier_labels(
        bars,
        vol_span=2,
        pt_mult=3.0,
        sl_mult=1.0,
        max_holding=3,
        volatility=constant_vol(6),
        side=side,
    )
    assert tb.loc[0, "label"] == 0.0
    assert tb.loc[0, "ret_trade"] == pytest.approx(-0.01)


def test_side_plus_one_equals_unsided():
    rng = np.random.default_rng(9)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 120)))
    bars = bars_from_closes(closes, spread=0.5)
    kwargs = {
        "vol_span": 2,
        "pt_mult": 2.0,
        "sl_mult": 1.0,
        "max_holding": 6,
        "volatility": constant_vol(120),
    }
    unsided = triple_barrier_labels(bars, **kwargs)
    sided = triple_barrier_labels(bars, side=pd.Series(np.ones(120)), **kwargs)
    pd.testing.assert_frame_equal(unsided, sided)


def test_zero_or_nan_side_gets_no_label():
    bars = bars_from_closes(np.full(10, 100.0), spread=2.0)
    side = pd.Series([1.0, 0.0, np.nan, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    tb = triple_barrier_labels(
        bars,
        vol_span=2,
        pt_mult=1.0,
        sl_mult=1.0,
        max_holding=3,
        volatility=constant_vol(10),
        side=side,
    )
    assert np.isnan(tb.loc[1, "label"])
    assert np.isnan(tb.loc[2, "label"])
    assert not np.isnan(tb.loc[0, "label"])


def test_momentum_side_signs_and_causality():
    close = pd.Series([100.0, 101.0, 102.0, 101.0, 100.0, 99.0])
    side = momentum_side(close, window=2)
    assert np.isnan(side.iloc[0]) and np.isnan(side.iloc[1])  # warm-up
    assert side.iloc[2] == 1.0  # 102 > 100
    assert side.iloc[4] == -1.0  # 100 < 102
    truncated = momentum_side(close.iloc[:5], window=2)
    assert truncated.iloc[4] == side.iloc[4]  # prefix invariance


def test_meta_effective_proba_never_fades_the_signal():
    p_win = np.array([0.8, 0.5, 0.2, 0.8])
    side = np.array([1.0, 1.0, 1.0, -1.0])
    p_dir = meta_effective_proba(p_win, side)
    assert p_dir[0] == pytest.approx(0.8)  # confident long
    assert p_dir[1] == pytest.approx(0.5)  # no conviction -> flat
    assert p_dir[2] == pytest.approx(0.5)  # low p_win -> flat, NOT short
    assert p_dir[3] == pytest.approx(0.2)  # confident short (side -1)


def test_simulate_trades_meta_mode():
    ts = np.arange(4, dtype="int64") * BAR_MS
    p_win = np.array([0.9, 0.2, 0.9, 0.9])
    ret_trade = np.array([0.03, 0.03, -0.01, 0.02])  # already side-adjusted
    side = np.array([-1, -1, 1, 1])
    result = simulate_trades(
        ts,
        p_win,
        ret_trade,
        np.ones(4, dtype=int),
        bar_ms=BAR_MS,
        fee_rate=0.0,
        side=side,
    )
    # Row 1 skipped (p_win < 0.5: never fade); rows 0, 2, 3 traded.
    assert len(result.trades) == 3
    assert list(result.trades["direction"]) == [-1, 1, 1]
    # Gross returns taken as-is (side-adjusted), no sign flip.
    assert list(result.trades["ret_gross"]) == [0.03, -0.01, 0.02]
