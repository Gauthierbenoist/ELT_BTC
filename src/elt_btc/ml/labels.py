"""Triple-barrier labeling (López de Prado, AFML ch. 3).

For each bar ``t`` a virtual trade is opened at ``close_t`` with three exits:

- upper barrier (profit-take) at ``close_t * (1 + pt_mult * sigma_t)``
- lower barrier (stop-loss)  at ``close_t * (1 - sl_mult * sigma_t)``
- vertical barrier ``max_holding`` bars later

``sigma_t`` is a causal EWMA volatility of one-bar log returns. The label is
1 if the upper barrier is touched first (bar highs/lows are used for touch
detection), 0 if the lower one is; on the vertical barrier it is the sign
of the realized return. When both barriers are touched inside the same bar
the order is unknowable, so the stop-loss wins (conservative convention).

CRITICAL leakage note: the label at ``t`` uses price data up to
``t + holding_bars <= t + max_holding``. Any purged split MUST therefore
use ``purge >= max_holding`` — enforced by
:class:`elt_btc.ml.config.BenchmarkSettings`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TB_COLUMNS = ["label", "ret_trade", "holding_bars"]


def ewma_volatility(close: pd.Series, span: int) -> pd.Series:
    """Causal EWMA standard deviation of one-bar log returns."""
    log_returns = pd.Series(np.log(close / close.shift(1)), index=close.index)
    return log_returns.ewm(span=span, min_periods=span).std()


def momentum_side(close: pd.Series, window: int) -> pd.Series:
    """Primary directional signal: sign of the trailing ``window``-bar return.

    Causal (uses closes up to ``t`` only). Returns +1/-1, 0 on an exactly
    flat window and NaN during warm-up — callers must treat 0/NaN rows as
    "no signal".
    """
    trailing = pd.Series(np.log(close / close.shift(window)), index=close.index)
    return pd.Series(np.sign(trailing), index=close.index)


def triple_barrier_labels(
    bars: pd.DataFrame,
    *,
    vol_span: int,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
    volatility: pd.Series | None = None,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """Label every bar with the outcome of its (possibly sided) virtual trade.

    Without ``side`` (or with side +1) the trade is long: profit-take at
    ``+pt_mult * sigma``, stop-loss at ``-sl_mult * sigma``. With side -1
    the barriers are set **in the trade's direction** (meta-labeling):
    profit-take ``pt_mult`` sigmas *below* the entry, stop-loss ``sl_mult``
    sigmas above — the asymmetry follows the trade, it is never inverted.

    Returns a frame indexed like ``bars`` with ``label`` (1 = the trade
    wins, i.e. its profit-take is touched first; vertical exits take the
    sign of the side-adjusted return), ``ret_trade`` (the *trade's* simple
    return: ``side * (exit / entry - 1)``, exit at the barrier price) and
    ``holding_bars``. Rows are NaN when the volatility or side is undefined
    or the vertical barrier falls beyond the data.

    Same-bar double touches resolve to the stop-loss (conservative).

    Args:
        volatility: Optional causal per-bar volatility overriding the EWMA
            estimate — used by tests to pin the barriers exactly.
        side: Optional +1/-1 primary signal per bar; 0/NaN rows get no label.
    """
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    if volatility is None:
        volatility = ewma_volatility(bars["close"], vol_span)
    sigma = volatility.to_numpy(dtype=float)
    sides = np.ones(len(bars)) if side is None else side.to_numpy(dtype=float)

    n = len(bars)
    label = np.full(n, np.nan)
    ret_trade = np.full(n, np.nan)
    holding = np.full(n, np.nan)
    # Barrier distances follow the trade direction: the profit-take always
    # sits pt_mult sigmas along the side, the stop sl_mult sigmas against it.
    up_mult = np.where(sides < 0, sl_mult, pt_mult)
    down_mult = np.where(sides < 0, pt_mult, sl_mult)
    barrier_up = close * (1.0 + up_mult * sigma)
    barrier_down = close * (1.0 - down_mult * sigma)

    for t in range(n):
        s = sides[t]
        if (
            not np.isfinite(sigma[t])
            or sigma[t] <= 0
            or not np.isfinite(s)
            or s == 0
            or t + max_holding >= n
        ):
            continue
        for h in range(1, max_holding + 1):
            i = t + h
            hit_up = high[i] >= barrier_up[t]
            hit_down = low[i] <= barrier_down[t]
            if not (hit_up or hit_down):
                continue
            # Stop-loss checked first on same-bar ambiguity (conservative):
            # for a long the stop is the lower barrier, for a short the upper.
            if s > 0:
                win, exit_price = (False, barrier_down[t]) if hit_down else (True, barrier_up[t])
            else:
                win, exit_price = (False, barrier_up[t]) if hit_up else (True, barrier_down[t])
            label[t] = 1.0 if win else 0.0
            ret_trade[t] = s * (exit_price / close[t] - 1.0)
            holding[t] = h
            break
        else:  # vertical barrier: sign of the side-adjusted realized return
            final_return = s * (close[t + max_holding] / close[t] - 1.0)
            label[t] = 1.0 if final_return > 0 else 0.0
            ret_trade[t] = final_return
            holding[t] = max_holding

    return pd.DataFrame(
        {"label": label, "ret_trade": ret_trade, "holding_bars": holding}, index=bars.index
    )
