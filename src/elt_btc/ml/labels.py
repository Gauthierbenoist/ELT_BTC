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


def triple_barrier_labels(
    bars: pd.DataFrame,
    *,
    vol_span: int,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
    volatility: pd.Series | None = None,
) -> pd.DataFrame:
    """Label every bar with its first-touched barrier.

    Returns a frame indexed like ``bars`` with columns ``label`` (1/0),
    ``ret_trade`` (simple return of the virtual trade, exit at the barrier
    price or at the vertical-barrier close) and ``holding_bars``. Rows are
    NaN when the volatility is not yet defined or when the vertical barrier
    falls beyond the available data (labels there are unknowable).

    Args:
        volatility: Optional causal per-bar volatility overriding the EWMA
            estimate — used by tests to pin the barriers exactly.
    """
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    if volatility is None:
        volatility = ewma_volatility(bars["close"], vol_span)
    sigma = volatility.to_numpy(dtype=float)

    n = len(bars)
    label = np.full(n, np.nan)
    ret_trade = np.full(n, np.nan)
    holding = np.full(n, np.nan)
    upper = close * (1.0 + pt_mult * sigma)
    lower = close * (1.0 - sl_mult * sigma)

    for t in range(n):
        if not np.isfinite(sigma[t]) or sigma[t] <= 0 or t + max_holding >= n:
            continue
        for h in range(1, max_holding + 1):
            i = t + h
            if low[i] <= lower[t]:  # stop-loss first on same-bar ambiguity
                label[t] = 0.0
                ret_trade[t] = lower[t] / close[t] - 1.0
                holding[t] = h
                break
            if high[i] >= upper[t]:
                label[t] = 1.0
                ret_trade[t] = upper[t] / close[t] - 1.0
                holding[t] = h
                break
        else:  # vertical barrier: sign of the realized return
            final_return = close[t + max_holding] / close[t] - 1.0
            label[t] = 1.0 if final_return > 0 else 0.0
            ret_trade[t] = final_return
            holding[t] = max_holding

    return pd.DataFrame(
        {"label": label, "ret_trade": ret_trade, "holding_bars": holding}, index=bars.index
    )
