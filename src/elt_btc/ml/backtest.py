"""Vectorized backtest of probability forecasts — evaluation only.

Converts P(up) forecasts into positions in {-1, 0, +1} and scores the
resulting PnL on next-bar returns. This is an *analysis* layer, not an
execution model: fees are a flat one-way rate per unit of position change,
there is no slippage, no sizing, no funding. Good enough to compare models
and to check whether an edge survives costs — nothing more.
"""

from __future__ import annotations

import math

import numpy as np


def meta_effective_proba(p_win: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Map meta-label win probabilities onto directional ``p_up`` space.

    Meta-labeling never fades its primary signal: a confident win pushes
    the effective probability toward the signal's side; ``p_win <= 0.5``
    maps to exactly 0.5 (stay flat). Feeding the result to
    :func:`positions_from_proba` reproduces the meta policy.
    """
    confidence = np.maximum(p_win - 0.5, 0.0)
    return 0.5 + side * confidence


def positions_from_proba(p_up: np.ndarray, threshold_band: float = 0.0) -> np.ndarray:
    """Map probabilities to positions: +1 above ``0.5 + band``, -1 below
    ``0.5 - band``, 0 inside the neutral band."""
    positions = np.zeros(len(p_up))
    positions[p_up > 0.5 + threshold_band] = 1.0
    positions[p_up < 0.5 - threshold_band] = -1.0
    return positions


def sharpe_ratio(returns: np.ndarray, bars_per_year: float) -> float:
    """Annualized Sharpe (no risk-free rate); NaN when undefined."""
    if len(returns) < 2:
        return float("nan")
    std = float(returns.std(ddof=1))
    if std == 0.0:
        return float("nan")
    return float(returns.mean()) / std * math.sqrt(bars_per_year)


def max_drawdown(returns: np.ndarray) -> float:
    """Max peak-to-trough drawdown of the compounded equity curve (<= 0)."""
    if len(returns) == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def strategy_returns(
    p_up: np.ndarray,
    next_returns: np.ndarray,
    *,
    fee_rate: float,
    threshold_band: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bar ``(gross, net, positions)`` of trading the forecasts.

    ``next_returns[t]`` must be the simple return of the bar the label of
    sample ``t`` refers to (``close_{t+1}/close_t - 1``) — an outcome
    column, never a feature. Fees are charged on every unit of position
    change, including the initial entry. Shared by :func:`backtest_metrics`
    and the dashboard so every consumer sees identical numbers.
    """
    positions = positions_from_proba(p_up, threshold_band)
    gross = positions * next_returns
    position_changes = np.abs(np.diff(positions, prepend=0.0))
    net = gross - fee_rate * position_changes
    return gross, net, positions


def backtest_metrics(
    p_up: np.ndarray,
    next_returns: np.ndarray,
    *,
    fee_rate: float,
    bars_per_year: float,
    threshold_band: float = 0.0,
) -> dict[str, float]:
    """Financial metrics of trading the forecasts on next-bar returns."""
    gross, net, positions = strategy_returns(
        p_up, next_returns, fee_rate=fee_rate, threshold_band=threshold_band
    )
    position_changes = np.abs(np.diff(positions, prepend=0.0))
    in_market = positions != 0
    hit_rate = float((gross[in_market] > 0).mean()) if in_market.any() else float("nan")
    ann_volatility = (
        float(net.std(ddof=1) * math.sqrt(bars_per_year)) if len(net) > 1 else float("nan")
    )
    return {
        "sharpe_gross": sharpe_ratio(gross, bars_per_year),
        "sharpe_net": sharpe_ratio(net, bars_per_year),
        "ann_return_net": float(net.mean() * bars_per_year),
        "ann_volatility_net": ann_volatility,
        "hit_rate": hit_rate,
        "max_drawdown_net": max_drawdown(net),
        "n_trades": float(position_changes.sum()),
        "turnover": float(position_changes.mean()),
        "exposure": float(np.abs(positions).mean()),
    }
