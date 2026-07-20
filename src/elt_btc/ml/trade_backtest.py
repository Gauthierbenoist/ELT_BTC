"""Sequential trade-level backtest for barrier-style predictions.

Unlike :mod:`elt_btc.ml.backtest` (per-bar positions), this simulates an
executable policy for labels that span several bars (triple-barrier): one
trade at a time, entered on a confident signal, held until its exit
(``holding_bars`` later); signals arriving while a trade is open are
ignored. Shorts mirror the long trade's return (barriers are symmetric).
Fees are charged per round trip (entry + exit). Still no slippage or
sizing — an evaluation tool, not an execution engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_MS_PER_YEAR = 365 * 86_400 * 1000

TRADE_COLUMNS = ["entry_ts", "direction", "holding_bars", "ret_gross", "ret_net"]


@dataclass(frozen=True)
class TradeBacktestResult:
    """Simulated trades and their aggregate metrics."""

    trades: pd.DataFrame
    metrics: dict[str, float]


def simulate_trades(
    timestamps: np.ndarray,
    p_up: np.ndarray,
    ret_trade: np.ndarray,
    holding_bars: np.ndarray,
    *,
    bar_ms: int,
    fee_rate: float,
    threshold_band: float = 0.0,
) -> TradeBacktestResult:
    """Run the sequential simulation over chronologically sorted samples.

    Args:
        timestamps: Bar open times (epoch ms), strictly increasing.
        ret_trade: Return of the *long* virtual trade opened at each bar
            (entry at close, exit at the touched barrier).
        holding_bars: Bars until that trade's exit.
        fee_rate: One-way fee; a round trip costs ``2 * fee_rate``.
    """
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")

    records: list[tuple[int, int, int, float, float]] = []
    busy_until = -np.inf
    for i in range(len(timestamps)):
        ts = int(timestamps[i])
        if ts < busy_until:
            continue  # a trade is still open: signal ignored
        if p_up[i] > 0.5 + threshold_band:
            direction = 1
        elif p_up[i] < 0.5 - threshold_band:
            direction = -1
        else:
            continue
        gross = direction * float(ret_trade[i])
        holding = int(holding_bars[i])
        records.append((ts, direction, holding, gross, gross - 2.0 * fee_rate))
        busy_until = ts + holding * bar_ms

    trades = pd.DataFrame(records, columns=TRADE_COLUMNS)
    elapsed_ms = int(timestamps[-1]) - int(timestamps[0]) + bar_ms if len(timestamps) else 0
    return TradeBacktestResult(
        trades=trades, metrics=_trade_metrics(trades, elapsed_ms=elapsed_ms, bar_ms=bar_ms)
    )


def _trade_metrics(trades: pd.DataFrame, *, elapsed_ms: int, bar_ms: int) -> dict[str, float]:
    n_trades = len(trades)
    if n_trades == 0 or elapsed_ms <= 0:
        return {
            "n_trades": 0.0,
            "win_rate": float("nan"),
            "avg_ret_net": float("nan"),
            "ann_return_net": 0.0,
            "sharpe_net": float("nan"),
            "max_drawdown_net": 0.0,
            "exposure": 0.0,
            "avg_holding_bars": float("nan"),
            "trades_per_year": 0.0,
        }
    net = trades["ret_net"].to_numpy()
    years = elapsed_ms / _MS_PER_YEAR
    trades_per_year = n_trades / years
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    std = float(net.std(ddof=1)) if n_trades > 1 else float("nan")
    sharpe = (
        float(net.mean()) / std * float(np.sqrt(trades_per_year))
        if std and np.isfinite(std) and std > 0
        else float("nan")
    )
    return {
        "n_trades": float(n_trades),
        "win_rate": float((net > 0).mean()),
        "avg_ret_net": float(net.mean()),
        "ann_return_net": float(equity[-1] ** (1.0 / years) - 1.0),
        "sharpe_net": sharpe,
        "max_drawdown_net": float((equity / peak - 1.0).min()),
        "exposure": float(trades["holding_bars"].sum() * bar_ms / elapsed_ms),
        "avg_holding_bars": float(trades["holding_bars"].mean()),
        "trades_per_year": float(trades_per_year),
    }
