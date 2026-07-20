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

TRADE_COLUMNS = [
    "entry_ts",
    "exit_ts",
    "direction",
    "holding_bars",
    "p_up",
    "ret_gross",
    "ret_net",
]


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
    side: np.ndarray | None = None,
) -> TradeBacktestResult:
    """Run the sequential simulation over chronologically sorted samples.

    Without ``side``, ``p_up`` is a directional probability: long above
    ``0.5 + band``, short below ``0.5 - band``, and ``ret_trade`` is the
    *long* trade's return (shorts take its negative — symmetric barriers).

    With ``side`` (meta-labeling), ``p_up`` is the probability that the
    *sided* trade wins and ``ret_trade`` is already side-adjusted: enter in
    the primary signal's direction when ``p_up > 0.5 + band``, never fade
    the signal.

    Args:
        timestamps: Bar open times (epoch ms), strictly increasing.
        holding_bars: Bars until each trade's exit.
        fee_rate: One-way fee; a round trip costs ``2 * fee_rate``.
    """
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")

    records: list[tuple[int, int, int, int, float, float, float]] = []
    busy_until = -np.inf
    for i in range(len(timestamps)):
        ts = int(timestamps[i])
        if ts < busy_until:
            continue  # a trade is still open: signal ignored
        if side is not None:
            if p_up[i] <= 0.5 + threshold_band or side[i] == 0:
                continue
            direction = int(side[i])
            gross = float(ret_trade[i])  # already side-adjusted
        elif p_up[i] > 0.5 + threshold_band:
            direction = 1
            gross = float(ret_trade[i])
        elif p_up[i] < 0.5 - threshold_band:
            direction = -1
            gross = -float(ret_trade[i])
        else:
            continue
        holding = int(holding_bars[i])
        exit_ts = ts + holding * bar_ms
        records.append(
            (ts, exit_ts, direction, holding, float(p_up[i]), gross, gross - 2.0 * fee_rate)
        )
        busy_until = exit_ts

    trades = pd.DataFrame(records, columns=TRADE_COLUMNS)
    elapsed_ms = int(timestamps[-1]) - int(timestamps[0]) + bar_ms if len(timestamps) else 0
    return TradeBacktestResult(
        trades=trades, metrics=_trade_metrics(trades, elapsed_ms=elapsed_ms, bar_ms=bar_ms)
    )


TRAILING_TRADE_COLUMNS = [
    "entry_ts",
    "exit_ts",
    "direction",
    "holding_bars",
    "p_up",
    "entry_price",
    "exit_price",
    "exit_reason",
    "n_updates",
    "ret_gross",
    "ret_net",
]


def simulate_trades_trailing(
    timestamps: np.ndarray,
    p_up: np.ndarray,
    side: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    sigma: np.ndarray,
    *,
    bar_ms: int,
    fee_rate: float,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
    threshold_band: float = 0.0,
) -> TradeBacktestResult:
    """Sequential meta policy **v2**: barriers ratchet on model re-signals.

    Entries follow the v1 meta rule (primary side + ``p_up`` above the
    band; the signal is never faded). While a trade is open, every bar
    where the model re-signals in the trade's direction recomputes
    candidate barriers from the current close and volatility, and each
    barrier ratchets **independently in the trade's favorable direction**:
    for a long, TP and SL only ever move up (the SL becomes a trailing
    stop that can lock in profit); for a short, only ever down. The
    vertical barrier stays anchored at entry + ``max_holding`` bars.

    Exit priority inside a bar: stop-loss, then take-profit (conservative,
    same convention as the labeling), then the vertical barrier at the
    bar close. A trade still open at the end of the data is discarded
    (unresolved). Fees: ``2 * fee_rate`` per round trip.
    """
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")

    records: list[tuple[int, int, int, int, float, float, float, str, int, float, float]] = []
    in_trade = False
    direction = 0
    entry_ts = n_updates = 0
    entry_p = entry_price = tp = sl = 0.0
    deadline_ts = 0

    for i in range(len(timestamps)):
        ts = int(timestamps[i])
        if in_trade:
            exit_price, reason = 0.0, ""
            if direction > 0:
                if low[i] <= sl:
                    exit_price, reason = sl, "sl"
                elif high[i] >= tp:
                    exit_price, reason = tp, "tp"
                elif ts >= deadline_ts:
                    exit_price, reason = float(close[i]), "vertical"
            else:
                if high[i] >= sl:
                    exit_price, reason = sl, "sl"
                elif low[i] <= tp:
                    exit_price, reason = tp, "tp"
                elif ts >= deadline_ts:
                    exit_price, reason = float(close[i]), "vertical"
            if reason:
                gross = direction * (exit_price / entry_price - 1.0)
                records.append(
                    (
                        entry_ts,
                        ts,
                        direction,
                        max(1, (ts - entry_ts) // bar_ms),
                        entry_p,
                        entry_price,
                        exit_price,
                        reason,
                        n_updates,
                        gross,
                        gross - 2.0 * fee_rate,
                    )
                )
                in_trade = False
                # fall through: a new entry on this same bar's close is allowed
            elif (
                side[i] == direction
                and p_up[i] > 0.5 + threshold_band
                and np.isfinite(sigma[i])
                and sigma[i] > 0
            ):
                # Re-signal: ratchet each barrier independently, favorably.
                candidate_tp = float(close[i]) * (1.0 + direction * pt_mult * float(sigma[i]))
                candidate_sl = float(close[i]) * (1.0 - direction * sl_mult * float(sigma[i]))
                if direction > 0:
                    tp, sl = max(tp, candidate_tp), max(sl, candidate_sl)
                else:
                    tp, sl = min(tp, candidate_tp), min(sl, candidate_sl)
                n_updates += 1
                continue
            else:
                continue
        if (
            not in_trade
            and side[i] != 0
            and p_up[i] > 0.5 + threshold_band
            and np.isfinite(sigma[i])
            and sigma[i] > 0
        ):
            in_trade = True
            direction = int(side[i])
            entry_ts, n_updates = ts, 0
            entry_p = float(p_up[i])
            entry_price = float(close[i])
            tp = entry_price * (1.0 + direction * pt_mult * float(sigma[i]))
            sl = entry_price * (1.0 - direction * sl_mult * float(sigma[i]))
            deadline_ts = ts + max_holding * bar_ms

    trades = pd.DataFrame(records, columns=TRAILING_TRADE_COLUMNS)
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
