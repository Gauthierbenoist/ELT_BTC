"""Fold-level and aggregate metrics for probability-of-up-move models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def evaluate_fold(y_true: pd.Series, p_up: np.ndarray) -> dict[str, float]:
    """Score one test fold. ``p_up`` is the predicted probability of class 1.

    ``auc`` is NaN when the fold contains a single class (degenerate fold).
    ``acc_lift_vs_prior`` is accuracy minus the best constant-guess accuracy
    on that fold — a model with no skill sits at 0.
    """
    y_arr = y_true.to_numpy()
    realized_rate = float(y_arr.mean())
    auc = float(roc_auc_score(y_arr, p_up)) if 0.0 < realized_rate < 1.0 else float("nan")
    accuracy = float(((p_up >= 0.5) == (y_arr == 1)).mean())
    return {
        "auc": auc,
        "log_loss": float(log_loss(y_arr, p_up, labels=[0, 1])),
        "brier": float(brier_score_loss(y_arr, p_up)),
        "accuracy": accuracy,
        "acc_lift_vs_prior": accuracy - max(realized_rate, 1.0 - realized_rate),
        "mean_p_up": float(np.mean(p_up)),
        "realized_up_rate": realized_rate,
    }


def calibration_table(
    y_true: pd.Series, p_up: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    """Realized up-rate per predicted-probability bin (reliability diagram data)."""
    y_arr = y_true.to_numpy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p_up, edges) - 1, 0, n_bins - 1)
    table: list[dict[str, float]] = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        table.append(
            {
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "mean_p_up": float(p_up[mask].mean()),
                "realized_up_rate": float(y_arr[mask].mean()),
                "count": float(mask.sum()),
            }
        )
    return table


def aggregate_folds(folds: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and std of each metric across folds (NaN-aware for ``auc``)."""
    aggregated: dict[str, dict[str, float]] = {}
    for metric in folds[0]:
        values = np.array([fold[metric] for fold in folds])
        aggregated[metric] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        }
    return aggregated
