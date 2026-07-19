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
