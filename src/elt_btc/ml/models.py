"""Benchmark model zoo: naive baselines, logistic regression, LightGBM.

All models expose the sklearn ``fit(X, y)`` / ``predict_proba(X)`` protocol
with ``X`` a feature DataFrame. Hyperparameters are fixed on purpose: this is
a benchmark floor, not a tuning exercise — any tuning would itself need
nested, purged validation to stay honest.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ProbClassifier(Protocol):
    """Minimal protocol shared by every benchmark model."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


class PriorClassifier:
    """Predicts the training up-rate for every sample — the absolute floor."""

    def __init__(self) -> None:
        self.p_up_: float = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PriorClassifier:
        self.p_up_ = float(y.mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.full(len(X), self.p_up_)
        return np.column_stack([1.0 - p, p])


class MomentumSignClassifier:
    """Conditions the up-rate on the sign of the last one-bar return.

    Uses the ``ret_1`` feature column; probabilities are the training
    frequencies of an up move given a positive vs non-positive last return.
    """

    def __init__(self, feature: str = "ret_1") -> None:
        self.feature = feature
        self.p_up_given_pos_: float = 0.5
        self.p_up_given_nonpos_: float = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MomentumSignClassifier:
        positive = X[self.feature].to_numpy() > 0
        y_arr = y.to_numpy()
        if positive.any():
            self.p_up_given_pos_ = float(y_arr[positive].mean())
        if (~positive).any():
            self.p_up_given_nonpos_ = float(y_arr[~positive].mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        positive = X[self.feature].to_numpy() > 0
        p = np.where(positive, self.p_up_given_pos_, self.p_up_given_nonpos_)
        return np.column_stack([1.0 - p, p])


def build_models(seed: int) -> dict[str, ProbClassifier]:
    """The benchmark zoo, keyed by report name."""
    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=seed)),
        ]
    )
    lgbm = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )
    return {
        "prior": PriorClassifier(),
        "momentum_sign": MomentumSignClassifier(),
        "logreg": logreg,
        "lightgbm": lgbm,
    }
