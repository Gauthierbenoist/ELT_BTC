"""Purged walk-forward cross-validation for time-ordered samples.

Rows must be in strict chronological order (as produced by
:func:`elt_btc.ml.dataset.build_dataset`). Each fold trains on an expanding
window of the past and tests on the next contiguous block; ``purge`` bars
immediately before each test block are excluded from training so that no
training label (which looks ``horizon`` bars ahead) overlaps the test window.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedWalkForwardSplit:
    """sklearn-style splitter: chronological expanding-train walk-forward.

    Attributes:
        n_splits: Number of consecutive test folds (covering the tail of the data).
        test_size: Length of each test fold, in rows (bars).
        min_train_size: Minimum rows required before the first test fold.
        purge: Rows dropped from the end of each training window; must be at
            least the label horizon (1 bar here) to prevent label overlap.
    """

    n_splits: int
    test_size: int
    min_train_size: int
    purge: int

    def split(self, X: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_indices, test_indices)`` per fold, oldest fold first.

        Raises:
            ValueError: If the data is too short for the requested layout.
        """
        n_samples = len(X)
        first_test_start = n_samples - self.n_splits * self.test_size
        if first_test_start - self.purge < self.min_train_size:
            raise ValueError(
                f"Not enough samples ({n_samples}) for {self.n_splits} folds of "
                f"{self.test_size} with min_train_size={self.min_train_size} "
                f"and purge={self.purge}"
            )
        for fold in range(self.n_splits):
            test_start = first_test_start + fold * self.test_size
            train_indices = np.arange(0, test_start - self.purge)
            test_indices = np.arange(test_start, test_start + self.test_size)
            yield train_indices, test_indices
