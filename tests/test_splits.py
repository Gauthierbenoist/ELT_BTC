"""Property tests for the purged walk-forward splitter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elt_btc.ml.splits import PurgedWalkForwardSplit


def frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({"x": np.arange(n)})


def test_fold_layout_and_purge():
    splitter = PurgedWalkForwardSplit(n_splits=4, test_size=100, min_train_size=400, purge=10)
    folds = list(splitter.split(frame(1000)))
    assert len(folds) == 4
    expected_starts = [600, 700, 800, 900]
    for (train_idx, test_idx), start in zip(folds, expected_starts, strict=True):
        assert test_idx[0] == start
        assert len(test_idx) == 100
        # Purge: last training row sits `purge` rows before the test window.
        assert train_idx[-1] == start - 10 - 1
        assert train_idx[0] == 0
        # Strict separation: no training label can overlap the test window.
        assert train_idx.max() < test_idx.min() - 10 + 1


def test_test_folds_are_contiguous_and_disjoint():
    splitter = PurgedWalkForwardSplit(n_splits=3, test_size=50, min_train_size=100, purge=5)
    folds = list(splitter.split(frame(400)))
    all_test = np.concatenate([test for _, test in folds])
    assert len(all_test) == len(set(all_test))  # disjoint
    assert list(all_test) == list(range(250, 400))  # contiguous tail coverage


def test_train_is_expanding():
    splitter = PurgedWalkForwardSplit(n_splits=3, test_size=50, min_train_size=100, purge=5)
    train_sizes = [len(train) for train, _ in splitter.split(frame(400))]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[0] < train_sizes[-1]


def test_raises_when_history_too_short():
    splitter = PurgedWalkForwardSplit(n_splits=4, test_size=100, min_train_size=700, purge=10)
    with pytest.raises(ValueError, match="Not enough samples"):
        list(splitter.split(frame(1000)))
