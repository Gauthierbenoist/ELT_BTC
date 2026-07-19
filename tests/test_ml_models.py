"""Baseline behavior and smoke tests for the benchmark model zoo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elt_btc.ml.models import MomentumSignClassifier, PriorClassifier, build_models


def test_prior_classifier_predicts_train_up_rate():
    X = pd.DataFrame({"ret_1": np.zeros(10)})
    y = pd.Series([1, 1, 1, 0, 0, 1, 1, 0, 1, 0])  # 60% up
    p = PriorClassifier().fit(X, y).predict_proba(X)
    assert p.shape == (10, 2)
    assert p[:, 1] == pytest.approx(np.full(10, 0.6))


def test_momentum_sign_conditional_rates():
    ret = pd.Series([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    y = pd.Series([1, 1, 1, 0, 0, 0, 1, 0])  # 75% up after +, 25% after -
    X = pd.DataFrame({"ret_1": ret})
    model = MomentumSignClassifier().fit(X, y)
    p = model.predict_proba(X)[:, 1]
    assert p[:4] == pytest.approx(np.full(4, 0.75))
    assert p[4:] == pytest.approx(np.full(4, 0.25))


def test_zoo_smoke_fit_predict():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"ret_1": rng.normal(size=n), "vol_24": np.abs(rng.normal(size=n))})
    y = pd.Series((rng.random(n) > 0.5).astype(int))
    for name, model in build_models(seed=0).items():
        model.fit(X, y)
        p = model.predict_proba(X)
        assert p.shape == (n, 2), name
        assert np.all((p >= 0) & (p <= 1)), name
        assert np.allclose(p.sum(axis=1), 1.0), name
