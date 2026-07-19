"""Benchmark runner: purged walk-forward evaluation of the model zoo.

Usage::

    uv run python -m elt_btc.ml.benchmark [--models logreg,lightgbm] [--config ...]

Writes ``metrics.json``, ``folds.csv`` and ``importances.json`` under
``outputs/benchmark/run_<UTC>/`` and logs an aggregate summary table.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from elt_btc.ml.config import BenchmarkSettings, load_benchmark_settings
from elt_btc.ml.dataset import Dataset, build_dataset
from elt_btc.ml.metrics import aggregate_folds, evaluate_fold
from elt_btc.ml.models import build_models
from elt_btc.ml.splits import PurgedWalkForwardSplit
from elt_btc.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _iso(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).isoformat()


def run_benchmark(
    settings: BenchmarkSettings, dataset: Dataset, model_names: list[str] | None = None
) -> tuple[dict[str, object], pd.DataFrame, dict[str, dict[str, float]]]:
    """Evaluate every requested model across the purged walk-forward folds.

    Returns ``(report, folds_frame, importances)`` where ``report`` is the
    JSON-serializable run summary.
    """
    splitter = PurgedWalkForwardSplit(
        n_splits=settings.split.n_splits,
        test_size=settings.split.test_size,
        min_train_size=settings.split.min_train_size,
        purge=settings.split.purge,
    )
    available = list(build_models(settings.models.seed))
    names = model_names or available
    unknown = set(names) - set(available)
    if unknown:
        raise ValueError(f"Unknown model(s) {sorted(unknown)}; available: {available}")

    fold_rows: list[dict[str, object]] = []
    per_model_folds: dict[str, list[dict[str, float]]] = {name: [] for name in names}
    importances: dict[str, dict[str, float]] = {}

    for fold, (train_idx, test_idx) in enumerate(splitter.split(dataset.X)):
        X_train, y_train = dataset.X.iloc[train_idx], dataset.y.iloc[train_idx]
        X_test, y_test = dataset.X.iloc[test_idx], dataset.y.iloc[test_idx]
        test_start = _iso(int(dataset.timestamps.iloc[test_idx[0]]))
        test_end = _iso(int(dataset.timestamps.iloc[test_idx[-1]]))
        logger.info(
            "Fold %d: train %d rows, test %d rows [%s -> %s]",
            fold,
            len(train_idx),
            len(test_idx),
            test_start,
            test_end,
        )
        models = build_models(settings.models.seed)  # fresh instances every fold
        for name in names:
            model = models[name]
            model.fit(X_train, y_train)
            p_up = model.predict_proba(X_test)[:, 1]
            scores = evaluate_fold(y_test, p_up)
            per_model_folds[name].append(scores)
            fold_rows.append(
                {"model": name, "fold": fold, "test_start": test_start, "test_end": test_end}
                | scores
            )
            if fold == settings.split.n_splits - 1:
                importances[name] = _extract_importances(name, model, list(dataset.X.columns))

    results = {
        name: {"folds": per_model_folds[name], "aggregate": aggregate_folds(per_model_folds[name])}
        for name in names
    }
    report: dict[str, object] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "config": settings.model_dump(mode="json"),
        "dataset": {
            "n_samples": len(dataset.X),
            "n_features": dataset.X.shape[1],
            "start": _iso(int(dataset.timestamps.iloc[0])),
            "end": _iso(int(dataset.timestamps.iloc[-1])),
            "up_rate": float(dataset.y.mean()),
        },
        "results": results,
    }
    return report, pd.DataFrame(fold_rows), importances


def _extract_importances(name: str, model: object, features: list[str]) -> dict[str, float]:
    """Feature importances of the last fitted fold (empty for naive models)."""
    if name == "logreg":
        coefs = np.abs(np.ravel(model.named_steps["clf"].coef_))  # type: ignore[attr-defined]
        values = coefs
    elif name == "lightgbm":
        values = np.asarray(model.feature_importances_, dtype=float)  # type: ignore[attr-defined]
    else:
        return {}
    order = np.argsort(values)[::-1]
    return {features[i]: float(values[i]) for i in order}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to benchmark.yaml")
    parser.add_argument("--models", default=None, help="Comma-separated subset of models to run")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    settings = load_benchmark_settings(args.config)
    np.random.seed(settings.models.seed)

    dataset = build_dataset(settings)
    model_names = args.models.split(",") if args.models else None
    report, folds_frame, importances = run_benchmark(settings, dataset, model_names)

    run_dir = args.output_dir / datetime.now(tz=UTC).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    folds_frame.to_csv(run_dir / "folds.csv", index=False)
    (run_dir / "importances.json").write_text(json.dumps(importances, indent=2), encoding="utf-8")

    results = report["results"]
    assert isinstance(results, dict)
    for name, payload in results.items():
        agg = payload["aggregate"]
        logger.info(
            "%-14s AUC %.4f±%.4f | log-loss %.4f±%.4f | Brier %.4f±%.4f | acc %.4f±%.4f",
            name,
            agg["auc"]["mean"],
            agg["auc"]["std"],
            agg["log_loss"]["mean"],
            agg["log_loss"]["std"],
            agg["brier"]["mean"],
            agg["brier"]["std"],
            agg["accuracy"]["mean"],
            agg["accuracy"]["std"],
        )
    logger.info("Report written to %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
