"""Benchmark runner: purged walk-forward evaluation of the model zoo.

Usage::

    uv run python -m elt_btc.ml.benchmark [--models logreg,lightgbm] [--config ...]

Each run writes a self-contained directory ``outputs/benchmark/run_<UTC>/``:

- ``metrics.json``      — config echo, git commit, per-fold + aggregate
  classification metrics and backtest metrics (gross/net Sharpe, drawdown,
  turnover), plus pooled out-of-sample backtest per model
- ``folds.csv``         — one row per (model, fold)
- ``predictions.parquet`` — every out-of-sample prediction
  (model, fold, timestamp, y, p_up, ret_next): the raw material for any
  further analysis (SHAP, calibration, custom backtests) without refitting
- ``calibration.json``  — reliability-diagram data per model (pooled OOS)
- ``importances.json``  — LightGBM gain importances / |logreg coefs|
- ``models/``           — every fitted model, one joblib file per fold
- ``contributions.parquet`` — per-prediction feature contributions
  (TreeSHAP via LightGBM's ``pred_contrib``, log-odds space), computed by
  the fold model that produced each OOS prediction
- ``features.parquet``  — feature values per decision time, for inspection
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from elt_btc.candles import timeframe_to_ms
from elt_btc.ml.backtest import backtest_metrics, meta_effective_proba
from elt_btc.ml.config import BenchmarkSettings, load_benchmark_settings
from elt_btc.ml.dataset import Dataset, build_dataset
from elt_btc.ml.metrics import aggregate_folds, calibration_table, evaluate_fold
from elt_btc.ml.models import build_models
from elt_btc.ml.splits import PurgedWalkForwardSplit
from elt_btc.ml.trade_backtest import simulate_trades, simulate_trades_trailing
from elt_btc.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_MS_PER_YEAR = 365 * 86_400 * 1000


def _iso(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).isoformat()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def run_benchmark(
    settings: BenchmarkSettings,
    dataset: Dataset,
    model_names: list[str] | None = None,
    models_dir: Path | None = None,
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, float]],
    pd.DataFrame | None,
]:
    """Evaluate every requested model across the purged walk-forward folds.

    Returns ``(report, folds_frame, predictions_frame, importances,
    contributions)`` — ``contributions`` holds per-prediction TreeSHAP
    values (LightGBM only, None otherwise). When ``models_dir`` is given,
    every fitted model is persisted there as ``<name>_fold<i>.joblib``.
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

    bars_per_year = _MS_PER_YEAR / timeframe_to_ms(settings.dataset.timeframe)
    fee_rate = settings.backtest.fee_bps / 10_000.0

    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    contribution_frames: list[pd.DataFrame] = []
    per_model_folds: dict[str, list[dict[str, float]]] = {name: [] for name in names}
    importances: dict[str, dict[str, float]] = {}

    for fold, (train_idx, test_idx) in enumerate(splitter.split(dataset.X)):
        X_train, y_train = dataset.X.iloc[train_idx], dataset.y.iloc[train_idx]
        X_test, y_test = dataset.X.iloc[test_idx], dataset.y.iloc[test_idx]
        ret_test = dataset.ret_next.iloc[test_idx].to_numpy()
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
        is_meta = settings.target.type == "meta_triple_barrier"
        side_test = dataset.side.iloc[test_idx].to_numpy()
        for name in names:
            model = models[name]
            model.fit(X_train, y_train)
            p_up = model.predict_proba(X_test)[:, 1]
            # For meta targets the bar-level layer needs directional probas.
            p_directional = meta_effective_proba(p_up, side_test) if is_meta else p_up
            scores = evaluate_fold(y_test, p_up) | backtest_metrics(
                p_directional,
                ret_test if not is_meta else side_test * ret_test,
                fee_rate=fee_rate,
                bars_per_year=bars_per_year,
                threshold_band=settings.backtest.threshold_band,
            )
            per_model_folds[name].append(scores)
            fold_rows.append(
                {"model": name, "fold": fold, "test_start": test_start, "test_end": test_end}
                | scores
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "fold": fold,
                        "timestamp": dataset.timestamps.iloc[test_idx].to_numpy(),
                        "y": y_test.to_numpy(),
                        "p_up": p_up,
                        "ret_next": ret_test,
                        "holding_bars": dataset.holding_bars.iloc[test_idx].to_numpy(),
                        "side": side_test,
                        "close": dataset.entry_close.iloc[test_idx].to_numpy(),
                        "sigma": dataset.sigma.iloc[test_idx].to_numpy(),
                    }
                )
            )
            if name == "lightgbm":
                # TreeSHAP contributions (log-odds), from THIS fold's model.
                contrib = model.booster_.predict(  # type: ignore[attr-defined]
                    X_test, pred_contrib=True
                )
                contrib_frame = pd.DataFrame(contrib, columns=[*dataset.X.columns, "bias"])
                contrib_frame.insert(0, "timestamp", dataset.timestamps.iloc[test_idx].to_numpy())
                contrib_frame.insert(0, "model", name)
                contribution_frames.append(contrib_frame)
            if models_dir is not None:
                # compress=3 (zlib): ~5x smaller LightGBM files, negligible load cost.
                joblib.dump(model, models_dir / f"{name}_fold{fold}.joblib", compress=3)
            if fold == settings.split.n_splits - 1:
                importances[name] = _extract_importances(name, model, list(dataset.X.columns))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    bar_ms = timeframe_to_ms(settings.dataset.timeframe)
    is_meta = settings.target.type == "meta_triple_barrier"
    results: dict[str, dict[str, object]] = {}
    for name in names:
        pooled = predictions.loc[predictions["model"] == name].sort_values("timestamp")
        pooled_p = pooled["p_up"].to_numpy()
        pooled_side = pooled["side"].to_numpy()
        pooled_ret = pooled["ret_next"].to_numpy()
        results[name] = {
            "folds": per_model_folds[name],
            "aggregate": aggregate_folds(per_model_folds[name]),
            "pooled_backtest": backtest_metrics(
                meta_effective_proba(pooled_p, pooled_side) if is_meta else pooled_p,
                pooled_side * pooled_ret if is_meta else pooled_ret,
                fee_rate=fee_rate,
                bars_per_year=bars_per_year,
                threshold_band=settings.backtest.threshold_band,
            ),
            # Executable policy: one trade at a time, held to its exit.
            "trade_backtest": simulate_trades(
                pooled["timestamp"].to_numpy(),
                pooled_p,
                pooled_ret,
                pooled["holding_bars"].to_numpy(),
                bar_ms=bar_ms,
                fee_rate=fee_rate,
                threshold_band=settings.backtest.threshold_band,
                side=pooled_side if is_meta else None,
            ).metrics,
        }
        if is_meta:
            aligned_bars = (
                dataset.bars.set_index("timestamp")
                .reindex(pooled["timestamp"].to_numpy())
                .reset_index()
            )
            results[name]["trade_backtest_trailing"] = simulate_trades_trailing(
                pooled["timestamp"].to_numpy(),
                pooled_p,
                pooled_side,
                aligned_bars["high"].to_numpy(),
                aligned_bars["low"].to_numpy(),
                aligned_bars["close"].to_numpy(),
                pooled["sigma"].to_numpy(),
                bar_ms=bar_ms,
                fee_rate=fee_rate,
                pt_mult=settings.target.pt_mult,
                sl_mult=settings.target.sl_mult,
                max_holding=settings.target.max_holding,
                threshold_band=settings.backtest.threshold_band,
            ).metrics

    report: dict[str, object] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "git_commit": _git_commit(),
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
    contributions = (
        pd.concat(contribution_frames, ignore_index=True) if contribution_frames else None
    )
    return report, pd.DataFrame(fold_rows), predictions, importances, contributions


def _extract_importances(name: str, model: object, features: list[str]) -> dict[str, float]:
    """Feature importances of the last fitted fold (empty for naive models).

    LightGBM importances use total split gain (more informative than split
    counts); logreg uses absolute coefficients on standardized features.
    """
    if name == "logreg":
        values = np.abs(np.ravel(model.named_steps["clf"].coef_))  # type: ignore[attr-defined]
    elif name == "lightgbm":
        booster = model.booster_  # type: ignore[attr-defined]
        values = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
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
    parser.add_argument(
        "--name",
        default=None,
        help="Run directory name suffix (run_<name>); defaults to a UTC timestamp",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    settings = load_benchmark_settings(args.config)
    np.random.seed(settings.models.seed)

    run_suffix = args.name or datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / f"run_{run_suffix}"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(settings)
    model_names = args.models.split(",") if args.models else None
    report, folds_frame, predictions, importances, contributions = run_benchmark(
        settings, dataset, model_names, models_dir=models_dir
    )

    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    folds_frame.to_csv(run_dir / "folds.csv", index=False)
    predictions.to_parquet(run_dir / "predictions.parquet", compression="zstd", index=False)
    # Full OHLCV bars for the dashboard's candlestick trade inspector.
    dataset.bars.to_parquet(run_dir / "bars.parquet", compression="zstd", index=False)
    if contributions is not None:
        contributions.to_parquet(run_dir / "contributions.parquet", compression="zstd", index=False)
    pd.concat([dataset.timestamps.rename("timestamp"), dataset.X], axis=1).to_parquet(
        run_dir / "features.parquet", compression="zstd", index=False
    )
    (run_dir / "importances.json").write_text(json.dumps(importances, indent=2), encoding="utf-8")
    calibration = {
        name: calibration_table(
            predictions.loc[predictions["model"] == name, "y"],
            predictions.loc[predictions["model"] == name, "p_up"].to_numpy(),
        )
        for name in predictions["model"].unique()
    }
    (run_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    results = report["results"]
    assert isinstance(results, dict)
    for name, payload in results.items():
        agg = payload["aggregate"]
        pooled = payload["pooled_backtest"]
        trades = payload["trade_backtest"]
        logger.info(
            "%-14s AUC %.4f±%.4f | acc %.4f | Sharpe gross %+.2f net %+.2f | "
            "maxDD %.1f%% | turnover %.3f",
            name,
            agg["auc"]["mean"],
            agg["auc"]["std"],
            agg["accuracy"]["mean"],
            pooled["sharpe_gross"],
            pooled["sharpe_net"],
            100 * pooled["max_drawdown_net"],
            pooled["turnover"],
        )
        logger.info(
            "%-14s trade-level: %d trades | win %.1f%% | Sharpe net %+.2f | "
            "ann ret %+.1f%% | maxDD %.1f%% | exposure %.2f",
            name,
            int(trades["n_trades"]),
            100 * trades["win_rate"],
            trades["sharpe_net"],
            100 * trades["ann_return_net"],
            100 * trades["max_drawdown_net"],
            trades["exposure"],
        )
        trailing = payload.get("trade_backtest_trailing")
        if trailing:
            logger.info(
                "%-14s trailing (v2): %d trades | win %.1f%% | Sharpe net %+.2f | "
                "ann ret %+.1f%% | maxDD %.1f%% | exposure %.2f",
                name,
                int(trailing["n_trades"]),
                100 * trailing["win_rate"],
                trailing["sharpe_net"],
                100 * trailing["ann_return_net"],
                100 * trailing["max_drawdown_net"],
                trailing["exposure"],
            )
    logger.info("Report written to %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
