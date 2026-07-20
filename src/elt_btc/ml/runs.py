"""Discovery and loading of benchmark run directories.

Kept free of any UI dependency so the dashboard stays a thin layer and this
logic is unit-testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RunInfo:
    """Lightweight handle on a run directory (no file content loaded)."""

    path: Path
    name: str


@dataclass(frozen=True)
class Run:
    """Fully loaded run artifacts.

    ``bars`` (full OHLCV bars), ``contributions`` (per-prediction TreeSHAP
    values) and ``features`` (feature values per decision time) are None
    for runs generated before those artifacts existed.
    """

    report: dict[str, Any]
    predictions: pd.DataFrame
    importances: dict[str, dict[str, float]]
    calibration: dict[str, list[dict[str, float]]]
    bars: pd.DataFrame | None = None
    contributions: pd.DataFrame | None = None
    features: pd.DataFrame | None = None


def list_runs(root: Path) -> list[RunInfo]:
    """Valid run directories under ``root``, newest first.

    A directory qualifies when it contains both ``metrics.json`` and
    ``predictions.parquet`` (older or aborted runs are skipped).
    """
    runs = [
        RunInfo(path=path, name=path.name)
        for path in root.glob("run_*")
        if (path / "metrics.json").is_file() and (path / "predictions.parquet").is_file()
    ]
    return sorted(runs, key=lambda run: run.name, reverse=True)


def load_run(run_dir: Path) -> Run:
    """Load every analysis artifact of one run (one disk read per file)."""
    report: dict[str, Any] = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    importances_path = run_dir / "importances.json"
    calibration_path = run_dir / "calibration.json"
    importances: dict[str, dict[str, float]] = (
        json.loads(importances_path.read_text(encoding="utf-8"))
        if importances_path.is_file()
        else {}
    )
    calibration: dict[str, list[dict[str, float]]] = (
        json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration_path.is_file()
        else {}
    )

    def _optional_parquet(name: str) -> pd.DataFrame | None:
        path = run_dir / name
        return pd.read_parquet(path) if path.is_file() else None

    return Run(
        report=report,
        predictions=predictions,
        importances=importances,
        calibration=calibration,
        bars=_optional_parquet("bars.parquet"),
        contributions=_optional_parquet("contributions.parquet"),
        features=_optional_parquet("features.parquet"),
    )
