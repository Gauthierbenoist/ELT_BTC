"""Run-directory discovery and loading tests."""

from __future__ import annotations

import json

import pandas as pd

from elt_btc.ml.runs import list_runs, load_run


def make_run_dir(root, name: str, with_predictions: bool = True):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps({"results": {}}), encoding="utf-8")
    if with_predictions:
        pd.DataFrame(
            {
                "model": ["prior"],
                "fold": [0],
                "timestamp": [0],
                "y": [1],
                "p_up": [0.5],
                "ret_next": [0.01],
            }
        ).to_parquet(run_dir / "predictions.parquet", index=False)
    return run_dir


def test_list_runs_newest_first_and_filters_incomplete(tmp_path):
    make_run_dir(tmp_path, "run_20260101T000000Z")
    make_run_dir(tmp_path, "run_20260301T000000Z")
    make_run_dir(tmp_path, "run_20260201T000000Z", with_predictions=False)  # aborted run
    (tmp_path / "not_a_run").mkdir()
    runs = list_runs(tmp_path)
    assert [r.name for r in runs] == ["run_20260301T000000Z", "run_20260101T000000Z"]


def test_list_runs_empty_root(tmp_path):
    assert list_runs(tmp_path / "missing") == []


def test_load_run_roundtrip(tmp_path):
    run_dir = make_run_dir(tmp_path, "run_20260101T000000Z")
    (run_dir / "importances.json").write_text(
        json.dumps({"lightgbm": {"ret_1": 10.0}}), encoding="utf-8"
    )
    run = load_run(run_dir)
    assert run.report == {"results": {}}
    assert len(run.predictions) == 1
    assert run.importances["lightgbm"]["ret_1"] == 10.0
    assert run.calibration == {}  # optional artifact absent
