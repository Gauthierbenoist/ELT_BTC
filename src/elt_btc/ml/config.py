"""Typed configuration for the ML benchmark (YAML -> Pydantic)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_BENCHMARK_CONFIG_PATH = Path("config/benchmark.yaml")
_CONFIG_PATH_ENV_VAR = "ELT_BTC_BENCHMARK_CONFIG"


class DatasetSettings(BaseModel):
    """Source lake and bar-construction parameters."""

    parquet_root: Path = Path("data/raw/1m")
    timeframe: str = "1h"
    min_minutes_per_bar: int = Field(default=45, gt=0)


class FeatureSettings(BaseModel):
    """Rolling-window lengths (in bars) for the feature set.

    ``volume_windows`` empty (the default) keeps the feature set OHLC-only;
    non-empty adds the volume features from
    :mod:`elt_btc.features.volume`.
    """

    momentum_windows: list[int] = [3, 6, 12, 24, 72, 168]
    vol_windows: list[int] = [24, 72, 168]
    range_vol_windows: list[int] = [24, 168]
    channel_windows: list[int] = [24, 168]
    rsi_period: int = Field(default=14, gt=1)
    volume_windows: list[int] = []


class TargetSettings(BaseModel):
    """Label definition.

    ``next_bar``: ``1{close_{t+1} > close_t}`` (horizon = 1 bar).
    ``triple_barrier``: López de Prado triple-barrier labels (horizon =
    ``max_holding`` bars) — see :mod:`elt_btc.ml.labels`.
    """

    type: Literal["next_bar", "triple_barrier"] = "next_bar"
    vol_span: int = Field(default=42, gt=1)
    pt_mult: float = Field(default=1.0, gt=0)
    sl_mult: float = Field(default=1.0, gt=0)
    max_holding: int = Field(default=42, gt=0)

    @property
    def horizon_bars(self) -> int:
        """How many future bars a label may depend on (min purge)."""
        return self.max_holding if self.type == "triple_barrier" else 1


class SplitSettings(BaseModel):
    """Purged walk-forward cross-validation layout (all sizes in bars)."""

    n_splits: int = Field(default=8, gt=0)
    test_size: int = Field(default=4380, gt=0)
    min_train_size: int = Field(default=26280, gt=0)
    purge: int = Field(default=24, ge=1)


class ModelSettings(BaseModel):
    """Model-level knobs."""

    seed: int = 42


class BacktestSettings(BaseModel):
    """Evaluation-only backtest parameters (see ``elt_btc.ml.backtest``)."""

    fee_bps: float = Field(default=10.0, ge=0)
    threshold_band: float = Field(default=0.0, ge=0, lt=0.5)


class BenchmarkSettings(BaseModel):
    """Root settings object, mirroring ``config/benchmark.yaml``."""

    dataset: DatasetSettings = DatasetSettings()
    features: FeatureSettings = FeatureSettings()
    target: TargetSettings = TargetSettings()
    split: SplitSettings = SplitSettings()
    models: ModelSettings = ModelSettings()
    backtest: BacktestSettings = BacktestSettings()

    @model_validator(mode="after")
    def _purge_covers_label_horizon(self) -> BenchmarkSettings:
        """Reject configs where training labels could overlap the test window.

        A label spanning ``horizon`` future bars leaks test information into
        training unless at least ``horizon`` bars are purged before each
        test fold (the "open trade" overlap of triple-barrier labels).
        """
        if self.split.purge < self.target.horizon_bars:
            raise ValueError(
                f"split.purge ({self.split.purge}) must be >= the label horizon "
                f"({self.target.horizon_bars} bars for target type "
                f"{self.target.type!r}): training labels would overlap the test window"
            )
        return self


def load_benchmark_settings(path: Path | None = None) -> BenchmarkSettings:
    """Load and validate benchmark settings from a YAML file.

    Resolution order: explicit ``path``, the ``ELT_BTC_BENCHMARK_CONFIG``
    environment variable, then ``config/benchmark.yaml`` relative to the
    current working directory.
    """
    resolved = path or Path(os.environ.get(_CONFIG_PATH_ENV_VAR, DEFAULT_BENCHMARK_CONFIG_PATH))
    with resolved.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return BenchmarkSettings.model_validate(raw)
