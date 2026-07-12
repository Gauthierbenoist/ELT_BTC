"""Typed configuration loading (YAML -> Pydantic models).

Secrets are never part of this file: ``DATABASE_URL`` is read directly from
the environment by the incremental entry point.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path("config/settings.yaml")
_CONFIG_PATH_ENV_VAR = "ELT_BTC_CONFIG"


class ExchangeSettings(BaseModel):
    """ccxt exchange identity and retry/pagination parameters.

    ``public_api_url`` overrides the exchange's public (market data) API
    base URL. For Binance, ``https://data-api.binance.vision/api/v3`` serves
    the same market data without the HTTP 451 geo-block that hits US IPs
    (e.g. GitHub Actions runners).
    """

    id: str = "binance"
    symbol: str = "BTC/USDT"
    public_api_url: str | None = None
    page_limit: int = Field(default=1000, gt=0)
    max_retries: int = Field(default=6, gt=0)
    backoff_base_seconds: float = Field(default=1.0, gt=0)
    backoff_max_seconds: float = Field(default=60.0, gt=0)


class BackfillSettings(BaseModel):
    """Historical 1m backfill to partitioned Parquet."""

    timeframe: str = "1m"
    start: date = date(2017, 8, 17)
    parquet_root: Path = Path("data/raw/1m")
    compression: Literal["zstd", "snappy"] = "zstd"


class IncrementalSettings(BaseModel):
    """Daily 15m incremental ingestion to Postgres."""

    timeframe: str = "15m"
    lookback_hours: int = Field(default=48, gt=0)
    table: str = "raw_ohlcv_15m"


class Settings(BaseModel):
    """Root settings object, mirroring ``config/settings.yaml``."""

    exchange: ExchangeSettings = ExchangeSettings()
    backfill: BackfillSettings = BackfillSettings()
    incremental: IncrementalSettings = IncrementalSettings()


def load_settings(path: Path | None = None) -> Settings:
    """Load and validate settings from a YAML file.

    Resolution order: explicit ``path`` argument, the ``ELT_BTC_CONFIG``
    environment variable, then ``config/settings.yaml`` relative to the
    current working directory.

    Raises:
        FileNotFoundError: If the resolved config file does not exist.
        pydantic.ValidationError: If the YAML content is invalid.
    """
    resolved = path or Path(os.environ.get(_CONFIG_PATH_ENV_VAR, DEFAULT_CONFIG_PATH))
    with resolved.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Settings.model_validate(raw)
