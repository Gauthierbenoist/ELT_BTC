# ELT_BTC

Data pipeline for BTC spot OHLCV, built as the data layer of a future ML
project (predicting the probability of an upward move on BTC spot). Two
independent flows:

1. **Historical 1m backfill** — full BTC/USDT 1-minute history from Binance
   (via [ccxt](https://github.com/ccxt/ccxt)) since 2017-08-17, stored locally
   as partitioned Parquet. High-resolution raw data, written once.
2. **Incremental 15m ingestion** — a `raw_ohlcv_15m` table on
   [Neon](https://neon.tech) Postgres, fed daily by GitHub Actions with the
   last 48 hours of closed 15-minute candles. Idempotent by construction.

## Data conventions

- **Timestamps are candle open times**, in UTC everywhere: epoch milliseconds
  (int64) in transit and in Parquet, `timestamptz` in Postgres.
- **Only closed candles are persisted.** A candle opening at `t` is closed
  once `t + timeframe <= now`; the in-progress candle is always excluded.
- **Idempotency**: the incremental flow uses
  `INSERT ... ON CONFLICT (timestamp) DO NOTHING`; the backfill skips
  partitions that are already complete. Re-running either flow is always safe.
- **Validation** (pandera) runs after every backfill partition and every
  incremental fetch: OHLC coherence (`low <= min(open, close)`,
  `high >= max(open, close)`), positive prices, non-negative volume, unique
  timestamps. Missing candles on the time grid (exchange maintenance windows,
  e.g. Binance 2018 outages) are logged as warnings but never block ingestion.

## Architecture

```
                 ┌──────────────────────────────┐
     Binance ────┤ ingestion/exchange.py        │  pagination + exponential backoff
     (ccxt)      └──────────────┬───────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        │                                               │
┌───────▼────────────┐                        ┌─────────▼──────────┐
│ backfill_1m        │  run once, resumable   │ incremental_15m    │  daily (GitHub Actions)
│ 1m since 2017      │                        │ last 48h, closed   │
└───────┬────────────┘                        └─────────┬──────────┘
        │  validation/schemas.py (pandera + gaps)       │
┌───────▼────────────┐                        ┌─────────▼──────────┐
│ Parquet (zstd)     │                        │ Neon Postgres      │
│ data/raw/1m/       │                        │ raw_ohlcv_15m      │
│  year=Y/month=M/   │                        │ (PK: timestamp)    │
└────────────────────┘                        └────────────────────┘
```

Business logic (dedup, closed-candle filter, gap detection, partition/month
math) lives in pure functions ([candles.py](src/elt_btc/candles.py),
[schemas.py](src/elt_btc/validation/schemas.py)) and is unit-tested without
any network or database access.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it installs the pinned Python
automatically):

```bash
git clone https://github.com/Gauthierbenoist/ELT_BTC
cd ELT_BTC
uv sync
```

Configuration lives in [config/settings.yaml](config/settings.yaml) (symbol,
timeframes, dates, paths, retry policy). No secrets there: the database URL
comes exclusively from the `DATABASE_URL` environment variable.

## Usage

### 1m backfill → Parquet

```bash
uv run python -m elt_btc.ingestion.backfill_1m            # full history to now
uv run python -m elt_btc.ingestion.backfill_1m --end 2017-10-01   # short test run
```

~4.7M candles ≈ 4,700 requests; expect 30–45 minutes under Binance rate
limits. The script is safe to interrupt: partitions are written atomically,
month by month, and complete months are skipped on the next run.

### Incremental 15m → Neon

Create the table once (e.g. with `psql "$DATABASE_URL" -f scripts/init_db.sql`),
then:

```bash
uv run python -m elt_btc.ingestion.incremental_15m --dry-run   # no DB required
DATABASE_URL=postgres://... uv run python -m elt_btc.ingestion.incremental_15m
```

Each run logs the number of candles fetched, rows actually inserted, and the
time range covered.

### GitHub Actions

[.github/workflows/incremental.yml](.github/workflows/incremental.yml) runs
the incremental flow daily at 06:15 UTC (plus manual `workflow_dispatch`).
One-time setup: add the `DATABASE_URL` secret in
*Settings → Secrets and variables → Actions*. Any unhandled exception fails
the job, so a red run means data did not land.

## Development

```bash
uv run pytest          # unit tests (no network / DB needed)
uv run ruff check .    # lint (print() is banned — logging only)
uv run ruff format .   # format
uv run mypy            # strict type checking on src/
```

## Troubleshooting: antivirus / corporate TLS interception

Both entry points call `truststore.inject_into_ssl()` at startup, so TLS
certificates are verified against the **OS trust store** (like pip does).
This keeps HTTPS working when an antivirus (e.g. Norton 360) or a corporate
proxy intercepts TLS with its own root CA, and is a no-op on plain
environments such as CI runners.

Two related quirks on such machines:

- `uv` itself may need the OS trust store too: `uv sync --native-tls`
  (or set `UV_NATIVE_TLS=1`).
- Some antivirus TLS hooks crash the OpenSSL bundled with uv-managed
  (python-build-standalone) interpreters with `no OPENSSL_Applink`. This
  project sets `python-preference = "system"` in `pyproject.toml` so uv
  prefers a python.org interpreter when one is installed.

## Project layout

```
config/settings.yaml           pipeline configuration (no secrets)
src/elt_btc/
  config.py                    YAML -> Pydantic settings
  candles.py                   pure candle logic (dedup, closed filter, timeframes)
  ingestion/exchange.py        ccxt pagination + backoff
  ingestion/backfill_1m.py     historical 1m -> partitioned Parquet
  ingestion/incremental_15m.py daily 15m -> Postgres
  validation/schemas.py        pandera schema + gap detection
  storage/parquet_io.py        atomic partition writes, resume helpers
  storage/postgres_io.py       idempotent inserts (psycopg 3)
  utils/logging.py             stdlib logging, UTC ISO timestamps
tests/                         pytest suite for all business logic
scripts/init_db.sql            DDL for raw_ohlcv_15m
```
