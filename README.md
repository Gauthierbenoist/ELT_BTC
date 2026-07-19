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

## ML benchmark (P(up), OHLC-only)

Baseline models predicting `P(close_{t+1} > close_t)` on **1h bars**
resampled from the 1m lake, evaluated with **purged walk-forward
cross-validation**. This is the reference floor every future model must
beat under the same protocol.

```bash
uv run python -m elt_btc.ml.benchmark                      # full zoo
uv run python -m elt_btc.ml.benchmark --models logreg      # subset
```

Configuration in [config/benchmark.yaml](config/benchmark.yaml). Each run
writes a self-contained, reproducible directory `outputs/benchmark/run_<UTC>/`:

| File | Content |
|---|---|
| `metrics.json` | config + git commit, per-fold & aggregate classification **and backtest** metrics (Sharpe gross/net, max drawdown, turnover, exposure) |
| `predictions.parquet` | every OOS prediction (`model, fold, timestamp, y, p_up, ret_next`) — the raw material for SHAP, calibration, custom backtests, without refitting |
| `models/` | every fitted model (`<name>_fold<i>.joblib`), reloadable via `joblib.load` |
| `importances.json` | LightGBM **gain** importances / abs. logreg coefficients |
| `calibration.json` | reliability-diagram data per model (pooled OOS) |
| `folds.csv` | one row per (model, fold) for quick spreadsheet analysis |

### Dashboard

Interactive comparison of every model in a run — metrics table (Sharpe
gross/net, annual return, volatility, max drawdown, accuracy, AUC, number
of trades…), net equity vs buy & hold, drawdown curves, net-return
distributions, feature importances and calibration:

```bash
uv sync --group dashboard        # streamlit + plotly (kept out of CI)
uv run streamlit run dashboard/app.py
```

Everything is recomputed live from `predictions.parquet` through the same
`strategy_returns` code path as the benchmark, so the **fee and
neutral-band sliders re-price all curves and metrics instantly** without
reloading any model. Models are persisted compressed (`joblib`, zlib-3,
~2x smaller) and reload transparently with `joblib.load`.

The backtest layer ([ml/backtest.py](src/elt_btc/ml/backtest.py)) is for
*evaluation only*: sign-of-probability positions, flat one-way fee per
position change (default 10 bps), no slippage/sizing. Reference run: the
LightGBM Sharpe is **+1.9 gross but deeply negative net of fees** (turnover
~0.8 position changes per hour) — the statistical edge is real, the trading
edge at 1h horizon is not. That, precisely, is what a benchmark is for.

### Look-ahead protections

1. Causal resampling: a 1h bar uses only the 1m candles inside its hour;
   bars with < 45 minutes of data (exchange outages) are dropped.
2. Causal features by construction ([features/ohlc.py](src/elt_btc/features/ohlc.py)):
   trailing `rolling`/`shift(k>=0)`/`ewm` only — the contract is enforced by a
   **prefix-invariance test** (truncating the future must not change past
   feature values) in [test_features_ohlc.py](tests/test_features_ohlc.py).
3. Only the target looks forward (`shift(-1)`), and the last row is dropped.
4. [PurgedWalkForwardSplit](src/elt_btc/ml/splits.py): expanding train,
   chronological test folds, `purge` bars removed before each test window so
   no training label overlaps it.

### Reference results (8 folds ≈ 6 months each, 2022-07 → 2026-07)

| Model | AUC | Log-loss | Accuracy |
|---|---|---|---|
| prior | 0.500 ± 0.000 | 0.6931 | 0.507 |
| momentum_sign | 0.530 ± 0.012 | 0.6915 | 0.530 |
| logreg | 0.555 ± 0.014 | 0.6891 | 0.539 |
| lightgbm | 0.558 ± 0.011 | 0.6919 | 0.543 |

Read this honestly: ~0.55 AUC on 1h BTC from OHLC alone reflects real but
weak short-horizon autocorrelation — before any excitement, remember it is
gross of fees/slippage and says nothing about a tradable edge. Any future
model claiming much more should first be suspected of leakage.

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
