"""ELT pipeline for BTC spot OHLCV data.

Conventions used across the package:
- Every candle timestamp is the candle *open time*.
- All timestamps are UTC, stored as milliseconds since the Unix epoch
  (int64) in transit and in Parquet, and as ``timestamptz`` in Postgres.
"""
