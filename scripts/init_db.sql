-- Raw 15m BTC/USDT spot OHLCV from Binance, fed daily by the incremental job.
-- Convention: "timestamp" is the candle OPEN time, UTC. Only closed candles
-- are ever inserted; the primary key makes re-runs idempotent.
CREATE TABLE IF NOT EXISTS raw_ohlcv_15m (
    "timestamp" timestamptz      PRIMARY KEY,
    open        double precision NOT NULL,
    high        double precision NOT NULL,
    low         double precision NOT NULL,
    close       double precision NOT NULL,
    volume      double precision NOT NULL,
    ingested_at timestamptz      NOT NULL DEFAULT now()
);
