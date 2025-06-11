#!/usr/bin/env python3
"""Load Unity options parquet into DuckDB."""

import os

import duckdb

parquet_path = "data/unity-options/processed/unity_ohlcv_3y.parquet"
duckdb_path = "data/cache/wheel_cache.duckdb"

print(f"Loading parquet into DuckDB at {duckdb_path}")
os.makedirs(os.path.dirname(duckdb_path), exist_ok=True)

conn = duckdb.connect(duckdb_path)

# Create table from parquet
conn.execute(
    f"""
    CREATE OR REPLACE TABLE unity_options_ohlcv AS
    SELECT * FROM read_parquet("{parquet_path}")
"""
)

# Create indexes
conn.execute(
    "CREATE INDEX IF NOT EXISTS idx_unity_symbol_date ON unity_options_ohlcv(symbol, ts_event)"
)

# Show row count and sample
count = conn.execute("SELECT COUNT(*) FROM unity_options_ohlcv").fetchone()[0]
print(f"\nLoaded {count:,} rows into DuckDB")

# Show sample data
print("\nSample Unity options:")
samples = conn.execute(
    """
    SELECT symbol, ts_event, close, volume
    FROM unity_options_ohlcv
    LIMIT 5
"""
).fetchall()

for row in samples:
    print(f"  {row[0]}: ${row[2]:.2f} on {row[1]} (vol: {row[3]:,})")

# Show date range
dates = conn.execute(
    """
    SELECT MIN(ts_event) as min_date, MAX(ts_event) as max_date
    FROM unity_options_ohlcv
"""
).fetchone()
print(f"\nDate range: {dates[0]} to {dates[1]}")

# Show liquidity analysis
print("\n💧 Most Liquid Put Strikes (last 30 days):")
liquidity = conn.execute(
    """
    SELECT
        CAST(SUBSTRING(symbol, 14, 8) AS FLOAT) / 1000 as strike,
        ROUND(AVG(volume)) as avg_volume,
        COUNT(*) as days
    FROM unity_options_ohlcv
    WHERE SUBSTRING(symbol, 13, 1) = 'P'
        AND ts_event >= CURRENT_DATE - INTERVAL 30 DAY
    GROUP BY strike
    HAVING AVG(volume) > 100
    ORDER BY avg_volume DESC
    LIMIT 10
"""
).fetchall()

for strike, vol, days in liquidity:
    print(f"   ${strike:.2f}: {vol:,.0f} avg volume ({days} days)")

conn.close()
print("\n✅ Data successfully loaded into DuckDB!")
