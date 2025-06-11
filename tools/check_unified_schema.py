#!/usr/bin/env python3
"""Check the actual schema of unified_wheel_trading.duckdb"""

from pathlib import Path

import duckdb

db_path = Path("data/unified_wheel_trading.duckdb")
conn = duckdb.connect(str(db_path))

print("=== Database Tables ===")
tables = conn.execute("SHOW TABLES").fetchall()
for table in tables:
    print(f"- {table[0]}")

print("\n=== market_data Schema ===")
schema = conn.execute("DESCRIBE market_data").fetchall()
for col in schema:
    print(f"  {col[0]:<20} {col[1]}")

print("\n=== Sample market_data Records ===")
samples = conn.execute(
    """
    SELECT symbol, date, open, high, low, close, volume, returns
    FROM market_data
    LIMIT 5
"""
).fetchall()

for row in samples:
    print(f"  {row}")

print("\n=== Data Summary ===")
summary = conn.execute(
    """
    SELECT
        symbol,
        COUNT(*) as records,
        MIN(date) as first_date,
        MAX(date) as last_date
    FROM market_data
    GROUP BY symbol
    ORDER BY records DESC
    LIMIT 10
"""
).fetchall()

for row in summary:
    print(f"  {row[0]:<25} Records: {row[1]:<7} Range: {row[2]} to {row[3]}")

conn.close()
