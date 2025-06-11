#!/usr/bin/env python3
"""Investigate where option prices are actually stored"""

from pathlib import Path

import duckdb

db_path = Path("data/unified_wheel_trading.duckdb")
conn = duckdb.connect(str(db_path))

print("=== Checking Option Price Data ===")
# Check a specific option to see all its data
option_data = conn.execute(
    """
    SELECT *
    FROM market_data
    WHERE symbol LIKE 'U %'
    LIMIT 5
"""
).fetchall()

print("\nSample option records (all columns):")
cols = conn.execute("DESCRIBE market_data").fetchall()
col_names = [col[0] for col in cols]
print(f"Columns: {col_names}")

for row in option_data:
    print("\nRecord:")
    for i, col in enumerate(col_names):
        print(f"  {col}: {row[i]}")

print("\n=== Checking if option prices are in close column ===")
option_prices = conn.execute(
    """
    SELECT
        symbol,
        date,
        open,
        high,
        low,
        close,
        volume,
        COUNT(*) OVER (PARTITION BY symbol) as days_of_data
    FROM market_data
    WHERE symbol LIKE 'U %'
    AND close IS NOT NULL
    AND close > 0
    ORDER BY symbol, date DESC
    LIMIT 20
"""
).fetchall()

print(f"\nOptions with non-null close prices: {len(option_prices)}")
for row in option_prices:
    print(f"  {row[0]:<25} {row[1]} close=${row[5]:.2f} volume={row[6]}")

print("\n=== Checking options_metadata table ===")
metadata_sample = conn.execute(
    """
    SELECT *
    FROM options_metadata
    LIMIT 5
"""
).fetchall()

metadata_cols = conn.execute("DESCRIBE options_metadata").fetchall()
print("\noptions_metadata columns:")
for col in metadata_cols:
    print(f"  {col[0]:<20} {col[1]}")

print("\n=== Checking if we need to join tables ===")
# See if we can join market_data with options_metadata
joined_data = conn.execute(
    """
    SELECT
        md.symbol,
        md.date,
        md.close as market_close,
        om.strike,
        om.expiration,
        om.option_type
    FROM market_data md
    JOIN options_metadata om ON md.symbol = om.symbol
    WHERE md.symbol LIKE 'U %'
    AND md.close IS NOT NULL
    LIMIT 10
"""
).fetchall()

print(f"\nJoined records found: {len(joined_data)}")
for row in joined_data:
    print(f"  {row}")

print("\n=== Checking other potential price sources ===")
# Check if there's data in other tables
for table in ["greeks_cache", "available_puts"]:
    try:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"\n{table}: {count} records")
        if count > 0:
            sample = conn.execute(f"SELECT * FROM {table} LIMIT 2").fetchall()
            for row in sample:
                print(f"  {row}")
    except:
        pass

conn.close()
