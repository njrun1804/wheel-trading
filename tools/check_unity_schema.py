#!/usr/bin/env python3
"""
Check Unity options table schema.
"""

from pathlib import Path
import duckdb

db_path = Path("~/.wheel_trading/cache/wheel_cache.duckdb").expanduser()
conn = duckdb.connect(str(db_path))

# Check table structure
print("Unity options_daily table schema:")
schema = conn.execute("DESCRIBE unity_options_daily").fetchall()
for col in schema:
    print(f"  {col[0]:<20} {col[1]}")

# Check a sample record
print("\nSample record:")
sample = conn.execute("SELECT * FROM unity_options_daily LIMIT 1").fetchone()
if sample:
    columns = conn.execute("DESCRIBE unity_options_daily").fetchall()
    for i, (col, _) in enumerate(columns):
        print(f"  {col}: {sample[i]}")
        
conn.close()