#!/usr/bin/env python3
"""Check schemas of views in unified database"""

from pathlib import Path

import duckdb

db_path = Path("data/unified_wheel_trading.duckdb")
conn = duckdb.connect(str(db_path))

views = ["current_risk_free_rate", "current_vix", "current_unity_stock"]

for view in views:
    print(f"\n=== {view} ===")
    try:
        schema = conn.execute(f"DESCRIBE {view}").fetchall()
        for col in schema:
            print(f"  {col[0]:<20} {col[1]}")

        # Show sample data
        data = conn.execute(f"SELECT * FROM {view} LIMIT 2").fetchall()
        print("  Sample data:", data)
    except Exception as e:
        print(f"  Error: {e}")

conn.close()
