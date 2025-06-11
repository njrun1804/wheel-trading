#!/usr/bin/env python3
"""Check the actual schemas of existing databases."""

import os

import duckdb

# Check home database
print("HOME DATABASE SCHEMA")
print("=" * 80)
home_db = os.path.expanduser("~/.wheel_trading/cache/wheel_cache.duckdb")
home_conn = duckdb.connect(home_db, read_only=True)

print("\nTables:")
tables = home_conn.execute("SHOW TABLES").fetchall()
for table in tables:
    print(f"\n{table[0]}:")
    schema = home_conn.execute(f"DESCRIBE {table[0]}").fetchall()
    for col in schema:
        print(f"  {col[0]}: {col[1]}")

    # Show sample data
    try:
        sample = home_conn.execute(f"SELECT * FROM {table[0]} LIMIT 2").fetchall()
        if sample:
            print(f"  Sample: {sample[0]}")
    except:
        pass

home_conn.close()

# Check project database
print("\n\nPROJECT DATABASE SCHEMA")
print("=" * 80)
project_db = "data/cache/wheel_cache.duckdb"
project_conn = duckdb.connect(project_db, read_only=True)

print("\nTables:")
tables = project_conn.execute("SHOW TABLES").fetchall()
for table in tables:
    print(f"\n{table[0]}:")
    schema = project_conn.execute(f"DESCRIBE {table[0]}").fetchall()
    for col in schema:
        print(f"  {col[0]}: {col[1]}")

    # Show row count
    count = project_conn.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
    print(f"  Rows: {count:,}")

project_conn.close()
