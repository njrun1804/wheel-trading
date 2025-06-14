#!/usr/bin/env python3
import duckdb
import os

db_path = 'data/wheel_trading_optimized.duckdb'
conn = duckdb.connect(db_path)

# Enable all performance optimizations
conn.execute("SET threads = 12")
conn.execute("SET memory_limit = '19GB'")
conn.execute("SET temp_directory = '/tmp'")
conn.execute("SET enable_progress_bar = false")
conn.execute("PRAGMA enable_profiling = 'no_output'")
conn.execute("SET checkpoint_threshold = '1GB'")

# Create indexes for common queries
print("Creating optimized indexes...")

# Index for active options queries
conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_active_options_symbol_type 
    ON active_options(symbol, option_type)
""")

# Index for market data queries
conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_market_data_symbol_date 
    ON market_data(symbol, date)
""")

# Index for options data queries
conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_options_data_composite 
    ON options_data(symbol, strike, expiration, option_type)
""")

# Analyze tables for query optimization
print("Analyzing tables...")
for table in conn.execute("SHOW TABLES").fetchall():
    conn.execute(f"ANALYZE {table[0]}")

# Compact database
print("Compacting database...")
conn.execute("VACUUM")
conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

# Show final stats
size_mb = os.path.getsize(db_path) / (1024 * 1024)
print(f"\nDatabase optimized!")
print(f"Size: {size_mb:.1f} MB")
print(f"Tables: {len(conn.execute('SHOW TABLES').fetchall())}")

conn.close()
print("✅ DuckDB optimization complete!")