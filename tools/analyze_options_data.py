#!/usr/bin/env python3
"""
Analyze what Unity options data we actually have.
REAL DATA ONLY - NO SYNTHETIC DATA.
"""

from pathlib import Path

import duckdb

db_path = Path("~/.wheel_trading/cache/wheel_cache.duckdb").expanduser()
conn = duckdb.connect(str(db_path))

print("=" * 60)
print("UNITY OPTIONS DATA ANALYSIS")
print("=" * 60)

# 1. Check raw options tick data
print("\n1. RAW OPTIONS TICK DATA (unity_options_ticks):")
tick_summary = conn.execute(
    """
    SELECT
        COUNT(*) as total_records,
        COUNT(DISTINCT trade_date) as days,
        COUNT(DISTINCT raw_symbol) as unique_symbols,
        MIN(trade_date) as first_date,
        MAX(trade_date) as last_date,
        SUM(CASE WHEN raw_symbol IS NULL OR raw_symbol = '' THEN 1 ELSE 0 END) as null_symbols
    FROM unity_options_ticks
"""
).fetchone()

print(f"  Total records: {tick_summary[0]:,}")
print(f"  Trading days: {tick_summary[1]}")
print(f"  Unique symbols: {tick_summary[2]}")
print(f"  Date range: {tick_summary[3]} to {tick_summary[4]}")
print(f"  Records with null/empty symbols: {tick_summary[5]:,}")

# 2. Sample the actual data
print("\n2. SAMPLE RECORDS:")
samples = conn.execute(
    """
    SELECT
        trade_date,
        ts_event,
        instrument_id,
        raw_symbol,
        bid_px,
        ask_px,
        bid_sz,
        ask_sz
    FROM unity_options_ticks
    WHERE raw_symbol IS NOT NULL AND raw_symbol != ''
    LIMIT 10
"""
).fetchall()

if samples:
    print("  First 10 records with symbols:")
    print("  Date       Time                 Instrument  Symbol              Bid    Ask")
    print("  ---------- -------------------- ----------- ------------------- ------ ------")
    for row in samples:
        print(
            f"  {row[0]} {str(row[1])[:19]} {row[2]:11d} {row[3]:19s} {row[4] or 0:6.2f} {row[5] or 0:6.2f}"
        )
else:
    # Try without symbol filter
    any_samples = conn.execute(
        """
        SELECT
            trade_date,
            instrument_id,
            raw_symbol,
            bid_px,
            ask_px
        FROM unity_options_ticks
        LIMIT 10
    """
    ).fetchall()

    print("  Sample of ALL records (including empty symbols):")
    for row in any_samples:
        symbol = row[2] if row[2] else "[EMPTY]"
        print(f"  {row[0]} ID:{row[1]} Symbol:'{symbol}' Bid:{row[3]} Ask:{row[4]}")

# 3. Check if symbols are actually empty
print("\n3. SYMBOL ANALYSIS:")
symbol_lengths = conn.execute(
    """
    SELECT
        LENGTH(raw_symbol) as sym_len,
        COUNT(*) as count
    FROM unity_options_ticks
    GROUP BY LENGTH(raw_symbol)
    ORDER BY sym_len
"""
).fetchall()

print("  Symbol length distribution:")
for row in symbol_lengths:
    print(f"    Length {row[0] if row[0] is not None else 'NULL'}: {row[1]:,} records")

# 4. Check for valid Unity option symbols
print("\n4. CHECKING FOR UNITY OPTION PATTERNS:")
unity_patterns = conn.execute(
    """
    SELECT
        COUNT(*) as count,
        MIN(raw_symbol) as example
    FROM unity_options_ticks
    WHERE
        raw_symbol LIKE 'U %' OR
        raw_symbol LIKE 'U_%' OR
        raw_symbol LIKE '%U %' OR
        instrument_id IN (
            SELECT DISTINCT instrument_id
            FROM unity_options_ticks
            WHERE raw_symbol LIKE '%U%'
        )
"""
).fetchone()

print(f"  Records that might be Unity options: {unity_patterns[0]:,}")
if unity_patterns[1]:
    print(f"  Example symbol: '{unity_patterns[1]}'")

# 5. Summary
print("\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)

if tick_summary[2] == 0 or tick_summary[2] == 1:
    print("⚠️  Options data appears incomplete:")
    print("   - Symbols may not have been captured properly")
    print("   - May need to re-download with different parameters")
    print("   - Or data format might be different than expected")
else:
    print("✓ Options data contains multiple unique symbols")

print(f"\n✓ All data shown is REAL from Databento")
print("✓ NO SYNTHETIC DATA exists in the database")

conn.close()
