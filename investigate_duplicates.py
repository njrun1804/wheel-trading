#!/usr/bin/env python3
"""Investigate the duplicate data issue in Unity options."""

from datetime import datetime

import duckdb
import pandas as pd


def investigate_duplicates():
    conn = duckdb.connect("data/cache/wheel_cache.duckdb")

    print("=" * 80)
    print("DUPLICATE DATA INVESTIGATION")
    print("=" * 80)

    # 1. Find the source of duplicates
    print("\n1. Checking for exact duplicates vs different data...")

    exact_dupes = conn.execute(
        """
        SELECT COUNT(*) as exact_duplicate_rows
        FROM (
            SELECT symbol, ts_event, open, high, low, close, volume,
                   COUNT(*) as cnt
            FROM unity_options_ohlcv
            GROUP BY symbol, ts_event, open, high, low, close, volume
            HAVING COUNT(*) > 1
        )
    """
    ).fetchone()[0]

    print(f"Exact duplicate rows (all fields identical): {exact_dupes}")

    # 2. Check if it's a loading issue
    print("\n2. Analyzing duplicate patterns...")

    # Get a sample of duplicates
    dupe_analysis = conn.execute(
        """
        WITH dupe_symbols AS (
            SELECT symbol, ts_event, COUNT(*) as cnt
            FROM unity_options_ohlcv
            GROUP BY symbol, ts_event
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC
            LIMIT 1
        )
        SELECT u.*
        FROM unity_options_ohlcv u
        JOIN dupe_symbols d ON u.symbol = d.symbol AND u.ts_event = d.ts_event
        ORDER BY u.symbol, u.ts_event, u.open
    """
    ).fetchall()

    if dupe_analysis:
        print(f"\nExample duplicate set ({dupe_analysis[0][0]} on {dupe_analysis[0][1]}):")
        print("instrument_id | symbol | ts_event | open | high | low | close | volume")
        print("-" * 80)
        for row in dupe_analysis[:10]:
            print(
                f"{row[0]} | {row[1]} | {row[2]} | {row[3]:.2f} | {row[4]:.2f} | {row[5]:.2f} | {row[6]:.2f} | {row[7]}"
            )

    # 3. Check instrument_id patterns
    print("\n3. Checking instrument_id uniqueness...")

    id_check = conn.execute(
        """
        SELECT
            COUNT(DISTINCT instrument_id) as unique_ids,
            COUNT(*) as total_rows,
            COUNT(DISTINCT symbol || ts_event) as unique_symbol_dates
        FROM unity_options_ohlcv
    """
    ).fetchone()

    print(f"Unique instrument_ids: {id_check[0]:,}")
    print(f"Total rows: {id_check[1]:,}")
    print(f"Unique symbol/date combinations: {id_check[2]:,}")
    print(f"=> Each symbol/date SHOULD have 1 row, but averages {id_check[1]/id_check[2]:.1f} rows")

    # 4. Check if duplicates come from different files
    print("\n4. Analyzing instrument_id patterns for file source...")

    # Instrument IDs might indicate different source files
    id_pattern = conn.execute(
        """
        SELECT
            MIN(instrument_id) as min_id,
            MAX(instrument_id) as max_id,
            COUNT(DISTINCT instrument_id) as unique_ids
        FROM unity_options_ohlcv
    """
    ).fetchone()

    print(f"Instrument ID range: {id_pattern[0]} to {id_pattern[1]}")

    # 5. Proposed fix
    print("\n5. PROPOSED FIX")
    print("-" * 40)

    # Count how many rows would remain after deduplication
    deduped_count = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT DISTINCT symbol, ts_event, open, high, low, close, volume
            FROM unity_options_ohlcv
        )
    """
    ).fetchone()[0]

    print(f"Current rows: {id_check[1]:,}")
    print(f"After deduplication: {deduped_count:,}")
    print(f"Rows to remove: {id_check[1] - deduped_count:,}")

    # Create deduplication query
    print("\nDeduplication SQL:")
    print(
        """
-- Create cleaned table
CREATE TABLE unity_options_clean AS
SELECT DISTINCT
    MIN(instrument_id) as instrument_id,  -- Keep lowest ID
    symbol,
    ts_event,
    open,
    high,
    low,
    close,
    volume
FROM unity_options_ohlcv
GROUP BY symbol, ts_event, open, high, low, close, volume;

-- Backup original
ALTER TABLE unity_options_ohlcv RENAME TO unity_options_ohlcv_backup;

-- Rename clean table
ALTER TABLE unity_options_clean RENAME TO unity_options_ohlcv;
    """
    )

    conn.close()


if __name__ == "__main__":
    investigate_duplicates()
