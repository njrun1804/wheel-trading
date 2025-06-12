#!/usr/bin/env python3
"""
Verify Unity data in DuckDB - check what real data we have.
"""

from datetime import datetime
from pathlib import Path

import duckdb

from unity_wheel.config.unified_config import get_config
config = get_config()


# Database path
db_path = Path(config.storage.database_path).expanduser()


def verify_unity_data():
    """Check all Unity-related data in the database."""
    conn = duckdb.connect(str(db_path))

    print("=" * 60)
    print("UNITY DATA VERIFICATION REPORT")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Report time: {datetime.now()}")
    print("=" * 60)

    # Check all tables
    tables = conn.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_catalog = 'wheel_cache'
        ORDER BY table_name
    """
    ).fetchall()

    print("\nAVAILABLE TABLES:")
    for table in tables:
        print(f"  - {table[0]}")

    # Check each relevant table
    print("\n" + "=" * 60)
    print("TABLE CONTENTS:")
    print("=" * 60)

    # 1. Price history (stock data)
    try:
        stock_stats = conn.execute(
            """
            SELECT
                COUNT(*) as records,
                COUNT(DISTINCT date) as days,
                MIN(date) as start_date,
                MAX(date) as end_date,
                AVG(close) as avg_price
            FROM price_history
            WHERE symbol = config.trading.symbol
        """
        ).fetchone()

        print("\n1. UNITY STOCK DATA (price_history):")
        print(f"   Records: {stock_stats[0]:,}")
        print(f"   Trading days: {stock_stats[1]}")
        print(f"   Date range: {stock_stats[2]} to {stock_stats[3]}")
        print(
            f"   Average price: ${stock_stats[4]:.2f}"
            if stock_stats[4]
            else "   Average price: N/A"
        )
    except Exception as e:
        print(f"\n1. UNITY STOCK DATA: Error - {e}")

    # 2. Options ticks
    try:
        options_stats = conn.execute(
            """
            SELECT
                COUNT(*) as records,
                COUNT(DISTINCT trade_date) as days,
                MIN(trade_date) as start_date,
                MAX(trade_date) as end_date
            FROM options_ticks
            WHERE raw_symbol LIKE 'U %' OR raw_symbol LIKE 'U_%'
        """
        ).fetchone()

        print("\n2. UNITY OPTIONS TICKS (options_ticks):")
        print(f"   Records: {options_stats[0]:,}")
        print(f"   Trading days: {options_stats[1]}")
        print(f"   Date range: {options_stats[2]} to {options_stats[3]}")
    except Exception as e:
        print(f"\n2. UNITY OPTIONS TICKS: Error - {e}")

    # 3. Unity stock 1-minute data
    try:
        stock_1min_stats = conn.execute(
            """
            SELECT
                COUNT(*) as records,
                COUNT(DISTINCT date) as days,
                MIN(date) as start_date,
                MAX(date) as end_date
            FROM unity_stock_1min
        """
        ).fetchone()

        print("\n3. UNITY 1-MINUTE STOCK DATA (unity_stock_1min):")
        print(f"   Records: {stock_1min_stats[0]:,}")
        print(f"   Trading days: {stock_1min_stats[1]}")
        print(f"   Date range: {stock_1min_stats[2]} to {stock_1min_stats[3]}")
    except Exception as e:
        print(f"\n3. UNITY 1-MINUTE STOCK DATA: Table not found or error - {e}")

    # 4. Unity options ticks (new table)
    try:
        unity_options_stats = conn.execute(
            """
            SELECT
                COUNT(*) as records,
                COUNT(DISTINCT trade_date) as days,
                COUNT(DISTINCT raw_symbol) as unique_contracts,
                MIN(trade_date) as start_date,
                MAX(trade_date) as end_date
            FROM unity_options_ticks
        """
        ).fetchone()

        print("\n4. UNITY OPTIONS DATA (unity_options_ticks):")
        print(f"   Records: {unity_options_stats[0]:,}")
        print(f"   Trading days: {unity_options_stats[1]}")
        print(f"   Unique contracts: {unity_options_stats[2]:,}")
        print(f"   Date range: {unity_options_stats[3]} to {unity_options_stats[4]}")

        # Sample some option symbols
        if unity_options_stats[0] > 0:
            sample_symbols = conn.execute(
                """
                SELECT DISTINCT raw_symbol
                FROM unity_options_ticks
                WHERE raw_symbol IS NOT NULL
                LIMIT 5
            """
            ).fetchall()
            print("   Sample symbols:")
            for sym in sample_symbols:
                print(f"     - {sym[0]}")
    except Exception as e:
        print(f"\n4. UNITY OPTIONS DATA: Table not found or error - {e}")

    # 5. Databento option chains (old table - should be empty)
    try:
        old_chains = conn.execute(
            """
            SELECT COUNT(*) FROM databento_option_chains WHERE symbol = config.trading.symbol
        """
        ).fetchone()[0]

        print(f"\n5. OLD DATABENTO_OPTION_CHAINS: {old_chains:,} records")
        if old_chains > 0:
            print("   ⚠️  WARNING: This table should be empty (contained synthetic data)")
    except Exception as e:
        print(f"\n5. OLD DATABENTO_OPTION_CHAINS: Table not found (good)")

    # 6. Summary statistics
    print("\n" + "=" * 60)
    print("DATA QUALITY SUMMARY:")
    print("=" * 60)

    # Check for real vs synthetic data indicators
    try:
        # Check if we have high-frequency data (sign of real data)
        freq_check = conn.execute(
            """
            SELECT
                date,
                COUNT(*) as records_per_day
            FROM unity_stock_1min
            GROUP BY date
            ORDER BY records_per_day DESC
            LIMIT 5
        """
        ).fetchall()

        if freq_check and freq_check[0][1] > 300:
            print("✅ High-frequency stock data detected (REAL DATA)")
            print(f"   Max records per day: {freq_check[0][1]} on {freq_check[0][0]}")
        else:
            print("⚠️  Low-frequency stock data")
    except:
        pass

    # Check options data patterns
    try:
        # Real options data should have varying spreads
        spread_check = conn.execute(
            """
            SELECT
                COUNT(DISTINCT ROUND((ask_px - bid_px), 2)) as unique_spreads,
                AVG(ask_px - bid_px) as avg_spread,
                MIN(ask_px - bid_px) as min_spread,
                MAX(ask_px - bid_px) as max_spread
            FROM unity_options_ticks
            WHERE bid_px > 0 AND ask_px > 0
        """
        ).fetchone()

        if spread_check and spread_check[0] > 10:
            print("✅ Variable bid-ask spreads detected (REAL DATA)")
            print(f"   Unique spread values: {spread_check[0]}")
            print(f"   Spread range: ${spread_check[2]:.2f} - ${spread_check[3]:.2f}")
        else:
            print("⚠️  Limited spread variation")
    except:
        pass

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)

    # Provide recommendations based on findings
    if "unity_options_stats" in locals() and unity_options_stats[0] > 0:
        print("✅ Unity options data found in unity_options_ticks table")
        print("   Continue downloading to get complete dataset")
    else:
        print("⚠️  No Unity options data found")
        print("   Run download_unity_options_final.py to get real data")

    if "stock_1min_stats" in locals() and stock_1min_stats[0] > 50000:
        print("✅ Substantial stock data downloaded")
    else:
        print("⚠️  Limited stock data")
        print("   Continue download process")

    conn.close()


if __name__ == "__main__":
    verify_unity_data()
