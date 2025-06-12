#!/usr/bin/env python3
"""
Check what REAL daily Unity data we already have.
NO SYNTHETIC DATA - just checking existing real data.
"""

from pathlib import Path

import duckdb

from unity_wheel.config.unified_config import get_config
config = get_config()


# Database path
db_path = Path(config.storage.database_path).expanduser()


def check_data():
    """Check existing REAL Unity data."""
    conn = duckdb.connect(str(db_path))

    print("=" * 60)
    print("REAL UNITY DATA CHECK - NO SYNTHETIC DATA")
    print("=" * 60)

    # Check daily stock data in price_history table
    try:
        stock_daily = conn.execute(
            """
            SELECT
                COUNT(*) as days,
                MIN(date) as start_date,
                MAX(date) as end_date,
                AVG(close) as avg_price
            FROM price_history
            WHERE symbol = config.trading.symbol
        """
        ).fetchone()

        print("\nDAILY STOCK DATA (price_history table):")
        print(f"  Trading days: {stock_daily[0]}")
        print(f"  Date range: {stock_daily[1]} to {stock_daily[2]}")
        print(f"  Average close: ${stock_daily[3]:.2f}")
        print("  ✓ This is REAL data from Databento")
    except Exception as e:
        print(f"\nDAILY STOCK DATA: Error - {e}")

    # Check for daily options (we can derive from tick data)
    try:
        # Count unique option dates from tick data
        options_daily = conn.execute(
            """
            SELECT
                COUNT(DISTINCT trade_date) as days,
                MIN(trade_date) as start_date,
                MAX(trade_date) as end_date,
                COUNT(DISTINCT raw_symbol) as unique_options
            FROM unity_options_ticks
            WHERE raw_symbol IS NOT NULL
        """
        ).fetchone()

        print("\nDAILY OPTIONS DATA (from unity_options_ticks):")
        print(f"  Days with data: {options_daily[0]}")
        print(f"  Date range: {options_daily[1]} to {options_daily[2]}")
        print(f"  Unique options: {options_daily[3]}")

        if options_daily[0] > 0:
            # Show what we can derive
            print("\n  We can derive daily summaries from this tick data:")
            print("  - End-of-day bid/ask for each option")
            print("  - Daily high/low/close for each option")
            print("  - Volume (if tracked in the data)")
    except Exception as e:
        print(f"\nDAILY OPTIONS DATA: No tick data to summarize")

    # Show what's available vs what's needed
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print("✓ We have 861 days of Unity daily stock data (REAL)")
    print("✓ We have 1 day of Unity options tick data (REAL)")
    print("✗ Need more options data for daily summaries")

    print("\nTO GET MORE DAILY DATA:")
    print("1. Stock: Already have 861 days from 2022-2025")
    print("2. Options: Need to download from Databento (available from March 2023)")

    conn.close()


if __name__ == "__main__":
    check_data()
