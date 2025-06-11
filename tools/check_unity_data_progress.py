#!/usr/bin/env python3
"""
Check the progress of Unity options data download.
"""

from datetime import datetime, timedelta
from pathlib import Path

import duckdb


def check_progress():
    db_path = Path("~/.wheel_trading/cache/wheel_cache.duckdb").expanduser()
    conn = duckdb.connect(str(db_path))

    # Check current status
    stats = conn.execute(
        """
        SELECT
            COUNT(DISTINCT date) as trading_days,
            COUNT(DISTINCT symbol) as unique_contracts,
            COUNT(*) as total_records,
            MIN(date) as first_date,
            MAX(date) as last_date,
            SUM(volume) as total_volume
        FROM unity_options_daily
    """
    ).fetchone()

    print("UNITY OPTIONS DATA PROGRESS")
    print("=" * 60)

    if stats and stats[2] > 0:
        print(f"Trading days with data: {stats[0]}")
        print(f"Unique option contracts: {stats[1]:,}")
        print(f"Total records: {stats[2]:,}")
        print(f"Date range: {stats[3]} to {stats[4]}")
        print(f"Total volume: {stats[5]:,}" if stats[5] else "Total volume: N/A")

        # Calculate coverage
        if stats[3] and stats[4]:
            start = datetime.strptime(str(stats[3]), "%Y-%m-%d")
            end = datetime.strptime(str(stats[4]), "%Y-%m-%d")
            total_days = (end - start).days + 1
            weekdays = sum(
                1 for i in range(total_days) if (start + timedelta(days=i)).weekday() < 5
            )
            coverage = (stats[0] / weekdays) * 100 if weekdays > 0 else 0

            print(f"Coverage: {coverage:.1f}% of weekdays")

        # Latest records
        print("\nLatest 5 records:")
        latest = conn.execute(
            """
            SELECT date, strike, option_type, close, volume
            FROM unity_options_daily
            ORDER BY date DESC, strike DESC
            LIMIT 5
        """
        ).fetchall()

        for date, strike, otype, close, volume in latest:
            close_str = f"${close:.2f}" if close else "N/A"
            print(f"  {date}: ${strike} {otype} close={close_str} vol={volume}")

    else:
        print("No data yet - download may still be processing...")

    # Check if data exists in other tables
    print("\nChecking other Unity data tables:")

    # Check stock data
    stock_count = conn.execute("SELECT COUNT(*) FROM price_history WHERE symbol = 'U'").fetchone()[
        0
    ]
    print(f"Unity stock price records: {stock_count:,}")

    conn.close()


if __name__ == "__main__":
    check_progress()
