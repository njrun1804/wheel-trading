#!/usr/bin/env python3
"""
Check the final Unity options data.
"""

from datetime import datetime, timedelta
from pathlib import Path

import duckdb


def check_final_data():
    db_path = Path("~/.wheel_trading/cache/wheel_cache.duckdb").expanduser()
    conn = duckdb.connect(str(db_path))

    print("UNITY OPTIONS DATA FINAL SUMMARY")
    print("=" * 60)

    # Check what we actually have
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

        # Sample records with data
        print("\nSample Options with Volume:")
        samples = conn.execute(
            """
            SELECT date, strike, option_type, last, volume
            FROM unity_options_daily
            WHERE volume > 0
            ORDER BY date DESC, volume DESC
            LIMIT 10
        """
        ).fetchall()

        if samples:
            for date, strike, otype, last, volume in samples:
                last_str = f"${last:.2f}" if last else "N/A"
                print(f"  {date}: ${strike} {otype} last={last_str} vol={volume:,}")

        # Check if download is still running
        print("\nChecking if download is still in progress...")
        latest = conn.execute("SELECT MAX(date) FROM unity_options_daily").fetchone()[0]

        if latest:
            latest_date = datetime.strptime(str(latest), "%Y-%m-%d").date()
            expected_date = datetime.now().date() - timedelta(days=1)
            days_behind = (expected_date - latest_date).days

            if days_behind > 7:
                print(f"Latest data is from {latest} ({days_behind} days ago)")
                print("Download may still be processing older data...")
            else:
                print(f"Latest data is from {latest} - looks complete!")

    else:
        print("No data found in unity_options_daily table")

    conn.close()


if __name__ == "__main__":
    check_final_data()
