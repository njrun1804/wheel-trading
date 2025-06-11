#!/usr/bin/env python3
"""Detailed investigation of data quality issues."""

from datetime import datetime

import duckdb
import numpy as np
import pandas as pd


def investigate_unity_options():
    """Deep dive into Unity options data quality."""
    conn = duckdb.connect("data/cache/wheel_cache.duckdb")

    print("=" * 80)
    print("DETAILED UNITY OPTIONS DATA INVESTIGATION")
    print("=" * 80)

    # 1. Investigate duplicates
    print("\n1. DUPLICATE ANALYSIS")
    print("-" * 40)

    # Get duplicate details
    dupes_detail = conn.execute(
        """
        WITH dupe_counts AS (
            SELECT symbol, ts_event, COUNT(*) as cnt
            FROM unity_options_ohlcv
            GROUP BY symbol, ts_event
            HAVING COUNT(*) > 1
        )
        SELECT
            d.symbol,
            d.ts_event,
            d.cnt as duplicate_count,
            u.open,
            u.high,
            u.low,
            u.close,
            u.volume
        FROM dupe_counts d
        JOIN unity_options_ohlcv u ON d.symbol = u.symbol AND d.ts_event = u.ts_event
        ORDER BY d.cnt DESC, d.symbol, d.ts_event
        LIMIT 20
    """
    ).fetchall()

    print("Sample duplicates (showing first 20):")
    for row in dupes_detail[:10]:
        print(
            f"  {row[0]} on {row[1]}: {row[2]} copies, OHLCV={row[3]:.2f}/{row[4]:.2f}/{row[5]:.2f}/{row[6]:.2f}, Vol={row[7]}"
        )

    # Check if duplicates have identical data
    identical_check = conn.execute(
        """
        WITH dupe_groups AS (
            SELECT symbol, ts_event
            FROM unity_options_ohlcv
            GROUP BY symbol, ts_event
            HAVING COUNT(*) > 1
        )
        SELECT
            COUNT(DISTINCT (symbol || ts_event || open || high || low || close || volume)) as unique_records,
            COUNT(*) as total_dupes
        FROM unity_options_ohlcv u
        JOIN dupe_groups d ON u.symbol = d.symbol AND u.ts_event = d.ts_event
    """
    ).fetchone()

    print(f"\nDuplicate data analysis:")
    print(f"  Total duplicate records: {identical_check[1]}")
    print(f"  Unique data combinations: {identical_check[0]}")
    print(
        f"  => {'IDENTICAL duplicates' if identical_check[0] < identical_check[1] else 'DIFFERENT data in duplicates'}"
    )

    # 2. Investigate synthetic patterns
    print("\n2. SYNTHETIC DATA PATTERNS")
    print("-" * 40)

    # Check for suspiciously round prices
    round_analysis = conn.execute(
        """
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN (close * 100) % 10 = 0 THEN 1 ELSE 0 END) as round_to_dime,
            SUM(CASE WHEN (close * 100) % 25 = 0 THEN 1 ELSE 0 END) as round_to_quarter,
            SUM(CASE WHEN (close * 100) % 100 = 0 THEN 1 ELSE 0 END) as round_to_dollar,
            SUM(CASE WHEN close = ROUND(close, 0) THEN 1 ELSE 0 END) as exact_dollars
        FROM unity_options_ohlcv
        WHERE close > 0
    """
    ).fetchone()

    print("Price rounding analysis:")
    print(f"  Total records: {round_analysis[0]:,}")
    print(
        f"  Round to dime (X.X0): {round_analysis[1]:,} ({round_analysis[1]/round_analysis[0]*100:.1f}%)"
    )
    print(
        f"  Round to quarter (X.25/X.50/X.75/X.00): {round_analysis[2]:,} ({round_analysis[2]/round_analysis[0]*100:.1f}%)"
    )
    print(
        f"  Round to dollar (X.00): {round_analysis[3]:,} ({round_analysis[3]/round_analysis[0]*100:.1f}%)"
    )
    print(
        f"  Exact dollars: {round_analysis[4]:,} ({round_analysis[4]/round_analysis[0]*100:.1f}%)"
    )

    # Check for mathematical relationships
    patterns = conn.execute(
        """
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN open = close THEN 1 ELSE 0 END) as open_equals_close,
            SUM(CASE WHEN high = low THEN 1 ELSE 0 END) as high_equals_low,
            SUM(CASE WHEN volume = 0 THEN 1 ELSE 0 END) as zero_volume,
            SUM(CASE WHEN volume % 100 = 0 AND volume > 0 THEN 1 ELSE 0 END) as round_volume
        FROM unity_options_ohlcv
    """
    ).fetchone()

    print("\nData pattern analysis:")
    print(f"  Open = Close: {patterns[1]:,} ({patterns[1]/patterns[0]*100:.1f}%)")
    print(f"  High = Low: {patterns[2]:,} ({patterns[2]/patterns[0]*100:.1f}%)")
    print(f"  Zero volume: {patterns[3]:,} ({patterns[3]/patterns[0]*100:.1f}%)")
    print(f"  Volume multiple of 100: {patterns[4]:,} ({patterns[4]/patterns[0]*100:.1f}%)")

    # 3. Check for realistic option pricing
    print("\n3. OPTION PRICING REALITY CHECK")
    print("-" * 40)

    # Parse strike prices and check if premiums make sense
    pricing_check = conn.execute(
        """
        WITH parsed AS (
            SELECT
                symbol,
                ts_event,
                CAST(SUBSTRING(symbol, 14, 8) AS FLOAT) / 1000 as strike,
                SUBSTRING(symbol, 13, 1) as option_type,
                close as premium,
                volume
            FROM unity_options_ohlcv
            WHERE close > 0
        )
        SELECT
            option_type,
            COUNT(*) as count,
            AVG(premium / strike) as avg_premium_ratio,
            MIN(premium / strike) as min_premium_ratio,
            MAX(premium / strike) as max_premium_ratio,
            AVG(CASE WHEN premium > strike THEN 1 ELSE 0 END) as pct_premium_exceeds_strike
        FROM parsed
        GROUP BY option_type
    """
    ).fetchall()

    print("Premium/Strike ratio analysis:")
    for row in pricing_check:
        print(
            f"  {row[0]}: Avg ratio={row[2]:.3f}, Min={row[3]:.3f}, Max={row[4]:.3f}, Premium>Strike={row[5]*100:.1f}%"
        )

    # Check for impossible option prices
    impossible = conn.execute(
        """
        WITH parsed AS (
            SELECT
                symbol,
                CAST(SUBSTRING(symbol, 14, 8) AS FLOAT) / 1000 as strike,
                SUBSTRING(symbol, 13, 1) as option_type,
                close as premium
            FROM unity_options_ohlcv
            WHERE close > 0
        )
        SELECT
            COUNT(*) as impossible_puts
        FROM parsed
        WHERE option_type = 'P' AND premium > strike
    """
    ).fetchone()[0]

    print(f"\nImpossible prices:")
    print(f"  Put premiums exceeding strike price: {impossible}")

    # 4. Volume analysis
    print("\n4. VOLUME ANALYSIS")
    print("-" * 40)

    volume_stats = conn.execute(
        """
        SELECT
            COUNT(*) as total,
            AVG(volume) as avg_volume,
            MEDIAN(volume) as median_volume,
            STDDEV(volume) as std_volume,
            MAX(volume) as max_volume,
            SUM(CASE WHEN volume = 0 THEN 1 ELSE 0 END) as zero_volume_count,
            SUM(CASE WHEN volume > 0 AND volume < 10 THEN 1 ELSE 0 END) as low_volume_count
        FROM unity_options_ohlcv
    """
    ).fetchone()

    print(f"Volume statistics:")
    print(f"  Average: {volume_stats[1]:.1f}")
    print(f"  Median: {volume_stats[2]}")
    print(f"  Std Dev: {volume_stats[3]:.1f}")
    print(f"  Max: {volume_stats[4]:,}")
    print(f"  Zero volume: {volume_stats[5]:,} ({volume_stats[5]/volume_stats[0]*100:.1f}%)")
    print(f"  Low volume (1-9): {volume_stats[6]:,} ({volume_stats[6]/volume_stats[0]*100:.1f}%)")

    # 5. Most frequently repeated values
    print("\n5. MOST REPEATED VALUES")
    print("-" * 40)

    repeated_prices = conn.execute(
        """
        SELECT close, COUNT(*) as count
        FROM unity_options_ohlcv
        WHERE close > 0
        GROUP BY close
        ORDER BY count DESC
        LIMIT 10
    """
    ).fetchall()

    print("Most repeated closing prices:")
    for price, count in repeated_prices:
        print(f"  ${price:.2f}: {count:,} times")

    # 6. Check data source
    print("\n6. DATA SOURCE VERIFICATION")
    print("-" * 40)

    # Check if this looks like real Databento data
    sample_symbols = conn.execute(
        """
        SELECT DISTINCT symbol
        FROM unity_options_ohlcv
        ORDER BY symbol
        LIMIT 5
    """
    ).fetchall()

    print("Sample symbols:")
    for sym in sample_symbols:
        print(f"  {sym[0]}")

    # Check timestamp precision
    timestamp_check = conn.execute(
        """
        SELECT
            COUNT(DISTINCT ts_event) as unique_timestamps,
            MIN(ts_event) as min_ts,
            MAX(ts_event) as max_ts
        FROM unity_options_ohlcv
    """
    ).fetchone()

    print(f"\nTimestamp analysis:")
    print(f"  Unique timestamps: {timestamp_check[0]}")
    print(f"  Date range: {timestamp_check[1]} to {timestamp_check[2]}")

    conn.close()


def check_other_data_sources():
    """Check for any other data files that might contain real data."""
    print("\n" + "=" * 80)
    print("CHECKING FOR OTHER DATA SOURCES")
    print("=" * 80)

    import glob
    import os

    # Check for parquet files
    parquet_files = glob.glob("data/**/*.parquet", recursive=True)
    print(f"\nParquet files found: {len(parquet_files)}")
    for f in parquet_files:
        size = os.path.getsize(f) / 1024 / 1024  # MB
        print(f"  {f}: {size:.1f} MB")

        # Check content
        try:
            df = pd.read_parquet(f)
            print(f"    Shape: {df.shape}")
            print(f"    Columns: {list(df.columns)[:5]}...")
        except Exception as e:
            print(f"    Error reading: {e}")

    # Check for CSV files
    csv_files = glob.glob("data/**/*.csv", recursive=True)
    print(f"\nCSV files found: {len(csv_files)}")
    for f in csv_files[:5]:  # Show first 5
        print(f"  {f}")

    # Check for other DuckDB files
    db_files = glob.glob("**/*.duckdb", recursive=True)
    print(f"\nDuckDB files found: {len(db_files)}")
    for f in db_files:
        size = os.path.getsize(f) / 1024 / 1024  # MB
        print(f"  {f}: {size:.1f} MB")


def generate_recommendations():
    """Generate actionable recommendations based on findings."""
    print("\n" + "=" * 80)
    print("DATA QUALITY RECOMMENDATIONS")
    print("=" * 80)

    print(
        """
1. CRITICAL ISSUES TO ADDRESS:
   - Remove duplicate entries in unity_options_ohlcv (135k duplicates)
   - Investigate why 61% of data shows synthetic patterns
   - Verify data source authenticity

2. DATA VALIDATION NEEDED:
   - Compare Unity options data with known market data
   - Validate option pricing against theoretical bounds
   - Cross-reference with other data providers

3. MISSING DATA:
   - No Unity stock price history (needed for Greeks/analytics)
   - Empty greeks_cache table
   - No position data

4. NEXT STEPS:
   - Run deduplication query
   - Implement data validation checks
   - Add Unity stock price data
   - Consider alternative data sources if current data is synthetic
"""
    )


if __name__ == "__main__":
    investigate_unity_options()
    check_other_data_sources()
    generate_recommendations()
