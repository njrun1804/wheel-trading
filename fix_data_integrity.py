#!/usr/bin/env python3
"""
Data Integrity Fixes - Critical Pre-Production Requirements
Addresses look-ahead bias, extreme strikes, and liquidity filters.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def fix_lookahead_bias(conn):
    """Remove any option data with negative DTE (look-ahead bias)."""

    print("1. FIXING LOOK-AHEAD BIAS")
    print("-" * 50)

    # First, analyze the extent of the problem
    analysis = conn.execute(
        """
        WITH option_timing AS (
            SELECT
                om.symbol,
                md.date as quote_date,
                om.expiration,
                DATEDIFF('day', md.date, om.expiration) as dte,
                md.close as option_price,
                om.strike
            FROM market_data md
            JOIN options_metadata om ON md.symbol = om.symbol
            WHERE om.underlying = 'U'
            AND md.data_type = 'option'
        )
        SELECT
            COUNT(*) as total_records,
            SUM(CASE WHEN dte < 0 THEN 1 ELSE 0 END) as future_records,
            MIN(dte) as min_dte,
            MAX(dte) as max_dte
        FROM option_timing
    """
    ).fetchone()

    total, future, min_dte, max_dte = analysis

    print(f"   Total option records: {total:,}")
    print(f"   Records with future data: {future:,} ({future/total*100:.1f}%)")
    print(f"   DTE range: {min_dte} to {max_dte} days")

    # Create cleaned table without look-ahead bias
    print("\n   Creating cleaned option data...")

    conn.execute(
        """
        CREATE OR REPLACE TABLE market_data_clean AS
        SELECT md.*
        FROM market_data md
        LEFT JOIN options_metadata om ON md.symbol = om.symbol
        WHERE
            -- Keep all non-option data
            (md.data_type != 'option')
            OR
            -- For options, only keep if quote date is before expiration
            (md.data_type = 'option'
             AND om.symbol IS NOT NULL
             AND md.date <= om.expiration
             AND DATEDIFF('day', md.date, om.expiration) >= 0)
    """
    )

    # Verify the fix
    clean_stats = conn.execute(
        """
        SELECT
            data_type,
            COUNT(*) as records
        FROM market_data_clean
        GROUP BY data_type
    """
    ).fetchall()

    print("\n   Cleaned data summary:")
    for dtype, count in clean_stats:
        print(f"     {dtype}: {count:,} records")

    # Check that no negative DTE remains
    negative_check = conn.execute(
        """
        SELECT COUNT(*) as negative_dte_count
        FROM market_data_clean md
        JOIN options_metadata om ON md.symbol = om.symbol
        WHERE md.data_type = 'option'
        AND DATEDIFF('day', md.date, om.expiration) < 0
    """
    ).fetchone()[0]

    if negative_check == 0:
        print("\n   ✅ Look-ahead bias eliminated!")
    else:
        print(f"\n   ⚠️  Warning: {negative_check} negative DTE records remain")

    return total - future  # Return clean record count


def fix_extreme_strikes(conn):
    """Remove strikes beyond reasonable trading boundaries."""

    print("\n\n2. FIXING EXTREME STRIKES")
    print("-" * 50)

    # Analyze strike distribution
    strike_analysis = conn.execute(
        """
        WITH strike_distances AS (
            SELECT
                om.strike,
                s.close as spot_price,
                ABS(om.strike - s.close) / s.close as distance_pct,
                COUNT(*) as records
            FROM options_metadata om
            JOIN market_data_clean s ON om.underlying = s.symbol
                AND s.data_type = 'stock'
                AND s.date = (
                    SELECT MAX(date)
                    FROM market_data_clean
                    WHERE symbol = om.underlying
                    AND data_type = 'stock'
                    AND date <= om.expiration - INTERVAL '7' DAY
                )
            WHERE om.underlying = 'U'
            GROUP BY om.strike, s.close
        )
        SELECT
            COUNT(DISTINCT strike) as total_strikes,
            SUM(CASE WHEN distance_pct > 3.0 THEN 1 ELSE 0 END) as extreme_strikes,
            MAX(distance_pct) as max_distance
        FROM strike_distances
    """
    ).fetchone()

    total_strikes, extreme, max_dist = strike_analysis

    print(f"   Total unique strikes: {total_strikes}")
    print(f"   Strikes >300% from spot: {extreme}")
    print(f"   Maximum distance: {max_dist:.1%}")

    # Apply strike filters based on listing rules
    # Most conservative: 20 × daily volatility or 300% distance
    print("\n   Applying strike filters...")

    conn.execute(
        """
        CREATE OR REPLACE TABLE options_metadata_clean AS
        WITH daily_stats AS (
            SELECT
                om.*,
                s.close as spot_price,
                s.volatility_20d as spot_vol,
                ABS(om.strike - s.close) / s.close as distance_pct,
                -- Maximum reasonable strike distance
                LEAST(
                    20 * s.volatility_20d * s.close / SQRT(252),  -- 20 × daily vol
                    3.0 * s.close  -- 300% of spot
                ) as max_distance
            FROM options_metadata om
            JOIN (
                SELECT
                    symbol,
                    date,
                    stock_price as close,
                    volatility_20d
                FROM backtest_features
                WHERE symbol = 'U'
            ) s ON om.underlying = s.symbol
                AND s.date = (
                    SELECT MAX(date)
                    FROM backtest_features
                    WHERE symbol = om.underlying
                    AND date <= om.expiration - INTERVAL '7' DAY
                )
        )
        SELECT
            symbol,
            underlying,
            option_type,
            strike,
            expiration
        FROM daily_stats
        WHERE ABS(strike - spot_price) <= max_distance
    """
    )

    # Verify the cleanup
    clean_strikes = conn.execute(
        """
        SELECT COUNT(DISTINCT strike) as strikes_remaining
        FROM options_metadata_clean
        WHERE underlying = 'U'
    """
    ).fetchone()[0]

    print(f"\n   Strikes after filtering: {clean_strikes}")
    print(f"   Removed: {total_strikes - clean_strikes} extreme strikes")
    print("   ✅ Extreme strikes removed!")


def add_liquidity_filters(conn):
    """Add minimum liquidity requirements for tradeable options."""

    print("\n\n3. ADDING LIQUIDITY FILTERS")
    print("-" * 50)

    # Check if we have volume/OI data
    try:
        # First check if columns exist
        columns = conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'market_data_clean'
        """
        ).fetchall()

        column_names = [col[0] for col in columns]
        has_oi = "open_interest" in column_names
        has_bid_ask = "bid" in column_names and "ask" in column_names

        if has_oi:
            liquidity_check = conn.execute(
                """
                SELECT
                    COUNT(*) as total_options,
                    SUM(CASE WHEN volume > 0 THEN 1 ELSE 0 END) as with_volume,
                    SUM(CASE WHEN open_interest > 0 THEN 1 ELSE 0 END) as with_oi
                FROM market_data_clean
                WHERE data_type = 'option'
                LIMIT 1
            """
            ).fetchone()
        else:
            # No OI data available
            liquidity_check = None
    except:
        liquidity_check = None

    if liquidity_check and liquidity_check[1] > 0:
        print(f"   Options with volume data: {liquidity_check[1]:,}/{liquidity_check[0]:,}")
        print(f"   Options with OI data: {liquidity_check[2]:,}/{liquidity_check[0]:,}")

        # Apply liquidity filters
        min_oi = 250  # Minimum open interest
        max_spread_pct = 0.04  # Maximum 4% effective spread

        print(f"\n   Applying filters:")
        print(f"   • Minimum open interest: {min_oi}")
        print(f"   • Maximum spread: {max_spread_pct:.1%}")

        conn.execute(
            f"""
            CREATE OR REPLACE TABLE tradeable_options AS
            SELECT
                md.*,
                om.strike,
                om.option_type,
                om.expiration,
                -- Calculate effective spread if bid/ask available
                CASE
                    WHEN md.bid > 0 AND md.ask > 0
                    THEN (md.ask - md.bid) / ((md.ask + md.bid) / 2)
                    ELSE NULL
                END as spread_pct
            FROM market_data_clean md
            JOIN options_metadata_clean om ON md.symbol = om.symbol
            WHERE md.data_type = 'option'
            AND (
                -- Liquidity requirements
                md.open_interest >= {min_oi}
                OR md.volume >= 100  -- Alternative: decent daily volume
            )
            AND (
                -- Spread requirements (if available)
                (md.bid IS NULL OR md.ask IS NULL)  -- No spread data
                OR (md.ask - md.bid) / ((md.ask + md.bid) / 2) <= {max_spread_pct}
            )
        """
        )

        tradeable_count = conn.execute(
            """
            SELECT COUNT(*) FROM tradeable_options
        """
        ).fetchone()[0]

        print(f"\n   Tradeable options after filters: {tradeable_count:,}")
        print("   ✅ Liquidity filters applied!")

    else:
        print("   ⚠️  No volume/OI data available - using price data only")
        print("   Recommendation: Source volume/OI from Databento for production")

        # Create placeholder without liquidity filters
        conn.execute(
            """
            CREATE OR REPLACE TABLE tradeable_options AS
            SELECT
                md.*,
                om.strike,
                om.option_type,
                om.expiration,
                NULL as spread_pct
            FROM market_data_clean md
            JOIN options_metadata_clean om ON md.symbol = om.symbol
            WHERE md.data_type = 'option'
        """
        )


def update_backtest_features(conn):
    """Update backtest_features table to use cleaned data."""

    print("\n\n4. UPDATING BACKTEST FEATURES")
    print("-" * 50)

    # Rebuild features with clean data
    conn.execute(
        """
        CREATE OR REPLACE TABLE backtest_features_clean AS
        WITH price_pairs AS (
            SELECT
                symbol,
                date,
                close,
                LAG(close) OVER (PARTITION BY symbol ORDER BY date) as prev_close
            FROM market_data_clean
            WHERE data_type = 'stock'
            AND symbol = 'U'
        )
        SELECT
            md.symbol,
            md.date,
            md.close as stock_price,
            md.volume,
            CASE
                WHEN pp.prev_close > 0
                THEN (pp.close - pp.prev_close) / pp.prev_close
                ELSE NULL
            END as returns,
            -- Calculate clean volatility metrics
            STDDEV(CASE
                WHEN pp.prev_close > 0
                THEN (pp.close - pp.prev_close) / pp.prev_close
                ELSE NULL
            END) OVER (
                PARTITION BY md.symbol
                ORDER BY md.date
                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            ) * SQRT(252) as volatility_20d,
            -- Risk-free rate and VIX (if available)
            COALESCE(bf.risk_free_rate, 0.05) as risk_free_rate,
            bf.vix
        FROM market_data_clean md
        JOIN price_pairs pp ON md.symbol = pp.symbol AND md.date = pp.date
        LEFT JOIN backtest_features bf ON md.symbol = bf.symbol AND md.date = bf.date
        WHERE md.data_type = 'stock'
        AND md.symbol = 'U'
        ORDER BY md.date
    """
    )

    # Verify the update
    stats = conn.execute(
        """
        SELECT
            COUNT(*) as total_days,
            AVG(volatility_20d) as avg_vol,
            MIN(date) as start_date,
            MAX(date) as end_date
        FROM backtest_features_clean
        WHERE symbol = 'U'
    """
    ).fetchone()

    print(f"   Clean backtest data:")
    print(f"   • Days: {stats[0]}")
    print(f"   • Average volatility: {stats[1]:.1%}")
    print(f"   • Period: {stats[2]} to {stats[3]}")
    print("   ✅ Backtest features updated!")


def generate_integrity_report(conn):
    """Generate summary report of data quality improvements."""

    print("\n\n5. DATA INTEGRITY SUMMARY")
    print("=" * 60)

    # Compare before/after statistics
    before_options = conn.execute(
        """
        SELECT COUNT(*) FROM market_data WHERE data_type = 'option'
    """
    ).fetchone()[0]

    after_options = conn.execute(
        """
        SELECT COUNT(*) FROM market_data_clean WHERE data_type = 'option'
    """
    ).fetchone()[0]

    print(f"\nOption Records:")
    print(f"   Before: {before_options:,}")
    print(f"   After:  {after_options:,}")
    print(
        f"   Removed: {before_options - after_options:,} ({(before_options - after_options)/before_options*100:.1f}%)"
    )

    # Check current data quality metrics
    quality_metrics = conn.execute(
        """
        WITH option_quality AS (
            SELECT
                om.symbol,
                md.date,
                om.strike,
                om.expiration,
                s.close as spot,
                DATEDIFF('day', md.date, om.expiration) as dte,
                ABS(om.strike - s.close) / s.close as moneyness
            FROM tradeable_options md
            JOIN options_metadata_clean om ON md.symbol = om.symbol
            JOIN market_data_clean s ON md.date = s.date
                AND s.symbol = 'U'
                AND s.data_type = 'stock'
        )
        SELECT
            MIN(dte) as min_dte,
            MAX(dte) as max_dte,
            AVG(dte) as avg_dte,
            MAX(moneyness) as max_moneyness,
            COUNT(DISTINCT DATE_TRUNC('month', date)) as months_covered
        FROM option_quality
    """
    ).fetchone()

    print(f"\nClean Data Quality Metrics:")
    print(f"   DTE range: {quality_metrics[0]} to {quality_metrics[1]} days")
    print(f"   Average DTE: {quality_metrics[2]:.0f} days")
    print(f"   Max strike distance: {quality_metrics[3]:.1%}")
    print(f"   Months with data: {quality_metrics[4]}")

    print("\n✅ DATA INTEGRITY FIXES COMPLETE!")
    print("\nNext steps:")
    print("1. Re-run validation suite A with clean data")
    print("2. Re-calculate optimal parameters")
    print("3. Verify Sharpe ratio remains > 1.2")


def main():
    """Run all data integrity fixes."""

    print("DATA INTEGRITY FIX SCRIPT")
    print("=" * 60)
    print("Fixing critical data quality issues before production...\n")

    # Connect to database
    db_path = Path("data/unified_wheel_trading.duckdb")
    if not db_path.exists():
        print("❌ Database not found!")
        print(f"   Expected at: {db_path}")
        sys.exit(1)

    conn = duckdb.connect(str(db_path))

    try:
        # 1. Fix look-ahead bias
        fix_lookahead_bias(conn)

        # 2. Fix extreme strikes
        fix_extreme_strikes(conn)

        # 3. Add liquidity filters
        add_liquidity_filters(conn)

        # 4. Update backtest features
        update_backtest_features(conn)

        # 5. Generate report
        generate_integrity_report(conn)

        # Create backup of original tables
        print("\nCreating backup of original tables...")
        conn.execute("CREATE TABLE IF NOT EXISTS market_data_original AS SELECT * FROM market_data")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS options_metadata_original AS SELECT * FROM options_metadata"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS backtest_features_original AS SELECT * FROM backtest_features"
        )

        print("✅ Backups created")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Rolling back changes...")
        conn.rollback()
        raise

    finally:
        conn.close()

    print("\n" + "=" * 60)
    print("✅ All data integrity fixes applied successfully!")
    print("\nIMPORTANT: Re-run all validation tests with clean data")


if __name__ == "__main__":
    main()
