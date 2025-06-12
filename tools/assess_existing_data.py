#!/usr/bin/env python3
"""
Assess existing Unity data in the database
Checks for real vs mock data, completeness, and errors
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unity_wheel.storage.duckdb_cache import CacheConfig, DuckDBCache


async def assess_database():
    """Comprehensive assessment of existing Unity data."""
    db_path = os.path.expanduser("~/.wheel_trading/cache/wheel_cache.duckdb")

    if not os.path.exists(db_path):
        print("❌ Database does not exist at:", db_path)
        return

    cache_config = CacheConfig(cache_dir=Path(db_path).parent)
    cache = DuckDBCache(cache_config)
    await cache.initialize()

    async with cache.connection() as conn:
        print("🔍 Unity Data Assessment")
        print("=" * 60)

        # 1. Check what tables exist
        tables = conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
        """
        ).fetchall()

        print("\n📊 Existing Tables:")
        for table in tables:
            print(f"  - {table[0]}")

        # 2. Assess stock data
        print("\n📈 Stock Data Assessment:")
        stock_exists = any("databento_stock_data" in t[0] for t in tables)

        if stock_exists:
            stock_stats = conn.execute(
                """
                SELECT
                    COUNT(*) as total_days,
                    MIN(date) as start_date,
                    MAX(date) as end_date,
                    MIN(close) as min_price,
                    MAX(close) as max_price,
                    AVG(close) as avg_price,
                    AVG(volume) as avg_volume
                FROM databento_stock_data
                WHERE symbol = 'U'
            """
            ).fetchone()

            if stock_stats and stock_stats[0] > 0:
                total, start, end, min_p, max_p, avg_p, avg_vol = stock_stats
                print(f"  Total days: {total}")
                print(f"  Date range: {start} to {end}")
                print(f"  Price range: ${min_p:.2f} - ${max_p:.2f} (avg: ${avg_p:.2f})")
                print(f"  Avg volume: {avg_vol:,.0f}")

                # Check for gaps
                gaps = conn.execute(
                    """
                    WITH date_series AS (
                        SELECT generate_series(
                            (SELECT MIN(date) FROM databento_stock_data WHERE symbol = 'U'),
                            (SELECT MAX(date) FROM databento_stock_data WHERE symbol = 'U'),
                            '1 day'::interval
                        )::date as date
                    ),
                    trading_days AS (
                        SELECT date FROM date_series
                        WHERE EXTRACT(DOW FROM date) NOT IN (0, 6)
                    )
                    SELECT COUNT(*)
                    FROM trading_days td
                    LEFT JOIN databento_stock_data sd
                        ON td.date = sd.date AND sd.symbol = 'U'
                    WHERE sd.date IS NULL
                """
                ).fetchone()

                print(f"  Missing trading days: {gaps[0] if gaps else 0}")

                # Check for suspicious data (mock)
                suspicious = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM databento_stock_data
                    WHERE symbol = 'U'
                    AND (
                        close = ROUND(close) OR  -- Exactly round numbers
                        volume = 0 OR            -- No volume
                        high = low OR            -- No price movement
                        close > 200 OR           -- Unity never hit $200
                        close < 10               -- Unity rarely below $10
                    )
                """
                ).fetchone()

                print(f"  Suspicious records: {suspicious[0] if suspicious else 0}")
            else:
                print("  ❌ No Unity stock data found")
        else:
            print("  ❌ Stock data table does not exist")

        # 3. Assess options data
        print("\n📊 Options Data Assessment:")
        options_exists = any("databento_option_chains" in t[0] for t in tables)

        if options_exists:
            option_stats = conn.execute(
                """
                SELECT
                    COUNT(*) as total_records,
                    COUNT(DISTINCT DATE(timestamp)) as unique_days,
                    COUNT(DISTINCT expiration) as unique_expirations,
                    COUNT(DISTINCT strike) as unique_strikes,
                    MIN(timestamp) as start_date,
                    MAX(timestamp) as end_date,
                    MIN(strike) as min_strike,
                    MAX(strike) as max_strike,
                    AVG(ask - bid) as avg_spread,
                    SUM(CASE WHEN bid = 0 OR ask = 0 THEN 1 ELSE 0 END) as zero_quotes
                FROM databento_option_chains
                WHERE symbol = 'U'
            """
            ).fetchone()

            if option_stats and option_stats[0] > 0:
                total, days, exps, strikes, start, end, min_s, max_s, spread, zeros = option_stats
                print(f"  Total option records: {total:,}")
                print(f"  Unique days: {days}")
                print(f"  Unique expirations: {exps}")
                print(f"  Unique strikes: {strikes}")
                print(f"  Date range: {start} to {end}")
                print(f"  Strike range: ${min_s:.2f} - ${max_s:.2f}")
                print(f"  Avg bid-ask spread: ${spread:.3f}")
                print(f"  Zero bid/ask quotes: {zeros} ({zeros/total*100:.1f}%)")

                # Check for mock data patterns
                mock_patterns = conn.execute(
                    """
                    SELECT
                        SUM(CASE WHEN bid = ROUND(bid, 1) AND ask = ROUND(ask, 1) THEN 1 ELSE 0 END) as round_quotes,
                        SUM(CASE WHEN volume IS NULL OR volume = 0 THEN 1 ELSE 0 END) as no_volume,
                        SUM(CASE WHEN implied_volatility IS NOT NULL THEN 1 ELSE 0 END) as has_iv,
                        SUM(CASE WHEN delta IS NOT NULL THEN 1 ELSE 0 END) as has_greeks
                    FROM databento_option_chains
                    WHERE symbol = 'U'
                """
                ).fetchone()

                round_q, no_vol, has_iv, has_greeks = mock_patterns
                print(f"\n  Data Quality Indicators:")
                print(f"    Round quotes: {round_q} ({round_q/total*100:.1f}%)")
                print(f"    No volume: {no_vol} ({no_vol/total*100:.1f}%)")
                print(f"    Has IV: {has_iv} ({has_iv/total*100:.1f}%)")
                print(f"    Has Greeks: {has_greeks} ({has_greeks/total*100:.1f}%)")

                # Check moneyness distribution
                print(f"\n  Moneyness Distribution:")
                moneyness_dist = conn.execute(
                    """
                    SELECT
                        CASE
                            WHEN moneyness < 0.7 THEN 'Far OTM (<70%)'
                            WHEN moneyness < 0.9 THEN 'OTM (70-90%)'
                            WHEN moneyness < 1.1 THEN 'ATM (90-110%)'
                            WHEN moneyness < 1.3 THEN 'ITM (110-130%)'
                            ELSE 'Far ITM (>130%)'
                        END as category,
                        COUNT(*) as count,
                        AVG(ask - bid) as avg_spread
                    FROM databento_option_chains
                    WHERE symbol = 'U' AND option_type = 'PUT'
                    GROUP BY category
                    ORDER BY
                        CASE category
                            WHEN 'Far OTM (<70%)' THEN 1
                            WHEN 'OTM (70-90%)' THEN 2
                            WHEN 'ATM (90-110%)' THEN 3
                            WHEN 'ITM (110-130%)' THEN 4
                            ELSE 5
                        END
                """
                ).fetchall()

                for category, count, spread in moneyness_dist:
                    print(f"    {category}: {count:,} records (avg spread: ${spread:.3f})")
            else:
                print("  ❌ No Unity options data found")
        else:
            print("  ❌ Options data table does not exist")

        # 4. Check price history table (old format)
        print("\n📊 Legacy Price History:")
        if any("price_history" in t[0] for t in tables):
            legacy_stats = conn.execute(
                """
                SELECT COUNT(*), MIN(date), MAX(date)
                FROM price_history
                WHERE symbol = 'U'
            """
            ).fetchone()

            if legacy_stats and legacy_stats[0] > 0:
                print(
                    f"  Found {legacy_stats[0]} records from {legacy_stats[1]} to {legacy_stats[2]}"
                )
                print("  ⚠️  Legacy format - should migrate to databento_stock_data")

        # 5. Overall assessment
        print("\n🎯 Assessment Summary:")
        if stock_exists and option_stats and option_stats[0] > 0:
            if suspicious[0] > 10 or zeros > total * 0.1:
                print("  ⚠️  Data quality issues detected - likely contains mock data")
                print("  Recommendation: OVERWRITE with fresh data")
            elif gaps[0] > 20:
                print("  ⚠️  Significant gaps in data")
                print("  Recommendation: OVERWRITE to ensure completeness")
            else:
                print("  ✅ Existing data appears valid")
                print("  Recommendation: UPDATE with missing dates only")
        else:
            print("  ❌ No meaningful data found")
            print("  Recommendation: FULL COLLECTION needed")


if __name__ == "__main__":
    asyncio.run(assess_database())
