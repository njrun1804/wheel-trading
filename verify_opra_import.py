#!/usr/bin/env python3
"""Verify OPRA data import and show sample queries."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.unity_wheel.storage.duckdb_cache import CacheConfig
from src.unity_wheel.storage.storage import Storage
from src.unity_wheel.utils import get_logger

logger = get_logger(__name__)


async def verify_import(cache_dir: str = "data/cache"):
    """Verify OPRA import and show useful queries."""

    # Initialize storage
    cache_config = CacheConfig(cache_dir=Path(cache_dir))
    storage = Storage()
    storage.cache = type(storage.cache)(cache_config)

    try:
        async with storage.cache.connection() as conn:
            # Check if options_data table exists
            tables = conn.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name = 'options_data'
            """
            ).fetchall()

            if not tables:
                print("❌ options_data table not found. Run import_opra_to_unified_db.py first.")
                return

            print("✅ OPRA Data Import Verification")
            print("=" * 50)

            # 1. Overall statistics
            stats = conn.execute(
                """
                SELECT
                    COUNT(*) as total_records,
                    COUNT(DISTINCT symbol) as symbols,
                    COUNT(DISTINCT instrument_id) as unique_options,
                    MIN(DATE(ts_event)) as earliest_date,
                    MAX(DATE(ts_event)) as latest_date
                FROM options_data
            """
            ).fetchone()

            print("\n📊 Overall Statistics:")
            print(f"   Total records: {stats[0]:,}")
            print(f"   Unique symbols: {stats[1]}")
            print(f"   Unique options: {stats[2]:,}")
            print(f"   Date range: {stats[3]} to {stats[4]}")

            # 2. Unity-specific stats
            unity_stats = conn.execute(
                """
                SELECT
                    COUNT(*) as records,
                    COUNT(DISTINCT DATE(ts_event)) as trading_days,
                    COUNT(DISTINCT expiration) as expirations,
                    MIN(strike) as min_strike,
                    MAX(strike) as max_strike
                FROM options_data
                WHERE symbol = 'U'
            """
            ).fetchone()

            print("\n🎯 Unity Options Statistics:")
            print(f"   Unity records: {unity_stats[0]:,}")
            print(f"   Trading days: {unity_stats[1]}")
            print(f"   Unique expirations: {unity_stats[2]}")
            print(f"   Strike range: ${unity_stats[3]:.2f} - ${unity_stats[4]:.2f}")

            # 3. Recent Unity price
            recent_price = conn.execute(
                """
                SELECT
                    DATE(ts_event) as date,
                    close as price
                FROM options_data
                WHERE symbol = 'U'
                    AND option_type = 'PUT'
                    AND strike = 35
                    AND expiration = (
                        SELECT MIN(expiration)
                        FROM options_data
                        WHERE symbol = 'U'
                            AND expiration > DATE('now')
                            AND option_type = 'PUT'
                    )
                ORDER BY ts_event DESC
                LIMIT 1
            """
            ).fetchone()

            if recent_price:
                print(f"\n📈 Recent Unity Data:")
                print(f"   Latest $35 put price: ${recent_price[1]:.2f} on {recent_price[0]}")

            # 4. Most liquid strikes (last 30 days)
            print("\n💧 Most Liquid Put Strikes (30-day avg volume):")
            liquid_strikes = conn.execute(
                """
                SELECT
                    strike,
                    ROUND(AVG(volume)) as avg_volume,
                    COUNT(*) as days
                FROM options_data
                WHERE symbol = 'U'
                    AND option_type = 'PUT'
                    AND DATE(ts_event) >= DATE('now', '-30 days')
                GROUP BY strike
                HAVING AVG(volume) > 50
                ORDER BY avg_volume DESC
                LIMIT 10
            """
            ).fetchall()

            for strike, vol, days in liquid_strikes:
                print(f"   ${strike:6.2f}: {vol:8,.0f} avg volume ({days} days)")

            # 5. Useful queries
            print("\n🔍 Useful Queries:")
            print("\n-- Connect to database:")
            print(f"duckdb {cache_config.cache_dir / 'wheel_cache.duckdb'}")

            print("\n-- Find optimal put strikes for wheel strategy:")
            print(
                """
SELECT
    expiration,
    strike,
    close as premium,
    volume,
    ROUND(close / strike * 365 / JULIANDAY(expiration) - JULIANDAY('now'), 3) as annualized_return
FROM options_data
WHERE symbol = 'U'
    AND option_type = 'PUT'
    AND DATE(ts_event) = (SELECT MAX(DATE(ts_event)) FROM options_data)
    AND expiration BETWEEN DATE('now', '+30 days') AND DATE('now', '+60 days')
    AND strike BETWEEN 30 AND 40
    AND volume > 100
ORDER BY annualized_return DESC;
"""
            )

            print("\n-- Analyze put skew over time:")
            print(
                """
WITH skew_data AS (
    SELECT
        DATE(ts_event) as date,
        MAX(CASE WHEN strike = 30 THEN close END) /
        MAX(CASE WHEN strike = 35 THEN close END) as put_skew_30_35
    FROM options_data
    WHERE symbol = 'U'
        AND option_type = 'PUT'
        AND strike IN (30, 35)
        AND expiration = (
            SELECT MIN(expiration)
            FROM options_data
            WHERE expiration > DATE(ts_event, '+20 days')
        )
    GROUP BY DATE(ts_event)
)
SELECT * FROM skew_data
WHERE date >= DATE('now', '-30 days')
ORDER BY date DESC;
"""
            )

            print("\n✅ Verification complete!")

    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify OPRA data import")
    parser.add_argument("--cache-dir", default="data/cache", help="Cache directory")

    args = parser.parse_args()

    asyncio.run(verify_import(args.cache_dir))
