#!/usr/bin/env python3
"""
Check the legacy price_history table for data quality
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.unity_wheel.storage.duckdb_cache import CacheConfig, DuckDBCache

from unity_wheel.config.unified_config import get_config
config = get_config()



async def check_price_history():
    """Check legacy price history data quality."""
    db_path = os.path.expanduser(config.storage.database_path)

    cache_config = CacheConfig(cache_dir=Path(db_path).parent)
    cache = DuckDBCache(cache_config)
    await cache.initialize()

    async with cache.connection() as conn:
        # Sample recent data
        recent = conn.execute(
            """
            SELECT date, open, high, low, close, volume
            FROM price_history
            WHERE symbol = config.trading.symbol
            ORDER BY date DESC
            LIMIT 10
        """
        ).fetchall()

        print("Recent Unity price data:")
        print("Date       | Open   | High   | Low    | Close  | Volume")
        print("-" * 60)
        for row in recent:
            date, o, h, l, c, v = row
            print(f"{date} | {o:6.2f} | {h:6.2f} | {l:6.2f} | {c:6.2f} | {v:,}")

        # Check data characteristics
        stats = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                COUNT(DISTINCT date) as unique_dates,
                SUM(CASE WHEN volume = 0 THEN 1 ELSE 0 END) as zero_volume,
                SUM(CASE WHEN high = low THEN 1 ELSE 0 END) as no_movement,
                SUM(CASE WHEN close = ROUND(close) THEN 1 ELSE 0 END) as round_prices,
                MIN(close) as min_price,
                MAX(close) as max_price,
                AVG(volume) as avg_volume
            FROM price_history
            WHERE symbol = config.trading.symbol
        """
        ).fetchone()

        print(f"\nData characteristics:")
        print(f"Total records: {stats[0]}")
        print(f"Zero volume days: {stats[2]} ({stats[2]/stats[0]*100:.1f}%)")
        print(f"No price movement: {stats[3]} ({stats[3]/stats[0]*100:.1f}%)")
        print(f"Round prices: {stats[4]} ({stats[4]/stats[0]*100:.1f}%)")
        print(f"Price range: ${stats[5]:.2f} - ${stats[6]:.2f}")
        print(f"Average volume: {stats[7]:,.0f}")


if __name__ == "__main__":
    asyncio.run(check_price_history())
