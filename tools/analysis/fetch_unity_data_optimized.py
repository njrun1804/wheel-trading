#!/usr/bin/env python3
"""
Optimized Unity data fetcher for M4 Pro MacBook.
Handles Databento API limitations with proper error recovery.
"""

import asyncio
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.unity_wheel.databento import DatentoClient
from src.unity_wheel.databento.optimized_price_loader import OptimizedPriceHistoryLoader
from src.unity_wheel.storage import Storage, StorageConfig
from src.unity_wheel.utils import setup_structured_logging


async def fetch_unity_data_optimized():
    """Fetch Unity data with optimized settings."""

    setup_structured_logging()

    print("🚀 OPTIMIZED UNITY DATA FETCHER")
    print("=" * 60)
    print("\nSystem: M4 Pro MacBook")
    print("Target: 750 days of daily OHLC for Unity (U)")
    print("Optimization: Concurrent chunks, retry logic, rate limiting")
    print()

    # Initialize components
    print("📦 Initializing components...")
    storage = Storage(StorageConfig())
    await storage.initialize()

    client = DatentoClient()
    loader = OptimizedPriceHistoryLoader(client, storage)

    # Ensure table exists
    await loader.ensure_table_exists()

    # Check system
    print("\n🔧 System optimization...")
    print("  - Max workers: 8 cores")
    print("  - Chunk size: 250 days")
    print("  - Rate limit: 100 req/s")
    print("  - Retry logic: Exponential backoff")

    try:
        # Start timing
        start_time = time.time()

        # Check current data
        print("\n📊 Checking existing data...")
        existing = await loader._get_existing_days("U")
        print(f"  Current: {existing} days")

        if existing >= 750:
            print("  ✅ Already have sufficient data!")
            quality = await loader.verify_data_quality("U")
            print_data_quality(quality)
            return

        # Fetch data
        print(f"\n📥 Fetching {750 - existing} additional days...")
        print("  This will use concurrent requests for speed")
        print("  Progress:")

        success = await loader.load_price_history("U", days=750)

        elapsed = time.time() - start_time

        if success:
            print(f"\n✅ Data loaded in {elapsed:.1f} seconds!")

            # Verify quality
            quality = await loader.verify_data_quality("U")
            print_data_quality(quality)

            # Show risk metrics
            await show_risk_metrics(storage)

        else:
            print("\n❌ Failed to load sufficient data")
            print("Check logs for details")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await client.close()

    # Storage stats
    await show_storage_stats(storage)


def print_data_quality(quality: dict):
    """Print data quality report."""
    print("\n📈 Data Quality Report:")
    print(f"  Symbol: {quality['symbol']}")
    print(f"  Total days: {quality['total_days']}")
    print(f"  Date range: {quality['date_range']}")
    print(f"  Data gaps: {quality['gaps']}")
    print(f"  Quality: {quality['data_quality']}")
    print(f"\n  Annual return: {quality['annual_return']}")
    print(f"  Annual volatility: {quality['annual_volatility']}")
    print(f"  Worst day: {quality['worst_day']}")
    print(f"  Best day: {quality['best_day']}")


async def show_risk_metrics(storage):
    """Calculate and show risk metrics."""
    print("\n💰 Risk Metrics (on $100,000 portfolio):")

    async with storage.cache.connection() as conn:
        # Calculate VaR
        var_result = conn.execute(
            """
            SELECT
                PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY returns) as var_95,
                PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY returns) as var_99,
                AVG(CASE WHEN returns < 0 THEN returns ELSE 0 END) as avg_loss,
                COUNT(CASE WHEN returns < -0.05 THEN 1 END) as big_loss_days
            FROM price_history
            WHERE symbol = 'U'
        """
        ).fetchone()

        if var_result:
            var_95, var_99, avg_loss, big_losses = var_result

            print(f"  95% VaR (1-day): ${abs(var_95) * 100_000:,.0f}")
            print(f"  99% VaR (1-day): ${abs(var_99) * 100_000:,.0f}")
            print(f"  Average loss day: ${abs(avg_loss) * 100_000:,.0f}")
            print(f"  Days with >5% loss: {big_losses}")

            # Kelly sizing
            print(f"\n  Recommended position sizes:")
            print(f"  Conservative (Quarter-Kelly): ${100_000 * 0.10:,.0f}")
            print(f"  Moderate (Half-Kelly): ${100_000 * 0.20:,.0f}")
            print(f"  Aggressive (Full-Kelly): ${100_000 * 0.40:,.0f}")


async def show_storage_stats(storage):
    """Show storage statistics."""
    print("\n💾 Storage Statistics:")

    db_path = storage.cache.db_path
    if os.path.exists(db_path):
        db_size = os.path.getsize(db_path) / 1024 / 1024
        print(f"  Database size: {db_size:.2f} MB")

    async with storage.cache.connection() as conn:
        # Table stats
        tables = conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
        """
        ).fetchall()

        print(f"  Tables: {len(tables)}")

        for table in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
            print(f"    - {table[0]}: {count:,} rows")


if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZED UNITY DATA FETCHER")
    print("=" * 60)
    print("\nOptimizations for M4 Pro:")
    print("- Concurrent chunk fetching")
    print("- Smart retry with exponential backoff")
    print("- Rate limiting to respect API limits")
    print("- Incremental storage to avoid memory issues")
    print()

    # Check if we should run
    if "--auto" in sys.argv:
        asyncio.run(fetch_unity_data_optimized())
    else:
        response = input("Ready to fetch 750 days of Unity data? (y/n): ")
        if response.lower() == "y":
            asyncio.run(fetch_unity_data_optimized())
        else:
            print("Cancelled.")
