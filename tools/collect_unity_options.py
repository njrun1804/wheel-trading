#!/usr/bin/env python3
"""
Historical Unity Options Collection Script
Collects 6 months of Friday Unity option chains from Databento
Focuses on 30-60 DTE puts for wheel strategy validation
"""
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config
from src.unity_wheel.data_providers.databento.client import DatabentoClient
from src.unity_wheel.data_providers.databento.databento_storage_adapter import (
    DatabentoStorageAdapter,
)
from src.unity_wheel.storage.duckdb_cache import DuckDBCache
from src.unity_wheel.storage.storage import Storage
from src.unity_wheel.utils.logging import StructuredLogger

logger = StructuredLogger(logging.getLogger(__name__))


def get_fridays(months_back: int = 6) -> List[datetime]:
    """Generate list of Fridays going back N months.

    Args:
        months_back: Number of months to go back

    Returns:
        List of Friday dates in chronological order
    """
    fridays = []
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=months_back * 30)

    # Find first Friday
    current = start_date
    while current.weekday() != 4:  # 4 = Friday
        current += timedelta(days=1)

    # Collect all Fridays
    while current <= end_date:
        fridays.append(current.replace(hour=15, minute=0, second=0, microsecond=0))  # Market close
        current += timedelta(days=7)

    return fridays


async def collect_historical_chains(
    symbol: str = "U",
    months_back: int = 6,
    dte_range: Tuple[int, int] = (30, 60),
    batch_size: int = 5,
) -> Dict[str, int]:
    """Collect historical Unity option chains for Fridays.

    Args:
        symbol: Underlying symbol (default: U)
        months_back: How many months of history to collect
        dte_range: Target DTE range for options
        batch_size: Number of concurrent requests

    Returns:
        Statistics about collected data
    """
    logger.info(
        "starting_historical_collection",
        extra={"symbol": symbol, "months_back": months_back, "dte_range": dte_range},
    )

    # Initialize components
    config = get_config()
    db_path = os.path.expanduser("~/.wheel_trading/cache/wheel_cache.duckdb")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Create storage stack
    from pathlib import Path

    from src.unity_wheel.storage.duckdb_cache import CacheConfig

    cache_config = CacheConfig(cache_dir=Path(db_path).parent)
    cache = DuckDBCache(cache_config)
    await cache.initialize()

    storage = Storage(cache=cache)
    adapter = DatabentoStorageAdapter(storage)
    await adapter.initialize()

    # Initialize Databento client
    client = DatabentoClient()

    # Get list of Fridays
    fridays = get_fridays(months_back)
    logger.info(f"collecting_data_for_fridays", extra={"count": len(fridays)})

    # Statistics
    stats = {
        "total_fridays": len(fridays),
        "successful_fetches": 0,
        "failed_fetches": 0,
        "total_options": 0,
        "total_puts": 0,
        "target_dte_puts": 0,
        "errors": [],
    }

    # Process in batches
    for i in range(0, len(fridays), batch_size):
        batch = fridays[i : i + batch_size]

        # Process batch concurrently
        tasks = []
        for friday in batch:
            tasks.append(process_friday(client, adapter, symbol, friday, dte_range, stats))

        # Wait for batch to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Progress update
        progress = (i + len(batch)) / len(fridays) * 100
        logger.info(
            "collection_progress",
            extra={
                "progress_pct": progress,
                "fetched": stats["successful_fetches"],
                "options_stored": stats["total_options"],
            },
        )

        # Rate limit between batches
        if i + batch_size < len(fridays):
            await asyncio.sleep(2)

    # Final summary
    logger.info(
        "collection_complete",
        extra={
            "stats": stats,
            "success_rate": (
                stats["successful_fetches"] / stats["total_fridays"]
                if stats["total_fridays"] > 0
                else 0
            ),
        },
    )

    # Print summary
    print("\n" + "=" * 60)
    print("📊 Historical Unity Options Collection Summary")
    print("=" * 60)
    print(f"Total Fridays processed: {stats['total_fridays']}")
    print(f"Successful fetches: {stats['successful_fetches']}")
    print(f"Failed fetches: {stats['failed_fetches']}")
    print(f"Total options stored: {stats['total_options']}")
    print(f"Total puts stored: {stats['total_puts']}")
    print(f"Target DTE puts (30-60): {stats['target_dte_puts']}")

    if stats["errors"]:
        print(f"\n⚠️  Errors encountered: {len(stats['errors'])}")
        for error in stats["errors"][:5]:  # Show first 5 errors
            print(f"  - {error}")

    # Get storage statistics
    storage_stats = await adapter.get_storage_stats()
    print(f"\n💾 Storage Statistics:")
    print(f"  Database size: {storage_stats['db_size_mb']:.1f} MB")
    print(f"  Unique expirations: {storage_stats['unique_expirations']}")
    print(f"  Cache hit rate: {storage_stats['cache_hit_rate']:.1%}")

    await client.close()
    await cache.close()

    return stats


async def process_friday(
    client: DatabentoClient,
    adapter: DatabentoStorageAdapter,
    symbol: str,
    friday: datetime,
    dte_range: Tuple[int, int],
    stats: Dict,
) -> None:
    """Process a single Friday's option data.

    Args:
        client: Databento client
        adapter: Storage adapter
        symbol: Underlying symbol
        friday: Friday date to process
        dte_range: Target DTE range
        stats: Statistics dictionary to update
    """
    try:
        logger.info("processing_friday", extra={"date": friday.isoformat(), "symbol": symbol})

        # Find expirations in our target DTE range
        min_expiry = friday + timedelta(days=dte_range[0])
        max_expiry = friday + timedelta(days=dte_range[1])

        # Get all Friday expirations in range
        target_expirations = []
        current_exp = min_expiry

        while current_exp <= max_expiry:
            # Find next Friday
            days_to_friday = (4 - current_exp.weekday()) % 7
            if days_to_friday == 0:
                days_to_friday = 7
            next_friday = current_exp + timedelta(days=days_to_friday)

            if next_friday <= max_expiry:
                target_expirations.append(next_friday)

            current_exp = next_friday

        logger.info(
            "target_expirations",
            extra={
                "count": len(target_expirations),
                "expirations": [exp.date().isoformat() for exp in target_expirations],
            },
        )

        # Collect option chains for each expiration
        options_count = 0
        puts_count = 0
        target_puts_count = 0

        for expiration in target_expirations:
            try:
                # Get option chain
                chain = await client.get_option_chain(
                    underlying=symbol,
                    expiration=expiration,
                    timestamp=friday,  # Historical data as of this Friday
                )

                # Get definitions
                definitions = await client._get_definitions(symbol, expiration)

                # Store in database
                success = await adapter.store_option_chain(
                    chain=chain,
                    definitions=definitions,
                    enriched=False,  # We'll calculate Greeks separately if needed
                )

                if success:
                    options_count += len(chain.calls) + len(chain.puts)
                    puts_count += len(chain.puts)

                    # Count puts in target DTE range
                    dte = (expiration - friday).days
                    if dte_range[0] <= dte <= dte_range[1]:
                        target_puts_count += len(chain.puts)

                    logger.info(
                        "chain_stored",
                        extra={
                            "expiration": expiration.date().isoformat(),
                            "calls": len(chain.calls),
                            "puts": len(chain.puts),
                            "spot": float(chain.spot_price),
                        },
                    )

            except Exception as e:
                error_msg = f"Failed to fetch {symbol} expiring {expiration.date()}: {str(e)}"
                logger.error("expiration_fetch_error", extra={"error": error_msg})
                stats["errors"].append(error_msg)

        # Update statistics
        stats["successful_fetches"] += 1
        stats["total_options"] += options_count
        stats["total_puts"] += puts_count
        stats["target_dte_puts"] += target_puts_count

    except Exception as e:
        error_msg = f"Failed to process {friday.date()}: {str(e)}"
        logger.error("friday_processing_error", extra={"error": error_msg})
        stats["failed_fetches"] += 1
        stats["errors"].append(error_msg)


async def validate_stored_data(symbol: str = "U") -> None:
    """Validate and display stored historical data.

    Args:
        symbol: Symbol to validate
    """
    from pathlib import Path

    from src.unity_wheel.storage.duckdb_cache import CacheConfig

    db_path = os.path.expanduser("~/.wheel_trading/cache/wheel_cache.duckdb")
    # Create cache config with the path
    cache_config = CacheConfig(cache_dir=Path(db_path).parent)
    cache = DuckDBCache(cache_config)
    await cache.initialize()

    # Query summary statistics
    async with cache.connection() as conn:
        # Check if table exists
        table_check = conn.execute(
            """
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_name = 'databento_option_chains'
        """
        ).fetchone()

        if not table_check or table_check[0] == 0:
            print("\n❌ No historical options data found.")
            print("Run without --validate-only to collect data.")
            return

        result = conn.execute(
            """
        SELECT
            COUNT(DISTINCT DATE_TRUNC('week', timestamp)) as weeks_collected,
            COUNT(DISTINCT expiration) as unique_expirations,
            COUNT(*) as total_records,
            MIN(timestamp) as oldest_data,
            MAX(timestamp) as newest_data,
            AVG(ask - bid) as avg_spread,
            COUNT(DISTINCT strike) as unique_strikes
        FROM databento_option_chains
        WHERE symbol = ?
        AND option_type = 'PUT'
        """,
            [symbol],
        ).fetchone()

        if result:
            weeks, exps, records, oldest, newest, avg_spread, strikes = result

            print("\n📊 Stored Data Validation")
            print("=" * 50)
            print(f"Symbol: {symbol}")
            print(f"Weeks collected: {weeks}")
            print(f"Unique expirations: {exps}")
            print(f"Total put records: {records}")
            print(f"Unique strikes: {strikes}")
            print(f"Date range: {oldest} to {newest}")
            print(f"Average bid-ask spread: ${avg_spread:.3f}")

            # Sample data
            print("\n🔍 Sample Put Options (most recent):")
            samples = conn.execute(
                """
            SELECT timestamp, expiration, strike, bid, ask,
                   (ask - bid) as spread, delta, implied_volatility
            FROM databento_option_chains
            WHERE symbol = ?
            AND option_type = 'PUT'
            ORDER BY timestamp DESC, expiration, strike
            LIMIT 10
            """,
                [symbol],
            ).fetchall()

            print(
                f"{'Date':10} | {'Expiry':10} | {'Strike':7} | {'Bid':6} | {'Ask':6} | {'Spread':6} | {'Delta':6}"
            )
            print("-" * 70)

            for sample in samples:
                ts, exp, strike, bid, ask, spread, delta, iv = sample
                delta_str = f"{delta:5.3f}" if delta else "  N/A"
                print(
                    f"{ts.date()} | {exp} | ${strike:6.2f} | "
                    f"${bid:5.2f} | ${ask:5.2f} | ${spread:5.2f} | {delta_str}"
                )

    # No need to close cache as it's handled by context manager


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect historical Unity options data from Databento"
    )
    parser.add_argument(
        "--months", type=int, default=6, help="Number of months of history to collect (default: 6)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing data, don't collect new",
    )
    parser.add_argument("--symbol", default="U", help="Symbol to collect (default: U)")

    args = parser.parse_args()

    # Check for API key
    if not args.validate_only and not os.getenv("DATABENTO_API_KEY"):
        print("❌ Error: DATABENTO_API_KEY environment variable not set")
        print("Set it with: export DATABENTO_API_KEY='your-key-here'")
        sys.exit(1)

    if args.validate_only:
        print("🔍 Validating stored data...")
        await validate_stored_data(args.symbol)
    else:
        print(f"🚀 Starting historical collection for {args.symbol}")
        print(f"📅 Collecting {args.months} months of Friday option chains")
        print(f"🎯 Focusing on 30-60 DTE puts for wheel strategy\n")

        stats = await collect_historical_chains(symbol=args.symbol, months_back=args.months)

        # Validate after collection
        if stats["successful_fetches"] > 0:
            print("\n🔍 Validating collected data...")
            await validate_stored_data(args.symbol)


if __name__ == "__main__":
    asyncio.run(main())
