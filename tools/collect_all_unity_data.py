#!/usr/bin/env python3
"""
Comprehensive Unity Data Collection Script
Collects ALL historical stock and options data needed for backtesting
Handles Unity's price variations from $20 to $100+ dynamically

Strike Range: 70-130% of spot price
- 70-100%: Put strikes for wheel entry (30% OTM to ATM)
- 100-130%: Call strikes for exit after assignment (ATM to 30% OTM)
"""
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.loader import get_config
from src.unity_wheel.data_providers.databento.client import DatabentoClient
from src.unity_wheel.data_providers.databento.databento_storage_adapter import (
    DatabentoStorageAdapter,
)
from src.unity_wheel.storage.duckdb_cache import CacheConfig, DuckDBCache
from src.unity_wheel.storage.storage import Storage, StorageConfig
from src.unity_wheel.utils.logging import StructuredLogger

logger = StructuredLogger(logging.getLogger(__name__))


def get_monthly_expirations(start_date: datetime, end_date: datetime) -> List[datetime]:
    """Get all 3rd Friday monthly expirations between start and end dates.

    Args:
        start_date: Start of period
        end_date: End of period

    Returns:
        List of expiration dates (3rd Fridays)
    """
    expirations = []
    current = start_date.replace(day=1)

    while current <= end_date:
        # Find third Friday of the month
        first_day = current.replace(day=1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)

        # Third Friday is 14 days after first Friday
        third_friday = first_friday + timedelta(days=14)

        if start_date <= third_friday <= end_date:
            expirations.append(third_friday.replace(hour=16, minute=0, second=0, microsecond=0))

        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    return expirations


def get_strikes_for_price(
    spot_price: float, moneyness_range: Tuple[float, float] = (0.70, 1.30)
) -> List[float]:
    """Calculate strikes needed for a given spot price.

    Args:
        spot_price: Current Unity price
        moneyness_range: Min/max moneyness (default 70-130%)

    Returns:
        List of strikes at $2.50 intervals
    """
    min_strike = spot_price * moneyness_range[0]
    max_strike = spot_price * moneyness_range[1]

    # Round to nearest $2.50
    min_strike = round(min_strike / 2.5) * 2.5
    max_strike = round(max_strike / 2.5) * 2.5

    strikes = []
    current = min_strike
    while current <= max_strike:
        strikes.append(current)
        current += 2.50

    return strikes


async def collect_stock_data(
    client: DatabentoClient,
    adapter: DatabentoStorageAdapter,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """Collect all Unity stock data.

    Args:
        client: Databento client
        adapter: Storage adapter
        start_date: Start of period
        end_date: End of period

    Returns:
        DataFrame with OHLCV data
    """
    logger.info("collecting_stock_data", extra={"start": start_date.date(), "end": end_date.date()})

    try:
        # Use the historical client directly
        # Per cheat sheet: Use XNYS.PILLAR for NYSE-listed stocks
        response = client.client.timeseries.get_range(
            dataset="XNYS.PILLAR",  # Unity trades on NYSE
            schema="ohlcv-1d",  # Use lowercase schema names per cheat sheet
            start=start_date,
            end=end_date,
            symbols=["U"],
        )

        # Convert to DataFrame
        data = []
        for record in response:
            # Per cheat sheet: ts_event is u64 nanoseconds, price fields are i64 (1e-9)
            # Divide prices by 1e9 for dollar price
            data.append(
                {
                    "date": datetime.fromtimestamp(record.ts_event / 1e9, tz=timezone.utc).date(),
                    "open": float(record.open) / 1e9,
                    "high": float(record.high) / 1e9,
                    "low": float(record.low) / 1e9,
                    "close": float(record.close) / 1e9,
                    "volume": record.volume,
                }
            )

        df = pd.DataFrame(data)
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        logger.info(
            "stock_data_collected",
            extra={"days": len(df), "min_price": df["low"].min(), "max_price": df["high"].max()},
        )

        # Store in database
        await store_stock_data(adapter, df)

        return df

    except Exception as e:
        logger.error(f"Failed to collect stock data: {e}")
        raise


async def store_stock_data(adapter: DatabentoStorageAdapter, df: pd.DataFrame) -> None:
    """Store stock data in existing price_history table.

    Args:
        adapter: Storage adapter
        df: DataFrame with OHLCV data
    """
    # Use existing price_history table
    cache = adapter.storage.cache
    async with cache.connection() as conn:
        # Check existing data range
        existing = conn.execute(
            """
            SELECT MIN(date), MAX(date), COUNT(*)
            FROM price_history
            WHERE symbol = config.trading.symbol
        """
        ).fetchone()

        if existing:
            min_date, max_date, count = existing
            logger.info(f"Existing data: {count} days from {min_date} to {max_date}")

        # Only insert new data that doesn't exist
        new_records = 0
        for date, row in df.iterrows():
            # Check if date already exists
            exists = conn.execute(
                """
                SELECT 1 FROM price_history
                WHERE symbol = config.trading.symbol AND date = ?
            """,
                [date],
            ).fetchone()

            if not exists:
                conn.execute(
                    """
                    INSERT INTO price_history
                    (symbol, date, open, high, low, close, volume, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                    ["U", date, row["open"], row["high"], row["low"], row["close"], row["volume"]],
                )
                new_records += 1

        conn.commit()
        logger.info(f"Added {new_records} new days to existing {count if existing else 0} days")


async def collect_options_for_date(
    client: DatabentoClient,
    adapter: DatabentoStorageAdapter,
    trade_date: datetime,
    spot_price: float,
    expirations: List[datetime],
    stats: Dict,
) -> None:
    """Collect options data for a specific date.

    Args:
        client: Databento client
        adapter: Storage adapter
        trade_date: Date to collect data for
        spot_price: Unity price on that date
        expirations: Next monthly expirations
        stats: Statistics tracker
    """
    # Get strikes based on spot price
    strikes = get_strikes_for_price(spot_price)

    # Filter expirations to those 21-49 DTE from trade_date
    valid_expirations = []
    for exp in expirations:
        dte = (exp - trade_date).days
        if 21 <= dte <= 49:
            valid_expirations.append(exp)

    if not valid_expirations:
        logger.warning(f"No valid expirations for {trade_date.date()}")
        return

    logger.debug(
        "collecting_options_for_date",
        extra={
            "date": trade_date.date(),
            "spot": spot_price,
            "strikes": len(strikes),
            "expirations": len(valid_expirations),
        },
    )

    # Collect option chains for each expiration
    options_collected = 0

    # Check if we should skip options collection
    skip_options = os.getenv("DATABENTO_SKIP_VALIDATION", "false").lower() == "true"
    if skip_options and options_collected == 0:  # Log once
        logger.info("Skipping all options collection due to DATABENTO_SKIP_VALIDATION=true")
        return

    for expiration in valid_expirations:
        try:

            # Get option chain
            chain = await client.get_option_chain(
                underlying="U",
                expiration=expiration,
                timestamp=trade_date,  # Historical data as of this date
            )

            # Get definitions
            definitions = await client._get_definitions("U", expiration)

            # Filter to relevant strikes (70-130% of spot)
            filtered_chain = filter_chain_by_strikes(chain, definitions, strikes)

            if filtered_chain:
                # Store in database
                success = await adapter.store_option_chain(
                    chain=filtered_chain["chain"],
                    definitions=filtered_chain["definitions"],
                    enriched=False,
                )

                if success:
                    options_collected += len(filtered_chain["chain"].puts)
                    stats["options_collected"] += len(filtered_chain["chain"].puts)

        except Exception as e:
            logger.error(
                "option_collection_error",
                extra={"date": trade_date.date(), "expiration": expiration.date(), "error": str(e)},
            )
            stats["errors"].append(f"{trade_date.date()} - {expiration.date()}: {str(e)}")

    if options_collected > 0:
        stats["successful_days"] += 1
        logger.info(
            "daily_options_collected",
            extra={"date": trade_date.date(), "options": options_collected, "spot": spot_price},
        )


def filter_chain_by_strikes(chain, definitions, target_strikes: List[float]) -> Optional[Dict]:
    """Filter option chain to only include target strikes.

    Args:
        chain: Full option chain
        definitions: Instrument definitions
        target_strikes: List of strikes to keep

    Returns:
        Filtered chain and definitions
    """
    # Create strike lookup from definitions
    strike_map = {}
    for defn in definitions:
        strike_map[defn.instrument_id] = float(defn.strike_price)

    # Filter puts to target strikes
    filtered_puts = []
    filtered_defs = []

    for put in chain.puts:
        if put.instrument_id in strike_map:
            strike = strike_map[put.instrument_id]
            if strike in target_strikes or any(abs(strike - ts) < 0.01 for ts in target_strikes):
                filtered_puts.append(put)
                # Find corresponding definition
                for defn in definitions:
                    if defn.instrument_id == put.instrument_id:
                        filtered_defs.append(defn)
                        break

    if not filtered_puts:
        return None

    # Create filtered chain
    chain.puts = filtered_puts
    chain.calls = []  # We only need puts for wheel strategy

    return {"chain": chain, "definitions": filtered_defs}


async def collect_all_options_data(
    client: DatabentoClient,
    adapter: DatabentoStorageAdapter,
    stock_data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
) -> Dict:
    """Collect all options data based on dynamic stock prices.

    Args:
        client: Databento client
        adapter: Storage adapter
        stock_data: DataFrame with stock prices
        start_date: Start of options collection
        end_date: End of options collection

    Returns:
        Collection statistics
    """
    # Get all monthly expirations
    all_expirations = get_monthly_expirations(
        start_date - timedelta(days=60),  # Start earlier to catch expirations
        end_date + timedelta(days=60),  # End later to catch expirations
    )

    logger.info(
        "starting_options_collection",
        extra={
            "start": start_date.date(),
            "end": end_date.date(),
            "expirations": len(all_expirations),
            "trading_days": len(stock_data[start_date.date() : end_date.date()]),
        },
    )

    # Statistics
    stats = {
        "total_days": 0,
        "successful_days": 0,
        "options_collected": 0,
        "errors": [],
        "price_range": {"min": float("inf"), "max": float("-inf")},
    }

    # Process each trading day
    trading_days = stock_data[start_date.date() : end_date.date()].index
    batch_size = 5  # Process in batches to manage rate limits

    for i in range(0, len(trading_days), batch_size):
        batch = trading_days[i : i + batch_size]
        tasks = []

        for date in batch:
            # Get closing price for this date
            spot_price = float(stock_data.loc[date, "close"])

            # Update price range stats
            stats["price_range"]["min"] = min(stats["price_range"]["min"], spot_price)
            stats["price_range"]["max"] = max(stats["price_range"]["max"], spot_price)

            # Create task for this date
            trade_date = datetime.combine(date, datetime.min.time()).replace(
                hour=16, tzinfo=timezone.utc  # Market close
            )

            tasks.append(
                collect_options_for_date(
                    client, adapter, trade_date, spot_price, all_expirations, stats
                )
            )

            stats["total_days"] += 1

        # Execute batch
        await asyncio.gather(*tasks, return_exceptions=True)

        # Progress update
        progress = (i + len(batch)) / len(trading_days) * 100
        logger.info(
            "collection_progress",
            extra={
                "progress_pct": progress,
                "days_processed": i + len(batch),
                "options_collected": stats["options_collected"],
            },
        )

        # Rate limit between batches
        if i + batch_size < len(trading_days):
            await asyncio.sleep(2)

    return stats


async def validate_data_completeness(adapter: DatabentoStorageAdapter) -> Dict:
    """Validate that we have complete data coverage.

    Args:
        adapter: Storage adapter

    Returns:
        Validation results
    """
    cache = adapter.storage.cache
    async with cache.connection() as conn:
        # Check stock data coverage (using price_history table)
        stock_coverage = conn.execute(
            """
            SELECT
                MIN(date) as start_date,
                MAX(date) as end_date,
                COUNT(*) as total_days,
                COUNT(DISTINCT EXTRACT(YEAR FROM date)) as years_covered
            FROM price_history
            WHERE symbol = config.trading.symbol
        """
        ).fetchone()

        # Check options data coverage
        options_coverage = conn.execute(
            """
            SELECT
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date,
                COUNT(DISTINCT DATE(timestamp)) as days_with_data,
                COUNT(DISTINCT expiration) as unique_expirations,
                COUNT(*) as total_options,
                AVG(ask - bid) as avg_spread
            FROM databento_option_chains
            WHERE symbol = config.trading.symbol
            AND option_type = 'PUT'
        """
        ).fetchone()

        # Check strike coverage by price range
        strike_coverage = conn.execute(
            """
            SELECT
                CASE
                    WHEN spot_price < 30 THEN 'Low ($20-30)'
                    WHEN spot_price < 50 THEN 'Medium ($30-50)'
                    ELSE 'High ($50+)'
                END as price_range,
                COUNT(DISTINCT strike) as unique_strikes,
                MIN(strike) as min_strike,
                MAX(strike) as max_strike
            FROM databento_option_chains
            WHERE symbol = config.trading.symbol
            AND option_type = 'PUT'
            GROUP BY price_range
        """
        ).fetchall()

        # Find gaps in data
        gaps = conn.execute(
            """
            WITH date_series AS (
                SELECT generate_series(
                    (SELECT MIN(date) FROM price_history WHERE symbol = config.trading.symbol),
                    (SELECT MAX(date) FROM price_history WHERE symbol = config.trading.symbol),
                    '1 day'::interval
                )::date as date
            ),
            trading_days AS (
                SELECT date FROM date_series
                WHERE EXTRACT(DOW FROM date) NOT IN (0, 6)  -- Exclude weekends
            )
            SELECT COUNT(*) as missing_days
            FROM trading_days td
            LEFT JOIN price_history ph
                ON td.date = ph.date AND ph.symbol = config.trading.symbol
            WHERE ph.date IS NULL
        """
        ).fetchone()

    return {
        "stock_coverage": stock_coverage,
        "options_coverage": options_coverage,
        "strike_coverage": strike_coverage,
        "missing_days": gaps[0] if gaps else 0,
    }


async def main():
    """Main entry point for comprehensive data collection."""
    # Import SecretInjector to handle API key
    from src.unity_wheel.secrets.integration import SecretInjector, get_databento_api_key

    # Check if API key exists in secrets
    try:
        api_key = get_databento_api_key()
        print("✅ Found Databento API key in secrets")
    except Exception as e:
        print("❌ Error: Databento API key not found in secrets")
        print(f"   {e}")
        print("   Run: python scripts/setup-secrets.py")
        sys.exit(1)

    # Use SecretInjector to inject API key into environment
    with SecretInjector(service="databento"):
        print("🚀 Unity Comprehensive Data Collection")
        print("=" * 60)
        print("✅ Found existing price_history data (513 days)")
        print("📊 Filling gaps in stock data (Jan 2022 - May 2023)")
        print("📈 Collecting ALL options data (Jan 2023 - June 2025)")
        print("🎯 Dynamic strike selection: 70-130% of spot price")
        print("   • 70-100%: Put strikes for wheel entry")
        print("   • 100-130%: Call strikes for exit after assignment")
        print("\n📅 Current date: June 10, 2025")
        print("⏳ This will take several minutes...\n")

        # Initialize components
        config = get_config()
        db_path = os.path.expanduser(config.storage.database_path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Create storage stack
        from pathlib import Path

        cache_config = CacheConfig(cache_dir=Path(db_path).parent)
        storage = Storage(StorageConfig(cache_config=cache_config))
        await storage.initialize()

        adapter = DatabentoStorageAdapter(storage)
        await adapter.initialize()

        # Initialize Databento client
        client = DatabentoClient()

        try:
            # Step 1: Collect stock data
            print("\n📊 Step 1: Collecting Unity stock data...")
            stock_start = datetime(2022, 1, 1, tzinfo=timezone.utc)
            stock_end = datetime(2025, 6, 10, tzinfo=timezone.utc)  # TODAY

            stock_data = await collect_stock_data(client, adapter, stock_start, stock_end)

            print(f"✅ Collected {len(stock_data)} days of new stock data")

            # Get full stock data from database (existing + new)
            cache = adapter.storage.cache
            async with cache.connection() as conn:
                full_data = conn.execute(
                    """
                    SELECT date, open, high, low, close, volume
                    FROM price_history
                    WHERE symbol = config.trading.symbol
                    ORDER BY date
                """
                ).fetchall()

                # Convert to DataFrame for options collection
                stock_data = pd.DataFrame(
                    full_data, columns=["date", "open", "high", "low", "close", "volume"]
                )
                stock_data.set_index("date", inplace=True)

            print(f"   Total stock data: {len(stock_data)} days")
            print(
                f"   Price range: ${stock_data['low'].min():.2f} - ${stock_data['high'].max():.2f}"
            )
            print(f"   Average volume: {stock_data['volume'].mean():,.0f}")

            # Step 2: Collect options data
            print("\n📈 Step 2: Collecting Unity options data...")
            print("   Using dynamic strikes based on daily prices")
            print("   Filtering to 21-49 DTE monthly expirations")

            options_start = datetime(2023, 1, 1, tzinfo=timezone.utc)
            options_end = datetime(2025, 6, 10, tzinfo=timezone.utc)

            stats = await collect_all_options_data(
                client, adapter, stock_data, options_start, options_end
            )

            print(f"\n✅ Options Collection Complete!")
            print(f"   Days processed: {stats['total_days']}")
            print(f"   Successful days: {stats['successful_days']}")
            print(f"   Options collected: {stats['options_collected']:,}")
            print(
                f"   Unity price range: ${stats['price_range']['min']:.2f} - ${stats['price_range']['max']:.2f}"
            )

            if stats["errors"]:
                print(f"\n⚠️  Errors encountered: {len(stats['errors'])}")
                for error in stats["errors"][:5]:
                    print(f"   - {error}")

            # Step 3: Validate completeness
            print("\n🔍 Step 3: Validating data completeness...")
            validation = await validate_data_completeness(adapter)

            stock_cov = validation["stock_coverage"]
            options_cov = validation["options_coverage"]

            print("\n📊 Data Coverage Summary")
            print("=" * 50)
            print(f"Stock Data:")
            print(f"  Period: {stock_cov[0]} to {stock_cov[1]}")
            print(f"  Trading days: {stock_cov[2]}")
            print(f"  Years covered: {stock_cov[3]}")
            print(f"  Missing days: {validation['missing_days']}")

            print(f"\nOptions Data:")
            print(f"  Period: {options_cov[0]} to {options_cov[1]}")
            print(f"  Days with data: {options_cov[2]}")
            print(f"  Unique expirations: {options_cov[3]}")
            print(f"  Total options: {options_cov[4]:,}")
            print(f"  Average spread: ${options_cov[5]:.3f}")

            print(f"\nStrike Coverage by Price Range:")
            for price_range, strikes, min_s, max_s in validation["strike_coverage"]:
                print(f"  {price_range}: {strikes} strikes (${min_s:.2f} - ${max_s:.2f})")

            # Get storage statistics
            storage_stats = await adapter.get_storage_stats()
            print(f"\n💾 Storage Statistics:")
            print(f"  Database size: {storage_stats['db_size_mb']:.1f} MB")
            print(f"  Cache efficiency: {storage_stats['cache_hit_rate']:.1%}")

            print("\n✅ Data collection complete! Ready for backtesting.")

        except Exception as e:
            logger.error(f"Collection failed: {e}")
            print(f"\n❌ Collection failed: {e}")
            raise

        finally:
            await client.close()
            # Cache closes automatically via context manager


if __name__ == "__main__":
    asyncio.run(main())
