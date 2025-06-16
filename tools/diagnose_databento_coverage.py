#!/usr/bin/env python3
"""
Diagnose why Databento is only showing 14 days when Unity options trade every day.
Test different approaches to get the missing 97% of data.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.data_providers.databento import DatabentoClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def diagnose_coverage():
    """Diagnose why we're missing 97% of Unity options data."""
    client = DatabentoClient()

    # Test a specific date we KNOW had options trading
    # May 30, 2025 had 46.6M share volume (earnings spike)
    test_date = "2025-05-30"

    logger.info("=" * 60)
    logger.info("DIAGNOSING DATABENTO UNITY OPTIONS COVERAGE")
    logger.info("=" * 60)
    logger.info(f"Testing date: {test_date} (known high-volume day)")
    logger.info("Unity averages 55,000 options contracts/day per CBOE")
    logger.info("=" * 60)

    # Test 1: Check if data exists for this specific date
    logger.info("\nTest 1: Checking specific date with different schemas...")

    schemas = ["ohlcv-1d", "trades", "mbp-10", "mbp-1", "bbo-1s", "statistics"]

    for schema in schemas:
        try:
            # Check if schema exists
            available = client.client.metadata.list_schemas("OPRA.PILLAR")
            if schema not in available:
                logger.info(f"  {schema}: Not available")
                continue

            # Try to get data for just this one day
            data = client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                symbols=["U.OPT"],
                stype_in="parent",
                schema=schema,
                start=test_date,
                end=datetime.strptime(test_date, "%Y-%m-%d") + timedelta(days=1),
                limit=100,  # Just a sample
            )

            df = data.to_df()

            if df.empty:
                logger.warning(f"  {schema}: NO DATA for {test_date}")
            else:
                logger.info(f"  ✅ {schema}: {len(df)} records found!")
                if "symbol" in df.columns:
                    logger.info(
                        f"     Sample symbols: {df['symbol'].unique()[:3].tolist()}"
                    )

        except Exception as e:
            logger.error(f"  {schema}: Error - {str(e)[:100]}")

    # Test 2: Check raw symbols instead of parent
    logger.info("\nTest 2: Testing specific Unity option symbols...")

    # Try some common Unity option symbols
    test_symbols = [
        "U  250620C00025000",  # June 2025 $25 call
        "U  250620P00020000",  # June 2025 $20 put
        "U  250516C00022500",  # May 2025 $22.50 call
    ]

    for symbol in test_symbols:
        try:
            data = client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                symbols=[symbol],
                stype_in="raw_symbol",
                schema="ohlcv-1d",
                start="2025-05-01",
                end="2025-06-09",
                limit=100,
            )

            df = data.to_df()

            if df.empty:
                logger.warning(f"  {symbol}: No data")
            else:
                logger.info(f"  ✅ {symbol}: {len(df)} days of data")

        except Exception as e:
            logger.error(f"  {symbol}: Error - {str(e)[:50]}")

    # Test 3: Check metadata to understand data availability
    logger.info("\nTest 3: Checking dataset metadata...")

    try:
        # Get dataset info
        info = client.client.metadata.get_dataset("OPRA.PILLAR")
        logger.info(f"  Dataset start: {info.get('start_date')}")
        logger.info(f"  Dataset end: {info.get('end_date')}")

        # Check specific date condition
        condition = client.client.metadata.get_dataset_condition(
            dataset="OPRA.PILLAR", date=test_date
        )
        logger.info(f"  Condition for {test_date}: {condition}")

    except Exception as e:
        logger.error(f"  Metadata error: {e}")

    # Test 4: Try getting ALL data types for one day
    logger.info("\nTest 4: Getting comprehensive data for one day...")

    try:
        # Use trades schema which should have everything
        data = client.client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            symbols=["U.OPT"],
            stype_in="parent",
            schema="trades",
            start=test_date + "T14:30:00",  # 9:30 AM ET
            end=test_date + "T21:00:00",  # 4:00 PM ET
            limit=1000,
        )

        df = data.to_df()

        if not df.empty:
            logger.info(f"  ✅ Found {len(df)} trades on {test_date}")

            # Analyze the trades
            if "symbol" in df.columns:
                unique_contracts = df["symbol"].nunique()
                total_volume = len(df)  # Each trade is a record

                logger.info(f"     Unique contracts traded: {unique_contracts}")
                logger.info(f"     Total trades: {total_volume}")

                # Show most active contracts
                top_contracts = df["symbol"].value_counts().head(5)
                logger.info("     Most active contracts:")
                for symbol, count in top_contracts.items():
                    logger.info(f"       {symbol}: {count} trades")
        else:
            logger.warning(f"  No trades found for {test_date}")

    except Exception as e:
        logger.error(f"  Trades error: {e}")

    # Test 5: Check if it's a minimum volume threshold issue
    logger.info("\nTest 5: Checking if ohlcv-1d has minimum thresholds...")

    # Get a week of data and analyze
    try:
        data = client.client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            symbols=["U.OPT"],
            stype_in="parent",
            schema="ohlcv-1d",
            start="2025-05-01",
            end="2025-05-08",
            limit=10000,
        )

        df = data.to_df()

        if not df.empty and "volume" in df.columns:
            min_vol = df["volume"].min()
            avg_vol = df["volume"].mean()

            logger.info("  Volume stats in ohlcv-1d data:")
            logger.info(f"    Minimum: {min_vol}")
            logger.info(f"    Average: {avg_vol:.0f}")
            logger.info(
                f"    Inference: ohlcv-1d might exclude contracts with volume < {min_vol}"
            )

    except Exception as e:
        logger.error(f"  Volume analysis error: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSIS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    diagnose_coverage()
