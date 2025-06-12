#!/usr/bin/env python3
"""
Test Unity options coverage with a small sample to understand the data better.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import databento as db
import duckdb
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.data_providers.databento import DatabentoClient

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


def test_recent_month():
    """Test the most recent month of Unity options data."""
    client = DatabentoClient()

    # Test just the last month
    START = "2025-05-01"
    END = "2025-06-09"

    logger.info("=" * 60)
    logger.info("TESTING UNITY OPTIONS COVERAGE - RECENT MONTH")
    logger.info("=" * 60)
    logger.info(f"Date range: {START} to {END}")

    # Test different schemas
    schemas_to_test = ["ohlcv-1d", "statistics", "tbbo", "trades"]

    for schema in schemas_to_test:
        logger.info(f"\nTesting schema: {schema}")

        try:
            # Check if schema is available
            available_schemas = client.client.metadata.list_schemas("OPRA.PILLAR")
            if schema not in available_schemas:
                logger.warning(f"  Schema {schema} not available")
                continue

            # Try to get data
            data = client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                symbols=["U.OPT"],
                stype_in="parent",
                schema=schema,
                start=START,
                end=END,
                limit=10000,  # Small sample
            )

            df = data.to_df()

            if df.empty:
                logger.info(f"  ❌ {schema}: No data")
                continue

            logger.info(f"  ✅ {schema}: {len(df):,} records")

            # Analyze coverage
            if "ts_event" in df.columns or df.index.name == "ts_event":
                if df.index.name == "ts_event":
                    unique_dates = df.index.normalize().unique()
                else:
                    unique_dates = pd.to_datetime(df["ts_event"]).dt.normalize().unique()

                logger.info(f"      Unique dates: {len(unique_dates)}")
                logger.info(f"      First date: {unique_dates.min()}")
                logger.info(f"      Last date: {unique_dates.max()}")

                if "symbol" in df.columns:
                    unique_symbols = df["symbol"].nunique()
                    logger.info(f"      Unique contracts: {unique_symbols:,}")

                    # Show sample symbols
                    sample_symbols = df["symbol"].unique()[:5]
                    logger.info(f"      Sample symbols: {sample_symbols}")

            # If this schema worked well, we found a good one
            if len(df) > 1000 and len(unique_dates) > 10:
                logger.info(f"  🎯 {schema} looks promising for full download!")

        except Exception as e:
            logger.error(f"  ❌ {schema}: Error - {e}")

    # Also test what Unity stock data looks like for comparison
    logger.info("\n" + "=" * 40)
    logger.info("UNITY STOCK DATA FOR COMPARISON")
    logger.info("=" * 40)

    db_path = Path(config.storage.database_path).expanduser()
    conn = duckdb.connect(str(db_path))

    # Check Unity stock data
    stock_data = conn.execute(
        """
        SELECT COUNT(*) as total_days, MIN(date), MAX(date)
        FROM price_history
        WHERE symbol = config.trading.symbol AND date >= '2025-05-01'
    """
    ).fetchone()

    logger.info(f"Unity stock trading days in May 2025: {stock_data[0]}")
    logger.info(f"Stock date range: {stock_data[1]} to {stock_data[2]}")

    # Check how many days Unity stock actually traded
    all_stock_data = conn.execute(
        """
        SELECT COUNT(*) as total_days, MIN(date), MAX(date)
        FROM price_history
        WHERE symbol = config.trading.symbol
    """
    ).fetchone()

    logger.info(f"\nTotal Unity stock trading days: {all_stock_data[0]}")
    logger.info(f"Full stock date range: {all_stock_data[1]} to {all_stock_data[2]}")

    conn.close()


if __name__ == "__main__":
    test_recent_month()
