#!/usr/bin/env python3
"""
Download ONLY Unity options data from Databento (March 2023 onward).
Focused script that skips stock data and goes straight to options.
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pytz

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.data_providers.databento import DatabentoClient

from unity_wheel.config.unified_config import get_config
config = get_config()


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


class UnityOptionsOnlyDownloader:
    """Downloads Unity options data from Databento."""

    def __init__(self):
        self.client = DatabentoClient()
        self.db_path = Path(config.storage.database_path).expanduser()
        self.eastern = pytz.timezone("US/Eastern")

        # Unity options data starts March 28, 2023
        self.OPTIONS_START_DATE = datetime(2023, 3, 28).date()

        # Connect to DuckDB
        self.conn = duckdb.connect(str(self.db_path))

        logger.info(f"Initialized - will download options from {self.OPTIONS_START_DATE}")

    def download_options_for_date(self, date: datetime.date) -> int:
        """Download Unity options data for a single date."""
        market_start = datetime.combine(date, datetime.min.time()).replace(
            hour=9, minute=30, tzinfo=self.eastern
        )
        market_end = datetime.combine(date, datetime.min.time()).replace(
            hour=16, minute=0, tzinfo=self.eastern
        )

        try:
            # Download options data
            data = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="cmbp-1",  # Consolidated NBBO
                symbols=["U.OPT"],
                stype_in="parent",
                start=market_start,
                end=market_end,
            )

            # Convert to dataframe
            df = data.to_df()

            if df.empty:
                return 0

            # Process and store
            records_inserted = 0

            # Process in batches
            batch_size = 1000
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i : i + batch_size]

                for _, row in batch.iterrows():
                    # Convert prices from integer representation
                    bid_px = row.get("bid_px_01", 0) / 10000.0 if "bid_px_01" in row else None
                    ask_px = row.get("ask_px_01", 0) / 10000.0 if "ask_px_01" in row else None

                    self.conn.execute(
                        """
                        INSERT OR IGNORE INTO unity_options_ticks
                        (trade_date, ts_event, instrument_id, raw_symbol,
                         bid_px, ask_px, bid_sz, ask_sz)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            date,
                            row.get("ts_event"),
                            row.get("instrument_id", 0),
                            row.get("raw_symbol", ""),
                            bid_px,
                            ask_px,
                            row.get("bid_sz_01", 0),
                            row.get("ask_sz_01", 0),
                        ),
                    )

                    records_inserted += 1

            # Commit after each day
            self.conn.commit()

            return records_inserted

        except Exception as e:
            if "422" in str(e) or "subscription" in str(e):
                # Expected for dates before options data availability
                return 0
            else:
                logger.warning(f"Error downloading {date}: {e}")
                return 0

    def download_all_options(self):
        """Download all available Unity options data."""
        end_date = datetime.now().date() - timedelta(days=1)
        current_date = self.OPTIONS_START_DATE

        logger.info(f"Downloading Unity options from {self.OPTIONS_START_DATE} to {end_date}")

        total_records = 0
        days_with_data = 0

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                logger.info(f"Processing {current_date}...")

                records = self.download_options_for_date(current_date)

                if records > 0:
                    total_records += records
                    days_with_data += 1
                    logger.info(f"  ✓ {current_date}: {records:,} records")
                else:
                    logger.debug(f"  - {current_date}: No data")

                # Brief pause every 5 days
                if days_with_data % 5 == 0 and days_with_data > 0:
                    time.sleep(1)

            current_date += timedelta(days=1)

        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("OPTIONS DOWNLOAD COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total option records: {total_records:,}")
        logger.info(f"Days with data: {days_with_data}")

        # Verify
        self.verify_options_data()

    def verify_options_data(self):
        """Verify downloaded options data."""
        stats = self.conn.execute(
            """
            SELECT
                COUNT(*) as records,
                COUNT(DISTINCT trade_date) as days,
                COUNT(DISTINCT raw_symbol) as contracts,
                MIN(trade_date) as start_date,
                MAX(trade_date) as end_date,
                AVG(CASE WHEN bid_px > 0 AND ask_px > 0 THEN ask_px - bid_px END) as avg_spread
            FROM unity_options_ticks
        """
        ).fetchone()

        logger.info(f"\nOPTIONS DATA VERIFICATION:")
        logger.info(f"  Records: {stats[0]:,}")
        logger.info(f"  Trading days: {stats[1]}")
        logger.info(f"  Unique contracts: {stats[2]:,}")
        logger.info(f"  Date range: {stats[3]} to {stats[4]}")
        logger.info(f"  Average spread: ${stats[5]:.3f}" if stats[5] else "  Average spread: N/A")

        if stats[0] > 0:
            # Show sample symbols
            samples = self.conn.execute(
                """
                SELECT DISTINCT raw_symbol
                FROM unity_options_ticks
                WHERE raw_symbol IS NOT NULL
                ORDER BY raw_symbol
                LIMIT 5
            """
            ).fetchall()

            logger.info("\n  Sample option symbols:")
            for sym in samples:
                logger.info(f"    - {sym[0]}")

            logger.info("\n✅ Unity options data successfully downloaded from Databento")
            logger.info("✅ All data is REAL market data - NO SYNTHETIC DATA")
        else:
            logger.warning("\n⚠️  No options data downloaded - check Databento subscription")

    def cleanup(self):
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()


def main():
    """Main entry point."""
    downloader = UnityOptionsOnlyDownloader()

    try:
        downloader.download_all_options()
    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        downloader.cleanup()


if __name__ == "__main__":
    main()
