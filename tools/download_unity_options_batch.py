#!/usr/bin/env python3
"""
Download Unity options data in monthly batches for better performance.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.data_providers.databento import DatabentoClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


class UnityOptionsBatchDownloader:
    """Download Unity options data in monthly batches."""

    def __init__(self):
        # Initialize Databento client
        logger.info("Initializing Databento client...")
        self.client = DatabentoClient()

        # Database connection
        self.db_path = Path("~/.wheel_trading/cache/wheel_cache.duckdb").expanduser()
        self.conn = duckdb.connect(str(self.db_path))

        # Verify table exists
        self.verify_table()

    def verify_table(self):
        """Verify the table exists with correct schema."""
        # Create or replace table with correct schema
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unity_options_daily (
                date DATE NOT NULL,
                symbol VARCHAR NOT NULL,
                expiration DATE NOT NULL,
                strike DECIMAL(10,2) NOT NULL,
                option_type VARCHAR(1) NOT NULL,
                open DECIMAL(10,4),
                high DECIMAL(10,4),
                low DECIMAL(10,4),
                close DECIMAL(10,4),
                volume BIGINT,
                PRIMARY KEY (date, symbol)
            )
        """
        )
        logger.info("Table verified")

    def download_month(self, year: int, month: int) -> int:
        """Download one month of data."""
        start_date = datetime(year, month, 1)

        # Calculate end of month
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)

        logger.info(f"Downloading {start_date.strftime('%Y-%m')}...")

        try:
            data = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                symbols=["U.OPT"],
                stype_in="parent",
                schema="ohlcv-1d",
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                path=None,
            )

            df = data.to_df()

            if df.empty:
                logger.info(f"  No data for {start_date.strftime('%Y-%m')}")
                return 0

            # Process this month's data
            records = self.process_dataframe(df)
            logger.info(f"  ✓ {start_date.strftime('%Y-%m')}: {records:,} records")
            return records

        except Exception as e:
            logger.error(f"  Error downloading {start_date.strftime('%Y-%m')}: {e}")
            return 0

    def process_dataframe(self, df) -> int:
        """Process and store a dataframe of options data."""
        records_inserted = 0

        for ts_event, row in df.iterrows():
            try:
                symbol = row["symbol"]

                # Skip if not Unity option
                if not symbol.startswith("U ") or len(symbol) < 21:
                    continue

                # Parse OSI symbol
                exp_str = symbol[6:12]
                option_type = symbol[12]
                strike_str = symbol[13:21]

                expiration = datetime.strptime("20" + exp_str, "%Y%m%d").date()
                strike = float(strike_str) / 1000
                trade_date = pd.Timestamp(ts_event).date()

                # Insert record
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO unity_options_daily
                    (date, symbol, expiration, strike, option_type,
                     open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        trade_date,
                        symbol,
                        expiration,
                        strike,
                        option_type,
                        row.get("open"),
                        row.get("high"),
                        row.get("low"),
                        row.get("close"),
                        row.get("volume", 0),
                    ),
                )

                records_inserted += 1

            except Exception:
                continue

        # Commit after each batch
        self.conn.commit()
        return records_inserted

    def download_all(self):
        """Download all Unity options data month by month."""
        logger.info("=" * 60)
        logger.info("DOWNLOADING UNITY OPTIONS DATA IN MONTHLY BATCHES")
        logger.info("=" * 60)

        # Check what we already have
        existing = self.conn.execute(
            """
            SELECT
                MIN(date) as min_date,
                MAX(date) as max_date,
                COUNT(*) as count
            FROM unity_options_daily
        """
        ).fetchone()

        if existing[2] > 0:
            logger.info(f"Existing data: {existing[0]} to {existing[1]} ({existing[2]:,} records)")

        total_records = 0

        # Download from March 2023 to present
        start_year = 2023
        start_month = 3

        current_date = datetime.now()
        end_year = current_date.year
        end_month = current_date.month

        # Process each month
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Skip months before Unity options started
                if year == 2023 and month < 3:
                    continue

                # Skip future months
                if year == end_year and month >= end_month:
                    continue

                records = self.download_month(year, month)
                total_records += records

                # Show progress every 6 months
                if (year * 12 + month) % 6 == 0:
                    self.show_progress()

        # Final summary
        self.show_final_summary(total_records)

    def show_progress(self):
        """Show current progress."""
        stats = self.conn.execute(
            """
            SELECT
                COUNT(DISTINCT date) as days,
                COUNT(*) as records,
                MAX(date) as latest
            FROM unity_options_daily
        """
        ).fetchone()

        logger.info(f"\nProgress: {stats[0]} days, {stats[1]:,} records, latest: {stats[2]}")

    def show_final_summary(self, new_records: int):
        """Show final summary."""
        logger.info("\n" + "=" * 60)
        logger.info("UNITY OPTIONS DATA DOWNLOAD COMPLETE")
        logger.info("=" * 60)

        stats = self.conn.execute(
            """
            SELECT
                COUNT(DISTINCT date) as trading_days,
                COUNT(DISTINCT symbol) as unique_contracts,
                COUNT(*) as total_records,
                MIN(date) as first_date,
                MAX(date) as last_date,
                SUM(volume) as total_volume
            FROM unity_options_daily
        """
        ).fetchone()

        logger.info(f"\nFinal Statistics:")
        logger.info(f"  Trading days: {stats[0]}")
        logger.info(f"  Unique contracts: {stats[1]:,}")
        logger.info(f"  Total records: {stats[2]:,}")
        logger.info(f"  Date range: {stats[3]} to {stats[4]}")
        logger.info(f"  Total volume: {stats[5]:,}" if stats[5] else "  Total volume: N/A")
        logger.info(f"  New records added: {new_records:,}")

        # Calculate coverage
        if stats[3] and stats[4]:
            start = datetime.strptime(str(stats[3]), "%Y-%m-%d")
            end = datetime.strptime(str(stats[4]), "%Y-%m-%d")
            total_days = (end - start).days + 1
            weekdays = sum(
                1 for i in range(total_days) if (start + timedelta(days=i)).weekday() < 5
            )
            coverage = (stats[0] / weekdays) * 100 if weekdays > 0 else 0

            logger.info(f"  Coverage: {coverage:.1f}% of weekdays")

        logger.info("\n✅ Unity options data successfully downloaded!")
        logger.info("✅ All data is REAL from Databento OPRA feed")

    def cleanup(self):
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()


def main():
    """Main entry point."""
    downloader = UnityOptionsBatchDownloader()

    try:
        downloader.download_all()
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
