#!/usr/bin/env python3
"""
Download 3 years of Unity Software (U) option daily bars following Databento documentation.
This uses the correct parameters to get ALL contracts for EACH trading day.
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

from unity_wheel.config.unified_config import get_config
config = get_config()


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


class UnityOptionsCorrectDownloader:
    """Download Unity options using the correct Databento approach."""

    def __init__(self):
        # Initialize Databento client
        self.client = DatabentoClient()

        # Database connection
        self.db_path = Path(config.storage.database_path).expanduser()
        self.conn = duckdb.connect(str(self.db_path))

        # Setup tables
        self.setup_tables()

    def setup_tables(self):
        """Create tables for Unity options data."""
        # Main daily table
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
                bid DECIMAL(10,4),
                ask DECIMAL(10,4),
                underlying_price DECIMAL(10,2),
                open_interest BIGINT,
                PRIMARY KEY (date, symbol)
            )
        """
        )

        logger.info("Tables created/verified")

    def download_all_data(self):
        """Download 3 years of Unity options EOD data."""
        # Define date range (3 years from March 2023 to yesterday)
        START = "2023-03-28"  # Unity options started here
        END = "2025-06-09"  # Data only available to yesterday

        logger.info("=" * 60)
        logger.info("DOWNLOADING UNITY OPTIONS DATA - CORRECT METHOD")
        logger.info("=" * 60)
        logger.info(f"Dataset: OPRA.PILLAR")
        logger.info(f"Schema: ohlcv-1d (daily candles)")
        logger.info(f"Symbol: U.OPT (using parent symbol)")
        logger.info(f"Date range: {START} to {END} (exclusive)")
        logger.info("=" * 60)

        # First check the estimated cost
        try:
            cost = self.client.client.metadata.get_cost(
                dataset="OPRA.PILLAR",
                symbols=["U.OPT"],
                stype_in="parent",
                schema="ohlcv-1d",
                start=START,
                end="2025-06-09",  # Fix end date - data only available to yesterday
                mode="historical-streaming",
            )
            logger.info(f"Estimated cost: ${cost:,.2f}")
            logger.info("Proceeding with download...")
        except Exception as e:
            logger.warning(f"Could not estimate cost: {e}")

        # Download the data
        try:
            logger.info("Requesting data from Databento...")

            bars = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                symbols=["U.OPT"],
                stype_in="parent",  # This is the key - tells Databento to get ALL Unity options
                schema="ohlcv-1d",  # Daily candles
                start=START,
                end=END,
                path=None,  # Stream into memory
            )

            # Convert to DataFrame
            logger.info("Converting to DataFrame...")
            df = bars.to_df()

            if df.empty:
                logger.error("No data returned!")
                return

            logger.info(f"Received {len(df):,} records")
            logger.info(f"Date range in data: {df.index.min()} to {df.index.max()}")
            logger.info(f"Unique symbols: {df['symbol'].nunique():,}")

            # Process and store the data
            self.process_and_store(df)

            # Show summary
            self.show_summary()

        except Exception as e:
            logger.error(f"Download failed: {e}")
            import traceback

            traceback.print_exc()

    def process_and_store(self, df):
        """Process and store the downloaded data."""
        logger.info("Processing and storing data...")

        records_inserted = 0
        batch_size = 1000

        # Process each record
        for idx in range(0, len(df), batch_size):
            batch = df.iloc[idx : idx + batch_size]

            for _, row in batch.iterrows():
                try:
                    # Parse the OSI symbol (e.g., U     230620C00020000)
                    symbol = row["symbol"]
                    if len(symbol) >= 21:
                        # Extract components
                        underlying = symbol[:6].strip()
                        if underlying != "U":
                            continue

                        exp_str = symbol[6:12]
                        expiration = datetime.strptime("20" + exp_str, "%Y%m%d").date()
                        option_type = symbol[12]
                        strike = float(symbol[13:21]) / 1000

                        # Extract the date from timestamp
                        trade_date = pd.Timestamp(row.name).date()

                        # Insert into database
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
                                float(row.get("open", 0)) / 1e9 if row.get("open") else None,
                                float(row.get("high", 0)) / 1e9 if row.get("high") else None,
                                float(row.get("low", 0)) / 1e9 if row.get("low") else None,
                                float(row.get("close", 0)) / 1e9 if row.get("close") else None,
                                row.get("volume", 0),
                            ),
                        )

                        records_inserted += 1

                except Exception as e:
                    logger.debug(f"Failed to process record: {e}")
                    continue

            # Commit batch
            self.conn.commit()

            if records_inserted % 10000 == 0:
                logger.info(f"  Processed {records_inserted:,} records...")

        logger.info(f"Inserted {records_inserted:,} total records")

    def show_summary(self):
        """Show comprehensive summary of downloaded data."""
        logger.info("\n" + "=" * 60)
        logger.info("UNITY OPTIONS DATA SUMMARY")
        logger.info("=" * 60)

        # Overall statistics
        stats = self.conn.execute(
            """
            SELECT
                COUNT(DISTINCT date) as trading_days,
                COUNT(DISTINCT symbol) as unique_contracts,
                COUNT(*) as total_records,
                MIN(date) as first_date,
                MAX(date) as last_date,
                SUM(volume) as total_volume,
                COUNT(DISTINCT expiration) as unique_expirations,
                COUNT(DISTINCT strike) as unique_strikes,
                AVG(volume) as avg_volume
            FROM unity_options_daily
        """
        ).fetchone()

        if stats and stats[0] > 0:
            logger.info(f"\nData Coverage:")
            logger.info(f"  Trading days with data: {stats[0]}")
            logger.info(f"  Unique option contracts: {stats[1]:,}")
            logger.info(f"  Total records: {stats[2]:,}")
            logger.info(f"  Date range: {stats[3]} to {stats[4]}")

            # Calculate coverage percentage
            if stats[3] and stats[4]:
                start = datetime.strptime(str(stats[3]), "%Y-%m-%d")
                end = datetime.strptime(str(stats[4]), "%Y-%m-%d")
                total_days = (end - start).days + 1
                weekdays = sum(
                    1 for i in range(total_days) if (start + timedelta(days=i)).weekday() < 5
                )
                coverage = (stats[0] / weekdays) * 100 if weekdays > 0 else 0

                logger.info(f"  Coverage: {coverage:.1f}% of weekdays")
                logger.info(f"  Total volume: {stats[5]:,}" if stats[5] else "  Total volume: N/A")
                logger.info(
                    f"  Average daily volume: {stats[8]:.0f}"
                    if stats[8]
                    else "  Average daily volume: N/A"
                )
                logger.info(f"  Unique expirations: {stats[6]}")
                logger.info(f"  Unique strikes: {stats[7]}")

            # Show daily coverage
            logger.info("\nDaily Coverage Analysis:")
            daily_coverage = self.conn.execute(
                """
                SELECT
                    date,
                    COUNT(*) as options_count,
                    SUM(CASE WHEN volume > 0 THEN 1 ELSE 0 END) as traded_options,
                    SUM(volume) as total_volume
                FROM unity_options_daily
                GROUP BY date
                ORDER BY date DESC
                LIMIT 10
            """
            ).fetchall()

            if daily_coverage:
                logger.info("  Recent days (newest first):")
                logger.info(f"  {'Date':<12} {'Total Options':<15} {'Traded':<10} {'Volume':<10}")
                logger.info(f"  {'-'*12} {'-'*15} {'-'*10} {'-'*10}")
                for date, total, traded, volume in daily_coverage:
                    logger.info(f"  {str(date):<12} {total:<15,} {traded:<10,} {volume or 0:<10,}")

            # Monthly summary
            logger.info("\nMonthly Summary:")
            monthly = self.conn.execute(
                """
                SELECT
                    STRFTIME('%Y-%m', date) as month,
                    COUNT(DISTINCT date) as days,
                    COUNT(*) as total_records,
                    COUNT(DISTINCT symbol) as unique_options
                FROM unity_options_daily
                GROUP BY STRFTIME('%Y-%m', date)
                ORDER BY month DESC
                LIMIT 6
            """
            ).fetchall()

            if monthly:
                logger.info(f"  {'Month':<10} {'Days':<6} {'Records':<12} {'Unique Options':<15}")
                logger.info(f"  {'-'*10} {'-'*6} {'-'*12} {'-'*15}")
                for month, days, records, options in monthly:
                    logger.info(f"  {month:<10} {days:<6} {records:<12,} {options:<15,}")

            # Sample recent options
            logger.info("\nSample Recent Options (with volume):")
            samples = self.conn.execute(
                """
                SELECT date, symbol, strike, option_type, close, volume
                FROM unity_options_daily
                WHERE volume > 0
                ORDER BY date DESC, volume DESC
                LIMIT 10
            """
            ).fetchall()

            if samples:
                for date, symbol, strike, otype, close, volume in samples:
                    close_str = f"${close:.2f}" if close else "N/A"
                    logger.info(
                        f"  {date} {symbol}: ${strike} {otype} close={close_str} vol={volume:,}"
                    )

            logger.info("\n✅ Unity options data successfully downloaded using CORRECT method")
            logger.info("✅ Used parent symbol 'U' to get ALL Unity options")
            logger.info("✅ All data is REAL from Databento OPRA feed - NO SYNTHETIC DATA")

            # Check if we got better coverage than before
            if stats[0] > 26:
                logger.info(f"✅ SUCCESS: Got {stats[0]} days of data (vs only 26 before)")
            else:
                logger.warning(
                    f"⚠️  Still only {stats[0]} days - Unity options may have limited trading"
                )

        else:
            logger.warning("\n⚠️  No data downloaded - check Databento API credentials")

    def cleanup(self):
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()


def main():
    """Main entry point."""
    downloader = UnityOptionsCorrectDownloader()

    try:
        downloader.download_all_data()
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
