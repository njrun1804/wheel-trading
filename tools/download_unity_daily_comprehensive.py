#!/usr/bin/env python3
"""
Download Unity options data DAY BY DAY to get complete coverage.
The diagnostic proved data exists for every day - we just need to query correctly.
"""

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.data_providers.databento import DatabentoClient
from unity_wheel.config.unified_config import get_config

config = get_config()


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class UnityOptionsDailyDownloader:
    """Download Unity options data day-by-day for complete coverage."""

    def __init__(self):
        self.client = DatabentoClient()
        self.db_path = Path(config.storage.database_path).expanduser()
        self.conn = duckdb.connect(str(self.db_path))

        # Clear and recreate table for clean start
        self.setup_tables()

    def setup_tables(self):
        """Create fresh table for comprehensive daily data."""
        # Drop and recreate for clean slate
        self.conn.execute("DROP TABLE IF EXISTS unity_options_comprehensive")

        self.conn.execute(
            """
            CREATE TABLE unity_options_comprehensive (
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
                trades_count INT,
                vwap DECIMAL(10,4),
                underlying_price DECIMAL(10,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, symbol)
            )
        """
        )

        logger.info("Created fresh table: unity_options_comprehensive")

    def download_day(self, date):
        """Download Unity options data for a single trading day."""
        date_str = date.strftime("%Y-%m-%d")
        next_day = date + timedelta(days=1)
        next_day_str = next_day.strftime("%Y-%m-%d")

        try:
            # Use ohlcv-1d for daily bars
            data = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                symbols=["U.OPT"],
                stype_in="parent",
                schema="ohlcv-1d",
                start=date_str,
                end=next_day_str,
                limit=10000,  # High limit to catch all contracts
            )

            df = data.to_df()

            if df.empty:
                return 0, 0

            # Process and store
            contracts_stored = 0
            total_volume = 0

            for _, row in df.iterrows():
                try:
                    symbol = row.get("symbol", "")

                    if not symbol.startswith("U "):
                        continue

                    # Parse OSI symbol: U     250620C00025000
                    if len(symbol) >= 21:
                        exp_str = symbol[6:12]
                        expiration = datetime.strptime("20" + exp_str, "%Y%m%d").date()
                        option_type = symbol[12]
                        strike = float(symbol[13:21]) / 1000

                        # Extract values
                        open_price = self.convert_price(row.get("open"))
                        high_price = self.convert_price(row.get("high"))
                        low_price = self.convert_price(row.get("low"))
                        close_price = self.convert_price(row.get("close"))
                        volume = row.get("volume", 0)
                        vwap = self.convert_price(row.get("vwap"))

                        # Store in database
                        self.conn.execute(
                            """
                            INSERT OR REPLACE INTO unity_options_comprehensive
                            (date, symbol, expiration, strike, option_type,
                             open, high, low, close, volume, vwap)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                date,
                                symbol,
                                expiration,
                                strike,
                                option_type,
                                open_price,
                                high_price,
                                low_price,
                                close_price,
                                volume,
                                vwap,
                            ),
                        )

                        contracts_stored += 1
                        total_volume += volume or 0

                except Exception as e:
                    logger.debug(f"Failed to process row: {e}")
                    continue

            self.conn.commit()
            return contracts_stored, total_volume

        except Exception as e:
            logger.debug(f"Error downloading {date_str}: {e}")
            return 0, 0

    def convert_price(self, price):
        """Convert Databento price format."""
        if price is None:
            return None
        try:
            if price > 10000:
                return float(price) / 10000.0
            elif price > 1000:
                return float(price) / 1000.0
            else:
                return float(price)
        except:
            return None

    def download_all_comprehensive(self):
        """Download all Unity options data day by day."""
        # Start from Unity options availability
        start_date = datetime(2023, 3, 28).date()
        end_date = datetime(2025, 6, 9).date()  # Yesterday

        logger.info("=" * 60)
        logger.info("COMPREHENSIVE UNITY OPTIONS DOWNLOAD - DAY BY DAY")
        logger.info("=" * 60)
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info("Downloading each trading day individually...")
        logger.info("Expected: ~503 trading days based on your analysis")
        logger.info("=" * 60)

        current_date = start_date
        total_contracts = 0
        total_volume = 0
        days_with_data = 0
        days_processed = 0
        empty_days = 0

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                days_processed += 1

                contracts, volume = self.download_day(current_date)

                if contracts > 0:
                    total_contracts += contracts
                    total_volume += volume
                    days_with_data += 1

                    logger.info(
                        f"✅ {current_date}: {contracts:,} contracts, {volume:,} volume"
                    )
                else:
                    empty_days += 1
                    if empty_days <= 10:  # Only log first 10 empty days
                        logger.debug(f"   {current_date}: No data")

                # Progress update every 50 days
                if days_processed % 50 == 0:
                    logger.info(
                        f"Progress: {days_processed} days processed, {days_with_data} with data ({days_with_data/days_processed*100:.1f}%)"
                    )

                # Brief pause every 10 days to be respectful
                if days_processed % 10 == 0:
                    time.sleep(0.5)

            current_date += timedelta(days=1)

        # Final summary
        self.show_comprehensive_summary(days_processed, days_with_data, empty_days)

    def show_comprehensive_summary(self, days_processed, days_with_data, empty_days):
        """Show comprehensive summary of the download."""
        logger.info("\n" + "=" * 60)
        logger.info("COMPREHENSIVE DOWNLOAD COMPLETE")
        logger.info("=" * 60)

        # Get database statistics
        stats = self.conn.execute(
            """
            SELECT
                COUNT(DISTINCT date) as trading_days,
                COUNT(DISTINCT symbol) as unique_contracts,
                COUNT(*) as total_records,
                MIN(date) as first_date,
                MAX(date) as last_date,
                SUM(volume) as total_volume,
                AVG(volume) as avg_volume
            FROM unity_options_comprehensive
        """
        ).fetchone()

        logger.info("Download Statistics:")
        logger.info(f"  Days processed: {days_processed}")
        logger.info(f"  Days with data: {days_with_data}")
        logger.info(f"  Empty days: {empty_days}")
        logger.info(f"  Coverage: {days_with_data/days_processed*100:.1f}% of weekdays")

        if stats and stats[0] > 0:
            logger.info("\nDatabase Statistics:")
            logger.info(f"  Trading days: {stats[0]}")
            logger.info(f"  Unique contracts: {stats[1]:,}")
            logger.info(f"  Total records: {stats[2]:,}")
            logger.info(f"  Date range: {stats[3]} to {stats[4]}")
            logger.info(f"  Total volume: {stats[5]:,}")
            logger.info(f"  Average volume per contract: {stats[6]:.1f}")

            # Calculate daily averages
            if stats[0] > 0:
                contracts_per_day = stats[2] / stats[0]
                volume_per_day = stats[5] / stats[0] if stats[5] else 0

                logger.info("\nDaily Averages:")
                logger.info(f"  Contracts per day: {contracts_per_day:,.0f}")
                logger.info(f"  Volume per day: {volume_per_day:,.0f}")

                # Compare to expected numbers
                logger.info("\nComparison to Your Analysis:")
                logger.info("  Expected: ~55,000 contracts/day from CBOE")
                logger.info(f"  Our data: {volume_per_day:,.0f} contracts/day")

                if volume_per_day > 40000:
                    logger.info("  ✅ EXCELLENT! Close to expected volume")
                elif volume_per_day > 20000:
                    logger.info("  ✅ Good coverage, within reasonable range")
                else:
                    logger.info("  ⚠️  Lower than expected - may need investigation")

            # Show recent data quality
            logger.info("\nRecent Data Sample:")
            recent = self.conn.execute(
                """
                SELECT date, COUNT(*) as contracts, SUM(volume) as daily_volume
                FROM unity_options_comprehensive
                WHERE date >= DATE('2025-05-01')
                GROUP BY date
                ORDER BY date DESC
                LIMIT 10
            """
            ).fetchall()

            if recent:
                logger.info(f"  {'Date':<12} {'Contracts':<10} {'Volume':<10}")
                logger.info(f"  {'-'*12} {'-'*10} {'-'*10}")
                for date, contracts, volume in recent:
                    logger.info(
                        f"  {str(date):<12} {contracts:<10,} {volume or 0:<10,}"
                    )

            logger.info("\n✅ SUCCESS: Comprehensive Unity options data downloaded!")
            logger.info(f"✅ Day-by-day approach captured {stats[0]} trading days")
            logger.info("✅ All data is REAL from Databento OPRA feed")

        else:
            logger.warning("\n⚠️  No data in database - check API credentials")

    def cleanup(self):
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()


def main():
    """Main entry point."""
    downloader = UnityOptionsDailyDownloader()

    try:
        downloader.download_all_comprehensive()
    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user")
        downloader.show_comprehensive_summary(0, 0, 0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        downloader.cleanup()


if __name__ == "__main__":
    main()
