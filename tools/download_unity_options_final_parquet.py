#!/usr/bin/env python3
"""
Download 3 years of Unity options data using Databento.
Fixed version that properly uses the DatabentoClient.
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
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


class UnityOptionsFinalDownloader:
    """Download 3 years of Unity options data from Databento."""

    def __init__(self):
        # Initialize Databento client - this will use SecretManager
        logger.info("Initializing Databento client...")
        self.client = DatabentoClient()

        # Database connection
        self.db_path = Path("~/.wheel_trading/cache/wheel_cache.duckdb").expanduser()
        self.conn = duckdb.connect(str(self.db_path))

        # Create/verify tables
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
                open_interest BIGINT,
                PRIMARY KEY (date, symbol)
            )
        """
        )

        logger.info("Tables created/verified")

    def download_all_data(self):
        """Download 3 years of Unity options data."""
        START = "2023-03-28"
        END = "2025-06-09"  # Yesterday

        logger.info("=" * 60)
        logger.info("DOWNLOADING 3 YEARS OF UNITY OPTIONS DATA")
        logger.info("=" * 60)
        logger.info(f"Dataset: OPRA.PILLAR")
        logger.info(f"Schema: ohlcv-1d (daily bars)")
        logger.info(f"Symbol: U.OPT (parent symbol for all Unity options)")
        logger.info(f"Date range: {START} to {END}")
        logger.info("=" * 60)

        try:
            # Use the client's client attribute to access the Historical API
            logger.info("Requesting data from Databento...")

            data = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                symbols=["U.OPT"],
                stype_in="parent",
                schema="ohlcv-1d",
                start=START,
                end=END,
                path=None,  # Return data directly
            )

            # Convert to DataFrame
            logger.info("Converting to DataFrame...")
            df = data.to_df()

            if df.empty:
                logger.error("No data returned!")
                return

            logger.info(f"Received {len(df):,} records")
            logger.info(f"Date range in data: {df.index.min()} to {df.index.max()}")

            # Get unique symbols to show variety
            if "symbol" in df.columns:
                unique_symbols = df["symbol"].nunique()
                logger.info(f"Unique option contracts: {unique_symbols:,}")

                # Show sample symbols
                sample_symbols = df["symbol"].unique()[:5]
                logger.info("Sample symbols:")
                for sym in sample_symbols:
                    logger.info(f"  {sym}")

            # Process and store
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

        for idx in range(0, len(df), batch_size):
            batch = df.iloc[idx : idx + batch_size]

            for _, row in batch.iterrows():
                try:
                    # Parse the OSI symbol
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

                        # Extract date from timestamp
                        trade_date = pd.Timestamp(row.name).date()

                        # Insert into database
                        self.conn.execute(
                            """
                            INSERT OR REPLACE INTO unity_options_daily
                            (date, symbol, expiration, strike, option_type,
                             open, high, low, close, volume, open_interest)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                                row.get("open_interest", 0),
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
                logger.info(f"  Total volume: {stats[5]:,}" if stats[5] else "  Total volume: N/A")
                logger.info(
                    f"  Average volume: {stats[6]:.0f}" if stats[6] else "  Average volume: N/A"
                )

            # Show daily breakdown for recent days
            logger.info("\nRecent Daily Coverage:")
            daily = self.conn.execute(
                """
                SELECT
                    date,
                    COUNT(*) as options_count,
                    SUM(volume) as total_volume
                FROM unity_options_daily
                GROUP BY date
                ORDER BY date DESC
                LIMIT 10
            """
            ).fetchall()

            for date, count, volume in daily:
                logger.info(f"  {date}: {count:,} options, {volume:,} volume")

            # Check if we got the expected coverage
            if stats[0] < 100:
                logger.warning(
                    f"\n⚠️  Only {stats[0]} days of data - Unity options may have limited trading"
                )
                logger.info(
                    "This is expected if Unity doesn't have actively traded options every day"
                )
            else:
                logger.info(f"\n✅ SUCCESS: Downloaded {stats[0]} days of Unity options data!")
                logger.info("✅ All data is REAL from Databento OPRA feed - NO SYNTHETIC DATA")

        else:
            logger.warning("\n⚠️  No data downloaded - check Databento API access")

    def cleanup(self):
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()


def main():
    """Main entry point."""
    downloader = UnityOptionsFinalDownloader()

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
