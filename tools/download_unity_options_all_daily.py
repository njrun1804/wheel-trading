#!/usr/bin/env python3
"""
Download ALL Unity options data with TRUE daily coverage.
Uses the statistics schema to get end-of-day data for ALL contracts, not just traded ones.
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


class UnityOptionsAllDailyDownloader:
    """Download ALL Unity options with true daily coverage."""

    def __init__(self):
        # Initialize Databento client
        self.client = DatabentoClient()

        # Database connection
        self.db_path = Path(config.storage.database_path).expanduser()
        self.conn = duckdb.connect(str(self.db_path))

        # Create a new table for comprehensive data
        self.setup_tables()

    def setup_tables(self):
        """Create tables for comprehensive Unity options data."""
        # Create new table for all daily data
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unity_options_all_daily (
                date DATE NOT NULL,
                symbol VARCHAR NOT NULL,
                expiration DATE NOT NULL,
                strike DECIMAL(10,2) NOT NULL,
                option_type VARCHAR(1) NOT NULL,
                open DECIMAL(10,4),
                high DECIMAL(10,4),
                low DECIMAL(10,4),
                close DECIMAL(10,4),
                settlement DECIMAL(10,4),
                volume BIGINT,
                open_interest BIGINT,
                bid_close DECIMAL(10,4),
                ask_close DECIMAL(10,4),
                bid_size INT,
                ask_size INT,
                underlying_close DECIMAL(10,2),
                iv DECIMAL(6,4),
                delta DECIMAL(6,4),
                gamma DECIMAL(6,4),
                theta DECIMAL(6,4),
                vega DECIMAL(6,4),
                PRIMARY KEY (date, symbol)
            )
        """
        )

        logger.info("Tables created/verified")

    def download_all_data(self):
        """Download Unity options with TRUE daily coverage."""
        # Define date range
        START = "2023-03-28"  # Unity options started here
        END = "2025-06-09"  # Data available to yesterday

        logger.info("=" * 60)
        logger.info("DOWNLOADING ALL UNITY OPTIONS - TRUE DAILY COVERAGE")
        logger.info("=" * 60)
        logger.info(f"Dataset: OPRA.PILLAR")
        logger.info(f"Symbol: U.OPT (using parent symbol)")
        logger.info(f"Date range: {START} to {END}")
        logger.info("=" * 60)

        # Try multiple schemas in order of preference
        schemas_to_try = [
            ("statistics", "End-of-day statistics for ALL contracts"),
            ("ohlcv-eod-1d", "End-of-day OHLCV for ALL contracts"),
            ("bbo-1d", "Best bid/offer daily for ALL contracts"),
            ("definition", "Contract definitions + last values"),
        ]

        for schema, description in schemas_to_try:
            logger.info(f"\nTrying schema: {schema} - {description}")

            try:
                # Check if schema is available
                available_schemas = self.client.client.metadata.list_schemas("OPRA.PILLAR")
                if schema not in available_schemas:
                    logger.warning(f"Schema {schema} not available, skipping...")
                    continue

                # Try to estimate cost first
                try:
                    cost = self.client.client.metadata.get_cost(
                        dataset="OPRA.PILLAR",
                        symbols=["U.OPT"],
                        stype_in="parent",
                        schema=schema,
                        start=START,
                        end=END,
                        mode="historical-streaming",
                    )
                    logger.info(f"Estimated cost for {schema}: ${cost:,.2f}")
                except Exception as e:
                    logger.warning(f"Could not estimate cost: {e}")

                # Download the data
                logger.info(f"Downloading with {schema} schema...")

                data = self.client.client.timeseries.get_range(
                    dataset="OPRA.PILLAR",
                    symbols=["U.OPT"],
                    stype_in="parent",
                    schema=schema,
                    start=START,
                    end=END,
                    path=None,
                )

                # Convert to DataFrame
                df = data.to_df()

                if df.empty:
                    logger.warning(f"No data returned for {schema}")
                    continue

                logger.info(f"SUCCESS with {schema}!")
                logger.info(f"Received {len(df):,} records")

                # Check coverage
                if "ts_event" in df.columns or df.index.name == "ts_event":
                    unique_dates = (
                        df.index.normalize().unique()
                        if df.index.name == "ts_event"
                        else pd.to_datetime(df["ts_event"]).dt.normalize().unique()
                    )
                    logger.info(f"Unique dates: {len(unique_dates)}")

                # Process and store the data
                self.process_and_store(df, schema)

                # Show summary
                self.show_summary()

                # Success - no need to try other schemas
                return

            except Exception as e:
                logger.error(f"Schema {schema} failed: {e}")
                continue

        logger.error("All schemas failed!")

        # If all schemas fail, try a different approach
        logger.info("\nTrying alternative approach: Daily snapshots by date...")
        self.download_daily_snapshots(START, END)

    def download_daily_snapshots(self, start_str: str, end_str: str):
        """Download daily snapshots for each trading day."""
        start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_str, "%Y-%m-%d").date()

        current_date = start_date
        total_records = 0
        days_with_data = 0

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                try:
                    # Get end-of-day snapshot for this specific date
                    logger.info(f"Getting snapshot for {current_date}...")

                    # Use a very narrow time window at market close
                    market_close = datetime.combine(current_date, datetime.min.time()).replace(
                        hour=16, minute=0, second=0
                    )

                    data = self.client.client.timeseries.get_range(
                        dataset="OPRA.PILLAR",
                        symbols=["U.OPT"],
                        stype_in="parent",
                        schema="tbbo",  # Top of book (best bid/offer)
                        start=market_close - timedelta(minutes=5),
                        end=market_close + timedelta(minutes=1),
                        limit=50000,
                    )

                    df = data.to_df()

                    if not df.empty:
                        records = self.process_snapshot_for_date(df, current_date)
                        total_records += records
                        days_with_data += 1
                        logger.info(f"  ✓ {current_date}: {records} options")
                    else:
                        logger.debug(f"  - {current_date}: No data")

                except Exception as e:
                    logger.debug(f"  Error for {current_date}: {e}")

            current_date += timedelta(days=1)

            # Brief pause every 10 days
            if days_with_data % 10 == 0 and days_with_data > 0:
                import time

                time.sleep(1)

        logger.info(f"\nTotal records: {total_records:,}")
        logger.info(f"Days with data: {days_with_data}")

        self.show_summary()

    def process_and_store(self, df, schema: str):
        """Process and store the downloaded data based on schema type."""
        logger.info(f"Processing {len(df):,} records from {schema} schema...")

        records_inserted = 0
        batch_size = 1000

        # Get column names for debugging
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")

        for idx in range(0, len(df), batch_size):
            batch = df.iloc[idx : idx + batch_size]

            for _, row in batch.iterrows():
                try:
                    # Get the symbol - could be in different columns
                    symbol = row.get("symbol", row.get("raw_symbol", ""))

                    # Skip if not Unity option
                    if not symbol.startswith("U "):
                        continue

                    # Parse OSI symbol
                    if len(symbol) >= 21:
                        exp_str = symbol[6:12]
                        expiration = datetime.strptime("20" + exp_str, "%Y%m%d").date()
                        option_type = symbol[12]
                        strike = float(symbol[13:21]) / 1000

                        # Extract date from timestamp
                        if hasattr(row, "name"):
                            trade_date = pd.Timestamp(row.name).date()
                        elif "ts_event" in row:
                            trade_date = pd.Timestamp(row["ts_event"]).date()
                        else:
                            continue

                        # Extract prices based on schema
                        if schema == "statistics":
                            record_data = {
                                "date": trade_date,
                                "symbol": symbol,
                                "expiration": expiration,
                                "strike": strike,
                                "option_type": option_type,
                                "open": self.convert_price(row.get("open_price")),
                                "high": self.convert_price(row.get("high_price")),
                                "low": self.convert_price(row.get("low_price")),
                                "close": self.convert_price(row.get("close_price")),
                                "settlement": self.convert_price(row.get("settlement_price")),
                                "volume": row.get("volume", 0),
                                "open_interest": row.get("open_interest", 0),
                                "bid_close": self.convert_price(row.get("bid_close")),
                                "ask_close": self.convert_price(row.get("ask_close")),
                            }
                        else:
                            # Default for ohlcv or other schemas
                            record_data = {
                                "date": trade_date,
                                "symbol": symbol,
                                "expiration": expiration,
                                "strike": strike,
                                "option_type": option_type,
                                "open": self.convert_price(row.get("open")),
                                "high": self.convert_price(row.get("high")),
                                "low": self.convert_price(row.get("low")),
                                "close": self.convert_price(row.get("close")),
                                "volume": row.get("volume", 0),
                                "open_interest": row.get("open_interest", 0),
                            }

                        # Insert into database
                        self.insert_record(record_data)
                        records_inserted += 1

                except Exception as e:
                    logger.debug(f"Failed to process record: {e}")
                    continue

            # Commit batch
            self.conn.commit()

            if records_inserted % 10000 == 0 and records_inserted > 0:
                logger.info(f"  Processed {records_inserted:,} records...")

        logger.info(f"Inserted {records_inserted:,} total records")

    def process_snapshot_for_date(self, df, date):
        """Process snapshot data for a specific date."""
        records = 0

        for _, row in df.iterrows():
            try:
                symbol = row.get("symbol", row.get("raw_symbol", ""))

                if not symbol.startswith("U "):
                    continue

                if len(symbol) >= 21:
                    exp_str = symbol[6:12]
                    expiration = datetime.strptime("20" + exp_str, "%Y%m%d").date()
                    option_type = symbol[12]
                    strike = float(symbol[13:21]) / 1000

                    record_data = {
                        "date": date,
                        "symbol": symbol,
                        "expiration": expiration,
                        "strike": strike,
                        "option_type": option_type,
                        "bid_close": self.convert_price(row.get("bid_px_01")),
                        "ask_close": self.convert_price(row.get("ask_px_01")),
                        "bid_size": row.get("bid_sz_01", 0),
                        "ask_size": row.get("ask_sz_01", 0),
                    }

                    self.insert_record(record_data)
                    records += 1

            except Exception:
                continue

        self.conn.commit()
        return records

    def convert_price(self, price):
        """Convert Databento price format."""
        if price is None:
            return None
        # Handle both integer fixed-point and float formats
        if isinstance(price, (int, float)):
            if price > 10000:
                return float(price) / 10000.0
            elif price > 1000:
                return float(price) / 1000.0
            else:
                return float(price)
        return None

    def insert_record(self, data):
        """Insert a record into the database."""
        columns = list(data.keys())
        values = [data.get(col) for col in columns]
        placeholders = ", ".join(["?" for _ in columns])

        query = f"""
            INSERT OR REPLACE INTO unity_options_all_daily ({', '.join(columns)})
            VALUES ({placeholders})
        """

        self.conn.execute(query, values)

    def show_summary(self):
        """Show comprehensive summary of all downloaded data."""
        logger.info("\n" + "=" * 60)
        logger.info("UNITY OPTIONS DATA SUMMARY - ALL DAILY")
        logger.info("=" * 60)

        # Check both tables
        for table in ["unity_options_daily", "unity_options_all_daily"]:
            try:
                stats = self.conn.execute(
                    f"""
                    SELECT
                        COUNT(DISTINCT date) as trading_days,
                        COUNT(DISTINCT symbol) as unique_contracts,
                        COUNT(*) as total_records,
                        MIN(date) as first_date,
                        MAX(date) as last_date
                    FROM {table}
                """
                ).fetchone()

                if stats and stats[2] > 0:
                    logger.info(f"\nTable: {table}")
                    logger.info(f"  Trading days: {stats[0]}")
                    logger.info(f"  Unique contracts: {stats[1]:,}")
                    logger.info(f"  Total records: {stats[2]:,}")
                    logger.info(f"  Date range: {stats[3]} to {stats[4]}")

                    # Calculate coverage
                    if stats[3] and stats[4]:
                        start = datetime.strptime(str(stats[3]), "%Y-%m-%d")
                        end = datetime.strptime(str(stats[4]), "%Y-%m-%d")
                        total_days = (end - start).days + 1
                        weekdays = sum(
                            1
                            for i in range(total_days)
                            if (start + timedelta(days=i)).weekday() < 5
                        )
                        coverage = (stats[0] / weekdays) * 100 if weekdays > 0 else 0

                        logger.info(f"  Coverage: {coverage:.1f}% of weekdays")

                        if coverage > 50:
                            logger.info("  ✅ EXCELLENT COVERAGE!")
                        elif coverage > 20:
                            logger.info("  ✅ Good coverage")
                        else:
                            logger.info("  ⚠️  Limited coverage - Unity may have sparse trading")

            except Exception as e:
                logger.debug(f"Error checking {table}: {e}")

    def cleanup(self):
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()


def main():
    """Main entry point."""
    downloader = UnityOptionsAllDailyDownloader()

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
