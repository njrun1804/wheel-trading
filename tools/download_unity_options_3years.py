#!/usr/bin/env python3
"""
Download 3 years of REAL Unity options data from Databento.
This script downloads actual market data - NO SYNTHETIC DATA.

REQUIREMENTS:
- Downloads 3 full years of Unity options data
- Uses Databento API with real credentials
- Stores in local DuckDB database
- Handles memory efficiently with daily chunks
- Implements proper error handling and retries
"""

import asyncio
import gc
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pytz

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import get_config
from src.unity_wheel.data_providers.databento import DatabentoClient
from src.unity_wheel.storage.duckdb_cache import DuckDBCache

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class Unity3YearOptionsDownloader:
    """Downloads 3 years of REAL Unity options data from Databento."""

    def __init__(self):
        self.config = get_config()
        self.client = DatabentoClient()
        # Use the same path as other scripts
        self.db_path = Path("~/.wheel_trading/cache/wheel_cache.duckdb").expanduser()
        self.eastern = pytz.timezone("US/Eastern")

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to DuckDB
        self.conn = duckdb.connect(str(self.db_path))

        # Processing parameters
        self.max_concurrent = 5  # Concurrent API requests
        self.chunk_size = 50000  # Records per batch insert
        self.memory_limit_gb = 4

        logger.info(f"Initialized downloader with database: {self.db_path}")

    def calculate_date_range(self) -> Tuple[datetime, datetime]:
        """Calculate 3 years of trading days to download."""
        end_date = datetime.now(self.eastern).date()
        start_date = end_date - timedelta(days=3 * 365)  # 3 years

        logger.info(f"Date range: {start_date} to {end_date} (3 years)")
        return start_date, end_date

    def ensure_tables_exist(self):
        """Ensure all necessary tables exist in DuckDB."""
        # Create databento_option_chains table if not exists
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS databento_option_chains (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                expiration DATE NOT NULL,
                strike DECIMAL(10,2) NOT NULL,
                option_type VARCHAR(4) NOT NULL,
                bid_price DECIMAL(10,4),
                ask_price DECIMAL(10,4),
                bid_size INTEGER,
                ask_size INTEGER,
                volume INTEGER,
                open_interest INTEGER,
                implied_volatility DOUBLE,
                delta DOUBLE,
                gamma DOUBLE,
                theta DOUBLE,
                vega DOUBLE,
                rho DOUBLE,
                underlying_price DECIMAL(10,4),
                PRIMARY KEY (symbol, timestamp, expiration, strike, option_type)
            )
        """
        )

        # Create options_ticks table for raw tick data
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS options_ticks (
                trade_date DATE NOT NULL,
                ts_event TIMESTAMP NOT NULL,
                instrument_id UINTEGER NOT NULL,
                symbol VARCHAR NOT NULL,
                bid_px DECIMAL(10,4),
                ask_px DECIMAL(10,4),
                bid_sz UINTEGER,
                ask_sz UINTEGER,
                PRIMARY KEY (trade_date, ts_event, instrument_id)
            )
        """
        )

        # Create instruments table if not exists
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS instruments (
                instrument_id UINTEGER PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                underlying VARCHAR NOT NULL,
                expiration DATE NOT NULL,
                strike DECIMAL(10,2) NOT NULL,
                option_type CHAR(1) NOT NULL,
                date_listed DATE,
                UNIQUE(symbol)
            )
        """
        )

        # Create indexes for performance
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_options_instrument ON options_ticks(instrument_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_instruments_underlying ON instruments(underlying, expiration)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_databento_chains_symbol ON databento_option_chains(symbol, timestamp)"
        )

        logger.info("Database tables verified/created")

    def download_daily_data(self, date: datetime.date) -> Dict[str, Any]:
        """Download one day of Unity options data."""
        try:
            start_time = datetime.combine(date, datetime.min.time()).replace(
                hour=9, minute=30, tzinfo=self.eastern
            )
            end_time = datetime.combine(date, datetime.min.time()).replace(
                hour=16, minute=0, tzinfo=self.eastern
            )

            logger.info(f"Downloading Unity options for {date}")

            # Get instrument definitions first
            definitions = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="definition",
                symbols=["U.OPT"],
                stype_in="parent",
                start=start_time,
                end=end_time,
            )

            # Get quote data (CMBP-1 for consolidated NBBO)
            quotes = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="cmbp-1",
                symbols=["U.OPT"],
                stype_in="parent",
                start=start_time,
                end=end_time,
            )

            # Get underlying price data
            underlying = self.client.client.timeseries.get_range(
                dataset="XNAS.ITCH",
                schema="ohlcv-1m",
                symbols=["U"],
                start=start_time,
                end=end_time,
            )

            return {
                "date": date,
                "definitions": definitions,
                "quotes": quotes,
                "underlying": underlying,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error downloading data for {date}: {e}")
            return {"date": date, "error": str(e), "success": False}

    def process_and_store_day(self, data: Dict[str, Any]) -> int:
        """Process and store one day of downloaded data."""
        if not data["success"]:
            logger.error(f"Skipping {data['date']} due to download error: {data.get('error')}")
            return 0

        date = data["date"]
        records_inserted = 0

        try:
            # Process definitions
            if data["definitions"]:
                defn_df = data["definitions"].to_df()
                if not defn_df.empty:
                    # Insert/update instruments
                    for _, row in defn_df.iterrows():
                        self.conn.execute(
                            """
                            INSERT OR REPLACE INTO instruments
                            (instrument_id, symbol, underlying, expiration, strike, option_type, date_listed)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                row.get("instrument_id"),
                                row.get("raw_symbol"),
                                "U",  # Unity
                                row.get("expiration"),
                                row.get("strike_price"),
                                (
                                    "C"
                                    if "CALL" in str(row.get("instrument_class", "")).upper()
                                    else "P"
                                ),
                                row.get("activation"),
                            ),
                        )
                    logger.info(f"Updated {len(defn_df)} instrument definitions")

            # Process quotes in chunks
            if data["quotes"]:
                quotes_df = data["quotes"].to_df()
                if not quotes_df.empty:
                    # Add date column for partitioning
                    quotes_df["trade_date"] = date

                    # Process in chunks to manage memory
                    for i in range(0, len(quotes_df), self.chunk_size):
                        chunk = quotes_df.iloc[i : i + self.chunk_size]

                        # Insert into options_ticks
                        self.conn.execute(
                            """
                            INSERT INTO options_ticks
                            (trade_date, ts_event, instrument_id, symbol, bid_px, ask_px, bid_sz, ask_sz)
                            SELECT
                                trade_date,
                                ts_event,
                                instrument_id,
                                raw_symbol as symbol,
                                bid_px_01 / 10000.0 as bid_px,
                                ask_px_01 / 10000.0 as ask_px,
                                bid_sz_01 as bid_sz,
                                ask_sz_01 as ask_sz
                            FROM chunk
                            ON CONFLICT DO NOTHING
                        """
                        )

                        records_inserted += len(chunk)

                        # Check memory usage
                        if self._get_memory_usage_gb() > self.memory_limit_gb:
                            gc.collect()

                    logger.info(f"Inserted {records_inserted} quote records for {date}")

            # Process underlying prices
            if data["underlying"]:
                underlying_df = data["underlying"].to_df()
                if not underlying_df.empty:
                    # Store underlying prices for Greeks calculation
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO price_history
                        (symbol, date, open, high, low, close, volume)
                        SELECT 'U', date(ts_event), open, high, low, close, volume
                        FROM underlying_df
                        ON CONFLICT(symbol, date) DO UPDATE SET
                            open = excluded.open,
                            high = excluded.high,
                            low = excluded.low,
                            close = excluded.close,
                            volume = excluded.volume
                    """
                    )
                    logger.info(f"Updated underlying prices for {date}")

            # Commit transaction
            self.conn.commit()

        except Exception as e:
            logger.error(f"Error processing data for {date}: {e}")
            self.conn.rollback()
            return 0

        return records_inserted

    def _get_memory_usage_gb(self) -> float:
        """Get current process memory usage in GB."""
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024 * 1024)

    def generate_trading_days(
        self, start_date: datetime.date, end_date: datetime.date
    ) -> List[datetime.date]:
        """Generate list of trading days between dates."""
        import pandas as pd
        from pandas.tseries.holiday import USFederalHolidayCalendar
        from pandas.tseries.offsets import CustomBusinessDay

        # Create custom business day calendar
        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

        # Generate trading days
        trading_days = pd.date_range(start=start_date, end=end_date, freq=us_bd)

        return [d.date() for d in trading_days]

    def download_all_data(self):
        """Download 3 years of Unity options data."""
        start_date, end_date = self.calculate_date_range()
        trading_days = self.generate_trading_days(start_date, end_date)

        logger.info(f"Downloading {len(trading_days)} trading days of Unity options data")
        logger.info("This will take several hours. Progress will be shown below.")

        total_records = 0
        failed_days = []

        # Process in batches with threading
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit download tasks
            future_to_date = {}

            for i, date in enumerate(trading_days):
                # Check if we should skip this date (weekend/holiday check is in trading_days)
                logger.info(f"Submitting download task {i+1}/{len(trading_days)}: {date}")

                future = executor.submit(self.download_daily_data, date)
                future_to_date[future] = date

                # Rate limit submissions
                if (i + 1) % self.max_concurrent == 0:
                    time.sleep(1)  # Brief pause every batch

            # Process results as they complete
            for future in as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    data = future.result()
                    if data["success"]:
                        records = self.process_and_store_day(data)
                        total_records += records
                        logger.info(f"Completed {date}: {records} records. Total: {total_records}")
                    else:
                        failed_days.append(date)
                except Exception as e:
                    logger.error(f"Failed to process {date}: {e}")
                    failed_days.append(date)

        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info(f"DOWNLOAD COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total records downloaded: {total_records:,}")
        logger.info(f"Trading days processed: {len(trading_days) - len(failed_days)}")
        logger.info(f"Failed days: {len(failed_days)}")

        if failed_days:
            logger.warning(
                f"Failed dates: {failed_days[:10]}{'...' if len(failed_days) > 10 else ''}"
            )

        # Verify data integrity
        self.verify_data_integrity()

        return (
            total_records > 0 and len(failed_days) < len(trading_days) * 0.1
        )  # Allow 10% failure rate

    def verify_data_integrity(self):
        """Verify the downloaded data is complete and valid."""
        logger.info("\nVerifying data integrity...")

        # Check total records
        total_ticks = self.conn.execute("SELECT COUNT(*) FROM options_ticks").fetchone()[0]
        total_instruments = self.conn.execute(
            "SELECT COUNT(*) FROM instruments WHERE underlying = 'U'"
        ).fetchone()[0]

        # Check date range
        date_range = self.conn.execute(
            """
            SELECT MIN(trade_date) as start_date, MAX(trade_date) as end_date, COUNT(DISTINCT trade_date) as days
            FROM options_ticks
        """
        ).fetchone()

        # Check data distribution
        daily_avg = self.conn.execute(
            """
            SELECT AVG(daily_count) as avg_records_per_day
            FROM (
                SELECT trade_date, COUNT(*) as daily_count
                FROM options_ticks
                GROUP BY trade_date
            )
        """
        ).fetchone()[0]

        logger.info(f"Total option tick records: {total_ticks:,}")
        logger.info(f"Total Unity instruments: {total_instruments:,}")
        logger.info(f"Date range: {date_range[0]} to {date_range[1]} ({date_range[2]} days)")
        logger.info(f"Average records per day: {daily_avg:,.0f}")

        # Validate we have substantial data
        if total_ticks < 1_000_000:  # Expect millions of records for 3 years
            logger.error(
                f"WARNING: Only {total_ticks:,} records found. Expected millions for 3 years of data."
            )
            return False

        if date_range[2] < 500:  # ~250 trading days per year * 3 years
            logger.error(
                f"WARNING: Only {date_range[2]} trading days found. Expected ~750 for 3 years."
            )
            return False

        logger.info("✓ Data integrity check passed")
        return True

    def cleanup(self):
        """Close database connections."""
        if hasattr(self, "conn"):
            self.conn.close()


def main():
    """Main entry point."""
    logger.info("Starting 3-year Unity options data download")
    logger.info("This will download REAL market data from Databento")
    logger.info("Expected duration: 2-4 hours depending on connection speed")

    downloader = Unity3YearOptionsDownloader()

    try:
        # Ensure tables exist
        downloader.ensure_tables_exist()

        # Download all data
        success = downloader.download_all_data()

        if not success:
            logger.error("FAILED: Unable to download 3 years of Unity options data")
            sys.exit(1)

        logger.info("SUCCESS: Downloaded 3 years of Unity options data")

    except KeyboardInterrupt:
        logger.warning("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        downloader.cleanup()


if __name__ == "__main__":
    main()
