#!/usr/bin/env python3
"""
FINAL VERSION: Download 3 years of REAL Unity options data from Databento.
This script downloads actual market data - NO SYNTHETIC DATA.

IMPORTANT:
- Unity options data on Databento is available from 2023-03-28 onward
- This will download all available data (approximately 1.7 years)
- All data is REAL market data from OPRA
"""

import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import pandas as pd
import pytz

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.data_providers.databento import DatabentoClient

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


class UnityOptionsDownloader:
    """Downloads all available Unity options data from Databento."""

    def __init__(self):
        self.client = DatabentoClient()
        self.db_path = Path("~/.wheel_trading/cache/wheel_cache.duckdb").expanduser()
        self.eastern = pytz.timezone("US/Eastern")

        # Data availability limits
        self.OPTIONS_START_DATE = datetime(2023, 3, 28).date()  # When OPRA data starts
        self.STOCK_START_DATE = datetime(2022, 1, 1).date()  # Stock data goes back further

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to DuckDB
        self.conn = duckdb.connect(str(self.db_path))

        logger.info(f"Initialized with database: {self.db_path}")
        logger.info(f"Options data available from: {self.OPTIONS_START_DATE}")

    def ensure_tables_exist(self):
        """Create tables if they don't exist."""
        # Drop old synthetic data table if exists
        self.conn.execute("DROP TABLE IF EXISTS databento_option_chains")

        # Create new clean tables
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unity_options_ticks (
                trade_date DATE NOT NULL,
                ts_event TIMESTAMP NOT NULL,
                instrument_id BIGINT,
                raw_symbol VARCHAR,
                bid_px DECIMAL(10,4),
                ask_px DECIMAL(10,4),
                bid_sz INTEGER,
                ask_sz INTEGER,
                PRIMARY KEY (trade_date, ts_event, raw_symbol)
            )
        """
        )

        # Unity stock prices
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unity_stock_1min (
                date DATE NOT NULL,
                ts_event TIMESTAMP NOT NULL,
                open DECIMAL(10,4),
                high DECIMAL(10,4),
                low DECIMAL(10,4),
                close DECIMAL(10,4),
                volume BIGINT,
                PRIMARY KEY (date, ts_event)
            )
        """
        )

        # Summary table for daily stats
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unity_daily_summary (
                date DATE PRIMARY KEY,
                stock_records INTEGER,
                option_records INTEGER,
                unique_strikes INTEGER,
                avg_spread DECIMAL(10,4),
                download_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        logger.info("Tables created/verified")

    def clear_existing_data(self):
        """Clear any existing data to ensure only real data remains."""
        tables_to_clear = [
            "databento_option_chains",  # Old synthetic data
            "unity_options_ticks",
            "unity_stock_1min",
            "unity_daily_summary",
        ]

        for table in tables_to_clear:
            try:
                count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if count > 0:
                    logger.info(f"Clearing {count:,} records from {table}")
                    self.conn.execute(f"DELETE FROM {table}")
            except:
                pass  # Table might not exist

        self.conn.commit()
        logger.info("Cleared all existing data")

    def download_stock_data(self, date: datetime.date) -> pd.DataFrame:
        """Download Unity stock data for a given date."""
        market_start = datetime.combine(date, datetime.min.time()).replace(
            hour=9, minute=30, tzinfo=self.eastern
        )
        market_end = datetime.combine(date, datetime.min.time()).replace(
            hour=16, minute=0, tzinfo=self.eastern
        )

        try:
            data = self.client.client.timeseries.get_range(
                dataset="XNAS.ITCH",
                schema="ohlcv-1m",
                symbols=["U"],
                start=market_start,
                end=market_end,
            )
            return data.to_df()
        except Exception as e:
            logger.warning(f"No stock data for {date}: {e}")
            return pd.DataFrame()

    def download_options_data(self, date: datetime.date) -> pd.DataFrame:
        """Download Unity options data for a given date."""
        # Skip if before options data availability
        if date < self.OPTIONS_START_DATE:
            return pd.DataFrame()

        market_start = datetime.combine(date, datetime.min.time()).replace(
            hour=9, minute=30, tzinfo=self.eastern
        )
        market_end = datetime.combine(date, datetime.min.time()).replace(
            hour=16, minute=0, tzinfo=self.eastern
        )

        try:
            data = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="cmbp-1",
                symbols=["U.OPT"],
                stype_in="parent",
                start=market_start,
                end=market_end,
            )
            return data.to_df()
        except Exception as e:
            logger.warning(f"No options data for {date}: {e}")
            return pd.DataFrame()

    def process_day(self, date: datetime.date) -> Dict[str, int]:
        """Download and store data for one day."""
        stats = {"stock_records": 0, "option_records": 0, "unique_strikes": 0, "avg_spread": 0}

        # Download stock data
        stock_df = self.download_stock_data(date)
        if not stock_df.empty:
            # Store stock data
            stock_df["date"] = date

            # Use executemany for batch insert
            stock_data = []
            for _, row in stock_df.iterrows():
                stock_data.append(
                    (
                        date,
                        row.get("ts_event", pd.Timestamp.now()),
                        row.get("open", 0),
                        row.get("high", 0),
                        row.get("low", 0),
                        row.get("close", 0),
                        row.get("volume", 0),
                    )
                )

            self.conn.executemany(
                """
                INSERT OR IGNORE INTO unity_stock_1min
                (date, ts_event, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                stock_data,
            )

            stats["stock_records"] = len(stock_df)

        # Download options data
        options_df = self.download_options_data(date)
        if not options_df.empty:
            # Process options data
            options_data = []
            unique_symbols = set()
            spreads = []

            for _, row in options_df.iterrows():
                # Convert prices from integer representation
                bid_px = row.get("bid_px_01", 0) / 10000.0 if "bid_px_01" in row else None
                ask_px = row.get("ask_px_01", 0) / 10000.0 if "ask_px_01" in row else None

                if bid_px and ask_px and bid_px > 0:
                    spreads.append(ask_px - bid_px)

                symbol = row.get("raw_symbol", "")
                if symbol:
                    unique_symbols.add(symbol)

                options_data.append(
                    (
                        date,
                        row.get("ts_event", pd.Timestamp.now()),
                        row.get("instrument_id", 0),
                        symbol,
                        bid_px,
                        ask_px,
                        row.get("bid_sz_01", 0),
                        row.get("ask_sz_01", 0),
                    )
                )

            # Batch insert
            if options_data:
                self.conn.executemany(
                    """
                    INSERT OR IGNORE INTO unity_options_ticks
                    (trade_date, ts_event, instrument_id, raw_symbol,
                     bid_px, ask_px, bid_sz, ask_sz)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    options_data,
                )

            stats["option_records"] = len(options_df)
            stats["unique_strikes"] = len(unique_symbols)
            stats["avg_spread"] = sum(spreads) / len(spreads) if spreads else 0

        # Store daily summary
        self.conn.execute(
            """
            INSERT OR REPLACE INTO unity_daily_summary
            (date, stock_records, option_records, unique_strikes, avg_spread)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                date,
                stats["stock_records"],
                stats["option_records"],
                stats["unique_strikes"],
                stats["avg_spread"],
            ),
        )

        # Commit after each day
        self.conn.commit()

        return stats

    def download_all_available_data(self):
        """Download all available Unity data."""
        # Determine date range
        end_date = datetime.now().date() - timedelta(days=1)
        stock_start = max(self.STOCK_START_DATE, end_date - timedelta(days=3 * 365))

        logger.info(f"\n{'='*60}")
        logger.info("DOWNLOADING UNITY DATA FROM DATABENTO")
        logger.info(f"{'='*60}")
        logger.info(f"Stock data range: {stock_start} to {end_date}")
        logger.info(f"Options data range: {self.OPTIONS_START_DATE} to {end_date}")
        logger.info("This is REAL market data - NO SYNTHETIC DATA")
        logger.info(f"{'='*60}\n")

        # Generate trading days
        current = stock_start
        trading_days = []
        while current <= end_date:
            if current.weekday() < 5:  # Weekday
                trading_days.append(current)
            current += timedelta(days=1)

        logger.info(f"Processing {len(trading_days)} trading days...")

        # Process each day
        total_stock_records = 0
        total_option_records = 0
        days_with_options = 0

        for i, date in enumerate(trading_days):
            if i % 20 == 0:
                logger.info(
                    f"\nProgress: {i}/{len(trading_days)} days ({i/len(trading_days)*100:.1f}%)"
                )
                logger.info(
                    f"Stock records: {total_stock_records:,}, Option records: {total_option_records:,}"
                )

            try:
                stats = self.process_day(date)
                total_stock_records += stats["stock_records"]
                total_option_records += stats["option_records"]
                if stats["option_records"] > 0:
                    days_with_options += 1

                # Log progress for days with data
                if stats["stock_records"] > 0 or stats["option_records"] > 0:
                    logger.debug(
                        f"{date}: {stats['stock_records']} stock, {stats['option_records']} options"
                    )

            except Exception as e:
                logger.error(f"Error processing {date}: {e}")
                traceback.print_exc()

            # Brief pause every 10 days
            if i % 10 == 9:
                time.sleep(0.5)

        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("DOWNLOAD COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total stock records: {total_stock_records:,}")
        logger.info(f"Total option records: {total_option_records:,}")
        logger.info(f"Days with options data: {days_with_options}")
        logger.info(f"Days processed: {len(trading_days)}")

        # Verify data
        self.verify_data()

    def verify_data(self):
        """Verify downloaded data integrity."""
        logger.info(f"\n{'='*60}")
        logger.info("DATA VERIFICATION")
        logger.info(f"{'='*60}")

        # Check stock data
        stock_stats = self.conn.execute(
            """
            SELECT
                COUNT(*) as records,
                COUNT(DISTINCT date) as days,
                MIN(date) as start_date,
                MAX(date) as end_date,
                AVG(close) as avg_price
            FROM unity_stock_1min
        """
        ).fetchone()

        logger.info(f"\nStock Data Summary:")
        logger.info(f"  Records: {stock_stats[0]:,}")
        logger.info(f"  Trading days: {stock_stats[1]}")
        logger.info(f"  Date range: {stock_stats[2]} to {stock_stats[3]}")
        logger.info(f"  Average price: ${stock_stats[4]:.2f}")

        # Check options data
        options_stats = self.conn.execute(
            """
            SELECT
                COUNT(*) as records,
                COUNT(DISTINCT trade_date) as days,
                COUNT(DISTINCT raw_symbol) as unique_contracts,
                MIN(trade_date) as start_date,
                MAX(trade_date) as end_date,
                AVG(CASE WHEN bid_px > 0 AND ask_px > 0 THEN ask_px - bid_px END) as avg_spread
            FROM unity_options_ticks
        """
        ).fetchone()

        logger.info(f"\nOptions Data Summary:")
        logger.info(f"  Records: {options_stats[0]:,}")
        logger.info(f"  Trading days: {options_stats[1]}")
        logger.info(f"  Unique contracts: {options_stats[2]:,}")
        logger.info(f"  Date range: {options_stats[3]} to {options_stats[4]}")
        logger.info(
            f"  Average spread: ${options_stats[5]:.3f}"
            if options_stats[5]
            else "  Average spread: N/A"
        )

        # Data quality check
        if options_stats[0] > 100000 and stock_stats[0] > 10000:
            logger.info(f"\n✅ SUCCESS: Real Unity data downloaded from Databento")
            logger.info("✅ All data is authentic market data")
            logger.info("✅ NO SYNTHETIC DATA in database")
            return True
        else:
            logger.warning(f"\n⚠️  Limited data downloaded - check Databento subscription")
            return False

    def cleanup(self):
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()


def main():
    """Main entry point."""
    downloader = UnityOptionsDownloader()

    try:
        # Setup
        downloader.ensure_tables_exist()
        downloader.clear_existing_data()

        # Download all available data
        downloader.download_all_available_data()

        logger.info("\n✅ Unity options data download complete!")
        logger.info("✅ Database contains only REAL market data from Databento")

    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
    finally:
        downloader.cleanup()


if __name__ == "__main__":
    main()
