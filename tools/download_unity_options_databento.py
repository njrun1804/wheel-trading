#!/usr/bin/env python3
"""
Download Unity options data using correct Databento schemas and conventions.
Based on official Databento documentation.
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytz

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.data_providers.databento import DatabentoClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


class UnityOptionsDatabentoDownloader:
    """Download Unity options using proper Databento conventions."""

    def __init__(self):
        self.client = DatabentoClient()
        self.db_path = Path("~/.wheel_trading/cache/wheel_cache.duckdb").expanduser()
        self.conn = duckdb.connect(str(self.db_path))
        self.eastern = pytz.timezone("US/Eastern")

        # Create tables
        self.setup_tables()

    def setup_tables(self):
        """Create tables for Unity options data."""
        # For MBO (Market by Order) data
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unity_options_mbo (
                ts_event TIMESTAMP,
                ts_recv TIMESTAMP,
                instrument_id BIGINT,
                action CHAR(1),
                side CHAR(1),
                price DECIMAL(10,4),
                size INTEGER,
                order_id BIGINT,
                flags INTEGER,
                PRIMARY KEY (ts_event, order_id)
            )
        """
        )

        # For trades data
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unity_options_trades (
                ts_event TIMESTAMP,
                ts_recv TIMESTAMP,
                instrument_id BIGINT,
                price DECIMAL(10,4),
                size INTEGER,
                aggressor_side CHAR(1),
                trade_id BIGINT,
                PRIMARY KEY (ts_event, trade_id)
            )
        """
        )

        # For definition data (option specifications)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unity_options_definitions (
                ts_recv TIMESTAMP,
                instrument_id BIGINT PRIMARY KEY,
                raw_symbol VARCHAR,
                exchange VARCHAR,
                instrument_class VARCHAR,
                strike_price DECIMAL(10,2),
                expiration DATE,
                currency VARCHAR,
                multiplier INTEGER
            )
        """
        )

        logger.info("Tables created/verified")

    def download_options_definitions(self, date: datetime.date):
        """Download option definitions for Unity."""
        # For definitions, use midnight UTC
        start = datetime.combine(date, datetime.min.time()).replace(tzinfo=pytz.UTC)
        end = start + timedelta(seconds=1)

        try:
            logger.info(f"Downloading Unity option definitions for {date}...")

            # Get definitions - NO parent symbology for definitions
            data = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="definition",  # Correct schema name
                start=start,
                end=end,
                symbols=["U"],  # Just 'U' for underlying
                limit=10000,
            )

            df = data.to_df()

            if not df.empty:
                # Filter for Unity options
                unity_options = df[df["raw_symbol"].str.startswith("U ")]

                if not unity_options.empty:
                    logger.info(f"  Found {len(unity_options)} Unity option definitions")

                    # Store definitions
                    for _, row in unity_options.iterrows():
                        self.conn.execute(
                            """
                            INSERT OR REPLACE INTO unity_options_definitions
                            (ts_recv, instrument_id, raw_symbol, exchange,
                             instrument_class, strike_price, expiration,
                             currency, multiplier)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                row.get("ts_recv"),
                                row.get("instrument_id"),
                                row.get("raw_symbol"),
                                row.get("exchange"),
                                row.get("instrument_class"),
                                row.get("strike_price"),
                                row.get("expiration"),
                                row.get("currency"),
                                row.get("multiplier", 100),
                            ),
                        )

                    self.conn.commit()
                    return len(unity_options)

        except Exception as e:
            logger.error(f"Error downloading definitions: {e}")

        return 0

    def download_options_trades(self, date: datetime.date):
        """Download Unity options trades data."""
        market_start = datetime.combine(date, datetime.min.time()).replace(
            hour=9, minute=30, tzinfo=self.eastern
        )
        market_end = datetime.combine(date, datetime.min.time()).replace(
            hour=16, minute=0, tzinfo=self.eastern
        )

        try:
            logger.info(f"Downloading Unity options trades for {date}...")

            # Get all Unity instruments for this date
            unity_instruments = self.conn.execute(
                """
                SELECT DISTINCT instrument_id
                FROM unity_options_definitions
                WHERE expiration >= ?
            """,
                (date,),
            ).fetchall()

            if not unity_instruments:
                logger.info("  No Unity instruments found")
                return 0

            instrument_ids = [row[0] for row in unity_instruments]

            # Download trades data by instrument ID
            data = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="trades",  # Trades schema
                start=market_start,
                end=market_end,
                instrument_ids=instrument_ids[:100],  # Limit to 100 instruments
                limit=50000,
            )

            df = data.to_df()

            if not df.empty:
                logger.info(f"  Got {len(df)} trade records")

                # Store trades
                for _, row in df.iterrows():
                    self.conn.execute(
                        """
                        INSERT OR IGNORE INTO unity_options_trades
                        (ts_event, ts_recv, instrument_id, price, size,
                         aggressor_side, trade_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            row.get("ts_event"),
                            row.get("ts_recv"),
                            row.get("instrument_id"),
                            row.get("price"),
                            row.get("size"),
                            row.get("aggressor_side"),
                            row.get("sequence", 0),  # Use sequence as trade_id
                        ),
                    )

                self.conn.commit()
                return len(df)

        except Exception as e:
            logger.error(f"Error downloading trades: {e}")

        return 0

    def download_options_mbo(self, date: datetime.date):
        """Download Unity options MBO (order book) data."""
        # Get a sample window (last 30 minutes of trading)
        close_start = datetime.combine(date, datetime.min.time()).replace(
            hour=15, minute=30, tzinfo=self.eastern
        )
        close_end = datetime.combine(date, datetime.min.time()).replace(
            hour=16, minute=0, tzinfo=self.eastern
        )

        try:
            logger.info(f"Downloading Unity options MBO for {date} (last 30 min)...")

            # Get near-the-money instruments only
            unity_price = self.get_unity_price(date)
            if not unity_price:
                return 0

            ntm_instruments = self.conn.execute(
                """
                SELECT instrument_id
                FROM unity_options_definitions
                WHERE expiration >= ?
                AND expiration <= ? + INTERVAL '60 days'
                AND ABS(strike_price - ?) <= 10
                LIMIT 20
            """,
                (date, date, unity_price),
            ).fetchall()

            if not ntm_instruments:
                return 0

            instrument_ids = [row[0] for row in ntm_instruments]

            # Download MBO data
            data = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="mbo",  # Market by Order
                start=close_start,
                end=close_end,
                instrument_ids=instrument_ids,
                limit=10000,
            )

            df = data.to_df()

            if not df.empty:
                logger.info(f"  Got {len(df)} MBO records")
                # Store sample of MBO data
                # (Full MBO data would be massive)

                return len(df)

        except Exception as e:
            logger.error(f"Error downloading MBO: {e}")

        return 0

    def get_unity_price(self, date: datetime.date) -> float:
        """Get Unity stock price for a date."""
        result = self.conn.execute(
            """
            SELECT close FROM price_history
            WHERE symbol = 'U' AND date = ?
        """,
            (date,),
        ).fetchone()

        return result[0] if result else None

    def download_all_data(self):
        """Download all Unity options data."""
        # Start from when OPRA data is available
        start_date = datetime(2023, 3, 28).date()
        end_date = datetime.now().date() - timedelta(days=1)

        logger.info("=" * 60)
        logger.info("DOWNLOADING UNITY OPTIONS FROM DATABENTO")
        logger.info("=" * 60)
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info("=" * 60)

        current_date = start_date
        total_definitions = 0
        total_trades = 0
        days_processed = 0

        while current_date <= end_date:
            if current_date.weekday() < 5:  # Weekday
                logger.info(f"\nProcessing {current_date}...")

                # Download definitions (only need periodically)
                if days_processed % 5 == 0:  # Every 5 days
                    defs = self.download_options_definitions(current_date)
                    total_definitions += defs

                # Download trades
                trades = self.download_options_trades(current_date)
                total_trades += trades

                # Download MBO sample (optional - uses lots of data)
                # self.download_options_mbo(current_date)

                days_processed += 1

                # Pause periodically
                if days_processed % 10 == 0:
                    time.sleep(1)

            current_date += timedelta(days=1)

        # Summary
        self.show_summary()

    def show_summary(self):
        """Show summary of downloaded data."""
        logger.info("\n" + "=" * 60)
        logger.info("UNITY OPTIONS DATA SUMMARY")
        logger.info("=" * 60)

        # Definitions summary
        defs = self.conn.execute(
            """
            SELECT COUNT(*) as total,
                   COUNT(DISTINCT DATE(expiration)) as expirations,
                   COUNT(DISTINCT strike_price) as strikes,
                   MIN(expiration) as min_exp,
                   MAX(expiration) as max_exp
            FROM unity_options_definitions
        """
        ).fetchone()

        logger.info(f"\nOption Definitions:")
        logger.info(f"  Total contracts: {defs[0]:,}")
        logger.info(f"  Unique expirations: {defs[1]}")
        logger.info(f"  Unique strikes: {defs[2]}")
        logger.info(f"  Expiration range: {defs[3]} to {defs[4]}")

        # Trades summary
        trades = self.conn.execute(
            """
            SELECT COUNT(*) as total,
                   COUNT(DISTINCT DATE(ts_event)) as days,
                   AVG(price) as avg_price,
                   SUM(size) as total_volume
            FROM unity_options_trades
        """
        ).fetchone()

        logger.info(f"\nOptions Trades:")
        logger.info(f"  Total trades: {trades[0]:,}")
        logger.info(f"  Trading days: {trades[1]}")
        logger.info(f"  Average price: ${trades[2]:.2f}" if trades[2] else "  Average price: N/A")
        logger.info(f"  Total volume: {trades[3]:,}" if trades[3] else "  Total volume: N/A")

        # Sample options
        logger.info("\nSample Unity Options:")
        samples = self.conn.execute(
            """
            SELECT raw_symbol, strike_price, expiration
            FROM unity_options_definitions
            ORDER BY expiration, strike_price
            LIMIT 10
        """
        ).fetchall()

        for symbol, strike, exp in samples:
            logger.info(f"  {symbol}: ${strike} exp {exp}")

    def cleanup(self):
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()


def main():
    """Main entry point."""
    downloader = UnityOptionsDatabentoDownloader()

    try:
        downloader.download_all_data()
    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        downloader.cleanup()


if __name__ == "__main__":
    main()
