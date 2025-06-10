#!/usr/bin/env python3
"""
Download Unity options data from the CORRECT Databento dataset.
Based on the technical guide - Unity options are in OPRA.PILLAR dataset.
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


class UnityOptionsCorrectDownloader:
    """Downloads Unity options from OPRA.PILLAR dataset."""

    def __init__(self):
        self.client = DatabentoClient()
        self.db_path = Path("~/.wheel_trading/cache/wheel_cache.duckdb").expanduser()
        self.conn = duckdb.connect(str(self.db_path))
        self.eastern = pytz.timezone("US/Eastern")

        # Unity options start date per technical guide
        self.OPTIONS_START_DATE = datetime(2023, 3, 28).date()

        # Setup tables
        self.setup_tables()

    def setup_tables(self):
        """Create proper tables for Unity options data."""
        # Create table for Unity options data
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unity_options (
                date DATE NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                symbol VARCHAR NOT NULL,
                expiration DATE,
                strike DECIMAL(10,2),
                option_type VARCHAR(4),
                bid DECIMAL(10,4),
                ask DECIMAL(10,4),
                bid_size INTEGER,
                ask_size INTEGER,
                last DECIMAL(10,4),
                volume INTEGER,
                open_interest INTEGER,
                instrument_id BIGINT,
                PRIMARY KEY (date, timestamp, symbol)
            )
        """
        )

        # Create index for performance
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_unity_options_date
            ON unity_options(date)
        """
        )

        logger.info("Tables created/verified")

    def download_options_day(self, date: datetime.date) -> int:
        """Download Unity options for a specific date using correct dataset."""
        if date < self.OPTIONS_START_DATE:
            return 0

        market_start = datetime.combine(date, datetime.min.time()).replace(
            hour=9, minute=30, tzinfo=self.eastern
        )
        market_end = datetime.combine(date, datetime.min.time()).replace(
            hour=16, minute=0, tzinfo=self.eastern
        )

        try:
            logger.info(f"Downloading Unity options for {date}...")

            # Use OPRA.PILLAR dataset with parent symbology as per guide
            data = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",  # Correct dataset
                schema="cmbp-1",  # Consolidated NBBO
                symbols=["U.OPT"],  # Unity parent symbol
                stype_in="parent",  # Critical parameter
                start=market_start,
                end=market_end,
            )

            df = data.to_df()

            if df.empty:
                logger.info(f"  No data for {date}")
                return 0

            # Process the data
            records = 0

            for _, row in df.iterrows():
                # Extract option details from raw_symbol
                symbol = row.get("raw_symbol", "")
                if not symbol or "U" not in symbol:
                    continue

                # Parse OCC symbol format: "U  240119C00050000"
                try:
                    # Parse the symbol
                    parts = symbol.strip().split()
                    if len(parts) >= 2 and parts[0] == "U":
                        details = parts[1]
                        exp_str = details[:6]  # YYMMDD
                        opt_type = details[6]  # C or P
                        strike_str = details[7:]  # Strike * 1000

                        expiration = datetime.strptime("20" + exp_str, "%Y%m%d").date()
                        strike = float(strike_str) / 1000
                        option_type = "CALL" if opt_type == "C" else "PUT"

                        # Convert prices
                        bid = row.get("bid_px_01", 0) / 10000.0 if "bid_px_01" in row else None
                        ask = row.get("ask_px_01", 0) / 10000.0 if "ask_px_01" in row else None

                        # Store in database
                        self.conn.execute(
                            """
                            INSERT OR REPLACE INTO unity_options
                            (date, timestamp, symbol, expiration, strike, option_type,
                             bid, ask, bid_size, ask_size, instrument_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                date,
                                row.get("ts_event"),
                                symbol,
                                expiration,
                                strike,
                                option_type,
                                bid,
                                ask,
                                row.get("bid_sz_01"),
                                row.get("ask_sz_01"),
                                row.get("instrument_id"),
                            ),
                        )

                        records += 1

                except Exception as e:
                    # Skip malformed symbols
                    continue

            # Commit after each day
            self.conn.commit()

            if records > 0:
                logger.info(f"  ✓ Stored {records:,} Unity option records")

            return records

        except Exception as e:
            logger.error(f"Error downloading {date}: {e}")
            return 0

    def download_all_available(self):
        """Download all available Unity options data."""
        end_date = datetime.now().date() - timedelta(days=1)
        current_date = self.OPTIONS_START_DATE

        logger.info("=" * 60)
        logger.info("DOWNLOADING UNITY OPTIONS FROM OPRA.PILLAR")
        logger.info("=" * 60)
        logger.info(f"Date range: {self.OPTIONS_START_DATE} to {end_date}")
        logger.info("Dataset: OPRA.PILLAR (Options Price Reporting Authority)")
        logger.info("Schema: CMBP-1 (Consolidated NBBO)")
        logger.info("=" * 60)

        total_records = 0
        days_with_data = 0

        while current_date <= end_date:
            if current_date.weekday() < 5:  # Weekday
                records = self.download_options_day(current_date)

                if records > 0:
                    total_records += records
                    days_with_data += 1

                # Pause every 10 days
                if days_with_data % 10 == 0 and days_with_data > 0:
                    logger.info(f"Progress: {days_with_data} days, {total_records:,} records")
                    time.sleep(1)

            current_date += timedelta(days=1)

        # Summary
        self.show_summary()

    def show_summary(self):
        """Show summary of downloaded Unity options."""
        logger.info("\n" + "=" * 60)
        logger.info("UNITY OPTIONS SUMMARY")
        logger.info("=" * 60)

        stats = self.conn.execute(
            """
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT date) as trading_days,
                COUNT(DISTINCT symbol) as unique_options,
                COUNT(DISTINCT expiration) as expirations,
                COUNT(DISTINCT strike) as strikes,
                MIN(date) as start_date,
                MAX(date) as end_date
            FROM unity_options
        """
        ).fetchone()

        logger.info(f"Total records: {stats[0]:,}")
        logger.info(f"Trading days: {stats[1]}")
        logger.info(f"Unique options: {stats[2]:,}")
        logger.info(f"Unique expirations: {stats[3]}")
        logger.info(f"Unique strikes: {stats[4]}")
        logger.info(f"Date range: {stats[5]} to {stats[6]}")

        # Sample data
        logger.info("\nSAMPLE OPTIONS (latest date):")
        samples = self.conn.execute(
            """
            SELECT symbol, expiration, strike, option_type, bid, ask
            FROM unity_options
            WHERE date = (SELECT MAX(date) FROM unity_options)
            ORDER BY strike, option_type
            LIMIT 10
        """
        ).fetchall()

        logger.info("Symbol                  Exp        Strike  Type  Bid    Ask")
        logger.info("----------------------  ---------  ------  ----  -----  -----")
        for row in samples:
            logger.info(
                f"{row[0]:22s}  {row[1]}  ${row[2]:6.2f}  {row[3]:4s}  ${row[4] or 0:5.2f}  ${row[5] or 0:5.2f}"
            )

        logger.info("\n✓ All data from OPRA.PILLAR dataset (REAL market data)")
        logger.info("✓ NO SYNTHETIC DATA")

    def cleanup(self):
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()


def main():
    """Main entry point."""
    downloader = UnityOptionsCorrectDownloader()

    try:
        downloader.download_all_available()
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
