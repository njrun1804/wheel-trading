#!/usr/bin/env python3
"""
Download DAILY Unity stock and options data from Databento.
Just end-of-day summaries, not tick data.

Stock: Daily OHLCV bars
Options: Daily snapshots of all strikes/expirations
"""

import logging
import os
import sys
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


class UnityDailyDataDownloader:
    """Downloads daily Unity stock and options data."""

    def __init__(self):
        self.client = DatabentoClient()
        self.db_path = Path(config.storage.database_path).expanduser()
        self.conn = duckdb.connect(str(self.db_path))

        # Create tables for daily data
        self.setup_tables()

    def setup_tables(self):
        """Create tables for daily data."""
        # Daily stock data
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unity_daily_stock (
                date DATE PRIMARY KEY,
                open DECIMAL(10,2),
                high DECIMAL(10,2),
                low DECIMAL(10,2),
                close DECIMAL(10,2),
                volume BIGINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Daily options data
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unity_daily_options (
                date DATE NOT NULL,
                expiration DATE NOT NULL,
                strike DECIMAL(10,2) NOT NULL,
                option_type VARCHAR(4) NOT NULL,
                bid DECIMAL(10,2),
                ask DECIMAL(10,2),
                last DECIMAL(10,2),
                volume INTEGER,
                open_interest INTEGER,
                implied_volatility DECIMAL(10,4),
                delta DECIMAL(10,4),
                gamma DECIMAL(10,4),
                theta DECIMAL(10,4),
                vega DECIMAL(10,4),
                PRIMARY KEY (date, expiration, strike, option_type)
            )
        """
        )

        logger.info("Tables created/verified")

    def download_daily_stock(self, date: datetime.date) -> bool:
        """Download daily stock bar for Unity."""
        try:
            eastern = pytz.timezone("US/Eastern")
            start = datetime.combine(date, datetime.min.time()).replace(
                hour=0, minute=0, tzinfo=eastern
            )
            end = datetime.combine(date, datetime.min.time()).replace(
                hour=23, minute=59, tzinfo=eastern
            )

            # Get daily OHLCV
            data = self.client.client.timeseries.get_range(
                dataset="XNAS.ITCH",
                schema="ohlcv-1d",  # Daily bars
                symbols=["U"],
                start=start,
                end=end,
            )

            df = data.to_df()
            if not df.empty:
                row = df.iloc[0]
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO unity_daily_stock
                    (date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (date, row["open"], row["high"], row["low"], row["close"], row["volume"]),
                )
                return True
        except Exception as e:
            logger.debug(f"No stock data for {date}: {e}")
        return False

    def download_daily_options(self, date: datetime.date) -> int:
        """Download end-of-day options snapshot."""
        if date < datetime(2023, 3, 28).date():
            return 0

        try:
            eastern = pytz.timezone("US/Eastern")
            # Get snapshot at market close
            close_time = datetime.combine(date, datetime.min.time()).replace(
                hour=15, minute=59, tzinfo=eastern
            )

            # Get options snapshot
            data = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="eod",  # End of day summary
                symbols=["U.OPT"],
                stype_in="parent",
                start=close_time,
                end=close_time + timedelta(minutes=1),
            )

            df = data.to_df()
            if df.empty:
                # Try alternate schema
                data = self.client.client.timeseries.get_range(
                    dataset="OPRA.PILLAR",
                    schema="cmbp-1",
                    symbols=["U.OPT"],
                    stype_in="parent",
                    start=close_time - timedelta(minutes=30),
                    end=close_time,
                    limit=10000,  # Get latest quotes
                )
                df = data.to_df()

            records = 0
            if not df.empty:
                # Group by symbol to get unique options
                for symbol, group in df.groupby("raw_symbol"):
                    if not symbol or "U" not in symbol:
                        continue

                    # Parse option symbol (e.g., "U  240119C00050000")
                    try:
                        # Extract expiration and strike from OCC symbol
                        parts = symbol.strip().split()
                        if len(parts) >= 2:
                            info = parts[1]
                            exp_str = info[:6]  # YYMMDD
                            opt_type = info[6]  # C or P
                            strike_str = info[7:]  # Strike * 1000

                            expiration = datetime.strptime("20" + exp_str, "%Y%m%d").date()
                            strike = float(strike_str) / 1000

                            # Get latest quote
                            latest = group.iloc[-1]
                            bid = (
                                latest.get("bid_px_01", 0) / 10000.0 if "bid_px_01" in latest else 0
                            )
                            ask = (
                                latest.get("ask_px_01", 0) / 10000.0 if "ask_px_01" in latest else 0
                            )

                            self.conn.execute(
                                """
                                INSERT OR REPLACE INTO unity_daily_options
                                (date, expiration, strike, option_type, bid, ask, last, volume, open_interest)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    date,
                                    expiration,
                                    strike,
                                    "CALL" if opt_type == "C" else "PUT",
                                    bid,
                                    ask,
                                    (bid + ask) / 2 if bid and ask else None,
                                    None,
                                    None,  # Volume and OI not in tick data
                                ),
                            )
                            records += 1
                    except:
                        continue

            return records

        except Exception as e:
            logger.debug(f"No options data for {date}: {e}")
            return 0

    def download_all_daily_data(self):
        """Download all available daily data."""
        # Date ranges
        stock_start = datetime(2022, 1, 1).date()
        options_start = datetime(2023, 3, 28).date()
        end_date = datetime.now().date() - timedelta(days=1)

        logger.info("=" * 60)
        logger.info("DOWNLOADING UNITY DAILY DATA")
        logger.info("=" * 60)
        logger.info(f"Stock data: {stock_start} to {end_date}")
        logger.info(f"Options data: {options_start} to {end_date}")
        logger.info("=" * 60)

        # Download stock data
        logger.info("\nDownloading daily stock data...")
        current = stock_start
        stock_days = 0

        while current <= end_date:
            if current.weekday() < 5:  # Weekday
                if self.download_daily_stock(current):
                    stock_days += 1
                    if stock_days % 50 == 0:
                        logger.info(f"  Stock: {stock_days} days downloaded...")
            current += timedelta(days=1)

        logger.info(f"✓ Stock data complete: {stock_days} days")

        # Download options data
        logger.info("\nDownloading daily options data...")
        current = options_start
        options_days = 0
        total_contracts = 0

        while current <= end_date:
            if current.weekday() < 5:  # Weekday
                contracts = self.download_daily_options(current)
                if contracts > 0:
                    options_days += 1
                    total_contracts += contracts
                    if options_days % 20 == 0:
                        logger.info(
                            f"  Options: {options_days} days, {total_contracts:,} contracts..."
                        )
            current += timedelta(days=1)

        logger.info(f"✓ Options data complete: {options_days} days, {total_contracts:,} contracts")

        # Commit all changes
        self.conn.commit()

        # Summary
        self.show_summary()

    def show_summary(self):
        """Show data summary."""
        logger.info("\n" + "=" * 60)
        logger.info("DATA SUMMARY")
        logger.info("=" * 60)

        # Stock summary
        stock_stats = self.conn.execute(
            """
            SELECT
                COUNT(*) as days,
                MIN(date) as start_date,
                MAX(date) as end_date,
                AVG(close) as avg_close,
                MIN(low) as min_price,
                MAX(high) as max_price
            FROM unity_daily_stock
        """
        ).fetchone()

        logger.info(f"\nSTOCK DATA:")
        logger.info(f"  Trading days: {stock_stats[0]}")
        logger.info(f"  Date range: {stock_stats[1]} to {stock_stats[2]}")
        logger.info(f"  Price range: ${stock_stats[4]:.2f} - ${stock_stats[5]:.2f}")
        logger.info(f"  Average close: ${stock_stats[3]:.2f}")

        # Options summary
        options_stats = self.conn.execute(
            """
            SELECT
                COUNT(DISTINCT date) as days,
                COUNT(*) as total_contracts,
                COUNT(DISTINCT expiration) as expirations,
                COUNT(DISTINCT strike) as strikes,
                MIN(date) as start_date,
                MAX(date) as end_date
            FROM unity_daily_options
        """
        ).fetchone()

        logger.info(f"\nOPTIONS DATA:")
        logger.info(f"  Trading days: {options_stats[0]}")
        logger.info(f"  Total contracts: {options_stats[1]:,}")
        logger.info(f"  Unique expirations: {options_stats[2]}")
        logger.info(f"  Unique strikes: {options_stats[3]}")
        logger.info(f"  Date range: {options_stats[4]} to {options_stats[5]}")

        # Sample options chain
        logger.info("\nSAMPLE OPTIONS CHAIN (latest date):")
        sample = self.conn.execute(
            """
            SELECT strike, option_type, bid, ask, (bid+ask)/2 as mid
            FROM unity_daily_options
            WHERE date = (SELECT MAX(date) FROM unity_daily_options)
            AND expiration > date + INTERVAL '20 days'
            AND expiration < date + INTERVAL '50 days'
            ORDER BY ABS(strike - (SELECT close FROM unity_daily_stock ORDER BY date DESC LIMIT 1))
            LIMIT 10
        """
        ).fetchall()

        logger.info("  Strike  Type  Bid    Ask    Mid")
        logger.info("  ------  ----  -----  -----  -----")
        for row in sample:
            logger.info(
                f"  ${row[0]:6.2f} {row[1]:4s}  ${row[2]:5.2f}  ${row[3]:5.2f}  ${row[4]:5.2f}"
            )

        logger.info("\n✓ All data is REAL from Databento - NO SYNTHETIC DATA")

    def cleanup(self):
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()


def main():
    """Main entry point."""
    downloader = UnityDailyDataDownloader()

    try:
        downloader.download_all_daily_data()
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
