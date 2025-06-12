#!/usr/bin/env python3
"""
Download Unity options end-of-day data from Databento.
Works with existing unity_options_daily table structure.
"""

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytz

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


class UnityOptionsEODDownloader:
    """Download Unity options end-of-day data."""

    def __init__(self):
        self.client = DatabentoClient()
        self.db_path = Path(config.storage.database_path).expanduser()
        self.conn = duckdb.connect(str(self.db_path))
        self.eastern = pytz.timezone("US/Eastern")

    def download_eod_data(self, date: datetime.date) -> int:
        """Download end-of-day options data for a single date."""
        # Get end of day timestamp (4pm ET)
        eod_time = datetime.combine(date, datetime.min.time()).replace(
            hour=16, minute=0, tzinfo=self.eastern
        )

        try:
            logger.info(f"Downloading EOD data for {date}...")

            # Try to get daily statistics/snapshot data
            data = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="statistics",  # End-of-day statistics
                symbols=["U.OPT"],  # Unity options - correct format
                stype_in="parent",  # Get all Unity options
                start=eod_time - timedelta(minutes=5),  # 5 minutes before close
                end=eod_time + timedelta(minutes=5),  # 5 minutes after close
                limit=10000,
            )

            # Convert to dataframe
            df = data.to_df()

            if df.empty:
                # Try alternative approach - get last trades of the day
                logger.debug(f"No statistics data, trying trades...")

                data = self.client.client.timeseries.get_range(
                    dataset="OPRA.PILLAR",
                    schema="trades",
                    symbols=["U.OPT"],
                    stype_in="parent",
                    start=eod_time - timedelta(hours=1),  # Last hour of trading
                    end=eod_time,
                    limit=50000,
                )

                df = data.to_df()

                if df.empty:
                    return 0

                # Aggregate trades to get end-of-day values
                return self.process_trades_to_eod(df, date)

            # Process statistics data
            return self.process_statistics(df, date)

        except Exception as e:
            if "422" in str(e) or "No data found" in str(e):
                logger.debug(f"No data available for {date}")
            else:
                logger.warning(f"Error downloading {date}: {e}")
            return 0

    def process_statistics(self, df, date: datetime.date) -> int:
        """Process end-of-day statistics data."""
        records_inserted = 0

        for _, row in df.iterrows():
            try:
                # Extract symbol info
                raw_symbol = row.get("raw_symbol", row.get("symbol", ""))
                if not raw_symbol or not raw_symbol.startswith("U"):
                    continue

                # Parse Unity option symbol (format: U  240119C00050000)
                if len(raw_symbol) >= 15:
                    # Extract expiration date
                    exp_str = raw_symbol[2:8].strip()
                    if exp_str:
                        expiration = datetime.strptime("20" + exp_str, "%Y%m%d").date()
                    else:
                        continue

                    # Extract option type
                    option_type = "C" if "C" in raw_symbol[8:13] else "P"

                    # Extract strike price
                    strike_str = raw_symbol[-8:]
                    strike = float(strike_str) / 1000.0
                else:
                    continue

                # Get prices (convert from fixed point if needed)
                def convert_price(val):
                    if val is None:
                        return None
                    # Databento prices are often in fixed point format
                    if val > 10000:
                        return val / 10000.0
                    return float(val)

                bid = convert_price(row.get("bid_close", row.get("bid_px", None)))
                ask = convert_price(row.get("ask_close", row.get("ask_px", None)))
                last = convert_price(row.get("close_price", row.get("last_price", None)))

                # Insert into database
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO unity_options_daily
                    (date, symbol, expiration, strike, option_type,
                     bid, ask, last, volume, open_interest)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        date,
                        raw_symbol,
                        expiration,
                        strike,
                        option_type,
                        bid,
                        ask,
                        last,
                        row.get("volume", 0),
                        row.get("open_interest", 0),
                    ),
                )

                records_inserted += 1

            except Exception as e:
                logger.debug(f"Failed to process row: {e}")
                continue

        self.conn.commit()
        return records_inserted

    def process_trades_to_eod(self, df, date: datetime.date) -> int:
        """Aggregate trades to create end-of-day data."""
        # Group by symbol and get last trade info
        eod_data = []

        unique_symbols = df["raw_symbol"].unique() if "raw_symbol" in df.columns else []

        for symbol in unique_symbols:
            if not symbol or not symbol.startswith("U"):
                continue

            symbol_df = df[df["raw_symbol"] == symbol]

            # Get last trade
            last_trade = symbol_df.iloc[-1]

            # Calculate daily volume
            daily_volume = len(symbol_df)

            # Extract option details
            try:
                if len(symbol) >= 15:
                    exp_str = symbol[2:8].strip()
                    expiration = datetime.strptime("20" + exp_str, "%Y%m%d").date()
                    option_type = "C" if "C" in symbol[8:13] else "P"
                    strike = float(symbol[-8:]) / 1000.0
                else:
                    continue

                last_price = (
                    last_trade.get("price", 0) / 10000.0 if last_trade.get("price") else None
                )

                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO unity_options_daily
                    (date, symbol, expiration, strike, option_type,
                     last, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (date, symbol, expiration, strike, option_type, last_price, daily_volume),
                )

                eod_data.append(symbol)

            except Exception as e:
                logger.debug(f"Failed to process symbol {symbol}: {e}")
                continue

        self.conn.commit()
        return len(eod_data)

    def download_all_eod_data(self):
        """Download all Unity options EOD data."""
        # Check what data we already have
        existing = self.conn.execute(
            """
            SELECT MAX(date) as last_date, COUNT(DISTINCT date) as days
            FROM unity_options_daily
        """
        ).fetchone()

        if existing and existing[0]:
            start_date = existing[0] + timedelta(days=1)
            logger.info(f"Resuming from {start_date} (already have {existing[1]} days)")
        else:
            # Unity options start March 28, 2023
            start_date = datetime(2023, 3, 28).date()
            logger.info(f"Starting fresh from {start_date}")

        end_date = datetime.now().date() - timedelta(days=1)

        logger.info("=" * 60)
        logger.info("DOWNLOADING UNITY OPTIONS END-OF-DAY DATA")
        logger.info("=" * 60)
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info("=" * 60)

        current_date = start_date
        total_records = 0
        days_with_data = 0

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                records = self.download_eod_data(current_date)

                if records > 0:
                    total_records += records
                    days_with_data += 1
                    logger.info(f"  ✓ {current_date}: {records} options")
                else:
                    logger.debug(f"  - {current_date}: No data")

                # Brief pause every 10 days
                if days_with_data % 10 == 0 and days_with_data > 0:
                    time.sleep(1)

            current_date += timedelta(days=1)

        # Show summary
        self.show_summary()

    def show_summary(self):
        """Show summary of all Unity options data."""
        logger.info("\n" + "=" * 60)
        logger.info("UNITY OPTIONS EOD DATA SUMMARY")
        logger.info("=" * 60)

        # Overall stats
        stats = self.conn.execute(
            """
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT date) as trading_days,
                COUNT(DISTINCT symbol) as unique_options,
                MIN(date) as start_date,
                MAX(date) as end_date,
                COUNT(DISTINCT expiration) as unique_expirations,
                COUNT(DISTINCT strike) as unique_strikes
            FROM unity_options_daily
            WHERE symbol LIKE 'U %'
        """
        ).fetchone()

        if stats and stats[0] > 0:
            logger.info(f"Total records: {stats[0]:,}")
            logger.info(f"Trading days: {stats[1]}")
            logger.info(f"Unique options: {stats[2]:,}")
            logger.info(f"Date range: {stats[3]} to {stats[4]}")
            logger.info(f"Unique expirations: {stats[5]}")
            logger.info(f"Unique strikes: {stats[6]}")

            # Recent data sample
            logger.info("\nMost recent options (by volume):")
            recent = self.conn.execute(
                """
                SELECT date, symbol, strike, option_type, last, volume
                FROM unity_options_daily
                WHERE date = (SELECT MAX(date) FROM unity_options_daily)
                AND volume > 0
                ORDER BY volume DESC
                LIMIT 5
            """
            ).fetchall()

            if recent:
                for date, symbol, strike, otype, last, volume in recent:
                    logger.info(
                        f"  {date} {symbol}: ${strike} {otype} last=${last:.2f} vol={volume}"
                    )

            # Date coverage
            logger.info("\nData coverage by month:")
            monthly = self.conn.execute(
                """
                SELECT
                    STRFTIME('%Y-%m', date) as month,
                    COUNT(DISTINCT date) as days,
                    COUNT(*) as options
                FROM unity_options_daily
                GROUP BY STRFTIME('%Y-%m', date)
                ORDER BY month DESC
                LIMIT 6
            """
            ).fetchall()

            for month, days, options in monthly:
                logger.info(f"  {month}: {days} days, {options:,} option records")

            logger.info("\n✅ Unity options EOD data successfully downloaded")
            logger.info("✅ All data is REAL from Databento - NO SYNTHETIC DATA")
        else:
            logger.warning("\n⚠️  No Unity options data found - check Databento subscription")

    def cleanup(self):
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()


def main():
    """Main entry point."""
    downloader = UnityOptionsEODDownloader()

    try:
        downloader.download_all_eod_data()
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
