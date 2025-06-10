#!/usr/bin/env python3
"""
Download complete daily Unity options data using CMBP-1 schema.
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

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class UnityDailyOptionsDownloader:
    """Download Unity options using CMBP-1 schema for daily data."""

    def __init__(self):
        self.client = DatabentoClient()
        self.db_path = Path("~/.wheel_trading/cache/wheel_cache.duckdb").expanduser()
        self.conn = duckdb.connect(str(self.db_path))
        self.eastern = pytz.timezone("US/Eastern")

        # Clear the existing incomplete data
        self.conn.execute("DELETE FROM unity_options_daily")
        self.conn.commit()
        logger.info("Cleared existing incomplete options data")

    def parse_option_symbol(self, symbol):
        """Parse Unity option symbol to extract details."""
        if not symbol or len(symbol) < 15 or not symbol.startswith("U"):
            return None, None, None

        try:
            # Unity format: "U     250613C00032000"
            # Find where the date starts (after spaces)
            for i in range(1, len(symbol)):
                if symbol[i].isdigit():
                    exp_str = symbol[i : i + 6]
                    if len(exp_str) == 6 and exp_str.isdigit():
                        expiration = datetime.strptime("20" + exp_str, "%Y%m%d").date()
                    break
            else:
                return None, None, None

            # Extract type (C or P)
            if "C" in symbol:
                option_type = "C"
                type_pos = symbol.index("C")
            elif "P" in symbol:
                option_type = "P"
                type_pos = symbol.index("P")
            else:
                return None, None, None

            # Extract strike (8 digits after C/P)
            if type_pos + 8 < len(symbol):
                strike_str = symbol[type_pos + 1 : type_pos + 9]
                if strike_str.isdigit():
                    strike = float(strike_str) / 1000.0
                else:
                    return None, None, None
            else:
                return None, None, None

            return expiration, strike, option_type

        except Exception:
            return None, None, None

    def download_day(self, date: datetime.date) -> int:
        """Download Unity options for a single day using CMBP-1."""
        # Use end of trading day for snapshot
        eod_time = datetime.combine(date, datetime.min.time()).replace(
            hour=16, minute=0, tzinfo=self.eastern
        )

        try:
            data = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="cmbp-1",  # Consolidated market by price
                symbols=["U.OPT"],
                stype_in="parent",
                start=eod_time - timedelta(minutes=30),  # 30 min before close
                end=eod_time,  # At close
                limit=5000,  # Get more data
            )

            df = data.to_df()

            if df.empty:
                return 0

            records_inserted = 0

            for _, row in df.iterrows():
                symbol = row.get("symbol", row.get("raw_symbol", ""))

                # Parse option details
                expiration, strike, option_type = self.parse_option_symbol(symbol)

                if expiration and strike and option_type:
                    # Convert prices (CMBP uses fixed point)
                    def convert_price(val):
                        if val is None or val == 0:
                            return None
                        if val > 1000:
                            return val / 10000.0
                        return float(val)

                    # Get bid/ask from CMBP data
                    bid = convert_price(row.get("bid_px_00"))
                    ask = convert_price(row.get("ask_px_00"))

                    # Use mid price as "last" if no trade price
                    if bid and ask:
                        last = (bid + ask) / 2
                    else:
                        last = None

                    volume = row.get("bid_sz_00", 0) + row.get("ask_sz_00", 0)

                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO unity_options_daily
                        (date, symbol, expiration, strike, option_type,
                         bid, ask, last, volume, open_interest)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            date,
                            symbol,
                            expiration,
                            strike,
                            option_type,
                            bid,
                            ask,
                            last,
                            volume,
                            0,  # open_interest not in CMBP
                        ),
                    )

                    records_inserted += 1

            if records_inserted > 0:
                self.conn.commit()
                logger.info(f"  ✓ {date}: {records_inserted} options")

            return records_inserted

        except Exception as e:
            if "422" in str(e) or "No data" in str(e):
                return 0
            else:
                logger.warning(f"Error downloading {date}: {e}")
                return 0

    def download_all_daily_data(self):
        """Download all Unity options daily data from March 2023."""
        # Start from Unity options availability
        start_date = datetime(2023, 3, 28).date()
        end_date = datetime.now().date() - timedelta(days=1)

        logger.info("=" * 70)
        logger.info("DOWNLOADING DAILY UNITY OPTIONS DATA")
        logger.info("=" * 70)
        logger.info(f"Using CMBP-1 schema for end-of-day snapshots")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info("=" * 70)

        current_date = start_date
        total_records = 0
        days_with_data = 0

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                records = self.download_day(current_date)

                if records > 0:
                    total_records += records
                    days_with_data += 1

                # Brief pause every 10 days
                if days_with_data % 10 == 0 and days_with_data > 0:
                    time.sleep(1)
                    logger.info(
                        f"Progress: {days_with_data} days processed, {total_records:,} total options"
                    )

            current_date += timedelta(days=1)

        # Show final summary
        self.show_summary()

    def show_summary(self):
        """Show comprehensive summary."""
        logger.info("\n" + "=" * 70)
        logger.info("DAILY UNITY OPTIONS DOWNLOAD COMPLETE")
        logger.info("=" * 70)

        stats = self.conn.execute(
            """
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT date) as trading_days,
                COUNT(DISTINCT symbol) as unique_options,
                MIN(date) as start_date,
                MAX(date) as end_date,
                COUNT(DISTINCT strike) as unique_strikes,
                COUNT(DISTINCT expiration) as unique_expirations
            FROM unity_options_daily
        """
        ).fetchone()

        logger.info(f"Total records: {stats[0]:,}")
        logger.info(f"Trading days: {stats[1]} (SHOULD BE ~500+ for daily data)")
        logger.info(f"Unique options: {stats[2]:,}")
        logger.info(f"Date range: {stats[3]} to {stats[4]}")
        logger.info(f"Strike range: {stats[5]} different strikes")
        logger.info(f"Expirations: {stats[6]} different expiry dates")

        # Check coverage by month
        logger.info("\nMonthly coverage:")
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
            logger.info(f"  {month}: {days} days, {options:,} options")

        if stats[1] > 400:  # Should have ~500+ trading days
            logger.info("\n✅ SUCCESS: Got daily Unity options data!")
            logger.info("✅ This is proper daily coverage for wheel strategy backtesting")
        else:
            logger.info(f"\n⚠️  Only {stats[1]} trading days - still missing daily data")

    def cleanup(self):
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()


def main():
    """Main entry point."""
    downloader = UnityDailyOptionsDownloader()

    try:
        downloader.download_all_daily_data()
    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted")
        downloader.show_summary()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        downloader.cleanup()


if __name__ == "__main__":
    main()
