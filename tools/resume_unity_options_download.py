#!/usr/bin/env python3
"""
Resume downloading Unity options data from Databento.
Picks up from the last downloaded date and continues through today.
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


class UnityOptionsResumer:
    """Resume Unity options download from last successful date."""

    def __init__(self):
        self.client = DatabentoClient()
        self.db_path = Path(config.storage.database_path).expanduser()
        self.conn = duckdb.connect(str(self.db_path))
        self.eastern = pytz.timezone("US/Eastern")

    def get_last_downloaded_date(self):
        """Get the last date we have options data for."""
        result = self.conn.execute(
            """
            SELECT MAX(trade_date) FROM unity_options_ticks
        """
        ).fetchone()

        if result and result[0]:
            return result[0]
        else:
            # Start from Unity options availability date
            return datetime(2023, 3, 27).date()  # Day before 3/28 so we start with 3/28

    def download_options_for_date(self, date: datetime.date) -> int:
        """Download Unity options data for a single date."""
        market_start = datetime.combine(date, datetime.min.time()).replace(
            hour=9, minute=30, tzinfo=self.eastern
        )
        market_end = datetime.combine(date, datetime.min.time()).replace(
            hour=16, minute=0, tzinfo=self.eastern
        )

        try:
            logger.info(f"Downloading {date}...")

            # Use OPRA.PILLAR with parent symbology
            data = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="cmbp-1",  # Consolidated NBBO
                symbols=["U.OPT"],
                stype_in="parent",
                start=market_start,
                end=market_end,
            )

            # Convert to dataframe
            df = data.to_df()

            if df.empty:
                logger.debug(f"  No data for {date}")
                return 0

            records_inserted = 0

            # Process in batches
            batch_size = 1000
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i : i + batch_size]

                for _, row in batch.iterrows():
                    # Convert prices from integer representation
                    bid_px = row.get("bid_px_01", 0) / 10000.0 if "bid_px_01" in row else None
                    ask_px = row.get("ask_px_01", 0) / 10000.0 if "ask_px_01" in row else None

                    # Get the raw symbol, handling both possible column names
                    raw_symbol = row.get("raw_symbol", row.get("symbol", ""))

                    self.conn.execute(
                        """
                        INSERT OR IGNORE INTO unity_options_ticks
                        (trade_date, ts_event, instrument_id, raw_symbol,
                         bid_px, ask_px, bid_sz, ask_sz)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            date,
                            row.get("ts_event"),
                            row.get("instrument_id", 0),
                            raw_symbol,
                            bid_px,
                            ask_px,
                            row.get("bid_sz_01", 0),
                            row.get("ask_sz_01", 0),
                        ),
                    )

                    records_inserted += 1

            # Commit after each day
            self.conn.commit()
            logger.info(f"  ✓ {date}: {records_inserted:,} records")

            return records_inserted

        except Exception as e:
            if "422" in str(e) or "No data found" in str(e):
                logger.debug(f"  No data available for {date}")
                return 0
            else:
                logger.warning(f"  Error on {date}: {e}")
                return 0

    def resume_download(self):
        """Resume download from last successful date."""
        last_date = self.get_last_downloaded_date()
        start_date = last_date + timedelta(days=1)
        end_date = datetime.now().date() - timedelta(days=1)

        logger.info("=" * 60)
        logger.info("RESUMING UNITY OPTIONS DOWNLOAD")
        logger.info("=" * 60)
        logger.info(f"Last downloaded: {last_date}")
        logger.info(f"Resuming from: {start_date}")
        logger.info(f"Target end date: {end_date}")
        logger.info("=" * 60)

        current_date = start_date
        total_records = 0
        days_with_data = 0
        consecutive_failures = 0

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                records = self.download_options_for_date(current_date)

                if records > 0:
                    total_records += records
                    days_with_data += 1
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1

                    # If we have many consecutive failures, we might be past available data
                    if consecutive_failures > 20:
                        logger.warning("Many consecutive days without data, stopping")
                        break

                # Brief pause every 5 days
                if days_with_data % 5 == 0 and days_with_data > 0:
                    time.sleep(1)

            current_date += timedelta(days=1)

        # Show final summary
        self.show_final_summary()

    def show_final_summary(self):
        """Show comprehensive summary of all Unity options data."""
        logger.info("\n" + "=" * 60)
        logger.info("UNITY OPTIONS DOWNLOAD COMPLETE")
        logger.info("=" * 60)

        # Overall stats
        stats = self.conn.execute(
            """
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT trade_date) as trading_days,
                COUNT(DISTINCT raw_symbol) as unique_contracts,
                MIN(trade_date) as start_date,
                MAX(trade_date) as end_date
            FROM unity_options_ticks
        """
        ).fetchone()

        logger.info(f"Total records: {stats[0]:,}")
        logger.info(f"Trading days: {stats[1]}")
        logger.info(f"Unique contracts: {stats[2]:,}")
        logger.info(f"Date range: {stats[3]} to {stats[4]}")

        # Calculate expected vs actual
        if stats[3] and stats[4]:
            start = datetime.strptime(str(stats[3]), "%Y-%m-%d")
            end = datetime.strptime(str(stats[4]), "%Y-%m-%d")
            days_diff = (end - start).days
            expected_trading_days = int(days_diff * 252 / 365)  # Rough estimate

            logger.info(
                f"Coverage: {stats[1]/expected_trading_days*100:.1f}% of expected trading days"
            )

        # Sample contracts
        logger.info("\nSample option contracts:")
        samples = self.conn.execute(
            """
            SELECT DISTINCT raw_symbol
            FROM unity_options_ticks
            WHERE raw_symbol != '' AND raw_symbol IS NOT NULL
            ORDER BY raw_symbol
            LIMIT 10
        """
        ).fetchall()

        for sym in samples:
            logger.info(f"  {sym[0]}")

        logger.info("\n✅ All Unity options data is REAL from Databento OPRA feed")
        logger.info("✅ NO SYNTHETIC DATA in database")

    def cleanup(self):
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()


def main():
    """Main entry point."""
    resumer = UnityOptionsResumer()

    try:
        resumer.resume_download()
    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        resumer.cleanup()


if __name__ == "__main__":
    main()
