#!/usr/bin/env python3
"""
Clean up bad data from the database.
Removes invalid prices and corrupted records identified during API investigation.
"""

import logging
import sys
from pathlib import Path

import duckdb

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataCleanup:
    """Clean up bad data from Unity Wheel database."""

    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.db_path = (
            Path(__file__).parent.parent / "data" / "wheel_trading_optimized.duckdb"
        )
        self.conn = duckdb.connect(str(self.db_path))
        self.stats = {
            "zero_prices": 0,
            "wrong_prices": 0,
            "zero_options": 0,
            "total_cleaned": 0,
        }

    def clean_zero_prices(self):
        """Remove stock records with zero prices."""
        logger.info("🧹 Cleaning zero prices...")

        # Find zero price records
        zero_records = self.conn.execute(
            """
            SELECT date, symbol, close 
            FROM market.price_data 
            WHERE symbol = 'U' AND close = 0
            ORDER BY date
        """
        ).fetchall()

        if zero_records:
            logger.info(f"Found {len(zero_records)} records with zero prices")
            for date, _symbol, close in zero_records[:5]:  # Show first 5
                logger.info(f"  - {date}: ${close}")

            if not self.dry_run:
                # Delete zero price records
                self.conn.execute(
                    """
                    DELETE FROM market.price_data 
                    WHERE symbol = 'U' AND close = 0
                """
                )
                logger.info(f"✅ Deleted {len(zero_records)} zero price records")
            else:
                logger.info(
                    f"DRY RUN: Would delete {len(zero_records)} zero price records"
                )

            self.stats["zero_prices"] = len(zero_records)
        else:
            logger.info("No zero price records found")

    def clean_wrong_price_range(self):
        """Remove Unity prices outside valid range ($10-50 typical, $5-100 max)."""
        logger.info("🧹 Cleaning wrong price range data...")

        # Find records with prices > $100 (the 10x error from May-June)
        wrong_prices = self.conn.execute(
            """
            SELECT date, close, high 
            FROM market.price_data 
            WHERE symbol = 'U' 
            AND (close > 100 OR high > 100)
            ORDER BY date
        """
        ).fetchall()

        if wrong_prices:
            logger.info(f"Found {len(wrong_prices)} records with wrong prices (>$100)")

            # Show date range
            dates = [record[0] for record in wrong_prices]
            logger.info(f"Date range: {min(dates)} to {max(dates)}")

            # Show sample prices
            for date, close, high in wrong_prices[:5]:
                logger.info(f"  - {date}: close=${close:.2f}, high=${high:.2f}")

            if not self.dry_run:
                # Delete wrong price records
                self.conn.execute(
                    """
                    DELETE FROM market.price_data 
                    WHERE symbol = 'U' 
                    AND (close > 100 OR high > 100)
                """
                )
                logger.info(f"✅ Deleted {len(wrong_prices)} wrong price records")
            else:
                logger.info(
                    f"DRY RUN: Would delete {len(wrong_prices)} wrong price records"
                )

            self.stats["wrong_prices"] = len(wrong_prices)
        else:
            logger.info("No wrong price range records found")

    def clean_zero_option_quotes(self):
        """Remove option records where both bid and ask are zero."""
        logger.info("🧹 Cleaning zero bid/ask options...")

        # Find zero bid/ask options
        zero_options = self.conn.execute(
            """
            SELECT COUNT(*) as count,
                   MIN(timestamp) as first_date,
                   MAX(timestamp) as last_date
            FROM options.contracts 
            WHERE symbol = 'U' 
            AND bid = 0 
            AND ask = 0
        """
        ).fetchone()

        count = zero_options[0] if zero_options else 0

        if count > 0:
            logger.info(f"Found {count} options with zero bid/ask")
            if zero_options[1] and zero_options[2]:
                logger.info(f"Date range: {zero_options[1]} to {zero_options[2]}")

            # Show sample of affected options
            samples = self.conn.execute(
                """
                SELECT expiration, strike, option_type
                FROM options.contracts 
                WHERE symbol = 'U' AND bid = 0 AND ask = 0
                LIMIT 5
            """
            ).fetchall()

            for exp, strike, otype in samples:
                logger.info(f"  - {exp} ${strike} {otype}")

            if not self.dry_run:
                # Delete zero bid/ask options
                self.conn.execute(
                    """
                    DELETE FROM options.contracts 
                    WHERE symbol = 'U' 
                    AND bid = 0 
                    AND ask = 0
                """
                )
                logger.info(f"✅ Deleted {count} zero bid/ask option records")
            else:
                logger.info(
                    f"DRY RUN: Would delete {count} zero bid/ask option records"
                )

            self.stats["zero_options"] = count
        else:
            logger.info("No zero bid/ask options found")

    def verify_cleanup(self):
        """Verify the cleanup was successful."""
        logger.info("\n📊 Verifying cleanup results...")

        # Check remaining data quality
        price_stats = self.conn.execute(
            """
            SELECT 
                COUNT(*) as total_records,
                MIN(close) as min_price,
                MAX(close) as max_price,
                AVG(close) as avg_price,
                COUNT(CASE WHEN close = 0 THEN 1 END) as zero_prices,
                COUNT(CASE WHEN close > 100 THEN 1 END) as high_prices
            FROM market.price_data
            WHERE symbol = 'U'
        """
        ).fetchone()

        logger.info("Stock data after cleanup:")
        logger.info(f"  - Total records: {price_stats[0]}")
        logger.info(f"  - Price range: ${price_stats[1]:.2f} - ${price_stats[2]:.2f}")
        logger.info(f"  - Average price: ${price_stats[3]:.2f}")
        logger.info(f"  - Zero prices remaining: {price_stats[4]}")
        logger.info(f"  - Prices > $100: {price_stats[5]}")

        # Check option data quality
        option_stats = self.conn.execute(
            """
            SELECT 
                COUNT(*) as total_options,
                COUNT(CASE WHEN bid = 0 AND ask = 0 THEN 1 END) as zero_quotes,
                AVG(CASE WHEN bid > 0 THEN bid END) as avg_bid,
                AVG(CASE WHEN ask > 0 THEN ask END) as avg_ask
            FROM options.contracts
            WHERE symbol = 'U'
        """
        ).fetchone()

        logger.info("\nOption data after cleanup:")
        logger.info(f"  - Total records: {option_stats[0]}")
        logger.info(f"  - Zero bid/ask remaining: {option_stats[1]}")
        if option_stats[2]:
            logger.info(f"  - Average bid: ${option_stats[2]:.4f}")
        if option_stats[3]:
            logger.info(f"  - Average ask: ${option_stats[3]:.4f}")

    def show_recent_data(self):
        """Show recent data to verify quality."""
        logger.info("\n📈 Recent Unity stock data:")

        recent = self.conn.execute(
            """
            SELECT date, open, high, low, close, volume
            FROM market.price_data
            WHERE symbol = 'U'
            ORDER BY date DESC
            LIMIT 10
        """
        ).fetchall()

        for date, open_p, high, low, close, volume in recent:
            # Handle None values in OHLC data
            o_str = f"${open_p:.2f}" if open_p is not None else "N/A"
            h_str = f"${high:.2f}" if high is not None else "N/A"
            l_str = f"${low:.2f}" if low is not None else "N/A"
            c_str = f"${close:.2f}" if close is not None else "N/A"
            vol_str = f"{volume:,}" if volume else "N/A"
            logger.info(
                f"  {date}: O={o_str} H={h_str} L={l_str} C={c_str} Vol={vol_str}"
            )

    def generate_summary(self):
        """Generate cleanup summary."""
        total = sum(self.stats.values())
        self.stats["total_cleaned"] = total

        logger.info("\n" + "=" * 50)
        logger.info("CLEANUP SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Zero prices removed: {self.stats['zero_prices']}")
        logger.info(f"Wrong prices removed: {self.stats['wrong_prices']}")
        logger.info(f"Zero options removed: {self.stats['zero_options']}")
        logger.info(f"TOTAL RECORDS CLEANED: {total}")
        logger.info("=" * 50)

        if self.dry_run:
            logger.info("\n⚠️  This was a DRY RUN - no data was actually deleted")
            logger.info("Run with --execute to perform actual cleanup")

    def run(self):
        """Run the complete cleanup process."""
        logger.info("Starting Unity Wheel data cleanup...")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'EXECUTE'}")
        logger.info("")

        try:
            # Run cleanup steps
            self.clean_zero_prices()
            self.clean_wrong_price_range()
            self.clean_zero_option_quotes()

            # Verify results
            if not self.dry_run:
                self.verify_cleanup()
                self.show_recent_data()

            # Generate summary
            self.generate_summary()

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise
        finally:
            self.conn.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean up bad data from Unity Wheel database"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform cleanup (default is dry run)",
    )

    args = parser.parse_args()

    # Create and run cleanup
    cleanup = DataCleanup(dry_run=not args.execute)
    cleanup.run()

    # Exit with error if data was found that needs cleaning
    if not args.execute and cleanup.stats["total_cleaned"] > 0:
        logger.info("\n❗ Bad data found. Run with --execute to clean it up.")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
