#!/usr/bin/env python3
"""
Resume Unity options download from where it left off.
Checks what dates are already downloaded and continues.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.download_unity_options_only import UnityOptionsOnlyDownloader

from unity_wheel.config.unified_config import get_config
config = get_config()


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


def check_download_progress():
    """Check current download progress."""
    db_path = Path(config.storage.database_path).expanduser()
    conn = duckdb.connect(str(db_path))

    # Get summary
    stats = conn.execute(
        """
        SELECT
            COUNT(DISTINCT trade_date) as days_downloaded,
            MIN(trade_date) as first_date,
            MAX(trade_date) as last_date,
            COUNT(*) as total_records
        FROM unity_options_ticks
    """
    ).fetchone()

    if stats[0] == 0:
        logger.info("No Unity options data downloaded yet")
        return None

    logger.info(f"\nCURRENT DOWNLOAD STATUS:")
    logger.info(f"Days downloaded: {stats[0]}")
    logger.info(f"Date range: {stats[1]} to {stats[2]}")
    logger.info(f"Total records: {stats[3]:,}")

    # Get list of downloaded dates
    downloaded_dates = conn.execute(
        """
        SELECT DISTINCT trade_date
        FROM unity_options_ticks
        ORDER BY trade_date
    """
    ).fetchall()

    downloaded_dates = [row[0] for row in downloaded_dates]

    conn.close()

    return stats[2]  # Return last downloaded date


def resume_download():
    """Resume download from last successful date."""
    last_date = check_download_progress()

    if last_date is None:
        start_date = datetime(2023, 3, 28).date()
        logger.info(f"\nStarting fresh download from {start_date}")
    else:
        start_date = last_date + timedelta(days=1)
        logger.info(f"\nResuming download from {start_date}")

    # Create downloader and modify start date
    downloader = UnityOptionsOnlyDownloader()

    # Override the download method to start from our date
    end_date = datetime.now().date() - timedelta(days=1)
    current_date = start_date

    logger.info(f"Downloading Unity options from {start_date} to {end_date}")

    total_records = 0
    days_with_data = 0

    try:
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                logger.info(f"Processing {current_date}...")

                records = downloader.download_options_for_date(current_date)

                if records > 0:
                    total_records += records
                    days_with_data += 1
                    logger.info(f"  ✓ {current_date}: {records:,} records")
                else:
                    logger.debug(f"  - {current_date}: No data")

                # Brief pause every 5 days
                if days_with_data % 5 == 0 and days_with_data > 0:
                    import time

                    time.sleep(1)

            current_date += timedelta(days=1)

            # Save progress every 10 days
            if days_with_data % 10 == 0 and days_with_data > 0:
                logger.info(
                    f"\nPROGRESS: Downloaded {days_with_data} days, {total_records:,} total records"
                )

    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user")
        logger.info(f"Progress saved. Downloaded {days_with_data} additional days")
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.info(f"Progress saved. Downloaded {days_with_data} additional days")
    finally:
        downloader.verify_options_data()
        downloader.cleanup()


if __name__ == "__main__":
    resume_download()
