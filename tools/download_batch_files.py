#!/usr/bin/env python3
"""
Download Unity options batch files and convert to DuckDB.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import databento as db

from src.unity_wheel.data_providers.databento import DatabentoClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def download_batch_files(job_id: str, specific_file: str = None):
    """Download batch files and convert to DuckDB."""

    # Initialize client
    client = DatabentoClient()

    # Create download directory
    download_dir = Path("unity_options_batch_data")
    download_dir.mkdir(exist_ok=True)

    logger.info(f"Downloading batch job: {job_id}")
    if specific_file:
        logger.info(f"Specific file: {specific_file}")
    logger.info(f"Download directory: {download_dir}")
    logger.info("=" * 60)

    try:
        # Download files
        logger.info("Downloading files from Databento...")

        files = client.client.batch.download(
            job_id=job_id,
            output_dir=str(download_dir),
            filename_to_download=specific_file,
        )

        logger.info(f"Downloaded {len(files)} files:")
        for file in files:
            logger.info(f"  {file}")

        # Find data files (not metadata)
        data_files = [f for f in files if f.suffix in [".dbn", ".zst"] and "metadata" not in f.name]

        if not data_files:
            logger.warning("No data files found to process")
            return

        # Process data files and load into DuckDB
        logger.info("\nProcessing data files and loading into DuckDB...")

        # Database connection
        db_path = Path("~/.wheel_trading/cache/wheel_cache.duckdb").expanduser()
        conn = duckdb.connect(str(db_path))

        # Clear existing data (since this is a complete dataset)
        logger.info("Clearing existing Unity options data...")
        conn.execute("DELETE FROM unity_options_daily")
        conn.commit()

        total_records = 0

        for data_file in data_files:
            logger.info(f"\nProcessing: {data_file}")

            try:
                # Read DBN file
                store = db.DBNStore.from_file(str(data_file))
                df = store.to_df()

                if df.empty:
                    logger.warning(f"  No data in {data_file}")
                    continue

                logger.info(f"  Records: {len(df):,}")

                # Process each record
                records_inserted = 0
                for ts_event, row in df.iterrows():
                    try:
                        symbol = row["symbol"]

                        # Skip if not Unity option
                        if not symbol.startswith("U ") or len(symbol) < 21:
                            continue

                        # Parse OSI symbol
                        exp_str = symbol[6:12]
                        option_type = symbol[12]
                        strike_str = symbol[13:21]

                        expiration = datetime.strptime("20" + exp_str, "%Y%m%d").date()
                        strike = float(strike_str) / 1000
                        trade_date = pd.Timestamp(ts_event).date()

                        # Insert record
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO unity_options_daily
                            (date, symbol, expiration, strike, option_type,
                             open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                trade_date,
                                symbol,
                                expiration,
                                strike,
                                option_type,
                                row.get("open"),
                                row.get("high"),
                                row.get("low"),
                                row.get("close"),
                                row.get("volume", 0),
                            ),
                        )

                        records_inserted += 1

                    except Exception as e:
                        logger.debug(f"Failed to process record: {e}")
                        continue

                # Commit batch
                conn.commit()
                total_records += records_inserted
                logger.info(f"  Inserted: {records_inserted:,} records")

            except Exception as e:
                logger.error(f"Error processing {data_file}: {e}")
                continue

        # Show final summary
        logger.info("\n" + "=" * 60)
        logger.info("UNITY OPTIONS BATCH DOWNLOAD COMPLETE")
        logger.info("=" * 60)

        # Get final stats
        stats = conn.execute(
            """
            SELECT
                COUNT(DISTINCT date) as trading_days,
                COUNT(DISTINCT symbol) as unique_contracts,
                COUNT(*) as total_records,
                MIN(date) as first_date,
                MAX(date) as last_date,
                SUM(volume) as total_volume
            FROM unity_options_daily
        """
        ).fetchone()

        if stats and stats[2] > 0:
            logger.info(f"Final Statistics:")
            logger.info(f"  Trading days: {stats[0]}")
            logger.info(f"  Unique contracts: {stats[1]:,}")
            logger.info(f"  Total records: {stats[2]:,}")
            logger.info(f"  Date range: {stats[3]} to {stats[4]}")
            logger.info(f"  Total volume: {stats[5]:,}" if stats[5] else "  Total volume: N/A")

            # Calculate coverage
            if stats[3] and stats[4]:
                start = datetime.strptime(str(stats[3]), "%Y-%m-%d")
                end = datetime.strptime(str(stats[4]), "%Y-%m-%d")
                years = (end - start).days / 365.25

                logger.info(f"  Years of data: {years:.1f}")

                if years >= 2.5:
                    logger.info("\n✅ SUCCESS: Downloaded ~3 years of Unity options data!")
                    logger.info("✅ All data is REAL from Databento OPRA feed")
                    logger.info("✅ Data now available in DuckDB for trading analysis")
                else:
                    logger.info(f"\n⚠️  Got {years:.1f} years of data")

        conn.close()

    except Exception as e:
        logger.error(f"Error downloading batch files: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python download_batch_files.py JOB_ID [SPECIFIC_FILE]")
        sys.exit(1)

    job_id = sys.argv[1]
    specific_file = sys.argv[2] if len(sys.argv) > 2 else None

    download_batch_files(job_id, specific_file)
