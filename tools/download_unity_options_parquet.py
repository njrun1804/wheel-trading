#!/usr/bin/env python3
"""
High-performance Unity options downloader - saves to Parquet files.
Downloads in parallel, saves to Parquet, then bulk loads to DuckDB.
"""

import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import databento as db
import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from unity_wheel.config.unified_config import get_config
config = get_config()


# Note: Using direct databento client to avoid import issues

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Enable Zstd multithreaded decompression
os.environ["DATABENTO_ZSTD_THREADS"] = str(os.cpu_count() or 4)

# Singleton Databento client
_CLIENT: Optional[db.Historical] = None


def get_databento_api_key():
    """Get Databento API key from Google Secret Manager."""
    try:
        from google.cloud import secretmanager

        client = secretmanager.SecretManagerServiceClient()
        name = "projects/wheel-strategy-202506/secrets/api-key/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logger.error(f"Could not get API key from Secret Manager: {e}")
        return None


def get_db_client() -> db.Historical:
    """Get or create singleton Databento client."""
    global _CLIENT

    if _CLIENT is None:
        # Get API key from environment or secret manager
        api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            logger.info("Getting API key from Secret Manager...")
            api_key = get_databento_api_key()

        if not api_key:
            logger.error("Could not get Databento API key")
            logger.info("Please set: export DATABENTO_API_KEY=your_key_here")
            raise ValueError("DATABENTO_API_KEY required")

        _CLIENT = db.Historical(api_key)
        logger.info("Databento client initialized")

    return _CLIENT


class ParquetUnityDownloader:
    """Download Unity options to Parquet files for maximum performance."""

    def __init__(self):
        self.client = get_db_client()

        # Create output directory
        self.output_dir = Path("~/.wheel_trading/unity_options_parquet").expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Thread-local storage
        self._local = threading.local()

        # Track progress
        self.completed_dates = []
        self.failed_dates = []
        self.lock = threading.Lock()

        logger.info(f"Output directory: {self.output_dir}")

    def download_single_date(self, date: datetime.date) -> Optional[str]:
        """Download Unity options for a single date and save to Parquet."""
        try:
            # Download data
            data = self.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                symbols=["U.OPT"],
                stype_in="parent",
                schema="ohlcv-1d",
                start=date.strftime("%Y-%m-%d"),
                end=(date + timedelta(days=1)).strftime("%Y-%m-%d"),
            )

            if not data:
                return None

            # Convert to DataFrame
            df = data.to_df()

            if df.empty:
                return None

            # Process the data
            processed_df = self.process_dataframe(df, date)

            if processed_df.empty:
                return None

            # Save to Parquet
            output_file = self.output_dir / f"unity_options_{date.strftime('%Y%m%d')}.parquet"

            # Write with compression
            processed_df.to_parquet(
                output_file, engine="pyarrow", compression="snappy", index=False  # Fast compression
            )

            with self.lock:
                self.completed_dates.append(date)

            return str(output_file)

        except Exception as e:
            logger.error(f"Error downloading {date}: {e}")
            with self.lock:
                self.failed_dates.append((date, str(e)))
            return None

    def process_dataframe(self, df: pd.DataFrame, date: datetime.date) -> pd.DataFrame:
        """Process raw Databento data into clean format."""
        processed_records = []

        for _, row in df.iterrows():
            try:
                symbol = row.get("symbol", "")

                if not symbol.startswith("U "):
                    continue

                # Parse OSI symbol: U     250620C00025000
                if len(symbol) >= 21:
                    exp_str = symbol[6:12]
                    expiration = datetime.strptime("20" + exp_str, "%Y%m%d").date()
                    option_type = symbol[12]
                    strike = float(symbol[13:21]) / 1000

                    processed_records.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "expiration": expiration,
                            "strike": strike,
                            "option_type": option_type,
                            "open": self.convert_price(row.get("open")),
                            "high": self.convert_price(row.get("high")),
                            "low": self.convert_price(row.get("low")),
                            "close": self.convert_price(row.get("close")),
                            "volume": row.get("volume", 0),
                            "trades_count": row.get("trades", 0),
                            "vwap": self.convert_price(row.get("vwap")),
                            "open_interest": row.get("open_interest", 0),
                        }
                    )

            except Exception as e:
                logger.debug(f"Failed to process row: {e}")
                continue

        return pd.DataFrame(processed_records)

    def convert_price(self, price) -> Optional[float]:
        """Convert Databento price format."""
        if price is None:
            return None
        try:
            if isinstance(price, (int, float)):
                if price > 10000:
                    return float(price) / 10000.0
                elif price > 1000:
                    return float(price) / 1000.0
                else:
                    return float(price)
        except:
            return None

    def download_all_parallel(self):
        """Download all Unity options data in parallel to Parquet files."""
        start_date = datetime(2023, 3, 28).date()
        end_date = datetime.now().date() - timedelta(days=1)

        logger.info("=" * 60)
        logger.info("UNITY OPTIONS DOWNLOAD TO PARQUET")
        logger.info("=" * 60)
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 60)

        # Build list of trading days
        trading_days = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Weekday
                trading_days.append(current)
            current += timedelta(days=1)

        logger.info(f"Total trading days: {len(trading_days)}")

        # Download in parallel
        start_time = time.time()
        max_workers = 8

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_date = {
                executor.submit(self.download_single_date, date): date for date in trading_days
            }

            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    result = future.result()
                    if result:
                        logger.info(f"✅ {date}: Saved to {Path(result).name}")
                    else:
                        logger.debug(f"   {date}: No data")

                except Exception as e:
                    logger.error(f"❌ {date}: {e}")

                completed += 1
                if completed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (len(trading_days) - completed) / rate
                    logger.info(
                        f"Progress: {completed}/{len(trading_days)} ({rate:.1f} days/sec, ETA: {eta/60:.1f} min)"
                    )

        duration = time.time() - start_time

        # Show summary
        logger.info("\n" + "=" * 60)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"Days processed: {len(trading_days)}")
        logger.info(f"Successful: {len(self.completed_dates)}")
        logger.info(f"Failed: {len(self.failed_dates)}")
        logger.info(f"Rate: {len(trading_days)/duration:.1f} days/second")

        if self.failed_dates:
            logger.warning(f"\nFailed dates:")
            for date, error in self.failed_dates[:10]:  # Show first 10
                logger.warning(f"  {date}: {error}")

        # List output files
        parquet_files = list(self.output_dir.glob("unity_options_*.parquet"))
        logger.info(f"\nCreated {len(parquet_files)} Parquet files")

        total_size = sum(f.stat().st_size for f in parquet_files) / (1024 * 1024)
        logger.info(f"Total size: {total_size:.1f} MB")

        return parquet_files

    def load_to_duckdb(self, parquet_files: List[Path]):
        """Bulk load all Parquet files into DuckDB."""
        logger.info("\n" + "=" * 60)
        logger.info("LOADING PARQUET FILES TO DUCKDB")
        logger.info("=" * 60)

        db_path = Path(config.storage.database_path).expanduser()
        conn = duckdb.connect(str(db_path))

        try:
            # Create table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS unity_options_parquet (
                    date DATE NOT NULL,
                    symbol VARCHAR NOT NULL,
                    expiration DATE NOT NULL,
                    strike DECIMAL(10,2) NOT NULL,
                    option_type VARCHAR(1) NOT NULL,
                    open DECIMAL(10,4),
                    high DECIMAL(10,4),
                    low DECIMAL(10,4),
                    close DECIMAL(10,4),
                    volume BIGINT,
                    trades_count INT,
                    vwap DECIMAL(10,4),
                    open_interest BIGINT,
                    PRIMARY KEY (date, symbol)
                )
            """
            )

            # Create indexes
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_unity_parquet_date_exp
                ON unity_options_parquet(date, expiration)
            """
            )

            # Bulk load all Parquet files
            start_time = time.time()

            # Use DuckDB's native Parquet reader - SUPER FAST!
            parquet_pattern = str(self.output_dir / "unity_options_*.parquet")

            logger.info(f"Loading {len(parquet_files)} files...")

            conn.execute(
                f"""
                INSERT INTO unity_options_parquet
                SELECT * FROM read_parquet('{parquet_pattern}')
            """
            )

            load_time = time.time() - start_time

            # Get statistics
            stats = conn.execute(
                """
                SELECT
                    COUNT(*) as total_records,
                    COUNT(DISTINCT date) as trading_days,
                    COUNT(DISTINCT symbol) as unique_contracts,
                    MIN(date) as first_date,
                    MAX(date) as last_date,
                    SUM(volume) as total_volume
                FROM unity_options_parquet
            """
            ).fetchone()

            logger.info(f"\nLoad complete in {load_time:.1f} seconds!")
            logger.info(f"Records: {stats[0]:,}")
            logger.info(f"Trading days: {stats[1]}")
            logger.info(f"Unique contracts: {stats[2]:,}")
            logger.info(f"Date range: {stats[3]} to {stats[4]}")
            logger.info(f"Total volume: {stats[5]:,}")

            # Validate against expected
            if stats[1] > 0 and stats[5]:
                daily_volume = stats[5] / stats[1]
                logger.info(f"\nDaily average volume: {daily_volume:,.0f}")
                logger.info(f"Expected (CBOE): ~55,000")

                if daily_volume > 40000:
                    logger.info("✅ EXCELLENT! Data matches expectations")
                elif daily_volume > 20000:
                    logger.info("✅ Good coverage")

        finally:
            conn.close()


def main():
    """Main entry point."""
    downloader = ParquetUnityDownloader()

    try:
        # Download all data to Parquet files
        parquet_files = downloader.download_all_parallel()

        if parquet_files:
            # Bulk load into DuckDB
            downloader.load_to_duckdb(parquet_files)

            logger.info("\n✅ SUCCESS! Unity options data downloaded and loaded")
            logger.info("✅ Parquet files saved for backup/reprocessing")
            logger.info("✅ All data is REAL from Databento OPRA feed")
        else:
            logger.warning("No Parquet files created")

    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
