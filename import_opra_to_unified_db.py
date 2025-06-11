#!/usr/bin/env python3
"""Import OPRA options data into unified DuckDB storage.

This script imports OPRA options data from Databento into the existing
unified storage structure used by the wheel trading system.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.unity_wheel.storage.storage import Storage, StorageConfig
from src.unity_wheel.utils import get_logger

logger = get_logger(__name__)

# Check for zstandard
try:
    import zstandard as zstd
except ImportError:
    print("❌ Error: zstandard package required for decompression")
    print("   Install with: pip install zstandard")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("❌ Error: pandas required for data processing")
    print("   Install with: pip install pandas")
    sys.exit(1)


class OPRAImporter:
    """Import OPRA options data into unified storage."""

    def __init__(self, storage: Storage):
        self.storage = storage
        self._decompressor = zstd.ZstdDecompressor()

    async def initialize_tables(self):
        """Create/update tables for OPRA data."""
        async with self.storage.cache.connection() as conn:
            # Create unified options data table if it doesn't exist
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS options_data (
                    ts_event TIMESTAMP NOT NULL,
                    instrument_id BIGINT NOT NULL,
                    symbol VARCHAR NOT NULL,           -- Parsed from raw_symbol
                    expiration DATE NOT NULL,          -- Parsed from raw_symbol
                    strike DECIMAL(10,2) NOT NULL,     -- Parsed from raw_symbol
                    option_type VARCHAR(4) NOT NULL,   -- PUT/CALL
                    open DECIMAL(10,4),
                    high DECIMAL(10,4),
                    low DECIMAL(10,4),
                    close DECIMAL(10,4),
                    volume BIGINT,
                    data_source VARCHAR DEFAULT 'OPRA',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (instrument_id, ts_event)
                )
            """
            )

            # Create symbology mapping table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS opra_symbology (
                    raw_symbol VARCHAR NOT NULL,
                    instrument_id BIGINT NOT NULL,
                    date DATE NOT NULL,
                    PRIMARY KEY (instrument_id, date)
                )
            """
            )

            # Create indexes for performance
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_options_symbol_date
                ON options_data(symbol, ts_event)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_options_strike_exp
                ON options_data(symbol, expiration, strike)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_options_type
                ON options_data(symbol, option_type, expiration)
            """
            )

            logger.info("Tables initialized for OPRA data import")

    def parse_opra_symbol(self, raw_symbol: str) -> Dict[str, any]:
        """Parse OPRA option symbol format.

        Format: SSSSSSYYMMDDTPPPPPPPP
        - SSSSSS: 6-char underlying symbol (right-padded)
        - YYMMDD: Expiration date
        - T: Option type (C=Call, P=Put)
        - PPPPPPPP: Strike price * 1000 (8 digits, left-padded)

        Example: U     250417P00039000
        """
        try:
            # Remove any extra whitespace
            symbol = raw_symbol.strip()

            # Parse components
            underlying = symbol[0:6].strip()  # Remove padding
            expiry_str = symbol[6:12]  # YYMMDD
            option_type = symbol[12]  # C or P
            strike_str = symbol[13:21]  # 8 digits

            # Convert expiration to date
            expiry_date = datetime.strptime("20" + expiry_str, "%Y%m%d").date()

            # Convert strike price (divided by 1000)
            strike = float(strike_str) / 1000.0

            return {
                "underlying": underlying,
                "expiration": expiry_date,
                "option_type": "CALL" if option_type == "C" else "PUT",
                "strike": strike,
            }
        except Exception as e:
            logger.error(f"Failed to parse symbol {raw_symbol}: {e}")
            return None

    def decompress_file(self, filepath: Path) -> bytes:
        """Decompress a .zst file."""
        with open(filepath, "rb") as f:
            return self._decompressor.decompress(f.read())

    async def import_symbology(self, symbology_path: Path) -> Dict[int, str]:
        """Import symbology mapping and return lookup dict."""
        logger.info(f"Importing symbology from {symbology_path}")

        # Read CSV
        df = pd.read_csv(symbology_path)

        # Store in database
        async with self.storage.cache.connection() as conn:
            # Use pandas to_sql equivalent in DuckDB
            conn.execute(
                """
                INSERT OR REPLACE INTO opra_symbology (raw_symbol, instrument_id, date)
                SELECT raw_symbol, instrument_id, date FROM df
            """
            )

        # Create lookup dict (instrument_id -> raw_symbol)
        # Use the most recent symbol for each instrument
        symbol_lookup = {}
        for _, row in df.iterrows():
            symbol_lookup[row["instrument_id"]] = row["raw_symbol"]

        logger.info(
            f"Imported {len(df)} symbology mappings, {len(symbol_lookup)} unique instruments"
        )
        return symbol_lookup

    async def import_ohlcv_file(
        self, filepath: Path, symbol_lookup: Dict[int, str], file_idx: int = 0
    ) -> int:
        """Import a single OHLCV file."""
        logger.info(f"Processing {filepath.name}")

        try:
            # Decompress file
            try:
                decompressed = self.decompress_file(filepath)
            except Exception as e:
                logger.error(f"Failed to decompress {filepath.name}: {e}")
                return 0

            # Debug decompression
            if file_idx == 0:  # First file
                logger.info(f"Decompressed size: {len(decompressed)} bytes")
                if len(decompressed) < 100:
                    logger.warning(f"Decompressed file seems too small: {len(decompressed)} bytes")
                    # Print first 100 chars
                    logger.info(f"First 100 bytes: {decompressed[:100]}")

            # Write to temporary file for pandas
            temp_path = f"/tmp/{filepath.stem}"
            with open(temp_path, "wb") as f:
                f.write(decompressed)

            # Read CSV
            try:
                df = pd.read_csv(temp_path)
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"Failed to read CSV: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return 0

            # Debug CSV reading
            if file_idx < 5:  # Debug first few files
                logger.info(f"CSV rows in {filepath.name}: {len(df)}")
                if len(df) > 0:
                    logger.info(f"CSV columns: {list(df.columns)}")
                    logger.info(f"First row: {df.iloc[0].to_dict()}")

            # Skip empty files (header only)
            if len(df) == 0:
                logger.info(f"Skipping empty file: {filepath.name}")
                return 0

            # Add parsed symbol information
            records = []
            skipped = 0
            unity_count = 0

            for idx, row in df.iterrows():
                instrument_id = row["instrument_id"]

                # Check if symbol field exists (newer format)
                if "symbol" in row and pd.notna(row["symbol"]):
                    raw_symbol = row["symbol"]
                elif instrument_id in symbol_lookup:
                    # Fall back to lookup
                    raw_symbol = symbol_lookup[instrument_id]
                else:
                    skipped += 1
                    continue

                # Debug first few Unity symbols
                if unity_count < 3 and raw_symbol.strip().startswith("U"):
                    logger.info(f"Unity symbol #{unity_count+1}: '{raw_symbol}'")

                parsed = self.parse_opra_symbol(raw_symbol)

                if not parsed:
                    skipped += 1
                    if unity_count < 3:
                        logger.warning(f"Failed to parse: '{raw_symbol}'")
                    continue

                # Import Unity options
                if parsed["underlying"] == "U":
                    unity_count += 1

                records.append(
                    {
                        "ts_event": pd.to_datetime(row["ts_event"]),
                        "instrument_id": instrument_id,
                        "symbol": parsed["underlying"],
                        "expiration": parsed["expiration"],
                        "strike": parsed["strike"],
                        "option_type": parsed["option_type"],
                        "open": row.get("open"),
                        "high": row.get("high"),
                        "low": row.get("low"),
                        "close": row.get("close"),
                        "volume": row.get("volume", 0),
                    }
                )

            if records:
                # Convert to DataFrame for bulk insert
                import_df = pd.DataFrame(records)

                async with self.storage.cache.connection() as conn:
                    # Insert using DuckDB's DataFrame support
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO options_data
                        (ts_event, instrument_id, symbol, expiration, strike,
                         option_type, open, high, low, close, volume)
                        SELECT ts_event, instrument_id, symbol, expiration, strike,
                               option_type, open, high, low, close, volume
                        FROM import_df
                    """
                    )

                logger.info(
                    f"Imported {len(records)} Unity options from {filepath.name} (found {unity_count} Unity, skipped {skipped})"
                )
                return len(records)
            else:
                logger.warning(
                    f"No Unity options found in {filepath.name} (examined {len(df)} rows, skipped {skipped})"
                )
                return 0

        except Exception as e:
            logger.error(f"Failed to import {filepath.name}: {e}")
            return 0

    async def import_all(self, data_dir: str) -> None:
        """Import all OPRA data files."""
        data_path = Path(data_dir)

        # Initialize tables
        await self.initialize_tables()

        # Import symbology first
        symbology_path = data_path / "symbology.csv"
        if not symbology_path.exists():
            logger.error(f"Symbology file not found: {symbology_path}")
            return

        symbol_lookup = await self.import_symbology(symbology_path)

        # Find all OHLCV files
        ohlcv_files = sorted(data_path.glob("*.ohlcv-1d.csv.zst"))
        logger.info(f"Found {len(ohlcv_files)} OHLCV files to import")

        # Import each file
        total_imported = 0
        for i, filepath in enumerate(ohlcv_files, 1):
            logger.info(f"Processing file {i}/{len(ohlcv_files)}")
            imported = await self.import_ohlcv_file(filepath, symbol_lookup, file_idx=i - 1)
            total_imported += imported

        # Get summary statistics
        await self.print_summary()

        logger.info(f"Import complete! Total Unity options imported: {total_imported:,}")

    async def print_summary(self):
        """Print summary statistics of imported data."""
        async with self.storage.cache.connection() as conn:
            # Overall stats
            stats = conn.execute(
                """
                SELECT
                    COUNT(*) as total_records,
                    COUNT(DISTINCT instrument_id) as unique_options,
                    COUNT(DISTINCT DATE(ts_event)) as trading_days,
                    MIN(DATE(ts_event)) as earliest_date,
                    MAX(DATE(ts_event)) as latest_date,
                    COUNT(DISTINCT expiration) as unique_expirations,
                    COUNT(DISTINCT strike) as unique_strikes
                FROM options_data
                WHERE symbol = 'U'
            """
            ).fetchone()

            print("\n📊 Import Summary:")
            print(f"   Total records: {stats[0]:,}")
            print(f"   Unique options: {stats[1]:,}")
            print(f"   Trading days: {stats[2]:,}")
            print(f"   Date range: {stats[3]} to {stats[4]}")
            print(f"   Unique expirations: {stats[5]:,}")
            print(f"   Unique strikes: {stats[6]:,}")

            # Sample recent puts
            print("\n📈 Sample Recent Unity Puts:")
            samples = conn.execute(
                """
                SELECT
                    DATE(ts_event) as date,
                    expiration,
                    strike,
                    close,
                    volume
                FROM options_data
                WHERE symbol = 'U'
                    AND option_type = 'PUT'
                    AND DATE(ts_event) >= '2025-01-01'
                    AND strike BETWEEN 30 AND 40
                ORDER BY ts_event DESC, strike
                LIMIT 5
            """
            ).fetchall()

            for date, exp, strike, close, vol in samples:
                exp_str = exp.strftime("%Y-%m-%d")
                print(f"   {date} - ${strike:.2f} put exp {exp_str}: ${close:.2f} (vol: {vol:,})")

            # Liquidity analysis
            print("\n💧 Liquidity Analysis (Last 30 Days):")
            liquidity = conn.execute(
                """
                SELECT
                    strike,
                    AVG(volume) as avg_volume,
                    COUNT(*) as data_points
                FROM options_data
                WHERE symbol = 'U'
                    AND option_type = 'PUT'
                    AND DATE(ts_event) >= CURRENT_DATE - INTERVAL 30 DAY
                GROUP BY strike
                HAVING avg_volume > 100
                ORDER BY avg_volume DESC
                LIMIT 10
            """
            ).fetchall()

            print("   Most liquid put strikes:")
            for strike, avg_vol, points in liquidity:
                print(f"   ${strike:.2f}: {avg_vol:.0f} avg volume ({points} days)")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Import OPRA options data into unified storage")
    parser.add_argument(
        "--data-dir",
        default="/Users/mikeedwards/Downloads/OPRA-20250611-NPFCXSAHG6",
        help="Directory containing OPRA data files",
    )
    parser.add_argument("--cache-dir", default="data/cache", help="Directory for DuckDB cache")

    args = parser.parse_args()

    # Check data directory exists
    if not Path(args.data_dir).exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Initialize storage
    from src.unity_wheel.storage.duckdb_cache import CacheConfig

    cache_config = CacheConfig(
        cache_dir=Path(args.cache_dir),
        max_size_gb=10.0,  # Allow larger size for options data
        ttl_days=365,  # Keep data for a year
    )

    storage_config = StorageConfig(cache_config=cache_config)
    storage = Storage(storage_config)

    # Initialize storage
    await storage.initialize()

    # Create importer and run import
    importer = OPRAImporter(storage)
    await importer.import_all(args.data_dir)

    print("\n✅ Import complete!")
    print(f"📁 Data location: {cache_config.cache_dir / 'wheel_cache.duckdb'}")
    print("\n🔍 Query examples:")
    print("   -- Connect to database:")
    print(f"   duckdb {cache_config.cache_dir / 'wheel_cache.duckdb'}")
    print("   -- Find liquid Unity puts:")
    print("   SELECT strike, AVG(volume) FROM options_data")
    print("   WHERE symbol='U' AND option_type='PUT'")
    print("   GROUP BY strike ORDER BY AVG(volume) DESC;")


if __name__ == "__main__":
    asyncio.run(main())
