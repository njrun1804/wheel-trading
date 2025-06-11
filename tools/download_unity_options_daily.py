#!/usr/bin/env python
"""
Download Unity options end-of-day (daily) data from Databento.
Stores OHLCV daily bars in DuckDB for historical analysis.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import databento as db
    import duckdb
    from databento_dbn import Schema, SType
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install: pip install duckdb databento")
    sys.exit(1)

from src.config import get_config
from src.unity_wheel.data_providers.databento.client import DatabentoClient
from src.unity_wheel.secrets.integration import get_databento_api_key
from src.unity_wheel.utils.logging import StructuredLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = StructuredLogger(logging.getLogger(__name__))


class UnityOptionsDownloader:
    """Downloads Unity options daily data from Databento."""

    def __init__(self):
        """Initialize downloader with database connection."""
        # Set up paths
        self.cache_dir = Path.home() / ".wheel_trading" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "wheel_cache.duckdb"

        # Get API key
        api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            logger.info("No API key in environment, retrieving from SecretManager")
            api_key = get_databento_api_key()

        if not api_key:
            raise ValueError("Databento API key required")

        # Initialize Databento client
        self.client = db.Historical(api_key)

        # Date range
        self.start_date = datetime(2023, 3, 28, tzinfo=timezone.utc)
        self.end_date = datetime.now(timezone.utc) - timedelta(days=1)  # Yesterday

        logger.info(
            "downloader_initialized",
            extra={
                "db_path": str(self.db_path),
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
            },
        )

    def create_table(self, conn: duckdb.DuckDBPyConnection):
        """Create the unity_options_daily table if it doesn't exist."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unity_options_daily (
                symbol VARCHAR NOT NULL,
                instrument_id BIGINT NOT NULL,
                strike_price DECIMAL(10,2) NOT NULL,
                expiration DATE NOT NULL,
                option_type VARCHAR(4) NOT NULL,
                date DATE NOT NULL,
                open DECIMAL(10,4),
                high DECIMAL(10,4),
                low DECIMAL(10,4),
                close DECIMAL(10,4),
                volume BIGINT,
                trades_count INTEGER,
                vwap DECIMAL(10,4),
                bid_close DECIMAL(10,4),
                ask_close DECIMAL(10,4),
                underlying_close DECIMAL(10,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (instrument_id, date)
            )
        """
        )

        # Create indexes for common queries
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_unity_daily_symbol_date
            ON unity_options_daily(symbol, date)
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_unity_daily_expiration_strike
            ON unity_options_daily(expiration, strike_price, date)
        """
        )

        logger.info("table_created", extra={"table": "unity_options_daily"})

    async def download_data(self):
        """Download Unity options daily data from Databento."""
        # Connect to database
        conn = duckdb.connect(str(self.db_path))

        try:
            # Create table
            self.create_table(conn)

            # Track statistics
            total_records = 0
            failed_chunks = 0
            successful_days = 0

            # Download in monthly chunks to manage data size
            current_start = self.start_date

            while current_start < self.end_date:
                # Calculate chunk end (1 month or remaining days)
                chunk_end = min(current_start + timedelta(days=30), self.end_date)

                logger.info(
                    "downloading_chunk",
                    extra={"start": current_start.isoformat(), "end": chunk_end.isoformat()},
                )

                try:
                    # Try different schemas in order of preference
                    schemas_to_try = [
                        (Schema.OHLCV_1D, "Daily OHLCV bars"),
                        (Schema.STATISTICS, "Daily statistics"),
                        (Schema.TRADES, "Trade data (will aggregate)"),
                    ]

                    chunk_data = None
                    used_schema = None

                    for schema, schema_desc in schemas_to_try:
                        try:
                            logger.info(
                                f"Trying schema: {schema_desc}", extra={"schema": str(schema)}
                            )

                            # Try with Unity option symbols
                            response = self.client.timeseries.get_range(
                                dataset="OPRA.PILLAR",
                                schema=schema,
                                start=current_start,
                                end=chunk_end,
                                symbols=["U.OPT"],  # Unity options
                                stype_in=SType.PARENT,
                            )

                            # Convert to list to check if we got data
                            chunk_data = list(response)
                            if chunk_data:
                                used_schema = schema
                                logger.info(
                                    f"Successfully retrieved {len(chunk_data)} records with {schema_desc}",
                                    extra={"count": len(chunk_data)},
                                )
                                break

                        except Exception as e:
                            logger.warning(
                                f"Schema {schema} failed: {str(e)}", extra={"error": str(e)}
                            )
                            continue

                    if not chunk_data:
                        logger.error(
                            "No data retrieved for chunk",
                            extra={
                                "start": current_start.isoformat(),
                                "end": chunk_end.isoformat(),
                            },
                        )
                        failed_chunks += 1
                        current_start = chunk_end + timedelta(days=1)
                        continue

                    # Process and store the data based on schema
                    if used_schema == Schema.OHLCV_1D:
                        records_added = self._process_ohlcv_data(conn, chunk_data)
                    elif used_schema == Schema.STATISTICS:
                        records_added = self._process_statistics_data(conn, chunk_data)
                    elif used_schema == Schema.TRADES:
                        records_added = self._aggregate_trades_to_daily(
                            conn, chunk_data, current_start, chunk_end
                        )
                    else:
                        logger.error("Unknown schema type")
                        records_added = 0

                    total_records += records_added
                    successful_days += (chunk_end - current_start).days

                    logger.info(
                        "chunk_complete",
                        extra={"records_added": records_added, "total_so_far": total_records},
                    )

                except Exception as e:
                    logger.error(
                        "chunk_failed",
                        extra={
                            "error": str(e),
                            "start": current_start.isoformat(),
                            "end": chunk_end.isoformat(),
                        },
                    )
                    failed_chunks += 1

                # Move to next chunk
                current_start = chunk_end + timedelta(days=1)

                # Small delay to respect rate limits
                await asyncio.sleep(0.5)

            # Show summary
            self._show_summary(conn, total_records, successful_days, failed_chunks)

        finally:
            conn.close()

    def _process_ohlcv_data(self, conn: duckdb.DuckDBPyConnection, data: List) -> int:
        """Process OHLCV daily bar data."""
        records = []

        for record in data:
            try:
                # Extract option details from symbol or instrument definition
                if hasattr(record, "symbol"):
                    # Parse Unity option symbol (e.g., "U     240119P00035000")
                    symbol_parts = self._parse_option_symbol(record.symbol)
                    if not symbol_parts:
                        continue

                    underlying, expiration, option_type, strike = symbol_parts
                else:
                    # Skip if we can't identify the option
                    continue

                # Create record
                records.append(
                    {
                        "symbol": underlying,
                        "instrument_id": (
                            record.instrument_id if hasattr(record, "instrument_id") else 0
                        ),
                        "strike_price": strike,
                        "expiration": expiration,
                        "option_type": option_type,
                        "date": datetime.fromtimestamp(
                            record.ts_event / 1e9, tz=timezone.utc
                        ).date(),
                        "open": float(record.open) / 1e9 if hasattr(record, "open") else None,
                        "high": float(record.high) / 1e9 if hasattr(record, "high") else None,
                        "low": float(record.low) / 1e9 if hasattr(record, "low") else None,
                        "close": float(record.close) / 1e9 if hasattr(record, "close") else None,
                        "volume": record.volume if hasattr(record, "volume") else 0,
                        "trades_count": record.trades if hasattr(record, "trades") else 0,
                        "vwap": float(record.vwap) / 1e9 if hasattr(record, "vwap") else None,
                        "bid_close": None,  # Would need separate quote data
                        "ask_close": None,
                        "underlying_close": None,  # Would need separate underlying data
                    }
                )

            except Exception as e:
                logger.debug(f"Failed to process record: {e}")
                continue

        # Batch insert
        if records:
            self._batch_insert(conn, records)

        return len(records)

    def _process_statistics_data(self, conn: duckdb.DuckDBPyConnection, data: List) -> int:
        """Process daily statistics data as fallback."""
        records = []

        for record in data:
            try:
                # Similar processing but adapted for statistics schema
                if hasattr(record, "symbol"):
                    symbol_parts = self._parse_option_symbol(record.symbol)
                    if not symbol_parts:
                        continue

                    underlying, expiration, option_type, strike = symbol_parts
                else:
                    continue

                # Statistics typically have different fields
                records.append(
                    {
                        "symbol": underlying,
                        "instrument_id": (
                            record.instrument_id if hasattr(record, "instrument_id") else 0
                        ),
                        "strike_price": strike,
                        "expiration": expiration,
                        "option_type": option_type,
                        "date": datetime.fromtimestamp(
                            record.ts_event / 1e9, tz=timezone.utc
                        ).date(),
                        "open": (
                            float(record.open_price) / 1e9
                            if hasattr(record, "open_price")
                            else None
                        ),
                        "high": (
                            float(record.high_price) / 1e9
                            if hasattr(record, "high_price")
                            else None
                        ),
                        "low": (
                            float(record.low_price) / 1e9 if hasattr(record, "low_price") else None
                        ),
                        "close": (
                            float(record.close_price) / 1e9
                            if hasattr(record, "close_price")
                            else None
                        ),
                        "volume": record.volume if hasattr(record, "volume") else 0,
                        "trades_count": 0,
                        "vwap": None,
                        "bid_close": (
                            float(record.bid_close) / 1e9 if hasattr(record, "bid_close") else None
                        ),
                        "ask_close": (
                            float(record.ask_close) / 1e9 if hasattr(record, "ask_close") else None
                        ),
                        "underlying_close": None,
                    }
                )

            except Exception as e:
                logger.debug(f"Failed to process statistics record: {e}")
                continue

        if records:
            self._batch_insert(conn, records)

        return len(records)

    def _aggregate_trades_to_daily(
        self, conn: duckdb.DuckDBPyConnection, data: List, start_date: datetime, end_date: datetime
    ) -> int:
        """Aggregate trade data into daily bars as last resort."""
        # This would be more complex - aggregate trades by day
        # For now, return 0 as this is a fallback
        logger.warning("Trade aggregation not implemented - skipping")
        return 0

    def _parse_option_symbol(self, symbol: str) -> Optional[Tuple[str, datetime, str, float]]:
        """Parse Unity option symbol format."""
        try:
            # Unity format: "U     240119P00035000"
            # Underlying (6 chars) + Date (6) + Type (1) + Strike (8)
            if len(symbol) < 21:
                return None

            underlying = symbol[:6].strip()
            if underlying != "U":
                return None

            date_str = symbol[6:12]
            expiration = datetime.strptime("20" + date_str, "%Y%m%d").date()

            option_type = "CALL" if symbol[12] == "C" else "PUT"

            strike_str = symbol[13:21]
            strike = float(strike_str) / 1000  # Convert from thousandths

            return (underlying, expiration, option_type, strike)

        except Exception:
            return None

    def _batch_insert(self, conn: duckdb.DuckDBPyConnection, records: List[Dict]):
        """Batch insert records into the database."""
        if not records:
            return

        # Prepare values for insert
        placeholders = ", ".join(["?"] * len(records[0]))
        columns = list(records[0].keys())

        # Convert records to tuples
        values = [tuple(r[col] for col in columns) for r in records]

        # Insert with ON CONFLICT handling
        query = f"""
            INSERT OR REPLACE INTO unity_options_daily ({", ".join(columns)})
            VALUES ({placeholders})
        """

        conn.executemany(query, values)

    def _show_summary(
        self,
        conn: duckdb.DuckDBPyConnection,
        total_records: int,
        successful_days: int,
        failed_chunks: int,
    ):
        """Show download summary statistics."""
        print("\n" + "=" * 60)
        print("Unity Options Daily Data Download Summary")
        print("=" * 60)

        # Basic stats
        print(f"\nDownload Statistics:")
        print(f"  Total records downloaded: {total_records:,}")
        print(f"  Successful days: {successful_days}")
        print(f"  Failed chunks: {failed_chunks}")
        print(f"  Date range: {self.start_date.date()} to {self.end_date.date()}")

        # Database statistics
        stats = conn.execute(
            """
            SELECT
                COUNT(DISTINCT date) as unique_days,
                COUNT(DISTINCT instrument_id) as unique_options,
                COUNT(DISTINCT expiration) as unique_expirations,
                COUNT(*) as total_records,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                AVG(volume) as avg_volume,
                SUM(volume) as total_volume
            FROM unity_options_daily
        """
        ).fetchone()

        if stats:
            print(f"\nDatabase Statistics:")
            print(f"  Total records in database: {stats[3]:,}")
            print(f"  Unique trading days: {stats[0]:,}")
            print(f"  Unique options: {stats[1]:,}")
            print(f"  Unique expirations: {stats[2]:,}")
            print(f"  Date range in DB: {stats[4]} to {stats[5]}")
            print(
                f"  Average daily volume: {stats[6]:,.0f}"
                if stats[6]
                else "  Average daily volume: N/A"
            )
            print(f"  Total volume: {stats[7]:,}" if stats[7] else "  Total volume: N/A")

        # Sample data
        print(f"\nSample Records (most recent):")
        samples = conn.execute(
            """
            SELECT date, strike_price, option_type, close, volume
            FROM unity_options_daily
            WHERE close IS NOT NULL
            ORDER BY date DESC, volume DESC
            LIMIT 5
        """
        ).fetchall()

        if samples:
            print(f"  {'Date':<12} {'Strike':<8} {'Type':<5} {'Close':<8} {'Volume':<10}")
            print(f"  {'-'*12} {'-'*8} {'-'*5} {'-'*8} {'-'*10}")
            for date, strike, opt_type, close, volume in samples:
                print(
                    f"  {str(date):<12} ${strike:<7.2f} {opt_type:<5} ${close:<7.2f} {volume:<10,}"
                )

        # File info
        db_size_mb = self.db_path.stat().st_size / (1024 * 1024)
        print(f"\nDatabase file: {self.db_path}")
        print(f"Database size: {db_size_mb:.1f} MB")

        print("\n" + "=" * 60)


async def main():
    """Main entry point."""
    print("Starting Unity options daily data download...")
    print("This may take several minutes depending on your connection speed.")

    try:
        downloader = UnityOptionsDownloader()
        await downloader.download_data()
        print("\nDownload completed successfully!")

    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Download failed: {e}")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
