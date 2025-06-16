"""
from __future__ import annotations

DuckDB-based local cache for all market data.
Provides SQL interface with automatic TTL and LRU eviction.
"""

import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import duckdb

    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False

    # Create dummy for when DuckDB isn't available
    class duckdb:  # type: ignore
        class DuckDBPyConnection:  # type: ignore
            pass

        @staticmethod
        def connect(*args, **kwargs):  # type: ignore
            raise ImportError("DuckDB not available on this platform")


# Import concurrent database management
try:
    from ....bolt_database_fixes import ConcurrentDatabase, DatabaseConfig

    HAS_CONCURRENT_DB = True
except ImportError:
    HAS_CONCURRENT_DB = False

    # Fallback implementations
    class ConcurrentDatabase:
        def __init__(self, *args, **kwargs):
            pass

        def query(self, *args, **kwargs):
            return []

        def close(self):
            pass

    class DatabaseConfig:
        pass


from src.config.loader import get_config
from unity_wheel.utils import get_logger

config = get_config()


logger = get_logger(__name__)

# Valid table names for security
VALID_TABLES = {"price_history", "options_data", "fred_data", "trades", "metrics"}


def validate_table_name(table: str) -> str:
    """Validate table name to prevent SQL injection."""
    # Only allow alphanumeric and underscore
    if not re.match(r"^[a-zA-Z0-9_]+$", table):
        raise ValueError(f"Invalid table name: {table}")
    # Check against whitelist
    if table not in VALID_TABLES:
        raise ValueError(f"Unknown table: {table}")
    return table


@dataclass
class CacheConfig:
    """Configuration for DuckDB cache."""

    cache_dir: Path = Path.home() / ".wheel_trading" / "cache"
    max_size_gb: float = 5.0
    ttl_days: int = 30
    vacuum_interval_hours: int = 24


class DuckDBCache:
    """Local-first cache using DuckDB with concurrent access support."""

    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        # Use a simple default database name instead of accessing non-existent storage attribute
        self.db_path = self.config.cache_dir / "wheel_trading_cache.duckdb"
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._last_vacuum = datetime.min

        # Initialize concurrent database manager if available
        self._concurrent_db: ConcurrentDatabase | None = None
        if HAS_CONCURRENT_DB and _HAS_DUCKDB:
            try:
                db_config = DatabaseConfig(
                    path=self.db_path,
                    max_connections=4,  # Conservative for cache
                    connection_timeout=15.0,
                    lock_timeout=15.0,
                    retry_attempts=2,
                    retry_delay=0.5,
                    enable_wal_mode=True,
                    enable_connection_pooling=True,
                )
                self._concurrent_db = ConcurrentDatabase(
                    str(self.db_path), **db_config.__dict__
                )
                logger.info(
                    f"Initialized concurrent database for cache: {self.db_path}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize concurrent database, using fallback: {e}"
                )
                self._concurrent_db = None

        if not _HAS_DUCKDB:
            logger.warning("DuckDB not available - cache will be disabled")

    def _check_duckdb_available(self):
        """Check if DuckDB is available, raise error if not."""
        if not _HAS_DUCKDB:
            raise ImportError(
                "DuckDB not available on this platform - storage operations disabled"
            )

    @asynccontextmanager
    async def connection(self):
        """Get thread-safe connection to DuckDB with concurrent access support."""
        self._check_duckdb_available()

        # Use concurrent database manager if available
        if self._concurrent_db:
            try:
                with self._concurrent_db.connection() as conn:
                    yield conn
                return
            except Exception as e:
                logger.warning(
                    f"Concurrent database connection failed, using fallback: {e}"
                )

        # Fallback to direct connection
        conn = duckdb.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()

    async def initialize(self):
        """Create cache tables if not exists."""
        if not _HAS_DUCKDB:
            logger.warning("DuckDB not available - skipping cache initialization")
            return
        async with self.connection() as conn:
            # Option chains table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS option_chains (
                    symbol VARCHAR NOT NULL,
                    expiration DATE NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    spot_price DECIMAL(10,2),
                    data JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, expiration, timestamp)
                )
            """
            )

            # Positions snapshot table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS position_snapshots (
                    account_id VARCHAR NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    positions JSON,
                    account_data JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (account_id, timestamp)
                )
            """
            )

            # Greeks calculations cache
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS greeks_cache (
                    option_symbol VARCHAR NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    spot_price DECIMAL(10,2),
                    risk_free_rate DECIMAL(6,4),
                    delta DECIMAL(6,4),
                    gamma DECIMAL(8,6),
                    theta DECIMAL(8,4),
                    vega DECIMAL(8,4),
                    rho DECIMAL(8,4),
                    iv DECIMAL(6,4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (option_symbol, timestamp)
                )
            """
            )

            # Model predictions cache
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions_cache (
                    prediction_id VARCHAR NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    input_features JSON,
                    predictions JSON,
                    model_version VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (prediction_id)
                )
            """
            )

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chains_symbol ON option_chains(symbol)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chains_created ON option_chains(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_positions_created ON position_snapshots(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_greeks_created ON greeks_cache(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions_cache(created_at)"
            )

    async def store_option_chain(
        self,
        symbol: str,
        expiration: datetime,
        timestamp: datetime,
        spot_price: float,
        chain_data: dict[str, Any],
    ):
        """Store option chain data with concurrent access support."""
        # Use concurrent database for writes if available
        if self._concurrent_db:
            try:
                self._concurrent_db.execute(
                    """
                    INSERT OR REPLACE INTO option_chains
                    (symbol, expiration, timestamp, spot_price, data)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (symbol, expiration.date(), timestamp, spot_price, chain_data),
                )
                await self._check_vacuum()
                return
            except Exception as e:
                logger.warning(f"Concurrent store failed, using fallback: {e}")

        # Fallback to regular connection
        async with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO option_chains
                (symbol, expiration, timestamp, spot_price, data)
                VALUES (?, ?, ?, ?, ?)
            """,
                [symbol, expiration.date(), timestamp, spot_price, chain_data],
            )

        await self._check_vacuum()

    async def get_option_chain(
        self, symbol: str, expiration: datetime, max_age_minutes: int = 15
    ) -> dict[str, Any] | None:
        """Get option chain if exists and fresh."""
        cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)

        async with self.connection() as conn:
            result = conn.execute(
                """
                SELECT data, spot_price, timestamp
                FROM option_chains
                WHERE symbol = ?
                AND expiration = ?
                AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 1
            """,
                [symbol, expiration.date(), cutoff],
            ).fetchone()

            if result:
                data, spot_price, timestamp = result
                data["spot_price"] = float(spot_price)
                data["cached_at"] = timestamp
                return data
            return None

    async def store_positions(
        self, account_id: str, positions: list[dict], account_data: dict[str, Any]
    ):
        """Store position snapshot."""
        timestamp = datetime.utcnow()
        async with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO position_snapshots
                (account_id, timestamp, positions, account_data)
                VALUES (?, ?, ?, ?)
            """,
                [account_id, timestamp, positions, account_data],
            )

        await self._check_vacuum()

    async def get_latest_positions(
        self, account_id: str, max_age_minutes: int = 30
    ) -> dict[str, Any] | None:
        """Get latest position snapshot if fresh."""
        cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)

        async with self.connection() as conn:
            result = conn.execute(
                """
                SELECT positions, account_data, timestamp
                FROM position_snapshots
                WHERE account_id = ?
                AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 1
            """,
                [account_id, cutoff],
            ).fetchone()

            if result:
                positions, account_data, timestamp = result
                return {
                    "positions": positions,
                    "account_data": account_data,
                    "timestamp": timestamp,
                }
            return None

    async def export_to_parquet(self, table: str, output_dir: Path):
        """Export table to Parquet for GCS backup."""
        table = validate_table_name(table)
        async with self.connection() as conn:
            df = conn.execute(
                f"SELECT * FROM {table}"
            ).df()  # nosec B608 - table validated

            output_file = output_dir / f"{table}_{datetime.utcnow():%Y%m%d}.parquet"
            df.to_parquet(output_file, compression="snappy")

            logger.info(
                "exported_to_parquet", table=table, rows=len(df), file=str(output_file)
            )

            return output_file

    async def import_from_parquet(self, parquet_file: Path, table: str):
        """Import data from Parquet file."""
        table = validate_table_name(table)
        async with self.connection() as conn:
            # Use parameterized query for file path
            conn.execute(
                f"""
                INSERT INTO {table}
                SELECT * FROM read_parquet(?)
            """,  # nosec B608 - table validated
                [str(parquet_file)],
            )

    async def get_storage_stats(self) -> dict[str, Any]:
        """Get cache storage statistics."""
        async with self.connection() as conn:
            stats = {}

            # Get table sizes
            tables = [
                "option_chains",
                "position_snapshots",
                "greeks_cache",
                "predictions_cache",
            ]
            for table in tables:
                safe_table = validate_table_name(table)
                count = conn.execute(f"SELECT COUNT(*) FROM {safe_table}").fetchone()[
                    0
                ]  # nosec B608
                stats[f"{table}_count"] = count

            # Get database file size
            db_size_mb = self.db_path.stat().st_size / (1024 * 1024)
            stats["db_size_mb"] = round(db_size_mb, 2)

            # Get age of oldest records
            for table in tables:
                safe_table = validate_table_name(table)
                oldest = conn.execute(
                    f"SELECT MIN(created_at) FROM {safe_table}"
                ).fetchone()[
                    0
                ]  # nosec B608
                if oldest:
                    age_days = (datetime.utcnow() - oldest).days
                    stats[f"{table}_oldest_days"] = age_days

            return stats

    async def _check_vacuum(self):
        """Run vacuum if needed to reclaim space."""
        if datetime.utcnow() - self._last_vacuum > timedelta(
            hours=self.config.vacuum_interval_hours
        ):
            await self._vacuum()

    async def _vacuum(self):
        """Clean up old data and reclaim space."""
        cutoff = datetime.utcnow() - timedelta(days=self.config.ttl_days)

        async with self.connection() as conn:
            # Delete old records
            tables = [
                "option_chains",
                "position_snapshots",
                "greeks_cache",
                "predictions_cache",
            ]
            for table in tables:
                safe_table = validate_table_name(table)
                deleted = conn.execute(
                    f"DELETE FROM {safe_table} WHERE created_at < ?",
                    [cutoff],  # nosec B608
                ).rowcount

                if deleted > 0:
                    logger.info("vacuum_deleted", table=table, rows=deleted)

            # Reclaim space
            conn.execute("VACUUM")

        self._last_vacuum = datetime.utcnow()

    async def clear_all(self):
        """Clear all cached data."""
        async with self.connection() as conn:
            tables = [
                "option_chains",
                "position_snapshots",
                "greeks_cache",
                "predictions_cache",
            ]
            for table in tables:
                safe_table = validate_table_name(table)
                conn.execute(f"DELETE FROM {safe_table}")  # nosec B608

        logger.warning("cache_cleared_all")

    def close(self):
        """Close the cache and cleanup resources."""
        if self._concurrent_db:
            try:
                self._concurrent_db.close()
                logger.info("Concurrent database cache closed")
            except Exception as e:
                logger.warning(f"Error closing concurrent database cache: {e}")

        if self._conn:
            try:
                self._conn.close()
                self._conn = None
            except Exception as e:
                logger.warning(f"Error closing direct DuckDB connection: {e}")

    def get_lock_info(self) -> dict[str, Any] | None:
        """Get information about current database locks."""
        if self._concurrent_db and hasattr(self._concurrent_db, "get_lock_info"):
            return self._concurrent_db.get_lock_info()
        return None

    def force_unlock(self) -> bool:
        """Force unlock the database (use with caution)."""
        if self._concurrent_db and hasattr(self._concurrent_db, "force_unlock"):
            return self._concurrent_db.force_unlock()
        return False
