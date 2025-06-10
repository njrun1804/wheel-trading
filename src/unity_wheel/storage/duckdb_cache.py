"""
DuckDB-based local cache for all market data.
Provides SQL interface with automatic TTL and LRU eviction.
"""

import asyncio
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

from ..utils import get_logger

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
    """Local-first cache using DuckDB for all data types."""

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.config.cache_dir / "wheel_cache.duckdb"
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._last_vacuum = datetime.min

    @asynccontextmanager
    async def connection(self):
        """Get thread-safe connection to DuckDB."""
        conn = duckdb.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()

    async def initialize(self):
        """Create cache tables if not exists."""
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chains_symbol ON option_chains(symbol)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chains_created ON option_chains(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_positions_created ON position_snapshots(created_at)"
            )

    async def store_option_chain(
        self,
        symbol: str,
        expiration: datetime,
        timestamp: datetime,
        spot_price: float,
        chain_data: Dict[str, Any],
    ):
        """Store option chain data."""
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
    ) -> Optional[Dict[str, Any]]:
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
        self, account_id: str, positions: List[Dict], account_data: Dict[str, Any]
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

    async def get_latest_positions(
        self, account_id: str, max_age_minutes: int = 30
    ) -> Optional[Dict[str, Any]]:
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
            df = conn.execute(f"SELECT * FROM {table}").df()

            output_file = output_dir / f"{table}_{datetime.utcnow():%Y%m%d}.parquet"
            df.to_parquet(output_file, compression="snappy")

            logger.info("exported_to_parquet", table=table, rows=len(df), file=str(output_file))

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
            """,
                [str(parquet_file)],
            )

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get cache storage statistics."""
        async with self.connection() as conn:
            stats = {}

            # Get table sizes
            tables = ["option_chains", "position_snapshots", "greeks_cache", "predictions_cache"]
            for table in tables:
                safe_table = validate_table_name(table)
                count = conn.execute(f"SELECT COUNT(*) FROM {safe_table}").fetchone()[0]
                stats[f"{table}_count"] = count

            # Get database file size
            db_size_mb = self.db_path.stat().st_size / (1024 * 1024)
            stats["db_size_mb"] = round(db_size_mb, 2)

            # Get age of oldest records
            for table in tables:
                safe_table = validate_table_name(table)
                oldest = conn.execute(f"SELECT MIN(created_at) FROM {safe_table}").fetchone()[0]
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
            tables = ["option_chains", "position_snapshots", "greeks_cache", "predictions_cache"]
            for table in tables:
                safe_table = validate_table_name(table)
                deleted = conn.execute(
                    f"DELETE FROM {safe_table} WHERE created_at < ?", [cutoff]
                ).rowcount

                if deleted > 0:
                    logger.info("vacuum_deleted", table=table, rows=deleted)

            # Reclaim space
            conn.execute("VACUUM")

        self._last_vacuum = datetime.utcnow()

    async def clear_all(self):
        """Clear all cached data."""
        async with self.connection() as conn:
            tables = ["option_chains", "position_snapshots", "greeks_cache", "predictions_cache"]
            for table in tables:
                safe_table = validate_table_name(table)
                conn.execute(f"DELETE FROM {safe_table}")

        logger.warning("cache_cleared_all")
