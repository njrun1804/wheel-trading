"""
Database Connection Manager for M4 Pro Optimizations

Handles concurrent database access without lock conflicts.
"""

import asyncio
import logging
import sqlite3
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class DatabaseConnectionPool:
    """
    Thread-safe database connection pool with lock conflict resolution.

    Manages multiple connections to avoid database lock issues while
    maintaining performance optimizations.
    """

    def __init__(self, db_path: str, pool_size: int = 8, db_type: str = "duckdb"):
        self.db_path = str(Path(db_path).resolve())
        self.pool_size = pool_size
        self.db_type = db_type.lower()
        self._connections = []
        self._available_connections = asyncio.Queue()
        self._lock = threading.RLock()
        self._initialized = False

        # Database-specific configurations
        self._db_configs = {
            "duckdb": {
                "config": {
                    "threads": 8,  # M4 Pro P-cores
                    "memory_limit": "4GB",  # Fixed: was empty string
                    "max_memory": "4GB",  # Fixed: was empty string
                    "temp_directory": "/tmp/duckdb_temp",
                    "preserve_insertion_order": False,
                    "enable_object_cache": True,
                    "checkpoint_threshold": "1GB",
                    "wal_autocheckpoint": 1000,
                },
                "connection_string": f"{self.db_path}?access_mode=read_write&cache=shared",
            },
            "sqlite": {
                "config": {
                    "timeout": 30.0,
                    "isolation_level": None,  # Autocommit mode
                    "check_same_thread": False,
                },
                "pragmas": [
                    "PRAGMA journal_mode=WAL",
                    "PRAGMA synchronous=NORMAL",
                    "PRAGMA cache_size=-64000",  # 64MB cache
                    "PRAGMA temp_store=MEMORY",
                    "PRAGMA mmap_size=268435456",  # 256MB mmap
                    "PRAGMA optimize",
                ],
            },
        }

        logger.debug(
            f"Initialized {db_type} connection pool for {db_path} with {pool_size} connections"
        )

    async def initialize(self):
        """Initialize the connection pool"""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            try:
                # Create database directory if needed
                db_dir = Path(self.db_path).parent
                db_dir.mkdir(parents=True, exist_ok=True)

                # Initialize connections
                if self.db_type == "duckdb" and DUCKDB_AVAILABLE:
                    await self._init_duckdb_connections()
                else:
                    await self._init_sqlite_connections()

                self._initialized = True
                logger.info(
                    f"Initialized {len(self._connections)} database connections"
                )

            except Exception as e:
                logger.error(f"Failed to initialize database pool: {e}")
                raise

    async def _init_duckdb_connections(self):
        """Initialize DuckDB connection pool"""
        config = self._db_configs["duckdb"]["config"]

        for i in range(self.pool_size):
            try:
                # Each connection gets its own instance to avoid locks
                conn_path = f"{self.db_path}?connection_id={i}"
                conn = duckdb.connect(conn_path, config=config)

                # Install required extensions
                try:
                    conn.install_extension("httpfs")
                    conn.install_extension("parquet")
                    conn.install_extension("json")
                    conn.load_extension("httpfs")
                    conn.load_extension("parquet")
                    conn.load_extension("json")
                except Exception as e:
                    logger.debug(
                        f"Extension loading failed (may already be loaded): {e}"
                    )

                self._connections.append(conn)
                await self._available_connections.put(conn)

            except Exception as e:
                logger.warning(f"Failed to create DuckDB connection {i}: {e}")
                # Continue with fewer connections
                continue

    async def _init_sqlite_connections(self):
        """Initialize SQLite connection pool"""
        config = self._db_configs["sqlite"]["config"]
        pragmas = self._db_configs["sqlite"]["pragmas"]

        for i in range(self.pool_size):
            try:
                conn = sqlite3.connect(self.db_path, **config)
                conn.row_factory = sqlite3.Row  # Enable dict-like access

                # Apply performance pragmas
                for pragma in pragmas:
                    conn.execute(pragma)

                self._connections.append(conn)
                await self._available_connections.put(conn)

            except Exception as e:
                logger.warning(f"Failed to create SQLite connection {i}: {e}")
                continue

    @asynccontextmanager
    async def get_connection(self, timeout: float = 30.0):
        """Get a database connection from the pool"""
        if not self._initialized:
            await self.initialize()

        conn = None

        try:
            # Wait for available connection
            conn = await asyncio.wait_for(
                self._available_connections.get(), timeout=timeout
            )

            # Verify connection is still valid
            if not self._is_connection_valid(conn):
                logger.warning("Invalid connection detected, creating new one")
                conn = await self._create_replacement_connection()

            yield conn

        except TimeoutError:
            logger.error(f"Database connection timeout after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                # Return connection to pool
                try:
                    await self._available_connections.put(conn)
                except Exception as e:
                    logger.error(f"Failed to return connection to pool: {e}")

    def _is_connection_valid(self, conn) -> bool:
        """Check if database connection is still valid"""
        try:
            if self.db_type == "duckdb":
                conn.execute("SELECT 1").fetchone()
            else:  # SQLite
                conn.execute("SELECT 1").fetchone()
            return True
        except (sqlite3.Error, RuntimeError, AttributeError) as e:
            logger.debug(f"Database connection validation failed: {e}")
            return False

    async def _create_replacement_connection(self):
        """Create a replacement connection when one becomes invalid"""
        try:
            if self.db_type == "duckdb" and DUCKDB_AVAILABLE:
                config = self._db_configs["duckdb"]["config"]
                conn_id = len(self._connections)
                conn_path = f"{self.db_path}?connection_id={conn_id}"
                conn = duckdb.connect(conn_path, config=config)
            else:
                config = self._db_configs["sqlite"]["config"]
                conn = sqlite3.connect(self.db_path, **config)
                conn.row_factory = sqlite3.Row

                # Apply pragmas
                for pragma in self._db_configs["sqlite"]["pragmas"]:
                    conn.execute(pragma)

            return conn

        except Exception as e:
            logger.error(f"Failed to create replacement connection: {e}")
            raise

    async def execute_query(
        self, query: str, params: list | None = None, fetch: bool = True
    ) -> Any:
        """Execute a query using a connection from the pool"""
        async with self.get_connection() as conn:
            try:
                if params:
                    cursor = conn.execute(query, params)
                else:
                    cursor = conn.execute(query)

                if fetch:
                    if self.db_type == "duckdb":
                        return cursor.fetchall()
                    else:
                        return [dict(row) for row in cursor.fetchall()]
                else:
                    conn.commit() if hasattr(conn, "commit") else None
                    return cursor.rowcount

            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                logger.error(f"Query: {query}")
                raise

    async def execute_many(self, query: str, params_list: list[list]) -> int:
        """Execute multiple queries with different parameters"""
        async with self.get_connection() as conn:
            try:
                cursor = conn.executemany(query, params_list)
                conn.commit() if hasattr(conn, "commit") else None
                return cursor.rowcount
            except Exception as e:
                logger.error(f"Batch execution failed: {e}")
                raise

    async def query_to_dataframe(self, query: str, params: list | None = None):
        """Execute query and return results as DataFrame"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for query_to_dataframe")

        async with self.get_connection() as conn:
            try:
                if self.db_type == "duckdb":
                    if params:
                        df = conn.execute(query, params).df()
                    else:
                        df = conn.execute(query).df()
                else:
                    # SQLite via pandas
                    df = pd.read_sql_query(query, conn, params=params)

                return df

            except Exception as e:
                logger.error(f"DataFrame query failed: {e}")
                raise

    def get_pool_stats(self) -> dict[str, Any]:
        """Get connection pool statistics"""
        return {
            "db_path": self.db_path,
            "db_type": self.db_type,
            "pool_size": self.pool_size,
            "total_connections": len(self._connections),
            "available_connections": self._available_connections.qsize(),
            "initialized": self._initialized,
        }

    async def close(self):
        """Close all connections in the pool"""
        with self._lock:
            for conn in self._connections:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")

            self._connections.clear()

            # Clear the queue
            while not self._available_connections.empty():
                try:
                    self._available_connections.get_nowait()
                except asyncio.QueueEmpty:
                    break

            self._initialized = False
            logger.info("Database connection pool closed")


# Global connection pools
_connection_pools: dict[str, DatabaseConnectionPool] = {}
_pools_lock = threading.RLock()


def get_database_pool(
    db_path: str, pool_size: int = 8, db_type: str = "duckdb"
) -> DatabaseConnectionPool:
    """Get or create a database connection pool"""
    db_path = str(Path(db_path).resolve())
    pool_key = f"{db_type}:{db_path}"

    with _pools_lock:
        if pool_key not in _connection_pools:
            _connection_pools[pool_key] = DatabaseConnectionPool(
                db_path, pool_size, db_type
            )

        return _connection_pools[pool_key]


async def execute_database_query(
    db_path: str, query: str, params: list | None = None, db_type: str = "duckdb"
) -> Any:
    """Execute a database query with automatic connection management"""
    pool = get_database_pool(db_path, db_type=db_type)
    return await pool.execute_query(query, params)


async def close_all_pools():
    """Close all database connection pools"""
    with _pools_lock:
        for pool in _connection_pools.values():
            await pool.close()
        _connection_pools.clear()
