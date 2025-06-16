"""
Unified Database Manager with Connection Pooling

Prevents SQLite locking issues and manages concurrent access.
"""
from __future__ import annotations

import queue
import sqlite3
import threading
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from unified_config import get_unified_config


class DatabaseConnectionPool:
    """Thread-safe SQLite connection pool."""

    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self.pool = queue.Queue(maxsize=pool_size)
        self.lock = threading.RLock()
        self._created_connections = 0

        # Ensure database directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Pre-populate pool with connections
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize connection pool with optimized SQLite settings."""
        for _ in range(self.pool_size):
            conn = self._create_connection()
            self.pool.put(conn)

    def _create_connection(self) -> sqlite3.Connection:
        """Create optimized SQLite connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)

        # Optimize for concurrent access
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=memory")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB

        self._created_connections += 1
        return conn

    @contextmanager
    def get_connection(self, timeout: float = 10.0):
        """Get connection from pool with automatic return."""
        conn = None
        try:
            conn = self.pool.get(timeout=timeout)
            yield conn
        except queue.Empty:
            raise RuntimeError(f"Database connection timeout after {timeout}s")
        finally:
            if conn:
                try:
                    conn.rollback()  # Ensure clean state
                    self.pool.put(conn)
                except Exception:
                    # Connection is bad, create new one
                    with suppress(Exception):
                        conn.close()
                    new_conn = self._create_connection()
                    self.pool.put(new_conn)

    def close_all(self):
        """Close all connections in pool."""
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                conn.close()
            except (queue.Empty, Exception):
                break


class UnifiedDatabaseManager:
    """Manages all database operations with connection pooling."""

    def __init__(self):
        self.config = get_unified_config()
        self.pools: dict[str, DatabaseConnectionPool] = {}
        self._initialize_pools()

    def _initialize_pools(self):
        """Initialize connection pools for all databases."""
        database_configs = {
            "evolution": self.config.meta.evolution_db,
            "monitoring": self.config.meta.monitoring_db,
            "reality": self.config.meta.reality_db,
        }

        pool_size = self.config.shared.database_connection_pool_size

        for name, db_path in database_configs.items():
            self.pools[name] = DatabaseConnectionPool(db_path, pool_size)

    @contextmanager
    def get_connection(self, database: str = "evolution"):
        """Get database connection with automatic pooling."""
        if database not in self.pools:
            raise ValueError(f"Unknown database: {database}")

        with self.pools[database].get_connection() as conn:
            yield conn

    def execute_query(
        self,
        query: str,
        params: tuple = (),
        database: str = "evolution",
        fetch: bool = False,
    ) -> list[tuple] | None:
        """Execute query with connection pooling."""
        with self.get_connection(database) as conn:
            cursor = conn.execute(query, params)
            if fetch:
                result = cursor.fetchall()
                conn.commit()
                return result
            else:
                conn.commit()
                return None

    def execute_transaction(
        self, queries: list[tuple], database: str = "evolution"
    ) -> bool:
        """Execute multiple queries in a transaction."""
        try:
            with self.get_connection(database) as conn:
                for query, params in queries:
                    conn.execute(query, params)
                conn.commit()
                return True
        except Exception as e:
            print(f"Transaction failed: {e}")
            return False

    def get_database_stats(self) -> dict[str, Any]:
        """Get statistics for all databases."""
        stats = {}

        for name, pool in self.pools.items():
            try:
                with pool.get_connection() as conn:
                    # Get database size
                    cursor = conn.execute("PRAGMA page_count")
                    page_count = cursor.fetchone()[0]
                    cursor = conn.execute("PRAGMA page_size")
                    page_size = cursor.fetchone()[0]
                    size_mb = (page_count * page_size) / (1024 * 1024)

                    # Get table count
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                    )
                    table_count = cursor.fetchone()[0]

                    stats[name] = {
                        "size_mb": round(size_mb, 2),
                        "table_count": table_count,
                        "pool_size": pool.pool_size,
                        "connections_created": pool._created_connections,
                    }
            except Exception as e:
                stats[name] = {"error": str(e)}

        return stats

    def cleanup(self):
        """Close all database connections."""
        for pool in self.pools.values():
            pool.close_all()


# Global instance
_db_manager: UnifiedDatabaseManager | None = None


def get_database_manager() -> UnifiedDatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = UnifiedDatabaseManager()
    return _db_manager


def cleanup_databases():
    """Cleanup all database connections."""
    global _db_manager
    if _db_manager:
        _db_manager.cleanup()
        _db_manager = None


if __name__ == "__main__":
    # Test database manager
    print("🗄️ Testing Database Manager...")

    db_manager = get_database_manager()

    # Test connection
    with db_manager.get_connection("evolution") as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"Evolution DB tables: {len(tables)}")

    # Test stats
    stats = db_manager.get_database_stats()
    print("Database statistics:")
    for name, data in stats.items():
        if "error" not in data:
            print(f"  {name}: {data['size_mb']}MB, {data['table_count']} tables")

    print("✅ Database manager test complete")
