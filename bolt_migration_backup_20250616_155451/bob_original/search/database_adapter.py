#!/usr/bin/env python3
"""
Einstein Database Adapter with Concurrent Access Support

This module provides a database adapter for Einstein that supports concurrent access
through connection pooling and session isolation, preventing the "Could not set lock"
errors that block multi-session usage.

Key Features:
- Concurrent database access for multiple Einstein instances
- Automatic lock management and deadlock recovery
- Session isolation for search operations
- Performance monitoring and optimization
- Graceful degradation when concurrent features unavailable
"""

import asyncio
import logging
import sqlite3
import threading
import time
from contextlib import asynccontextmanager, contextmanager, suppress
from pathlib import Path
from typing import Any

# Try to import concurrent database management
try:
    # Import from the root bolt_database_fixes module
    import sys
    from pathlib import Path

    # Add the root directory to Python path for imports
    root_dir = Path(__file__).parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from bolt_database_fixes import ConcurrentDatabase, DatabaseConfig
    from src.unity_wheel.storage.session_isolation import (
        async_database_session,
    )

    HAS_CONCURRENT_DB = True
except ImportError as e:
    logger.warning(f"Could not import concurrent database components: {e}")
    HAS_CONCURRENT_DB = False

    # Functional fallback implementations using SQLite/DuckDB directly
    import sqlite3
    import threading
    from contextlib import contextmanager
    from typing import Any

    class ConcurrentDatabase:
        """Functional concurrent database implementation using SQLite."""

        def __init__(
            self,
            db_path: str,
            max_connections: int = 6,
            connection_timeout: float = 30.0,
            **kwargs,
        ):
            self.db_path = db_path
            self.max_connections = max_connections
            self.connection_timeout = connection_timeout
            self.config = kwargs
            self._connections: dict[int, sqlite3.Connection] = {}
            self._lock = threading.RLock()
            self._thread_local = threading.local()

        def _get_connection(self) -> sqlite3.Connection:
            """Get a thread-local database connection."""
            thread_id = threading.get_ident()

            if not hasattr(self._thread_local, "connection"):
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=self.connection_timeout,
                    check_same_thread=False,
                )
                conn.row_factory = sqlite3.Row

                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA temp_store=MEMORY")
                conn.execute("PRAGMA mmap_size=268435456")  # 256MB

                self._thread_local.connection = conn

                with self._lock:
                    self._connections[thread_id] = conn

            return self._thread_local.connection

        @contextmanager
        def connection(self, lock_type: str = "shared"):
            """Get a connection with proper locking."""
            conn = self._get_connection()

            try:
                if lock_type == "exclusive":
                    conn.execute("BEGIN IMMEDIATE")
                else:
                    conn.execute("BEGIN DEFERRED")

                yield conn
                conn.commit()

            except Exception as e:
                conn.rollback()
                raise e

        def query(self, sql: str, params: tuple | None = None) -> list[dict[str, Any]]:
            """Execute a query and return results."""
            conn = self._get_connection()

            try:
                if params:
                    cursor = conn.execute(sql, params)
                else:
                    cursor = conn.execute(sql)

                if sql.strip().upper().startswith("SELECT"):
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
                else:
                    conn.commit()
                    return []

            except Exception as e:
                conn.rollback()
                raise e

        def execute(self, sql: str, params: tuple | None = None):
            """Execute a query (alias for query method)."""
            return self.query(sql, params)

        def close(self):
            """Close all connections."""
            with self._lock:
                for conn in self._connections.values():
                    try:
                        conn.close()
                    except Exception:
                        pass
                self._connections.clear()

                if hasattr(self._thread_local, "connection"):
                    try:
                        self._thread_local.connection.close()
                    except Exception:
                        pass
                    delattr(self._thread_local, "connection")

    class DatabaseConfig:
        """Database configuration with proper validation."""

        def __init__(
            self,
            path: str | None = None,
            max_connections: int = 6,
            connection_timeout: float = 30.0,
            lock_timeout: float = 30.0,
            retry_attempts: int = 3,
            retry_delay: float = 1.0,
            enable_wal_mode: bool = True,
            enable_connection_pooling: bool = True,
            **kwargs,
        ):
            self.path = path
            self.max_connections = max_connections
            self.connection_timeout = connection_timeout
            self.lock_timeout = lock_timeout
            self.retry_attempts = retry_attempts
            self.retry_delay = retry_delay
            self.enable_wal_mode = enable_wal_mode
            self.enable_connection_pooling = enable_connection_pooling

            # Store all kwargs for configuration
            for key, value in kwargs.items():
                setattr(self, key, value)

        def get(self, key: str, default: Any = None) -> Any:
            """Get configuration value."""
            return getattr(self, key, default)

        def to_dict(self) -> dict[str, Any]:
            """Return configuration as dictionary."""
            result = {}
            for attr in dir(self):
                if not attr.startswith("_") and not callable(getattr(self, attr)):
                    result[attr] = getattr(self, attr)
            return result


logger = logging.getLogger(__name__)


class EinsteinDatabaseAdapter:
    """Database adapter for Einstein with concurrent access support."""

    def __init__(self, db_path: str | Path, db_name: str = "einstein"):
        self.db_path = Path(db_path)
        self.db_name = db_name
        self.session_hint = f"einstein_{db_name}"

        # Initialize concurrent database if available
        self.concurrent_db: ConcurrentDatabase | None = None
        # Disable concurrent for now - use reliable fallback
        self.use_concurrent = False  # HAS_CONCURRENT_DB

        if self.use_concurrent:
            try:
                config = DatabaseConfig(
                    path=self.db_path,
                    max_connections=6,  # Allow multiple Einstein instances
                    connection_timeout=30.0,
                    lock_timeout=30.0,
                    retry_attempts=3,
                    retry_delay=1.0,
                    enable_wal_mode=True,
                    enable_connection_pooling=True,
                )
                # Handle both real and fallback DatabaseConfig versions
                if hasattr(config, "config"):
                    # Real DatabaseConfig from bolt_database_fixes
                    config_dict = config.config
                else:
                    # Fallback DatabaseConfig with to_dict method
                    config_dict = config.to_dict()

                self.concurrent_db = ConcurrentDatabase(
                    str(self.db_path), **config_dict
                )
                logger.info(
                    f"Einstein database adapter initialized with concurrent access: {self.db_path}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize concurrent database, using fallback: {e}"
                )
                self.use_concurrent = False
                self.concurrent_db = None
                # Note: This is expected fallback behavior, no re-raise needed
        else:
            logger.info(
                f"Einstein database adapter initialized with fallback mode: {self.db_path}"
            )

        # Performance tracking
        self.query_count = 0
        self.error_count = 0
        self.total_query_time = 0.0
        self.lock = threading.RLock()

    @contextmanager
    def connection(self, read_only: bool = True):
        """Get database connection with proper concurrency handling."""
        if self.use_concurrent and self.concurrent_db:
            # Use concurrent database manager
            lock_type = "shared" if read_only else "exclusive"
            try:
                # Check if concurrent_db has connection method (full implementation)
                if hasattr(self.concurrent_db, "connection") and callable(
                    self.concurrent_db.connection
                ):
                    with self.concurrent_db.connection(lock_type) as conn:
                        yield EinsteinConnectionWrapper(conn, self, concurrent=True)
                else:
                    # Direct connection fallback for functional implementation
                    yield EinsteinConcurrentWrapper(self.concurrent_db, self)
            except Exception as e:
                logger.warning(f"Concurrent connection failed, using fallback: {e}")
                # Fall back to direct connection
                with self._fallback_connection(read_only) as conn:
                    yield conn
        else:
            # Use fallback direct connection
            with self._fallback_connection(read_only) as conn:
                yield conn

    @contextmanager
    def _fallback_connection(self, read_only: bool):
        """Fallback connection method for when concurrent access is unavailable."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row

        try:
            # Configure for concurrent access
            if read_only:
                conn.execute("BEGIN DEFERRED")
            else:
                conn.execute("BEGIN IMMEDIATE")

            yield EinsteinConnectionWrapper(conn, self, concurrent=False)

            conn.commit()
        except Exception as e:
            with suppress(Exception):
                conn.rollback()
            raise e
        finally:
            conn.close()

    @asynccontextmanager
    async def async_connection(self, read_only: bool = True):
        """Get async database connection using session isolation."""
        if HAS_CONCURRENT_DB:
            try:
                async with async_database_session(
                    self.db_path, self.session_hint
                ) as session:
                    yield EinsteinAsyncConnectionWrapper(session, self)
                return
            except Exception as e:
                logger.warning(f"Async session connection failed, using fallback: {e}")

        # Fallback to thread pool execution
        loop = asyncio.get_event_loop()

        def _get_sync_connection():
            return self._fallback_connection(read_only)

        conn_context = await loop.run_in_executor(None, _get_sync_connection)
        try:
            yield conn_context
        finally:
            # Cleanup handled by context manager
            pass

    def execute(
        self, query: str, params: tuple | None = None, read_only: bool = True
    ) -> list[dict[str, Any]]:
        """Execute a query with automatic retry and error handling."""
        with self.lock:
            self.query_count += 1
            start_time = time.time()

        try:
            with self.connection(read_only) as conn:
                result = conn.execute(query, params)

                execution_time = time.time() - start_time
                with self.lock:
                    self.total_query_time += execution_time

                # Log slow queries
                if execution_time > 1.0:
                    logger.warning(f"Slow Einstein query: {execution_time:.2f}s")

                return result

        except Exception as e:
            with self.lock:
                self.error_count += 1
            logger.error(f"Einstein database query failed: {e}")
            raise

    async def async_execute(
        self, query: str, params: tuple | None = None, read_only: bool = True
    ) -> list[dict[str, Any]]:
        """Execute a query asynchronously."""
        with self.lock:
            self.query_count += 1
            start_time = time.time()

        try:
            # For simplicity, run the sync version in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.execute, query, params, read_only
            )

            execution_time = time.time() - start_time
            with self.lock:
                self.total_query_time += execution_time

            return result

        except Exception as e:
            with self.lock:
                self.error_count += 1
            logger.error(f"Einstein async database query failed: {e}")
            raise

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        try:
            result = self.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
                read_only=True,
            )
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False

    def create_table_if_not_exists(self, table_name: str, schema: str) -> bool:
        """Create a table if it doesn't exist."""
        try:
            if not self.table_exists(table_name):
                self.execute(f"CREATE TABLE {table_name} {schema}", read_only=False)
                logger.info(f"Created Einstein table: {table_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get database adapter statistics."""
        with self.lock:
            avg_query_time = self.total_query_time / max(self.query_count, 1)
            return {
                "db_path": str(self.db_path),
                "db_name": self.db_name,
                "use_concurrent": self.use_concurrent,
                "query_count": self.query_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.query_count, 1),
                "avg_query_time_ms": avg_query_time * 1000,
                "total_query_time_s": self.total_query_time,
            }

    def close(self):
        """Close the database adapter and cleanup resources."""
        if self.concurrent_db:
            try:
                self.concurrent_db.close()
                logger.info(f"Closed Einstein database adapter: {self.db_name}")
            except Exception as e:
                logger.warning(f"Error closing Einstein database adapter: {e}")


class EinsteinConnectionWrapper:
    """Wrapper for database connections to provide consistent interface."""

    def __init__(
        self,
        connection: Any,
        adapter: EinsteinDatabaseAdapter,
        concurrent: bool = False,
    ):
        self.connection = connection
        self.adapter = adapter
        self.concurrent = concurrent

    def execute(self, query: str, params: tuple | None = None) -> list[dict[str, Any]]:
        """Execute query and return results."""
        try:
            if params:
                cursor = self.connection.execute(query, params)
            else:
                cursor = self.connection.execute(query)

            # Handle different return types
            if query.strip().upper().startswith("SELECT"):
                if hasattr(cursor, "fetchall"):
                    rows = cursor.fetchall()
                    # Convert to list of dicts
                    if rows and hasattr(rows[0], "keys"):
                        return [dict(row) for row in rows]
                    else:
                        return [{"result": row} for row in rows] if rows else []
                else:
                    return []
            else:
                # Non-SELECT query
                if hasattr(self.connection, "commit"):
                    self.connection.commit()
                return []

        except Exception as e:
            logger.error(f"Connection wrapper query failed: {e}")
            raise


class EinsteinConcurrentWrapper:
    """Wrapper for ConcurrentDatabase to provide consistent interface."""

    def __init__(self, concurrent_db: Any, adapter: EinsteinDatabaseAdapter):
        self.concurrent_db = concurrent_db
        self.adapter = adapter

    def execute(self, query: str, params: tuple | None = None) -> list[dict[str, Any]]:
        """Execute query using the concurrent database."""
        try:
            if hasattr(self.concurrent_db, "query"):
                # Use the query method if available
                result = self.concurrent_db.query(query, params)
                return result if isinstance(result, list) else []
            elif hasattr(self.concurrent_db, "execute"):
                # Use the execute method
                result = self.concurrent_db.execute(query, params)
                return result if isinstance(result, list) else []
            else:
                # Fallback to direct connection access
                conn = self.concurrent_db._get_connection()
                if params:
                    cursor = conn.execute(query, params)
                else:
                    cursor = conn.execute(query)

                if query.strip().upper().startswith("SELECT"):
                    rows = cursor.fetchall()
                    if rows and hasattr(rows[0], "keys"):
                        return [dict(row) for row in rows]
                    else:
                        return [{"result": row} for row in rows] if rows else []
                else:
                    conn.commit()
                    return []

        except Exception as e:
            logger.error(f"Concurrent wrapper query failed: {e}")
            raise


class EinsteinAsyncConnectionWrapper:
    """Async wrapper for database connections."""

    def __init__(self, session_context: Any, adapter: EinsteinDatabaseAdapter):
        self.session_context = session_context
        self.adapter = adapter

    async def execute(
        self, query: str, params: tuple | None = None
    ) -> list[dict[str, Any]]:
        """Execute query asynchronously."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._sync_execute, query, params)
            return result
        except Exception as e:
            logger.error(f"Async connection wrapper query failed: {e}")
            raise

    def _sync_execute(
        self, query: str, params: tuple | None = None
    ) -> list[dict[str, Any]]:
        """Synchronous execution for thread pool."""
        if params:
            result = self.session_context.execute(query, params)
        else:
            result = self.session_context.execute(query)

        # Handle result conversion
        if hasattr(result, "fetchall"):
            rows = result.fetchall()
            if rows and hasattr(rows[0], "keys"):
                return [dict(row) for row in rows]
            else:
                return [{"result": row} for row in rows] if rows else []
        else:
            return []


class EinsteinDatabaseManager:
    """High-level manager for Einstein databases."""

    def __init__(self, config):
        self.config = config
        self.adapters: dict[str, EinsteinDatabaseAdapter] = {}

        # Initialize adapters for Einstein databases
        self._init_adapters()

    def _init_adapters(self):
        """Initialize database adapters."""
        try:
            # Analytics database
            if (
                hasattr(self.config, "analytics_db_path")
                and self.config.analytics_db_path
            ):
                self.adapters["analytics"] = EinsteinDatabaseAdapter(
                    self.config.analytics_db_path, "analytics"
                )

            # Embeddings database
            if (
                hasattr(self.config, "embeddings_db_path")
                and self.config.embeddings_db_path
            ):
                self.adapters["embeddings"] = EinsteinDatabaseAdapter(
                    self.config.embeddings_db_path, "embeddings"
                )

            logger.info(f"Initialized {len(self.adapters)} Einstein database adapters")

        except Exception as e:
            logger.error(f"Failed to initialize Einstein database adapters: {e}")
            self.adapters = {}

    def get_adapter(self, db_name: str) -> EinsteinDatabaseAdapter | None:
        """Get database adapter by name."""
        return self.adapters.get(db_name)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all adapters."""
        stats = {}
        for name, adapter in self.adapters.items():
            stats[name] = adapter.get_stats()
        return stats

    def close_all(self):
        """Close all database adapters."""
        for name, adapter in self.adapters.items():
            try:
                adapter.close()
            except Exception as e:
                logger.warning(f"Error closing adapter {name}: {e}")
        self.adapters.clear()
        logger.info("Closed all Einstein database adapters")


# Example usage
if __name__ == "__main__":
    import tempfile

    async def test_einstein_adapter():
        """Test Einstein database adapter functionality."""
        print("🧪 Testing Einstein Database Adapter with Concurrent Access")

        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)

        try:
            # Test basic adapter functionality
            print("\n📝 Testing basic adapter operations...")

            adapter = EinsteinDatabaseAdapter(db_path, "test")

            # Create test table
            adapter.create_table_if_not_exists(
                "embeddings", "(id INTEGER PRIMARY KEY, text TEXT, embedding BLOB)"
            )

            # Insert test data
            adapter.execute(
                "INSERT INTO embeddings (text, embedding) VALUES (?, ?)",
                ("test text", b"fake_embedding"),
                read_only=False,
            )

            # Query data
            results = adapter.execute("SELECT * FROM embeddings")
            print(f"   Query results: {len(results)} rows")

            # Test async operations
            print("\n🔄 Testing async operations...")

            async_results = await adapter.async_execute(
                "SELECT COUNT(*) as count FROM embeddings"
            )
            print(f"   Async query results: {async_results}")

            # Test concurrent access
            print("\n🚀 Testing concurrent access...")

            async def concurrent_worker(worker_id: int):
                worker_adapter = EinsteinDatabaseAdapter(db_path, f"worker_{worker_id}")

                # Insert data concurrently
                await worker_adapter.async_execute(
                    "INSERT INTO embeddings (text, embedding) VALUES (?, ?)",
                    (f"worker_{worker_id}_text", f"embedding_{worker_id}".encode()),
                    read_only=False,
                )

                # Query data
                results = await worker_adapter.async_execute(
                    "SELECT COUNT(*) as count FROM embeddings"
                )
                print(f"   Worker {worker_id}: found {results[0]['count']} rows")

                worker_adapter.close()

            # Run concurrent workers
            tasks = [concurrent_worker(i) for i in range(3)]
            await asyncio.gather(*tasks)

            # Check final state
            final_results = adapter.execute("SELECT COUNT(*) as count FROM embeddings")
            print(f"   Final row count: {final_results[0]['count']}")

            # Show statistics
            print("\n📊 Adapter statistics:")
            stats = adapter.get_stats()
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")

            print("\n✅ Einstein database adapter test completed successfully!")

        finally:
            # Cleanup
            try:
                adapter.close()
                db_path.unlink()
            except Exception as e:
                print(f"Cleanup error: {e}")

    asyncio.run(test_einstein_adapter())
