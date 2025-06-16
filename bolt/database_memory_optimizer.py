#!/usr/bin/env python3
"""
Database Memory Optimizer for M4 Pro - DuckDB Connection Pool Management
Target: Reduce database memory usage from 9GB to under 1GB

Key Features:
1. Optimized connection pooling
2. Memory-constrained query execution
3. Result set streaming and pagination
4. Connection-level memory limits
5. Query result caching with eviction
6. Batch query optimization
7. Memory pressure monitoring
"""

import asyncio
import gc
import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DatabaseMemoryStats:
    """Database memory usage statistics."""

    connection_count: int
    active_connections: int
    total_queries: int
    cached_results: int
    memory_limit_mb: int
    estimated_usage_mb: float
    cache_hits: int
    cache_misses: int
    evictions: int
    last_cleanup: float


@dataclass
class QueryResult:
    """Cached query result with metadata."""

    result: Any
    query_hash: str
    timestamp: float
    size_estimate_mb: float
    access_count: int
    last_accessed: float


class StreamingResultSet:
    """Streaming result set to minimize memory usage."""

    def __init__(self, connection, query: str, chunk_size: int = 10000):
        self.connection = connection
        self.query = query
        self.chunk_size = chunk_size
        self._cursor = None
        self._exhausted = False

    def __enter__(self):
        """Context manager entry."""
        self._cursor = self.connection.execute(self.query)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._cursor:
            with suppress(Exception):
                self._cursor.close()

    def __iter__(self) -> Iterator[list[tuple]]:
        """Iterate over result chunks."""
        if not self._cursor:
            raise RuntimeError("StreamingResultSet not properly initialized")

        while not self._exhausted:
            chunk = self._cursor.fetchmany(self.chunk_size)
            if not chunk:
                self._exhausted = True
                break
            yield chunk

    def to_pandas_chunks(self, max_memory_mb: float = 100) -> Iterator[pd.DataFrame]:
        """Convert to pandas DataFrames in chunks with memory limit."""
        estimated_rows_per_mb = 1000  # Rough estimate
        max_chunk_size = max(1000, int(max_memory_mb * estimated_rows_per_mb))

        for chunk in self:
            try:
                df = pd.DataFrame(
                    chunk, columns=[desc[0] for desc in self._cursor.description]
                )

                # If chunk is too large, split it further
                if len(df) > max_chunk_size:
                    for i in range(0, len(df), max_chunk_size):
                        yield df.iloc[i : i + max_chunk_size]
                else:
                    yield df
            except Exception as e:
                logger.error(f"Error converting chunk to DataFrame: {e}")
                break


class QueryResultCache:
    """LRU cache for query results with memory pressure awareness."""

    def __init__(self, max_cache_size_mb: float = 200, max_items: int = 100):
        self.max_cache_size_mb = max_cache_size_mb
        self.max_items = max_items
        self._cache: OrderedDict[str, QueryResult] = OrderedDict()
        self._current_size_mb = 0.0
        self._lock = threading.RLock()

        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, query_hash: str) -> Any | None:
        """Get cached result."""
        with self._lock:
            if query_hash not in self._cache:
                self.misses += 1
                return None

            # Move to end (most recent)
            result_entry = self._cache[query_hash]
            self._cache.move_to_end(query_hash)

            # Update access stats
            result_entry.access_count += 1
            result_entry.last_accessed = time.time()

            self.hits += 1
            return result_entry.result

    def put(self, query_hash: str, result: Any, size_estimate_mb: float):
        """Cache query result."""
        with self._lock:
            # Don't cache if result is too large
            if size_estimate_mb > self.max_cache_size_mb / 2:
                logger.debug(
                    f"Query result too large to cache: {size_estimate_mb:.1f}MB"
                )
                return

            # Create result entry
            result_entry = QueryResult(
                result=result,
                query_hash=query_hash,
                timestamp=time.time(),
                size_estimate_mb=size_estimate_mb,
                access_count=1,
                last_accessed=time.time(),
            )

            # Add to cache
            self._cache[query_hash] = result_entry
            self._current_size_mb += size_estimate_mb

            # Evict if necessary
            self._evict_if_needed()

    def _evict_if_needed(self):
        """Evict entries if cache is too large."""
        while (
            len(self._cache) > self.max_items
            or self._current_size_mb > self.max_cache_size_mb
        ):
            if not self._cache:
                break

            # Evict least recently used
            query_hash, result_entry = self._cache.popitem(last=False)
            self._current_size_mb -= result_entry.size_estimate_mb
            self.evictions += 1

            logger.debug(
                f"Evicted cached query result: {query_hash[:16]}... ({result_entry.size_estimate_mb:.1f}MB)"
            )

    def clear(self):
        """Clear all cached results."""
        with self._lock:
            self._cache.clear()
            self._current_size_mb = 0.0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = (
                self.hits / (self.hits + self.misses)
                if (self.hits + self.misses) > 0
                else 0
            )
            return {
                "cached_items": len(self._cache),
                "cache_size_mb": self._current_size_mb,
                "max_cache_size_mb": self.max_cache_size_mb,
                "hit_rate": hit_rate,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }


class OptimizedDatabaseConnection:
    """Optimized database connection with memory constraints."""

    def __init__(self, connection_string: str, memory_limit_mb: int = 200):
        self.connection_string = connection_string
        self.memory_limit_mb = memory_limit_mb
        self._connection = None
        self._in_use = False
        self._created_at = time.time()
        self._last_used = time.time()
        self._query_count = 0
        self._lock = threading.Lock()

    def _create_connection(self):
        """Create optimized database connection."""
        if ":memory:" in self.connection_string or self.connection_string == "":
            # In-memory DuckDB
            if DUCKDB_AVAILABLE:
                conn = duckdb.connect(":memory:")
                # Configure for low memory usage
                conn.execute(f"SET memory_limit='{self.memory_limit_mb}MB'")
                conn.execute("SET threads=2")  # Limit parallel threads for M4 Pro
                conn.execute(f"SET max_memory='{self.memory_limit_mb}MB'")
                conn.execute(
                    "SET enable_profiling='false'"
                )  # Disable profiling to save memory
                conn.execute("SET enable_progress_bar='false'")
                return conn
            else:
                # Fallback to SQLite
                conn = sqlite3.connect(":memory:")
                conn.execute("PRAGMA cache_size = -10000")  # 10MB cache
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                return conn
        else:
            # File-based database
            path = Path(self.connection_string)
            if DUCKDB_AVAILABLE and path.suffix.lower() in [".db", ".duckdb", ""]:
                conn = duckdb.connect(str(path))
                conn.execute(f"SET memory_limit='{self.memory_limit_mb}MB'")
                conn.execute("SET threads=2")
                conn.execute(f"SET max_memory='{self.memory_limit_mb}MB'")
                return conn
            else:
                # SQLite
                conn = sqlite3.connect(str(path))
                conn.execute("PRAGMA cache_size = -10000")  # 10MB cache
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                return conn

    @contextmanager
    def acquire(self):
        """Acquire connection for use."""
        with self._lock:
            if self._in_use:
                raise RuntimeError("Connection already in use")

            if self._connection is None:
                self._connection = self._create_connection()

            self._in_use = True
            self._last_used = time.time()

        try:
            yield self._connection
        finally:
            with self._lock:
                self._in_use = False
                self._query_count += 1

    def execute_streaming(
        self, query: str, chunk_size: int = 10000
    ) -> StreamingResultSet:
        """Execute query with streaming results."""
        # This should be used within acquire() context
        if not self._connection:
            raise RuntimeError("Connection not initialized")

        return StreamingResultSet(self._connection, query, chunk_size)

    def is_expired(self, max_age_minutes: int = 30) -> bool:
        """Check if connection should be refreshed."""
        age_minutes = (time.time() - self._created_at) / 60
        idle_minutes = (time.time() - self._last_used) / 60

        return age_minutes > max_age_minutes or idle_minutes > 10

    def close(self):
        """Close the connection."""
        with self._lock:
            if self._connection:
                with suppress(Exception):
                    self._connection.close()
                self._connection = None

    def get_stats(self) -> dict[str, Any]:
        """Get connection statistics."""
        return {
            "in_use": self._in_use,
            "age_minutes": (time.time() - self._created_at) / 60,
            "idle_minutes": (time.time() - self._last_used) / 60,
            "query_count": self._query_count,
            "memory_limit_mb": self.memory_limit_mb,
        }


class M4ProDatabaseMemoryManager:
    """Optimized database memory manager for M4 Pro."""

    def __init__(self, max_connections: int = 3, memory_budget_mb: int = 1000):
        self.max_connections = max_connections
        self.memory_budget_mb = memory_budget_mb
        self.connection_memory_limit_mb = (
            memory_budget_mb // max_connections
        )  # Split budget

        # Connection pool
        self._connections: list[OptimizedDatabaseConnection] = []
        self._connection_lock = threading.Lock()

        # Query result cache
        cache_memory_mb = min(200, memory_budget_mb // 4)  # 25% of budget for caching
        self.result_cache = QueryResultCache(max_cache_size_mb=cache_memory_mb)

        # Stats
        self.stats = DatabaseMemoryStats(
            connection_count=0,
            active_connections=0,
            total_queries=0,
            cached_results=0,
            memory_limit_mb=memory_budget_mb,
            estimated_usage_mb=0,
            cache_hits=0,
            cache_misses=0,
            evictions=0,
            last_cleanup=time.time(),
        )

        # Monitoring
        self._monitoring_active = False
        self._monitor_task = None

        logger.info(
            f"M4 Pro Database Memory Manager initialized (budget: {memory_budget_mb}MB, "
            f"{max_connections} connections @ {self.connection_memory_limit_mb}MB each)"
        )

    @contextmanager
    def get_connection(self, connection_string: str = ":memory:"):
        """Get optimized database connection."""
        connection = None
        try:
            with self._connection_lock:
                # Find available connection
                for conn in self._connections:
                    if not conn._in_use and conn.connection_string == connection_string:
                        if not conn.is_expired():
                            connection = conn
                            break
                        else:
                            # Remove expired connection
                            conn.close()
                            self._connections.remove(conn)

                # Create new connection if needed
                if connection is None and len(self._connections) < self.max_connections:
                    connection = OptimizedDatabaseConnection(
                        connection_string, self.connection_memory_limit_mb
                    )
                    self._connections.append(connection)

                if connection is None:
                    raise RuntimeError("No database connections available")

            # Use the connection
            with connection.acquire() as conn:
                self.stats.total_queries += 1
                yield conn

        finally:
            # Update stats
            self._update_stats()

    def execute_query_cached(
        self,
        query: str,
        connection_string: str = ":memory:",
        cache_key: str | None = None,
    ) -> Any:
        """Execute query with result caching."""
        # Generate cache key
        if cache_key is None:
            cache_key = f"{hash(query)}"

        # Check cache first
        cached_result = self.result_cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Query cache hit: {cache_key[:16]}...")
            return cached_result

        # Execute query
        with self.get_connection(connection_string) as conn:
            if DUCKDB_AVAILABLE and hasattr(conn, "execute"):
                result = conn.execute(query).fetchall()
            else:
                cursor = conn.execute(query)
                result = cursor.fetchall()
                cursor.close()

        # Estimate result size and cache if reasonable
        size_estimate_mb = self._estimate_result_size(result)
        if size_estimate_mb < 50:  # Only cache results under 50MB
            self.result_cache.put(cache_key, result, size_estimate_mb)

        return result

    def execute_query_streaming(
        self, query: str, connection_string: str = ":memory:", chunk_size: int = 10000
    ) -> Iterator[list[tuple]]:
        """Execute query with streaming results to minimize memory."""
        with self.get_connection(connection_string) as conn:
            if DUCKDB_AVAILABLE and hasattr(conn, "execute"):
                cursor = conn.execute(query)
                while True:
                    chunk = cursor.fetchmany(chunk_size)
                    if not chunk:
                        break
                    yield chunk
                cursor.close()
            else:
                cursor = conn.execute(query)
                while True:
                    chunk = cursor.fetchmany(chunk_size)
                    if not chunk:
                        break
                    yield chunk
                cursor.close()

    def execute_query_to_pandas(
        self,
        query: str,
        connection_string: str = ":memory:",
        max_memory_mb: float = 100,
    ) -> Iterator[pd.DataFrame]:
        """Execute query and return pandas DataFrames in memory-limited chunks."""
        for chunk in self.execute_query_streaming(query, connection_string):
            try:
                # Convert chunk to DataFrame
                if DUCKDB_AVAILABLE:
                    # Use DuckDB's built-in pandas integration if available
                    with self.get_connection(connection_string) as conn:
                        if hasattr(conn, "execute") and hasattr(conn, "df"):
                            # This is a DuckDB connection
                            df = conn.execute(query).df()

                            # Split large DataFrames
                            rows_per_mb = max(
                                1000, int(max_memory_mb * 1000)
                            )  # Rough estimate
                            if len(df) > rows_per_mb:
                                for i in range(0, len(df), rows_per_mb):
                                    yield df.iloc[i : i + rows_per_mb]
                            else:
                                yield df
                            return

                # Fallback: convert tuple chunks to DataFrames
                df = pd.DataFrame(chunk)
                yield df

            except Exception as e:
                logger.error(f"Error converting query result to pandas: {e}")
                break

    def _estimate_result_size(self, result: Any) -> float:
        """Estimate memory size of query result in MB."""
        if not result:
            return 0.0

        try:
            if isinstance(result, list) and result:
                # Estimate based on first row
                first_row = result[0]
                if isinstance(first_row, list | tuple):
                    row_size = (
                        sum(len(str(item)) for item in first_row) * 2
                    )  # Rough estimate
                    return (len(result) * row_size) / (1024 * 1024)
                else:
                    return len(str(result)) / (1024 * 1024)
            else:
                return len(str(result)) / (1024 * 1024)
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Could not estimate result memory usage: {e}")
            return 1.0  # Default 1MB estimate

    def cleanup_connections(self, force: bool = False):
        """Clean up expired connections and caches."""
        with self._connection_lock:
            expired_connections = []

            for conn in self._connections:
                if force or (conn.is_expired() and not conn._in_use):
                    expired_connections.append(conn)

            for conn in expired_connections:
                conn.close()
                self._connections.remove(conn)

            logger.debug(f"Cleaned up {len(expired_connections)} database connections")

        # Clear result cache if under memory pressure
        if force or self._is_under_memory_pressure():
            cache_stats = self.result_cache.get_stats()
            self.result_cache.clear()
            logger.info(
                f"Cleared database result cache: {cache_stats['cached_items']} items, {cache_stats['cache_size_mb']:.1f}MB"
            )

        # Force garbage collection
        if force:
            collected = gc.collect()
            logger.debug(f"Database cleanup GC collected {collected} objects")

        self.stats.last_cleanup = time.time()

    def _is_under_memory_pressure(self) -> bool:
        """Check if database memory usage is high."""
        estimated_usage = self._estimate_total_memory_usage()
        return estimated_usage > self.memory_budget_mb * 0.8  # 80% threshold

    def _estimate_total_memory_usage(self) -> float:
        """Estimate total database memory usage."""
        connection_usage = len(self._connections) * self.connection_memory_limit_mb
        cache_usage = self.result_cache.get_stats()["cache_size_mb"]
        return connection_usage + cache_usage

    def _update_stats(self):
        """Update internal statistics."""
        cache_stats = self.result_cache.get_stats()

        self.stats.connection_count = len(self._connections)
        self.stats.active_connections = sum(
            1 for conn in self._connections if conn._in_use
        )
        self.stats.cached_results = cache_stats["cached_items"]
        self.stats.estimated_usage_mb = self._estimate_total_memory_usage()
        self.stats.cache_hits = cache_stats["hits"]
        self.stats.cache_misses = cache_stats["misses"]
        self.stats.evictions = cache_stats["evictions"]

    async def start_monitoring(self, interval_seconds: float = 30.0):
        """Start background memory monitoring."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitor_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
        logger.info("Started database memory monitoring")

    async def stop_monitoring(self):
        """Stop background memory monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._monitor_task

        logger.info("Stopped database memory monitoring")

    async def _monitoring_loop(self, interval_seconds: float):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                self._update_stats()

                # Check for memory pressure
                if self._is_under_memory_pressure():
                    logger.warning(
                        f"Database memory pressure: {self.stats.estimated_usage_mb:.1f}MB / {self.memory_budget_mb}MB"
                    )
                    self.cleanup_connections(force=False)

                # Periodic cleanup of expired connections
                if time.time() - self.stats.last_cleanup > 300:  # 5 minutes
                    self.cleanup_connections(force=False)

                # Periodic logging
                if time.time() % 300 < interval_seconds:  # Log every 5 minutes
                    cache_stats = self.result_cache.get_stats()
                    logger.info(
                        f"Database Memory: {self.stats.estimated_usage_mb:.1f}MB used, "
                        f"{self.stats.connection_count} connections, "
                        f"Cache hit rate: {cache_stats['hit_rate']:.1%}"
                    )

                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Database memory monitoring error: {e}")
                await asyncio.sleep(interval_seconds)

    def get_memory_report(self) -> dict[str, Any]:
        """Get comprehensive memory report."""
        self._update_stats()
        cache_stats = self.result_cache.get_stats()

        connection_stats = []
        for i, conn in enumerate(self._connections):
            connection_stats.append(
                {
                    "id": i,
                    "connection_string": conn.connection_string,
                    **conn.get_stats(),
                }
            )

        return {
            "timestamp": time.time(),
            "memory_budget_mb": self.memory_budget_mb,
            "estimated_usage_mb": self.stats.estimated_usage_mb,
            "usage_percent": (self.stats.estimated_usage_mb / self.memory_budget_mb)
            * 100,
            "under_pressure": self._is_under_memory_pressure(),
            "connections": {
                "total": self.stats.connection_count,
                "active": self.stats.active_connections,
                "max": self.max_connections,
                "per_connection_limit_mb": self.connection_memory_limit_mb,
                "details": connection_stats,
            },
            "cache": cache_stats,
            "queries": {
                "total": self.stats.total_queries,
                "cache_hit_rate": cache_stats["hit_rate"],
            },
            "optimization_recommendations": self._get_optimization_recommendations(),
        }

    def _get_optimization_recommendations(self) -> list[str]:
        """Get database memory optimization recommendations."""
        recommendations = []

        if self._is_under_memory_pressure():
            recommendations.append(
                f"High database memory usage: {self.stats.estimated_usage_mb:.1f}MB / {self.memory_budget_mb}MB"
            )

        cache_stats = self.result_cache.get_stats()
        if cache_stats["hit_rate"] < 0.3:
            recommendations.append(
                f"Low cache hit rate: {cache_stats['hit_rate']:.1%} - consider query optimization"
            )

        if self.stats.connection_count >= self.max_connections:
            recommendations.append(
                "Connection pool at capacity - consider connection reuse"
            )

        expired_count = sum(1 for conn in self._connections if conn.is_expired())
        if expired_count > 0:
            recommendations.append(f"{expired_count} expired connections need cleanup")

        if not recommendations:
            recommendations.append("Database memory usage is optimal")

        return recommendations

    def shutdown(self):
        """Shutdown database memory manager."""
        if self._monitoring_active:
            asyncio.create_task(self.stop_monitoring())

        self.cleanup_connections(force=True)
        self.result_cache.clear()
        logger.info("M4 Pro Database Memory Manager shutdown complete")


# Global instance
_database_memory_manager: M4ProDatabaseMemoryManager | None = None


def get_database_memory_manager() -> M4ProDatabaseMemoryManager:
    """Get the global database memory manager."""
    global _database_memory_manager
    if _database_memory_manager is None:
        _database_memory_manager = M4ProDatabaseMemoryManager()
    return _database_memory_manager


# Convenience functions
def get_optimized_connection(connection_string: str = ":memory:"):
    """Get optimized database connection."""
    return get_database_memory_manager().get_connection(connection_string)


def execute_cached_query(
    query: str, connection_string: str = ":memory:", cache_key: str | None = None
):
    """Execute query with result caching."""
    return get_database_memory_manager().execute_query_cached(
        query, connection_string, cache_key
    )


def execute_streaming_query(
    query: str, connection_string: str = ":memory:", chunk_size: int = 10000
):
    """Execute query with streaming results."""
    return get_database_memory_manager().execute_query_streaming(
        query, connection_string, chunk_size
    )


def execute_pandas_query(
    query: str, connection_string: str = ":memory:", max_memory_mb: float = 100
):
    """Execute query and return pandas DataFrames."""
    return get_database_memory_manager().execute_query_to_pandas(
        query, connection_string, max_memory_mb
    )


def cleanup_database_memory(force: bool = False):
    """Clean up database memory."""
    get_database_memory_manager().cleanup_connections(force)


def get_database_memory_report() -> dict[str, Any]:
    """Get database memory usage report."""
    return get_database_memory_manager().get_memory_report()


if __name__ == "__main__":
    # Test the database memory manager
    print("Testing M4 Pro Database Memory Manager...")

    manager = get_database_memory_manager()

    # Test connection
    with get_optimized_connection() as conn:
        if DUCKDB_AVAILABLE:
            # Test DuckDB
            conn.execute("CREATE TABLE test (id INTEGER, value TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'hello'), (2, 'world')")
            result = conn.execute("SELECT * FROM test").fetchall()
            print(f"DuckDB test result: {result}")
        else:
            # Test SQLite
            conn.execute("CREATE TABLE test (id INTEGER, value TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'hello'), (2, 'world')")
            cursor = conn.execute("SELECT * FROM test")
            result = cursor.fetchall()
            cursor.close()
            print(f"SQLite test result: {result}")

    # Test cached query
    cached_result = execute_cached_query("SELECT 1 as test_col")
    print(f"Cached query result: {cached_result}")

    # Test cache hit
    cached_result2 = execute_cached_query("SELECT 1 as test_col")
    print(f"Cached query result (should be cache hit): {cached_result2}")

    # Get report
    report = get_database_memory_report()
    print("\nDatabase Memory Report:")
    print(
        f"Usage: {report['estimated_usage_mb']:.1f}MB / {report['memory_budget_mb']}MB ({report['usage_percent']:.1f}%)"
    )
    print(
        f"Connections: {report['connections']['active']} / {report['connections']['total']}"
    )
    print(f"Cache hit rate: {report['cache']['hit_rate']:.1%}")
    print(f"Recommendations: {', '.join(report['optimization_recommendations'])}")

    print("Test completed successfully!")
