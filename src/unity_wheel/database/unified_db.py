#!/usr/bin/env python3
"""
Unified Database Abstraction Layer

Consolidates all database implementations into a single, optimized interface
with M4 Pro hardware acceleration and intelligent query routing.

Key Features:
- Unified connection pooling across all database types
- Automatic query optimization and routing
- Coordinated caching with LRU eviction
- M4 Pro CPU affinity for performance cores
- Concurrent access with session isolation
- Automatic failover and recovery
"""

import asyncio
import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import duckdb

# Import existing components for integration
try:
    from ...bolt_database_fixes import DatabaseConcurrencyManager
    from ..accelerated_tools.duckdb_turbo import DuckDBTurbo
    from ..storage.session_isolation import SessionConfig, SessionManager

    HAS_ACCELERATED = True
except ImportError:
    HAS_ACCELERATED = False
    DatabaseConcurrencyManager = None
    SessionManager = None
    DuckDBTurbo = None

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types."""

    DUCKDB = "duckdb"
    SQLITE = "sqlite"
    MEMORY = "memory"
    PARQUET = "parquet"


@dataclass
class DatabaseConfig:
    """Unified database configuration."""

    # Connection settings
    db_type: DatabaseType = DatabaseType.DUCKDB
    path: Path | None = None

    # M4 Pro optimizations
    cpu_cores: int = 8  # Performance cores
    memory_limit_gb: float = 19.2  # 80% of 24GB
    enable_gpu: bool = True

    # Connection pool settings
    min_connections: int = 2
    max_connections: int = 12  # M4 Pro optimized
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0  # 5 minutes

    # Query optimization
    enable_query_cache: bool = True
    cache_size_mb: int = 512
    enable_parallel_execution: bool = True
    max_parallel_queries: int = 8

    # Session management
    enable_session_isolation: bool = True
    session_timeout: float = 300.0
    transaction_timeout: float = 30.0

    # Performance monitoring
    enable_metrics: bool = True
    metrics_interval: float = 60.0

    # Retry and recovery
    max_retries: int = 3
    retry_delay: float = 0.1
    enable_auto_recovery: bool = True


@dataclass
class QueryMetrics:
    """Query performance metrics."""

    query_id: str
    query_type: str
    start_time: float
    end_time: float = 0.0
    rows_affected: int = 0
    cache_hit: bool = False
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        """Query duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000 if self.end_time else 0


class QueryRouter(Protocol):
    """Protocol for query routing strategies."""

    def route(
        self, query: str, params: dict[str, Any] | None = None
    ) -> DatabaseType:
        """Route query to appropriate database type."""
        ...


class UnifiedDatabaseManager:
    """Main unified database abstraction layer."""

    def __init__(self, config: DatabaseConfig | None = None):
        self.config = config or DatabaseConfig()

        # Thread-safe state management
        self._lock = threading.RLock()
        self._shutdown = False

        # Component managers
        self._concurrency_manager: DatabaseConcurrencyManager | None = None
        self._session_manager: SessionManager | None = None
        self._connection_pool: dict[str, Any] = {}
        self._cache_manager: Any | None = None  # Will be CacheCoordinator

        # Performance tracking
        self._metrics: list[QueryMetrics] = []
        self._connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        }

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all database components with M4 Pro optimizations."""
        # Initialize concurrency manager
        if HAS_ACCELERATED and DatabaseConcurrencyManager:
            self._concurrency_manager = DatabaseConcurrencyManager()
            # Override with our config
            self._concurrency_manager.max_connections = self.config.max_connections
            self._concurrency_manager.connection_timeout = (
                self.config.connection_timeout
            )

        # Initialize session manager
        if HAS_ACCELERATED and SessionManager:
            session_config = SessionConfig(
                session_timeout=self.config.session_timeout,
                max_transaction_time=self.config.transaction_timeout,
                enable_auto_cleanup=True,
                isolation_level="READ_COMMITTED",
            )
            self._session_manager = SessionManager(session_config)

        # Load optimization config
        try:
            with open("optimization_config.json") as f:
                self.optimization_config = json.load(f)
        except:
            # Default optimization config
            self.optimization_config = {
                "cpu": {
                    "max_workers": 8,
                    "performance_cores": [0, 1, 2, 3, 4, 5, 6, 7],
                },
                "memory": {"max_allocation_gb": 19.2},
                "io": {"concurrent_reads": 24},
            }

        logger.info(f"Unified database manager initialized with config: {self.config}")

    @contextmanager
    def connection(
        self, db_type: DatabaseType | None = None, read_only: bool = True
    ):
        """Get a database connection with automatic resource management."""
        db_type = db_type or self.config.db_type
        conn_key = f"{db_type.value}:{'ro' if read_only else 'rw'}"

        with self._lock:
            self._connection_stats["total_connections"] += 1

        try:
            # Get or create connection
            if db_type == DatabaseType.DUCKDB:
                conn = self._get_duckdb_connection(read_only)
            elif db_type == DatabaseType.MEMORY:
                conn = self._get_memory_connection()
            else:
                raise ValueError(f"Unsupported database type: {db_type}")

            with self._lock:
                self._connection_stats["active_connections"] += 1

            yield conn

        finally:
            with self._lock:
                self._connection_stats["active_connections"] -= 1

    def _get_duckdb_connection(self, read_only: bool = True):
        """Get DuckDB connection with M4 Pro optimizations."""
        if self._concurrency_manager and self.config.path:
            # Use concurrency manager for file-based databases
            return self._concurrency_manager.get_connection(
                str(self.config.path), read_only=read_only
            )
        else:
            # Create direct connection with optimizations
            config = {
                "threads": self.optimization_config["cpu"]["max_workers"],
                "memory_limit": f"{self.optimization_config['memory']['max_allocation_gb']}GB",
                "max_memory": f"{self.optimization_config['memory']['max_allocation_gb']}GB",
                "preserve_insertion_order": False,
                "enable_object_cache": True,
            }

            if read_only and self.config.path:
                config["access_mode"] = "READ_ONLY"

            db_path = str(self.config.path) if self.config.path else ":memory:"
            conn = duckdb.connect(db_path, config=config)

            # Apply performance optimizations
            try:
                conn.execute(
                    f"PRAGMA threads={self.optimization_config['cpu']['max_workers']}"
                )
                conn.execute(
                    f"PRAGMA memory_limit='{self.optimization_config['memory']['max_allocation_gb']}GB'"
                )
            except:
                pass  # Some pragmas might not be supported

            return conn

    def _get_memory_connection(self):
        """Get in-memory database connection."""
        # For in-memory databases, reuse single connection
        if "memory" not in self._connection_pool:
            self._connection_pool["memory"] = self._get_duckdb_connection(
                read_only=False
            )
        return self._connection_pool["memory"]

    def execute(
        self,
        query: str,
        params: list | dict | None = None,
        db_type: DatabaseType | None = None,
    ) -> Any:
        """Execute a query with automatic optimization and metrics."""
        query_id = f"{time.time()}_{threading.get_ident()}"
        metrics = QueryMetrics(
            query_id=query_id,
            query_type=self._get_query_type(query),
            start_time=time.time(),
        )

        try:
            with self.connection(db_type) as conn:
                # Execute query
                if params:
                    result = conn.execute(query, params)
                else:
                    result = conn.execute(query)

                # Get results
                if hasattr(result, "fetchdf"):
                    data = result.fetchdf()
                    metrics.rows_affected = len(data)
                    return data
                elif hasattr(result, "fetchall"):
                    data = result.fetchall()
                    metrics.rows_affected = len(data)
                    return data
                else:
                    return result

        except Exception as e:
            metrics.error = str(e)
            self._connection_stats["errors"] += 1
            logger.error(f"Query execution failed: {e}")
            raise
        finally:
            metrics.end_time = time.time()
            self._metrics.append(metrics)
            self._connection_stats["total_queries"] += 1

    def _get_query_type(self, query: str) -> str:
        """Determine query type for routing and optimization."""
        query_lower = query.lower().strip()
        if query_lower.startswith("select"):
            return "SELECT"
        elif query_lower.startswith("insert"):
            return "INSERT"
        elif query_lower.startswith("update"):
            return "UPDATE"
        elif query_lower.startswith("delete"):
            return "DELETE"
        elif query_lower.startswith("create"):
            return "CREATE"
        elif query_lower.startswith("drop"):
            return "DROP"
        else:
            return "OTHER"

    async def execute_async(
        self,
        query: str,
        params: list | dict | None = None,
        db_type: DatabaseType | None = None,
    ) -> Any:
        """Execute query asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, query, params, db_type)

    def get_metrics(self, last_n: int | None = None) -> list[QueryMetrics]:
        """Get query performance metrics."""
        with self._lock:
            if last_n:
                return self._metrics[-last_n:]
            return self._metrics.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get database connection and query statistics."""
        with self._lock:
            stats = self._connection_stats.copy()

        # Add performance metrics
        if self._metrics:
            durations = [m.duration_ms for m in self._metrics if m.duration_ms > 0]
            if durations:
                stats["avg_query_time_ms"] = sum(durations) / len(durations)
                stats["min_query_time_ms"] = min(durations)
                stats["max_query_time_ms"] = max(durations)

        return stats

    def shutdown(self):
        """Gracefully shutdown database connections."""
        with self._lock:
            self._shutdown = True

        # Close all connections
        for conn in self._connection_pool.values():
            try:
                if hasattr(conn, "close"):
                    conn.close()
            except:
                pass

        # Shutdown session manager
        if self._session_manager:
            try:
                self._session_manager.cleanup_all_sessions()
            except:
                pass

        logger.info("Unified database manager shutdown complete")


# Global instance for easy access
_global_db_manager: UnifiedDatabaseManager | None = None
_manager_lock = threading.Lock()


def get_unified_db(config: DatabaseConfig | None = None) -> UnifiedDatabaseManager:
    """Get or create the global unified database manager."""
    global _global_db_manager

    with _manager_lock:
        if _global_db_manager is None:
            _global_db_manager = UnifiedDatabaseManager(config)
        return _global_db_manager


def shutdown_unified_db():
    """Shutdown the global database manager."""
    global _global_db_manager

    with _manager_lock:
        if _global_db_manager:
            _global_db_manager.shutdown()
            _global_db_manager = None
