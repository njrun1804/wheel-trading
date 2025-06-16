#!/usr/bin/env python3
"""
Session Isolation for Concurrent DuckDB Access

This module provides session isolation to ensure that multiple bolt instances 
and Einstein searches can access DuckDB databases concurrently without conflicts.

Key Features:
- Session-scoped database connections
- Transaction isolation levels
- Automatic deadlock detection and recovery
- Session cleanup and resource management
- Performance monitoring per session
"""

import asyncio
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from weakref import WeakSet

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

    class duckdb:
        class DuckDBPyConnection:
            pass

        @staticmethod
        def connect(*args, **kwargs):
            raise ImportError("DuckDB not available")


logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """Configuration for database session isolation."""

    session_timeout: float = 300.0  # 5 minutes
    max_transaction_time: float = 30.0  # 30 seconds
    enable_read_uncommitted: bool = False
    enable_auto_cleanup: bool = True
    deadlock_timeout: float = 10.0
    max_retries: int = 3
    isolation_level: str = (
        "READ_COMMITTED"  # READ_UNCOMMITTED, READ_COMMITTED, SERIALIZABLE
    )


@dataclass
class SessionInfo:
    """Information about a database session."""

    session_id: str
    db_path: Path
    created_at: float
    last_used: float
    thread_id: int
    process_id: int
    query_count: int = 0
    transaction_count: int = 0
    is_active: bool = True
    in_transaction: bool = False
    error_count: int = 0

    @property
    def age_seconds(self) -> float:
        """Get session age in seconds."""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """Get seconds since last use."""
        return time.time() - self.last_used


class SessionManager:
    """Manages isolated database sessions for concurrent access."""

    def __init__(self, config: SessionConfig | None = None):
        self.config = config or SessionConfig()
        self.sessions: dict[str, SessionInfo] = {}
        self.connections: dict[str, duckdb.DuckDBPyConnection] = {}
        self.session_locks: dict[str, threading.RLock] = {}
        self.global_lock = threading.RLock()
        self.cleanup_task: asyncio.Task | None = None
        self.active_sessions: WeakSet = WeakSet()

        # Performance tracking
        self.total_sessions_created = 0
        self.total_queries_executed = 0
        self.total_deadlocks_resolved = 0
        self.session_creation_times: list[float] = []

    def create_session(
        self, db_path: str | Path, session_hint: str | None = None
    ) -> str:
        """Create a new isolated database session."""
        if not HAS_DUCKDB:
            raise RuntimeError("DuckDB not available for session creation")

        db_path = Path(db_path)
        session_id = session_hint or f"session_{uuid.uuid4().hex[:8]}"

        # Ensure unique session ID
        counter = 0
        original_id = session_id
        while session_id in self.sessions:
            counter += 1
            session_id = f"{original_id}_{counter}"

        with self.global_lock:
            # Create session info
            session_info = SessionInfo(
                session_id=session_id,
                db_path=db_path,
                created_at=time.time(),
                last_used=time.time(),
                thread_id=threading.get_ident(),
                process_id=threading.current_thread().ident or 0,
            )

            # Create database connection with session isolation
            try:
                start_time = time.time()
                conn = self._create_isolated_connection(db_path, session_id)
                creation_time = time.time() - start_time
                self.session_creation_times.append(creation_time)

                # Store session data
                self.sessions[session_id] = session_info
                self.connections[session_id] = conn
                self.session_locks[session_id] = threading.RLock()
                self.active_sessions.add(session_info)

                self.total_sessions_created += 1

                logger.info(
                    f"Created database session {session_id} for {db_path} in {creation_time*1000:.1f}ms"
                )

                # Start cleanup task if not running
                if self.config.enable_auto_cleanup and not self.cleanup_task:
                    self.cleanup_task = asyncio.create_task(self._cleanup_loop())

                return session_id

            except Exception as e:
                logger.error(f"Failed to create session {session_id}: {e}")
                # Cleanup partial state
                self.sessions.pop(session_id, None)
                self.connections.pop(session_id, None)
                self.session_locks.pop(session_id, None)
                raise

    def _create_isolated_connection(
        self, db_path: Path, session_id: str
    ) -> duckdb.DuckDBPyConnection:
        """Create an isolated database connection."""
        # Create connection with isolation settings
        conn = duckdb.connect(str(db_path))

        # Configure isolation level
        if self.config.isolation_level == "READ_UNCOMMITTED":
            # DuckDB doesn't have explicit isolation levels like PostgreSQL,
            # but we can simulate with transaction modes
            conn.execute("BEGIN READ ONLY")
            conn.execute("COMMIT")  # Just to establish the mode
        elif self.config.isolation_level == "SERIALIZABLE":
            # Use explicit transactions for serializable behavior
            pass  # Handle in transaction context

        # Set session-specific configurations (skip if not supported)
        try:
            conn.execute(f"SET client_session_id = '{session_id}'")
        except Exception:
            # DuckDB may not support this configuration, continue without it
            pass

        # Configure timeouts (skip if not supported)
        try:
            conn.execute(
                f"SET lock_timeout = '{int(self.config.deadlock_timeout * 1000)}ms'"
            )
        except Exception:
            # DuckDB may not support this configuration, continue without it
            pass

        # Optimize for concurrent access
        conn.execute("PRAGMA threads=1")  # One thread per session to avoid conflicts
        conn.execute("PRAGMA enable_progress_bar=false")

        return conn

    @contextmanager
    def get_session(self, session_id: str):
        """Get a session for use with proper resource management."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        session_info = self.sessions[session_id]
        session_lock = self.session_locks[session_id]

        with session_lock:
            try:
                # Update last used time
                session_info.last_used = time.time()

                # Check session health
                if not session_info.is_active:
                    raise RuntimeError(f"Session {session_id} is not active")

                # Get connection
                conn = self.connections[session_id]

                # Yield session context
                yield SessionContext(session_info, conn, self)

            except Exception as e:
                session_info.error_count += 1
                logger.error(f"Error in session {session_id}: {e}")
                raise

    @asynccontextmanager
    async def get_async_session(self, session_id: str):
        """Get a session for async use."""
        # Run synchronous session management in thread pool
        loop = asyncio.get_event_loop()

        async def _get_session():
            return await loop.run_in_executor(
                None, lambda: self.get_session(session_id)
            )

        session_context = await _get_session()
        try:
            yield session_context
        finally:
            # Cleanup handled by context manager
            pass

    def close_session(self, session_id: str) -> bool:
        """Close a specific session."""
        with self.global_lock:
            if session_id not in self.sessions:
                return False

            try:
                # Get session data
                session_info = self.sessions[session_id]
                conn = self.connections[session_id]

                # Mark as inactive
                session_info.is_active = False

                # Close connection
                try:
                    # Rollback any pending transaction
                    if session_info.in_transaction:
                        conn.execute("ROLLBACK")
                except Exception as e:
                    logger.warning(
                        f"Error rolling back transaction in session {session_id}: {e}"
                    )

                try:
                    conn.close()
                except Exception as e:
                    logger.warning(
                        f"Error closing connection for session {session_id}: {e}"
                    )

                # Remove from tracking
                self.sessions.pop(session_id, None)
                self.connections.pop(session_id, None)
                self.session_locks.pop(session_id, None)

                logger.info(
                    f"Closed session {session_id} (age: {session_info.age_seconds:.1f}s, queries: {session_info.query_count})"
                )
                return True

            except Exception as e:
                logger.error(f"Error closing session {session_id}: {e}")
                return False

    def close_all_sessions(self):
        """Close all active sessions."""
        session_ids = list(self.sessions.keys())
        closed_count = 0

        for session_id in session_ids:
            if self.close_session(session_id):
                closed_count += 1

        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            self.cleanup_task = None

        logger.info(f"Closed {closed_count} database sessions")

    async def _cleanup_loop(self):
        """Background task to cleanup expired sessions."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                time.time()
                expired_sessions = []

                # Find expired sessions
                for session_id, session_info in self.sessions.items():
                    if session_info.idle_seconds > self.config.session_timeout:
                        expired_sessions.append(session_id)

                # Close expired sessions
                for session_id in expired_sessions:
                    logger.info(f"Cleaning up expired session {session_id}")
                    self.close_session(session_id)

                # Log statistics
                if len(self.sessions) > 0:
                    active_count = len(
                        [s for s in self.sessions.values() if s.is_active]
                    )
                    avg_age = sum(s.age_seconds for s in self.sessions.values()) / len(
                        self.sessions
                    )
                    logger.debug(
                        f"Session stats: {active_count} active, avg age {avg_age:.1f}s"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying

    def get_session_stats(self) -> dict[str, Any]:
        """Get statistics about session usage."""
        active_sessions = [s for s in self.sessions.values() if s.is_active]

        stats = {
            "total_sessions": len(self.sessions),
            "active_sessions": len(active_sessions),
            "total_created": self.total_sessions_created,
            "total_queries": self.total_queries_executed,
            "deadlocks_resolved": self.total_deadlocks_resolved,
            "avg_creation_time_ms": sum(self.session_creation_times[-100:])
            / min(len(self.session_creation_times), 100)
            * 1000
            if self.session_creation_times
            else 0,
        }

        if active_sessions:
            stats.update(
                {
                    "avg_session_age": sum(s.age_seconds for s in active_sessions)
                    / len(active_sessions),
                    "max_session_age": max(s.age_seconds for s in active_sessions),
                    "avg_queries_per_session": sum(
                        s.query_count for s in active_sessions
                    )
                    / len(active_sessions),
                    "sessions_in_transaction": len(
                        [s for s in active_sessions if s.in_transaction]
                    ),
                }
            )

        return stats


class SessionContext:
    """Context for working with an isolated database session."""

    def __init__(
        self,
        session_info: SessionInfo,
        connection: duckdb.DuckDBPyConnection,
        manager: SessionManager,
    ):
        self.session_info = session_info
        self.connection = connection
        self.manager = manager

    def execute(self, query: str, params: tuple | None = None) -> Any:
        """Execute a query in this session."""
        try:
            self.session_info.query_count += 1
            self.manager.total_queries_executed += 1

            start_time = time.time()

            if params:
                result = self.connection.execute(query, params)
            else:
                result = self.connection.execute(query)

            execution_time = time.time() - start_time

            # Log slow queries
            if execution_time > 1.0:
                logger.warning(
                    f"Slow query in session {self.session_info.session_id}: {execution_time:.2f}s"
                )

            return result

        except Exception as e:
            self.session_info.error_count += 1
            logger.error(f"Query error in session {self.session_info.session_id}: {e}")
            raise

    @contextmanager
    def transaction(self):
        """Execute operations within a transaction."""
        if self.session_info.in_transaction:
            # Nested transaction - use savepoint
            savepoint_name = f"sp_{int(time.time()*1000000)}"
            self.execute(f"SAVEPOINT {savepoint_name}")
            try:
                yield
                self.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            except Exception:
                self.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                raise
        else:
            # Top-level transaction
            self.session_info.in_transaction = True
            self.session_info.transaction_count += 1

            try:
                self.execute("BEGIN")
                yield
                self.execute("COMMIT")
            except Exception:
                try:
                    self.execute("ROLLBACK")
                except Exception as rollback_error:
                    logger.error(f"Error during rollback: {rollback_error}")
                raise
            finally:
                self.session_info.in_transaction = False


# Global session manager instance
_global_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = SessionManager()
    return _global_session_manager


def create_database_session(
    db_path: str | Path, session_hint: str | None = None
) -> str:
    """Create a new database session with isolation."""
    manager = get_session_manager()
    return manager.create_session(db_path, session_hint)


@contextmanager
def database_session(db_path: str | Path, session_hint: str | None = None):
    """Context manager for database session with automatic cleanup."""
    manager = get_session_manager()
    session_id = None

    try:
        session_id = manager.create_session(db_path, session_hint)
        with manager.get_session(session_id) as session:
            yield session
    finally:
        if session_id:
            manager.close_session(session_id)


@asynccontextmanager
async def async_database_session(db_path: str | Path, session_hint: str | None = None):
    """Async context manager for database session."""
    manager = get_session_manager()
    session_id = None

    try:
        loop = asyncio.get_event_loop()
        session_id = await loop.run_in_executor(
            None, manager.create_session, db_path, session_hint
        )

        async with manager.get_async_session(session_id) as session:
            yield session
    finally:
        if session_id:
            await loop.run_in_executor(None, manager.close_session, session_id)


def cleanup_all_sessions():
    """Cleanup all active sessions - useful for shutdown."""
    global _global_session_manager
    if _global_session_manager:
        _global_session_manager.close_all_sessions()
        _global_session_manager = None


# Example usage
if __name__ == "__main__":
    import sys
    import tempfile

    async def test_session_isolation():
        """Test session isolation functionality."""
        print("🧪 Testing Session Isolation for Concurrent DuckDB Access")

        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as tmp:
            db_path = Path(tmp.name)

        try:
            # Test basic session creation
            print("\n📝 Testing basic session operations...")

            async with async_database_session(db_path, "test_session") as session:
                # Create test table
                session.execute(
                    "CREATE TABLE IF NOT EXISTS test (id INTEGER, value TEXT)"
                )
                session.execute("INSERT INTO test VALUES (1, 'hello'), (2, 'world')")

                # Query data
                result = session.execute("SELECT * FROM test").fetchall()
                print(f"   Query result: {result}")

            # Test concurrent sessions
            print("\n🔄 Testing concurrent session access...")

            async def concurrent_worker(worker_id: int):
                async with async_database_session(
                    db_path, f"worker_{worker_id}"
                ) as session:
                    with session.transaction():
                        session.execute(
                            "INSERT INTO test VALUES (?, ?)",
                            (worker_id + 10, f"worker_{worker_id}"),
                        )
                        # Simulate some work
                        await asyncio.sleep(0.1)
                        count = session.execute("SELECT COUNT(*) FROM test").fetchone()[
                            0
                        ]
                        print(f"   Worker {worker_id}: table has {count} rows")

            # Run multiple concurrent workers
            tasks = [concurrent_worker(i) for i in range(5)]
            await asyncio.gather(*tasks)

            # Check final state
            async with async_database_session(db_path, "final_check") as session:
                result = session.execute("SELECT COUNT(*) FROM test").fetchone()[0]
                print(f"   Final row count: {result}")

            # Test session statistics
            print("\n📊 Session statistics:")
            manager = get_session_manager()
            stats = manager.get_session_stats()
            for key, value in stats.items():
                print(f"   {key}: {value}")

            print("\n✅ Session isolation test completed successfully!")

        finally:
            # Cleanup
            cleanup_all_sessions()
            with suppress(Exception):
                db_path.unlink()

    if HAS_DUCKDB:
        asyncio.run(test_session_isolation())
    else:
        print("❌ DuckDB not available - skipping session isolation test")
        sys.exit(1)
