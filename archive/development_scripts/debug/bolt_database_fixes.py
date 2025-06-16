#!/usr/bin/env python3
"""
Database Concurrency Fixes - Solutions for bolt's database locking issues.

This module provides robust database access patterns that solve the critical
database concurrency issues preventing multiple bolt sessions and Einstein
integration from working simultaneously.

Key Problems Solved:
- "Could not set lock on analytics.db: Conflicting lock is held in PID..."
- Single-session limitation preventing concurrent usage
- Database connection pooling failures
- Analytics database unavailable during Einstein sessions
"""

import asyncio
import fcntl
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration with concurrency settings."""

    path: Path
    max_connections: int = 10
    connection_timeout: float = 30.0
    lock_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_wal_mode: bool = True
    enable_connection_pooling: bool = True


@dataclass
class ConnectionInfo:
    """Information about database connection."""

    connection: sqlite3.Connection
    thread_id: int
    created_at: float
    last_used: float
    in_use: bool = False


class DatabaseLockManager:
    """Advanced file-based locking system for database concurrency."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.lock_dir = db_path.parent / ".db_locks"
        self.lock_dir.mkdir(exist_ok=True)
        self.lock_file = self.lock_dir / f"{db_path.name}.lock"
        self.process_info_file = self.lock_dir / f"{db_path.name}.info"

    @contextmanager
    def acquire_lock(self, timeout: float = 30.0, lock_type: str = "shared"):
        """Acquire database lock with advanced coordination."""
        lock_acquired = False
        lock_fd = None

        try:
            # Create lock file if it doesn't exist
            self.lock_file.touch()

            # Open lock file
            lock_fd = open(self.lock_file, "w")

            # Determine lock type
            if lock_type == "exclusive":
                lock_flags = fcntl.LOCK_EX
            else:
                lock_flags = fcntl.LOCK_SH

            # Try to acquire lock with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    fcntl.flock(lock_fd.fileno(), lock_flags | fcntl.LOCK_NB)
                    lock_acquired = True
                    break
                except OSError:
                    # Check if blocking process still exists
                    if self._cleanup_stale_locks():
                        continue
                    time.sleep(0.1)

            if not lock_acquired:
                raise TimeoutError(
                    f"Could not acquire {lock_type} lock within {timeout}s"
                )

            # Write process information
            self._write_process_info(lock_type)

            yield

        finally:
            if lock_acquired and lock_fd:
                try:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                    self._cleanup_process_info()
                except Exception as e:
                    logger.warning(f"Failed to release lock: {e}")

            if lock_fd:
                with suppress(Exception):
                    lock_fd.close()

    def _write_process_info(self, lock_type: str):
        """Write process information for lock debugging."""
        try:
            info = {
                "pid": os.getpid(),
                "lock_type": lock_type,
                "timestamp": time.time(),
                "command_line": " ".join(psutil.Process().cmdline()),
            }

            with open(self.process_info_file, "w") as f:
                import json

                json.dump(info, f)

        except Exception as e:
            logger.warning(f"Failed to write process info: {e}")

    def _cleanup_process_info(self):
        """Clean up process information file."""
        with suppress(Exception):
            self.process_info_file.unlink(missing_ok=True)

    def _cleanup_stale_locks(self) -> bool:
        """Clean up stale locks from dead processes."""
        try:
            if not self.process_info_file.exists():
                return False

            with open(self.process_info_file) as f:
                import json

                info = json.load(f)

            pid = info.get("pid")
            if pid and not psutil.pid_exists(pid):
                logger.info(f"Cleaning up stale lock from dead process {pid}")
                self.process_info_file.unlink(missing_ok=True)
                return True

        except Exception as e:
            logger.warning(f"Failed to cleanup stale locks: {e}")

        return False

    def get_lock_info(self) -> dict[str, Any] | None:
        """Get information about current lock holder."""
        try:
            if self.process_info_file.exists():
                with open(self.process_info_file) as f:
                    import json

                    return json.load(f)
        except Exception:
            pass
        return None


class ConnectionPool:
    """Thread-safe connection pool with advanced management."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connections: dict[int, ConnectionInfo] = {}
        self.lock = threading.RLock()
        self.total_connections = 0
        self.lock_manager = DatabaseLockManager(config.path)

    @contextmanager
    def get_connection(self, lock_type: str = "shared"):
        """Get connection from pool with proper locking."""
        thread_id = threading.get_ident()
        connection_info = None

        try:
            # Acquire database-level lock first
            with self.lock_manager.acquire_lock(self.config.lock_timeout, lock_type):
                # Get or create connection
                with self.lock:
                    connection_info = self._get_or_create_connection(thread_id)
                    connection_info.in_use = True
                    connection_info.last_used = time.time()

                yield connection_info.connection

        finally:
            if connection_info:
                with self.lock:
                    connection_info.in_use = False

    def _get_or_create_connection(self, thread_id: int) -> ConnectionInfo:
        """Get existing connection or create new one."""
        # Check if thread already has a connection
        if thread_id in self.connections:
            conn_info = self.connections[thread_id]
            if self._is_connection_healthy(conn_info.connection):
                return conn_info
            else:
                # Close and remove unhealthy connection
                self._close_connection(thread_id)

        # Create new connection
        if self.total_connections >= self.config.max_connections:
            self._cleanup_idle_connections()

        if self.total_connections >= self.config.max_connections:
            # Find and close least recently used connection
            lru_thread = min(
                self.connections.keys(), key=lambda tid: self.connections[tid].last_used
            )
            self._close_connection(lru_thread)

        # Create new connection with proper settings
        conn = self._create_connection()
        conn_info = ConnectionInfo(
            connection=conn,
            thread_id=thread_id,
            created_at=time.time(),
            last_used=time.time(),
        )

        self.connections[thread_id] = conn_info
        self.total_connections += 1

        return conn_info

    def _create_connection(self) -> sqlite3.Connection:
        """Create new database connection with optimized settings."""
        conn = sqlite3.connect(
            str(self.config.path),
            timeout=self.config.connection_timeout,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode
        )

        # Set row factory for dict-like access
        conn.row_factory = sqlite3.Row

        # Optimize connection settings
        if self.config.enable_wal_mode:
            conn.execute("PRAGMA journal_mode=WAL")

        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB

        return conn

    def _is_connection_healthy(self, conn: sqlite3.Connection) -> bool:
        """Check if connection is healthy."""
        try:
            conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    def _cleanup_idle_connections(self):
        """Close idle connections to free up pool space."""
        current_time = time.time()
        idle_timeout = 300  # 5 minutes

        idle_threads = [
            thread_id
            for thread_id, conn_info in self.connections.items()
            if not conn_info.in_use
            and current_time - conn_info.last_used > idle_timeout
        ]

        for thread_id in idle_threads:
            self._close_connection(thread_id)

    def _close_connection(self, thread_id: int):
        """Close and remove connection."""
        if thread_id in self.connections:
            with suppress(Exception):
                self.connections[thread_id].connection.close()

            del self.connections[thread_id]
            self.total_connections -= 1

    def close_all(self):
        """Close all connections in pool."""
        with self.lock:
            for thread_id in list(self.connections.keys()):
                self._close_connection(thread_id)


class ConcurrentDatabase:
    """High-level database interface with full concurrency support."""

    def __init__(self, db_path: str, **config_kwargs):
        self.config = DatabaseConfig(path=Path(db_path), **config_kwargs)
        self.pool = (
            ConnectionPool(self.config)
            if self.config.enable_connection_pooling
            else None
        )
        self.lock_manager = DatabaseLockManager(self.config.path)

    @contextmanager
    def connection(self, lock_type: str = "shared"):
        """Get database connection with proper concurrency control."""
        if self.pool:
            with self.pool.get_connection(lock_type) as conn:
                yield conn
        else:
            # Direct connection without pooling
            with self.lock_manager.acquire_lock(self.config.lock_timeout, lock_type):
                conn = sqlite3.connect(
                    str(self.config.path), timeout=self.config.connection_timeout
                )
                conn.row_factory = sqlite3.Row
                try:
                    yield conn
                finally:
                    conn.close()

    def query(
        self, sql: str, params: tuple | None = None, lock_type: str = "shared"
    ) -> list[dict[str, Any]]:
        """Execute query with automatic retry and error handling."""
        last_error = None

        for attempt in range(self.config.retry_attempts):
            try:
                with self.connection(lock_type) as conn:
                    cursor = conn.cursor()

                    if params:
                        cursor.execute(sql, params)
                    else:
                        cursor.execute(sql)

                    if sql.strip().upper().startswith("SELECT"):
                        return [dict(row) for row in cursor.fetchall()]
                    else:
                        conn.commit()
                        return []

            except sqlite3.OperationalError as e:
                last_error = e
                if "database is locked" in str(e).lower():
                    logger.warning(
                        f"Database locked, attempt {attempt + 1}/{self.config.retry_attempts}"
                    )
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                else:
                    raise
            except Exception as e:
                last_error = e
                logger.error(f"Database query failed: {e}")
                break

        if last_error:
            raise last_error

    def execute_transaction(
        self, statements: list[tuple[str, tuple | None]], lock_type: str = "exclusive"
    ) -> bool:
        """Execute multiple statements in a transaction."""
        try:
            with self.connection(lock_type) as conn:
                conn.execute("BEGIN IMMEDIATE")

                try:
                    for sql, params in statements:
                        if params:
                            conn.execute(sql, params)
                        else:
                            conn.execute(sql)

                    conn.commit()
                    return True

                except Exception:
                    conn.rollback()
                    raise

        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            return False

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        try:
            result = self.query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            return len(result) > 0
        except Exception:
            return False

    def get_schema_info(self) -> dict[str, list[dict[str, Any]]]:
        """Get database schema information."""
        try:
            tables = self.query("SELECT name FROM sqlite_master WHERE type='table'")
            schema = {}

            for table in tables:
                table_name = table["name"]
                columns = self.query(f"PRAGMA table_info({table_name})")
                schema[table_name] = [dict(col) for col in columns]

            return schema
        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            return {}

    def get_lock_info(self) -> dict[str, Any] | None:
        """Get current lock information."""
        return self.lock_manager.get_lock_info()

    def force_unlock(self) -> bool:
        """Force unlock database (use with caution)."""
        try:
            lock_info = self.get_lock_info()
            if lock_info:
                pid = lock_info.get("pid")
                if pid and not psutil.pid_exists(pid):
                    self.lock_manager._cleanup_process_info()
                    logger.info(f"Removed stale lock from dead process {pid}")
                    return True
                else:
                    logger.warning(f"Lock held by active process {pid}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Failed to force unlock: {e}")
            return False

    def close(self):
        """Close database and cleanup resources."""
        if self.pool:
            self.pool.close_all()


class AsyncConcurrentDatabase:
    """Async wrapper for concurrent database operations."""

    def __init__(self, db_path: str, **config_kwargs):
        self.sync_db = ConcurrentDatabase(db_path, **config_kwargs)
        self.executor = None

    async def query(
        self, sql: str, params: tuple | None = None
    ) -> list[dict[str, Any]]:
        """Async query execution."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.sync_db.query, sql, params
        )

    async def execute_transaction(
        self, statements: list[tuple[str, tuple | None]]
    ) -> bool:
        """Async transaction execution."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.sync_db.execute_transaction, statements
        )

    async def close(self):
        """Close async database."""
        self.sync_db.close()


def create_database_manager(db_path: str, **kwargs) -> ConcurrentDatabase:
    """Factory function to create database manager with concurrency support."""
    return ConcurrentDatabase(db_path, **kwargs)


def fix_existing_database_locks(project_root: str = ".") -> dict[str, bool]:
    """Fix existing database lock issues in the project."""
    project_path = Path(project_root)
    results = {}

    # Find database files
    db_patterns = ["*.db", "*.duckdb", "*.sqlite", "*.sqlite3"]
    db_files = []

    for pattern in db_patterns:
        db_files.extend(project_path.rglob(pattern))

    for db_file in db_files:
        try:
            db_manager = ConcurrentDatabase(str(db_file))

            # Check if database is accessible
            try:
                db_manager.query("SELECT 1")
                results[str(db_file)] = True
            except Exception as e:
                logger.warning(f"Database {db_file} has issues: {e}")

                # Try to force unlock
                if db_manager.force_unlock():
                    # Test again
                    try:
                        db_manager.query("SELECT 1")
                        results[str(db_file)] = True
                        logger.info(f"Successfully fixed {db_file}")
                    except Exception:
                        results[str(db_file)] = False
                else:
                    results[str(db_file)] = False

            db_manager.close()

        except Exception as e:
            logger.error(f"Failed to check {db_file}: {e}")
            results[str(db_file)] = False

    return results


# CLI for testing and fixing database issues
if __name__ == "__main__":
    import sys

    def main():
        if len(sys.argv) < 2:
            print("Database Concurrency Fixes")
            print("Usage:")
            print(
                "  python bolt_database_fixes.py test <db_path>     # Test database access"
            )
            print(
                "  python bolt_database_fixes.py fix <project_root> # Fix all database locks"
            )
            print("  python bolt_database_fixes.py info <db_path>     # Show lock info")
            return

        command = sys.argv[1]

        if command == "test" and len(sys.argv) > 2:
            db_path = sys.argv[2]
            print(f"Testing database access: {db_path}")

            try:
                db = ConcurrentDatabase(db_path)

                # Test basic query
                result = db.query("SELECT 1 as test")
                print(f"✅ Basic query successful: {result}")

                # Test schema info
                schema = db.get_schema_info()
                print(f"✅ Schema info: {len(schema)} tables")

                # Test concurrent access
                import threading

                def concurrent_test():
                    try:
                        db.query("SELECT 1")
                        print("✅ Concurrent access successful")
                    except Exception as e:
                        print(f"❌ Concurrent access failed: {e}")

                threads = [threading.Thread(target=concurrent_test) for _ in range(5)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                db.close()

            except Exception as e:
                print(f"❌ Database test failed: {e}")

        elif command == "fix":
            project_root = sys.argv[2] if len(sys.argv) > 2 else "."
            print(f"Fixing database locks in: {project_root}")

            results = fix_existing_database_locks(project_root)

            print("\nResults:")
            for db_path, success in results.items():
                status = "✅ Fixed" if success else "❌ Failed"
                print(f"  {status} {db_path}")

        elif command == "info" and len(sys.argv) > 2:
            db_path = sys.argv[2]
            print(f"Database lock info: {db_path}")

            try:
                db = ConcurrentDatabase(db_path)
                lock_info = db.get_lock_info()

                if lock_info:
                    print(f"Lock held by PID: {lock_info.get('pid')}")
                    print(f"Lock type: {lock_info.get('lock_type')}")
                    print(f"Timestamp: {lock_info.get('timestamp')}")
                    print(f"Command: {lock_info.get('command_line')}")
                else:
                    print("✅ No active locks")

                db.close()

            except Exception as e:
                print(f"❌ Failed to get lock info: {e}")

        else:
            print("Invalid command. Use 'test', 'fix', or 'info'")

    main()
