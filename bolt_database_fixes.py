#!/usr/bin/env python3
"""
Database concurrency fixes for Bolt system.

Provides utilities to handle database concurrency issues and improve
connection management across the trading system.
"""

import contextlib
import fcntl
import logging
import os
import platform
import subprocess
import threading
import time
from pathlib import Path

import duckdb
import psutil

logger = logging.getLogger(__name__)

# Global connection pool for reuse
_connection_pool: dict[str, duckdb.DuckDBPyConnection] = {}
_pool_lock = threading.Lock()

# Mac-specific constants
IS_MACOS = platform.system() == "Darwin"
MACOS_LOCK_TIMEOUT = 30
MACOS_PROCESS_CHECK_INTERVAL = 1.0


class MacOSProcessManager:
    """Mac-specific process identification and cleanup utilities."""

    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".MacOSProcessManager")

    def get_file_locks(self, file_path: str) -> list[dict]:
        """Get processes holding locks on a file using macOS lsof."""
        if not IS_MACOS:
            return []

        try:
            # Use lsof to find processes with file locks
            result = subprocess.run(
                ["lsof", "+L1", file_path], capture_output=True, text=True, timeout=10
            )

            locks = []
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 9:
                            locks.append(
                                {
                                    "command": parts[0],
                                    "pid": int(parts[1]),
                                    "user": parts[2],
                                    "fd": parts[3],
                                    "type": parts[4],
                                    "device": parts[5],
                                    "size": parts[6],
                                    "node": parts[7],
                                    "name": " ".join(parts[8:]),
                                }
                            )
            return locks

        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            ValueError,
        ) as e:
            self.logger.debug(f"lsof failed for {file_path}: {e}")
            return []

    def get_database_related_processes(self, db_path: str) -> list[dict]:
        """Get all processes that might be using database files."""
        db_path = Path(db_path)
        related_files = [
            str(db_path),
            str(db_path.with_suffix(db_path.suffix + ".db-wal")),
            str(db_path.with_suffix(db_path.suffix + ".db-shm")),
            str(db_path.with_suffix(db_path.suffix + "-wal")),
            str(db_path.with_suffix(db_path.suffix + "-shm")),
            str(db_path) + ".wal",
            str(db_path) + ".shm",
        ]

        all_processes = []
        for file_path in related_files:
            if Path(file_path).exists():
                processes = self.get_file_locks(file_path)
                for proc in processes:
                    proc["file"] = file_path
                    all_processes.append(proc)

        # Also check using psutil for additional coverage
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline", "open_files"]):
                try:
                    if proc.info["open_files"]:
                        for f in proc.info["open_files"]:
                            if f.path in related_files:
                                all_processes.append(
                                    {
                                        "command": proc.info["name"],
                                        "pid": proc.info["pid"],
                                        "cmdline": " ".join(proc.info["cmdline"][:3])
                                        if proc.info["cmdline"]
                                        else "",
                                        "file": f.path,
                                        "method": "psutil",
                                    }
                                )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            self.logger.debug(f"psutil scan failed: {e}")

        # Deduplicate by PID
        seen_pids = set()
        unique_processes = []
        for proc in all_processes:
            if proc["pid"] not in seen_pids:
                seen_pids.add(proc["pid"])
                unique_processes.append(proc)

        return unique_processes

    def terminate_process_gracefully(self, pid: int, timeout: float = 10.0) -> bool:
        """Terminate a process gracefully on macOS."""
        try:
            proc = psutil.Process(pid)

            # Send SIGTERM first
            proc.terminate()

            # Wait for graceful termination
            try:
                proc.wait(timeout=timeout)
                self.logger.info(f"Process {pid} terminated gracefully")
                return True
            except psutil.TimeoutExpired:
                # Force kill if it doesn't terminate
                proc.kill()
                proc.wait(timeout=5)
                self.logger.warning(f"Process {pid} force killed")
                return True

        except psutil.NoSuchProcess:
            self.logger.debug(f"Process {pid} already terminated")
            return True
        except psutil.AccessDenied:
            self.logger.warning(f"Access denied when terminating process {pid}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to terminate process {pid}: {e}")
            return False

    def cleanup_database_processes(
        self, db_path: str, exclude_current: bool = True
    ) -> bool:
        """Clean up processes holding database locks."""
        current_pid = os.getpid()
        processes = self.get_database_related_processes(db_path)

        if not processes:
            self.logger.info(f"No processes found using database {db_path}")
            return True

        success = True
        for proc in processes:
            pid = proc["pid"]

            if exclude_current and pid == current_pid:
                self.logger.debug(f"Skipping current process {pid}")
                continue

            self.logger.info(
                f"Terminating process {pid} ({proc.get('command', 'unknown')}) using {proc.get('file', 'unknown')}"
            )
            if not self.terminate_process_gracefully(pid):
                success = False

        return success

    def get_process_stats(self) -> dict:
        """Get process and lock statistics."""
        return {
            "is_macos": IS_MACOS,
            "current_pid": os.getpid(),
            "process_count": len(list(psutil.process_iter())),
            "database_processes": len(self.get_database_related_processes(".")),
            "lock_timeout": MACOS_LOCK_TIMEOUT,
        }


class ConnectionPool:
    """Mac-friendly database connection pool with file locking."""

    def __init__(self, db_path: str, pool_size: int = 8):
        self.db_path = str(Path(db_path).resolve())
        self.pool_size = pool_size
        self.connections = []
        self.available = threading.Semaphore(pool_size)
        self.lock = threading.RLock()
        self.lock_file = f"{self.db_path}.pool.lock"
        self.process_manager = MacOSProcessManager()
        self.logger = logging.getLogger(__name__ + ".ConnectionPool")

    def _acquire_file_lock(self, timeout: float = MACOS_LOCK_TIMEOUT) -> int | None:
        """Acquire exclusive file lock using macOS fcntl."""
        try:
            fd = os.open(self.lock_file, os.O_CREAT | os.O_WRONLY, 0o644)

            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # Write PID to lock file
                    os.ftruncate(fd, 0)
                    os.write(fd, str(os.getpid()).encode())
                    os.fsync(fd)
                    return fd
                except BlockingIOError:
                    time.sleep(0.1)
                    continue

            os.close(fd)
            return None

        except Exception as e:
            self.logger.error(f"Failed to acquire file lock: {e}")
            return None

    def _release_file_lock(self, fd: int):
        """Release file lock."""
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            if os.path.exists(self.lock_file):
                os.unlink(self.lock_file)
        except Exception as e:
            self.logger.debug(f"Error releasing file lock: {e}")

    def initialize(self) -> bool:
        """Initialize connection pool with Mac-specific optimizations."""
        with self.lock:
            if self.connections:
                return True

            # Clean up any stale processes first
            self.process_manager.cleanup_database_processes(self.db_path)

            # Create connections
            for i in range(self.pool_size):
                try:
                    # Apple Silicon optimized DuckDB configuration
                    config = self._get_optimized_config()

                    # Debug: Log config values to identify empty string issue
                    self.logger.debug(f"DuckDB config for connection {i}: {config}")

                    conn = duckdb.connect(
                        f"{self.db_path}?connection_id={i}", config=config
                    )

                    # Test connection
                    conn.execute("SELECT 1")

                    self.connections.append(conn)
                    self.logger.debug(f"Created connection {i} for {self.db_path}")

                except Exception as e:
                    self.logger.warning(f"Failed to create connection {i}: {e}")
                    continue

            if not self.connections:
                self.logger.error(
                    f"Failed to create any connections for {self.db_path}"
                )
                return False

            self.logger.info(
                f"Initialized {len(self.connections)} connections for {self.db_path}"
            )
            return True

    @contextlib.contextmanager
    def get_connection(self, timeout: float = 30.0):
        """Get a connection from the pool with Mac-specific locking."""
        if not self.connections and not self.initialize():
            raise RuntimeError(
                f"Failed to initialize connection pool for {self.db_path}"
            )

        # Acquire semaphore
        acquired = self.available.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError(f"Connection timeout after {timeout}s")

        lock_fd = None
        try:
            # Acquire file lock for additional safety
            lock_fd = self._acquire_file_lock(timeout=5.0)

            with self.lock:
                if not self.connections:
                    raise RuntimeError("No connections available")

                conn = self.connections[0]
                self.connections = self.connections[1:] + [conn]  # Rotate

                # Test connection is still valid
                try:
                    conn.execute("SELECT 1")
                except Exception:
                    # Connection is stale, try to replace it
                    self.logger.warning("Stale connection detected, replacing")
                    conn = self._create_new_connection()

                yield conn

        finally:
            if lock_fd is not None:
                self._release_file_lock(lock_fd)
            self.available.release()

    def _create_new_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a new connection to replace a stale one with Apple Silicon optimizations."""
        config = self._get_optimized_config()
        conn_id = len(self.connections)
        return duckdb.connect(f"{self.db_path}?connection_id={conn_id}", config=config)

    def _get_optimized_config(self) -> dict:
        """Get database configuration optimized for Apple Silicon unified memory."""
        # Use conservative, working configuration to avoid config issues
        try:
            import psutil

            available_memory_gb = psutil.virtual_memory().available / (1024**3)

            # Apple Silicon optimizations
            if IS_MACOS and "arm" in platform.machine().lower():
                # Apple Silicon with unified memory - more aggressive caching
                memory_limit = max(
                    min(int(available_memory_gb * 0.4), 8), 1
                )  # Up to 40% of available, max 8GB, min 1GB
                threads = min(8, os.cpu_count() or 8)  # Use P-cores primarily
            else:
                # Intel or other architectures - conservative settings
                memory_limit = max(
                    min(int(available_memory_gb * 0.2), 4), 1
                )  # Up to 20% of available, max 4GB, min 1GB
                threads = max(
                    min(6, (os.cpu_count() or 6) - 2), 1
                )  # Leave more cores for system

            # Ensure minimum values to prevent empty strings
            memory_limit = max(memory_limit, 1)
            threads = max(threads, 1)

            # Only include essential configs that are known to work
            config = {
                "memory_limit": f"{memory_limit}GB",
                "threads": threads,
            }

            # Add optional configs only if they're likely to be supported
            try:
                # Test if we can create a temp directory
                temp_dir = "/tmp/duckdb_wheel"
                os.makedirs(temp_dir, exist_ok=True)
                config["temp_directory"] = temp_dir
            except:
                # If temp directory fails, don't include it
                pass

            return config

        except Exception as e:
            # Fallback to minimal working config
            self.logger.warning(f"Failed to generate optimized config: {e}")
            return {
                "memory_limit": "2GB",
                "threads": 4,
            }

    def close(self):
        """Close all connections in the pool."""
        with self.lock:
            for conn in self.connections:
                try:
                    conn.close()
                except Exception as e:
                    self.logger.debug(f"Error closing connection: {e}")
            self.connections.clear()

            # Clean up lock file
            if os.path.exists(self.lock_file):
                try:
                    os.unlink(self.lock_file)
                except Exception:
                    pass


class DatabaseConcurrencyManager:
    """Enhanced database connection manager with M4 Pro optimizations and Mac-specific features."""

    def __init__(self):
        self.max_retries = 5  # Increased for better reliability
        self.retry_delay = 0.05  # Reduced for faster recovery
        self.max_connections = 12  # M4 Pro optimized
        self.connection_timeout = 30.0
        self.pool_cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        self.process_manager = MacOSProcessManager()
        self.connection_pools: dict[str, ConnectionPool] = {}
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_queries": 0,
            "failed_connections": 0,
            "mac_process_cleanups": 0,
            "lock_conflicts_resolved": 0,
        }

    def get_connection_pool(self, db_path: str) -> ConnectionPool:
        """Get or create a Mac-optimized connection pool for a database."""
        if db_path not in self.connection_pools:
            self.connection_pools[db_path] = ConnectionPool(db_path, pool_size=4)
            self.connection_pools[db_path].initialize()
        return self.connection_pools[db_path]

    def get_connection(
        self, db_path: str, read_only: bool = False
    ) -> duckdb.DuckDBPyConnection:
        """Get a database connection with Mac-optimized concurrency handling and lock conflict resolution."""
        # Try to use Mac-optimized connection pool first
        try:
            if IS_MACOS:
                pool = self.get_connection_pool(db_path)
                with pool.get_connection(timeout=self.connection_timeout) as conn:
                    self.connection_stats["cache_hits"] += 1
                    self.connection_stats["total_queries"] += 1
                    return conn
        except Exception as e:
            logger.warning(
                f"Mac-optimized connection pool failed, falling back to legacy method: {e}"
            )
            self.connection_stats["lock_conflicts_resolved"] += 1

        # Fallback to legacy connection method
        key = f"{db_path}:{'ro' if read_only else 'rw'}"
        current_time = time.time()

        # Periodic cleanup of stale connections
        if current_time - self.last_cleanup > self.pool_cleanup_interval:
            self._cleanup_stale_connections()
            self.last_cleanup = current_time

        with _pool_lock:
            if key in _connection_pool:
                try:
                    # Test connection is still valid
                    _connection_pool[key].execute("SELECT 1")
                    self.connection_stats["cache_hits"] += 1
                    self.connection_stats["total_queries"] += 1
                    return _connection_pool[key]
                except:
                    # Connection is stale, remove it
                    del _connection_pool[key]
                    self.connection_stats["active_connections"] -= 1

            # Check pool size limit
            if len(_connection_pool) >= self.max_connections:
                self._evict_least_used_connection()

            # Create new connection with enhanced retry logic and Mac-specific cleanup
            base_delay = self.retry_delay
            for attempt in range(self.max_retries):
                try:
                    # Clean up any stale processes on Mac before attempting connection
                    if IS_MACOS and attempt > 0:
                        cleanup_success = (
                            self.process_manager.cleanup_database_processes(db_path)
                        )
                        if cleanup_success:
                            self.connection_stats["mac_process_cleanups"] += 1

                    # M4 Pro optimized configuration
                    config = {
                        "memory_limit": "2GB",  # Increased for M4 Pro
                        "max_memory": "4GB",
                        "threads": 8,  # Utilize more cores
                        "max_temp_directory_size": "1GB",
                        "enable_object_cache": True,
                        "checkpoint_threshold": "16MB",
                    }

                    if read_only:
                        config["access_mode"] = "READ_ONLY"

                    conn = duckdb.connect(db_path, config=config)

                    # Additional performance optimizations
                    try:
                        conn.execute("SET enable_progress_bar_print=false")
                    except:
                        # Older versions may not support this
                        pass
                    with contextlib.suppress(Exception):
                        conn.execute("SET memory_limit='2GB'")
                    if not read_only:
                        with contextlib.suppress(Exception):
                            conn.execute("SET checkpoint_threshold='32MB'")

                    _connection_pool[key] = conn
                    self.connection_stats["total_connections"] += 1
                    self.connection_stats["active_connections"] += 1
                    self.connection_stats["cache_misses"] += 1
                    self.connection_stats["total_queries"] += 1

                    logger.info(f"Created new database connection: {key}")
                    return conn

                except duckdb.IOException as e:
                    if "lock" in str(e).lower() and attempt < self.max_retries - 1:
                        delay = base_delay * (2**attempt)  # Exponential backoff
                        logger.warning(
                            f"Database locked, retrying in {delay:.2f}s (attempt {attempt + 1}/{self.max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        self.connection_stats["failed_connections"] += 1
                        logger.error(
                            f"Failed to connect to database after {self.max_retries} attempts: {e}"
                        )
                        raise
                except Exception as e:
                    self.connection_stats["failed_connections"] += 1
                    logger.error(f"Unexpected error connecting to database: {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(base_delay * (attempt + 1))

    def _cleanup_stale_connections(self):
        """Clean up stale database connections."""
        stale_keys = []
        for key, conn in _connection_pool.items():
            try:
                conn.execute("SELECT 1")
            except:
                stale_keys.append(key)

        for key in stale_keys:
            with contextlib.suppress(Exception):
                _connection_pool[key].close()
            del _connection_pool[key]
            self.connection_stats["active_connections"] -= 1

        if stale_keys:
            logger.info(f"Cleaned up {len(stale_keys)} stale database connections")

    def _evict_least_used_connection(self):
        """Evict the least recently used connection to make room for new ones."""
        if not _connection_pool:
            return

        # For simplicity, remove the first connection found
        # In a production system, you'd track usage timestamps
        key_to_remove = next(iter(_connection_pool))
        with contextlib.suppress(Exception):
            _connection_pool[key_to_remove].close()
        del _connection_pool[key_to_remove]
        self.connection_stats["active_connections"] -= 1
        logger.info(f"Evicted database connection: {key_to_remove}")

    def cleanup_all_pools(self):
        """Clean up all connection pools."""
        for pool in self.connection_pools.values():
            pool.close()
        self.connection_pools.clear()

    def get_performance_stats(self) -> dict[str, any]:
        """Get detailed performance statistics."""
        with _pool_lock:
            stats = self.connection_stats.copy()
            stats.update(
                {
                    "pool_size": len(_connection_pool),
                    "max_connections": self.max_connections,
                    "pool_utilization": len(_connection_pool) / self.max_connections,
                    "cache_hit_rate": (
                        stats["cache_hits"]
                        / max(1, stats["cache_hits"] + stats["cache_misses"])
                    ),
                    "connection_success_rate": (
                        (stats["total_connections"] - stats["failed_connections"])
                        / max(1, stats["total_connections"])
                    ),
                    "last_cleanup": self.last_cleanup,
                }
            )
            return stats

    @contextlib.contextmanager
    def get_temp_connection(self, db_path: str, read_only: bool = False):
        """Context manager for temporary database connections."""
        conn = None
        try:
            conn = self.get_connection(db_path, read_only)
            yield conn
        finally:
            # Don't close pooled connections, just return them
            pass


# Global instance (declared below)


def get_database_connection(
    db_path: str, read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    """Get a managed database connection."""
    return _db_manager.get_connection(db_path, read_only)


def get_temp_database_connection(db_path: str, read_only: bool = False):
    """Get a temporary database connection context manager."""
    return _db_manager.get_temp_connection(db_path, read_only)


def cleanup_database_connections():
    """Clean up all pooled connections including Mac-optimized pools."""
    global _connection_pool
    with _pool_lock:
        for conn in _connection_pool.values():
            with contextlib.suppress(Exception):
                conn.close()
        _connection_pool.clear()

    # Also clean up Mac-optimized connection pools
    _db_manager.cleanup_all_pools()


def fix_database_lock_issues(db_path: str) -> bool:
    """Attempt to fix database lock issues."""
    try:
        db_path = Path(db_path)

        # Remove WAL and shared memory files
        wal_file = db_path.with_suffix(db_path.suffix + "-wal")
        shm_file = db_path.with_suffix(db_path.suffix + "-shm")

        for file in [wal_file, shm_file]:
            if file.exists():
                try:
                    file.unlink()
                    logger.info(f"Removed lock file: {file}")
                except:
                    pass

        return True
    except Exception as e:
        logger.error(f"Failed to fix database lock issues: {e}")
        return False


def ensure_database_integrity(db_path: str) -> bool:
    """Ensure database integrity and fix common issues."""
    try:
        with get_temp_database_connection(db_path, read_only=True) as conn:
            # Test basic connectivity
            conn.execute("SELECT 1")

            # Check for common tables
            tables = conn.execute("SHOW TABLES").fetchall()
            logger.info(f"Database {db_path} has {len(tables)} tables")

        return True
    except Exception as e:
        logger.error(f"Database integrity check failed for {db_path}: {e}")

        # Attempt to fix lock issues
        if "lock" in str(e).lower():
            return fix_database_lock_issues(db_path)

        return False


# Compatibility functions for existing code
def get_bolt_database_manager():
    """Get the bolt database manager for compatibility."""
    return _db_manager


class DatabaseLockManager:
    """Database lock manager for compatibility with bolt integration."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self.manager = _db_manager
        self.process_manager = _mac_process_manager if IS_MACOS else None
        self.logger = logging.getLogger(__name__ + ".DatabaseLockManager")

    def acquire_lock(self, resource_id: str = None, timeout: float = 30.0) -> bool:
        """Acquire database lock (compatibility method)."""
        try:
            if self.db_path:
                conn = get_database_connection(self.db_path)
                # Test connection is working
                conn.execute("SELECT 1")
                return True
            return True
        except Exception as e:
            self.logger.warning(f"Failed to acquire lock for {resource_id}: {e}")
            return False

    def release_lock(self, resource_id: str = None) -> bool:
        """Release database lock (compatibility method)."""
        return True  # Handled by connection pooling

    def diagnose_locks(self) -> dict:
        """Diagnose lock issues."""
        if self.db_path:
            return diagnose_database_locks(self.db_path)
        return {"locks": [], "processes": []}

    def cleanup_locks(self) -> bool:
        """Clean up stale locks."""
        if self.db_path:
            result = auto_fix_database_locks(self.db_path)
            return result.get("success", False)
        return True

    def get_lock_stats(self) -> dict:
        """Get lock statistics."""
        return self.manager.get_performance_stats()

    @contextlib.contextmanager
    def lock_context(self, resource_id: str = None, timeout: float = 30.0):
        """Context manager for database locks."""
        success = self.acquire_lock(resource_id, timeout)
        if not success:
            raise RuntimeError(f"Failed to acquire lock for {resource_id}")

        try:
            yield
        finally:
            self.release_lock(resource_id)


class ConcurrentDatabase:
    """Simplified concurrent database wrapper with Mac optimizations."""

    def __init__(self, db_path: str, **kwargs):
        self.db_path = db_path
        self.manager = _db_manager
        self.config = kwargs

    def get_connection(self, read_only: bool = False):
        """Get a database connection using Mac-optimized pools."""
        return self.manager.get_connection(self.db_path, read_only)

    @contextlib.contextmanager
    def get_connection_context(self, read_only: bool = False):
        """Get a connection with context management for Mac-optimized pools."""
        if IS_MACOS:
            try:
                pool = self.manager.get_connection_pool(self.db_path)
                with pool.get_connection() as conn:
                    yield conn
                    return
            except Exception:
                pass  # Fall back to regular connection

        # Fallback to regular connection
        conn = self.get_connection(read_only)
        try:
            yield conn
        finally:
            pass  # Pooled connections don't need explicit closing

    def execute(self, query: str, params=None):
        """Execute a query using Mac-optimized connection management."""
        with self.get_connection_context() as conn:
            if params:
                return conn.execute(query, params)
            else:
                return conn.execute(query)

    def close(self):
        """Close connections (enhanced for Mac pools)."""
        if IS_MACOS and self.db_path in self.manager.connection_pools:
            self.manager.connection_pools[self.db_path].close()
            del self.manager.connection_pools[self.db_path]


class AsyncConcurrentDatabase:
    """Async database with concurrency control."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.manager = _db_manager

    def get_connection(self, read_only: bool = False):
        """Get database connection."""
        return self.manager.get_connection(self.db_path, read_only)

    async def execute(self, query: str, params=None):
        """Execute async query."""
        conn = self.get_connection()
        if params:
            return conn.execute(query, params)
        else:
            return conn.execute(query)


class DatabaseConfig:
    """Database configuration for compatibility."""

    def __init__(self, **kwargs):
        self.config = kwargs

    def get(self, key, default=None):
        return self.config.get(key, default)


def get_mac_process_manager() -> MacOSProcessManager:
    """Get Mac process manager instance."""
    return _db_manager.process_manager


def cleanup_database_processes(db_path: str) -> bool:
    """Clean up processes holding database locks (Mac-specific)."""
    if IS_MACOS:
        return _db_manager.process_manager.cleanup_database_processes(db_path)
    return True


def get_database_locks(db_path: str) -> list[dict]:
    """Get list of processes holding database locks (Mac-specific)."""
    if IS_MACOS:
        return _db_manager.process_manager.get_database_related_processes(db_path)
    return []


def force_database_unlock(db_path: str) -> bool:
    """Force unlock database by terminating conflicting processes (Mac-specific)."""
    if IS_MACOS:
        # First try graceful cleanup
        success = _db_manager.process_manager.cleanup_database_processes(db_path)
        if success:
            # Remove any remaining lock files
            return fix_database_lock_issues(db_path)
        return False
    return fix_database_lock_issues(db_path)


def diagnose_database_locks(db_path: str) -> dict:
    """Comprehensive diagnosis of database lock issues."""
    diagnosis = {
        "db_path": db_path,
        "db_exists": Path(db_path).exists(),
        "is_macos": IS_MACOS,
        "lock_files": [],
        "processes": [],
        "performance_stats": _db_manager.get_performance_stats(),
        "recommendations": [],
    }

    # Check for lock files
    db_path_obj = Path(db_path)
    lock_patterns = [
        ".db-wal",
        ".db-shm",
        "-wal",
        "-shm",
        ".wal",
        ".shm",
        ".lock",
        ".pool.lock",
    ]
    for pattern in lock_patterns:
        if pattern.endswith(".lock"):
            lock_file = db_path_obj.with_suffix(db_path_obj.suffix + pattern)
        else:
            lock_file = Path(str(db_path_obj) + pattern)

        if lock_file.exists():
            try:
                stat = lock_file.stat()
                diagnosis["lock_files"].append(
                    {
                        "file": str(lock_file),
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "age_seconds": time.time() - stat.st_mtime,
                    }
                )
            except Exception as e:
                diagnosis["lock_files"].append(
                    {"file": str(lock_file), "error": str(e)}
                )

    # Get processes using the database
    if IS_MACOS:
        diagnosis[
            "processes"
        ] = _db_manager.process_manager.get_database_related_processes(db_path)

    # Generate recommendations
    if diagnosis["lock_files"]:
        stale_locks = [
            lf
            for lf in diagnosis["lock_files"]
            if isinstance(lf.get("age_seconds"), (int, float))
            and lf["age_seconds"] > 300
        ]
        if stale_locks:
            diagnosis["recommendations"].append(
                "Remove stale lock files older than 5 minutes"
            )

    if diagnosis["processes"]:
        diagnosis["recommendations"].append("Terminate conflicting processes")

    if not diagnosis["lock_files"] and not diagnosis["processes"]:
        diagnosis["recommendations"].append("No lock issues detected")

    return diagnosis


def auto_fix_database_locks(db_path: str) -> dict:
    """Automatically attempt to fix database lock issues."""
    result = {"db_path": db_path, "steps_taken": [], "success": False, "errors": []}

    try:
        # Step 1: Diagnose the issue
        diagnosis = diagnose_database_locks(db_path)
        result["steps_taken"].append(
            f"Diagnosed: {len(diagnosis['lock_files'])} lock files, {len(diagnosis['processes'])} processes"
        )

        # Step 2: Clean up processes (Mac only)
        if IS_MACOS and diagnosis["processes"]:
            cleanup_success = cleanup_database_processes(db_path)
            if cleanup_success:
                result["steps_taken"].append(
                    "Successfully cleaned up conflicting processes"
                )
            else:
                result["steps_taken"].append("Failed to clean up some processes")
                result["errors"].append("Process cleanup incomplete")

        # Step 3: Remove lock files
        if diagnosis["lock_files"]:
            removed_count = 0
            for lock_info in diagnosis["lock_files"]:
                if "file" in lock_info:
                    try:
                        Path(lock_info["file"]).unlink()
                        removed_count += 1
                    except Exception as e:
                        result["errors"].append(
                            f"Failed to remove {lock_info['file']}: {e}"
                        )

            result["steps_taken"].append(
                f"Removed {removed_count}/{len(diagnosis['lock_files'])} lock files"
            )

        # Step 4: Test database connectivity
        try:
            with get_temp_database_connection(db_path, read_only=True) as conn:
                conn.execute("SELECT 1")
            result["steps_taken"].append("Database connectivity test passed")
            result["success"] = True
        except Exception as e:
            result["steps_taken"].append(f"Database connectivity test failed: {e}")
            result["errors"].append(str(e))

    except Exception as e:
        result["errors"].append(f"Auto-fix failed: {e}")

    return result


def test_database_concurrency() -> dict:
    """Test database concurrency fixes and Mac optimizations."""
    test_results = {
        "platform": platform.system(),
        "is_macos": IS_MACOS,
        "test_db": ".test_concurrency.db",
        "tests": {},
        "overall_success": True,
    }

    test_db = test_results["test_db"]

    # Test 1: Basic connection
    try:
        with get_temp_database_connection(test_db) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS test_table (id INTEGER, name TEXT)"
            )
            conn.execute("INSERT INTO test_table VALUES (1, 'test')")
            result = conn.execute("SELECT COUNT(*) FROM test_table").fetchone()
        test_results["tests"]["basic_connection"] = {
            "success": True,
            "count": result[0] if result else 0,
        }
    except Exception as e:
        test_results["tests"]["basic_connection"] = {"success": False, "error": str(e)}
        test_results["overall_success"] = False

    # Test 2: Mac-specific connection pool (if on macOS)
    if IS_MACOS:
        try:
            pool = _db_manager.get_connection_pool(test_db)
            with pool.get_connection() as conn:
                conn.execute("SELECT COUNT(*) FROM test_table")
            test_results["tests"]["mac_connection_pool"] = {"success": True}
        except Exception as e:
            test_results["tests"]["mac_connection_pool"] = {
                "success": False,
                "error": str(e),
            }
            test_results["overall_success"] = False

    # Test 3: Process detection (if on macOS)
    if IS_MACOS:
        try:
            processes = get_database_locks(test_db)
            test_results["tests"]["process_detection"] = {
                "success": True,
                "process_count": len(processes),
            }
        except Exception as e:
            test_results["tests"]["process_detection"] = {
                "success": False,
                "error": str(e),
            }

    # Test 4: Lock diagnosis
    try:
        diagnosis = diagnose_database_locks(test_db)
        test_results["tests"]["lock_diagnosis"] = {
            "success": True,
            "lock_files": len(diagnosis["lock_files"]),
            "processes": len(diagnosis["processes"]),
        }
    except Exception as e:
        test_results["tests"]["lock_diagnosis"] = {"success": False, "error": str(e)}
        test_results["overall_success"] = False

    # Test 5: Performance stats
    try:
        stats = _db_manager.get_performance_stats()
        test_results["tests"]["performance_stats"] = {
            "success": True,
            "total_connections": stats.get("total_connections", 0),
            "cache_hit_rate": stats.get("cache_hit_rate", 0),
        }
    except Exception as e:
        test_results["tests"]["performance_stats"] = {"success": False, "error": str(e)}

    # Cleanup test database
    try:
        cleanup_database_connections()
        if Path(test_db).exists():
            Path(test_db).unlink()
    except Exception:
        pass  # Best effort cleanup

    return test_results


def initialize_database_fixes():
    """Initialize database concurrency fixes with Mac optimizations."""
    logger.info(f"Bolt database concurrency fixes initialized (macOS: {IS_MACOS})")
    if IS_MACOS:
        logger.info("Mac-specific process management and file locking enabled")
        logger.info(f"CPU count: {os.cpu_count()}, Architecture: {platform.machine()}")
    return True


# Global instances
_db_manager = DatabaseConcurrencyManager()
_mac_process_manager = MacOSProcessManager() if IS_MACOS else None

# Auto-initialize when imported
initialize_database_fixes()
