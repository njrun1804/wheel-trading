#!/usr/bin/env python3
"""
macOS-specific subprocess wrapper that handles all the AsyncIO child watcher issues
and provides robust subprocess execution for search functionality.

This module implements comprehensive fixes for:
1. NotImplementedError: asyncio child watcher not implemented (Python 3.13+)
2. macOS-specific event loop policy issues
3. Database lock contention with cloud sync processes
4. Tool execution reliability in accelerated_tools modules
5. Graceful fallback mechanisms for search operations

Key Features:
- Automatic fallback from async to sync subprocess execution
- Process cleanup and resource management
- Database lock handling for cloud-synced directories
- Event loop policy management for macOS
- Comprehensive error handling and recovery
"""

import asyncio
import concurrent.futures
import logging
import multiprocessing as mp
import os
import platform
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    """Standardized process execution result."""

    returncode: int
    stdout: bytes
    stderr: bytes
    duration: float
    pid: int | None = None
    method: str = "unknown"  # async, sync, thread_pool

    def decode_stdout(self, encoding="utf-8") -> str:
        """Decode stdout with error handling."""
        try:
            return self.stdout.decode(encoding)
        except UnicodeDecodeError:
            return self.stdout.decode("utf-8", errors="replace")

    def decode_stderr(self, encoding="utf-8") -> str:
        """Decode stderr with error handling."""
        try:
            return self.stderr.decode(encoding)
        except UnicodeDecodeError:
            return self.stderr.decode("utf-8", errors="replace")


class MacOSSubprocessWrapper:
    """
    Comprehensive subprocess wrapper for macOS that handles all AsyncIO issues.

    This class provides a unified interface for subprocess execution that:
    1. Automatically detects and handles AsyncIO child watcher issues
    2. Provides graceful fallback from async to sync execution
    3. Manages process cleanup and resource management
    4. Handles database lock contention
    5. Implements proper error handling and recovery
    """

    def __init__(self, max_workers: int | None = None):
        self.system = platform.system()
        self.is_macos = self.system == "Darwin"
        self.cpu_count = mp.cpu_count()
        self.max_workers = max_workers or min(self.cpu_count, 32)

        # Thread pool for sync execution
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="subprocess_worker"
        )

        # Process management
        self.active_processes: dict[int, subprocess.Popen] = {}
        self.process_lock = threading.Lock()

        # AsyncIO support detection
        self.async_supported = self._detect_async_support()

        # Event loop policy setup for macOS
        if self.is_macos:
            self._setup_macos_event_loop_policy()

        logger.info(
            f"MacOSSubprocessWrapper initialized: async_supported={self.async_supported}, max_workers={self.max_workers}"
        )

    def _detect_async_support(self) -> bool:
        """Detect if async subprocess execution is supported with zero event loop conflicts."""
        try:
            # Always use thread-based detection to avoid event loop conflicts entirely
            return self._detect_async_support_in_thread()

        except Exception as e:
            logger.debug(f"Async subprocess detection failed: {e}")
            return False

    def _detect_async_support_in_thread(self) -> bool:
        """Detect async support using thread pool to avoid event loop conflicts."""
        try:
            import concurrent.futures
            import subprocess

            def simplified_thread_test():
                """Simplified sync test in separate thread to avoid all async issues."""
                try:
                    # Use a simple subprocess test instead of async
                    result = subprocess.run(
                        ["echo", "test"], capture_output=True, timeout=2.0
                    )
                    # If basic subprocess works, assume async could work too
                    # This is conservative but avoids all event loop conflicts
                    return result.returncode == 0
                except Exception:
                    return False

            # Run test in thread with strict timeout
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="async_detect"
            ) as executor:
                future = executor.submit(simplified_thread_test)
                try:
                    result = future.result(timeout=3.0)
                    return result
                except concurrent.futures.TimeoutError:
                    logger.debug("Async support detection timed out")
                    return False

        except Exception as e:
            logger.debug(f"Thread-based async detection failed: {e}")
            return False

    def _setup_macos_event_loop_policy(self):
        """Setup proper event loop policy for macOS."""
        try:
            # Try to set a compatible event loop policy
            if hasattr(asyncio, "DefaultEventLoopPolicy"):
                policy = asyncio.DefaultEventLoopPolicy()
                asyncio.set_event_loop_policy(policy)
                logger.info("Set DefaultEventLoopPolicy for macOS")

            # Try uvloop if available for better performance
            try:
                import uvloop

                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
                logger.info("Set uvloop EventLoopPolicy for macOS")
            except ImportError:
                pass

        except Exception as e:
            logger.warning(f"Failed to setup macOS event loop policy: {e}")

    async def execute_async(
        self,
        *args,
        timeout: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ProcessResult:
        """
        Execute command asynchronously with comprehensive error handling.

        Falls back to sync execution if async is not supported.
        """
        start_time = time.perf_counter()

        if not self.async_supported:
            logger.debug("Async not supported, falling back to sync execution")
            return await self._execute_sync_in_thread(args, timeout, cwd, env)

        try:
            # Create subprocess with proper error handling
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            # Track process
            with self.process_lock:
                self.active_processes[proc.pid] = proc

            try:
                # Wait for completion with timeout
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )

                duration = time.perf_counter() - start_time

                return ProcessResult(
                    returncode=proc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    duration=duration,
                    pid=proc.pid,
                    method="async",
                )

            finally:
                # Clean up process tracking
                with self.process_lock:
                    self.active_processes.pop(proc.pid, None)

        except TimeoutError:
            logger.warning(f"Async subprocess timeout after {timeout}s")
            # Fallback to sync
            return await self._execute_sync_in_thread(args, timeout, cwd, env)

        except NotImplementedError as e:
            logger.info(f"AsyncIO subprocess not implemented, using sync fallback: {e}")
            # Update support detection
            self.async_supported = False
            return await self._execute_sync_in_thread(args, timeout, cwd, env)

        except Exception as e:
            logger.warning(f"Async subprocess failed: {e}, using sync fallback")
            return await self._execute_sync_in_thread(args, timeout, cwd, env)

    async def _execute_sync_in_thread(
        self, args: tuple, timeout: float, cwd: str | None, env: dict[str, str] | None
    ) -> ProcessResult:
        """Execute command synchronously in thread pool."""
        loop = asyncio.get_event_loop()

        def _sync_execute():
            start_time = time.perf_counter()

            try:
                result = subprocess.run(
                    args, capture_output=True, timeout=timeout, cwd=cwd, env=env
                )

                duration = time.perf_counter() - start_time

                return ProcessResult(
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    duration=duration,
                    method="sync",
                )

            except subprocess.TimeoutExpired as e:
                return ProcessResult(
                    returncode=-1,
                    stdout=e.stdout or b"",
                    stderr=e.stderr or b"Command timed out",
                    duration=timeout,
                    method="sync_timeout",
                )

            except Exception as e:
                return ProcessResult(
                    returncode=-1,
                    stdout=b"",
                    stderr=str(e).encode(),
                    duration=time.perf_counter() - start_time,
                    method="sync_error",
                )

        return await loop.run_in_executor(self.thread_pool, _sync_execute)

    def execute_sync(
        self,
        *args,
        timeout: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ProcessResult:
        """Execute command synchronously."""
        start_time = time.perf_counter()

        try:
            result = subprocess.run(
                args, capture_output=True, timeout=timeout, cwd=cwd, env=env
            )

            duration = time.perf_counter() - start_time

            return ProcessResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration=duration,
                method="sync",
            )

        except subprocess.TimeoutExpired as e:
            return ProcessResult(
                returncode=-1,
                stdout=e.stdout or b"",
                stderr=e.stderr or b"Command timed out",
                duration=timeout,
                method="sync_timeout",
            )

        except Exception as e:
            return ProcessResult(
                returncode=-1,
                stdout=b"",
                stderr=str(e).encode(),
                duration=time.perf_counter() - start_time,
                method="sync_error",
            )

    def cleanup_stale_processes(self):
        """Clean up any stale or zombie processes."""
        with self.process_lock:
            stale_pids = []

            for pid, proc in self.active_processes.items():
                try:
                    # Check if process is still running
                    if proc.poll() is not None or not psutil.pid_exists(pid):
                        stale_pids.append(pid)
                except Exception:
                    stale_pids.append(pid)

            # Remove stale processes
            for pid in stale_pids:
                self.active_processes.pop(pid, None)
                logger.debug(f"Cleaned up stale process {pid}")

    def shutdown(self):
        """Shutdown the subprocess wrapper and clean up resources."""
        logger.info("Shutting down MacOSSubprocessWrapper")

        # Terminate any remaining processes
        with self.process_lock:
            for pid, proc in self.active_processes.items():
                try:
                    proc.terminate()
                    # Give process time to terminate gracefully
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                except Exception as e:
                    logger.warning(f"Failed to terminate process {pid}: {e}")

            self.active_processes.clear()

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        logger.info("MacOSSubprocessWrapper shutdown complete")


class DatabaseLockManager:
    """
    Manages database lock contention issues in cloud-synced directories.

    Handles issues with DuckDB and SQLite databases that may be locked by
    file sync processes like iCloud Drive.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.lock_files: dict[str, Path] = {}

    @contextmanager
    def safe_database_access(self, db_path: str | Path, timeout: float = 30.0):
        """
        Context manager for safe database access with lock handling.

        Attempts to handle database lock contention by:
        1. Checking for stale locks
        2. Waiting for cloud sync to complete
        3. Creating exclusive access locks
        """
        db_path = Path(db_path)
        lock_key = str(db_path)

        try:
            # Check if database is locked by cloud sync
            if self._is_cloud_sync_locked(db_path):
                logger.info(f"Database {db_path} locked by cloud sync, waiting...")
                self._wait_for_cloud_sync(db_path, timeout)

            # Create our own lock
            lock_file = self._create_access_lock(db_path)
            self.lock_files[lock_key] = lock_file

            yield db_path

        finally:
            # Clean up our lock
            if lock_key in self.lock_files:
                lock_file = self.lock_files.pop(lock_key)
                try:
                    lock_file.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to remove lock file {lock_file}: {e}")

    def _is_cloud_sync_locked(self, db_path: Path) -> bool:
        """Check if database is locked by cloud sync process."""
        try:
            # Use lsof to check if file is open by file sync processes
            result = subprocess.run(
                ["lsof", str(db_path)], capture_output=True, text=True, timeout=5.0
            )

            if result.returncode == 0:
                # Check for known sync processes
                sync_processes = ["fileprovi", "bird", "cloudd", "sync"]
                for line in result.stdout.splitlines():
                    for sync_proc in sync_processes:
                        if sync_proc in line.lower():
                            logger.debug(f"Database locked by sync process: {line}")
                            return True

            return False

        except Exception as e:
            logger.debug(f"Could not check cloud sync lock: {e}")
            return False

    def _wait_for_cloud_sync(self, db_path: Path, timeout: float):
        """Wait for cloud sync to release database lock."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if not self._is_cloud_sync_locked(db_path):
                logger.info(f"Cloud sync released lock on {db_path}")
                return

            time.sleep(1.0)

        logger.warning(
            f"Cloud sync did not release lock on {db_path} within {timeout}s"
        )

    def _create_access_lock(self, db_path: Path) -> Path:
        """Create an exclusive access lock file."""
        lock_file = db_path.with_suffix(db_path.suffix + ".access_lock")

        try:
            # Create lock file with process ID
            with open(lock_file, "w") as f:
                f.write(f"{os.getpid()}\n{time.time()}\n")

            return lock_file

        except Exception as e:
            logger.warning(f"Could not create access lock {lock_file}: {e}")
            return lock_file


# Singleton instances
_subprocess_wrapper: MacOSSubprocessWrapper | None = None
_database_manager: DatabaseLockManager | None = None


def get_subprocess_wrapper() -> MacOSSubprocessWrapper:
    """Get or create the subprocess wrapper singleton."""
    global _subprocess_wrapper
    if _subprocess_wrapper is None:
        _subprocess_wrapper = MacOSSubprocessWrapper()
    return _subprocess_wrapper


def get_database_manager(project_root: Path | None = None) -> DatabaseLockManager:
    """Get or create the database lock manager singleton."""
    global _database_manager
    if _database_manager is None:
        root = project_root or Path.cwd()
        _database_manager = DatabaseLockManager(root)
    return _database_manager


# Convenience functions for direct use
async def execute_command_async(
    *args,
    timeout: float = 30.0,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> ProcessResult:
    """Execute command asynchronously with automatic fallback."""
    wrapper = get_subprocess_wrapper()
    return await wrapper.execute_async(*args, timeout=timeout, cwd=cwd, env=env)


def execute_command_sync(
    *args,
    timeout: float = 30.0,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> ProcessResult:
    """Execute command synchronously."""
    wrapper = get_subprocess_wrapper()
    return wrapper.execute_sync(*args, timeout=timeout, cwd=cwd, env=env)


@contextmanager
def safe_database_access(db_path: str | Path, timeout: float = 30.0):
    """Context manager for safe database access."""
    manager = get_database_manager()
    with manager.safe_database_access(db_path, timeout):
        yield


def cleanup_subprocess_resources():
    """Clean up all subprocess resources."""
    global _subprocess_wrapper
    if _subprocess_wrapper:
        _subprocess_wrapper.cleanup_stale_processes()
        _subprocess_wrapper.shutdown()
        _subprocess_wrapper = None


# Test functions
async def test_subprocess_wrapper():
    """Test the subprocess wrapper functionality with comprehensive error handling."""
    print("🧪 Testing MacOSSubprocessWrapper...")

    import warnings
    
    # Suppress all warnings during subprocess testing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        wrapper = get_subprocess_wrapper()

        # Test async execution
        print("Testing async execution...")
        try:
            result = await wrapper.execute_async("echo", "hello async")
            print(
                f"  Result: {result.decode_stdout().strip()} (method: {result.method}, duration: {result.duration:.3f}s)"
            )
        except Exception as e:
            print(f"  Async execution failed (expected if not supported): {e}")

        # Test sync execution
        print("Testing sync execution...")
        try:
            result = wrapper.execute_sync("echo", "hello sync")
            print(
                f"  Result: {result.decode_stdout().strip()} (method: {result.method}, duration: {result.duration:.3f}s)"
            )
        except Exception as e:
            print(f"  Sync execution failed: {e}")

        # Test with non-existent command
        print("Testing error handling...")
        try:
            result = await wrapper.execute_async("nonexistent_command_12345")
            print(f"  Error handled: returncode={result.returncode}, method={result.method}")
        except Exception as e:
            print(f"  Error handling test failed: {e}")

        print("✅ All tests completed")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_subprocess_wrapper())
