#!/usr/bin/env python3
"""
Database Lock Manager for Bolt System
Handles database connection conflicts and lock management.
"""

import contextlib
import logging
import os
import time
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)


class DatabaseLockManager:
    """Manages database locks and connection conflicts."""

    def __init__(self):
        self.active_locks: dict[str, int] = {}
        self.lock_files: dict[str, str] = {}

    def cleanup_stale_locks(self, db_path: str) -> bool:
        """Clean up stale database locks."""
        try:
            db_path = Path(db_path)

            # Find related lock files
            lock_files = []
            for suffix in [".db-wal", ".db-shm", ".db-lock"]:
                lock_file = db_path.with_suffix(db_path.suffix + suffix)
                if lock_file.exists():
                    lock_files.append(lock_file)

            # Check for processes using these files
            stale_locks = []
            for lock_file in lock_files:
                if self._is_lock_stale(str(lock_file)):
                    stale_locks.append(lock_file)

            # Remove stale locks
            for lock_file in stale_locks:
                try:
                    lock_file.unlink()
                    logger.info(f"Removed stale lock file: {lock_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove lock file {lock_file}: {e}")

            return len(stale_locks) > 0

        except Exception as e:
            logger.error(f"Failed to cleanup stale locks for {db_path}: {e}")
            return False

    def _is_lock_stale(self, lock_file: str) -> bool:
        """Check if a lock file is stale."""
        try:
            # Check if any process is using this file
            for proc in psutil.process_iter(["pid", "name", "open_files"]):
                try:
                    if proc.info["open_files"]:
                        for f in proc.info["open_files"]:
                            if f.path == lock_file:
                                return False  # File is in use
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Check file age - if older than 5 minutes, likely stale
            stat = os.stat(lock_file)
            age = time.time() - stat.st_mtime
            if age > 300:  # 5 minutes
                return True

            return False

        except Exception as e:
            logger.debug(f"Error checking lock file {lock_file}: {e}")
            return False

    def acquire_exclusive_lock(self, db_path: str, timeout: int = 30) -> bool:
        """Acquire exclusive lock on database."""
        try:
            lock_file = f"{db_path}.lock"

            # Try to acquire lock with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    # Create lock file
                    fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, str(os.getpid()).encode())
                    os.close(fd)

                    self.active_locks[db_path] = os.getpid()
                    self.lock_files[db_path] = lock_file

                    logger.info(f"Acquired exclusive lock on {db_path}")
                    return True

                except FileExistsError:
                    # Lock file exists, check if it's stale
                    if self._is_lock_stale(lock_file):
                        try:
                            os.unlink(lock_file)
                            continue
                        except Exception:
                            pass

                    time.sleep(0.1)
                    continue

            logger.warning(f"Failed to acquire lock on {db_path} within {timeout}s")
            return False

        except Exception as e:
            logger.error(f"Error acquiring lock on {db_path}: {e}")
            return False

    def release_lock(self, db_path: str) -> bool:
        """Release lock on database."""
        try:
            if db_path not in self.active_locks:
                return True

            lock_file = self.lock_files.get(db_path)
            if lock_file and os.path.exists(lock_file):
                os.unlink(lock_file)

            del self.active_locks[db_path]
            if db_path in self.lock_files:
                del self.lock_files[db_path]

            logger.info(f"Released lock on {db_path}")
            return True

        except Exception as e:
            logger.error(f"Error releasing lock on {db_path}: {e}")
            return False

    @contextlib.contextmanager
    def exclusive_access(self, db_path: str, timeout: int = 30):
        """Context manager for exclusive database access."""
        acquired = False
        try:
            # Clean up stale locks first
            self.cleanup_stale_locks(db_path)

            # Acquire lock
            acquired = self.acquire_exclusive_lock(db_path, timeout)
            if not acquired:
                raise RuntimeError(f"Could not acquire exclusive lock on {db_path}")

            yield

        finally:
            if acquired:
                self.release_lock(db_path)

    def get_conflicting_processes(self, db_path: str) -> list[dict]:
        """Get list of processes that might be conflicting with database access."""
        conflicts = []

        try:
            db_path = Path(db_path)
            related_files = [
                str(db_path),
                str(db_path.with_suffix(db_path.suffix + ".db-wal")),
                str(db_path.with_suffix(db_path.suffix + ".db-shm")),
            ]

            for proc in psutil.process_iter(["pid", "name", "cmdline", "open_files"]):
                try:
                    if proc.info["open_files"]:
                        for f in proc.info["open_files"]:
                            if f.path in related_files:
                                conflicts.append(
                                    {
                                        "pid": proc.info["pid"],
                                        "name": proc.info["name"],
                                        "cmdline": " ".join(proc.info["cmdline"][:3]),
                                        "file": f.path,
                                    }
                                )
                                break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except Exception as e:
            logger.error(f"Error finding conflicting processes: {e}")

        return conflicts

    def force_cleanup_database(self, db_path: str) -> bool:
        """Force cleanup of database locks and connections."""
        try:
            logger.info(f"Force cleaning up database: {db_path}")

            # Find and terminate conflicting processes
            conflicts = self.get_conflicting_processes(db_path)
            for conflict in conflicts:
                try:
                    logger.info(f"Terminating conflicting process: {conflict}")
                    proc = psutil.Process(conflict["pid"])
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception as e:
                    logger.warning(
                        f"Failed to terminate process {conflict['pid']}: {e}"
                    )

            # Clean up lock files
            self.cleanup_stale_locks(db_path)

            # Release any locks we're holding
            self.release_lock(db_path)

            logger.info(f"Force cleanup completed for {db_path}")
            return True

        except Exception as e:
            logger.error(f"Force cleanup failed for {db_path}: {e}")
            return False


# Global instance
_lock_manager: DatabaseLockManager | None = None


def get_database_lock_manager() -> DatabaseLockManager:
    """Get global database lock manager."""
    global _lock_manager
    if _lock_manager is None:
        _lock_manager = DatabaseLockManager()
    return _lock_manager


def cleanup_database_locks(db_path: str) -> bool:
    """Convenience function to cleanup database locks."""
    manager = get_database_lock_manager()
    return manager.force_cleanup_database(db_path)


def safe_database_access(db_path: str, timeout: int = 30):
    """Context manager for safe database access."""
    manager = get_database_lock_manager()
    return manager.exclusive_access(db_path, timeout)
