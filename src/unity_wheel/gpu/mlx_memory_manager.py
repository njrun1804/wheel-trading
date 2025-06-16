"""
MLX GPU Memory Manager with Leak Prevention and Recovery

This module provides comprehensive memory management for MLX operations:
- Memory leak detection and prevention
- Automatic cleanup and garbage collection
- Memory pooling for efficient reuse
- Real-time monitoring and alerts
- Recovery mechanisms for out-of-memory scenarios
"""
from __future__ import annotations

import asyncio
import gc
import logging
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    total_allocated_mb: float
    peak_allocated_mb: float
    current_arrays: int
    pooled_arrays: int
    cleanup_count: int
    leak_warnings: int
    gc_runs: int
    metal_clears: int


class MLXArrayTracker:
    """Tracks MLX arrays to detect memory leaks."""

    def __init__(self):
        self._arrays = weakref.WeakSet()
        self._creation_times = {}
        self._allocation_sizes = {}
        self._leak_threshold_seconds = 300  # 5 minutes
        self._lock = threading.Lock()

    def track_array(self, array: mx.array, context: str = "unknown"):
        """Track a new MLX array."""
        if not MLX_AVAILABLE:
            return

        with self._lock:
            self._arrays.add(array)
            array_id = id(array)
            self._creation_times[array_id] = time.time()
            self._allocation_sizes[array_id] = array.nbytes

            logger.debug(
                f"Tracking array {array_id} ({array.nbytes} bytes) from {context}"
            )

    def check_for_leaks(self) -> list[dict[str, Any]]:
        """Check for potential memory leaks."""
        leaks = []
        current_time = time.time()

        with self._lock:
            for array in list(self._arrays):
                array_id = id(array)
                creation_time = self._creation_times.get(array_id, current_time)
                age = current_time - creation_time

                if age > self._leak_threshold_seconds:
                    leaks.append(
                        {
                            "array_id": array_id,
                            "age_seconds": age,
                            "size_bytes": self._allocation_sizes.get(array_id, 0),
                            "shape": array.shape,
                            "dtype": str(array.dtype),
                        }
                    )

        return leaks

    def cleanup_stale_tracking(self):
        """Clean up tracking data for garbage collected arrays."""
        with self._lock:
            active_ids = {id(array) for array in self._arrays}

            # Remove tracking data for arrays that were garbage collected
            stale_ids = set(self._creation_times.keys()) - active_ids
            for array_id in stale_ids:
                self._creation_times.pop(array_id, None)
                self._allocation_sizes.pop(array_id, None)


class MLXMemoryPool:
    """Memory pool for efficient MLX array reuse."""

    def __init__(self, max_pool_size_mb: int = 2048):
        self.max_pool_size_mb = max_pool_size_mb
        self.pools = {}  # (shape, dtype) -> list of arrays
        self.pool_size_mb = 0
        self._lock = threading.Lock()

    def get_array(self, shape: tuple[int, ...], dtype=mx.float32) -> mx.array:
        """Get array from pool or create new one."""
        if not MLX_AVAILABLE:
            return None

        key = (shape, str(dtype))

        with self._lock:
            if key in self.pools and self.pools[key]:
                array = self.pools[key].pop()
                # Zero out for reuse
                array = mx.zeros_like(array)
                mx.eval(array)
                logger.debug(f"Reused array from pool: {shape} {dtype}")
                return array

        # Create new array
        array = mx.zeros(shape, dtype=dtype)
        mx.eval(array)
        logger.debug(f"Created new array: {shape} {dtype}")
        return array

    def return_array(self, array: mx.array):
        """Return array to pool for reuse."""
        if not MLX_AVAILABLE or array is None:
            return

        try:
            key = (array.shape, str(array.dtype))
            array_size_mb = array.nbytes / (1024 * 1024)

            with self._lock:
                # Check if we have room in the pool
                if self.pool_size_mb + array_size_mb <= self.max_pool_size_mb:
                    if key not in self.pools:
                        self.pools[key] = []

                    # Limit pool size per key
                    if len(self.pools[key]) < 10:
                        self.pools[key].append(array)
                        self.pool_size_mb += array_size_mb
                        mx.eval(array)  # Ensure all operations complete
                        logger.debug(f"Returned array to pool: {array.shape}")
                        return

            # If we can't pool it, just let it be garbage collected
            logger.debug(f"Array not pooled (pool full): {array.shape}")

        except Exception as e:
            logger.warning(f"Failed to return array to pool: {e}")

    def clear_pool(self):
        """Clear the entire memory pool."""
        with self._lock:
            cleared_count = sum(len(arrays) for arrays in self.pools.values())
            self.pools.clear()
            self.pool_size_mb = 0
            logger.info(f"Cleared memory pool: {cleared_count} arrays")

    def get_pool_stats(self) -> dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            total_arrays = sum(len(arrays) for arrays in self.pools.values())
            return {
                "total_arrays": total_arrays,
                "pool_size_mb": self.pool_size_mb,
                "pool_types": len(self.pools),
                "utilization": self.pool_size_mb / self.max_pool_size_mb,
            }


class MLXMemoryManager:
    """Comprehensive MLX memory manager with leak prevention."""

    def __init__(
        self,
        max_memory_mb: int = 4096,
        cleanup_threshold: int = 100,
        monitoring_interval: float = 30.0,
    ):
        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold = cleanup_threshold
        self.monitoring_interval = monitoring_interval

        # Components
        self.tracker = MLXArrayTracker()
        self.pool = MLXMemoryPool(max_pool_size_mb=max_memory_mb // 2)

        # Statistics
        self.stats = MemoryStats(
            total_allocated_mb=0,
            peak_allocated_mb=0,
            current_arrays=0,
            pooled_arrays=0,
            cleanup_count=0,
            leak_warnings=0,
            gc_runs=0,
            metal_clears=0,
        )

        # Monitoring
        self.operation_count = 0
        self.monitoring_active = False
        self.monitoring_task = None
        self._lock = threading.Lock()

        # Thread pool for cleanup operations
        self.cleanup_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="MLX-Cleanup"
        )

        logger.info(f"MLX Memory Manager initialized: {max_memory_mb}MB limit")

    def start_monitoring(self):
        """Start memory monitoring background task."""
        if not MLX_AVAILABLE or self.monitoring_active:
            return

        self.monitoring_active = True
        loop = asyncio.get_event_loop()
        self.monitoring_task = loop.create_task(self._monitoring_loop())
        logger.info("Started MLX memory monitoring")

    async def stop_monitoring(self):
        """Stop memory monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.monitoring_task

        self.cleanup_executor.shutdown(wait=True)
        logger.info("Stopped MLX memory monitoring")

    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                await asyncio.sleep(self.monitoring_interval)

                # Check for leaks
                leaks = self.tracker.check_for_leaks()
                if leaks:
                    self.stats.leak_warnings += 1
                    logger.warning(f"Detected {len(leaks)} potential memory leaks")

                    # Log details of worst leaks
                    for leak in sorted(
                        leaks, key=lambda x: x["size_bytes"], reverse=True
                    )[:3]:
                        logger.warning(
                            f"Leak: array {leak['array_id']}, age {leak['age_seconds']:.1f}s, size {leak['size_bytes']} bytes"
                        )

                # Automatic cleanup if needed
                if self.operation_count >= self.cleanup_threshold:
                    await self.cleanup_memory()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")

    @contextmanager
    def track_operation(self, operation_name: str = "unknown"):
        """Context manager to track MLX operations and ensure cleanup."""
        arrays_created = []
        start_time = time.time()

        try:
            # Increment operation count
            with self._lock:
                self.operation_count += 1

            # Yield control to the operation
            yield arrays_created

        except Exception as e:
            logger.error(f"Error in MLX operation '{operation_name}': {e}")
            raise
        finally:
            # Ensure all created arrays are evaluated and tracked
            try:
                if arrays_created:
                    mx.eval(*arrays_created)
                    for array in arrays_created:
                        self.tracker.track_array(array, operation_name)

                # Periodic cleanup
                if self.operation_count % self.cleanup_threshold == 0:
                    # Schedule cleanup in background to avoid blocking
                    self.cleanup_executor.submit(self._sync_cleanup)

            except Exception as e:
                logger.warning(f"Error in operation cleanup: {e}")

            duration = time.time() - start_time
            logger.debug(
                f"MLX operation '{operation_name}' completed in {duration:.3f}s"
            )

    def create_array(
        self,
        shape: tuple[int, ...],
        dtype=mx.float32,
        operation_name: str = "create_array",
    ) -> mx.array:
        """Create MLX array with memory tracking."""
        if not MLX_AVAILABLE:
            return None

        # Try to get from pool first
        array = self.pool.get_array(shape, dtype)
        if array is None:
            array = mx.zeros(shape, dtype=dtype)
            mx.eval(array)

        # Track the array
        self.tracker.track_array(array, operation_name)

        # Update stats
        with self._lock:
            self.stats.current_arrays += 1
            size_mb = array.nbytes / (1024 * 1024)
            self.stats.total_allocated_mb += size_mb
            self.stats.peak_allocated_mb = max(
                self.stats.peak_allocated_mb, self.stats.total_allocated_mb
            )

        return array

    def release_array(self, array: mx.array):
        """Release MLX array back to pool or for garbage collection."""
        if not MLX_AVAILABLE or array is None:
            return

        try:
            # Ensure all operations complete
            mx.eval(array)

            # Try to return to pool
            self.pool.return_array(array)

            # Update stats
            with self._lock:
                self.stats.current_arrays = max(0, self.stats.current_arrays - 1)
                size_mb = array.nbytes / (1024 * 1024)
                self.stats.total_allocated_mb = max(
                    0, self.stats.total_allocated_mb - size_mb
                )

        except Exception as e:
            logger.warning(f"Error releasing array: {e}")

    def _sync_cleanup(self):
        """Synchronous cleanup for thread pool execution."""
        try:
            if MLX_AVAILABLE:
                # Force evaluation of all pending operations
                gc.collect()

                # Clear Metal memory if available
                if hasattr(mx, "metal") and hasattr(mx.metal, "clear_memory"):
                    mx.metal.clear_memory()
                    self.stats.metal_clears += 1

                # Clean up tracking data
                self.tracker.cleanup_stale_tracking()

                # Update stats
                self.stats.cleanup_count += 1
                self.stats.gc_runs += 1

                logger.debug("Completed synchronous MLX memory cleanup")

        except Exception as e:
            logger.error(f"Error in sync cleanup: {e}")

    async def cleanup_memory(self, force: bool = False):
        """Comprehensive memory cleanup."""
        if not MLX_AVAILABLE:
            return

        logger.info("Starting MLX memory cleanup...")
        start_time = time.time()

        try:
            # Run sync cleanup in executor to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                self.cleanup_executor, self._sync_cleanup
            )

            # Reset operation count
            with self._lock:
                self.operation_count = 0

            # Clear pool if forced or if we're using too much memory
            pool_stats = self.pool.get_pool_stats()
            if force or pool_stats["utilization"] > 0.8:
                self.pool.clear_pool()

            duration = time.time() - start_time
            logger.info(f"MLX memory cleanup completed in {duration:.3f}s")

        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        with self._lock:
            # Update current arrays count from tracker
            self.stats.current_arrays = len(self.tracker._arrays)

            # Update pooled arrays count
            pool_stats = self.pool.get_pool_stats()
            self.stats.pooled_arrays = pool_stats["total_arrays"]

            return self.stats

    def print_memory_report(self):
        """Print detailed memory usage report."""
        stats = self.get_memory_stats()
        pool_stats = self.pool.get_pool_stats()

        print("\n=== MLX Memory Manager Report ===")
        print(f"Total Allocated: {stats.total_allocated_mb:.1f} MB")
        print(f"Peak Allocated: {stats.peak_allocated_mb:.1f} MB")
        print(f"Current Arrays: {stats.current_arrays}")
        print(f"Pooled Arrays: {stats.pooled_arrays}")
        print(f"Pool Utilization: {pool_stats['utilization']:.1%}")
        print(f"Cleanup Runs: {stats.cleanup_count}")
        print(f"Leak Warnings: {stats.leak_warnings}")
        print(f"GC Runs: {stats.gc_runs}")
        print(f"Metal Clears: {stats.metal_clears}")
        print("=================================\n")

    async def __aenter__(self):
        """Async context manager entry."""
        self.start_monitoring()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_monitoring()
        await self.cleanup_memory(force=True)


# Global memory manager instance
_memory_manager: MLXMemoryManager | None = None


def get_mlx_memory_manager() -> MLXMemoryManager:
    """Get or create the global MLX memory manager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MLXMemoryManager()
    return _memory_manager


def safe_mx_eval(*arrays: mx.array):
    """Safely evaluate MLX arrays with error handling."""
    if not MLX_AVAILABLE:
        return

    try:
        mx.eval(*arrays)
    except Exception as e:
        logger.warning(f"Error evaluating MLX arrays: {e}")


def safe_release_arrays(*arrays: mx.array):
    """Safely release MLX arrays."""
    manager = get_mlx_memory_manager()
    for array in arrays:
        if array is not None:
            manager.release_array(array)


# Decorator for automatic memory management
def mlx_memory_managed(func):
    """Decorator to add automatic MLX memory management to functions."""

    def wrapper(*args, **kwargs):
        manager = get_mlx_memory_manager()
        with manager.track_operation(func.__name__):
            return func(*args, **kwargs)

    async def async_wrapper(*args, **kwargs):
        manager = get_mlx_memory_manager()
        with manager.track_operation(func.__name__):
            return await func(*args, **kwargs)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return wrapper


if __name__ == "__main__":
    # Test the memory manager
    async def test_memory_manager():
        print("Testing MLX Memory Manager...")

        async with MLXMemoryManager() as manager:
            # Create some test arrays
            for i in range(50):
                array = manager.create_array((1000, 1000), operation_name=f"test_{i}")
                # Simulate some operations
                if MLX_AVAILABLE:
                    result = mx.sum(array)
                    mx.eval(result)
                # Release the array
                manager.release_array(array)

                if i % 10 == 0:
                    await manager.cleanup_memory()

            # Print final report
            manager.print_memory_report()

    if MLX_AVAILABLE:
        asyncio.run(test_memory_manager())
    else:
        print("MLX not available - skipping test")
