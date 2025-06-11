"""Memory optimization utilities for Unity Wheel Trading Bot.

Provides memory-efficient data structures, monitoring tools, and automatic
cleanup routines to keep memory usage within reasonable bounds.
"""

import gc
import logging
import os
import sys
import time
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterator, List, Optional, Set, Tuple, Union
from weakref import WeakKeyDictionary, WeakSet

import numpy as np

from src.unity_wheel.utils.logging import StructuredLogger

logger = StructuredLogger(logging.getLogger(__name__))


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    total_mb: float
    available_mb: float
    used_mb: float
    used_percent: float
    process_mb: float
    process_percent: float

    def __str__(self) -> str:
        return (
            f"Memory: {self.process_mb:.1f}MB process "
            f"({self.process_percent:.1f}% of {self.total_mb:.0f}MB system)"
        )


@dataclass
class ObjectStats:
    """Statistics for tracked objects."""

    count: int
    total_size_mb: float
    avg_size_mb: float
    max_size_mb: float
    types: Set[str]


class MemoryMonitor:
    """Monitor and track memory usage across the application."""

    def __init__(
        self,
        warning_threshold_mb: float = 100.0,
        critical_threshold_mb: float = 200.0,
        cleanup_interval_seconds: float = 300.0,
    ):
        """Initialize memory monitor.

        Args:
            warning_threshold_mb: Log warning when process memory exceeds this
            critical_threshold_mb: Trigger cleanup when exceeding this
            cleanup_interval_seconds: Automatic cleanup interval
        """
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.cleanup_interval_seconds = cleanup_interval_seconds

        self._last_cleanup = 0.0
        self._tracked_objects: WeakSet = WeakSet()
        self._size_cache: WeakKeyDictionary = WeakKeyDictionary()
        self._object_counters: Dict[str, int] = defaultdict(int)

        # Peak usage tracking
        self._peak_memory_mb = 0.0
        self._peak_objects = 0

        logger.info(
            "memory_monitor_initialized",
            extra={
                "warning_threshold_mb": warning_threshold_mb,
                "critical_threshold_mb": critical_threshold_mb,
                "cleanup_interval_seconds": cleanup_interval_seconds,
            },
        )

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        try:
            import psutil

            # System memory
            system_memory = psutil.virtual_memory()
            total_mb = system_memory.total / (1024 * 1024)
            available_mb = system_memory.available / (1024 * 1024)
            used_mb = system_memory.used / (1024 * 1024)
            used_percent = system_memory.percent

            # Process memory
            process = psutil.Process(os.getpid())
            process_info = process.memory_info()
            process_mb = process_info.rss / (1024 * 1024)
            process_percent = (process_mb / total_mb) * 100

            # Update peak tracking
            self._peak_memory_mb = max(self._peak_memory_mb, process_mb)

            return MemoryStats(
                total_mb=total_mb,
                available_mb=available_mb,
                used_mb=used_mb,
                used_percent=used_percent,
                process_mb=process_mb,
                process_percent=process_percent,
            )

        except ImportError:
            # Fallback if psutil not available
            return self._get_basic_memory_stats()

    def _get_basic_memory_stats(self) -> MemoryStats:
        """Basic memory stats without psutil."""
        # Very basic estimation
        try:
            import resource

            process_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            if sys.platform == "darwin":  # macOS reports in bytes
                process_mb = process_mb / 1024
        except ImportError:
            process_mb = 50.0  # Default estimate

        return MemoryStats(
            total_mb=8192.0,  # Estimate 8GB
            available_mb=4096.0,  # Estimate 4GB available
            used_mb=4096.0,
            used_percent=50.0,
            process_mb=process_mb,
            process_percent=process_mb / 8192.0 * 100,
        )

    def track_object(self, obj: Any, size_hint: Optional[float] = None) -> None:
        """Track an object for memory monitoring.

        Args:
            obj: Object to track
            size_hint: Optional size hint in MB
        """
        try:
            self._tracked_objects.add(obj)

            if size_hint is not None:
                self._size_cache[obj] = size_hint

            obj_type = type(obj).__name__
            self._object_counters[obj_type] += 1

            # Update peak object count
            self._peak_objects = max(self._peak_objects, len(self._tracked_objects))

        except Exception as e:
            logger.warning(
                "failed_to_track_object", extra={"error": str(e), "object_type": type(obj).__name__}
            )

    def get_object_stats(self) -> Dict[str, ObjectStats]:
        """Get statistics for tracked objects."""
        stats_by_type = defaultdict(
            lambda: {"count": 0, "total_size": 0.0, "sizes": [], "types": set()}
        )

        # Gather stats for live objects
        for obj in self._tracked_objects:
            obj_type = type(obj).__name__
            size_mb = self._estimate_object_size(obj)

            stats_by_type[obj_type]["count"] += 1
            stats_by_type[obj_type]["total_size"] += size_mb
            stats_by_type[obj_type]["sizes"].append(size_mb)
            stats_by_type[obj_type]["types"].add(type(obj).__module__ + "." + type(obj).__name__)

        # Convert to ObjectStats
        result = {}
        for obj_type, data in stats_by_type.items():
            sizes = data["sizes"]
            result[obj_type] = ObjectStats(
                count=data["count"],
                total_size_mb=data["total_size"],
                avg_size_mb=data["total_size"] / data["count"] if data["count"] > 0 else 0.0,
                max_size_mb=max(sizes) if sizes else 0.0,
                types=data["types"],
            )

        return result

    def _estimate_object_size(self, obj: Any) -> float:
        """Estimate object size in MB."""
        if obj in self._size_cache:
            return self._size_cache[obj]

        try:
            # For numpy arrays, use actual memory usage
            if hasattr(obj, "nbytes"):
                size_mb = obj.nbytes / (1024 * 1024)
            elif hasattr(obj, "__sizeof__"):
                size_mb = obj.__sizeof__() / (1024 * 1024)
            else:
                size_mb = sys.getsizeof(obj) / (1024 * 1024)

            # Cache the result
            self._size_cache[obj] = size_mb
            return size_mb

        except Exception:
            return 0.1  # Default 100KB estimate

    def check_memory_usage(self, force_cleanup: bool = False) -> Tuple[bool, str]:
        """Check memory usage and trigger cleanup if needed.

        Returns:
            Tuple of (cleanup_triggered, reason)
        """
        stats = self.get_memory_stats()
        current_time = time.time()

        # Check thresholds
        cleanup_needed = False
        reason = ""

        if stats.process_mb > self.critical_threshold_mb:
            cleanup_needed = True
            reason = f"Critical threshold exceeded: {stats.process_mb:.1f}MB > {self.critical_threshold_mb:.1f}MB"
        elif stats.process_mb > self.warning_threshold_mb:
            logger.warning(
                "memory_warning",
                extra={
                    "process_mb": stats.process_mb,
                    "warning_threshold_mb": self.warning_threshold_mb,
                    "process_percent": stats.process_percent,
                },
            )

        # Check cleanup interval
        if not cleanup_needed and current_time - self._last_cleanup > self.cleanup_interval_seconds:
            cleanup_needed = True
            reason = f"Scheduled cleanup (interval: {self.cleanup_interval_seconds}s)"

        # Force cleanup if requested
        if force_cleanup:
            cleanup_needed = True
            reason = "Forced cleanup requested"

        if cleanup_needed:
            self._perform_cleanup(reason)
            self._last_cleanup = current_time
            return True, reason

        return False, ""

    def _perform_cleanup(self, reason: str) -> None:
        """Perform memory cleanup."""
        logger.info("memory_cleanup_started", extra={"reason": reason})

        before_stats = self.get_memory_stats()
        before_objects = len(self._tracked_objects)

        # Clear object tracking for dead objects
        self._tracked_objects = WeakSet([obj for obj in self._tracked_objects if obj is not None])

        # Clear size cache of dead references
        dead_keys = [key for key in self._size_cache.keys() if key not in self._tracked_objects]
        for key in dead_keys:
            del self._size_cache[key]

        # Force garbage collection
        collected = gc.collect()

        after_stats = self.get_memory_stats()
        after_objects = len(self._tracked_objects)

        memory_freed = before_stats.process_mb - after_stats.process_mb
        objects_freed = before_objects - after_objects

        logger.info(
            "memory_cleanup_completed",
            extra={
                "memory_freed_mb": memory_freed,
                "objects_freed": objects_freed,
                "gc_collected": collected,
                "before_mb": before_stats.process_mb,
                "after_mb": after_stats.process_mb,
                "before_objects": before_objects,
                "after_objects": after_objects,
            },
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory summary."""
        stats = self.get_memory_stats()
        object_stats = self.get_object_stats()

        return {
            "current_stats": stats,
            "peak_memory_mb": self._peak_memory_mb,
            "peak_objects": self._peak_objects,
            "tracked_objects": len(self._tracked_objects),
            "object_stats": object_stats,
            "thresholds": {
                "warning_mb": self.warning_threshold_mb,
                "critical_mb": self.critical_threshold_mb,
            },
            "cleanup": {
                "last_cleanup": self._last_cleanup,
                "interval_seconds": self.cleanup_interval_seconds,
            },
        }


class MemoryEfficientArray:
    """Memory-efficient array that automatically manages memory usage."""

    def __init__(
        self, initial_capacity: int = 1000, max_size: int = 10000, dtype: np.dtype = np.float64
    ):
        """Initialize memory-efficient array.

        Args:
            initial_capacity: Initial array capacity
            max_size: Maximum array size before recycling
            dtype: Numpy data type
        """
        self.max_size = max_size
        self.dtype = dtype

        self._data = np.empty(initial_capacity, dtype=dtype)
        self._size = 0
        self._capacity = initial_capacity

        # Register with memory monitor
        _global_monitor.track_object(self, self._estimate_size())

    def _estimate_size(self) -> float:
        """Estimate array size in MB."""
        return self._data.nbytes / (1024 * 1024)

    def append(self, value: Union[float, np.ndarray]) -> None:
        """Append value(s) to array."""
        if isinstance(value, np.ndarray):
            needed_size = self._size + value.size
        else:
            needed_size = self._size + 1

        # Resize if needed
        if needed_size > self._capacity:
            self._resize(max(needed_size * 2, self._capacity * 2))

        # Recycle if too large
        if needed_size > self.max_size:
            self._recycle()

        # Add data
        if isinstance(value, np.ndarray):
            self._data[self._size : self._size + value.size] = value.flatten()
            self._size += value.size
        else:
            self._data[self._size] = value
            self._size += 1

    def _resize(self, new_capacity: int) -> None:
        """Resize internal array."""
        new_data = np.empty(new_capacity, dtype=self.dtype)
        new_data[: self._size] = self._data[: self._size]
        self._data = new_data
        self._capacity = new_capacity

    def _recycle(self) -> None:
        """Recycle array by keeping only recent data."""
        keep_size = self.max_size // 2
        if self._size > keep_size:
            # Keep the most recent half
            self._data[:keep_size] = self._data[self._size - keep_size : self._size]
            self._size = keep_size

            logger.debug(
                "array_recycled", extra={"new_size": self._size, "max_size": self.max_size}
            )

    def get_data(self) -> np.ndarray:
        """Get view of current data."""
        return self._data[: self._size]

    def clear(self) -> None:
        """Clear all data."""
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index) -> Union[float, np.ndarray]:
        if index >= self._size:
            raise IndexError("Index out of range")
        return self._data[index]


class StreamingDataProcessor:
    """Process large datasets in streaming fashion to minimize memory usage."""

    def __init__(self, chunk_size: int = 1000):
        """Initialize streaming processor.

        Args:
            chunk_size: Size of processing chunks
        """
        self.chunk_size = chunk_size
        self._processed_count = 0

    def process_chunks(
        self,
        data_source: Iterator[Any],
        processor_func: callable,
        accumulator_func: Optional[callable] = None,
    ) -> Generator[Any, None, None]:
        """Process data in chunks to minimize memory usage.

        Args:
            data_source: Iterator over data
            processor_func: Function to process each chunk
            accumulator_func: Optional function to accumulate results

        Yields:
            Processed results
        """
        chunk = []
        accumulated_result = None

        for item in data_source:
            chunk.append(item)

            if len(chunk) >= self.chunk_size:
                # Process chunk
                result = processor_func(chunk)

                # Accumulate if needed
                if accumulator_func:
                    accumulated_result = accumulator_func(accumulated_result, result)
                else:
                    yield result

                # Clear chunk to free memory
                chunk.clear()
                self._processed_count += self.chunk_size

                # Check memory periodically
                if self._processed_count % (self.chunk_size * 10) == 0:
                    _global_monitor.check_memory_usage()

        # Process remaining items
        if chunk:
            result = processor_func(chunk)
            if accumulator_func:
                accumulated_result = accumulator_func(accumulated_result, result)
            else:
                yield result

        # Yield final accumulated result
        if accumulator_func and accumulated_result is not None:
            yield accumulated_result

    def batch_process(
        self, items: List[Any], batch_func: callable, combine_func: Optional[callable] = None
    ) -> Any:
        """Process items in batches.

        Args:
            items: List of items to process
            batch_func: Function to process each batch
            combine_func: Function to combine batch results

        Returns:
            Combined result or list of batch results
        """
        results = []

        for i in range(0, len(items), self.chunk_size):
            batch = items[i : i + self.chunk_size]
            result = batch_func(batch)
            results.append(result)

            # Periodic memory check
            if len(results) % 10 == 0:
                _global_monitor.check_memory_usage()

        if combine_func:
            return combine_func(results)
        else:
            return results


class MemoryPool:
    """Memory pool for reusing objects to reduce allocation overhead."""

    def __init__(
        self, factory_func: callable, max_size: int = 100, reset_func: Optional[callable] = None
    ):
        """Initialize memory pool.

        Args:
            factory_func: Function to create new objects
            max_size: Maximum pool size
            reset_func: Function to reset objects before reuse
        """
        self.factory_func = factory_func
        self.max_size = max_size
        self.reset_func = reset_func

        self._pool: deque = deque()
        self._created_count = 0
        self._reused_count = 0

    def get(self) -> Any:
        """Get object from pool or create new one."""
        if self._pool:
            obj = self._pool.popleft()
            if self.reset_func:
                self.reset_func(obj)
            self._reused_count += 1
            return obj
        else:
            obj = self.factory_func()
            self._created_count += 1
            _global_monitor.track_object(obj)
            return obj

    def put(self, obj: Any) -> None:
        """Return object to pool."""
        if len(self._pool) < self.max_size:
            self._pool.append(obj)

    def clear(self) -> None:
        """Clear the pool."""
        self._pool.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return {
            "pool_size": len(self._pool),
            "max_size": self.max_size,
            "created_count": self._created_count,
            "reused_count": self._reused_count,
            "reuse_rate": self._reused_count / max(1, self._created_count + self._reused_count),
        }


@contextmanager
def memory_profiler(operation_name: str):
    """Context manager for profiling memory usage of operations."""
    before_stats = _global_monitor.get_memory_stats()
    before_time = time.time()

    try:
        yield
    finally:
        after_stats = _global_monitor.get_memory_stats()
        after_time = time.time()

        memory_diff = after_stats.process_mb - before_stats.process_mb
        duration = after_time - before_time

        logger.info(
            "memory_profile_completed",
            extra={
                "operation": operation_name,
                "memory_change_mb": memory_diff,
                "duration_seconds": duration,
                "before_mb": before_stats.process_mb,
                "after_mb": after_stats.process_mb,
            },
        )

        # Trigger cleanup if significant memory increase
        if memory_diff > 50.0:  # More than 50MB increase
            _global_monitor.check_memory_usage(force_cleanup=True)


def optimize_numpy_memory(array: np.ndarray, target_dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Optimize numpy array memory usage."""
    if target_dtype is None:
        # Auto-select optimal dtype
        if array.dtype == np.float64:
            # Check if we can downcast to float32
            array_f32 = array.astype(np.float32)
            if np.allclose(array, array_f32, rtol=1e-6):
                target_dtype = np.float32
            else:
                target_dtype = array.dtype
        elif array.dtype == np.int64:
            # Check if we can use smaller int type
            if np.all((array >= np.iinfo(np.int32).min) & (array <= np.iinfo(np.int32).max)):
                target_dtype = np.int32
            else:
                target_dtype = array.dtype
        else:
            target_dtype = array.dtype

    if target_dtype != array.dtype:
        optimized = array.astype(target_dtype)
        original_size = array.nbytes
        optimized_size = optimized.nbytes

        logger.debug(
            "numpy_memory_optimized",
            extra={
                "original_dtype": str(array.dtype),
                "optimized_dtype": str(target_dtype),
                "original_size_mb": original_size / (1024 * 1024),
                "optimized_size_mb": optimized_size / (1024 * 1024),
                "memory_saved_mb": (original_size - optimized_size) / (1024 * 1024),
            },
        )

        return optimized

    return array


def create_memory_efficient_dict(max_size: int = 10000) -> Dict[Any, Any]:
    """Create a memory-efficient dictionary with size limits."""

    class LimitedDict(dict):
        def __init__(self):
            super().__init__()
            self.max_size = max_size
            self.access_order = deque()

        def __setitem__(self, key, value):
            if key in self:
                # Update existing key
                self.access_order.remove(key)
            elif len(self) >= self.max_size:
                # Remove least recently used item
                lru_key = self.access_order.popleft()
                del super(LimitedDict, self)[lru_key]

            super().__setitem__(key, value)
            self.access_order.append(key)

        def __getitem__(self, key):
            value = super().__getitem__(key)
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return value

        def __delitem__(self, key):
            super().__delitem__(key)
            self.access_order.remove(key)

    return LimitedDict()


# Global memory monitor instance
_global_monitor = MemoryMonitor()


def get_memory_monitor() -> MemoryMonitor:
    """Get the global memory monitor instance."""
    return _global_monitor


def get_memory_summary() -> Dict[str, Any]:
    """Get comprehensive memory usage summary."""
    return _global_monitor.get_summary()


def force_memory_cleanup() -> None:
    """Force immediate memory cleanup."""
    _global_monitor.check_memory_usage(force_cleanup=True)


def set_memory_thresholds(warning_mb: float, critical_mb: float) -> None:
    """Set memory warning and critical thresholds."""
    _global_monitor.warning_threshold_mb = warning_mb
    _global_monitor.critical_threshold_mb = critical_mb

    logger.info(
        "memory_thresholds_updated", extra={"warning_mb": warning_mb, "critical_mb": critical_mb}
    )
