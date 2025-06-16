#!/usr/bin/env python3
"""
GPU Memory Optimizer for M4 Pro - MLX Memory Management
Target: Reduce GPU memory usage and prevent memory leaks

Key Features:
1. Smart memory pool management
2. Automatic garbage collection triggers
3. Memory pressure detection
4. Lazy allocation strategies
5. Metal cache optimization
6. Operation batching for efficiency
"""

import asyncio
import gc
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
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
class GPUMemoryStats:
    """GPU memory usage statistics."""

    allocated_mb: float
    peak_mb: float
    cached_mb: float
    active_arrays: int
    total_allocations: int
    gc_runs: int
    cache_hits: int
    cache_misses: int
    last_cleanup: float


@dataclass
class MemoryOperation:
    """Represents a memory operation."""

    operation_id: str
    size_mb: float
    operation_type: str
    timestamp: float
    priority: int = 5  # 1-10, higher = more important


class MLXMemoryPool:
    """Memory pool for MLX arrays with size-based allocation."""

    def __init__(self, max_pool_size: int = 100):
        self.max_pool_size = max_pool_size
        self._pools: dict[tuple[int, ...], list[Any]] = defaultdict(list)
        self._lock = threading.Lock()
        self.allocations = 0
        self.reuses = 0
        self.pool_hits = 0
        self.pool_misses = 0

    def get_array(self, shape: tuple[int, ...], dtype=None) -> Any | None:
        """Get array from pool or None if not available."""
        if not MLX_AVAILABLE:
            return None

        pool_key = shape
        with self._lock:
            pool = self._pools.get(pool_key, [])
            if pool:
                self.reuses += 1
                self.pool_hits += 1
                return pool.pop()
            else:
                self.pool_misses += 1
                return None

    def return_array(self, array: Any):
        """Return array to pool if there's space."""
        if not MLX_AVAILABLE or array is None:
            return

        try:
            shape = array.shape
            with self._lock:
                pool = self._pools[shape]
                if (
                    len(pool) < self.max_pool_size // len(self._pools)
                    if self._pools
                    else self.max_pool_size
                ):
                    pool.append(array)
        except (ValueError, TypeError, KeyError) as e:
            logger.debug(f"Failed to return array to pool: {e}")
            # Array will be garbage collected instead

    def clear_pools(self):
        """Clear all pools."""
        with self._lock:
            for pool in self._pools.values():
                pool.clear()
            self._pools.clear()

    def get_pool_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            total_pooled = sum(len(pool) for pool in self._pools.values())
            return {
                "total_pooled_arrays": total_pooled,
                "pool_types": len(self._pools),
                "allocations": self.allocations,
                "reuses": self.reuses,
                "hit_rate": self.pool_hits / (self.pool_hits + self.pool_misses)
                if (self.pool_hits + self.pool_misses) > 0
                else 0,
            }


class GPUMemoryTracker:
    """Track GPU memory allocations and detect leaks."""

    def __init__(self, max_tracked_operations: int = 1000):
        self.max_tracked_operations = max_tracked_operations
        self._operations: deque[MemoryOperation] = deque(maxlen=max_tracked_operations)
        self._active_operations: dict[str, MemoryOperation] = {}
        self._memory_timeline: list[tuple[float, float]] = []  # (timestamp, memory_mb)
        self._lock = threading.Lock()

    def track_allocation(
        self, operation_id: str, size_mb: float, operation_type: str, priority: int = 5
    ):
        """Track memory allocation."""
        operation = MemoryOperation(
            operation_id=operation_id,
            size_mb=size_mb,
            operation_type=operation_type,
            timestamp=time.time(),
            priority=priority,
        )

        with self._lock:
            self._operations.append(operation)
            self._active_operations[operation_id] = operation

    def track_deallocation(self, operation_id: str):
        """Track memory deallocation."""
        with self._lock:
            self._active_operations.pop(operation_id, None)

    def record_memory_usage(self, memory_mb: float):
        """Record current memory usage."""
        with self._lock:
            self._memory_timeline.append((time.time(), memory_mb))
            # Keep only last 100 measurements
            if len(self._memory_timeline) > 100:
                self._memory_timeline = self._memory_timeline[-100:]

    def detect_potential_leaks(self) -> list[MemoryOperation]:
        """Detect operations that might be leaking memory."""
        current_time = time.time()
        leak_threshold = 300  # 5 minutes

        with self._lock:
            potential_leaks = []
            for operation in self._active_operations.values():
                if current_time - operation.timestamp > leak_threshold:
                    potential_leaks.append(operation)

            return sorted(potential_leaks, key=lambda x: x.timestamp)

    def get_memory_growth_rate(self) -> float:
        """Calculate memory growth rate in MB/minute."""
        with self._lock:
            if len(self._memory_timeline) < 2:
                return 0.0

            # Use recent measurements
            recent_timeline = (
                self._memory_timeline[-10:]
                if len(self._memory_timeline) >= 10
                else self._memory_timeline
            )

            if len(recent_timeline) < 2:
                return 0.0

            time_span_minutes = (recent_timeline[-1][0] - recent_timeline[0][0]) / 60
            memory_growth = recent_timeline[-1][1] - recent_timeline[0][1]

            return memory_growth / max(0.1, time_span_minutes)


class M4ProGPUMemoryManager:
    """Optimized GPU memory manager for M4 Pro."""

    def __init__(self, max_gpu_memory_mb: float = 600):  # 600MB budget for M4 Pro
        if not MLX_AVAILABLE:
            logger.warning("MLX not available - GPU memory management disabled")
            return

        self.max_gpu_memory_mb = max_gpu_memory_mb
        self.memory_pressure_threshold = max_gpu_memory_mb * 0.8  # 480MB
        self.critical_threshold = max_gpu_memory_mb * 0.9  # 540MB

        # Memory management components
        self.memory_pool = MLXMemoryPool()
        self.memory_tracker = GPUMemoryTracker()

        # State tracking
        self.stats = GPUMemoryStats(
            allocated_mb=0,
            peak_mb=0,
            cached_mb=0,
            active_arrays=0,
            total_allocations=0,
            gc_runs=0,
            cache_hits=0,
            cache_misses=0,
            last_cleanup=time.time(),
        )

        # Optimization settings
        self.auto_cleanup_enabled = True
        self.gc_threshold = 100  # MB threshold for automatic GC
        self.batch_operations = True
        self._pending_operations: list[Callable] = []
        self._batch_lock = threading.Lock()

        # Monitoring
        self._monitoring_active = False
        self._monitor_task = None

        logger.info(
            f"M4 Pro GPU Memory Manager initialized (budget: {max_gpu_memory_mb}MB)"
        )

    def get_current_usage_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if not MLX_AVAILABLE:
            return 0.0

        try:
            return mx.metal.get_active_memory() / (1024 * 1024)
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"Could not get GPU memory usage: {e}")
            return 0.0

    def get_cached_memory_mb(self) -> float:
        """Get cached GPU memory in MB."""
        if not MLX_AVAILABLE:
            return 0.0

        try:
            return mx.metal.get_cache_memory() / (1024 * 1024)
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"Could not get GPU cache memory: {e}")
            return 0.0

    @contextmanager
    def allocate_operation(
        self, operation_name: str, estimated_mb: float, priority: int = 5
    ):
        """Context manager for GPU memory allocation."""
        if not MLX_AVAILABLE:
            yield
            return

        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"

        # Check if allocation is possible
        current_usage = self.get_current_usage_mb()
        if current_usage + estimated_mb > self.memory_pressure_threshold:
            # Try to free memory
            freed = self.cleanup_memory(force=False)
            logger.info(
                f"GPU memory pressure: freed {freed:.1f}MB for {operation_name}"
            )

            current_usage = self.get_current_usage_mb()
            if current_usage + estimated_mb > self.critical_threshold:
                logger.warning(
                    f"GPU allocation may fail: {estimated_mb:.1f}MB requested, {current_usage:.1f}MB in use"
                )

        # Track allocation
        self.memory_tracker.track_allocation(
            operation_id, estimated_mb, operation_name, priority
        )
        self.stats.total_allocations += 1

        try:
            yield operation_id
        finally:
            # Track deallocation
            self.memory_tracker.track_deallocation(operation_id)

    def create_optimized_array(
        self, shape: tuple[int, ...], dtype=None, init_fn=None
    ) -> Any:
        """Create array with pool optimization."""
        if not MLX_AVAILABLE:
            return None

        # Try to get from pool first
        pooled_array = self.memory_pool.get_array(shape, dtype)
        if pooled_array is not None:
            self.stats.cache_hits += 1
            if init_fn:
                # Reinitialize pooled array
                pooled_array = init_fn(shape, dtype)
            return pooled_array

        # Allocate new array
        self.stats.cache_misses += 1
        if init_fn:
            return init_fn(shape, dtype)
        else:
            return mx.zeros(shape, dtype=dtype or mx.float32)

    def return_array_to_pool(self, array: Any):
        """Return array to memory pool."""
        if MLX_AVAILABLE and array is not None:
            self.memory_pool.return_array(array)

    def cleanup_memory(self, force: bool = False) -> float:
        """Clean up GPU memory."""
        if not MLX_AVAILABLE:
            return 0.0

        initial_usage = self.get_current_usage_mb()

        # Clear memory pool
        self.memory_pool.clear_pools()

        # Force garbage collection
        if force or initial_usage > self.gc_threshold:
            collected = gc.collect()
            self.stats.gc_runs += 1
            logger.debug(f"GPU GC collected {collected} objects")

        # Clear MLX caches
        with suppress(Exception):
            mx.metal.clear_cache()

        # Update stats
        final_usage = self.get_current_usage_mb()
        freed = initial_usage - final_usage

        self.stats.last_cleanup = time.time()
        logger.debug(
            f"GPU memory cleanup: {initial_usage:.1f}MB -> {final_usage:.1f}MB (freed {freed:.1f}MB)"
        )

        return freed

    def batch_operation(self, operation: Callable):
        """Add operation to batch for efficient execution."""
        if not self.batch_operations:
            return operation()

        with self._batch_lock:
            self._pending_operations.append(operation)

            # Execute batch if it's getting large
            if len(self._pending_operations) >= 10:
                self._execute_batch()

    def _execute_batch(self):
        """Execute batched operations."""
        if not self._pending_operations:
            return

        logger.debug(
            f"Executing GPU operation batch of {len(self._pending_operations)} operations"
        )

        try:
            # Execute all operations
            results = []
            for operation in self._pending_operations:
                try:
                    result = operation()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch operation failed: {e}")
                    results.append(None)

            # Force evaluation of all results together
            if MLX_AVAILABLE:
                mx.eval(results)

        finally:
            self._pending_operations.clear()

    def flush_batch(self):
        """Flush any pending batched operations."""
        with self._batch_lock:
            if self._pending_operations:
                self._execute_batch()

    async def start_monitoring(self, interval_seconds: float = 10.0):
        """Start background memory monitoring."""
        if not MLX_AVAILABLE or self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitor_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
        logger.info("Started GPU memory monitoring")

    async def stop_monitoring(self):
        """Stop background memory monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._monitor_task

        logger.info("Stopped GPU memory monitoring")

    async def _monitoring_loop(self, interval_seconds: float):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                # Update stats
                current_usage = self.get_current_usage_mb()
                cached_usage = self.get_cached_memory_mb()

                self.stats.allocated_mb = current_usage
                self.stats.cached_mb = cached_usage
                self.stats.peak_mb = max(self.stats.peak_mb, current_usage)

                # Record for tracking
                self.memory_tracker.record_memory_usage(current_usage)

                # Check for memory pressure
                if current_usage > self.memory_pressure_threshold:
                    logger.warning(
                        f"GPU memory pressure: {current_usage:.1f}MB / {self.max_gpu_memory_mb:.1f}MB"
                    )
                    if self.auto_cleanup_enabled:
                        freed = self.cleanup_memory(force=True)
                        logger.info(f"Auto-cleanup freed {freed:.1f}MB")

                # Check for potential leaks
                potential_leaks = self.memory_tracker.detect_potential_leaks()
                if potential_leaks:
                    growth_rate = self.memory_tracker.get_memory_growth_rate()
                    logger.warning(
                        f"Potential GPU memory leaks detected: {len(potential_leaks)} operations, growth rate: {growth_rate:.1f}MB/min"
                    )

                # Periodic logging
                if time.time() % 60 < interval_seconds:  # Log once per minute
                    pool_stats = self.memory_pool.get_pool_stats()
                    logger.info(
                        f"GPU Memory: {current_usage:.1f}MB allocated, {cached_usage:.1f}MB cached, "
                        f"Pool hit rate: {pool_stats['hit_rate']:.1%}"
                    )

                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"GPU memory monitoring error: {e}")
                await asyncio.sleep(interval_seconds)

    def get_memory_report(self) -> dict[str, Any]:
        """Get comprehensive memory report."""
        current_usage = self.get_current_usage_mb()
        cached_usage = self.get_cached_memory_mb()
        pool_stats = self.memory_pool.get_pool_stats()

        # Update stats
        self.stats.allocated_mb = current_usage
        self.stats.cached_mb = cached_usage

        return {
            "timestamp": time.time(),
            "gpu_available": MLX_AVAILABLE,
            "memory_stats": {
                "allocated_mb": current_usage,
                "cached_mb": cached_usage,
                "peak_mb": self.stats.peak_mb,
                "budget_mb": self.max_gpu_memory_mb,
                "usage_percent": (current_usage / self.max_gpu_memory_mb) * 100,
                "pressure_threshold_mb": self.memory_pressure_threshold,
                "critical_threshold_mb": self.critical_threshold,
            },
            "pool_stats": pool_stats,
            "tracker_stats": {
                "active_operations": len(self.memory_tracker._active_operations),
                "total_allocations": self.stats.total_allocations,
                "gc_runs": self.stats.gc_runs,
                "memory_growth_rate_mb_per_min": self.memory_tracker.get_memory_growth_rate(),
            },
            "potential_leaks": len(self.memory_tracker.detect_potential_leaks()),
            "optimization_recommendations": self._get_optimization_recommendations(),
        }

    def _get_optimization_recommendations(self) -> list[str]:
        """Get memory optimization recommendations."""
        recommendations = []
        current_usage = self.get_current_usage_mb()

        if current_usage > self.memory_pressure_threshold:
            recommendations.append(
                f"High GPU memory usage: {current_usage:.1f}MB / {self.max_gpu_memory_mb:.1f}MB"
            )

        potential_leaks = self.memory_tracker.detect_potential_leaks()
        if potential_leaks:
            recommendations.append(
                f"Potential memory leaks detected: {len(potential_leaks)} operations"
            )

        growth_rate = self.memory_tracker.get_memory_growth_rate()
        if growth_rate > 10:  # More than 10MB/min growth
            recommendations.append(f"High memory growth rate: {growth_rate:.1f}MB/min")

        pool_stats = self.memory_pool.get_pool_stats()
        if pool_stats["hit_rate"] < 0.3:  # Less than 30% hit rate
            recommendations.append(
                f"Low pool hit rate: {pool_stats['hit_rate']:.1%} - consider adjusting pool size"
            )

        if not recommendations:
            recommendations.append("GPU memory usage is optimal")

        return recommendations

    def shutdown(self):
        """Shutdown GPU memory manager."""
        if self._monitoring_active:
            asyncio.create_task(self.stop_monitoring())

        self.cleanup_memory(force=True)
        self.memory_pool.clear_pools()
        logger.info("M4 Pro GPU Memory Manager shutdown complete")


# Global instance
_gpu_memory_manager: M4ProGPUMemoryManager | None = None


def get_gpu_memory_manager() -> M4ProGPUMemoryManager:
    """Get the global GPU memory manager."""
    global _gpu_memory_manager
    if _gpu_memory_manager is None:
        _gpu_memory_manager = M4ProGPUMemoryManager()
    return _gpu_memory_manager


# Convenience functions
def allocate_gpu_operation(operation_name: str, estimated_mb: float, priority: int = 5):
    """Allocate GPU memory for an operation."""
    return get_gpu_memory_manager().allocate_operation(
        operation_name, estimated_mb, priority
    )


def create_optimized_array(shape: tuple[int, ...], dtype=None, init_fn=None):
    """Create GPU array with pool optimization."""
    return get_gpu_memory_manager().create_optimized_array(shape, dtype, init_fn)


def cleanup_gpu_memory(force: bool = False) -> float:
    """Clean up GPU memory."""
    return get_gpu_memory_manager().cleanup_memory(force)


def get_gpu_memory_report() -> dict[str, Any]:
    """Get GPU memory usage report."""
    return get_gpu_memory_manager().get_memory_report()


if __name__ == "__main__":
    # Test the GPU memory manager
    print("Testing M4 Pro GPU Memory Manager...")

    manager = get_gpu_memory_manager()

    if MLX_AVAILABLE:
        # Test memory allocation
        with allocate_gpu_operation("test_operation", 50.0) as op_id:
            print(f"Allocated operation: {op_id}")

            # Create some arrays
            arrays = []
            for _i in range(10):
                arr = create_optimized_array((100, 100))
                if arr is not None:
                    arrays.append(arr)

            print(f"Created {len(arrays)} arrays")

        # Test cleanup
        freed = cleanup_gpu_memory(force=True)
        print(f"Cleanup freed: {freed:.1f}MB")

        # Get report
        report = get_gpu_memory_report()
        print("\nGPU Memory Report:")
        print(f"Allocated: {report['memory_stats']['allocated_mb']:.1f}MB")
        print(f"Usage: {report['memory_stats']['usage_percent']:.1f}%")
        print(f"Recommendations: {', '.join(report['optimization_recommendations'])}")
    else:
        print("MLX not available - GPU memory management disabled")

    print("Test completed successfully!")
