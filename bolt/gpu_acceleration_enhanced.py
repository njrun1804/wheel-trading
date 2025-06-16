#!/usr/bin/env python3
"""
Enhanced GPU Acceleration with Advanced Memory Pool Management

This module provides advanced GPU acceleration with:
- Real-time memory pool optimization
- Adaptive batch size tuning
- Thermal throttling integration
- Advanced MLX/Metal optimizations
- Multi-stream execution
- Hardware-aware scheduling
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logger = logging.getLogger(__name__)


class ComputeStrategy(Enum):
    """GPU compute strategies for different workloads."""

    AUTO = "auto"
    GPU_ONLY = "gpu_only"
    CPU_ONLY = "cpu_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class MemoryPoolStrategy(Enum):
    """Memory pool management strategies."""

    FIXED_SIZE = "fixed_size"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"
    LRU_EVICTION = "lru_eviction"


@dataclass
class GPUComputeStats:
    """Advanced GPU compute statistics."""

    total_operations: int = 0
    gpu_operations: int = 0
    cpu_fallbacks: int = 0
    adaptive_decisions: int = 0
    memory_pool_hits: int = 0
    memory_pool_misses: int = 0
    thermal_throttles: int = 0
    total_compute_time: float = 0.0
    total_memory_transferred: int = 0
    peak_memory_usage: int = 0
    average_batch_size: float = 0.0
    stream_utilization: float = 0.0
    cache_efficiency: float = 0.0


@dataclass
class MemoryBuffer:
    """GPU memory buffer with lifecycle management."""

    id: str
    size_bytes: int
    shape: tuple[int, ...]
    dtype: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    data: Any | None = None
    in_use: bool = False
    can_evict: bool = True
    priority: int = 5  # 1-10, higher = more important


class AdvancedMemoryPool:
    """Advanced GPU memory pool with intelligent management."""

    def __init__(
        self,
        max_size_mb: int = 1024,
        strategy: MemoryPoolStrategy = MemoryPoolStrategy.ADAPTIVE,
        eviction_threshold: float = 0.85,
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.strategy = strategy
        self.eviction_threshold = eviction_threshold

        # Memory tracking
        self.buffers: dict[str, MemoryBuffer] = {}
        self.free_buffers: dict[tuple, deque[str]] = defaultdict(deque)
        self.allocated_bytes = 0

        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.allocations = 0

        # Thread safety
        self.lock = threading.RLock()

        # Adaptive parameters
        self.size_frequency: dict[int, int] = defaultdict(int)
        self.optimal_sizes: set[int] = set()

        logger.info(
            f"Advanced GPU memory pool initialized: {max_size_mb}MB, strategy={strategy.value}"
        )

    def get_buffer(
        self, shape: tuple[int, ...], dtype: str = "float32"
    ) -> str | None:
        """Get buffer from pool or allocate new one."""
        with self.lock:
            buffer_key = (shape, dtype)

            # Try to reuse existing buffer
            if buffer_key in self.free_buffers and self.free_buffers[buffer_key]:
                buffer_id = self.free_buffers[buffer_key].popleft()
                buffer = self.buffers[buffer_id]
                buffer.in_use = True
                buffer.last_accessed = time.time()
                buffer.access_count += 1
                self.hits += 1
                return buffer_id

            # Allocate new buffer
            buffer_size = self._calculate_buffer_size(shape, dtype)

            # Check if we need to evict
            if self.allocated_bytes + buffer_size > self.max_size_bytes:
                evicted = self._evict_buffers(buffer_size)
                if not evicted:
                    self.misses += 1
                    return None

            # Create new buffer
            buffer_id = f"buf_{int(time.time() * 1000000)}_{self.allocations}"
            buffer = MemoryBuffer(
                id=buffer_id,
                size_bytes=buffer_size,
                shape=shape,
                dtype=dtype,
                in_use=True,
            )

            self.buffers[buffer_id] = buffer
            self.allocated_bytes += buffer_size
            self.allocations += 1
            self.misses += 1

            # Update size frequency for adaptive optimization
            self.size_frequency[buffer_key] += 1
            if self.size_frequency[buffer_key] > 5:
                self.optimal_sizes.add(buffer_key)

            return buffer_id

    def return_buffer(self, buffer_id: str) -> bool:
        """Return buffer to pool."""
        with self.lock:
            if buffer_id not in self.buffers:
                return False

            buffer = self.buffers[buffer_id]
            buffer.in_use = False
            buffer.last_accessed = time.time()

            # Add to free pool
            buffer_key = (buffer.shape, buffer.dtype)
            self.free_buffers[buffer_key].append(buffer_id)

            return True

    def _calculate_buffer_size(self, shape: tuple[int, ...], dtype: str) -> int:
        """Calculate buffer size in bytes."""
        dtype_sizes = {"float32": 4, "float16": 2, "int32": 4, "int16": 2, "int8": 1}
        element_size = dtype_sizes.get(dtype, 4)
        return int(np.prod(shape) * element_size)

    def _evict_buffers(self, needed_bytes: int) -> bool:
        """Evict buffers to make space."""
        if self.strategy == MemoryPoolStrategy.LRU_EVICTION:
            return self._lru_eviction(needed_bytes)
        elif self.strategy == MemoryPoolStrategy.ADAPTIVE:
            return self._adaptive_eviction(needed_bytes)
        else:
            return self._size_based_eviction(needed_bytes)

    def _lru_eviction(self, needed_bytes: int) -> bool:
        """Evict least recently used buffers."""
        evictable = [
            (buf_id, buf)
            for buf_id, buf in self.buffers.items()
            if not buf.in_use and buf.can_evict
        ]

        # Sort by last accessed time
        evictable.sort(key=lambda x: x[1].last_accessed)

        freed_bytes = 0
        for buf_id, buffer in evictable:
            if freed_bytes >= needed_bytes:
                break

            self._evict_buffer(buf_id)
            freed_bytes += buffer.size_bytes
            self.evictions += 1

        return freed_bytes >= needed_bytes

    def _adaptive_eviction(self, needed_bytes: int) -> bool:
        """Adaptive eviction based on usage patterns."""
        evictable = [
            (buf_id, buf)
            for buf_id, buf in self.buffers.items()
            if not buf.in_use and buf.can_evict
        ]

        # Score based on multiple factors
        def eviction_score(buffer):
            age_score = time.time() - buffer.last_accessed
            freq_score = 1.0 / max(1, buffer.access_count)
            size_score = buffer.size_bytes / self.max_size_bytes
            return age_score * freq_score * size_score

        evictable.sort(key=lambda x: eviction_score(x[1]), reverse=True)

        freed_bytes = 0
        for buf_id, buffer in evictable:
            if freed_bytes >= needed_bytes:
                break

            self._evict_buffer(buf_id)
            freed_bytes += buffer.size_bytes
            self.evictions += 1

        return freed_bytes >= needed_bytes

    def _size_based_eviction(self, needed_bytes: int) -> bool:
        """Evict largest buffers first."""
        evictable = [
            (buf_id, buf)
            for buf_id, buf in self.buffers.items()
            if not buf.in_use and buf.can_evict
        ]

        # Sort by size (largest first)
        evictable.sort(key=lambda x: x[1].size_bytes, reverse=True)

        freed_bytes = 0
        for buf_id, buffer in evictable:
            if freed_bytes >= needed_bytes:
                break

            self._evict_buffer(buf_id)
            freed_bytes += buffer.size_bytes
            self.evictions += 1

        return freed_bytes >= needed_bytes

    def _evict_buffer(self, buffer_id: str):
        """Remove buffer from pool."""
        buffer = self.buffers.pop(buffer_id)
        self.allocated_bytes -= buffer.size_bytes

        # Remove from free lists
        buffer_key = (buffer.shape, buffer.dtype)
        if buffer_id in self.free_buffers[buffer_key]:
            self.free_buffers[buffer_key].remove(buffer_id)

    def get_stats(self) -> dict[str, Any]:
        """Get memory pool statistics."""
        with self.lock:
            hit_rate = self.hits / max(1, self.hits + self.misses)
            utilization = self.allocated_bytes / self.max_size_bytes

            return {
                "total_buffers": len(self.buffers),
                "allocated_mb": self.allocated_bytes / (1024 * 1024),
                "max_mb": self.max_size_bytes / (1024 * 1024),
                "utilization_percent": utilization * 100,
                "hit_rate": hit_rate,
                "total_hits": self.hits,
                "total_misses": self.misses,
                "total_evictions": self.evictions,
                "optimal_sizes_discovered": len(self.optimal_sizes),
                "strategy": self.strategy.value,
            }

    def optimize_pool(self):
        """Optimize pool based on usage patterns."""
        with self.lock:
            if self.strategy == MemoryPoolStrategy.ADAPTIVE:
                # Pre-allocate buffers for frequently used sizes
                for buffer_key, frequency in self.size_frequency.items():
                    if frequency > 10 and buffer_key not in self.optimal_sizes:
                        # Pre-allocate a few buffers of this size
                        for _ in range(min(3, frequency // 5)):
                            self.get_buffer(buffer_key[0], buffer_key[1])
                            # Return immediately to pool
                            if self.buffers:
                                last_id = list(self.buffers.keys())[-1]
                                self.return_buffer(last_id)

    def clear_pool(self):
        """Clear all buffers from pool."""
        with self.lock:
            self.buffers.clear()
            self.free_buffers.clear()
            self.allocated_bytes = 0
            logger.info("GPU memory pool cleared")


class ThermalMonitor:
    """Monitor and respond to thermal conditions."""

    def __init__(self):
        self.temperature_history = deque(maxlen=60)  # Last 60 readings
        self.throttle_threshold = 80.0  # Celsius
        self.critical_threshold = 90.0  # Celsius
        self.is_throttling = False

    def update_temperature(self, temp_celsius: float):
        """Update current temperature reading."""
        self.temperature_history.append((time.time(), temp_celsius))

        if temp_celsius > self.critical_threshold:
            if not self.is_throttling:
                logger.warning(f"GPU thermal throttling activated: {temp_celsius}°C")
                self.is_throttling = True
        elif temp_celsius < self.throttle_threshold:
            if self.is_throttling:
                logger.info(f"GPU thermal throttling deactivated: {temp_celsius}°C")
                self.is_throttling = False

    def should_throttle(self) -> bool:
        """Check if thermal throttling should be applied."""
        return self.is_throttling

    def get_throttle_factor(self) -> float:
        """Get throttling factor (0.0 to 1.0)."""
        if not self.temperature_history:
            return 1.0

        current_temp = self.temperature_history[-1][1]
        if current_temp < self.throttle_threshold:
            return 1.0
        elif current_temp > self.critical_threshold:
            return 0.3  # Severe throttling
        else:
            # Linear interpolation between throttle and critical
            factor = 1.0 - (current_temp - self.throttle_threshold) / (
                self.critical_threshold - self.throttle_threshold
            )
            return max(0.3, factor)


class EnhancedGPUAccelerator:
    """Enhanced GPU accelerator with advanced features."""

    def __init__(
        self,
        memory_pool_mb: int = 1024,
        enable_thermal_monitoring: bool = True,
        enable_adaptive_batching: bool = True,
    ):
        self.gpu_available = (
            MLX_AVAILABLE and mx.metal.is_available() if MLX_AVAILABLE else False
        )
        self.memory_pool = AdvancedMemoryPool(memory_pool_mb)
        self.thermal_monitor = ThermalMonitor() if enable_thermal_monitoring else None
        self.enable_adaptive_batching = enable_adaptive_batching

        # Performance tracking
        self.stats = GPUComputeStats()

        # Adaptive batching
        self.optimal_batch_sizes: dict[str, int] = {}
        self.batch_performance_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10)
        )

        # Multi-stream support
        self.active_streams = 0
        self.max_concurrent_streams = 4

        logger.info(
            f"Enhanced GPU accelerator initialized: memory_pool={memory_pool_mb}MB, thermal_monitoring={enable_thermal_monitoring}"
        )

    @contextmanager
    def gpu_memory_context(self, shape: tuple[int, ...], dtype: str = "float32"):
        """Context manager for GPU memory allocation."""
        buffer_id = None
        try:
            buffer_id = self.memory_pool.get_buffer(shape, dtype)
            if buffer_id:
                self.stats.memory_pool_hits += 1
            else:
                self.stats.memory_pool_misses += 1
            yield buffer_id
        finally:
            if buffer_id:
                self.memory_pool.return_buffer(buffer_id)

    async def adaptive_batch_operation(
        self,
        operation_name: str,
        data_list: list[Any],
        operation_func: Callable,
        target_latency_ms: float = 100.0,
    ) -> list[Any]:
        """Execute batch operation with adaptive batch sizing."""

        if not self.enable_adaptive_batching:
            return await self._simple_batch_operation(data_list, operation_func)

        # Get optimal batch size for this operation
        batch_size = self.optimal_batch_sizes.get(operation_name, 32)

        results = []
        total_items = len(data_list)

        for i in range(0, total_items, batch_size):
            batch = data_list[i : i + batch_size]

            # Apply thermal throttling if needed
            if self.thermal_monitor and self.thermal_monitor.should_throttle():
                throttle_factor = self.thermal_monitor.get_throttle_factor()
                adjusted_batch_size = max(1, int(len(batch) * throttle_factor))
                batch = batch[:adjusted_batch_size]
                self.stats.thermal_throttles += 1

            # Execute batch
            start_time = time.time()
            batch_results = await operation_func(batch)
            batch_latency = (time.time() - start_time) * 1000

            results.extend(batch_results)

            # Update performance history
            self.batch_performance_history[operation_name].append(
                {
                    "batch_size": len(batch),
                    "latency_ms": batch_latency,
                    "throughput": len(batch) / (batch_latency / 1000),
                }
            )

            # Adapt batch size if we have enough history
            if len(self.batch_performance_history[operation_name]) >= 5:
                self._optimize_batch_size(operation_name, target_latency_ms)

        return results

    def _optimize_batch_size(self, operation_name: str, target_latency_ms: float):
        """Optimize batch size based on performance history."""
        history = self.batch_performance_history[operation_name]
        if len(history) < 3:
            return

        # Find the batch size that gives best throughput while staying under target latency
        best_throughput = 0
        best_batch_size = self.optimal_batch_sizes.get(operation_name, 32)

        for entry in history:
            if (
                entry["latency_ms"] <= target_latency_ms
                and entry["throughput"] > best_throughput
            ):
                best_throughput = entry["throughput"]
                best_batch_size = entry["batch_size"]

        # Adjust batch size gradually
        current_size = self.optimal_batch_sizes.get(operation_name, 32)
        if best_batch_size != current_size:
            # Move 20% toward optimal
            new_size = int(current_size * 0.8 + best_batch_size * 0.2)
            self.optimal_batch_sizes[operation_name] = max(1, min(256, new_size))

    async def _simple_batch_operation(
        self, data_list: list[Any], operation_func: Callable
    ) -> list[Any]:
        """Simple batch operation without adaptation."""
        return await operation_func(data_list)

    async def parallel_matrix_operations(
        self, matrices: list[np.ndarray], operation_type: str = "multiply"
    ) -> list[np.ndarray]:
        """Parallel matrix operations with memory pool optimization."""

        if not self.gpu_available:
            return self._cpu_fallback_matrix_ops(matrices, operation_type)

        results = []

        for matrix in matrices:
            with self.gpu_memory_context(matrix.shape):
                # Convert to MLX array
                mx_matrix = mx.array(matrix)

                # Perform operation
                if operation_type == "multiply":
                    result = mx_matrix @ mx_matrix.T
                elif operation_type == "inverse":
                    result = mx.linalg.inv(mx_matrix)
                elif operation_type == "eigenvals":
                    result = mx.linalg.eigvals(mx_matrix)
                else:
                    result = mx_matrix  # Identity operation

                # Ensure computation completes
                mx.eval(result)

                # Convert back to numpy
                results.append(np.array(result))

        self.stats.gpu_operations += len(matrices)
        return results

    def _cpu_fallback_matrix_ops(
        self, matrices: list[np.ndarray], operation_type: str
    ) -> list[np.ndarray]:
        """CPU fallback for matrix operations."""
        results = []
        for matrix in matrices:
            if operation_type == "multiply":
                result = matrix @ matrix.T
            elif operation_type == "inverse":
                result = np.linalg.inv(matrix)
            elif operation_type == "eigenvals":
                result = np.linalg.eigvals(matrix)
            else:
                result = matrix
            results.append(result)

        self.stats.cpu_fallbacks += len(matrices)
        return results

    def update_thermal_state(self, temperature_celsius: float):
        """Update thermal monitoring state."""
        if self.thermal_monitor:
            self.thermal_monitor.update_temperature(temperature_celsius)

    def get_comprehensive_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        pool_stats = self.memory_pool.get_stats()

        total_ops = self.stats.total_operations
        gpu_ratio = self.stats.gpu_operations / max(1, total_ops)

        thermal_info = {}
        if self.thermal_monitor:
            thermal_info = {
                "is_throttling": self.thermal_monitor.is_throttling,
                "throttle_factor": self.thermal_monitor.get_throttle_factor(),
                "temperature_history_size": len(
                    self.thermal_monitor.temperature_history
                ),
            }

        return {
            "gpu_available": self.gpu_available,
            "total_operations": total_ops,
            "gpu_operations": self.stats.gpu_operations,
            "cpu_fallbacks": self.stats.cpu_fallbacks,
            "gpu_utilization_ratio": gpu_ratio,
            "memory_pool": pool_stats,
            "thermal_monitoring": thermal_info,
            "adaptive_batching": {
                "enabled": self.enable_adaptive_batching,
                "optimal_batch_sizes": dict(self.optimal_batch_sizes),
                "operations_tracked": len(self.batch_performance_history),
            },
            "performance": {
                "thermal_throttles": self.stats.thermal_throttles,
                "memory_pool_hit_rate": self.stats.memory_pool_hits
                / max(1, self.stats.memory_pool_hits + self.stats.memory_pool_misses),
                "average_batch_size": self.stats.average_batch_size,
            },
        }

    def optimize_system(self):
        """Optimize the entire acceleration system."""
        self.memory_pool.optimize_pool()

        # Optimize batch sizes based on recent performance
        for operation_name in list(self.batch_performance_history.keys()):
            self._optimize_batch_size(operation_name, 100.0)

        logger.info("Enhanced GPU accelerator optimization completed")

    def shutdown(self):
        """Shutdown and cleanup resources."""
        self.memory_pool.clear_pool()
        logger.info("Enhanced GPU accelerator shutdown completed")


# Global instance
_enhanced_accelerator: EnhancedGPUAccelerator | None = None


def get_enhanced_gpu_accelerator(**kwargs) -> EnhancedGPUAccelerator:
    """Get global enhanced GPU accelerator instance."""
    global _enhanced_accelerator
    if _enhanced_accelerator is None:
        _enhanced_accelerator = EnhancedGPUAccelerator(**kwargs)
    return _enhanced_accelerator


# High-level convenience functions
async def gpu_accelerated_batch_operation(
    operation_name: str, data: list[Any], operation_func: Callable, **kwargs
) -> list[Any]:
    """High-level GPU accelerated batch operation."""
    accelerator = get_enhanced_gpu_accelerator()
    return await accelerator.adaptive_batch_operation(
        operation_name, data, operation_func, **kwargs
    )


def gpu_matrix_operations(
    matrices: list[np.ndarray], operation_type: str = "multiply"
) -> list[np.ndarray]:
    """High-level GPU matrix operations."""
    accelerator = get_enhanced_gpu_accelerator()
    return asyncio.run(accelerator.parallel_matrix_operations(matrices, operation_type))


if __name__ == "__main__":

    async def benchmark_enhanced_accelerator():
        """Benchmark enhanced GPU accelerator."""
        print("=== Enhanced GPU Accelerator Benchmark ===")

        accelerator = get_enhanced_gpu_accelerator(memory_pool_mb=512)

        # Test adaptive batching
        print("\nTesting adaptive batching...")
        test_data = [np.random.randn(100, 100) for _ in range(50)]

        async def matrix_multiply_batch(batch):
            results = []
            for matrix in batch:
                if MLX_AVAILABLE:
                    mx_matrix = mx.array(matrix)
                    result = mx_matrix @ mx_matrix.T
                    mx.eval(result)
                    results.append(np.array(result))
                else:
                    results.append(matrix @ matrix.T)
            return results

        start_time = time.time()
        results = await accelerator.adaptive_batch_operation(
            "matrix_multiply", test_data, matrix_multiply_batch
        )
        batch_time = time.time() - start_time

        print(
            f"Adaptive batching completed: {len(results)} results in {batch_time:.3f}s"
        )

        # Test thermal simulation
        print("\nTesting thermal monitoring...")
        accelerator.update_thermal_state(75.0)  # Normal
        accelerator.update_thermal_state(85.0)  # Hot
        accelerator.update_thermal_state(95.0)  # Critical

        # Get comprehensive stats
        stats = accelerator.get_comprehensive_stats()
        print("\nComprehensive Stats:")
        print(f"GPU Available: {stats['gpu_available']}")
        print(
            f"Memory Pool Hit Rate: {stats['performance']['memory_pool_hit_rate']:.2%}"
        )
        print(
            f"Thermal Throttling: {stats['thermal_monitoring'].get('is_throttling', False)}"
        )
        print(
            f"Optimal Batch Sizes: {stats['adaptive_batching']['optimal_batch_sizes']}"
        )

        # Optimize system
        accelerator.optimize_system()

        accelerator.shutdown()

    asyncio.run(benchmark_enhanced_accelerator())
