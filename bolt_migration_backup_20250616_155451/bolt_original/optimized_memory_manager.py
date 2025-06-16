#!/usr/bin/env python3
"""
Optimized Memory Manager for Bolt System - M4 Pro Optimizations
Target: Reduce memory usage from 9GB to under 4GB

Key optimizations:
1. Lazy loading with weak references
2. Memory pressure-aware eviction
3. Compressed caching strategies
4. GPU memory pool management
5. Database connection optimization
6. Smart garbage collection
"""

import gc
import gzip
import logging
import pickle
import threading
import time
import weakref
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import psutil

# Try to import MLX for GPU memory management
try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logger = logging.getLogger(__name__)

T = TypeVar("T")

# M4 Pro Memory Configuration (24GB total, targeting 4GB max)
M4_PRO_CONFIG = {
    "total_memory_gb": 24,
    "target_max_allocation_gb": 4.0,  # Reduced from 18GB to 4GB
    "memory_pressure_threshold": 0.75,  # 75% of 4GB = 3GB
    "critical_threshold": 0.90,  # 90% of 4GB = 3.6GB
    "component_budgets": {
        "agents": 0.30,  # 30% = 1.2GB (reduced from 3.06GB)
        "einstein": 0.20,  # 20% = 0.8GB (reduced from 1.44GB)
        "database": 0.25,  # 25% = 1.0GB (reduced from 9GB)
        "gpu": 0.15,  # 15% = 0.6GB
        "cache": 0.08,  # 8% = 0.32GB (reduced from 1.8GB)
        "other": 0.02,  # 2% = 0.08GB (buffer)
    },
}


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    component: str
    allocated_mb: float
    peak_mb: float
    budget_mb: float
    usage_percent: float
    evictions: int
    gc_runs: int
    compress_ratio: float
    last_cleanup: float


@dataclass
class MemoryPressureEvent:
    """Memory pressure event data."""

    timestamp: float
    component: str
    usage_mb: float
    threshold_mb: float
    action_taken: str
    freed_mb: float


class CompressedWeakCache(Generic[T]):
    """Memory-efficient cache with compression and weak references."""

    def __init__(self, max_items: int = 100, compress_threshold: int = 512):
        self.max_items = max_items
        self.compress_threshold = compress_threshold
        self._cache: OrderedDict[str, T | bytes] = OrderedDict()
        self._metadata: dict[str, dict] = {}
        self._weak_refs: dict[str, weakref.ref] = {}
        self._lock = threading.RLock()

        # Stats
        self.hits = 0
        self.misses = 0
        self.compressions = 0
        self.evictions = 0

    def get(self, key: str) -> T | None:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                # Check weak references
                if key in self._weak_refs:
                    ref = self._weak_refs[key]
                    value = ref()
                    if value is not None:
                        self.hits += 1
                        return value
                    else:
                        del self._weak_refs[key]

                self.misses += 1
                return None

            value = self._cache[key]
            metadata = self._metadata.get(key, {})

            # Move to end (LRU)
            self._cache.move_to_end(key)

            # Decompress if needed
            if metadata.get("compressed", False):
                try:
                    value = pickle.loads(gzip.decompress(value))
                    # Store decompressed value temporarily as weak ref
                    self._weak_refs[key] = weakref.ref(value)
                except Exception as e:
                    logger.warning(f"Decompression failed for {key}: {e}")
                    del self._cache[key]
                    del self._metadata[key]
                    self.misses += 1
                    return None

            self.hits += 1
            return value

    def put(self, key: str, value: T) -> None:
        """Store item in cache."""
        with self._lock:
            # Check if we should compress
            compressed = False
            stored_value = value

            try:
                serialized = pickle.dumps(value)
                if len(serialized) > self.compress_threshold:
                    compressed_data = gzip.compress(serialized)
                    if (
                        len(compressed_data) < len(serialized) * 0.8
                    ):  # Only if 20% savings
                        stored_value = compressed_data
                        compressed = True
                        self.compressions += 1

                        # Also store weak reference for fast access
                        self._weak_refs[key] = weakref.ref(value)
            except Exception:
                pass  # Use original value

            # Store in cache
            self._cache[key] = stored_value
            self._metadata[key] = {
                "compressed": compressed,
                "size": len(serialized) if "serialized" in locals() else 0,
                "timestamp": time.time(),
            }

            # Evict if necessary
            while len(self._cache) > self.max_items:
                self._evict_lru()

    def _evict_lru(self):
        """Evict least recently used item."""
        if self._cache:
            key, _ = self._cache.popitem(last=False)
            self._metadata.pop(key, None)
            self._weak_refs.pop(key, None)
            self.evictions += 1

    def clear(self):
        """Clear all caches."""
        with self._lock:
            self._cache.clear()
            self._metadata.clear()
            self._weak_refs.clear()

    def get_memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        with self._lock:
            total_bytes = 0
            for _key, value in self._cache.items():
                if isinstance(value, bytes):
                    total_bytes += len(value)
                else:
                    try:
                        total_bytes += len(pickle.dumps(value))
                    except (TypeError, pickle.PickleError) as e:
                        logger.debug(f"Could not serialize cache value: {e}")
                        total_bytes += 1024  # Rough estimate
            return total_bytes / (1024 * 1024)


class LazyLoader:
    """Lazy loading wrapper to defer expensive operations."""

    def __init__(self, loader_func: Callable, *args, **kwargs):
        self.loader_func = loader_func
        self.args = args
        self.kwargs = kwargs
        self._loaded = False
        self._value = None
        self._lock = threading.Lock()

    def get(self):
        """Get the loaded value."""
        if not self._loaded:
            with self._lock:
                if not self._loaded:  # Double-check locking
                    self._value = self.loader_func(*self.args, **self.kwargs)
                    self._loaded = True
        return self._value

    def is_loaded(self) -> bool:
        """Check if value is loaded."""
        return self._loaded

    def unload(self):
        """Unload to free memory."""
        with self._lock:
            self._value = None
            self._loaded = False


class MemoryPool:
    """Reusable memory pool for arrays and buffers."""

    def __init__(self, max_pool_size: int = 50):
        self.max_pool_size = max_pool_size
        self._pools: dict[int, list[Any]] = defaultdict(list)
        self._lock = threading.Lock()
        self.allocations = 0
        self.reuses = 0

    def get_buffer(self, size: int) -> bytearray:
        """Get buffer from pool or allocate new."""
        # Round to nearest power of 2
        pool_size = 1 << (size - 1).bit_length()

        with self._lock:
            pool = self._pools[pool_size]
            if pool:
                self.reuses += 1
                return pool.pop()
            else:
                self.allocations += 1
                return bytearray(pool_size)

    def return_buffer(self, buffer: Any):
        """Return buffer to pool."""
        size = len(buffer)
        with self._lock:
            pool = self._pools[size]
            if len(pool) < self.max_pool_size:
                pool.append(buffer)

    def clear_pools(self):
        """Clear all pools."""
        with self._lock:
            self._pools.clear()


class DatabaseConnectionPool:
    """Optimized database connection pool with memory limits."""

    def __init__(self, max_connections: int = 3, max_memory_mb: int = 250):
        self.max_connections = max_connections
        self.max_memory_mb = max_memory_mb
        self._connections: list[Any] = []
        self._in_use: set = set()
        self._lock = threading.Lock()
        self._memory_usage_mb = 0

    @contextmanager
    def get_connection(self):
        """Get connection from pool."""
        conn = None
        try:
            with self._lock:
                # Find available connection
                for c in self._connections:
                    if id(c) not in self._in_use:
                        conn = c
                        self._in_use.add(id(c))
                        break

                # Create new if needed and under limit
                if conn is None and len(self._connections) < self.max_connections:
                    conn = self._create_connection()
                    self._connections.append(conn)
                    self._in_use.add(id(conn))

            if conn is None:
                raise RuntimeError("No database connections available")

            yield conn

        finally:
            if conn:
                with self._lock:
                    self._in_use.discard(id(conn))

    def _create_connection(self):
        """Create optimized database connection."""
        import duckdb

        conn = duckdb.connect(":memory:")

        # Configure for low memory usage
        conn.execute("SET memory_limit='200MB'")
        conn.execute("SET threads=2")  # Limit parallel threads
        conn.execute("SET max_memory='200MB'")

        return conn

    def close_all(self):
        """Close all connections."""
        with self._lock:
            for conn in self._connections:
                with suppress(Exception):
                    conn.close()
            self._connections.clear()
            self._in_use.clear()


class OptimizedMemoryManager:
    """Optimized memory manager targeting 4GB total usage."""

    def __init__(self):
        self.config = M4_PRO_CONFIG
        self.target_mb = self.config["target_max_allocation_gb"] * 1024

        # Component pools with reduced budgets
        self.component_budgets = {}
        for component, ratio in self.config["component_budgets"].items():
            self.component_budgets[component] = self.target_mb * ratio

        # Memory management components
        self.caches: dict[str, CompressedWeakCache] = {}
        self.lazy_loaders: dict[str, LazyLoader] = {}
        self.memory_pools: dict[str, MemoryPool] = defaultdict(MemoryPool)
        self.db_pool = DatabaseConnectionPool()

        # Stats and monitoring
        self.stats: dict[str, MemoryStats] = {}
        self.pressure_events: list[MemoryPressureEvent] = []
        self.last_gc = time.time()
        self.gc_interval = 30  # Run GC every 30 seconds

        # Locks
        self.main_lock = threading.RLock()

        # Start monitoring
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        logger.info(
            f"Optimized Memory Manager initialized (target: {self.target_mb:.0f}MB)"
        )

    def register_component(
        self, component: str, cache_size: int = 50
    ) -> CompressedWeakCache:
        """Register a component and get its cache."""
        with self.main_lock:
            if component not in self.caches:
                budget_mb = self.component_budgets.get(component, self.target_mb * 0.02)
                # Adjust cache size based on budget
                adjusted_cache_size = min(
                    cache_size, int(budget_mb / 10)
                )  # ~10MB per 100 items

                self.caches[component] = CompressedWeakCache(
                    max_items=adjusted_cache_size
                )
                self.stats[component] = MemoryStats(
                    component=component,
                    allocated_mb=0,
                    peak_mb=0,
                    budget_mb=budget_mb,
                    usage_percent=0,
                    evictions=0,
                    gc_runs=0,
                    compress_ratio=0,
                    last_cleanup=time.time(),
                )

                logger.info(
                    f"Registered component '{component}' with {budget_mb:.1f}MB budget"
                )

            return self.caches[component]

    def create_lazy_loader(
        self, key: str, loader_func: Callable, *args, **kwargs
    ) -> LazyLoader:
        """Create lazy loader for expensive operations."""
        loader = LazyLoader(loader_func, *args, **kwargs)
        self.lazy_loaders[key] = loader
        return loader

    def get_database_connection(self):
        """Get optimized database connection."""
        return self.db_pool.get_connection()

    def allocate_gpu_memory(self, size_mb: float, operation: str) -> str | None:
        """Allocate GPU memory with tracking."""
        if not MLX_AVAILABLE:
            return None

        budget_mb = self.component_budgets.get("gpu", 600)
        current_usage = self._get_gpu_memory_usage()

        if current_usage + size_mb > budget_mb:
            # Try to free GPU memory
            freed = self._cleanup_gpu_memory()
            logger.info(f"GPU memory pressure: freed {freed:.1f}MB")

            current_usage = self._get_gpu_memory_usage()
            if current_usage + size_mb > budget_mb:
                logger.warning(
                    f"GPU allocation denied: {size_mb:.1f}MB would exceed budget"
                )
                return None

        # Create allocation ID for tracking
        alloc_id = f"gpu_{int(time.time() * 1000000)}"
        logger.debug(f"GPU allocated {size_mb:.1f}MB for {operation}")
        return alloc_id

    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if not MLX_AVAILABLE:
            return 0.0
        try:
            return mx.metal.get_active_memory() / (1024 * 1024)
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"Could not get GPU memory usage: {e}")
            return 0.0

    def _cleanup_gpu_memory(self) -> float:
        """Clean up GPU memory."""
        if not MLX_AVAILABLE:
            return 0.0

        initial_usage = self._get_gpu_memory_usage()
        try:
            mx.metal.clear_cache()
            gc.collect()
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"Could not clear GPU cache: {e}")
        final_usage = self._get_gpu_memory_usage()
        return initial_usage - final_usage

    def cleanup_component(self, component: str, aggressive: bool = False) -> float:
        """Clean up memory for a specific component."""
        freed_mb = 0

        with self.main_lock:
            # Clear component cache
            if component in self.caches:
                cache = self.caches[component]
                freed_mb += cache.get_memory_usage_mb()
                cache.clear()

                if component in self.stats:
                    self.stats[component].last_cleanup = time.time()

            # Unload lazy loaders for this component
            component_loaders = [
                k for k in self.lazy_loaders if k.startswith(component)
            ]
            for key in component_loaders:
                loader = self.lazy_loaders[key]
                if loader.is_loaded():
                    loader.unload()
                    freed_mb += 1  # Rough estimate

            # Clear memory pools
            if component in self.memory_pools:
                self.memory_pools[component].clear_pools()
                freed_mb += 5  # Rough estimate

            # Component-specific cleanup
            if component == "gpu":
                freed_mb += self._cleanup_gpu_memory()
            elif component == "database":
                self.db_pool.close_all()
                freed_mb += 50  # Rough estimate

            if aggressive:
                # Force garbage collection
                collected = gc.collect()
                logger.debug(f"Aggressive GC collected {collected} objects")

        logger.info(f"Cleaned up {freed_mb:.1f}MB from {component}")
        return freed_mb

    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        current_mb = self.get_total_memory_usage()
        pressure_threshold = self.target_mb * self.config["memory_pressure_threshold"]
        return current_mb > pressure_threshold

    def handle_memory_pressure(self):
        """Handle memory pressure situation."""
        current_mb = self.get_total_memory_usage()
        target_reduction = current_mb - (
            self.target_mb * 0.7
        )  # Try to get to 70% of target

        logger.warning(
            f"Memory pressure detected: {current_mb:.1f}MB, need to free {target_reduction:.1f}MB"
        )

        freed_total = 0

        # Clean up components in order of priority (least important first)
        cleanup_priority = ["other", "cache", "database", "einstein", "agents", "gpu"]

        for component in cleanup_priority:
            if freed_total >= target_reduction:
                break

            freed = self.cleanup_component(component, aggressive=True)
            freed_total += freed

            # Record pressure event
            event = MemoryPressureEvent(
                timestamp=time.time(),
                component=component,
                usage_mb=current_mb,
                threshold_mb=self.target_mb * self.config["memory_pressure_threshold"],
                action_taken=f"cleanup_{component}",
                freed_mb=freed,
            )
            self.pressure_events.append(event)

        # Final garbage collection
        collected = gc.collect()
        logger.info(
            f"Memory pressure handled: freed {freed_total:.1f}MB, GC collected {collected} objects"
        )

        return freed_total

    def get_total_memory_usage(self) -> float:
        """Get total memory usage across all components."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def get_component_usage(self, component: str) -> float:
        """Get memory usage for a specific component."""
        if component in self.caches:
            return self.caches[component].get_memory_usage_mb()
        return 0.0

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                self.get_total_memory_usage()

                # Update stats
                for component in self.stats:
                    usage_mb = self.get_component_usage(component)
                    stats = self.stats[component]
                    stats.allocated_mb = usage_mb
                    stats.usage_percent = (usage_mb / stats.budget_mb) * 100
                    if usage_mb > stats.peak_mb:
                        stats.peak_mb = usage_mb

                # Check for pressure
                if self.check_memory_pressure():
                    self.handle_memory_pressure()

                # Periodic GC
                if time.time() - self.last_gc > self.gc_interval:
                    collected = gc.collect()
                    self.last_gc = time.time()
                    if collected > 100:
                        logger.debug(f"Periodic GC collected {collected} objects")

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
                time.sleep(30)

    def get_memory_report(self) -> dict[str, Any]:
        """Get comprehensive memory report."""
        current_mb = self.get_total_memory_usage()

        return {
            "timestamp": time.time(),
            "total_usage_mb": current_mb,
            "target_mb": self.target_mb,
            "usage_percent": (current_mb / self.target_mb) * 100,
            "under_pressure": self.check_memory_pressure(),
            "components": {
                name: {
                    "allocated_mb": stats.allocated_mb,
                    "budget_mb": stats.budget_mb,
                    "usage_percent": stats.usage_percent,
                    "peak_mb": stats.peak_mb,
                    "evictions": self.caches[name].evictions
                    if name in self.caches
                    else 0,
                    "compressions": self.caches[name].compressions
                    if name in self.caches
                    else 0,
                }
                for name, stats in self.stats.items()
            },
            "recent_pressure_events": len(
                [e for e in self.pressure_events if time.time() - e.timestamp < 3600]
            ),
            "gpu_usage_mb": self._get_gpu_memory_usage(),
            "system_memory_percent": psutil.virtual_memory().percent,
        }

    def shutdown(self):
        """Shutdown memory manager."""
        self._monitoring_active = False
        self.db_pool.close_all()
        for cache in self.caches.values():
            cache.clear()
        logger.info("Optimized Memory Manager shutdown complete")


# Global instance
_memory_manager: OptimizedMemoryManager | None = None


def get_optimized_memory_manager() -> OptimizedMemoryManager:
    """Get the global optimized memory manager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = OptimizedMemoryManager()
    return _memory_manager


# Convenience functions for components
def register_component_memory(
    component: str, cache_size: int = 50
) -> CompressedWeakCache:
    """Register a component for memory management."""
    return get_optimized_memory_manager().register_component(component, cache_size)


def create_lazy_loader(key: str, loader_func: Callable, *args, **kwargs) -> LazyLoader:
    """Create a lazy loader for expensive operations."""
    return get_optimized_memory_manager().create_lazy_loader(
        key, loader_func, *args, **kwargs
    )


@contextmanager
def optimized_database_connection():
    """Get an optimized database connection."""
    with get_optimized_memory_manager().get_database_connection() as conn:
        yield conn


def allocate_gpu_memory(size_mb: float, operation: str = "operation") -> str | None:
    """Allocate GPU memory with tracking."""
    return get_optimized_memory_manager().allocate_gpu_memory(size_mb, operation)


def cleanup_component_memory(component: str, aggressive: bool = False) -> float:
    """Clean up memory for a component."""
    return get_optimized_memory_manager().cleanup_component(component, aggressive)


def get_memory_report() -> dict[str, Any]:
    """Get current memory usage report."""
    return get_optimized_memory_manager().get_memory_report()


if __name__ == "__main__":
    # Test the optimized memory manager
    print("Testing Optimized Memory Manager...")

    manager = get_optimized_memory_manager()

    # Register some components
    agent_cache = manager.register_component("agents", 100)
    einstein_cache = manager.register_component("einstein", 50)

    # Test caching
    agent_cache.put("test_key", {"data": "x" * 1000})
    result = agent_cache.get("test_key")
    print(f"Cache test: {'PASS' if result else 'FAIL'}")

    # Test lazy loading
    def expensive_operation():
        return [i * i for i in range(10000)]

    lazy = manager.create_lazy_loader("test_loader", expensive_operation)
    print(f"Lazy loader test: {'PASS' if not lazy.is_loaded() else 'FAIL'}")

    data = lazy.get()
    print(
        f"Lazy loader data: {'PASS' if lazy.is_loaded() and len(data) == 10000 else 'FAIL'}"
    )

    # Get report
    report = manager.get_memory_report()
    print("\nMemory Report:")
    print(
        f"Total Usage: {report['total_usage_mb']:.1f}MB / {report['target_mb']:.1f}MB"
    )
    print(f"Usage: {report['usage_percent']:.1f}%")
    print(f"Under Pressure: {report['under_pressure']}")

    for component, stats in report["components"].items():
        print(
            f"  {component}: {stats['allocated_mb']:.1f}MB / {stats['budget_mb']:.1f}MB ({stats['usage_percent']:.1f}%)"
        )

    print("Test completed successfully!")
