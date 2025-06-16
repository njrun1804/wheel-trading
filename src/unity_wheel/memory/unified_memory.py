#!/usr/bin/env python3
"""
Unified Memory Management System
Consolidates 474 memory functions into a cohesive API
"""

import asyncio
import logging
import threading
import time
import weakref
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Unified memory statistics"""

    total_memory: int
    available_memory: int
    used_memory: int
    memory_percent: float
    swap_total: int
    swap_used: int
    gpu_memory: int | None = None
    buffer_memory: int | None = None
    cache_memory: int | None = None


@dataclass
class CacheStats:
    """Cache performance statistics"""

    hits: int = 0
    misses: int = 0
    size: int = 0
    max_size: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class UnifiedMemoryPool:
    """Consolidated memory pool management"""

    def __init__(self, pool_size: int = 10, max_overflow: int = 5):
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self._pool = asyncio.Queue(maxsize=pool_size)
        self._overflow_count = 0
        self._created_count = 0
        self._lock = asyncio.Lock()
        self._stats = {"created": 0, "reused": 0, "overflow": 0}

    async def get_connection(self):
        """Get a connection from the pool"""
        try:
            # Try to get from pool first
            return await asyncio.wait_for(self._pool.get(), timeout=0.1)
        except TimeoutError:
            # Pool is empty, create new connection
            async with self._lock:
                if self._overflow_count < self.max_overflow:
                    self._overflow_count += 1
                    self._stats["overflow"] += 1
                    return self._create_connection()
                else:
                    # Wait for a connection to be returned
                    return await self._pool.get()

    async def return_connection(self, conn):
        """Return connection to pool"""
        try:
            await asyncio.wait_for(self._pool.put(conn), timeout=0.1)
            self._stats["reused"] += 1
        except TimeoutError:
            # Pool is full, discard connection
            async with self._lock:
                if self._overflow_count > 0:
                    self._overflow_count -= 1

    def _create_connection(self):
        """Create new connection - to be overridden"""
        self._created_count += 1
        self._stats["created"] += 1
        return f"connection_{self._created_count}"

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics"""
        return {
            "pool_size": self.pool_size,
            "current_size": self._pool.qsize(),
            "overflow_count": self._overflow_count,
            "stats": self._stats.copy(),
        }


class UnifiedCache:
    """High-performance unified cache with LRU eviction"""

    def __init__(self, max_size: int = 1000, ttl: int | None = None):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: dict[str, Any] = {}
        self._access_times: dict[str, float] = {}
        self._stats = CacheStats(max_size=max_size)
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        """Get item from cache"""
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None

            # Check TTL
            if self.ttl and time.time() - self._access_times[key] > self.ttl:
                self._evict_key(key)
                self._stats.misses += 1
                return None

            # Update access time and return value
            self._access_times[key] = time.time()
            self._stats.hits += 1
            return self._cache[key]

    def put(self, key: str, value: Any):
        """Put item in cache"""
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()

            self._cache[key] = value
            self._access_times[key] = time.time()
            self._stats.size = len(self._cache)

    def _evict_lru(self):
        """Evict least recently used item"""
        if not self._access_times:
            return

        lru_key = min(self._access_times.keys(), key=self._access_times.get)
        self._evict_key(lru_key)

    def _evict_key(self, key: str):
        """Evict specific key"""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._stats.evictions += 1
        self._stats.size = len(self._cache)

    def clear(self):
        """Clear all cached items"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._stats.size = 0

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self._stats


class MemoryMonitor:
    """Unified memory monitoring and pressure detection"""

    def __init__(self, check_interval: float = 5.0, pressure_threshold: float = 0.85):
        self.check_interval = check_interval
        self.pressure_threshold = pressure_threshold
        self._running = False
        self._callbacks: list[Callable[[MemoryStats], None]] = []
        self._last_stats: MemoryStats | None = None

    def add_pressure_callback(self, callback: Callable[[MemoryStats], None]):
        """Add callback for memory pressure events"""
        self._callbacks.append(callback)

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()

        stats = MemoryStats(
            total_memory=vm.total,
            available_memory=vm.available,
            used_memory=vm.used,
            memory_percent=vm.percent,
            swap_total=swap.total,
            swap_used=swap.used,
        )

        # Add GPU memory if available
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                stats.gpu_memory = sum(gpu.memoryUsed for gpu in gpus)
        except ImportError:
            pass

        self._last_stats = stats
        return stats

    def is_under_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        stats = self.get_memory_stats()
        return stats.memory_percent / 100.0 > self.pressure_threshold

    async def start_monitoring(self):
        """Start background memory monitoring"""
        self._running = True
        while self._running:
            try:
                stats = self.get_memory_stats()

                # Check for pressure and notify callbacks
                if stats.memory_percent / 100.0 > self.pressure_threshold:
                    for callback in self._callbacks:
                        try:
                            callback(stats)
                        except Exception as e:
                            logger.error(f"Memory pressure callback failed: {e}")

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(self.check_interval)

    def stop_monitoring(self):
        """Stop memory monitoring"""
        self._running = False


class MemoryOptimizer:
    """Unified memory optimization and cleanup"""

    def __init__(self):
        self._cleanup_functions: list[Callable[[], None]] = []
        self._weak_refs: weakref.WeakSet = weakref.WeakSet()

    def register_cleanup(self, cleanup_func: Callable[[], None]):
        """Register cleanup function"""
        self._cleanup_functions.append(cleanup_func)

    def register_object(self, obj: Any):
        """Register object for weak reference tracking"""
        self._weak_refs.add(obj)

    def cleanup_memory(self, aggressive: bool = False):
        """Perform memory cleanup"""
        logger.info(f"Starting memory cleanup (aggressive={aggressive})")

        # Run registered cleanup functions
        for cleanup_func in self._cleanup_functions:
            try:
                cleanup_func()
            except Exception as e:
                logger.error(f"Cleanup function failed: {e}")

        # Force garbage collection
        import gc

        gc.collect()

        if aggressive:
            # Additional aggressive cleanup
            self._aggressive_cleanup()

        logger.info("Memory cleanup completed")

    def _aggressive_cleanup(self):
        """Aggressive memory cleanup"""
        import gc

        # Multiple GC passes
        for _ in range(3):
            gc.collect()

        # Clear weak references
        self._weak_refs.clear()

        # Try to compact memory (platform specific)
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass  # Not available on all platforms


class UnifiedMemoryManager:
    """Main interface for all memory operations"""

    def __init__(self):
        self.pool = UnifiedMemoryPool()
        self.cache = UnifiedCache()
        self.monitor = MemoryMonitor()
        self.optimizer = MemoryOptimizer()

        # Set up memory pressure handling
        self.monitor.add_pressure_callback(self._handle_memory_pressure)

    def _handle_memory_pressure(self, stats: MemoryStats):
        """Handle memory pressure events"""
        logger.warning(f"Memory pressure detected: {stats.memory_percent:.1f}%")

        # Clear caches first
        self.cache.clear()

        # Run cleanup
        self.optimizer.cleanup_memory(aggressive=True)

    @asynccontextmanager
    async def connection(self):
        """Context manager for pool connections"""
        conn = await self.pool.get_connection()
        try:
            yield conn
        finally:
            await self.pool.return_connection(conn)

    def get_system_stats(self) -> dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            "memory": self.monitor.get_memory_stats(),
            "cache": self.cache.get_stats(),
            "pool": self.pool.get_stats(),
            "pressure": self.monitor.is_under_pressure(),
        }

    async def start(self):
        """Start all memory services"""
        logger.info("Starting unified memory manager")
        await self.monitor.start_monitoring()

    def stop(self):
        """Stop all memory services"""
        logger.info("Stopping unified memory manager")
        self.monitor.stop_monitoring()


# Global memory manager instance
_memory_manager: UnifiedMemoryManager | None = None


def get_memory_manager() -> UnifiedMemoryManager:
    """Get the global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = UnifiedMemoryManager()
    return _memory_manager


# Convenience functions for backward compatibility
def get_memory_stats() -> MemoryStats:
    """Get current memory statistics"""
    return get_memory_manager().monitor.get_memory_stats()


def is_under_memory_pressure() -> bool:
    """Check if system is under memory pressure"""
    return get_memory_manager().monitor.is_under_pressure()


def cleanup_memory(aggressive: bool = False):
    """Perform memory cleanup"""
    get_memory_manager().optimizer.cleanup_memory(aggressive)


@asynccontextmanager
async def memory_connection():
    """Get a connection from the global pool"""
    async with get_memory_manager().connection() as conn:
        yield conn


# Export main classes and functions
__all__ = [
    "UnifiedMemoryManager",
    "MemoryStats",
    "CacheStats",
    "UnifiedMemoryPool",
    "UnifiedCache",
    "MemoryMonitor",
    "MemoryOptimizer",
    "get_memory_manager",
    "get_memory_stats",
    "is_under_memory_pressure",
    "cleanup_memory",
    "memory_connection",
]
