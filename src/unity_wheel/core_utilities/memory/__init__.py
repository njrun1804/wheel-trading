"""
Unified Memory Management System.

Consolidates 15 memory-related implementations into a single, optimized framework.
Provides thread-safe memory operations with automatic pressure detection and cleanup.

Key Features:
- Unified allocation and deallocation
- Automatic memory pressure detection
- Thread-safe operations with minimal overhead
- Zero-copy data sharing where possible
- M4 Pro optimized (24GB aware)
"""

from .allocator import AllocationContext, MemoryAllocator
from .cache import CacheEntry, CacheStats, UnifiedCache
from .pools import MemoryPool, PooledBuffer
from .pressure import MemoryPressure, PressureMonitor

# Singleton instances for global use
_allocator = None
_cache = None
_monitor = None


def get_allocator() -> MemoryAllocator:
    """Get the global memory allocator instance."""
    global _allocator
    if _allocator is None:
        _allocator = MemoryAllocator()
    return _allocator


def get_cache() -> UnifiedCache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = UnifiedCache()
    return _cache


def get_monitor() -> PressureMonitor:
    """Get the global pressure monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = PressureMonitor()
    return _monitor


# Convenience functions
allocate = lambda size, component="default": get_allocator().allocate(size, component)
deallocate = lambda allocation_id: get_allocator().deallocate(allocation_id)
cache_get = lambda key: get_cache().get(key)
cache_set = lambda key, value, ttl=None: get_cache().set(key, value, ttl)
memory_pressure = lambda: get_monitor().current_pressure

__all__ = [
    "MemoryAllocator",
    "AllocationContext",
    "UnifiedCache",
    "CacheEntry",
    "CacheStats",
    "PressureMonitor",
    "MemoryPressure",
    "MemoryPool",
    "PooledBuffer",
    "get_allocator",
    "get_cache",
    "get_monitor",
    "allocate",
    "deallocate",
    "cache_get",
    "cache_set",
    "memory_pressure",
]
