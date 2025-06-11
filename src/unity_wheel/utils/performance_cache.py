"""Performance-optimized caching layer for Unity Wheel Trading Bot.

Provides memory-aware LRU caching with performance monitoring for expensive calculations.
"""

import asyncio
import functools
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from weakref import WeakKeyDictionary

from src.unity_wheel.utils.logging import StructuredLogger

logger = StructuredLogger(logging.getLogger(__name__))

T = TypeVar("T")


@dataclass
class CacheStats:
    """Cache performance statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_bytes: int = 0
    avg_computation_time_ms: float = 0.0
    max_computation_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset(self) -> None:
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.memory_bytes = 0
        self.avg_computation_time_ms = 0.0
        self.max_computation_time_ms = 0.0


class MemoryAwareLRUCache:
    """Memory-aware LRU cache with performance monitoring."""

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 50.0,
        ttl_seconds: Optional[float] = None,
        name: str = "cache",
    ):
        """Initialize cache.

        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            ttl_seconds: Time-to-live for entries (None = no expiration)
            name: Cache name for logging
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.ttl_seconds = ttl_seconds
        self.name = name

        self._cache: OrderedDict = OrderedDict()
        self._stats = CacheStats()
        self._lock = asyncio.Lock()

        # Performance tracking
        self._computation_times: List[float] = []
        self._last_cleanup = time.time()

        logger.debug(
            "cache_initialized",
            extra={
                "name": name,
                "max_size": max_size,
                "max_memory_mb": max_memory_mb,
                "ttl_seconds": ttl_seconds,
            },
        )

    def _get_memory_usage(self, value: Any) -> int:
        """Estimate memory usage of a value."""
        try:
            import sys

            return sys.getsizeof(value)
        except Exception:
            # Fallback estimation
            if isinstance(value, (int, float)):
                return 24
            elif isinstance(value, str):
                return 24 + len(value)
            elif isinstance(value, (list, tuple)):
                return 56 + sum(
                    self._get_memory_usage(item) for item in value[:10]
                )  # Sample first 10
            elif isinstance(value, dict):
                return 240 + sum(
                    self._get_memory_usage(k) + self._get_memory_usage(v)
                    for k, v in list(value.items())[:10]  # Sample first 10
                )
            else:
                return 64  # Default estimate

    def _is_expired(self, timestamp: float) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - timestamp > self.ttl_seconds

    async def get(self, key: str) -> Tuple[Any, bool]:
        """Get value from cache.

        Returns:
            Tuple of (value, found) where found indicates if key was in cache
        """
        async with self._lock:
            if key in self._cache:
                value, timestamp, memory_size = self._cache[key]

                # Check expiration
                if self._is_expired(timestamp):
                    del self._cache[key]
                    self._stats.memory_bytes -= memory_size
                    self._stats.misses += 1
                    return None, False

                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._stats.hits += 1

                return value, True

            self._stats.misses += 1
            return None, False

    async def put(self, key: str, value: Any, computation_time_ms: Optional[float] = None) -> None:
        """Put value in cache.

        Args:
            key: Cache key
            value: Value to cache
            computation_time_ms: Time taken to compute value (for stats)
        """
        memory_size = self._get_memory_usage(value)
        timestamp = time.time()

        async with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                _, _, old_memory = self._cache[key]
                self._stats.memory_bytes -= old_memory

            # Check memory limit
            if memory_size > self.max_memory_bytes:
                logger.warning(
                    "cache_value_too_large",
                    extra={
                        "name": self.name,
                        "key": key,
                        "size_mb": memory_size / 1024 / 1024,
                        "max_mb": self.max_memory_bytes / 1024 / 1024,
                    },
                )
                return

            # Evict entries if needed
            await self._evict_if_needed(memory_size)

            # Add new entry
            self._cache[key] = (value, timestamp, memory_size)
            self._stats.memory_bytes += memory_size

            # Update computation time stats
            if computation_time_ms is not None:
                self._computation_times.append(computation_time_ms)
                if len(self._computation_times) > 100:
                    self._computation_times.pop(0)

                self._stats.avg_computation_time_ms = sum(self._computation_times) / len(
                    self._computation_times
                )
                self._stats.max_computation_time_ms = max(
                    self._stats.max_computation_time_ms, computation_time_ms
                )

    async def _evict_if_needed(self, new_memory: int) -> None:
        """Evict entries if memory or size limits would be exceeded."""
        # Check size limit
        while len(self._cache) >= self.max_size:
            self._evict_oldest()

        # Check memory limit
        while self._stats.memory_bytes + new_memory > self.max_memory_bytes and self._cache:
            self._evict_oldest()

    def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) entry."""
        if not self._cache:
            return

        key, (value, timestamp, memory_size) = self._cache.popitem(last=False)
        self._stats.memory_bytes -= memory_size
        self._stats.evictions += 1

        logger.debug(
            "cache_eviction",
            extra={
                "name": self.name,
                "key": key,
                "age_seconds": time.time() - timestamp,
                "size_bytes": memory_size,
            },
        )

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._stats.memory_bytes = 0
            logger.info("cache_cleared", extra={"name": self.name})

    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        return self._stats

    async def cleanup_expired(self) -> int:
        """Remove expired entries. Returns number of entries removed."""
        if self.ttl_seconds is None:
            return 0

        now = time.time()
        if now - self._last_cleanup < 60:  # Cleanup max once per minute
            return 0

        removed = 0
        async with self._lock:
            expired_keys = [
                key for key, (_, timestamp, _) in self._cache.items() if self._is_expired(timestamp)
            ]

            for key in expired_keys:
                _, _, memory_size = self._cache[key]
                del self._cache[key]
                self._stats.memory_bytes -= memory_size
                removed += 1

        self._last_cleanup = now

        if removed > 0:
            logger.debug("cache_cleanup", extra={"name": self.name, "removed_count": removed})

        return removed


class CacheManager:
    """Global cache manager for the application."""

    def __init__(self):
        self._caches: Dict[str, MemoryAwareLRUCache] = {}
        self._default_cache = MemoryAwareLRUCache(name="default")
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()

    def get_cache(self, name: str = "default") -> MemoryAwareLRUCache:
        """Get or create a named cache."""
        if name == "default":
            return self._default_cache

        if name not in self._caches:
            self._caches[name] = MemoryAwareLRUCache(name=name)

        return self._caches[name]

    def create_cache(
        self,
        name: str,
        max_size: int = 1000,
        max_memory_mb: float = 50.0,
        ttl_seconds: Optional[float] = None,
    ) -> MemoryAwareLRUCache:
        """Create a new named cache with specific parameters."""
        cache = MemoryAwareLRUCache(
            max_size=max_size, max_memory_mb=max_memory_mb, ttl_seconds=ttl_seconds, name=name
        )
        self._caches[name] = cache
        return cache

    def get_global_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all caches."""
        stats = {"default": self._default_cache.get_stats()}
        stats.update({name: cache.get_stats() for name, cache in self._caches.items()})
        return stats

    async def clear_all(self) -> None:
        """Clear all caches."""
        await self._default_cache.clear()
        for cache in self._caches.values():
            await cache.clear()

    def _start_cleanup_task(self) -> None:
        """Start background cleanup task."""

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Cleanup every 5 minutes

                    # Cleanup all caches
                    await self._default_cache.cleanup_expired()
                    for cache in self._caches.values():
                        await cache.cleanup_expired()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("cache_cleanup_error", extra={"error": str(e)})

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    def shutdown(self) -> None:
        """Shutdown the cache manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()


# Global cache manager instance
_cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager."""
    return _cache_manager


def cached(
    cache_name: str = "default",
    ttl_seconds: Optional[float] = None,
    key_func: Optional[Callable] = None,
    include_self: bool = False,
):
    """Decorator for caching function results.

    Args:
        cache_name: Name of cache to use
        ttl_seconds: Time-to-live for cached results
        key_func: Custom function to generate cache keys
        include_self: Include 'self' parameter in cache key for methods
    """

    def decorator(func: Callable) -> Callable:
        cache = _cache_manager.get_cache(cache_name)

        def generate_key(*args, **kwargs) -> str:
            if key_func:
                return key_func(*args, **kwargs)

            # Default key generation
            key_parts = [func.__name__]

            # Handle args
            start_idx = 1 if not include_self and args and hasattr(args[0], func.__name__) else 0
            for arg in args[start_idx:]:
                if isinstance(arg, (int, float, str, bool)):
                    key_parts.append(str(arg))
                else:
                    key_parts.append(str(hash(str(arg))))

            # Handle kwargs
            for k, v in sorted(kwargs.items()):
                if isinstance(v, (int, float, str, bool)):
                    key_parts.append(f"{k}={v}")
                else:
                    key_parts.append(f"{k}={hash(str(v))}")

            key = ":".join(key_parts)
            return hashlib.md5(key.encode()).hexdigest()

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                key = generate_key(*args, **kwargs)

                # Try to get from cache
                cached_result, found = await cache.get(key)
                if found:
                    return cached_result

                # Compute result
                start_time = time.time()
                result = await func(*args, **kwargs)
                computation_time_ms = (time.time() - start_time) * 1000

                # Cache result
                await cache.put(key, result, computation_time_ms)

                return result

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                key = generate_key(*args, **kwargs)

                # Try to get from cache (need to run in event loop)
                try:
                    loop = asyncio.get_event_loop()
                    cached_result, found = loop.run_until_complete(cache.get(key))
                    if found:
                        return cached_result
                except RuntimeError:
                    # No event loop - skip caching
                    return func(*args, **kwargs)

                # Compute result
                start_time = time.time()
                result = func(*args, **kwargs)
                computation_time_ms = (time.time() - start_time) * 1000

                # Cache result
                try:
                    loop.run_until_complete(cache.put(key, result, computation_time_ms))
                except RuntimeError:
                    pass  # No event loop - skip caching

                return result

            return sync_wrapper

    return decorator


def cache_key_for_options(
    spot_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str,
) -> str:
    """Generate cache key for options calculations."""
    # Round to reasonable precision to improve cache hit rates
    key_parts = [
        f"S:{spot_price:.2f}",
        f"K:{strike_price:.2f}",
        f"T:{time_to_expiry:.4f}",
        f"r:{risk_free_rate:.4f}",
        f"v:{volatility:.3f}",
        f"type:{option_type.lower()}",
    ]
    key = "|".join(key_parts)
    return hashlib.md5(key.encode()).hexdigest()


# Specialized cache for options calculations
options_cache = _cache_manager.create_cache(
    name="options",
    max_size=5000,
    max_memory_mb=100.0,
    ttl_seconds=3600,  # 1 hour TTL for options calculations
)

# Specialized cache for risk calculations
risk_cache = _cache_manager.create_cache(
    name="risk",
    max_size=2000,
    max_memory_mb=75.0,
    ttl_seconds=1800,  # 30 minutes TTL for risk calculations
)

# Specialized cache for market data
market_data_cache = _cache_manager.create_cache(
    name="market_data",
    max_size=10000,
    max_memory_mb=200.0,
    ttl_seconds=300,  # 5 minutes TTL for market data
)
