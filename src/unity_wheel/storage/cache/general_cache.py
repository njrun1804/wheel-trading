"""Intelligent caching layer for expensive calculations with TTL and invalidation."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from src.unity_wheel.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""

    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    computation_time_ms: float = 0.0
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return datetime.now(timezone.utc) >= self.expires_at

    @property
    def age_seconds(self) -> float:
        """Age of cache entry in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    def record_hit(self) -> None:
        """Record a cache hit."""
        self.hit_count += 1


class CacheStatistics:
    """Track cache performance statistics."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.computation_time_saved_ms = 0.0
        self.entries_by_key: Dict[str, int] = {}

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def record_hit(self, key: str, time_saved_ms: float) -> None:
        """Record cache hit."""
        self.hits += 1
        self.computation_time_saved_ms += time_saved_ms
        self.entries_by_key[key] = self.entries_by_key.get(key, 0) + 1

    def record_miss(self, key: str) -> None:
        """Record cache miss."""
        self.misses += 1

    def record_eviction(self, key: str) -> None:
        """Record cache eviction."""
        self.evictions += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "evictions": self.evictions,
            "time_saved_ms": round(self.computation_time_saved_ms, 2),
            "unique_keys": len(self.entries_by_key),
            "top_keys": sorted(self.entries_by_key.items(), key=lambda x: x[1], reverse=True)[:5],
        }


class IntelligentCache:
    """
    Intelligent caching system with:
    - TTL-based expiration
    - Size limits
    - Smart eviction (LRU + computation cost aware)
    - Serialization support
    - Performance tracking
    """

    def __init__(
        self,
        max_size_mb: float = 100.0,
        default_ttl: timedelta = timedelta(minutes=15),
        persistence_path: Optional[Path] = None,
    ):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.default_ttl = default_ttl
        self.persistence_path = persistence_path

        self._cache: Dict[str, CacheEntry] = {}
        self._total_size = 0
        self._stats = CacheStatistics()

        # Load persisted cache if available
        if persistence_path and persistence_path.exists():
            self._load_from_disk()

    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function call."""
        # Create hashable representation
        key_data = {
            "func": func_name,
            "args": args,
            "kwargs": sorted(kwargs.items()),
        }

        # Use JSON for serialization (handles most types)
        try:
            key_str = json.dumps(key_data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # Fallback to repr for non-JSON serializable objects
            key_str = repr(key_data)

        # Create hash for compact key
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of cached value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            return len(str(value))

    def _evict_if_needed(self, required_space: int) -> None:
        """Evict entries if needed to make space."""
        if self._total_size + required_space <= self.max_size_bytes:
            return

        # Sort by score: age * hit_count / computation_time
        # (prefer evicting old, rarely used, cheap-to-compute entries)
        entries = list(self._cache.values())
        entries.sort(key=lambda e: e.age_seconds / (e.hit_count + 1) / (e.computation_time_ms + 1))

        # Evict until we have space
        while self._total_size + required_space > self.max_size_bytes and entries:
            entry = entries.pop(0)
            self._remove_entry(entry.key)

    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache[key]
            self._total_size -= entry.size_bytes
            del self._cache[key]
            self._stats.record_eviction(key)

    def get(
        self,
        key: str,
        compute_func: Optional[Callable[[], T]] = None,
        ttl: Optional[timedelta] = None,
    ) -> Optional[T]:
        """
        Get value from cache or compute it.

        Args:
            key: Cache key
            compute_func: Function to compute value if not cached
            ttl: Time to live for this entry

        Returns:
            Cached or computed value
        """
        # Check if we have valid cached value
        if key in self._cache:
            entry = self._cache[key]

            if not entry.is_expired:
                entry.record_hit()
                self._stats.record_hit(key, entry.computation_time_ms)
                logger.debug(
                    f"Cache hit for {key}",
                    extra={
                        "key": key,
                        "age_seconds": entry.age_seconds,
                        "hit_count": entry.hit_count,
                    },
                )
                return entry.value
            else:
                # Remove expired entry
                self._remove_entry(key)

        # Cache miss
        self._stats.record_miss(key)

        if compute_func is None:
            return None

        # Compute value
        start_time = time.time()
        try:
            value = compute_func()
        except Exception as e:
            logger.error(
                f"Failed to compute value for cache key {key}", extra={"key": key, "error": str(e)}
            )
            raise

        computation_time_ms = (time.time() - start_time) * 1000

        # Store in cache
        self.set(key, value, ttl, computation_time_ms)

        return value

    def set(
        self,
        key: str,
        value: T,
        ttl: Optional[timedelta] = None,
        computation_time_ms: float = 0.0,
    ) -> None:
        """Store value in cache."""
        ttl = ttl or self.default_ttl

        # Estimate size
        size_bytes = self._estimate_size(value)

        # Evict if needed
        self._evict_if_needed(size_bytes)

        # Create entry
        now = datetime.now(timezone.utc)
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            expires_at=now + ttl,
            computation_time_ms=computation_time_ms,
            size_bytes=size_bytes,
        )

        # Remove old entry if exists
        if key in self._cache:
            self._remove_entry(key)

        # Store new entry
        self._cache[key] = entry
        self._total_size += size_bytes

        logger.debug(
            f"Cached value for {key}",
            extra={
                "key": key,
                "size_bytes": size_bytes,
                "ttl_seconds": ttl.total_seconds(),
                "computation_time_ms": round(computation_time_ms, 2),
            },
        )

    def invalidate(self, key: str) -> None:
        """Invalidate cache entry."""
        if key in self._cache:
            self._remove_entry(key)
            logger.info(f"Invalidated cache key {key}")

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        import re

        regex = re.compile(pattern)

        keys_to_remove = [k for k in self._cache if regex.match(k)]
        for key in keys_to_remove:
            self._remove_entry(key)

        if keys_to_remove:
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries matching {pattern}")

        return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._total_size = 0
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self._stats.get_summary()
        stats.update(
            {
                "total_entries": len(self._cache),
                "total_size_mb": round(self._total_size / 1024 / 1024, 2),
                "size_limit_mb": round(self.max_size_bytes / 1024 / 1024, 2),
                "utilization": f"{(self._total_size / self.max_size_bytes):.1%}",
            }
        )
        return stats

    def cleanup_expired(self) -> int:
        """Remove all expired entries."""
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired]

        for key in expired_keys:
            self._remove_entry(key)

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    def _save_to_disk(self) -> None:
        """Persist cache to disk."""
        if not self.persistence_path:
            return

        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to serializable format
            cache_data = {
                "version": 1,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "entries": {
                    key: {
                        "value": entry.value,
                        "created_at": entry.created_at.isoformat(),
                        "expires_at": entry.expires_at.isoformat(),
                        "hit_count": entry.hit_count,
                        "computation_time_ms": entry.computation_time_ms,
                    }
                    for key, entry in self._cache.items()
                    if not entry.is_expired
                },
            }

            # Use JSON instead of pickle for security
            with open(self.persistence_path, "w") as f:
                json.dump(cache_data, f, default=str)

            logger.info(f"Saved {len(cache_data['entries'])} cache entries to disk")

        except Exception as e:
            logger.error(f"Failed to save cache to disk: {e}")

    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            # Use JSON instead of pickle for security
            with open(self.persistence_path, "r") as f:
                cache_data = json.load(f)

            loaded = 0
            for key, data in cache_data.get("entries", {}).items():
                # Reconstruct entry
                entry = CacheEntry(
                    key=key,
                    value=data["value"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    expires_at=datetime.fromisoformat(data["expires_at"]),
                    hit_count=data.get("hit_count", 0),
                    computation_time_ms=data.get("computation_time_ms", 0),
                )

                # Only load non-expired entries
                if not entry.is_expired:
                    entry.size_bytes = self._estimate_size(entry.value)
                    self._cache[key] = entry
                    self._total_size += entry.size_bytes
                    loaded += 1

            logger.info(f"Loaded {loaded} cache entries from disk")

        except Exception as e:
            logger.error(f"Failed to load cache from disk: {e}")


# Global cache instance
_cache = IntelligentCache()


def cached(
    ttl: Union[timedelta, int] = timedelta(minutes=15),
    key_prefix: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator for caching function results.

    Args:
        ttl: Time to live (timedelta or seconds)
        key_prefix: Optional prefix for cache key
    """
    if isinstance(ttl, int):
        ttl = timedelta(seconds=ttl)

    def decorator(func: F) -> F:
        func_name = f"{func.__module__}.{func.__name__}"
        if key_prefix:
            func_name = f"{key_prefix}.{func_name}"

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            cache_key = _cache._generate_key(func_name, args, kwargs)

            # Try to get from cache
            def compute():
                return func(*args, **kwargs)

            return _cache.get(cache_key, compute, ttl)

        # Add cache control methods
        wrapper.invalidate = lambda: _cache.invalidate_pattern(f"{func_name}.*")  # type: ignore
        wrapper.cache_stats = lambda: _cache.get_stats()  # type: ignore

        return wrapper  # type: ignore

    return decorator


def invalidate_cache(pattern: Optional[str] = None) -> None:
    """Invalidate cache entries matching pattern."""
    if pattern:
        _cache.invalidate_pattern(pattern)
    else:
        _cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    return _cache.get_stats()
