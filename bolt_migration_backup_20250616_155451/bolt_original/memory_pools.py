"""
Advanced Memory Pool Management for M4 Pro Unified Memory

Implements intelligent memory pools and caching strategies optimized for
large embedding indexes and unified memory architecture.
"""

import gc
import logging
import mmap
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from .unified_memory import BufferType, UnifiedMemoryBuffer

logger = logging.getLogger(__name__)


class PoolType(Enum):
    """Types of memory pools"""

    EMBEDDING = "embedding"  # Large embedding matrices
    CACHE = "cache"  # Frequently accessed cache data
    TEMPORARY = "temporary"  # Short-lived temporary data
    PERSISTENT = "persistent"  # Long-lived application data
    MMAP = "mmap"  # Memory-mapped file data


class EvictionPolicy(Enum):
    """Cache eviction policies"""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


@dataclass
class PoolStats:
    """Memory pool statistics"""

    pool_name: str
    pool_type: PoolType
    total_size_mb: float = 0.0
    used_size_mb: float = 0.0
    free_size_mb: float = 0.0
    num_allocations: int = 0
    num_deallocations: int = 0
    num_cache_hits: int = 0
    num_cache_misses: int = 0
    num_evictions: int = 0
    average_allocation_size_mb: float = 0.0
    peak_usage_mb: float = 0.0
    fragmentation_ratio: float = 0.0


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    key: str
    data: Any
    size_bytes: int
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float | None = None

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return self.age_seconds > self.ttl_seconds

    def touch(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_access = time.time()


class MemoryPool(ABC):
    """Abstract base class for memory pools"""

    def __init__(self, name: str, pool_type: PoolType, max_size_mb: float):
        self.name = name
        self.pool_type = pool_type
        self.max_size_mb = max_size_mb
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.stats = PoolStats(name, pool_type)
        self._lock = threading.RLock()
        self.created_at = time.time()

        logger.debug(
            f"Created memory pool '{name}' ({pool_type.value}, {max_size_mb}MB)"
        )

    @abstractmethod
    def allocate(self, size_bytes: int, key: str | None = None) -> Any:
        """Allocate memory from pool"""
        pass

    @abstractmethod
    def deallocate(self, ptr: Any, key: str | None = None) -> bool:
        """Deallocate memory back to pool"""
        pass

    @abstractmethod
    def get_free_space(self) -> int:
        """Get available free space in bytes"""
        pass

    def _update_stats(self, allocated_bytes: int, is_allocation: bool):
        """Update pool statistics"""
        with self._lock:
            if is_allocation:
                self.stats.num_allocations += 1
                self.stats.used_size_mb += allocated_bytes / (1024 * 1024)
            else:
                self.stats.num_deallocations += 1
                self.stats.used_size_mb -= allocated_bytes / (1024 * 1024)

            self.stats.free_size_mb = self.stats.total_size_mb - self.stats.used_size_mb
            self.stats.peak_usage_mb = max(
                self.stats.peak_usage_mb, self.stats.used_size_mb
            )

            if self.stats.num_allocations > 0:
                total_allocated_mb = (
                    self.stats.num_allocations * self.stats.used_size_mb
                )
                self.stats.average_allocation_size_mb = (
                    total_allocated_mb / self.stats.num_allocations
                )


class EmbeddingPool(MemoryPool):
    """
    Specialized memory pool for large embedding matrices.

    Optimized for M4 Pro unified memory with zero-copy operations
    and efficient memory-mapped file support.
    """

    def __init__(self, name: str, max_size_mb: float = 4096):
        super().__init__(name, PoolType.EMBEDDING, max_size_mb)
        self._allocations: dict[str, UnifiedMemoryBuffer] = {}
        self._memory_mapped_files: dict[str, tuple[mmap.mmap, int]] = {}
        self._allocation_order: list[str] = []  # For LRU eviction

        # Initialize stats
        self.stats.total_size_mb = max_size_mb
        self.stats.free_size_mb = max_size_mb

    def allocate(self, size_bytes: int, key: str | None = None) -> UnifiedMemoryBuffer:
        """Allocate unified memory buffer for embeddings"""
        if key is None:
            key = f"embedding_{len(self._allocations)}"

        with self._lock:
            # Check if already allocated
            if key in self._allocations:
                self._update_access_order(key)
                return self._allocations[key]

            # Check space and evict if necessary
            if not self._ensure_space(size_bytes):
                raise MemoryError(
                    f"Cannot allocate {size_bytes} bytes in pool {self.name}"
                )

            # Create unified memory buffer
            try:
                buffer = UnifiedMemoryBuffer(
                    size_bytes, BufferType.EMBEDDING_MATRIX, key
                )
                self._allocations[key] = buffer
                self._allocation_order.append(key)

                self._update_stats(size_bytes, is_allocation=True)
                logger.debug(
                    f"Allocated {size_bytes/1024/1024:.1f}MB embedding buffer '{key}'"
                )

                return buffer

            except Exception as e:
                logger.error(f"Failed to allocate embedding buffer: {e}")
                raise

    def allocate_from_mmap(self, file_path: Path, key: str | None = None) -> np.ndarray:
        """Allocate memory-mapped embedding from file"""
        if key is None:
            key = f"mmap_{file_path.stem}"

        with self._lock:
            if key in self._memory_mapped_files:
                mmap_obj, size = self._memory_mapped_files[key]
                return np.frombuffer(mmap_obj, dtype=np.float32)

            try:
                # Open file for memory mapping
                with open(file_path, "r+b") as f:
                    file_size = os.path.getsize(file_path)
                    mmap_obj = mmap.mmap(f.fileno(), file_size, access=mmap.ACCESS_READ)

                    self._memory_mapped_files[key] = (mmap_obj, file_size)
                    self._allocation_order.append(key)

                    # Create numpy array view
                    embedding_array = np.frombuffer(mmap_obj, dtype=np.float32)

                    self._update_stats(file_size, is_allocation=True)
                    logger.info(
                        f"Memory-mapped embedding file '{file_path}' as '{key}' ({file_size/1024/1024:.1f}MB)"
                    )

                    return embedding_array

            except Exception as e:
                logger.error(f"Failed to memory-map file {file_path}: {e}")
                raise

    def deallocate(self, key: str) -> bool:
        """Deallocate embedding buffer"""
        with self._lock:
            if key in self._allocations:
                buffer = self._allocations.pop(key)
                if key in self._allocation_order:
                    self._allocation_order.remove(key)

                self._update_stats(buffer.size_bytes, is_allocation=False)
                logger.debug(f"Deallocated embedding buffer '{key}'")
                return True

            elif key in self._memory_mapped_files:
                mmap_obj, size = self._memory_mapped_files.pop(key)
                mmap_obj.close()
                if key in self._allocation_order:
                    self._allocation_order.remove(key)

                self._update_stats(size, is_allocation=False)
                logger.debug(f"Unmapped memory-mapped file '{key}'")
                return True

            return False

    def get_free_space(self) -> int:
        """Get available free space in bytes"""
        with self._lock:
            used_bytes = sum(buf.size_bytes for buf in self._allocations.values())
            used_bytes += sum(size for _, size in self._memory_mapped_files.values())
            return max(0, self.max_size_bytes - used_bytes)

    def _ensure_space(self, required_bytes: int) -> bool:
        """Ensure sufficient space by evicting if necessary"""
        while self.get_free_space() < required_bytes and self._allocation_order:
            # Evict least recently used
            lru_key = self._allocation_order[0]
            logger.info(f"Evicting LRU embedding '{lru_key}' to free space")
            if not self.deallocate(lru_key):
                break
            self.stats.num_evictions += 1

        return self.get_free_space() >= required_bytes

    def _update_access_order(self, key: str):
        """Update LRU access order"""
        if key in self._allocation_order:
            self._allocation_order.remove(key)
            self._allocation_order.append(key)


class CachePool(MemoryPool):
    """
    High-performance cache pool with configurable eviction policies.

    Supports LRU, LFU, TTL, and adaptive eviction strategies
    optimized for M4 Pro memory bandwidth.
    """

    def __init__(
        self,
        name: str,
        max_size_mb: float = 1024,
        eviction_policy: EvictionPolicy = EvictionPolicy.ADAPTIVE,
        default_ttl_seconds: float | None = None,
    ):
        super().__init__(name, PoolType.CACHE, max_size_mb)
        self.eviction_policy = eviction_policy
        self.default_ttl_seconds = default_ttl_seconds

        # Cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._access_frequencies: defaultdict[str, int] = defaultdict(int)

        # Adaptive eviction parameters
        self._hit_rate_window = 1000  # Track hit rate over last N operations
        self._recent_operations = []
        self._adaptive_threshold = 0.7  # Switch policies if hit rate drops below this

        # Background cleanup
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()

        # Initialize stats
        self.stats.total_size_mb = max_size_mb
        self.stats.free_size_mb = max_size_mb

    def get(self, key: str) -> Any | None:
        """Get item from cache"""
        with self._lock:
            if key not in self._cache:
                self.stats.num_cache_misses += 1
                self._record_operation(False)
                return None

            entry = self._cache[key]

            # Check TTL expiration
            if entry.is_expired:
                self._remove_entry(key)
                self.stats.num_cache_misses += 1
                self._record_operation(False)
                return None

            # Update access statistics
            entry.touch()
            self._access_frequencies[key] += 1

            # Move to end for LRU
            if self.eviction_policy in [EvictionPolicy.LRU, EvictionPolicy.ADAPTIVE]:
                self._cache.move_to_end(key)

            self.stats.num_cache_hits += 1
            self._record_operation(True)

            return entry.data

    def put(self, key: str, data: Any, ttl_seconds: float | None = None) -> bool:
        """Put item in cache"""
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl_seconds

        # Estimate data size
        size_bytes = self._estimate_size(data)

        with self._lock:
            # Check if update
            if key in self._cache:
                old_entry = self._cache[key]
                self._update_stats(-old_entry.size_bytes, is_allocation=False)

            # Ensure space
            if not self._ensure_space(size_bytes, exclude_key=key):
                logger.warning(f"Cannot cache item '{key}' - insufficient space")
                return False

            # Create entry
            entry = CacheEntry(
                key=key, data=data, size_bytes=size_bytes, ttl_seconds=ttl_seconds
            )

            self._cache[key] = entry
            self._update_stats(size_bytes, is_allocation=True)

            return True

    def allocate(self, size_bytes: int, key: str | None = None) -> bytes:
        """Allocate raw bytes (for compatibility)"""
        if key is None:
            key = f"raw_{len(self._cache)}"

        # Allocate raw bytes
        data = bytearray(size_bytes)

        if self.put(key, data):
            return data
        else:
            raise MemoryError(f"Cannot allocate {size_bytes} bytes in cache pool")

    def deallocate(self, key: str) -> bool:
        """Remove item from cache"""
        with self._lock:
            return self._remove_entry(key)

    def get_free_space(self) -> int:
        """Get available free space in bytes"""
        with self._lock:
            used_bytes = sum(entry.size_bytes for entry in self._cache.values())
            return max(0, self.max_size_bytes - used_bytes)

    def _ensure_space(
        self, required_bytes: int, exclude_key: str | None = None
    ) -> bool:
        """Ensure sufficient space by evicting entries"""
        while self.get_free_space() < required_bytes:
            victim_key = self._select_eviction_victim(exclude_key)
            if victim_key is None:
                return False

            if not self._remove_entry(victim_key):
                return False

            self.stats.num_evictions += 1

        return True

    def _select_eviction_victim(self, exclude_key: str | None = None) -> str | None:
        """Select victim for eviction based on policy"""
        if not self._cache:
            return None

        candidates = [k for k in self._cache if k != exclude_key]
        if not candidates:
            return None

        if self.eviction_policy == EvictionPolicy.LRU:
            return candidates[0]  # OrderedDict maintains insertion/access order

        elif self.eviction_policy == EvictionPolicy.LFU:
            return min(candidates, key=lambda k: self._access_frequencies[k])

        elif self.eviction_policy == EvictionPolicy.TTL:
            # Evict expired entries first, then oldest
            time.time()
            expired = [k for k in candidates if self._cache[k].is_expired]
            if expired:
                return expired[0]
            return min(candidates, key=lambda k: self._cache[k].created_at)

        elif self.eviction_policy == EvictionPolicy.ADAPTIVE:
            return self._adaptive_eviction_victim(candidates)

        return candidates[0]  # Fallback

    def _adaptive_eviction_victim(self, candidates: list[str]) -> str:
        """Adaptive eviction based on recent hit rate"""
        # Calculate recent hit rate
        recent_hits = sum(self._recent_operations[-self._hit_rate_window :])
        hit_rate = (
            recent_hits / len(self._recent_operations) if self._recent_operations else 0
        )

        if hit_rate < self._adaptive_threshold:
            # Poor hit rate - use LFU to evict rarely used items
            return min(candidates, key=lambda k: self._access_frequencies[k])
        else:
            # Good hit rate - use LRU to maintain working set
            return candidates[0]

    def _remove_entry(self, key: str) -> bool:
        """Remove entry from cache"""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._access_frequencies.pop(key, None)
            self._update_stats(-entry.size_bytes, is_allocation=False)
            return True
        return False

    def _record_operation(self, was_hit: bool):
        """Record cache operation for adaptive policy"""
        self._recent_operations.append(1 if was_hit else 0)
        if len(self._recent_operations) > self._hit_rate_window:
            self._recent_operations.pop(0)

    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data"""
        if isinstance(data, bytes | bytearray):
            return len(data)
        elif isinstance(data, str):
            return len(data.encode("utf-8"))
        elif (
            isinstance(data, np.ndarray) or MLX_AVAILABLE and isinstance(data, mx.array)
        ):
            return data.nbytes
        else:
            # Rough estimate for Python objects
            try:
                import sys

                return sys.getsizeof(data)
            except (ImportError, TypeError, ValueError) as e:
                logger.debug(f"Failed to estimate object size: {e}")
                return 1024  # Default estimate

    def cleanup_expired(self):
        """Clean up expired entries"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired
            ]

            for key in expired_keys:
                self._remove_entry(key)

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

            self._last_cleanup = time.time()

    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total_ops = self.stats.num_cache_hits + self.stats.num_cache_misses
        if total_ops == 0:
            return 0.0
        return self.stats.num_cache_hits / total_ops


class MemoryPoolManager:
    """
    Central manager for all memory pools.

    Coordinates memory allocation across different pool types
    and provides global memory pressure management.
    """

    def __init__(self, total_memory_limit_mb: float = 8192):
        self.total_memory_limit_mb = total_memory_limit_mb
        self.pools: dict[str, MemoryPool] = {}
        self._lock = threading.RLock()

        # Global pressure management
        self._pressure_threshold = 0.85  # 85% of total limit
        self._cleanup_interval = 60  # 1 minute
        self._last_global_cleanup = time.time()

        # Background monitoring
        self._monitoring_active = False
        self._monitor_thread: threading.Thread | None = None

        logger.info(
            f"Initialized MemoryPoolManager with {total_memory_limit_mb}MB limit"
        )

    def create_embedding_pool(self, name: str, max_size_mb: float) -> EmbeddingPool:
        """Create embedding pool"""
        with self._lock:
            if name in self.pools:
                raise ValueError(f"Pool '{name}' already exists")

            pool = EmbeddingPool(name, max_size_mb)
            self.pools[name] = pool

            logger.info(f"Created embedding pool '{name}' ({max_size_mb}MB)")
            return pool

    def create_cache_pool(
        self,
        name: str,
        max_size_mb: float,
        eviction_policy: EvictionPolicy = EvictionPolicy.ADAPTIVE,
        default_ttl_seconds: float | None = None,
    ) -> CachePool:
        """Create cache pool"""
        with self._lock:
            if name in self.pools:
                raise ValueError(f"Pool '{name}' already exists")

            pool = CachePool(name, max_size_mb, eviction_policy, default_ttl_seconds)
            self.pools[name] = pool

            logger.info(
                f"Created cache pool '{name}' ({max_size_mb}MB, {eviction_policy.value})"
            )
            return pool

    def get_pool(self, name: str) -> MemoryPool | None:
        """Get pool by name"""
        return self.pools.get(name)

    def get_total_used_memory_mb(self) -> float:
        """Get total memory used across all pools"""
        with self._lock:
            return sum(pool.stats.used_size_mb for pool in self.pools.values())

    def get_memory_pressure(self) -> float:
        """Get current memory pressure (0.0 to 1.0)"""
        return self.get_total_used_memory_mb() / self.total_memory_limit_mb

    def is_memory_pressure_high(self) -> bool:
        """Check if memory pressure is high"""
        return self.get_memory_pressure() > self._pressure_threshold

    def global_cleanup(self):
        """Perform global cleanup across all pools"""
        with self._lock:
            logger.info("Performing global memory cleanup")

            # Cleanup cache pools first
            for pool in self.pools.values():
                if isinstance(pool, CachePool):
                    pool.cleanup_expired()

            # Force garbage collection
            gc.collect()

            # If still under pressure, more aggressive cleanup
            if self.is_memory_pressure_high():
                logger.warning("High memory pressure - performing aggressive cleanup")
                self._aggressive_cleanup()

            self._last_global_cleanup = time.time()

    def _aggressive_cleanup(self):
        """Aggressive cleanup when under high memory pressure"""
        # Sort pools by usage and priority
        cache_pools = [
            (name, pool)
            for name, pool in self.pools.items()
            if isinstance(pool, CachePool)
        ]

        # Evict from cache pools starting with lowest hit rate
        cache_pools.sort(
            key=lambda x: x[1].get_hit_rate() if hasattr(x[1], "get_hit_rate") else 0
        )

        for name, pool in cache_pools:
            if not self.is_memory_pressure_high():
                break

            # Evict 25% of cache entries
            if hasattr(pool, "_cache"):
                num_to_evict = max(1, len(pool._cache) // 4)
                keys_to_evict = list(pool._cache.keys())[:num_to_evict]

                for key in keys_to_evict:
                    pool.deallocate(key)

                logger.info(
                    f"Aggressive cleanup: evicted {len(keys_to_evict)} entries from '{name}'"
                )

    def start_monitoring(self):
        """Start background monitoring"""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="memory-pool-monitor", daemon=True
        )
        self._monitor_thread.start()
        logger.debug("Started memory pool monitoring")

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                current_time = time.time()

                # Periodic cleanup
                if current_time - self._last_global_cleanup > self._cleanup_interval:
                    self.global_cleanup()

                # Check memory pressure
                if self.is_memory_pressure_high():
                    logger.warning(
                        f"High memory pressure: {self.get_memory_pressure():.1%}"
                    )

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(60)  # Back off on error

    def get_global_stats(self) -> dict[str, Any]:
        """Get global memory statistics"""
        with self._lock:
            pool_stats = {
                name: {
                    "type": pool.pool_type.value,
                    "used_mb": pool.stats.used_size_mb,
                    "free_mb": pool.stats.free_size_mb,
                    "hit_rate": pool.get_hit_rate()
                    if hasattr(pool, "get_hit_rate")
                    else 0.0,
                    "allocations": pool.stats.num_allocations,
                    "evictions": pool.stats.num_evictions,
                }
                for name, pool in self.pools.items()
            }

            return {
                "total_used_mb": self.get_total_used_memory_mb(),
                "total_limit_mb": self.total_memory_limit_mb,
                "memory_pressure": self.get_memory_pressure(),
                "num_pools": len(self.pools),
                "pools": pool_stats,
            }

    def shutdown(self):
        """Shutdown memory pool manager"""
        logger.info("Shutting down MemoryPoolManager")

        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)

        # Close all memory-mapped files
        for pool in self.pools.values():
            if isinstance(pool, EmbeddingPool):
                for mmap_obj, _ in pool._memory_mapped_files.values():
                    mmap_obj.close()

        logger.info("MemoryPoolManager shutdown complete")


# Global instance
_memory_pool_manager: MemoryPoolManager | None = None


def get_memory_pool_manager() -> MemoryPoolManager:
    """Get global memory pool manager instance"""
    global _memory_pool_manager
    if _memory_pool_manager is None:
        _memory_pool_manager = MemoryPoolManager()
        _memory_pool_manager.start_monitoring()
    return _memory_pool_manager


def create_optimized_embedding_pool(
    name: str = "main_embeddings", size_mb: float = 2048
) -> EmbeddingPool:
    """Create optimized embedding pool for M4 Pro"""
    manager = get_memory_pool_manager()
    return manager.create_embedding_pool(name, size_mb)


def create_high_performance_cache(
    name: str = "main_cache", size_mb: float = 512
) -> CachePool:
    """Create high-performance cache pool"""
    manager = get_memory_pool_manager()
    return manager.create_cache_pool(
        name,
        size_mb,
        EvictionPolicy.ADAPTIVE,
        default_ttl_seconds=3600,  # 1 hour default TTL
    )
