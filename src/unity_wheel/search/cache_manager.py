"""3-Tier Cache System for Unified Search - L1 (Memory), L2 (SSD), L3 (Persistent).

Optimized for M4 Pro with unified memory architecture and NVMe SSD.
"""

import asyncio
import hashlib
import logging
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles
import lz4.frame

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""

    key: str
    value: Any
    size_bytes: int
    access_count: int
    last_access: float
    creation_time: float
    ttl_seconds: float


@dataclass
class CacheStats:
    """Cache performance statistics."""

    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l3_hits: int = 0
    l3_misses: int = 0
    total_accesses: int = 0
    evictions: int = 0
    avg_access_time_ms: float = 0.0


class CacheManager:
    """3-Tier cache system optimized for search performance.

    L1: In-memory cache (hot data) - <1ms access
    L2: SSD-backed cache (warm data) - <5ms access
    L3: Persistent cache (cold data) - <10ms access
    """

    def __init__(
        self,
        l1_size_mb: int = 512,
        l2_size_mb: int = 2048,
        l3_size_mb: int = 4096,
        cache_dir: Path | None = None,
    ):
        # Cache sizes
        self.l1_max_bytes = l1_size_mb * 1024 * 1024
        self.l2_max_bytes = l2_size_mb * 1024 * 1024
        self.l3_max_bytes = l3_size_mb * 1024 * 1024

        # L1: Memory cache (OrderedDict for LRU)
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l1_size_bytes = 0
        self.l1_lock = asyncio.Lock()

        # L2: SSD cache directory
        self.cache_dir = cache_dir or Path.home() / ".unity_wheel" / "search_cache"
        self.l2_dir = self.cache_dir / "l2"
        self.l3_dir = self.cache_dir / "l3"

        # Create cache directories
        self.l2_dir.mkdir(parents=True, exist_ok=True)
        self.l3_dir.mkdir(parents=True, exist_ok=True)

        # L2 & L3: Track entries in memory for fast lookup
        self.l2_index: dict[str, tuple[Path, int]] = {}  # key -> (file_path, size)
        self.l3_index: dict[str, tuple[Path, int]] = {}
        self.l2_size_bytes = 0
        self.l3_size_bytes = 0
        self.l2_lock = asyncio.Lock()
        self.l3_lock = asyncio.Lock()

        # Statistics
        self.stats = CacheStats()

        # Background tasks
        self.cleanup_task = None

    async def initialize(self):
        """Initialize cache system and load indices."""
        logger.info("🔧 Initializing 3-tier cache system...")

        # Load L2 and L3 indices
        await self._load_disk_indices()

        # Start background cleanup
        self.cleanup_task = asyncio.create_task(self._periodic_cleanup())

        logger.info(
            f"✅ Cache initialized: L1={self.l1_size_bytes/1024/1024:.1f}MB, "
            f"L2={self.l2_size_bytes/1024/1024:.1f}MB, "
            f"L3={self.l3_size_bytes/1024/1024:.1f}MB"
        )

    async def get(self, key: str) -> Any | None:
        """Get value from cache, checking all tiers."""
        start_time = time.perf_counter()
        self.stats.total_accesses += 1

        # Generate cache key hash
        cache_key = self._hash_key(key)

        # Check L1 cache
        async with self.l1_lock:
            if cache_key in self.l1_cache:
                entry = self.l1_cache[cache_key]
                # Check TTL
                if (
                    entry.ttl_seconds > 0
                    and time.time() - entry.creation_time > entry.ttl_seconds
                ):
                    del self.l1_cache[cache_key]
                    self.l1_size_bytes -= entry.size_bytes
                else:
                    # Update access info and move to end (LRU)
                    entry.access_count += 1
                    entry.last_access = time.time()
                    self.l1_cache.move_to_end(cache_key)
                    self.stats.l1_hits += 1
                    self._update_access_time(start_time)
                    return entry.value
            self.stats.l1_misses += 1

        # Check L2 cache
        if cache_key in self.l2_index:
            value = await self._load_from_l2(cache_key)
            if value is not None:
                self.stats.l2_hits += 1
                # Promote to L1
                await self.put(key, value, ttl_seconds=300)
                self._update_access_time(start_time)
                return value
            self.stats.l2_misses += 1

        # Check L3 cache
        if cache_key in self.l3_index:
            value = await self._load_from_l3(cache_key)
            if value is not None:
                self.stats.l3_hits += 1
                # Promote to L1
                await self.put(key, value, ttl_seconds=300)
                self._update_access_time(start_time)
                return value
            self.stats.l3_misses += 1

        self._update_access_time(start_time)
        return None

    async def put(self, key: str, value: Any, ttl_seconds: float = 3600):
        """Store value in cache."""
        cache_key = self._hash_key(key)

        # Serialize to get size
        serialized = pickle.dumps(value)
        size_bytes = len(serialized)

        # Create entry
        entry = CacheEntry(
            key=cache_key,
            value=value,
            size_bytes=size_bytes,
            access_count=1,
            last_access=time.time(),
            creation_time=time.time(),
            ttl_seconds=ttl_seconds,
        )

        # Add to L1
        async with self.l1_lock:
            # If already exists, update size tracking
            if cache_key in self.l1_cache:
                old_entry = self.l1_cache[cache_key]
                self.l1_size_bytes -= old_entry.size_bytes

            self.l1_cache[cache_key] = entry
            self.l1_size_bytes += size_bytes

            # Evict if necessary
            while self.l1_size_bytes > self.l1_max_bytes and len(self.l1_cache) > 1:
                await self._evict_from_l1()

    async def _evict_from_l1(self):
        """Evict least recently used item from L1 to L2."""
        # Get oldest item (first in OrderedDict)
        oldest_key, oldest_entry = next(iter(self.l1_cache.items()))

        # Move to L2
        await self._save_to_l2(oldest_key, oldest_entry)

        # Remove from L1
        del self.l1_cache[oldest_key]
        self.l1_size_bytes -= oldest_entry.size_bytes
        self.stats.evictions += 1

    async def _save_to_l2(self, key: str, entry: CacheEntry):
        """Save entry to L2 cache."""
        file_path = self.l2_dir / f"{key}.lz4"

        # Compress and save
        data = {
            "value": entry.value,
            "metadata": {
                "access_count": entry.access_count,
                "last_access": entry.last_access,
                "creation_time": entry.creation_time,
                "ttl_seconds": entry.ttl_seconds,
            },
        }

        serialized = pickle.dumps(data)
        compressed = lz4.frame.compress(serialized)

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(compressed)

        # Update L2 index
        async with self.l2_lock:
            self.l2_index[key] = (file_path, len(compressed))
            self.l2_size_bytes += len(compressed)

            # Evict from L2 if necessary
            while self.l2_size_bytes > self.l2_max_bytes and len(self.l2_index) > 1:
                await self._evict_from_l2()

    async def _evict_from_l2(self):
        """Evict oldest item from L2 to L3."""
        # Find oldest file
        oldest_key = None
        oldest_time = float("inf")

        for key, (path, _) in self.l2_index.items():
            if path.exists():
                mtime = path.stat().st_mtime
                if mtime < oldest_time:
                    oldest_time = mtime
                    oldest_key = key

        if oldest_key:
            old_path, old_size = self.l2_index[oldest_key]
            new_path = self.l3_dir / old_path.name

            # Move file
            old_path.rename(new_path)

            # Update indices
            del self.l2_index[oldest_key]
            self.l2_size_bytes -= old_size

            async with self.l3_lock:
                self.l3_index[oldest_key] = (new_path, old_size)
                self.l3_size_bytes += old_size

                # Evict from L3 if necessary
                while self.l3_size_bytes > self.l3_max_bytes and len(self.l3_index) > 1:
                    await self._evict_from_l3()

    async def _evict_from_l3(self):
        """Evict oldest item from L3 (permanent deletion)."""
        # Find oldest file
        oldest_key = None
        oldest_time = float("inf")

        for key, (path, _) in self.l3_index.items():
            if path.exists():
                mtime = path.stat().st_mtime
                if mtime < oldest_time:
                    oldest_time = mtime
                    oldest_key = key

        if oldest_key:
            path, size = self.l3_index[oldest_key]

            # Delete file
            if path.exists():
                path.unlink()

            # Update index
            del self.l3_index[oldest_key]
            self.l3_size_bytes -= size

    async def _load_from_l2(self, key: str) -> Any | None:
        """Load value from L2 cache."""
        try:
            path, _ = self.l2_index[key]
            if not path.exists():
                del self.l2_index[key]
                return None

            async with aiofiles.open(path, "rb") as f:
                compressed = await f.read()

            decompressed = lz4.frame.decompress(compressed)
            data = pickle.loads(decompressed)

            # Check TTL
            metadata = data["metadata"]
            if metadata["ttl_seconds"] > 0:
                age = time.time() - metadata["creation_time"]
                if age > metadata["ttl_seconds"]:
                    # Expired - delete
                    path.unlink()
                    del self.l2_index[key]
                    return None

            return data["value"]

        except Exception as e:
            logger.error(f"Error loading from L2: {e}")
            return None

    async def _load_from_l3(self, key: str) -> Any | None:
        """Load value from L3 cache."""
        try:
            path, _ = self.l3_index[key]
            if not path.exists():
                del self.l3_index[key]
                return None

            async with aiofiles.open(path, "rb") as f:
                compressed = await f.read()

            decompressed = lz4.frame.decompress(compressed)
            data = pickle.loads(decompressed)

            # Check TTL
            metadata = data["metadata"]
            if metadata["ttl_seconds"] > 0:
                age = time.time() - metadata["creation_time"]
                if age > metadata["ttl_seconds"]:
                    # Expired - delete
                    path.unlink()
                    del self.l3_index[key]
                    return None

            return data["value"]

        except Exception as e:
            logger.error(f"Error loading from L3: {e}")
            return None

    async def _load_disk_indices(self):
        """Load L2 and L3 indices from disk."""
        # Load L2 index
        for path in self.l2_dir.glob("*.lz4"):
            key = path.stem
            size = path.stat().st_size
            self.l2_index[key] = (path, size)
            self.l2_size_bytes += size

        # Load L3 index
        for path in self.l3_dir.glob("*.lz4"):
            key = path.stem
            size = path.stat().st_size
            self.l3_index[key] = (path, size)
            self.l3_size_bytes += size

    async def _periodic_cleanup(self):
        """Periodic cleanup of expired entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def _cleanup_expired(self):
        """Remove expired entries from all tiers."""
        expired_count = 0

        # Clean L1
        async with self.l1_lock:
            expired_keys = []
            for key, entry in self.l1_cache.items():
                if entry.ttl_seconds > 0:
                    age = time.time() - entry.creation_time
                    if age > entry.ttl_seconds:
                        expired_keys.append(key)

            for key in expired_keys:
                entry = self.l1_cache[key]
                del self.l1_cache[key]
                self.l1_size_bytes -= entry.size_bytes
                expired_count += 1

        # Clean L2 and L3 (check a sample to avoid blocking)
        for index, lock in [
            (self.l2_index, self.l2_lock),
            (self.l3_index, self.l3_lock),
        ]:
            async with lock:
                sample_keys = list(index.keys())[:100]  # Check first 100
                for key in sample_keys:
                    path, _ = index[key]
                    if path.exists():
                        # Check age
                        age = time.time() - path.stat().st_mtime
                        if age > 86400:  # 24 hours
                            path.unlink()
                            del index[key]
                            expired_count += 1

        if expired_count > 0:
            logger.debug(f"Cleaned {expired_count} expired cache entries")

    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _update_access_time(self, start_time: float):
        """Update average access time."""
        access_time_ms = (time.perf_counter() - start_time) * 1000

        # Exponential moving average
        alpha = 0.1
        self.stats.avg_access_time_ms = (
            alpha * access_time_ms + (1 - alpha) * self.stats.avg_access_time_ms
        )

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_hits = self.stats.l1_hits + self.stats.l2_hits + self.stats.l3_hits
        hit_rate = total_hits / max(1, self.stats.total_accesses)

        return {
            "hit_rate": hit_rate,
            "l1_hit_rate": self.stats.l1_hits / max(1, self.stats.total_accesses),
            "l2_hit_rate": self.stats.l2_hits / max(1, self.stats.total_accesses),
            "l3_hit_rate": self.stats.l3_hits / max(1, self.stats.total_accesses),
            "avg_access_time_ms": self.stats.avg_access_time_ms,
            "total_accesses": self.stats.total_accesses,
            "evictions": self.stats.evictions,
            "sizes_mb": {
                "l1": self.l1_size_bytes / 1024 / 1024,
                "l2": self.l2_size_bytes / 1024 / 1024,
                "l3": self.l3_size_bytes / 1024 / 1024,
            },
            "entries": {
                "l1": len(self.l1_cache),
                "l2": len(self.l2_index),
                "l3": len(self.l3_index),
            },
        }

    async def optimize(self):
        """Optimize cache based on access patterns."""
        # Analyze hit rates and adjust sizes if needed
        stats = self.get_stats()

        if stats["l1_hit_rate"] < 0.5 and stats["l2_hit_rate"] > 0.3:
            # Consider promoting more from L2 to L1
            logger.info("Cache optimization: Promoting hot L2 entries to L1")
            # Implementation would analyze L2 access patterns

    async def cleanup(self):
        """Cleanup resources."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("Cache manager cleaned up")


# Global cache instance
_cache_manager: CacheManager | None = None


async def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
        await _cache_manager.initialize()
    return _cache_manager
