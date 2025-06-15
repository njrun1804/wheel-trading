#!/usr/bin/env python3
"""
Memory Optimizer for Einstein System

Optimizes memory usage for Claude Code CLI to stay under configured limits while maintaining performance:
1. Smart memory management with weak references
2. Compressed data structures
3. LRU eviction policies
4. Memory-mapped files for large indexes
5. Garbage collection optimization
6. Memory pool management
"""

import gc
import gzip
import logging
import os
import mmap
import pickle
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import psutil

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MemoryProfile:
    """Memory usage profile."""
    total_mb: float
    heap_mb: float
    cache_mb: float
    indexes_mb: float
    system_available_mb: float
    gc_collections: int
    target_usage_mb: float = 2048  # Will be overridden by config


class CompressedLRUCache(Generic[T]):
    """LRU cache with compression for memory efficiency."""
    
    def __init__(self, max_size: int, compress_threshold: int = 1024):
        self.max_size = max_size
        self.compress_threshold = compress_threshold
        self._cache: OrderedDict[str, tuple[T, bool, float]] = OrderedDict()
        self._lock = threading.RLock()
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.compressions = 0
        self.decompressions = 0
    
    def get(self, key: str) -> T | None:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            # Move to end (most recent)
            value, compressed, timestamp = self._cache[key]
            self._cache.move_to_end(key)
            self.hits += 1
            
            if compressed:
                try:
                    value = pickle.loads(gzip.decompress(value))
                    self.decompressions += 1
                except Exception as e:
                    logger.warning(f"Decompression failed for key {key}: {e}",
                                  extra={
                                      'operation': 'cache_decompression',
                                      'error_type': type(e).__name__,
                                      'cache_key': key,
                                      'cache_size': len(self._cache),
                                      'compression_enabled': True,
                                      'compressed_size': len(value) if isinstance(value, bytes) else 'unknown',
                                      'cache_hit_rate': round(self.hits / (self.hits + self.misses) * 100, 1) if (self.hits + self.misses) > 0 else 0
                                  })
                    del self._cache[key]
                    return None
            
            return value
    
    def put(self, key: str, value: T) -> None:
        """Put item in cache."""
        with self._lock:
            # Check if we need to compress
            compressed = False
            stored_value = value
            
            if self._should_compress(value):
                try:
                    compressed_data = gzip.compress(pickle.dumps(value))
                    if len(compressed_data) < len(pickle.dumps(value)):
                        stored_value = compressed_data
                        compressed = True
                        self.compressions += 1
                except Exception as e:
                    logger.warning(f"Compression failed for key {key}: {e}",
                                  extra={
                                      'operation': 'cache_compression',
                                      'error_type': type(e).__name__,
                                      'cache_key': key,
                                      'cache_size': len(self._cache),
                                      'compression_threshold': self.compress_threshold,
                                      'value_type': type(value).__name__,
                                      'max_cache_size': self.max_size
                                  })
            
            # Store in cache
            self._cache[key] = (stored_value, compressed, time.time())
            
            # Move to end if already exists
            if key in self._cache:
                self._cache.move_to_end(key)
            
            # Evict if necessary
            while len(self._cache) > self.max_size:
                self._evict_lru()
    
    def _should_compress(self, value: T) -> bool:
        """Determine if value should be compressed."""
        try:
            size = len(pickle.dumps(value))
            return size > self.compress_threshold
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            logger.debug(f"Failed to serialize value for compression check: {e}",
                        extra={
                            'operation': 'compression_check',
                            'error_type': type(e).__name__,
                            'value_type': type(value).__name__,
                            'compression_threshold': self.compress_threshold
                        })
            return False
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if self._cache:
            self._cache.popitem(last=False)
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': round(hit_rate * 100, 1),
                'hits': self.hits,
                'misses': self.misses,
                'compressions': self.compressions,
                'decompressions': self.decompressions
            }


class MemoryMappedIndex:
    """Memory-mapped index for large data with minimal RAM usage."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._mmap_file = None
        self._mmap = None
        self._index: dict[str, tuple[int, int]] | None = None  # key -> (offset, length)
        self._loaded = False
    
    def load(self) -> bool:
        """Load the memory-mapped index."""
        try:
            if not self.file_path.exists():
                return False
            
            self._mmap_file = open(self.file_path, 'rb')
            self._mmap = mmap.mmap(self._mmap_file.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Load index from end of file (last 8KB)
            file_size = self.file_path.stat().st_size
            if file_size > 8192:
                self._mmap.seek(file_size - 8192)
                index_data = self._mmap.read(8192)
                try:
                    self._index = pickle.loads(gzip.decompress(index_data))
                    self._loaded = True
                    return True
                except (pickle.PickleError, gzip.BadGzipFile, EOFError, ValueError) as e:
                    logger.debug(f"Failed to decompress/deserialize memory-mapped index from {self.file_path}: {e}",
                                extra={
                                    'operation': 'decompress_mmap_index',
                                    'error_type': type(e).__name__,
                                    'file_path': str(self.file_path),
                                    'file_size': self.file_path.stat().st_size if self.file_path.exists() else 0,
                                    'index_offset': file_size - 8192,
                                    'index_data_size': len(index_data) if 'index_data' in locals() else 0
                                })
                    pass
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to load memory-mapped index {self.file_path}: {e}", exc_info=True,
                        extra={
                            'operation': 'load_memory_mapped_index',
                            'error_type': type(e).__name__,
                            'file_path': str(self.file_path),
                            'file_exists': self.file_path.exists(),
                            'file_size': self.file_path.stat().st_size if self.file_path.exists() else 0,
                            'mmap_available': hasattr(self, '_mmap_file'),
                            'file_readable': self.file_path.is_file() and os.access(self.file_path, os.R_OK) if self.file_path.exists() else False
                        })
            return False
    
    def get(self, key: str) -> bytes | None:
        """Get data by key."""
        if not self._loaded or not self._index or key not in self._index:
            return None
        
        try:
            offset, length = self._index[key]
            self._mmap.seek(offset)
            return self._mmap.read(length)
        except Exception as e:
            logger.error(f"Failed to read from memory-mapped index: {e}", exc_info=True,
                        extra={
                            'operation': 'mmap_index_read',
                            'error_type': type(e).__name__,
                            'key': key,
                            'index_loaded': self._loaded,
                            'index_size': len(self._index) if self._index else 0,
                            'mmap_available': self._mmap is not None,
                            'offset': offset if 'offset' in locals() else 'unknown',
                            'length': length if 'length' in locals() else 'unknown'
                        })
            return None
    
    def close(self):
        """Close memory-mapped file."""
        if self._mmap:
            self._mmap.close()
        if self._mmap_file:
            self._mmap_file.close()
        self._loaded = False


class MemoryPool:
    """Pool for reusing memory allocations."""
    
    def __init__(self):
        self._pools: dict[int, list[bytes]] = {}
        self._lock = threading.Lock()
        self._allocated = 0
        self._reused = 0
    
    def get_buffer(self, size: int) -> bytes:
        """Get buffer from pool or allocate new one."""
        # Round up to nearest power of 2 for better pooling
        pool_size = 1
        while pool_size < size:
            pool_size *= 2
        
        with self._lock:
            pool = self._pools.get(pool_size, [])
            
            if pool:
                self._reused += 1
                return pool.pop()
            else:
                self._allocated += 1
                return bytearray(pool_size)
    
    def return_buffer(self, buffer: bytes):
        """Return buffer to pool."""
        size = len(buffer)
        
        with self._lock:
            pool = self._pools.setdefault(size, [])
            
            # Limit pool size to prevent unbounded growth
            if len(pool) < 10:
                pool.append(buffer)
    
    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            total_buffers = sum(len(pool) for pool in self._pools.values())
            return {
                'allocated': self._allocated,
                'reused': self._reused,
                'pooled_buffers': total_buffers,
                'pool_sizes': {str(size): len(pool) for size, pool in self._pools.items()}
            }


class WeakValueCache:
    """Cache using weak references to allow garbage collection."""
    
    def __init__(self):
        self._cache: dict[str, weakref.ref] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Any | None:
        """Get value from weak cache."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            ref = self._cache[key]
            value = ref()
            
            if value is None:
                # Object was garbage collected
                del self._cache[key]
                self.evictions += 1
                self.misses += 1
                return None
            
            self.hits += 1
            return value
    
    def put(self, key: str, value: Any):
        """Put value in weak cache."""
        def cleanup(ref):
            with self._lock:
                if key in self._cache and self._cache[key] is ref:
                    del self._cache[key]
                    self.evictions += 1
        
        with self._lock:
            self._cache[key] = weakref.ref(value, cleanup)
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': round(self.hits / (self.hits + self.misses) * 100, 1) if (self.hits + self.misses) > 0 else 0
            }


class MemoryOptimizer:
    """Main memory optimization manager."""
    
    def __init__(self, target_usage_mb: float = 2048):
        self.target_usage_mb = target_usage_mb
        
        # Get Einstein config
        from .einstein_config import get_einstein_config
        self.config = get_einstein_config()
        
        # Memory management components
        self.search_cache = CompressedLRUCache[list[dict[str, Any]]](max_size=self.config.cache.search_cache_size)
        self.file_cache = CompressedLRUCache[str](max_size=self.config.cache.file_cache_size)
        self.weak_cache = WeakValueCache()
        self.memory_pool = MemoryPool()
        
        # Memory-mapped indexes
        self.mmap_indexes: dict[str, MemoryMappedIndex] = {}
        
        # GC optimization
        self._gc_threshold = (700, 10, 10)  # More aggressive collection
        self._last_gc_time = time.time()
        self._gc_interval = self.config.monitoring.gc_interval_s
        
        # Memory monitoring
        self._memory_history: list[tuple[float, float]] = []  # (timestamp, usage_mb)
        self._monitoring_active = False
        
        self._setup_gc_optimization()
    
    def _setup_gc_optimization(self):
        """Setup optimized garbage collection."""
        gc.set_threshold(*self._gc_threshold)
        
        # Disable automatic GC for generation 2 (we'll handle it manually)
        gc.disable()
        gc.set_debug(0)  # Disable GC debugging for performance
    
    async def optimize_memory_usage(self):
        """Optimize current memory usage."""
        logger.info("🧹 Optimizing memory usage...")
        
        # Get current memory usage
        current_usage = self.get_memory_usage_mb()
        
        if current_usage > self.target_usage_mb:
            # Apply aggressive optimization
            freed_mb = self._aggressive_cleanup()
            logger.info(f"Freed {freed_mb:.1f}MB through aggressive cleanup")
        
        # Run garbage collection
        self._run_gc()
        
        # Update memory history
        self._memory_history.append((time.time(), current_usage))
        
        # Keep only last 100 measurements
        if len(self._memory_history) > 100:
            self._memory_history = self._memory_history[-100:]
        
        new_usage = self.get_memory_usage_mb()
        freed_total = current_usage - new_usage
        
        logger.info(f"Memory optimization complete: {current_usage:.1f}MB -> {new_usage:.1f}MB (freed {freed_total:.1f}MB)")
        
        return freed_total
    
    def _aggressive_cleanup(self) -> float:
        """Perform aggressive memory cleanup."""
        initial_usage = self.get_memory_usage_mb()
        
        # Clear caches
        self.search_cache.clear()
        self.file_cache.clear()
        self.weak_cache.clear()
        
        # Close memory-mapped files
        for mmap_index in self.mmap_indexes.values():
            mmap_index.close()
        self.mmap_indexes.clear()
        
        # Force garbage collection
        self._run_gc()
        
        final_usage = self.get_memory_usage_mb()
        return initial_usage - final_usage
    
    def _run_gc(self) -> int:
        """Run optimized garbage collection."""
        start_time = time.time()
        
        # Collect all generations
        collected = 0
        for generation in range(3):
            collected += gc.collect(generation)
        
        gc_time = (time.time() - start_time) * 1000
        self._last_gc_time = time.time()
        
        logger.debug(f"GC collected {collected} objects in {gc_time:.1f}ms")
        return collected
    
    def should_run_gc(self) -> bool:
        """Check if garbage collection should be run."""
        return (time.time() - self._last_gc_time) > self._gc_interval
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_memory_profile(self) -> MemoryProfile:
        """Get detailed memory profile."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Estimate cache memory usage
        cache_mb = 0
        try:
            # Rough estimation of cache memory
            cache_stats = [
                self.search_cache.get_stats(),
                self.file_cache.get_stats(),
                self.weak_cache.get_stats()
            ]
            # Each cache entry estimated at ~1KB average
            cache_mb = sum(stats.get('size', 0) for stats in cache_stats) / 1024
        except (AttributeError, TypeError, KeyError) as e:
            logger.debug(f"Failed to calculate cache memory usage: {e}",
                        extra={
                            'operation': 'calculate_cache_memory_usage',
                            'error_type': type(e).__name__,
                            'cache_components': ['search_cache', 'file_cache', 'weak_cache'],
                            'has_search_cache': hasattr(self, 'search_cache'),
                            'has_file_cache': hasattr(self, 'file_cache'),
                            'has_weak_cache': hasattr(self, 'weak_cache')
                        })
            pass
        
        # Get GC stats
        gc_stats = gc.get_stats()
        total_collections = sum(gen['collections'] for gen in gc_stats)
        
        return MemoryProfile(
            total_mb=memory_info.rss / 1024 / 1024,
            heap_mb=memory_info.rss / 1024 / 1024 - cache_mb,  # Rough estimation
            cache_mb=cache_mb,
            indexes_mb=sum(mmap.file_path.stat().st_size / 1024 / 1024 
                          for mmap in self.mmap_indexes.values() 
                          if mmap.file_path.exists()),
            system_available_mb=psutil.virtual_memory().available / 1024 / 1024,
            gc_collections=total_collections
        )
    
    def register_memory_mapped_index(self, name: str, file_path: Path):
        """Register a memory-mapped index."""
        mmap_index = MemoryMappedIndex(file_path)
        if mmap_index.load():
            self.mmap_indexes[name] = mmap_index
            logger.debug(f"Registered memory-mapped index: {name}")
    
    def get_optimization_recommendations(self) -> list[str]:
        """Get memory optimization recommendations."""
        profile = self.get_memory_profile()
        recommendations = []
        
        if profile.total_mb > profile.target_usage_mb:
            recommendations.append(f"Memory usage {profile.total_mb:.1f}MB exceeds target {profile.target_usage_mb:.1f}MB")
        
        if profile.cache_mb > 500:
            recommendations.append(f"Cache usage {profile.cache_mb:.1f}MB is high - consider reducing cache sizes")
        
        if profile.system_available_mb < 1024:
            recommendations.append(f"System memory low ({profile.system_available_mb:.1f}MB available)")
        
        # Check memory growth trend
        if len(self._memory_history) >= 10:
            recent_usage = [usage for _, usage in self._memory_history[-10:]]
            if len(recent_usage) >= 2:
                growth_rate = (recent_usage[-1] - recent_usage[0]) / len(recent_usage)
                if growth_rate > self.config.cache.cache_memory_limit_mb / 50:  # Growing by more than 2% of cache limit per measurement
                    recommendations.append(f"Memory usage growing rapidly ({growth_rate:.1f}MB per interval)")
        
        if not recommendations:
            recommendations.append("Memory usage is optimal")
        
        return recommendations
    
    async def start_memory_monitoring(self, interval_seconds: int = None):
        if interval_seconds is None:
            interval_seconds = self.config.monitoring.memory_check_interval_s
        """Start background memory monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        logger.info(f"Starting memory monitoring (interval: {interval_seconds}s)")
        
        async def monitor_loop():
            while self._monitoring_active:
                try:
                    profile = self.get_memory_profile()
                    
                    # Auto-optimize if memory usage is high
                    if profile.total_mb > profile.target_usage_mb * 0.9:  # 90% of target
                        logger.warning(f"High memory usage detected: {profile.total_mb:.1f}MB")
                        await self.optimize_memory_usage()
                    
                    # Run GC if needed
                    if self.should_run_gc():
                        self._run_gc()
                    
                    await asyncio.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}", exc_info=True,
                                extra={
                                    'operation': 'memory_monitoring',
                                    'error_type': type(e).__name__,
                                    'interval_seconds': interval_seconds,
                                    'monitoring_active': self._monitoring_active,
                                    'memory_usage_mb': self.get_memory_usage_mb(),
                                    'target_usage_mb': self.target_usage_mb,
                                    'gc_due': self.should_run_gc(),
                                    'memory_history_size': len(self._memory_history)
                                })
                    await asyncio.sleep(interval_seconds)
        
        import asyncio
        asyncio.create_task(monitor_loop())
    
    def stop_memory_monitoring(self):
        """Stop background memory monitoring."""
        self._monitoring_active = False
        logger.info("Memory monitoring stopped")
    
    def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'search_cache': self.search_cache.get_stats(),
            'file_cache': self.file_cache.get_stats(),
            'weak_cache': self.weak_cache.get_stats(),
            'memory_pool': self.memory_pool.get_stats(),
            'mmap_indexes': len(self.mmap_indexes)
        }


# Global memory optimizer
_memory_optimizer: MemoryOptimizer | None = None


def get_memory_optimizer() -> MemoryOptimizer:
    """Get the global memory optimizer."""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer


async def benchmark_memory_optimization():
    """Benchmark memory optimization effectiveness."""
    print("🧠 Benchmarking Memory Optimization...")
    
    optimizer = get_memory_optimizer()
    
    # Get baseline memory usage
    initial_profile = optimizer.get_memory_profile()
    print("\n📊 Initial Memory Profile:")
    print(f"   Total usage: {initial_profile.total_mb:.1f}MB")
    print(f"   Cache usage: {initial_profile.cache_mb:.1f}MB")
    print(f"   Available system: {initial_profile.system_available_mb:.1f}MB")
    print(f"   Target usage: {initial_profile.target_usage_mb:.1f}MB")
    
    # Simulate memory usage by adding data to caches
    print("\n📈 Simulating memory usage...")
    
    # Add data to search cache
    for i in range(1000):
        key = f"query_{i}"
        results = [{'content': f'result_{j}' * 100, 'file': f'file_{j}.py', 'line': j} for j in range(10)]
        optimizer.search_cache.put(key, results)
    
    # Add data to file cache
    for i in range(2000):
        key = f"file_{i}"
        content = f"# File content {i}\n" + "def function():\n    pass\n" * 50
        optimizer.file_cache.put(key, content)
    
    # Check memory usage after simulation
    loaded_profile = optimizer.get_memory_profile()
    print(f"   After loading: {loaded_profile.total_mb:.1f}MB (+{loaded_profile.total_mb - initial_profile.total_mb:.1f}MB)")
    
    # Test memory optimization
    print("\n🧹 Testing memory optimization...")
    freed_mb = await optimizer.optimize_memory_usage()
    
    # Get final memory usage
    final_profile = optimizer.get_memory_profile()
    print("\n📊 Final Memory Profile:")
    print(f"   Total usage: {final_profile.total_mb:.1f}MB")
    print(f"   Memory freed: {freed_mb:.1f}MB")
    print(f"   Cache usage: {final_profile.cache_mb:.1f}MB")
    
    # Get cache statistics
    cache_stats = optimizer.get_cache_stats()
    print("\n💾 Cache Statistics:")
    for cache_name, stats in cache_stats.items():
        if isinstance(stats, dict):
            print(f"   {cache_name}:")
            for key, value in stats.items():
                print(f"     {key}: {value}")
    
    # Get optimization recommendations
    recommendations = optimizer.get_optimization_recommendations()
    print("\n💡 Optimization Recommendations:")
    for recommendation in recommendations:
        print(f"   • {recommendation}")
    
    # Performance assessment
    memory_efficiency = (initial_profile.target_usage_mb - final_profile.total_mb) / initial_profile.target_usage_mb * 100
    
    if final_profile.total_mb <= initial_profile.target_usage_mb:
        print("\n✅ Memory optimization successful!")
        print(f"   Efficiency: {memory_efficiency:.1f}% under target")
    else:
        print("\n⚠️ Memory usage still above target")
        print(f"   Overage: {final_profile.total_mb - initial_profile.target_usage_mb:.1f}MB")


if __name__ == "__main__":
    import asyncio
    asyncio.run(benchmark_memory_optimization())