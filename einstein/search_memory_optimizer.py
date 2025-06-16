#!/usr/bin/env python3
"""
Search Memory Optimizer for Einstein

Optimizes memory usage for search operations through:
1. Intelligent garbage collection scheduling
2. Memory pool management for search results
3. Weak reference caching to prevent memory leaks
4. Memory pressure detection and response
5. Object lifecycle management
"""

import asyncio
import gc
import logging
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Optional, Set, Dict, List

import psutil

from einstein.einstein_config import get_einstein_config

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    
    timestamp: float
    rss_mb: float          # Resident Set Size
    vms_mb: float          # Virtual Memory Size
    percent: float         # Memory percentage
    available_mb: float    # Available system memory
    gc_collections: int    # Number of GC collections
    gc_collected: int      # Objects collected by GC
    weak_refs: int         # Active weak references


class MemoryPool:
    """Memory pool for reusing search result objects."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.pool = deque(maxlen=max_size)
        self.allocations = 0
        self.reuses = 0
    
    def get_object(self, object_type: type = dict) -> Any:
        """Get object from pool or create new one."""
        if self.pool:
            obj = self.pool.popleft()
            # Reset object state
            if hasattr(obj, 'clear'):
                obj.clear()
            self.reuses += 1
            return obj
        else:
            self.allocations += 1
            return object_type()
    
    def return_object(self, obj: Any):
        """Return object to pool."""
        if len(self.pool) < self.max_size:
            self.pool.append(obj)
    
    def get_stats(self) -> dict:
        """Get pool statistics."""
        total_ops = self.allocations + self.reuses
        reuse_rate = (self.reuses / total_ops * 100) if total_ops > 0 else 0
        
        return {
            "pool_size": len(self.pool),
            "max_size": self.max_size,
            "allocations": self.allocations,
            "reuses": self.reuses,
            "reuse_rate_percent": reuse_rate,
        }


class WeakReferenceCache:
    """Cache using weak references to prevent memory leaks."""
    
    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self._cache: Dict[str, weakref.ReferenceType] = {}
        self._access_order = deque()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get object from cache."""
        if key in self._cache:
            ref = self._cache[key]
            obj = ref()  # Dereference weak reference
            
            if obj is not None:
                # Move to end (most recently used)
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
                self._access_order.append(key)
                self.hits += 1
                return obj
            else:
                # Object was garbage collected
                del self._cache[key]
        
        self.misses += 1
        return None
    
    def put(self, key: str, obj: Any):
        """Store object in cache with weak reference."""
        # Clean up dead references first
        self._cleanup_dead_references()
        
        # Evict if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            old_key = self._access_order.popleft()
            if old_key in self._cache:
                del self._cache[old_key]
                self.evictions += 1
        
        # Store with weak reference and callback to clean up
        def cleanup_callback(ref):
            if key in self._cache and self._cache[key] is ref:
                del self._cache[key]
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
        
        self._cache[key] = weakref.ref(obj, cleanup_callback)
        self._access_order.append(key)
    
    def _cleanup_dead_references(self):
        """Remove dead weak references."""
        dead_keys = []
        for key, ref in self._cache.items():
            if ref() is None:
                dead_keys.append(key)
        
        for key in dead_keys:
            del self._cache[key]
            try:
                self._access_order.remove(key)
            except ValueError:
                pass
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        # Count live references
        live_refs = sum(1 for ref in self._cache.values() if ref() is not None)
        
        return {
            "cache_size": len(self._cache),
            "live_references": live_refs,
            "hit_rate_percent": hit_rate,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


class GarbageCollectionScheduler:
    """Intelligent garbage collection scheduler."""
    
    def __init__(self, config):
        self.config = config
        self.last_gc_time = time.time()
        self.gc_interval_seconds = 30.0
        self.memory_threshold_mb = 800  # Trigger GC if above this
        self.force_gc_threshold_mb = 1200  # Force full GC if above this
        
        # GC statistics
        self.gc_runs = 0
        self.objects_collected = 0
        self.time_spent_gc_ms = 0
    
    async def schedule_gc_if_needed(self) -> bool:
        """Schedule garbage collection if needed."""
        current_time = time.time()
        current_memory = self._get_memory_usage_mb()
        
        should_gc = False
        gc_type = "incremental"
        
        # Force GC if memory usage is very high
        if current_memory > self.force_gc_threshold_mb:
            should_gc = True
            gc_type = "full"
        
        # Regular GC if above threshold and enough time passed
        elif (current_memory > self.memory_threshold_mb and 
              current_time - self.last_gc_time > self.gc_interval_seconds):
            should_gc = True
            gc_type = "incremental"
        
        # Periodic GC regardless of memory usage
        elif current_time - self.last_gc_time > self.gc_interval_seconds * 2:
            should_gc = True
            gc_type = "maintenance"
        
        if should_gc:
            await self._run_garbage_collection(gc_type)
            return True
        
        return False
    
    async def _run_garbage_collection(self, gc_type: str):
        """Run garbage collection."""
        start_time = time.time()
        
        try:
            if gc_type == "full":
                # Full GC - collect all generations
                collected = 0
                for generation in range(3):
                    collected += gc.collect(generation)
                logger.info(f"Full GC collected {collected} objects")
            
            elif gc_type == "incremental":
                # Incremental GC - just generation 0
                collected = gc.collect(0)
                logger.debug(f"Incremental GC collected {collected} objects")
            
            else:  # maintenance
                # Light GC for maintenance
                collected = gc.collect(0)
                logger.debug(f"Maintenance GC collected {collected} objects")
            
            # Update statistics
            self.gc_runs += 1
            self.objects_collected += collected
            self.time_spent_gc_ms += (time.time() - start_time) * 1000
            self.last_gc_time = time.time()
            
            # Log if significant collection
            if collected > 100:
                logger.info(f"GC ({gc_type}) collected {collected} objects in {(time.time() - start_time) * 1000:.1f}ms")
        
        except Exception as e:
            logger.error(f"Garbage collection failed: {e}")
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0
    
    def get_stats(self) -> dict:
        """Get GC statistics."""
        return {
            "gc_runs": self.gc_runs,
            "objects_collected": self.objects_collected,
            "time_spent_gc_ms": self.time_spent_gc_ms,
            "avg_gc_time_ms": self.time_spent_gc_ms / self.gc_runs if self.gc_runs > 0 else 0,
            "last_gc_seconds_ago": time.time() - self.last_gc_time,
        }


class MemoryPressureDetector:
    """Detects and responds to memory pressure."""
    
    def __init__(self, config):
        self.config = config
        self.memory_snapshots = deque(maxlen=100)
        self.pressure_callbacks = []
        
        # Pressure thresholds
        self.warning_threshold_percent = 80  # System memory usage
        self.critical_threshold_percent = 90
        self.process_warning_mb = 1000
        self.process_critical_mb = 1500
    
    def add_pressure_callback(self, callback):
        """Add callback for memory pressure events."""
        self.pressure_callbacks.append(callback)
    
    def check_memory_pressure(self) -> dict:
        """Check current memory pressure and return status."""
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            system_usage_percent = system_memory.percent
            
            # Process memory
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / (1024 * 1024)
            
            # Determine pressure level
            pressure_level = "normal"
            
            if (system_usage_percent > self.critical_threshold_percent or 
                process_memory_mb > self.process_critical_mb):
                pressure_level = "critical"
            elif (system_usage_percent > self.warning_threshold_percent or
                  process_memory_mb > self.process_warning_mb):
                pressure_level = "warning"
            
            pressure_info = {
                "level": pressure_level,
                "system_usage_percent": system_usage_percent,
                "process_usage_mb": process_memory_mb,
                "available_mb": system_memory.available / (1024 * 1024),
                "recommendations": self._get_pressure_recommendations(pressure_level),
            }
            
            # Record snapshot
            snapshot = MemorySnapshot(
                timestamp=time.time(),
                rss_mb=process_memory_mb,
                vms_mb=process.memory_info().vms / (1024 * 1024),
                percent=system_usage_percent,
                available_mb=system_memory.available / (1024 * 1024),
                gc_collections=sum(gc.get_stats()),
                gc_collected=0,  # Would need to track separately
                weak_refs=len(gc.get_referrers()),
            )
            self.memory_snapshots.append(snapshot)
            
            # Trigger callbacks if pressure detected
            if pressure_level != "normal":
                for callback in self.pressure_callbacks:
                    try:
                        callback(pressure_info)
                    except Exception as e:
                        logger.error(f"Memory pressure callback failed: {e}")
            
            return pressure_info
            
        except Exception as e:
            logger.error(f"Memory pressure check failed: {e}")
            return {"level": "unknown", "error": str(e)}
    
    def _get_pressure_recommendations(self, pressure_level: str) -> List[str]:
        """Get recommendations based on pressure level."""
        if pressure_level == "critical":
            return [
                "Force garbage collection immediately",
                "Clear all non-essential caches",
                "Reduce search result limits",
                "Consider restarting the search service",
            ]
        elif pressure_level == "warning":
            return [
                "Schedule garbage collection",
                "Clear older cache entries", 
                "Reduce concurrent search operations",
                "Monitor memory usage closely",
            ]
        else:
            return ["Memory usage is normal"]
    
    def get_memory_trend(self) -> dict:
        """Get memory usage trend analysis."""
        if len(self.memory_snapshots) < 10:
            return {"status": "insufficient_data"}
        
        recent_snapshots = list(self.memory_snapshots)[-20:]
        memory_usage = [s.rss_mb for s in recent_snapshots]
        
        # Calculate trend
        import numpy as np
        x = np.arange(len(memory_usage))
        trend_slope = np.polyfit(x, memory_usage, 1)[0]
        
        return {
            "trend": "increasing" if trend_slope > 1 else "stable" if abs(trend_slope) < 1 else "decreasing",
            "slope_mb_per_sample": trend_slope,
            "current_usage_mb": memory_usage[-1],
            "peak_usage_mb": max(memory_usage),
            "min_usage_mb": min(memory_usage),
        }


class SearchMemoryOptimizer:
    """Main memory optimizer for search operations."""
    
    def __init__(self, project_root: Optional[str] = None):
        self.config = get_einstein_config()
        
        # Core components
        self.memory_pool = MemoryPool(max_size=2000)
        self.weak_cache = WeakReferenceCache(max_size=5000)
        self.gc_scheduler = GarbageCollectionScheduler(self.config)
        self.pressure_detector = MemoryPressureDetector(self.config)
        
        # Optimization state
        self.optimization_active = False
        self.optimization_task = None
        self.check_interval_seconds = 15
        
        # Statistics
        self.optimizations_run = 0
        self.memory_freed_mb = 0
        
        # Set up pressure response
        self.pressure_detector.add_pressure_callback(self._handle_memory_pressure)
    
    async def start_optimization(self):
        """Start memory optimization loop."""
        if self.optimization_active:
            logger.warning("Memory optimization already active")
            return
        
        logger.info("Starting search memory optimization...")
        self.optimization_active = True
        self.optimization_task = asyncio.create_task(self._optimization_loop())
    
    async def stop_optimization(self):
        """Stop memory optimization."""
        if not self.optimization_active:
            return
        
        logger.info("Stopping search memory optimization...")
        self.optimization_active = False
        
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
    
    async def _optimization_loop(self):
        """Main optimization loop."""
        try:
            while self.optimization_active:
                # Check memory pressure
                pressure_info = self.pressure_detector.check_memory_pressure()
                
                # Schedule GC if needed
                gc_ran = await self.gc_scheduler.schedule_gc_if_needed()
                
                # Run optimizations if pressure detected
                if pressure_info["level"] != "normal":
                    await self._run_memory_optimization()
                
                # Log status periodically
                if self.optimizations_run % 10 == 0:
                    await self._log_memory_status()
                
                await asyncio.sleep(self.check_interval_seconds)
                
        except asyncio.CancelledError:
            logger.info("Memory optimization loop cancelled")
        except Exception as e:
            logger.error(f"Memory optimization loop failed: {e}")
    
    async def _run_memory_optimization(self):
        """Run memory optimization procedures."""
        start_memory = self._get_memory_usage_mb()
        
        try:
            # Clear weak reference cache of dead references
            self.weak_cache._cleanup_dead_references()
            
            # Force garbage collection
            await self.gc_scheduler._run_garbage_collection("full")
            
            # Clear memory pool if needed
            if len(self.memory_pool.pool) > self.memory_pool.max_size // 2:
                self.memory_pool.pool.clear()
                logger.debug("Cleared memory pool")
            
            self.optimizations_run += 1
            
            # Calculate memory freed
            end_memory = self._get_memory_usage_mb()
            memory_freed = start_memory - end_memory
            
            if memory_freed > 0:
                self.memory_freed_mb += memory_freed
                logger.info(f"Memory optimization freed {memory_freed:.1f}MB")
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
    
    def _handle_memory_pressure(self, pressure_info: dict):
        """Handle memory pressure events."""
        level = pressure_info["level"]
        
        if level == "critical":
            logger.warning("Critical memory pressure detected - running emergency optimization")
            # Run optimization immediately
            asyncio.create_task(self._run_memory_optimization())
        
        elif level == "warning":
            logger.info("Memory pressure warning - scheduling optimization")
            # Schedule optimization for next cycle
            pass
    
    async def _log_memory_status(self):
        """Log current memory status."""
        pressure_info = self.pressure_detector.check_memory_pressure()
        trend = self.pressure_detector.get_memory_trend()
        
        logger.info(
            f"Memory status: {pressure_info['process_usage_mb']:.1f}MB used, "
            f"{pressure_info['level']} pressure, trend: {trend.get('trend', 'unknown')}"
        )
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0
    
    def get_object_from_pool(self, object_type: type = dict) -> Any:
        """Get object from memory pool."""
        return self.memory_pool.get_object(object_type)
    
    def return_object_to_pool(self, obj: Any):
        """Return object to memory pool."""
        self.memory_pool.return_object(obj)
    
    def cache_with_weak_ref(self, key: str, obj: Any):
        """Cache object with weak reference."""
        self.weak_cache.put(key, obj)
    
    def get_from_weak_cache(self, key: str) -> Optional[Any]:
        """Get object from weak reference cache."""
        return self.weak_cache.get(key)
    
    def get_optimization_stats(self) -> dict:
        """Get comprehensive optimization statistics."""
        pressure_info = self.pressure_detector.check_memory_pressure()
        trend = self.pressure_detector.get_memory_trend()
        
        return {
            "optimization_active": self.optimization_active,
            "optimizations_run": self.optimizations_run,
            "memory_freed_mb": self.memory_freed_mb,
            "current_memory": pressure_info,
            "memory_trend": trend,
            "memory_pool_stats": self.memory_pool.get_stats(),
            "weak_cache_stats": self.weak_cache.get_stats(),
            "gc_stats": self.gc_scheduler.get_stats(),
        }


# Global optimizer instance
_memory_optimizer = None


def get_search_memory_optimizer() -> SearchMemoryOptimizer:
    """Get global search memory optimizer instance."""
    global _memory_optimizer
    
    if _memory_optimizer is None:
        _memory_optimizer = SearchMemoryOptimizer()
    
    return _memory_optimizer


if __name__ == "__main__":
    async def demo_memory_optimization():
        """Demonstrate memory optimization capabilities."""
        print("🧠 Search Memory Optimizer Demo")
        print("=" * 40)
        
        optimizer = get_search_memory_optimizer()
        
        # Start optimization
        await optimizer.start_optimization()
        
        print("\n1️⃣ Creating memory pressure...")
        
        # Create some objects to simulate memory usage
        large_objects = []
        for i in range(1000):
            # Create large objects
            obj = {f"key_{j}": f"value_{j}" * 100 for j in range(100)}
            large_objects.append(obj)
            
            # Use memory pool occasionally
            if i % 10 == 0:
                pool_obj = optimizer.get_object_from_pool()
                pool_obj.update({"pooled": True, "index": i})
                optimizer.return_object_to_pool(pool_obj)
            
            # Use weak cache
            if i % 5 == 0:
                optimizer.cache_with_weak_ref(f"cache_key_{i}", obj)
        
        print(f"   Created {len(large_objects)} large objects")
        
        # Check initial memory status
        stats = optimizer.get_optimization_stats()
        print(f"   Memory usage: {stats['current_memory']['process_usage_mb']:.1f}MB")
        print(f"   Memory pressure: {stats['current_memory']['level']}")
        
        # Wait for optimization to run
        await asyncio.sleep(5)
        
        print("\n2️⃣ Triggering optimization...")
        
        # Clear references to trigger garbage collection
        large_objects.clear()
        
        # Force optimization run
        await optimizer._run_memory_optimization()
        
        # Wait for GC
        await asyncio.sleep(2)
        
        # Check final stats
        final_stats = optimizer.get_optimization_stats()
        
        print("\n3️⃣ Optimization Results:")
        print(f"   Memory usage: {final_stats['current_memory']['process_usage_mb']:.1f}MB")
        print(f"   Memory freed: {final_stats['memory_freed_mb']:.1f}MB")
        print(f"   Optimizations run: {final_stats['optimizations_run']}")
        print(f"   Memory trend: {final_stats['memory_trend'].get('trend', 'unknown')}")
        
        print("\n4️⃣ Component Statistics:")
        pool_stats = final_stats['memory_pool_stats']
        cache_stats = final_stats['weak_cache_stats']
        gc_stats = final_stats['gc_stats']
        
        print(f"   Memory Pool:")
        print(f"     - Pool size: {pool_stats['pool_size']}/{pool_stats['max_size']}")
        print(f"     - Reuse rate: {pool_stats['reuse_rate_percent']:.1f}%")
        
        print(f"   Weak Cache:")
        print(f"     - Cache size: {cache_stats['cache_size']}")
        print(f"     - Hit rate: {cache_stats['hit_rate_percent']:.1f}%")
        print(f"     - Live references: {cache_stats['live_references']}")
        
        print(f"   Garbage Collection:")
        print(f"     - GC runs: {gc_stats['gc_runs']}")
        print(f"     - Objects collected: {gc_stats['objects_collected']}")
        print(f"     - Average GC time: {gc_stats['avg_gc_time_ms']:.1f}ms")
        
        # Stop optimization
        await optimizer.stop_optimization()
        
        print("\n✅ Memory optimization demo complete!")
    
    asyncio.run(demo_memory_optimization())