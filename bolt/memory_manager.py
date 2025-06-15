"""
Bolt Memory Manager - Dynamic Memory Budgeting with Overcommit Protection
Prevents memory crashes by enforcing hard limits and graceful degradation
"""

import os
import time
import psutil
import threading
import logging
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager
import weakref
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System constants
TOTAL_SYSTEM_MEMORY_GB = 24
MAX_ALLOCATION_GB = 18  # 75% of 24GB
MEMORY_PRESSURE_THRESHOLD = 0.85  # 85% triggers pressure response
CRITICAL_THRESHOLD = 0.95  # 95% triggers emergency measures

# Component memory budgets (percentage of MAX_ALLOCATION_GB)
COMPONENT_BUDGETS = {
    "duckdb": 0.50,      # 50% = 9GB
    "jarvis": 0.17,      # 17% = 3.06GB
    "einstein": 0.08,    # 8% = 1.44GB
    "meta_system": 0.10, # 10% = 1.8GB
    "cache": 0.10,       # 10% = 1.8GB
    "other": 0.05        # 5% = 0.9GB (buffer)
}

@dataclass
class MemoryAllocation:
    """Tracks a single memory allocation"""
    component: str
    size_bytes: int
    allocated_at: float
    description: str
    can_evict: bool = True
    priority: int = 5  # 1-10, higher = more important
    
@dataclass
class ComponentMemoryStats:
    """Memory statistics for a component"""
    allocated_bytes: int = 0
    peak_bytes: int = 0
    allocation_count: int = 0
    eviction_count: int = 0
    pressure_events: int = 0
    last_gc: float = 0
    
class MemoryPool:
    """Thread-safe memory pool for a component"""
    
    def __init__(self, name: str, max_size_bytes: int):
        self.name = name
        self.max_size_bytes = max_size_bytes
        self.allocated_bytes = 0
        self.allocations: Dict[str, MemoryAllocation] = {}
        self.lock = threading.RLock()
        self.stats = ComponentMemoryStats()
        
    def allocate(self, size_bytes: int, description: str, 
                 can_evict: bool = True, priority: int = 5) -> Optional[str]:
        """Allocate memory from pool, returns allocation ID or None if failed"""
        with self.lock:
            # Check if allocation would exceed limit
            if self.allocated_bytes + size_bytes > self.max_size_bytes:
                # Try to evict lower priority allocations
                evicted = self._evict_for_space(size_bytes, priority)
                if not evicted:
                    logger.warning(f"{self.name}: Cannot allocate {size_bytes} bytes, pool full")
                    return None
                    
            # Create allocation
            alloc_id = f"{self.name}_{int(time.time() * 1000000)}"
            self.allocations[alloc_id] = MemoryAllocation(
                component=self.name,
                size_bytes=size_bytes,
                allocated_at=time.time(),
                description=description,
                can_evict=can_evict,
                priority=priority
            )
            
            self.allocated_bytes += size_bytes
            self.stats.allocated_bytes = self.allocated_bytes
            self.stats.allocation_count += 1
            
            if self.allocated_bytes > self.stats.peak_bytes:
                self.stats.peak_bytes = self.allocated_bytes
                
            return alloc_id
            
    def deallocate(self, alloc_id: str) -> bool:
        """Deallocate memory"""
        with self.lock:
            if alloc_id not in self.allocations:
                return False
                
            alloc = self.allocations.pop(alloc_id)
            self.allocated_bytes -= alloc.size_bytes
            self.stats.allocated_bytes = self.allocated_bytes
            return True
            
    def _evict_for_space(self, needed_bytes: int, min_priority: int) -> bool:
        """Evict lower priority allocations to make space"""
        evictable = sorted(
            [a for a in self.allocations.values() if a.can_evict and a.priority < min_priority],
            key=lambda x: (x.priority, x.allocated_at)
        )
        
        evicted_bytes = 0
        evicted_ids = []
        
        for alloc in evictable:
            if evicted_bytes >= needed_bytes:
                break
            evicted_bytes += alloc.size_bytes
            evicted_ids.append([k for k, v in self.allocations.items() if v == alloc][0])
            
        if evicted_bytes < needed_bytes:
            return False
            
        # Perform evictions
        for alloc_id in evicted_ids:
            self.deallocate(alloc_id)
            self.stats.eviction_count += 1
            
        logger.info(f"{self.name}: Evicted {len(evicted_ids)} allocations to free {evicted_bytes} bytes")
        return True
        
    def get_usage_percent(self) -> float:
        """Get pool usage percentage"""
        with self.lock:
            return (self.allocated_bytes / self.max_size_bytes) * 100 if self.max_size_bytes > 0 else 0


class BoltMemoryManager:
    """Central memory manager with dynamic budgeting and overcommit protection"""
    
    def __init__(self):
        self.total_memory_bytes = TOTAL_SYSTEM_MEMORY_GB * 1024 * 1024 * 1024
        self.max_allocation_bytes = MAX_ALLOCATION_GB * 1024 * 1024 * 1024
        
        # Initialize component pools
        self.pools: Dict[str, MemoryPool] = {}
        for component, budget_percent in COMPONENT_BUDGETS.items():
            pool_size = int(self.max_allocation_bytes * budget_percent)
            self.pools[component] = MemoryPool(component, pool_size)
            
        # Tracking
        self.allocation_registry: Dict[str, str] = {}  # alloc_id -> component
        self.pressure_callbacks: List[Callable] = []
        self.emergency_callbacks: List[Callable] = []
        
        # Monitoring
        self.monitor_thread = None
        self.running = False
        self.lock = threading.RLock()
        
        # Start monitoring
        self.start_monitoring()
        
    def allocate(self, component: str, size_mb: float, description: str,
                 can_evict: bool = True, priority: int = 5) -> Optional[str]:
        """
        Allocate memory for a component
        
        Args:
            component: Component name (must be in COMPONENT_BUDGETS)
            size_mb: Size in megabytes
            description: What the memory is for
            can_evict: Whether this can be evicted under pressure
            priority: 1-10, higher = more important
            
        Returns:
            Allocation ID or None if failed
        """
        if component not in self.pools:
            logger.error(f"Unknown component: {component}")
            return None
            
        size_bytes = int(size_mb * 1024 * 1024)
        
        # Check system memory pressure first
        system_usage = self.get_system_memory_usage()
        if system_usage > CRITICAL_THRESHOLD:
            self._trigger_emergency_measures()
            if system_usage > CRITICAL_THRESHOLD:  # Still critical after measures
                logger.error(f"System memory critical ({system_usage:.1%}), allocation denied")
                return None
                
        # Attempt allocation
        with self.lock:
            alloc_id = self.pools[component].allocate(
                size_bytes, description, can_evict, priority
            )
            
            if alloc_id:
                self.allocation_registry[alloc_id] = component
                logger.info(f"Allocated {size_mb:.1f}MB for {component}: {description}")
            else:
                logger.warning(f"Failed to allocate {size_mb:.1f}MB for {component}")
                
            return alloc_id
            
    def deallocate(self, alloc_id: str) -> bool:
        """Deallocate memory"""
        with self.lock:
            if alloc_id not in self.allocation_registry:
                return False
                
            component = self.allocation_registry.pop(alloc_id)
            return self.pools[component].deallocate(alloc_id)
            
    @contextmanager
    def allocate_context(self, component: str, size_mb: float, description: str, **kwargs):
        """Context manager for automatic allocation/deallocation"""
        alloc_id = self.allocate(component, size_mb, description, **kwargs)
        if not alloc_id:
            raise MemoryError(f"Could not allocate {size_mb}MB for {component}")
            
        try:
            yield alloc_id
        finally:
            self.deallocate(alloc_id)
            
    def get_component_stats(self, component: str) -> Dict[str, any]:
        """Get detailed stats for a component"""
        if component not in self.pools:
            return {}
            
        pool = self.pools[component]
        with pool.lock:
            return {
                "allocated_mb": pool.allocated_bytes / 1024 / 1024,
                "max_mb": pool.max_size_bytes / 1024 / 1024,
                "usage_percent": pool.get_usage_percent(),
                "allocation_count": pool.stats.allocation_count,
                "eviction_count": pool.stats.eviction_count,
                "pressure_events": pool.stats.pressure_events,
                "peak_mb": pool.stats.peak_bytes / 1024 / 1024
            }
            
    def get_system_memory_usage(self) -> float:
        """Get system-wide memory usage percentage"""
        try:
            mem = psutil.virtual_memory()
            return mem.percent / 100
        except:
            return 0.0
            
    def get_total_allocated_mb(self) -> float:
        """Get total allocated memory across all components"""
        total = 0
        for pool in self.pools.values():
            with pool.lock:
                total += pool.allocated_bytes
        return total / 1024 / 1024
        
    def register_pressure_callback(self, callback: Callable):
        """Register callback for memory pressure events"""
        self.pressure_callbacks.append(callback)
        
    def register_emergency_callback(self, callback: Callable):
        """Register callback for emergency memory situations"""
        self.emergency_callbacks.append(callback)
        
    def _monitor_memory(self):
        """Background thread to monitor memory usage"""
        logger.info("Memory monitor started")
        
        while self.running:
            try:
                system_usage = self.get_system_memory_usage()
                total_allocated = self.get_total_allocated_mb()
                
                # Log status
                if system_usage > MEMORY_PRESSURE_THRESHOLD:
                    logger.warning(f"Memory pressure: System {system_usage:.1%}, Allocated {total_allocated:.1f}MB")
                    
                # Check for pressure conditions
                if system_usage > MEMORY_PRESSURE_THRESHOLD:
                    self._handle_memory_pressure(system_usage)
                    
                # Component-specific checks
                for component, pool in self.pools.items():
                    usage = pool.get_usage_percent()
                    if usage > 90:
                        logger.warning(f"{component} pool usage high: {usage:.1f}%")
                        pool.stats.pressure_events += 1
                        
                # Periodic GC for components using >80% of budget
                current_time = time.time()
                for pool in self.pools.values():
                    if (pool.get_usage_percent() > 80 and 
                        current_time - pool.stats.last_gc > 60):  # GC every minute max
                        gc.collect()
                        pool.stats.last_gc = current_time
                        
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(10)
                
    def _handle_memory_pressure(self, system_usage: float):
        """Handle memory pressure situation"""
        logger.warning(f"Handling memory pressure: {system_usage:.1%}")
        
        # Notify callbacks
        for callback in self.pressure_callbacks:
            try:
                callback(system_usage)
            except Exception as e:
                logger.error(f"Pressure callback error: {e}")
                
        # Progressive eviction based on pressure level
        if system_usage > CRITICAL_THRESHOLD:
            self._trigger_emergency_measures()
        elif system_usage > MEMORY_PRESSURE_THRESHOLD:
            self._evict_low_priority_allocations()
            
    def _evict_low_priority_allocations(self):
        """Evict low priority allocations across all pools"""
        total_evicted = 0
        
        for pool in self.pools.values():
            with pool.lock:
                # Find evictable allocations with priority <= 3
                evictable = [
                    (aid, alloc) for aid, alloc in pool.allocations.items()
                    if alloc.can_evict and alloc.priority <= 3
                ]
                
                # Sort by priority (ascending) and age
                evictable.sort(key=lambda x: (x[1].priority, x[1].allocated_at))
                
                # Evict up to 20% of pool
                target_bytes = int(pool.allocated_bytes * 0.2)
                evicted_bytes = 0
                
                for alloc_id, alloc in evictable:
                    if evicted_bytes >= target_bytes:
                        break
                    pool.deallocate(alloc_id)
                    evicted_bytes += alloc.size_bytes
                    total_evicted += alloc.size_bytes
                    
        logger.info(f"Evicted {total_evicted / 1024 / 1024:.1f}MB of low priority allocations")
        
    def _trigger_emergency_measures(self):
        """Emergency measures when memory is critical"""
        logger.critical("Triggering emergency memory measures")
        
        # Force garbage collection
        gc.collect(2)  # Full collection
        
        # Notify emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Emergency callback error: {e}")
                
        # Aggressive eviction - evict everything with priority < 7
        total_evicted = 0
        for pool in self.pools.values():
            with pool.lock:
                evictable = [
                    (aid, alloc) for aid, alloc in pool.allocations.items()
                    if alloc.can_evict and alloc.priority < 7
                ]
                
                for alloc_id, alloc in evictable:
                    pool.deallocate(alloc_id)
                    total_evicted += alloc.size_bytes
                    
        logger.critical(f"Emergency: Evicted {total_evicted / 1024 / 1024:.1f}MB")
        
    def start_monitoring(self):
        """Start background monitoring"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_memory,
                daemon=True,
                name="BoltMemoryMonitor"
            )
            self.monitor_thread.start()
            
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
    def get_status_report(self) -> Dict[str, any]:
        """Get comprehensive status report"""
        report = {
            "timestamp": time.time(),
            "system": {
                "total_memory_gb": TOTAL_SYSTEM_MEMORY_GB,
                "max_allocation_gb": MAX_ALLOCATION_GB,
                "system_usage_percent": self.get_system_memory_usage() * 100,
                "total_allocated_mb": self.get_total_allocated_mb()
            },
            "components": {}
        }
        
        for component in self.pools:
            report["components"][component] = self.get_component_stats(component)
            
        return report
        
    def enforce_limits(self, strict: bool = True):
        """Enforce memory limits across all components"""
        if not strict:
            return
            
        for component, pool in self.pools.items():
            if pool.get_usage_percent() > 100:
                # Force eviction to get back under limit
                excess = pool.allocated_bytes - pool.max_size_bytes
                logger.warning(f"{component} over limit by {excess / 1024 / 1024:.1f}MB, forcing eviction")
                
                # Evict lowest priority allocations until under limit
                evicted = 0
                allocations = sorted(
                    pool.allocations.items(),
                    key=lambda x: (x[1].priority, x[1].allocated_at)
                )
                
                for alloc_id, alloc in allocations:
                    if not alloc.can_evict:
                        continue
                    pool.deallocate(alloc_id)
                    evicted += alloc.size_bytes
                    if evicted >= excess:
                        break
                        
                logger.info(f"{component}: Forced eviction of {evicted / 1024 / 1024:.1f}MB")


# Global instance
_memory_manager: Optional[BoltMemoryManager] = None

def get_memory_manager() -> BoltMemoryManager:
    """Get or create the global memory manager"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = BoltMemoryManager()
    return _memory_manager


# Component-specific helpers

class DuckDBMemoryGuard:
    """Memory guard specifically for DuckDB operations"""
    
    def __init__(self, connection):
        self.connection = connection
        self.memory_manager = get_memory_manager()
        self._configure_limits()
        
    def _configure_limits(self):
        """Configure DuckDB memory limits"""
        # Get DuckDB allocation in MB
        stats = self.memory_manager.get_component_stats("duckdb")
        max_memory_mb = int(stats["max_mb"] * 0.9)  # Use 90% of allocation
        
        # Set DuckDB configuration
        self.connection.execute(f"SET memory_limit='{max_memory_mb}MB'")
        self.connection.execute("SET temp_directory='.'")  # Use local temp
        
    @contextmanager
    def allocate_for_query(self, estimated_mb: float, query_description: str):
        """Allocate memory for a specific query"""
        with self.memory_manager.allocate_context(
            "duckdb",
            estimated_mb,
            f"Query: {query_description}",
            priority=6
        ):
            yield
            

class JarvisMemoryGuard:
    """Memory guard for Jarvis operations"""
    
    def __init__(self):
        self.memory_manager = get_memory_manager()
        
    def check_allocation(self, size_mb: float, operation: str) -> bool:
        """Check if allocation is possible"""
        stats = self.memory_manager.get_component_stats("jarvis")
        available_mb = stats["max_mb"] - stats["allocated_mb"]
        return size_mb <= available_mb
        
    @contextmanager
    def allocate_for_index(self, index_size_mb: float, index_name: str):
        """Allocate memory for index operations"""
        with self.memory_manager.allocate_context(
            "jarvis",
            index_size_mb,
            f"Index: {index_name}",
            priority=7,  # Indexes are important
            can_evict=False  # Don't evict active indexes
        ):
            yield
            

class EinsteinMemoryGuard:
    """Memory guard for Einstein operations"""
    
    def __init__(self):
        self.memory_manager = get_memory_manager()
        
    @contextmanager
    def allocate_for_embeddings(self, batch_size: int, embedding_dim: int):
        """Allocate memory for embedding operations"""
        # Estimate memory needed
        estimated_mb = (batch_size * embedding_dim * 4) / (1024 * 1024)  # 4 bytes per float
        
        with self.memory_manager.allocate_context(
            "einstein",
            estimated_mb,
            f"Embeddings: batch={batch_size}",
            priority=8  # High priority for ML operations
        ):
            yield


# Graceful degradation strategies

def setup_pressure_handlers():
    """Setup handlers for memory pressure situations"""
    manager = get_memory_manager()
    
    def handle_pressure(usage: float):
        """Handle memory pressure"""
        logger.info(f"Memory pressure handler triggered at {usage:.1%}")
        
        # Clear caches
        if hasattr(cache, 'clear'):
            cache.clear()
            
        # Reduce batch sizes
        os.environ['BATCH_SIZE_REDUCTION'] = '0.5'
        
        # Trigger garbage collection
        gc.collect()
        
    def handle_emergency():
        """Handle emergency memory situation"""
        logger.critical("Emergency memory handler triggered")
        
        # Stop non-critical operations
        os.environ['EMERGENCY_MODE'] = '1'
        
        # Clear all caches
        gc.collect(2)
        
        # Notify components to reduce memory usage
        # Components should check os.environ['EMERGENCY_MODE']
        
    manager.register_pressure_callback(handle_pressure)
    manager.register_emergency_callback(handle_emergency)


# Example usage and testing
if __name__ == "__main__":
    # Initialize manager
    manager = get_memory_manager()
    setup_pressure_handlers()
    
    # Example: DuckDB allocation
    with manager.allocate_context("duckdb", 1000, "Test query", priority=8):
        print("DuckDB query running with 1GB allocation")
        time.sleep(2)
        
    # Example: Jarvis allocation
    alloc_id = manager.allocate("jarvis", 500, "Search index", can_evict=False, priority=9)
    if alloc_id:
        print("Jarvis index allocated")
        
    # Get status
    report = manager.get_status_report()
    print("\nMemory Status Report:")
    print(f"System usage: {report['system']['system_usage_percent']:.1f}%")
    print(f"Total allocated: {report['system']['total_allocated_mb']:.1f}MB")
    
    for component, stats in report['components'].items():
        print(f"\n{component}:")
        print(f"  Allocated: {stats['allocated_mb']:.1f}MB / {stats['max_mb']:.1f}MB")
        print(f"  Usage: {stats['usage_percent']:.1f}%")
        print(f"  Peak: {stats['peak_mb']:.1f}MB")