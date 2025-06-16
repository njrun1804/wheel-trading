"""
Unified Memory Manager - Central coordinator for all memory operations

Provides centralized memory management across all trading system components
with intelligent allocation, pressure detection, and automatic cleanup.
"""

import gc
import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import psutil

from .allocation_strategies import get_strategy_for_component
from .cleanup_system import CleanupLevel, CleanupSystem
from .memory_pools import MemoryPool, StandardMemoryPool
from .pressure_monitor import PressureMonitor

logger = logging.getLogger(__name__)

# System configuration for M4 Pro (24GB system)
SYSTEM_CONFIG = {
    "total_memory_gb": 24,
    "max_usable_gb": 20,  # Leave 4GB for system
    "pressure_threshold": 0.80,
    "critical_threshold": 0.90,
    "emergency_threshold": 0.95,
}

# Component memory budgets (percentage of max_usable_gb)
COMPONENT_BUDGETS = {
    "trading_data": 0.35,  # 7GB - Price data, options chains, market data
    "ml_models": 0.25,  # 5GB - Neural networks, embeddings, training
    "database": 0.25,  # 5GB - DuckDB, SQLite operations
    "cache": 0.10,  # 2GB - General caching
    "system_buffer": 0.05,  # 1GB - Emergency buffer
}


@dataclass
class MemoryAllocation:
    """Tracks a single memory allocation with metadata"""

    component: str
    size_bytes: int
    description: str
    allocated_at: float
    priority: int = 5  # 1-10, higher = more important
    can_evict: bool = True
    tags: list[str] = field(default_factory=list)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def update_access(self):
        """Update access tracking"""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class ComponentStats:
    """Statistics for a component's memory usage"""

    allocated_bytes: int = 0
    peak_bytes: int = 0
    allocation_count: int = 0
    eviction_count: int = 0
    pressure_events: int = 0
    last_cleanup: float = 0
    budget_violations: int = 0


class UnifiedMemoryManager:
    """
    Central memory manager providing unified allocation across all trading components
    """

    def __init__(self):
        self.total_memory = SYSTEM_CONFIG["total_memory_gb"] * 1024**3
        self.max_usable = SYSTEM_CONFIG["max_usable_gb"] * 1024**3

        # Initialize component pools
        self.pools: dict[str, MemoryPool] = {}
        self._init_component_pools()

        # Allocation tracking
        self.allocations: dict[str, MemoryAllocation] = {}
        self.component_stats: dict[str, ComponentStats] = {}
        self._init_component_stats()

        # Thread safety
        self.lock = threading.RLock()

        # Subsystems
        self.pressure_monitor = PressureMonitor(self)
        self.cleanup_system = CleanupSystem(self)

        # Callbacks
        self.pressure_callbacks: list[Callable] = []
        self.emergency_callbacks: list[Callable] = []

        # State tracking
        self.emergency_mode = False
        self.last_pressure_check = 0

        logger.info(
            f"UnifiedMemoryManager initialized with {SYSTEM_CONFIG['max_usable_gb']}GB usable memory"
        )

    def _init_component_pools(self):
        """Initialize memory pools for each component"""
        for component, budget_percent in COMPONENT_BUDGETS.items():
            if component == "system_buffer":
                continue  # Reserve buffer, don't create pool

            pool_size = int(self.max_usable * budget_percent)
            strategy = get_strategy_for_component(component)

            self.pools[component] = StandardMemoryPool(
                name=component, max_size_bytes=pool_size, strategy=strategy
            )

    def _init_component_stats(self):
        """Initialize statistics for each component"""
        for component in COMPONENT_BUDGETS:
            if component == "system_buffer":
                continue
            self.component_stats[component] = ComponentStats()

    def allocate(
        self,
        component: str,
        size_mb: float,
        description: str,
        priority: int = 5,
        can_evict: bool = True,
        tags: list[str] | None = None,
    ) -> str | None:
        """
        Allocate memory for a component

        Args:
            component: Component name ('trading_data', 'ml_models', 'database', 'cache')
            size_mb: Size in megabytes
            description: Description of what memory is for
            priority: Priority 1-10 (higher = more important)
            can_evict: Whether allocation can be evicted under pressure
            tags: Optional tags for categorization

        Returns:
            Allocation ID or None if failed
        """
        if component not in self.pools:
            logger.error(f"Unknown component: {component}")
            return None

        size_bytes = int(size_mb * 1024 * 1024)
        tags = tags or []

        with self.lock:
            # Check system pressure first
            if self._should_deny_allocation(size_bytes, priority):
                logger.warning(
                    f"Allocation denied due to memory pressure: {size_mb}MB for {component}"
                )
                return None

            # Try allocation in component pool
            pool = self.pools[component]
            alloc_id = pool.allocate(size_bytes, description, priority, can_evict)

            if not alloc_id:
                # Try eviction within component
                if self._try_eviction_for_allocation(component, size_bytes, priority):
                    alloc_id = pool.allocate(
                        size_bytes, description, priority, can_evict
                    )

            if not alloc_id:
                # Try global eviction from other components
                if self._try_global_eviction(
                    size_bytes, priority, exclude_component=component
                ):
                    alloc_id = pool.allocate(
                        size_bytes, description, priority, can_evict
                    )

            if alloc_id:
                # Track allocation
                self.allocations[alloc_id] = MemoryAllocation(
                    component=component,
                    size_bytes=size_bytes,
                    description=description,
                    allocated_at=time.time(),
                    priority=priority,
                    can_evict=can_evict,
                    tags=tags,
                )

                # Update stats
                stats = self.component_stats[component]
                stats.allocated_bytes += size_bytes
                stats.allocation_count += 1
                if stats.allocated_bytes > stats.peak_bytes:
                    stats.peak_bytes = stats.allocated_bytes

                logger.debug(
                    f"Allocated {size_mb:.1f}MB for {component}: {description}"
                )
                return alloc_id
            else:
                logger.error(f"Failed to allocate {size_mb:.1f}MB for {component}")
                return None

    def deallocate(self, alloc_id: str) -> bool:
        """Deallocate memory by allocation ID"""
        with self.lock:
            if alloc_id not in self.allocations:
                return False

            allocation = self.allocations.pop(alloc_id)
            pool = self.pools[allocation.component]

            if pool.deallocate(alloc_id):
                # Update stats
                stats = self.component_stats[allocation.component]
                stats.allocated_bytes -= allocation.size_bytes
                return True

            return False

    def access_allocation(self, alloc_id: str):
        """Record access to allocation for LRU tracking"""
        if alloc_id in self.allocations:
            self.allocations[alloc_id].update_access()

    @contextmanager
    def allocate_context(
        self, component: str, size_mb: float, description: str, **kwargs
    ):
        """Context manager for automatic allocation/deallocation"""
        alloc_id = self.allocate(component, size_mb, description, **kwargs)
        if not alloc_id:
            raise MemoryError(f"Could not allocate {size_mb}MB for {component}")

        try:
            yield alloc_id
        finally:
            self.deallocate(alloc_id)

    def get_component_usage(self, component: str) -> dict[str, Any]:
        """Get detailed usage statistics for component"""
        if component not in self.pools:
            return {}

        pool = self.pools[component]
        stats = self.component_stats[component]

        return {
            "allocated_mb": stats.allocated_bytes / (1024 * 1024),
            "budget_mb": pool.max_size_bytes / (1024 * 1024),
            "usage_percent": (stats.allocated_bytes / pool.max_size_bytes * 100)
            if pool.max_size_bytes > 0
            else 0,
            "peak_mb": stats.peak_bytes / (1024 * 1024),
            "allocation_count": stats.allocation_count,
            "eviction_count": stats.eviction_count,
            "pressure_events": stats.pressure_events,
            "budget_violations": stats.budget_violations,
        }

    def get_system_usage(self) -> dict[str, Any]:
        """Get system-wide memory usage"""
        total_allocated = sum(
            stats.allocated_bytes for stats in self.component_stats.values()
        )
        system_memory = psutil.virtual_memory()

        return {
            "total_system_gb": self.total_memory / (1024**3),
            "usable_gb": self.max_usable / (1024**3),
            "allocated_mb": total_allocated / (1024 * 1024),
            "system_usage_percent": system_memory.percent,
            "available_gb": system_memory.available / (1024**3),
            "pressure_level": self.pressure_monitor.get_pressure_level(),
            "emergency_mode": self.emergency_mode,
        }

    def _should_deny_allocation(self, size_bytes: int, priority: int) -> bool:
        """Check if allocation should be denied due to pressure"""
        pressure = self.pressure_monitor.get_pressure_level()

        # Always allow high priority allocations
        if priority >= 8:
            return False

        # Deny low priority allocations under pressure
        if pressure > SYSTEM_CONFIG["pressure_threshold"] and priority < 5:
            return True

        # Deny all non-critical allocations in emergency
        return bool(pressure > SYSTEM_CONFIG["emergency_threshold"] and priority < 9)

    def _try_eviction_for_allocation(
        self, component: str, needed_bytes: int, min_priority: int
    ) -> bool:
        """Try to evict from component pool to make space"""
        pool = self.pools[component]
        return pool.evict_for_space(needed_bytes, min_priority)

    def _try_global_eviction(
        self, needed_bytes: int, min_priority: int, exclude_component: str
    ) -> bool:
        """Try to evict from other components to make space"""
        evicted_bytes = 0

        # Sort components by current usage (evict from highest usage first)
        components = [
            (name, self.component_stats[name].allocated_bytes)
            for name in self.pools
            if name != exclude_component
        ]
        components.sort(key=lambda x: x[1], reverse=True)

        for component, _ in components:
            pool = self.pools[component]

            # Try to evict low priority allocations
            target_evict = min(needed_bytes - evicted_bytes, pool.allocated_bytes // 4)
            evicted = pool.evict_for_space(target_evict, min_priority)

            if evicted:
                evicted_bytes += evicted
                if evicted_bytes >= needed_bytes:
                    return True

        return evicted_bytes >= needed_bytes

    def trigger_cleanup(self, aggressive: bool = False):
        """Trigger memory cleanup"""
        level = CleanupLevel.AGGRESSIVE if aggressive else CleanupLevel.MODERATE
        self.cleanup_system.run_cleanup(level=level)

    def handle_pressure(self, pressure_level: float):
        """Handle memory pressure situation"""
        logger.warning(f"Memory pressure detected: {pressure_level:.2%}")

        # Update component stats
        for stats in self.component_stats.values():
            stats.pressure_events += 1

        # Trigger callbacks
        for callback in self.pressure_callbacks:
            try:
                callback(pressure_level)
            except Exception as e:
                logger.error(f"Pressure callback error: {e}")

        # Progressive response based on pressure level
        if pressure_level > SYSTEM_CONFIG["emergency_threshold"]:
            self._handle_emergency()
        elif pressure_level > SYSTEM_CONFIG["critical_threshold"]:
            self._handle_critical_pressure()
        elif pressure_level > SYSTEM_CONFIG["pressure_threshold"]:
            self._handle_normal_pressure()

    def _handle_normal_pressure(self):
        """Handle normal memory pressure"""
        # Evict low priority allocations
        self._evict_by_priority(max_priority=3)

        # Trigger garbage collection
        gc.collect()

    def _handle_critical_pressure(self):
        """Handle critical memory pressure"""
        # More aggressive eviction
        self._evict_by_priority(max_priority=5)

        # Trigger component-specific cleanup
        self.trigger_cleanup(aggressive=False)

        # Force garbage collection
        gc.collect(2)

    def _handle_emergency(self):
        """Handle emergency memory situation"""
        logger.critical("Emergency memory situation - triggering all measures")

        self.emergency_mode = True

        # Notify emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Emergency callback error: {e}")

        # Aggressive eviction
        self._evict_by_priority(max_priority=7)

        # Aggressive cleanup
        self.trigger_cleanup(aggressive=True)

        # Full garbage collection
        gc.collect(2)

        # Set environment flag for components
        os.environ["TRADING_MEMORY_EMERGENCY"] = "1"

    def _evict_by_priority(self, max_priority: int):
        """Evict allocations up to specified priority"""
        evicted_total = 0

        for component, pool in self.pools.items():
            evicted = pool.evict_by_priority(max_priority)
            evicted_total += evicted

            if evicted > 0:
                self.component_stats[component].eviction_count += 1

        logger.info(
            f"Evicted {evicted_total / (1024*1024):.1f}MB across all components"
        )

    def register_pressure_callback(self, callback: Callable[[float], None]):
        """Register callback for pressure events"""
        self.pressure_callbacks.append(callback)

    def register_emergency_callback(self, callback: Callable[[], None]):
        """Register callback for emergency situations"""
        self.emergency_callbacks.append(callback)

    def get_status_report(self) -> dict[str, Any]:
        """Get comprehensive status report"""
        return {
            "timestamp": time.time(),
            "system": self.get_system_usage(),
            "components": {comp: self.get_component_usage(comp) for comp in self.pools},
            "pressure_monitor": self.pressure_monitor.get_stats(),
            "cleanup_system": self.cleanup_system.get_stats(),
        }

    def start_monitoring(self):
        """Start background monitoring"""
        self.pressure_monitor.start()
        self.cleanup_system.start()

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.pressure_monitor.stop()
        self.cleanup_system.stop()

    def shutdown(self):
        """Clean shutdown of memory manager"""
        logger.info("Shutting down UnifiedMemoryManager")

        self.stop_monitoring()

        # Clean up all allocations
        with self.lock:
            for alloc_id in list(self.allocations.keys()):
                self.deallocate(alloc_id)

        # Clean up pools
        for pool in self.pools.values():
            pool.cleanup()

        # Clear environment flags
        os.environ.pop("TRADING_MEMORY_EMERGENCY", None)


# Global instance
_unified_manager: UnifiedMemoryManager | None = None


def get_memory_manager() -> UnifiedMemoryManager:
    """Get or create the global unified memory manager"""
    global _unified_manager
    if _unified_manager is None:
        _unified_manager = UnifiedMemoryManager()
        _unified_manager.start_monitoring()
    return _unified_manager
