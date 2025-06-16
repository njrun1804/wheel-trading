"""
Unified Memory Manager - Central coordination for all memory management

Consolidates 26 different memory management implementations into a single,
coherent system optimized for M4 Pro with 24GB unified memory.

Key Features:
- Central coordination across all components
- Predictive pressure handling
- Zero-copy buffer management
- Component-specific allocators
- M4 Pro unified memory optimization
"""

import gc
import logging
import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import psutil

logger = logging.getLogger(__name__)

# M4 Pro System Constants
TOTAL_MEMORY_GB = 24
MAX_SYSTEM_ALLOCATION_GB = 18  # 75% of total
SAFETY_BUFFER_GB = 2  # Emergency reserve

# Component Budget Allocations (percentage of MAX_SYSTEM_ALLOCATION_GB)
COMPONENT_BUDGETS = {
    "duckdb": 0.40,  # 40% = 7.2GB (reduced from 9GB)
    "jarvis": 0.15,  # 15% = 2.7GB (reduced from 3GB)
    "einstein": 0.05,  # 5% = 0.9GB (reduced from 1.44GB)
    "meta": 0.08,  # 8% = 1.44GB (reduced from 1.8GB)
    "gpu_mlx": 0.20,  # 20% = 3.6GB for MLX operations
    "cache": 0.08,  # 8% = 1.44GB (reduced from 1.8GB)
    "shared": 0.04,  # 4% = 0.72GB for shared buffers
}


class PressureLevel(Enum):
    """Memory pressure levels"""

    NORMAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class AllocationPriority(Enum):
    """Allocation priorities from lowest to highest"""

    CACHE = 1
    TEMPORARY = 2
    STANDARD = 3
    IMPORTANT = 4
    CRITICAL = 5
    SYSTEM = 6


@dataclass
class MemoryAllocation:
    """Represents a single memory allocation"""

    id: str
    component: str
    size_bytes: int
    priority: AllocationPriority
    description: str
    created_at: float
    can_evict: bool = True
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def touch(self):
        """Update last access time"""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class ComponentMemoryStats:
    """Memory statistics for a component"""

    budget_bytes: int
    allocated_bytes: int = 0
    peak_bytes: int = 0
    allocation_count: int = 0
    eviction_count: int = 0
    pressure_events: int = 0
    failed_allocations: int = 0

    @property
    def available_bytes(self) -> int:
        return max(0, self.budget_bytes - self.allocated_bytes)

    @property
    def usage_percent(self) -> float:
        return (
            (self.allocated_bytes / self.budget_bytes * 100)
            if self.budget_bytes > 0
            else 0
        )


class UnifiedMemoryManager:
    """
    Central memory manager for all components

    Provides unified memory allocation, tracking, and pressure management
    across all system components with M4 Pro optimizations.
    """

    def __init__(self):
        # System configuration
        self.total_memory_bytes = TOTAL_MEMORY_GB * 1024 * 1024 * 1024
        self.max_allocation_bytes = MAX_SYSTEM_ALLOCATION_GB * 1024 * 1024 * 1024
        self.safety_buffer_bytes = SAFETY_BUFFER_GB * 1024 * 1024 * 1024

        # Component tracking
        self.components: dict[str, ComponentMemoryStats] = {}
        self.allocations: dict[str, MemoryAllocation] = {}
        self.allocation_lock = threading.RLock()

        # Initialize component budgets
        self._initialize_components()

        # Pressure management
        self.current_pressure = PressureLevel.NORMAL
        self.pressure_callbacks: dict[PressureLevel, list[Callable]] = defaultdict(list)
        self.pressure_history = []

        # Monitoring
        self.monitor_thread = None
        self.monitoring_active = False
        self.monitor_interval = 2.0  # seconds

        # Memory pools
        from .memory_pools import UnifiedMemoryPool

        self.memory_pool = UnifiedMemoryPool(self)

        # Pressure monitor
        from .pressure_monitor import PressureMonitor

        self.pressure_monitor = PressureMonitor(self)

        # Start monitoring
        self.start_monitoring()

        logger.info(
            "UnifiedMemoryManager initialized with %.1fGB budget",
            MAX_SYSTEM_ALLOCATION_GB,
        )

    def _initialize_components(self):
        """Initialize component memory budgets"""
        for component, budget_percent in COMPONENT_BUDGETS.items():
            budget_bytes = int(self.max_allocation_bytes * budget_percent)
            self.components[component] = ComponentMemoryStats(budget_bytes=budget_bytes)
            logger.info(f"Component {component}: {budget_bytes / 1024**3:.2f}GB budget")

    def allocate(
        self,
        component: str,
        size_mb: float,
        description: str,
        priority: AllocationPriority = AllocationPriority.STANDARD,
        can_evict: bool = True,
    ) -> str | None:
        """
        Allocate memory for a component

        Returns allocation ID if successful, None if failed
        """
        if component not in self.components:
            logger.error(f"Unknown component: {component}")
            return None

        size_bytes = int(size_mb * 1024 * 1024)

        with self.allocation_lock:
            # Check system pressure first
            if not self._can_allocate(size_bytes):
                logger.warning(f"System under pressure, cannot allocate {size_mb}MB")
                self.components[component].failed_allocations += 1
                return None

            # Check component budget
            stats = self.components[component]
            if stats.allocated_bytes + size_bytes > stats.budget_bytes:
                # Try to evict to make space
                freed = self._evict_for_component(component, size_bytes, priority)
                if freed < size_bytes:
                    logger.warning(
                        f"{component}: Cannot allocate {size_mb}MB, budget exceeded"
                    )
                    stats.failed_allocations += 1
                    return None

            # Create allocation
            alloc_id = f"{component}_{int(time.time() * 1000000)}"
            allocation = MemoryAllocation(
                id=alloc_id,
                component=component,
                size_bytes=size_bytes,
                priority=priority,
                description=description,
                created_at=time.time(),
                can_evict=can_evict,
            )

            # Track allocation
            self.allocations[alloc_id] = allocation
            stats.allocated_bytes += size_bytes
            stats.allocation_count += 1
            stats.peak_bytes = max(stats.peak_bytes, stats.allocated_bytes)

            logger.debug(f"Allocated {size_mb:.1f}MB for {component}: {description}")
            return alloc_id

    def deallocate(self, alloc_id: str) -> bool:
        """Deallocate memory"""
        with self.allocation_lock:
            if alloc_id not in self.allocations:
                return False

            allocation = self.allocations.pop(alloc_id)
            stats = self.components[allocation.component]
            stats.allocated_bytes -= allocation.size_bytes

            logger.debug(
                f"Deallocated {allocation.size_bytes / 1024**2:.1f}MB from {allocation.component}"
            )
            return True

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

    @asynccontextmanager
    async def allocate_context_async(
        self, component: str, size_mb: float, description: str, **kwargs
    ):
        """Async context manager for automatic allocation/deallocation"""
        alloc_id = self.allocate(component, size_mb, description, **kwargs)
        if not alloc_id:
            raise MemoryError(f"Could not allocate {size_mb}MB for {component}")

        try:
            yield alloc_id
        finally:
            self.deallocate(alloc_id)

    def _can_allocate(self, size_bytes: int) -> bool:
        """Check if allocation is safe given system pressure"""
        # Get current system usage
        memory = psutil.virtual_memory()
        available_bytes = memory.available

        # Check if we have enough with safety buffer
        return available_bytes - size_bytes > self.safety_buffer_bytes

    def _evict_for_component(
        self, component: str, needed_bytes: int, min_priority: AllocationPriority
    ) -> int:
        """Evict allocations to make space for component"""
        # Get evictable allocations for this component
        evictable = [
            alloc
            for alloc in self.allocations.values()
            if (
                alloc.component == component
                and alloc.can_evict
                and alloc.priority.value < min_priority.value
            )
        ]

        # Sort by priority (ascending) and age
        evictable.sort(key=lambda a: (a.priority.value, a.last_accessed))

        evicted_bytes = 0
        evicted_ids = []

        for alloc in evictable:
            if evicted_bytes >= needed_bytes:
                break
            evicted_bytes += alloc.size_bytes
            evicted_ids.append(alloc.id)

        # Perform evictions
        for alloc_id in evicted_ids:
            self.deallocate(alloc_id)
            self.components[component].eviction_count += 1

        if evicted_ids:
            logger.info(
                f"Evicted {len(evicted_ids)} allocations ({evicted_bytes / 1024**2:.1f}MB) from {component}"
            )

        return evicted_bytes

    def handle_pressure(self, pressure_level: PressureLevel):
        """Handle memory pressure event"""
        self.current_pressure = pressure_level

        # Record pressure event
        for component in self.components.values():
            if pressure_level.value >= PressureLevel.MEDIUM.value:
                component.pressure_events += 1

        # Execute callbacks
        for level in PressureLevel:
            if level.value <= pressure_level.value:
                for callback in self.pressure_callbacks[level]:
                    try:
                        callback(pressure_level)
                    except Exception as e:
                        logger.error(f"Pressure callback error: {e}")

        # Take action based on pressure level
        if pressure_level == PressureLevel.HIGH:
            self._handle_high_pressure()
        elif pressure_level == PressureLevel.CRITICAL:
            self._handle_critical_pressure()
        elif pressure_level == PressureLevel.EMERGENCY:
            self._handle_emergency_pressure()

    def _handle_high_pressure(self):
        """Handle high memory pressure"""
        logger.warning("Handling HIGH memory pressure")

        # Evict low priority allocations across all components
        with self.allocation_lock:
            evicted = 0
            for alloc in list(self.allocations.values()):
                if (
                    alloc.can_evict
                    and alloc.priority.value <= AllocationPriority.TEMPORARY.value
                ):
                    self.deallocate(alloc.id)
                    evicted += 1

            if evicted > 0:
                logger.info(f"Evicted {evicted} low priority allocations")

        # Clear caches
        self.memory_pool.clear_caches()

        # Force garbage collection
        gc.collect()

    def _handle_critical_pressure(self):
        """Handle critical memory pressure"""
        logger.critical("Handling CRITICAL memory pressure")

        # More aggressive eviction
        with self.allocation_lock:
            evicted = 0
            for alloc in list(self.allocations.values()):
                if (
                    alloc.can_evict
                    and alloc.priority.value <= AllocationPriority.IMPORTANT.value
                ):
                    self.deallocate(alloc.id)
                    evicted += 1

            if evicted > 0:
                logger.info(f"Evicted {evicted} allocations under critical pressure")

        # Clear all pools
        self.memory_pool.clear_all()

        # Aggressive garbage collection
        gc.collect(2)

    def _handle_emergency_pressure(self):
        """Handle emergency memory pressure"""
        logger.critical("EMERGENCY memory pressure - taking drastic measures")

        # Evict everything except critical allocations
        with self.allocation_lock:
            evicted = 0
            for alloc in list(self.allocations.values()):
                if (
                    alloc.can_evict
                    and alloc.priority.value < AllocationPriority.CRITICAL.value
                ):
                    self.deallocate(alloc.id)
                    evicted += 1

            logger.critical(f"Emergency evicted {evicted} allocations")

        # Clear everything
        self.memory_pool.emergency_clear()

        # Multiple GC passes
        for _ in range(3):
            gc.collect(2)

        # Notify all components to reduce memory
        os.environ["MEMORY_EMERGENCY"] = "1"

    def register_pressure_callback(self, level: PressureLevel, callback: Callable):
        """Register callback for pressure level"""
        self.pressure_callbacks[level].append(callback)

    def get_component_stats(self, component: str) -> ComponentMemoryStats | None:
        """Get statistics for a component"""
        return self.components.get(component)

    def get_system_stats(self) -> dict[str, Any]:
        """Get comprehensive system statistics"""
        memory = psutil.virtual_memory()
        total_allocated = sum(c.allocated_bytes for c in self.components.values())

        return {
            "system": {
                "total_gb": self.total_memory_bytes / 1024**3,
                "available_gb": memory.available / 1024**3,
                "used_percent": memory.percent,
                "pressure_level": self.current_pressure.name,
            },
            "allocation": {
                "total_allocated_gb": total_allocated / 1024**3,
                "max_allowed_gb": self.max_allocation_bytes / 1024**3,
                "utilization_percent": (
                    total_allocated / self.max_allocation_bytes * 100
                ),
            },
            "components": {
                name: {
                    "budget_gb": stats.budget_bytes / 1024**3,
                    "allocated_gb": stats.allocated_bytes / 1024**3,
                    "usage_percent": stats.usage_percent,
                    "allocations": stats.allocation_count,
                    "evictions": stats.eviction_count,
                    "failures": stats.failed_allocations,
                }
                for name, stats in self.components.items()
            },
        }

    def start_monitoring(self):
        """Start background monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True, name="UnifiedMemoryMonitor"
            )
            self.monitor_thread.start()

            # Start pressure monitor
            self.pressure_monitor.start()

            logger.info("Memory monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False

        # Stop pressure monitor
        self.pressure_monitor.stop()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        logger.info("Memory monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Check for stale allocations
                self._cleanup_stale_allocations()

                # Check component health
                self._check_component_health()

                time.sleep(self.monitor_interval)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(5)

    def _cleanup_stale_allocations(self):
        """Clean up stale allocations"""
        current_time = time.time()
        stale_threshold = 3600  # 1 hour

        with self.allocation_lock:
            stale_ids = []
            for alloc_id, alloc in self.allocations.items():
                if (
                    alloc.can_evict
                    and alloc.priority.value <= AllocationPriority.TEMPORARY.value
                    and current_time - alloc.last_accessed > stale_threshold
                ):
                    stale_ids.append(alloc_id)

            for alloc_id in stale_ids:
                self.deallocate(alloc_id)

            if stale_ids:
                logger.info(f"Cleaned up {len(stale_ids)} stale allocations")

    def _check_component_health(self):
        """Check component memory health"""
        for name, stats in self.components.items():
            if stats.usage_percent > 90:
                logger.warning(
                    f"Component {name} at {stats.usage_percent:.1f}% capacity"
                )

            if stats.failed_allocations > 10:
                logger.warning(
                    f"Component {name} has {stats.failed_allocations} failed allocations"
                )

    def optimize_allocations(self):
        """Optimize memory allocations across components"""
        logger.info("Optimizing memory allocations...")

        with self.allocation_lock:
            # Analyze allocation patterns
            component_efficiency = {}

            for name, stats in self.components.items():
                # Calculate efficiency metrics
                allocations = [
                    a for a in self.allocations.values() if a.component == name
                ]
                if allocations:
                    avg_access_time = sum(a.last_accessed for a in allocations) / len(
                        allocations
                    )
                    avg_access_count = sum(a.access_count for a in allocations) / len(
                        allocations
                    )
                    efficiency = avg_access_count / max(
                        1, time.time() - avg_access_time
                    )
                else:
                    efficiency = 0

                component_efficiency[name] = efficiency

            # Rebalance budgets based on efficiency
            # This is a placeholder for more sophisticated rebalancing
            logger.info(f"Component efficiency: {component_efficiency}")

    def get_allocation_summary(self) -> str:
        """Get human-readable allocation summary"""
        stats = self.get_system_stats()

        lines = [
            "=== Memory Allocation Summary ===",
            f"System: {stats['system']['used_percent']:.1f}% used, "
            f"{stats['system']['available_gb']:.1f}GB available",
            f"Allocated: {stats['allocation']['total_allocated_gb']:.1f}GB / "
            f"{stats['allocation']['max_allowed_gb']:.1f}GB "
            f"({stats['allocation']['utilization_percent']:.1f}%)",
            f"Pressure: {stats['system']['pressure_level']}",
            "\nComponent Usage:",
        ]

        for name, component in stats["components"].items():
            lines.append(
                f"  {name}: {component['allocated_gb']:.1f}GB / "
                f"{component['budget_gb']:.1f}GB ({component['usage_percent']:.1f}%) "
                f"[{component['allocations']} allocs, {component['evictions']} evicts]"
            )

        return "\n".join(lines)


# Global instance
_memory_manager: UnifiedMemoryManager | None = None


def get_memory_manager() -> UnifiedMemoryManager:
    """Get or create the global memory manager"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = UnifiedMemoryManager()
    return _memory_manager


# Component-specific helpers


def allocate_for_duckdb(size_mb: float, description: str, **kwargs) -> str | None:
    """Allocate memory for DuckDB operations"""
    return get_memory_manager().allocate("duckdb", size_mb, description, **kwargs)


def allocate_for_jarvis(size_mb: float, description: str, **kwargs) -> str | None:
    """Allocate memory for Jarvis operations"""
    return get_memory_manager().allocate("jarvis", size_mb, description, **kwargs)


def allocate_for_einstein(size_mb: float, description: str, **kwargs) -> str | None:
    """Allocate memory for Einstein operations"""
    return get_memory_manager().allocate("einstein", size_mb, description, **kwargs)


def allocate_for_gpu(size_mb: float, description: str, **kwargs) -> str | None:
    """Allocate memory for GPU/MLX operations"""
    return get_memory_manager().allocate("gpu_mlx", size_mb, description, **kwargs)
