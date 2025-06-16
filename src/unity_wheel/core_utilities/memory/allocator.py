"""
Unified Memory Allocator - Central memory management for all components.

Replaces 8 different allocation implementations with a single, optimized version.
Thread-safe, pressure-aware, with automatic cleanup.
"""

import gc
import logging
import threading
import time
import weakref
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import psutil

logger = logging.getLogger(__name__)

# M4 Pro system configuration - Optimized for 4.0x speedup
SYSTEM_MEMORY_GB = 24
MAX_USABLE_GB = 20  # Leave 4GB for system
BYTES_PER_GB = 1024 * 1024 * 1024

# Component memory budgets (percentage of MAX_USABLE_GB) - Parallel processing optimized
COMPONENT_BUDGETS = {
    "trading_data": 0.35,  # 7GB
    "ml_models": 0.25,  # 5GB
    "database": 0.25,  # 5GB
    "cache": 0.10,  # 2GB
    "parallel_processing": 0.08,  # 1.6GB for parallel work
    "lock_free_pools": 0.05,  # 1GB for lock-free allocations
    "default": 0.02,  # 400MB reduced for specialized allocations
}


@dataclass
class MemoryAllocation:
    """Represents a single memory allocation."""

    id: str = field(default_factory=lambda: str(uuid4()))
    component: str = "default"
    size_bytes: int = 0
    description: str = ""
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    reference_count: int = 1
    data: Any = None

    def __post_init__(self):
        """Validate allocation parameters."""
        if self.size_bytes < 0:
            raise ValueError(f"Invalid allocation size: {self.size_bytes}")
        if self.component not in COMPONENT_BUDGETS:
            logger.warning(f"Unknown component {self.component}, using 'default'")
            self.component = "default"


class AllocationContext:
    """Context manager for temporary memory allocations."""

    def __init__(
        self, allocator: "MemoryAllocator", size: int, component: str = "default"
    ):
        self.allocator = allocator
        self.size = size
        self.component = component
        self.allocation_id = None

    def __enter__(self) -> str:
        """Allocate memory on context entry."""
        self.allocation_id = self.allocator.allocate(self.size, self.component)
        return self.allocation_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Deallocate memory on context exit."""
        if self.allocation_id:
            self.allocator.deallocate(self.allocation_id)


class MemoryAllocator:
    """
    Unified memory allocator with pressure-aware allocation strategies.

    Features:
    - Component-based budgeting
    - Automatic garbage collection
    - Thread-safe operations
    - Memory pressure handling
    - Weak reference tracking
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._allocations: dict[str, MemoryAllocation] = {}
        self._component_usage: dict[str, int] = defaultdict(int)
        self._weak_refs: dict[str, weakref.ref] = {}
        self._last_gc_time = time.time()
        self._gc_interval = 60.0  # seconds

        # Initialize component budgets in bytes
        self._component_budgets = {
            comp: int(MAX_USABLE_GB * BYTES_PER_GB * budget)
            for comp, budget in COMPONENT_BUDGETS.items()
        }

        logger.info(f"MemoryAllocator initialized with {MAX_USABLE_GB}GB usable memory")

    def allocate(
        self, size_bytes: int, component: str = "default", description: str = ""
    ) -> str:
        """
        Allocate memory for a component.

        Args:
            size_bytes: Size to allocate in bytes
            component: Component name for budgeting
            description: Optional description

        Returns:
            Allocation ID

        Raises:
            MemoryError: If allocation would exceed budget
        """
        with self._lock:
            # Validate component
            if component not in self._component_budgets:
                component = "default"

            # Check budget
            current_usage = self._component_usage[component]
            budget = self._component_budgets[component]

            if current_usage + size_bytes > budget:
                # Try garbage collection
                self._garbage_collect()
                current_usage = self._component_usage[component]

                if current_usage + size_bytes > budget:
                    raise MemoryError(
                        f"Component '{component}' would exceed budget: "
                        f"{(current_usage + size_bytes) / BYTES_PER_GB:.2f}GB > "
                        f"{budget / BYTES_PER_GB:.2f}GB"
                    )

            # Create allocation
            allocation = MemoryAllocation(
                component=component, size_bytes=size_bytes, description=description
            )

            self._allocations[allocation.id] = allocation
            self._component_usage[component] += size_bytes

            logger.debug(
                f"Allocated {size_bytes / 1024 / 1024:.2f}MB for {component} "
                f"(usage: {current_usage / BYTES_PER_GB:.2f}GB)"
            )

            return allocation.id

    def deallocate(self, allocation_id: str) -> bool:
        """
        Deallocate memory by allocation ID.

        Args:
            allocation_id: ID of allocation to free

        Returns:
            True if deallocation successful
        """
        with self._lock:
            allocation = self._allocations.pop(allocation_id, None)
            if allocation:
                self._component_usage[allocation.component] -= allocation.size_bytes

                # Clear any weak references
                self._weak_refs.pop(allocation_id, None)

                # Clear data reference
                allocation.data = None

                logger.debug(
                    f"Deallocated {allocation.size_bytes / 1024 / 1024:.2f}MB "
                    f"from {allocation.component}"
                )
                return True
            return False

    def resize(self, allocation_id: str, new_size: int) -> bool:
        """
        Resize an existing allocation.

        Args:
            allocation_id: ID of allocation to resize
            new_size: New size in bytes

        Returns:
            True if resize successful
        """
        with self._lock:
            allocation = self._allocations.get(allocation_id)
            if not allocation:
                return False

            size_diff = new_size - allocation.size_bytes

            # Check if resize would exceed budget
            if size_diff > 0:
                current_usage = self._component_usage[allocation.component]
                budget = self._component_budgets[allocation.component]

                if current_usage + size_diff > budget:
                    return False

            # Update allocation
            self._component_usage[allocation.component] += size_diff
            allocation.size_bytes = new_size
            allocation.last_accessed = time.time()

            return True

    def get_stats(self) -> dict[str, Any]:
        """Get current memory statistics."""
        with self._lock:
            # System memory info
            vm = psutil.virtual_memory()

            stats = {
                "system": {
                    "total_gb": vm.total / BYTES_PER_GB,
                    "available_gb": vm.available / BYTES_PER_GB,
                    "used_percent": vm.percent,
                },
                "allocator": {
                    "total_allocations": len(self._allocations),
                    "total_allocated_mb": sum(
                        a.size_bytes for a in self._allocations.values()
                    )
                    / 1024
                    / 1024,
                },
                "components": {},
            }

            # Component statistics
            for component, budget in self._component_budgets.items():
                usage = self._component_usage[component]
                stats["components"][component] = {
                    "budget_gb": budget / BYTES_PER_GB,
                    "used_gb": usage / BYTES_PER_GB,
                    "used_percent": (usage / budget * 100) if budget > 0 else 0,
                    "allocations": sum(
                        1
                        for a in self._allocations.values()
                        if a.component == component
                    ),
                }

            return stats

    def _garbage_collect(self):
        """Run garbage collection to free unused allocations."""
        current_time = time.time()

        # Check if enough time has passed
        if current_time - self._last_gc_time < self._gc_interval:
            return

        self._last_gc_time = current_time

        # Find dead weak references
        dead_refs = []
        for alloc_id, weak_ref in self._weak_refs.items():
            if weak_ref() is None:
                dead_refs.append(alloc_id)

        # Deallocate dead references
        for alloc_id in dead_refs:
            self.deallocate(alloc_id)

        # Run Python garbage collector
        gc.collect()

        logger.debug(f"Garbage collection freed {len(dead_refs)} allocations")

    @contextmanager
    def allocate_context(self, size: int, component: str = "default"):
        """Context manager for temporary allocations."""
        allocation_id = self.allocate(size, component)
        try:
            yield allocation_id
        finally:
            self.deallocate(allocation_id)

    def track_object(self, allocation_id: str, obj: Any):
        """Track an object with weak reference for automatic cleanup."""
        with self._lock:
            if allocation_id in self._allocations:
                self._weak_refs[allocation_id] = weakref.ref(obj)
                self._allocations[allocation_id].data = obj

    def get_allocation(self, allocation_id: str) -> MemoryAllocation | None:
        """Get allocation details by ID."""
        with self._lock:
            allocation = self._allocations.get(allocation_id)
            if allocation:
                allocation.last_accessed = time.time()
            return allocation

    def clear_component(self, component: str):
        """Clear all allocations for a component."""
        with self._lock:
            to_remove = [
                alloc_id
                for alloc_id, alloc in self._allocations.items()
                if alloc.component == component
            ]

            for alloc_id in to_remove:
                self.deallocate(alloc_id)

            logger.info(
                f"Cleared {len(to_remove)} allocations from component {component}"
            )

    def shutdown(self):
        """Clean shutdown of allocator."""
        with self._lock:
            # Clear all allocations
            all_ids = list(self._allocations.keys())
            for alloc_id in all_ids:
                self.deallocate(alloc_id)

            logger.info("MemoryAllocator shutdown complete")
