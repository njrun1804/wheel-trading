"""
Memory Context Managers - Convenient context managers for memory allocation

Provides high-level context managers that automatically handle allocation and cleanup
for different trading system components with proper error handling and resource management.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any

import numpy as np

from .unified_manager import get_memory_manager

logger = logging.getLogger(__name__)


@contextmanager
def allocate_for_trading(
    size_mb: float,
    description: str,
    priority: int = 6,
    can_evict: bool = True,
    tags: list[str] | None = None,
):
    """
    Context manager for trading data allocations

    Args:
        size_mb: Size in megabytes
        description: Description of what memory is for
        priority: Priority 1-10 (higher = more important)
        can_evict: Whether allocation can be evicted under pressure
        tags: Optional tags for categorization

    Yields:
        allocation_id: String ID for the allocation

    Example:
        with allocate_for_trading(100, "SPY options chain", priority=8, tags=['options', 'real_time']) as alloc_id:
            # Use memory for options processing
            pass
    """
    manager = get_memory_manager()
    alloc_id = None

    try:
        # Set default tags for trading data
        if tags is None:
            tags = ["trading_data"]
        elif "trading_data" not in tags:
            tags.append("trading_data")

        alloc_id = manager.allocate(
            component="trading_data",
            size_mb=size_mb,
            description=description,
            priority=priority,
            can_evict=can_evict,
            tags=tags,
        )

        if not alloc_id:
            raise MemoryError(
                f"Could not allocate {size_mb}MB for trading data: {description}"
            )

        logger.debug(f"Allocated {size_mb}MB for trading: {description}")
        yield alloc_id

    except Exception as e:
        logger.error(f"Error in trading memory context: {e}")
        raise
    finally:
        if alloc_id:
            success = manager.deallocate(alloc_id)
            if success:
                logger.debug(f"Deallocated trading memory: {description}")
            else:
                logger.warning(f"Failed to deallocate trading memory: {alloc_id}")


@contextmanager
def allocate_for_ml(
    size_mb: float,
    description: str,
    priority: int = 7,
    can_evict: bool = True,
    tags: list[str] | None = None,
):
    """
    Context manager for ML model allocations

    Args:
        size_mb: Size in megabytes
        description: Description of ML operation
        priority: Priority 1-10 (higher = more important)
        can_evict: Whether allocation can be evicted under pressure
        tags: Optional tags for categorization

    Yields:
        allocation_id: String ID for the allocation

    Example:
        with allocate_for_ml(500, "BERT model loading", priority=9, tags=['model', 'active']) as alloc_id:
            # Load and use ML model
            pass
    """
    manager = get_memory_manager()
    alloc_id = None

    try:
        # Set default tags for ML operations
        if tags is None:
            tags = ["ml_models"]
        elif "ml_models" not in tags:
            tags.append("ml_models")

        alloc_id = manager.allocate(
            component="ml_models",
            size_mb=size_mb,
            description=description,
            priority=priority,
            can_evict=can_evict,
            tags=tags,
        )

        if not alloc_id:
            raise MemoryError(f"Could not allocate {size_mb}MB for ML: {description}")

        logger.debug(f"Allocated {size_mb}MB for ML: {description}")
        yield alloc_id

    except Exception as e:
        logger.error(f"Error in ML memory context: {e}")
        raise
    finally:
        if alloc_id:
            success = manager.deallocate(alloc_id)
            if success:
                logger.debug(f"Deallocated ML memory: {description}")
            else:
                logger.warning(f"Failed to deallocate ML memory: {alloc_id}")


@contextmanager
def allocate_for_database(
    size_mb: float,
    description: str,
    priority: int = 6,
    can_evict: bool = True,
    tags: list[str] | None = None,
):
    """
    Context manager for database operations

    Args:
        size_mb: Size in megabytes
        description: Description of database operation
        priority: Priority 1-10 (higher = more important)
        can_evict: Whether allocation can be evicted under pressure
        tags: Optional tags for categorization

    Yields:
        allocation_id: String ID for the allocation

    Example:
        with allocate_for_database(200, "DuckDB query results", priority=7, tags=['query_result']) as alloc_id:
            # Process database query results
            pass
    """
    manager = get_memory_manager()
    alloc_id = None

    try:
        # Set default tags for database operations
        if tags is None:
            tags = ["database"]
        elif "database" not in tags:
            tags.append("database")

        alloc_id = manager.allocate(
            component="database",
            size_mb=size_mb,
            description=description,
            priority=priority,
            can_evict=can_evict,
            tags=tags,
        )

        if not alloc_id:
            raise MemoryError(
                f"Could not allocate {size_mb}MB for database: {description}"
            )

        logger.debug(f"Allocated {size_mb}MB for database: {description}")
        yield alloc_id

    except Exception as e:
        logger.error(f"Error in database memory context: {e}")
        raise
    finally:
        if alloc_id:
            success = manager.deallocate(alloc_id)
            if success:
                logger.debug(f"Deallocated database memory: {description}")
            else:
                logger.warning(f"Failed to deallocate database memory: {alloc_id}")


@contextmanager
def allocate_for_cache(
    size_mb: float,
    description: str,
    priority: int = 4,
    can_evict: bool = True,
    tags: list[str] | None = None,
):
    """
    Context manager for cache allocations

    Args:
        size_mb: Size in megabytes
        description: Description of cached data
        priority: Priority 1-10 (lower for cache as it's evictable)
        can_evict: Whether allocation can be evicted (should be True for cache)
        tags: Optional tags for categorization

    Yields:
        allocation_id: String ID for the allocation

    Example:
        with allocate_for_cache(50, "Options price cache", priority=5, tags=['price_cache']) as alloc_id:
            # Use cache memory
            pass
    """
    manager = get_memory_manager()
    alloc_id = None

    try:
        # Set default tags for cache operations
        if tags is None:
            tags = ["cache"]
        elif "cache" not in tags:
            tags.append("cache")

        alloc_id = manager.allocate(
            component="cache",
            size_mb=size_mb,
            description=description,
            priority=priority,
            can_evict=can_evict,
            tags=tags,
        )

        if not alloc_id:
            raise MemoryError(
                f"Could not allocate {size_mb}MB for cache: {description}"
            )

        logger.debug(f"Allocated {size_mb}MB for cache: {description}")
        yield alloc_id

    except Exception as e:
        logger.error(f"Error in cache memory context: {e}")
        raise
    finally:
        if alloc_id:
            success = manager.deallocate(alloc_id)
            if success:
                logger.debug(f"Deallocated cache memory: {description}")
            else:
                logger.warning(f"Failed to deallocate cache memory: {alloc_id}")


@contextmanager
def allocate_tensor_memory(
    shape: tuple,
    dtype: np.dtype | str,
    description: str,
    priority: int = 7,
    component: str = "ml_models",
):
    """
    Context manager for tensor allocations with automatic memory calculation

    Args:
        shape: Tensor shape tuple
        dtype: NumPy dtype or string
        description: Description of tensor usage
        priority: Priority 1-10
        component: Memory component to allocate from

    Yields:
        allocation_id: String ID for the allocation

    Example:
        with allocate_tensor_memory((1000, 512), np.float32, "Embeddings") as alloc_id:
            # Use tensor memory
            pass
    """
    # Ensure dtype is a numpy dtype object
    if isinstance(dtype, str) or not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)

    # Calculate memory needed
    elements = np.prod(shape)
    size_bytes = elements * dtype.itemsize
    size_mb = size_bytes / (1024 * 1024)

    # Use appropriate context manager based on component
    if component == "ml_models":
        context_manager = allocate_for_ml
    elif component == "trading_data":
        context_manager = allocate_for_trading
    elif component == "database":
        context_manager = allocate_for_database
    else:
        context_manager = allocate_for_cache

    tags = ["tensor", f"shape_{len(shape)}d", f"dtype_{dtype.name}"]

    with context_manager(size_mb, description, priority, True, tags) as alloc_id:
        yield alloc_id


@contextmanager
def allocate_batch_memory(
    batch_size: int,
    item_size_mb: float,
    description: str,
    component: str = "trading_data",
    priority: int = 6,
):
    """
    Context manager for batch processing allocations

    Args:
        batch_size: Number of items in batch
        item_size_mb: Size per item in MB
        description: Description of batch operation
        component: Memory component to allocate from
        priority: Priority 1-10

    Yields:
        allocation_id: String ID for the allocation

    Example:
        with allocate_batch_memory(100, 0.5, "Options batch processing") as alloc_id:
            # Process batch of 100 items, 0.5MB each
            pass
    """
    total_size_mb = batch_size * item_size_mb
    tags = ["batch", f"batch_size_{batch_size}"]

    # Use appropriate context manager
    if component == "ml_models":
        context_manager = allocate_for_ml
    elif component == "database":
        context_manager = allocate_for_database
    elif component == "cache":
        context_manager = allocate_for_cache
    else:
        context_manager = allocate_for_trading

    with context_manager(total_size_mb, description, priority, True, tags) as alloc_id:
        yield alloc_id


@contextmanager
def allocate_temporary_memory(
    size_mb: float, description: str, max_lifetime_seconds: int = 300
):
    """
    Context manager for temporary allocations with automatic timeout

    Args:
        size_mb: Size in megabytes
        description: Description of temporary usage
        max_lifetime_seconds: Maximum lifetime before forced cleanup

    Yields:
        allocation_id: String ID for the allocation

    Example:
        with allocate_temporary_memory(20, "Temporary calculation buffer", 60) as alloc_id:
            # Use memory for temporary calculations
            pass
    """
    start_time = time.time()
    tags = ["temporary", f"max_lifetime_{max_lifetime_seconds}s"]

    # Use cache component for temporary allocations (low priority, evictable)
    with allocate_for_cache(
        size_mb, f"TEMP: {description}", priority=2, tags=tags
    ) as alloc_id:
        try:
            yield alloc_id

            # Check if allocation exceeded lifetime
            elapsed = time.time() - start_time
            if elapsed > max_lifetime_seconds:
                logger.warning(
                    f"Temporary allocation exceeded lifetime: {elapsed:.1f}s > {max_lifetime_seconds}s"
                )

        except Exception as e:
            logger.error(f"Error in temporary memory context: {e}")
            raise


@contextmanager
def allocate_with_fallback(
    primary_size_mb: float,
    fallback_size_mb: float,
    description: str,
    component: str = "trading_data",
    priority: int = 6,
):
    """
    Context manager that tries primary allocation, falls back to smaller size

    Args:
        primary_size_mb: Preferred allocation size
        fallback_size_mb: Fallback size if primary fails
        description: Description of allocation
        component: Memory component
        priority: Priority 1-10

    Yields:
        tuple: (allocation_id, actual_size_mb)

    Example:
        with allocate_with_fallback(100, 50, "Options processing") as (alloc_id, size):
            # Process with actual allocated size
            pass
    """
    manager = get_memory_manager()

    # Try primary allocation first
    alloc_id = manager.allocate(
        component=component,
        size_mb=primary_size_mb,
        description=f"{description} (primary)",
        priority=priority,
        can_evict=True,
        tags=["fallback_primary"],
    )

    actual_size = primary_size_mb

    if not alloc_id:
        # Try fallback allocation
        alloc_id = manager.allocate(
            component=component,
            size_mb=fallback_size_mb,
            description=f"{description} (fallback)",
            priority=priority,
            can_evict=True,
            tags=["fallback_secondary"],
        )
        actual_size = fallback_size_mb

        if not alloc_id:
            raise MemoryError(
                f"Could not allocate even fallback size {fallback_size_mb}MB"
            )

        logger.info(
            f"Used fallback allocation: {fallback_size_mb}MB instead of {primary_size_mb}MB"
        )

    try:
        yield alloc_id, actual_size
    finally:
        if alloc_id:
            manager.deallocate(alloc_id)


@contextmanager
def allocate_adaptive_memory(
    base_size_mb: float,
    description: str,
    component: str = "trading_data",
    priority: int = 6,
    growth_factor: float = 1.5,
    max_attempts: int = 5,
):
    """
    Context manager that adaptively reduces allocation size until successful

    Args:
        base_size_mb: Initial size to try
        description: Description of allocation
        component: Memory component
        priority: Priority 1-10
        growth_factor: Factor to reduce size by each attempt (e.g., 1.5 = 67% of previous)
        max_attempts: Maximum allocation attempts

    Yields:
        tuple: (allocation_id, actual_size_mb)

    Example:
        with allocate_adaptive_memory(200, "Large dataset") as (alloc_id, size):
            # Work with whatever size was successfully allocated
            pass
    """
    manager = get_memory_manager()
    current_size = base_size_mb

    for attempt in range(max_attempts):
        alloc_id = manager.allocate(
            component=component,
            size_mb=current_size,
            description=f"{description} (attempt {attempt + 1})",
            priority=priority,
            can_evict=True,
            tags=["adaptive", f"attempt_{attempt + 1}"],
        )

        if alloc_id:
            if attempt > 0:
                logger.info(
                    f"Adaptive allocation succeeded on attempt {attempt + 1}: "
                    f"{current_size:.1f}MB (requested {base_size_mb:.1f}MB)"
                )

            try:
                yield alloc_id, current_size
                return
            finally:
                manager.deallocate(alloc_id)

        # Reduce size for next attempt
        current_size = current_size / growth_factor
        logger.debug(
            f"Allocation attempt {attempt + 1} failed, trying {current_size:.1f}MB"
        )

    raise MemoryError(
        f"Could not allocate memory after {max_attempts} attempts, "
        f"smallest attempted: {current_size:.1f}MB"
    )


@contextmanager
def allocate_with_pressure_monitoring(
    size_mb: float,
    description: str,
    component: str = "trading_data",
    priority: int = 6,
    pressure_callback: callable = None,
):
    """
    Context manager that monitors memory pressure during allocation lifetime

    Args:
        size_mb: Size in megabytes
        description: Description of allocation
        component: Memory component
        priority: Priority 1-10
        pressure_callback: Optional callback for pressure events

    Yields:
        dict: Contains allocation_id and pressure monitoring info

    Example:
        def on_pressure(level):
            print(f"Memory pressure: {level}")

        with allocate_with_pressure_monitoring(100, "Data processing",
                                             pressure_callback=on_pressure) as info:
            alloc_id = info['allocation_id']
            # Use memory while monitoring pressure
            pass
    """
    manager = get_memory_manager()

    # Register pressure callback if provided
    if pressure_callback:
        manager.register_pressure_callback(pressure_callback)

    alloc_id = None
    try:
        alloc_id = manager.allocate(
            component=component,
            size_mb=size_mb,
            description=description,
            priority=priority,
            can_evict=True,
            tags=["pressure_monitored"],
        )

        if not alloc_id:
            raise MemoryError(f"Could not allocate {size_mb}MB for {description}")

        # Get initial pressure level
        initial_pressure = manager.pressure_monitor.get_pressure_level()

        info = {
            "allocation_id": alloc_id,
            "size_mb": size_mb,
            "initial_pressure": initial_pressure,
            "component": component,
            "description": description,
        }

        yield info

    finally:
        if alloc_id:
            manager.deallocate(alloc_id)


# Utility functions for memory context managers


def estimate_tensor_memory_mb(shape: tuple, dtype: np.dtype | str) -> float:
    """Estimate memory needed for tensor in MB"""
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)

    elements = np.prod(shape)
    size_bytes = elements * dtype.itemsize
    return size_bytes / (1024 * 1024)


def estimate_dataframe_memory_mb(
    num_rows: int, num_cols: int, avg_bytes_per_cell: float = 8
) -> float:
    """Estimate memory needed for DataFrame in MB"""
    total_cells = num_rows * num_cols
    total_bytes = total_cells * avg_bytes_per_cell
    return total_bytes / (1024 * 1024)


def get_optimal_batch_size(
    total_items: int, available_memory_mb: float, item_size_mb: float
) -> int:
    """Calculate optimal batch size given memory constraints"""
    max_batch_size = int(available_memory_mb / item_size_mb)
    return min(total_items, max(1, max_batch_size))


def memory_usage_report() -> dict[str, Any]:
    """Get current memory usage report"""
    manager = get_memory_manager()
    return manager.get_status_report()
