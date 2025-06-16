"""
Unified Memory Management System for Trading Components

This module provides centralized memory management with:
- Dynamic allocation strategies for different component types
- Memory pressure detection and automatic cleanup
- Intelligent resource pooling and eviction policies
- Integration with trading system components
- Memory-aware resource scheduling
- Specialized memory pools for different data types

Core Features:
- Component-specific memory budgets and priorities
- Real-time pressure monitoring with adaptive responses
- Automatic cleanup of unused allocations
- Shared memory pools for inter-process communication
- Emergency measures to prevent system crashes
- Memory-aware task scheduling and resource management
- Context managers for safe memory allocation
- Garbage collection optimization
"""

from .allocation_strategies import (
    AllocationRequest,
    AllocationStrategy,
    CacheStrategy,
    DatabaseStrategy,
    EvictionPolicy,
    MLModelStrategy,
    TradingDataStrategy,
)
from .cleanup_system import CleanupLevel, CleanupSystem, get_cleanup_system
from .context_managers import (
    allocate_adaptive_memory,
    allocate_batch_memory,
    allocate_for_cache,
    allocate_for_database,
    allocate_for_ml,
    allocate_for_trading,
    allocate_temporary_memory,
    allocate_tensor_memory,
    allocate_with_fallback,
    allocate_with_pressure_monitoring,
    estimate_dataframe_memory_mb,
    estimate_tensor_memory_mb,
    get_optimal_batch_size,
    memory_usage_report,
)
from .memory_pools import (
    CircularPool,
    MemoryPool,
    ObjectPool,
    PoolType,
    SharedTensorPool,
    StandardMemoryPool,
    create_cache_pool,
    create_circular_buffer_pool,
    create_ml_tensor_pool,
    create_trading_data_pool,
)
from .pressure_monitor import (
    MemoryReading,
    PressureLevel,
    PressureMonitor,
    get_pressure_monitor,
)
from .resource_scheduler import (
    ResourceRequirements,
    ResourceScheduler,
    ScheduledTask,
    SchedulingStrategy,
    TaskPriority,
    TaskState,
    get_resource_scheduler,
    schedule_database_task,
    schedule_ml_task,
    schedule_trading_task,
)
from .unified_manager import UnifiedMemoryManager, get_memory_manager

__all__ = [
    # Core memory management
    "UnifiedMemoryManager",
    "get_memory_manager",
    # Allocation strategies
    "AllocationStrategy",
    "TradingDataStrategy",
    "MLModelStrategy",
    "CacheStrategy",
    "DatabaseStrategy",
    "EvictionPolicy",
    "AllocationRequest",
    # Pressure monitoring
    "PressureMonitor",
    "get_pressure_monitor",
    "PressureLevel",
    "MemoryReading",
    # Cleanup system
    "CleanupSystem",
    "get_cleanup_system",
    "CleanupLevel",
    # Memory pools
    "MemoryPool",
    "StandardMemoryPool",
    "SharedTensorPool",
    "CircularPool",
    "ObjectPool",
    "PoolType",
    "create_trading_data_pool",
    "create_ml_tensor_pool",
    "create_cache_pool",
    "create_circular_buffer_pool",
    # Context managers
    "allocate_for_trading",
    "allocate_for_ml",
    "allocate_for_cache",
    "allocate_for_database",
    "allocate_tensor_memory",
    "allocate_batch_memory",
    "allocate_temporary_memory",
    "allocate_with_fallback",
    "allocate_adaptive_memory",
    "allocate_with_pressure_monitoring",
    "estimate_tensor_memory_mb",
    "estimate_dataframe_memory_mb",
    "get_optimal_batch_size",
    "memory_usage_report",
    # Resource scheduling
    "ResourceScheduler",
    "get_resource_scheduler",
    "ScheduledTask",
    "ResourceRequirements",
    "TaskPriority",
    "TaskState",
    "SchedulingStrategy",
    "schedule_trading_task",
    "schedule_ml_task",
    "schedule_database_task",
]
