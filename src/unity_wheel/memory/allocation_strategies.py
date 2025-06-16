"""
Memory Allocation Strategies for Different Trading System Components

Each component has different memory usage patterns and requirements.
These strategies optimize allocation behavior for each use case.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EvictionPolicy(Enum):
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    PRIORITY = "priority"  # Priority-based
    TTL = "ttl"  # Time To Live
    HYBRID = "hybrid"  # Combination approach


@dataclass
class AllocationRequest:
    """Request for memory allocation"""

    size_bytes: int
    description: str
    priority: int
    can_evict: bool
    tags: list[str]
    ttl: float | None = None  # Time to live in seconds


class AllocationStrategy(ABC):
    """Base class for component-specific allocation strategies"""

    def __init__(self, component_name: str):
        self.component_name = component_name
        self.eviction_policy = EvictionPolicy.HYBRID
        self.allocation_history: list[dict] = []

    @abstractmethod
    def should_accept_allocation(
        self, request: AllocationRequest, current_usage: float, max_size: int
    ) -> bool:
        """Determine if allocation should be accepted"""
        pass

    @abstractmethod
    def select_eviction_candidates(
        self, allocations: dict[str, Any], needed_bytes: int, min_priority: int
    ) -> list[str]:
        """Select allocations to evict"""
        pass

    @abstractmethod
    def optimize_allocation_size(self, requested_bytes: int, description: str) -> int:
        """Optimize the requested allocation size"""
        pass

    def get_priority_adjustment(self, request: AllocationRequest) -> int:
        """Adjust priority based on component-specific logic"""
        return 0  # No adjustment by default

    def record_allocation(self, alloc_id: str, request: AllocationRequest):
        """Record allocation for strategy optimization"""
        self.allocation_history.append(
            {
                "timestamp": time.time(),
                "alloc_id": alloc_id,
                "size_bytes": request.size_bytes,
                "description": request.description,
                "priority": request.priority,
            }
        )

        # Keep only recent history
        if len(self.allocation_history) > 1000:
            self.allocation_history = self.allocation_history[-500:]


class TradingDataStrategy(AllocationStrategy):
    """Strategy optimized for trading data (price data, options chains, market data)"""

    def __init__(self):
        super().__init__("trading_data")
        self.eviction_policy = EvictionPolicy.LRU

        # Trading data specific parameters
        self.high_frequency_threshold = 100 * 1024 * 1024  # 100MB
        self.batch_size_preference = 50 * 1024 * 1024  # 50MB chunks

    def should_accept_allocation(
        self, request: AllocationRequest, current_usage: float, max_size: int
    ) -> bool:
        """Accept allocation based on trading data patterns"""
        usage_percent = current_usage / max_size

        # Always accept high priority market data
        if request.priority >= 8 and "market_data" in request.tags:
            return True

        # Accept options data if under 80% usage
        if "options" in request.tags and usage_percent < 0.8:
            return True

        # Accept price data if under 70% usage
        if "price_data" in request.tags and usage_percent < 0.7:
            return True

        # General acceptance criteria
        return usage_percent < 0.6

    def select_eviction_candidates(
        self, allocations: dict[str, Any], needed_bytes: int, min_priority: int
    ) -> list[str]:
        """Select trading data allocations for eviction using LRU + priority"""
        candidates = []

        # Filter evictable allocations below priority threshold
        evictable = [
            (alloc_id, alloc)
            for alloc_id, alloc in allocations.items()
            if alloc.can_evict and alloc.priority < min_priority
        ]

        # Prioritize historical data over current data
        historical_data = [
            (aid, alloc) for aid, alloc in evictable if "historical" in alloc.tags
        ]

        # Sort by last access time (LRU)
        historical_data.sort(key=lambda x: x[1].last_accessed)
        candidates.extend([aid for aid, _ in historical_data])

        # Add other data sorted by LRU if still need space
        other_data = [
            (aid, alloc) for aid, alloc in evictable if "historical" not in alloc.tags
        ]
        other_data.sort(key=lambda x: x[1].last_accessed)
        candidates.extend([aid for aid, _ in other_data])

        return candidates

    def optimize_allocation_size(self, requested_bytes: int, description: str) -> int:
        """Optimize allocation size for trading data patterns"""
        # Round up to batch size for better memory alignment
        if requested_bytes < self.batch_size_preference:
            return (
                (requested_bytes - 1) // self.batch_size_preference + 1
            ) * self.batch_size_preference

        # For large allocations, round to 10MB boundaries
        if requested_bytes > self.high_frequency_threshold:
            boundary = 10 * 1024 * 1024
            return ((requested_bytes - 1) // boundary + 1) * boundary

        return requested_bytes

    def get_priority_adjustment(self, request: AllocationRequest) -> int:
        """Adjust priority for trading data"""
        adjustment = 0

        # Boost priority for real-time data
        if "real_time" in request.tags:
            adjustment += 2

        # Boost priority for options data during market hours
        if "options" in request.tags:
            current_hour = time.localtime().tm_hour
            if 9 <= current_hour <= 16:  # Market hours
                adjustment += 1

        return adjustment


class MLModelStrategy(AllocationStrategy):
    """Strategy optimized for ML models and embeddings"""

    def __init__(self):
        super().__init__("ml_models")
        self.eviction_policy = EvictionPolicy.PRIORITY

        # ML specific parameters
        self.model_size_threshold = 500 * 1024 * 1024  # 500MB
        self.embedding_batch_size = 32 * 1024 * 1024  # 32MB

    def should_accept_allocation(
        self, request: AllocationRequest, current_usage: float, max_size: int
    ) -> bool:
        """Accept allocation based on ML patterns"""
        usage_percent = current_usage / max_size

        # Always accept high priority model loading
        if request.priority >= 9 and "model_loading" in request.tags:
            return True

        # Accept embeddings if under 75% usage
        if "embeddings" in request.tags and usage_percent < 0.75:
            return True

        # Accept training data if under 60% usage
        if "training" in request.tags and usage_percent < 0.6:
            return True

        # General acceptance
        return usage_percent < 0.5

    def select_eviction_candidates(
        self, allocations: dict[str, Any], needed_bytes: int, min_priority: int
    ) -> list[str]:
        """Select ML allocations for eviction using priority + size"""

        # Filter evictable allocations
        evictable = [
            (alloc_id, alloc)
            for alloc_id, alloc in allocations.items()
            if alloc.can_evict and alloc.priority < min_priority
        ]

        # Prioritize by type: temporary > cache > embeddings > models
        type_priority = {"temporary": 1, "cache": 2, "embeddings": 3, "model": 4}

        def get_type_priority(alloc):
            for tag in alloc.tags:
                if tag in type_priority:
                    return type_priority[tag]
            return 5

        # Sort by type priority, then by allocation priority (ascending)
        evictable.sort(key=lambda x: (get_type_priority(x[1]), x[1].priority))

        return [alloc_id for alloc_id, _ in evictable]

    def optimize_allocation_size(self, requested_bytes: int, description: str) -> int:
        """Optimize allocation size for ML operations"""
        # Round embeddings to batch boundaries
        if "embedding" in description.lower():
            return (
                (requested_bytes - 1) // self.embedding_batch_size + 1
            ) * self.embedding_batch_size

        # Round model allocations to 64MB boundaries for better GPU transfer
        if "model" in description.lower():
            boundary = 64 * 1024 * 1024
            return ((requested_bytes - 1) // boundary + 1) * boundary

        return requested_bytes

    def get_priority_adjustment(self, request: AllocationRequest) -> int:
        """Adjust priority for ML operations"""
        adjustment = 0

        # Boost active model priority
        if "active_model" in request.tags:
            adjustment += 3

        # Boost training operations
        if "training" in request.tags:
            adjustment += 1

        return adjustment


class DatabaseStrategy(AllocationStrategy):
    """Strategy optimized for database operations (DuckDB, SQLite)"""

    def __init__(self):
        super().__init__("database")
        self.eviction_policy = EvictionPolicy.HYBRID

        # Database specific parameters
        self.query_cache_threshold = 10 * 1024 * 1024  # 10MB
        self.result_set_threshold = 100 * 1024 * 1024  # 100MB

    def should_accept_allocation(
        self, request: AllocationRequest, current_usage: float, max_size: int
    ) -> bool:
        """Accept allocation based on database patterns"""
        usage_percent = current_usage / max_size

        # Always accept critical database operations
        if request.priority >= 8 and "critical_query" in request.tags:
            return True

        # Accept query results if under 80% usage
        if "query_result" in request.tags and usage_percent < 0.8:
            return True

        # Accept cache operations if under 70% usage
        if "cache" in request.tags and usage_percent < 0.7:
            return True

        # General acceptance
        return usage_percent < 0.6

    def select_eviction_candidates(
        self, allocations: dict[str, Any], needed_bytes: int, min_priority: int
    ) -> list[str]:
        """Select database allocations for eviction using hybrid approach"""
        candidates = []

        # Filter evictable allocations
        evictable = [
            (alloc_id, alloc)
            for alloc_id, alloc in allocations.items()
            if alloc.can_evict and alloc.priority < min_priority
        ]

        # Separate by type for different eviction strategies
        cache_allocs = [
            (aid, alloc) for aid, alloc in evictable if "cache" in alloc.tags
        ]
        result_allocs = [
            (aid, alloc) for aid, alloc in evictable if "query_result" in alloc.tags
        ]
        other_allocs = [
            (aid, alloc)
            for aid, alloc in evictable
            if "cache" not in alloc.tags and "query_result" not in alloc.tags
        ]

        # Evict cache using LRU
        cache_allocs.sort(key=lambda x: x[1].last_accessed)
        candidates.extend([aid for aid, _ in cache_allocs])

        # Evict old results using TTL + size
        current_time = time.time()
        result_allocs.sort(
            key=lambda x: (
                current_time - x[1].allocated_at,
                -x[1].size_bytes,  # Larger allocations first
            )
        )
        candidates.extend([aid for aid, _ in result_allocs])

        # Evict others by priority
        other_allocs.sort(key=lambda x: x[1].priority)
        candidates.extend([aid for aid, _ in other_allocs])

        return candidates

    def optimize_allocation_size(self, requested_bytes: int, description: str) -> int:
        """Optimize allocation size for database operations"""
        # Round query results to page boundaries (64KB)
        if "query" in description.lower():
            page_size = 64 * 1024
            return ((requested_bytes - 1) // page_size + 1) * page_size

        # Round cache to 1MB boundaries
        if "cache" in description.lower():
            boundary = 1024 * 1024
            return ((requested_bytes - 1) // boundary + 1) * boundary

        return requested_bytes


class CacheStrategy(AllocationStrategy):
    """Strategy optimized for general caching operations"""

    def __init__(self):
        super().__init__("cache")
        self.eviction_policy = EvictionPolicy.LRU

        # Cache specific parameters
        self.small_object_threshold = 1024 * 1024  # 1MB
        self.large_object_threshold = 10 * 1024 * 1024  # 10MB

    def should_accept_allocation(
        self, request: AllocationRequest, current_usage: float, max_size: int
    ) -> bool:
        """Accept allocation based on caching patterns"""
        usage_percent = current_usage / max_size

        # Always accept high priority cache items
        if request.priority >= 7:
            return True

        # Accept if under usage threshold
        return usage_percent < 0.8

    def select_eviction_candidates(
        self, allocations: dict[str, Any], needed_bytes: int, min_priority: int
    ) -> list[str]:
        """Select cache allocations for eviction using LRU"""

        # Filter evictable allocations (caches are always evictable)
        evictable = [
            (alloc_id, alloc)
            for alloc_id, alloc in allocations.items()
            if alloc.priority < min_priority
        ]

        # Sort by last access time (LRU)
        evictable.sort(key=lambda x: x[1].last_accessed)

        return [alloc_id for alloc_id, _ in evictable]

    def optimize_allocation_size(self, requested_bytes: int, description: str) -> int:
        """Optimize allocation size for cache operations"""
        # Round small objects to 4KB boundaries
        if requested_bytes <= self.small_object_threshold:
            boundary = 4096
            return ((requested_bytes - 1) // boundary + 1) * boundary

        # Round large objects to 1MB boundaries
        if requested_bytes >= self.large_object_threshold:
            boundary = 1024 * 1024
            return ((requested_bytes - 1) // boundary + 1) * boundary

        # Medium objects to 64KB boundaries
        boundary = 64 * 1024
        return ((requested_bytes - 1) // boundary + 1) * boundary


# Strategy registry
_strategies = {
    "trading_data": TradingDataStrategy,
    "ml_models": MLModelStrategy,
    "database": DatabaseStrategy,
    "cache": CacheStrategy,
}


def get_strategy_for_component(component: str) -> AllocationStrategy:
    """Get the allocation strategy for a component"""
    if component in _strategies:
        return _strategies[component]()
    else:
        logger.warning(
            f"No specific strategy for component {component}, using base strategy"
        )
        return AllocationStrategy(component)


def register_strategy(component: str, strategy_class: type):
    """Register a custom strategy for a component"""
    _strategies[component] = strategy_class
