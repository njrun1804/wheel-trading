"""
Memory Pool System - Specialized memory pools for different data types

Provides optimized memory pools with different allocation strategies,
tensor pools for ML operations, and shared memory for inter-process communication.
"""

import contextlib
import logging
import mmap
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PoolType(Enum):
    """Memory pool types"""

    GENERAL = "general"
    TENSOR = "tensor"
    SHARED = "shared"
    CIRCULAR = "circular"
    OBJECT = "object"


@dataclass
class PoolAllocation:
    """Represents a single pool allocation"""

    alloc_id: str
    offset: int
    size: int
    allocated_at: float
    description: str
    priority: int = 5
    can_evict: bool = True
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    tags: list[str] = field(default_factory=list)

    def update_access(self):
        """Update access tracking"""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class PoolStats:
    """Pool statistics"""

    total_allocations: int = 0
    active_allocations: int = 0
    total_allocated_bytes: int = 0
    peak_allocated_bytes: int = 0
    evictions_count: int = 0
    fragmentations_count: int = 0
    last_defrag_time: float = 0
    allocation_failures: int = 0
    average_allocation_size: float = 0


class MemoryPool(ABC):
    """Base class for memory pools"""

    def __init__(self, name: str, max_size_bytes: int, strategy=None):
        self.name = name
        self.max_size_bytes = max_size_bytes
        self.strategy = strategy
        self.allocated_bytes = 0
        self.allocations: dict[str, PoolAllocation] = {}
        self.stats = PoolStats()
        self.lock = threading.RLock()

        logger.debug(
            f"Created {self.__class__.__name__}: {name}, max_size={max_size_bytes/(1024**2):.1f}MB"
        )

    @abstractmethod
    def allocate(
        self,
        size_bytes: int,
        description: str,
        priority: int = 5,
        can_evict: bool = True,
    ) -> str | None:
        """Allocate memory from pool"""
        pass

    @abstractmethod
    def deallocate(self, alloc_id: str) -> bool:
        """Deallocate memory from pool"""
        pass

    @abstractmethod
    def get_memory_view(self, alloc_id: str) -> memoryview | None:
        """Get memory view for allocation"""
        pass

    def evict_for_space(self, needed_bytes: int, min_priority: int) -> int:
        """Evict allocations to make space, returns bytes evicted"""
        if not self.strategy:
            return 0

        with self.lock:
            # Get eviction candidates from strategy
            candidates = self.strategy.select_eviction_candidates(
                self.allocations, needed_bytes, min_priority
            )

            evicted_bytes = 0
            evicted_ids = []

            for alloc_id in candidates:
                if evicted_bytes >= needed_bytes:
                    break

                if alloc_id in self.allocations:
                    alloc = self.allocations[alloc_id]
                    evicted_bytes += alloc.size
                    evicted_ids.append(alloc_id)

            # Perform evictions
            for alloc_id in evicted_ids:
                self.deallocate(alloc_id)
                self.stats.evictions_count += 1

            logger.debug(
                f"{self.name}: Evicted {len(evicted_ids)} allocations, "
                f"freed {evicted_bytes / (1024**2):.1f}MB"
            )

            return evicted_bytes

    def evict_by_priority(self, max_priority: int) -> int:
        """Evict all allocations with priority <= max_priority"""
        with self.lock:
            evictable = [
                (alloc_id, alloc)
                for alloc_id, alloc in self.allocations.items()
                if alloc.can_evict and alloc.priority <= max_priority
            ]

            evicted_bytes = 0
            for alloc_id, alloc in evictable:
                evicted_bytes += alloc.size
                self.deallocate(alloc_id)
                self.stats.evictions_count += 1

            return evicted_bytes

    def cleanup(self):
        """Clean up pool resources"""
        with self.lock:
            # Deallocate all
            for alloc_id in list(self.allocations.keys()):
                self.deallocate(alloc_id)

    def get_fragmentation_ratio(self) -> float:
        """Get memory fragmentation ratio (0.0 = no fragmentation, 1.0 = highly fragmented)"""
        with self.lock:
            if not self.allocations:
                return 0.0

            # Simple fragmentation metric based on allocation spread
            allocated_regions = sorted(
                [
                    (alloc.offset, alloc.offset + alloc.size)
                    for alloc in self.allocations.values()
                ]
            )

            if len(allocated_regions) <= 1:
                return 0.0

            # Calculate gaps between allocations
            total_gaps = 0
            for i in range(1, len(allocated_regions)):
                gap = allocated_regions[i][0] - allocated_regions[i - 1][1]
                if gap > 0:
                    total_gaps += gap

            # Fragmentation ratio = wasted space / total used space
            return min(1.0, total_gaps / max(1, self.allocated_bytes))


class StandardMemoryPool(MemoryPool):
    """Standard memory pool with virtual allocation tracking"""

    def __init__(self, name: str, max_size_bytes: int, strategy=None):
        super().__init__(name, max_size_bytes, strategy)
        self.next_offset = 0
        self.free_blocks: list[tuple[int, int]] = []  # (offset, size) pairs

    def allocate(
        self,
        size_bytes: int,
        description: str,
        priority: int = 5,
        can_evict: bool = True,
    ) -> str | None:
        """Allocate memory from pool"""
        with self.lock:
            if self.allocated_bytes + size_bytes > self.max_size_bytes:
                return None

            # Apply strategy optimization if available
            if self.strategy:
                size_bytes = self.strategy.optimize_allocation_size(
                    size_bytes, description
                )

            # Find suitable free block or allocate at end
            offset = self._find_free_block(size_bytes)
            if offset is None:
                return None

            # Create allocation
            alloc_id = f"{self.name}_{int(time.time() * 1000000)}"
            self.allocations[alloc_id] = PoolAllocation(
                alloc_id=alloc_id,
                offset=offset,
                size=size_bytes,
                allocated_at=time.time(),
                description=description,
                priority=priority,
                can_evict=can_evict,
            )

            self.allocated_bytes += size_bytes
            self.stats.total_allocations += 1
            self.stats.active_allocations += 1
            self.stats.total_allocated_bytes += size_bytes

            if self.allocated_bytes > self.stats.peak_allocated_bytes:
                self.stats.peak_allocated_bytes = self.allocated_bytes

            # Update average allocation size
            self.stats.average_allocation_size = (
                self.stats.total_allocated_bytes / self.stats.total_allocations
            )

            return alloc_id

    def deallocate(self, alloc_id: str) -> bool:
        """Deallocate memory from pool"""
        with self.lock:
            if alloc_id not in self.allocations:
                return False

            alloc = self.allocations.pop(alloc_id)
            self.allocated_bytes -= alloc.size
            self.stats.active_allocations -= 1

            # Add to free blocks
            self._add_free_block(alloc.offset, alloc.size)

            return True

    def get_memory_view(self, alloc_id: str) -> memoryview | None:
        """Get memory view for allocation (placeholder - no actual memory)"""
        with self.lock:
            if alloc_id in self.allocations:
                # This is a virtual pool, return None
                # Subclasses should override for actual memory access
                return None
            return None

    def _find_free_block(self, size: int) -> int | None:
        """Find a suitable free block for allocation"""
        # Try to use existing free blocks first
        for i, (offset, block_size) in enumerate(self.free_blocks):
            if block_size >= size:
                # Remove block or split it
                if block_size == size:
                    self.free_blocks.pop(i)
                else:
                    self.free_blocks[i] = (offset + size, block_size - size)
                return offset

        # Allocate at the end
        if self.next_offset + size <= self.max_size_bytes:
            offset = self.next_offset
            self.next_offset += size
            return offset

        return None

    def _add_free_block(self, offset: int, size: int):
        """Add a free block and merge with adjacent blocks"""
        # Insert in sorted order
        inserted = False
        for i, (block_offset, _block_size) in enumerate(self.free_blocks):
            if offset < block_offset:
                self.free_blocks.insert(i, (offset, size))
                inserted = True
                break

        if not inserted:
            self.free_blocks.append((offset, size))

        # Merge adjacent blocks
        self._merge_free_blocks()

    def _merge_free_blocks(self):
        """Merge adjacent free blocks"""
        if len(self.free_blocks) <= 1:
            return

        merged = []
        current_offset, current_size = self.free_blocks[0]

        for offset, size in self.free_blocks[1:]:
            if current_offset + current_size == offset:
                # Merge blocks
                current_size += size
            else:
                merged.append((current_offset, current_size))
                current_offset, current_size = offset, size

        merged.append((current_offset, current_size))
        self.free_blocks = merged


class SharedTensorPool(MemoryPool):
    """Shared memory pool optimized for tensor operations"""

    def __init__(self, name: str, max_size_bytes: int, strategy=None):
        super().__init__(name, max_size_bytes, strategy)

        # Create shared memory region
        self.shm_name = f"/tensor_pool_{name}_{os.getpid()}"
        try:
            # Try to create new shared memory
            self.shm_fd = os.open(
                f"/tmp{self.shm_name}", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
            )
            os.ftruncate(self.shm_fd, max_size_bytes)
            self.memory_map = mmap.mmap(
                self.shm_fd,
                max_size_bytes,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
            )
            self.owns_memory = True
        except (OSError, FileExistsError):
            # Fallback to regular memory mapping
            self.shm_fd = None
            self.memory_map = mmap.mmap(-1, max_size_bytes)
            self.owns_memory = True

        self.tensor_registry: dict[str, np.ndarray] = {}
        self.next_offset = 0

        logger.info(
            f"SharedTensorPool created: {name}, size={max_size_bytes/(1024**2):.1f}MB"
        )

    def allocate_tensor(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype,
        description: str,
        priority: int = 5,
    ) -> str | None:
        """Allocate tensor in shared memory"""
        # Convert dtype to numpy dtype if needed
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        size_bytes = int(np.prod(shape) * dtype.itemsize)

        with self.lock:
            if self.allocated_bytes + size_bytes > self.max_size_bytes:
                # Try eviction
                evicted = self.evict_for_space(size_bytes, priority)
                if evicted < size_bytes:
                    self.stats.allocation_failures += 1
                    return None

            # Find offset for tensor (must be aligned)
            offset = self._find_aligned_offset(size_bytes, dtype.itemsize)
            if offset is None:
                self.stats.allocation_failures += 1
                return None

            # Create tensor view
            try:
                buffer = self.memory_map[offset : offset + size_bytes]
                tensor = np.frombuffer(buffer, dtype=dtype).reshape(shape)

                alloc_id = f"tensor_{self.name}_{len(self.allocations)}"

                self.allocations[alloc_id] = PoolAllocation(
                    alloc_id=alloc_id,
                    offset=offset,
                    size=size_bytes,
                    allocated_at=time.time(),
                    description=description,
                    priority=priority,
                    tags=["tensor"],
                )

                self.tensor_registry[alloc_id] = tensor
                self.allocated_bytes += size_bytes
                self.stats.total_allocations += 1
                self.stats.active_allocations += 1

                return alloc_id

            except Exception as e:
                logger.error(f"Failed to create tensor: {e}")
                return None

    def allocate(
        self,
        size_bytes: int,
        description: str,
        priority: int = 5,
        can_evict: bool = True,
    ) -> str | None:
        """Allocate raw memory from pool"""
        with self.lock:
            if self.allocated_bytes + size_bytes > self.max_size_bytes:
                return None

            offset = self._find_aligned_offset(size_bytes, 8)  # 8-byte alignment
            if offset is None:
                return None

            alloc_id = f"raw_{self.name}_{len(self.allocations)}"

            self.allocations[alloc_id] = PoolAllocation(
                alloc_id=alloc_id,
                offset=offset,
                size=size_bytes,
                allocated_at=time.time(),
                description=description,
                priority=priority,
                can_evict=can_evict,
            )

            self.allocated_bytes += size_bytes
            self.stats.total_allocations += 1
            self.stats.active_allocations += 1

            return alloc_id

    def deallocate(self, alloc_id: str) -> bool:
        """Deallocate memory from pool"""
        with self.lock:
            if alloc_id not in self.allocations:
                return False

            alloc = self.allocations.pop(alloc_id)
            self.allocated_bytes -= alloc.size
            self.stats.active_allocations -= 1

            # Remove from tensor registry if present
            if alloc_id in self.tensor_registry:
                del self.tensor_registry[alloc_id]

            return True

    def get_tensor(self, alloc_id: str) -> np.ndarray | None:
        """Get tensor by allocation ID"""
        with self.lock:
            return self.tensor_registry.get(alloc_id)

    def get_memory_view(self, alloc_id: str) -> memoryview | None:
        """Get memory view for allocation"""
        with self.lock:
            if alloc_id not in self.allocations:
                return None

            alloc = self.allocations[alloc_id]
            return memoryview(self.memory_map[alloc.offset : alloc.offset + alloc.size])

    def _find_aligned_offset(self, size: int, alignment: int) -> int | None:
        """Find aligned offset for allocation"""
        # Align next_offset to required boundary
        aligned_offset = ((self.next_offset + alignment - 1) // alignment) * alignment

        if aligned_offset + size <= self.max_size_bytes:
            self.next_offset = aligned_offset + size
            return aligned_offset

        return None

    def cleanup(self):
        """Clean up shared memory resources"""
        super().cleanup()

        if hasattr(self, "memory_map"):
            self.memory_map.close()

        if hasattr(self, "shm_fd") and self.shm_fd:
            os.close(self.shm_fd)
            with contextlib.suppress(Exception):
                os.unlink(f"/tmp{self.shm_name}")


class CircularPool(MemoryPool):
    """Circular buffer pool for streaming data"""

    def __init__(self, name: str, max_size_bytes: int, strategy=None):
        super().__init__(name, max_size_bytes, strategy)
        self.buffer = bytearray(max_size_bytes)
        self.head = 0  # Next write position
        self.tail = 0  # Next read position
        self.full = False
        self.active_regions: list[tuple[int, int, str]] = []  # (start, end, alloc_id)

    def allocate(
        self,
        size_bytes: int,
        description: str,
        priority: int = 5,
        can_evict: bool = True,
    ) -> str | None:
        """Allocate space in circular buffer"""
        with self.lock:
            if size_bytes > self.max_size_bytes:
                return None

            # Check if we have contiguous space
            if not self._has_space(size_bytes):
                # Try to advance tail (evict old data)
                if not self._make_space(size_bytes):
                    return None

            alloc_id = f"circular_{self.name}_{len(self.allocations)}"
            start_pos = self.head
            end_pos = (self.head + size_bytes) % self.max_size_bytes

            self.allocations[alloc_id] = PoolAllocation(
                alloc_id=alloc_id,
                offset=start_pos,
                size=size_bytes,
                allocated_at=time.time(),
                description=description,
                priority=priority,
                can_evict=can_evict,
            )

            self.active_regions.append((start_pos, end_pos, alloc_id))
            self.head = end_pos
            self.allocated_bytes += size_bytes
            self.stats.total_allocations += 1
            self.stats.active_allocations += 1

            if self.head == self.tail:
                self.full = True

            return alloc_id

    def deallocate(self, alloc_id: str) -> bool:
        """Deallocate from circular buffer"""
        with self.lock:
            if alloc_id not in self.allocations:
                return False

            alloc = self.allocations.pop(alloc_id)
            self.allocated_bytes -= alloc.size
            self.stats.active_allocations -= 1

            # Remove from active regions
            self.active_regions = [
                (start, end, aid)
                for start, end, aid in self.active_regions
                if aid != alloc_id
            ]

            # Update tail if this was the oldest allocation
            if self.active_regions:
                self.tail = min(start for start, _, _ in self.active_regions)
            else:
                self.tail = self.head

            self.full = False
            return True

    def get_memory_view(self, alloc_id: str) -> memoryview | None:
        """Get memory view for allocation"""
        with self.lock:
            if alloc_id not in self.allocations:
                return None

            alloc = self.allocations[alloc_id]
            start = alloc.offset
            size = alloc.size

            # Handle wrap-around
            if start + size <= self.max_size_bytes:
                return memoryview(self.buffer[start : start + size])
            else:
                # Allocation wraps around - need to concatenate
                first_part = self.buffer[start:]
                second_part = self.buffer[: (start + size) % self.max_size_bytes]
                combined = bytearray(first_part + second_part)
                return memoryview(combined)

    def _has_space(self, size: int) -> bool:
        """Check if circular buffer has contiguous space"""
        if not self.full:
            # Simple case: buffer not full
            return size <= self.max_size_bytes - self.head

        # Buffer is full, check if we can wrap around
        return size <= (self.tail - self.head) if self.tail > self.head else False

    def _make_space(self, size: int) -> bool:
        """Make space by advancing tail (evicting old data)"""
        space_needed = size

        while space_needed > 0 and self.active_regions:
            # Find oldest region to evict
            oldest_region = min(self.active_regions, key=lambda x: x[0])
            start, end, alloc_id = oldest_region

            # Evict the allocation
            if alloc_id in self.allocations:
                self.deallocate(alloc_id)
                space_needed -= (end - start) % self.max_size_bytes

        return space_needed <= 0


class ObjectPool(MemoryPool):
    """Pool for managing reusable objects"""

    def __init__(
        self,
        name: str,
        max_objects: int,
        object_factory: callable,
        object_reset: callable = None,
    ):
        # Convert max_objects to approximate byte size (rough estimate)
        super().__init__(name, max_objects * 1024, None)  # 1KB per object estimate

        self.max_objects = max_objects
        self.object_factory = object_factory
        self.object_reset = object_reset or (lambda x: None)

        self.available_objects: list[Any] = []
        self.active_objects: dict[str, Any] = {}
        self.object_count = 0

    def allocate(
        self,
        size_bytes: int = 0,
        description: str = "",
        priority: int = 5,
        can_evict: bool = True,
    ) -> str | None:
        """Allocate an object from the pool"""
        with self.lock:
            if self.object_count >= self.max_objects and not self.available_objects:
                return None

            # Get object from pool or create new one
            if self.available_objects:
                obj = self.available_objects.pop()
            else:
                try:
                    obj = self.object_factory()
                    self.object_count += 1
                except Exception as e:
                    logger.error(f"Failed to create object: {e}")
                    return None

            alloc_id = f"obj_{self.name}_{len(self.allocations)}"

            self.allocations[alloc_id] = PoolAllocation(
                alloc_id=alloc_id,
                offset=0,  # Not applicable for objects
                size=1024,  # Rough estimate
                allocated_at=time.time(),
                description=description,
                priority=priority,
                can_evict=can_evict,
            )

            self.active_objects[alloc_id] = obj
            self.stats.total_allocations += 1
            self.stats.active_allocations += 1

            return alloc_id

    def deallocate(self, alloc_id: str) -> bool:
        """Return object to pool"""
        with self.lock:
            if alloc_id not in self.allocations:
                return False

            self.allocations.pop(alloc_id)
            obj = self.active_objects.pop(alloc_id)

            # Reset object and return to available pool
            try:
                self.object_reset(obj)
                self.available_objects.append(obj)
            except Exception as e:
                logger.warning(f"Failed to reset object: {e}")
                # Object unusable, don't return to pool
                self.object_count -= 1

            self.stats.active_allocations -= 1
            return True

    def get_object(self, alloc_id: str) -> Any | None:
        """Get object by allocation ID"""
        with self.lock:
            return self.active_objects.get(alloc_id)

    def get_memory_view(self, alloc_id: str) -> memoryview | None:
        """Not applicable for object pools"""
        return None

    def cleanup(self):
        """Clean up object pool"""
        super().cleanup()
        with self.lock:
            self.available_objects.clear()
            self.active_objects.clear()
            self.object_count = 0


# Pool factory functions
def create_trading_data_pool(size_mb: int) -> StandardMemoryPool:
    """Create optimized pool for trading data"""
    from .allocation_strategies import TradingDataStrategy

    return StandardMemoryPool(
        "trading_data", size_mb * 1024 * 1024, TradingDataStrategy()
    )


def create_ml_tensor_pool(size_mb: int) -> SharedTensorPool:
    """Create shared tensor pool for ML operations"""
    from .allocation_strategies import MLModelStrategy

    return SharedTensorPool("ml_tensors", size_mb * 1024 * 1024, MLModelStrategy())


def create_cache_pool(size_mb: int) -> StandardMemoryPool:
    """Create pool for general caching"""
    from .allocation_strategies import CacheStrategy

    return StandardMemoryPool("cache", size_mb * 1024 * 1024, CacheStrategy())


def create_circular_buffer_pool(size_mb: int) -> CircularPool:
    """Create circular buffer for streaming data"""
    return CircularPool("stream_buffer", size_mb * 1024 * 1024)
