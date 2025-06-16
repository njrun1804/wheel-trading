"""
Memory Pools - Efficient allocation for fixed-size objects.

Consolidates buffer pool implementations.
"""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class MemoryPool(ABC):
    """Abstract base class for memory pools."""

    @abstractmethod
    def acquire(self) -> Any:
        """Acquire an object from the pool."""
        pass

    @abstractmethod
    def release(self, obj: Any):
        """Release an object back to the pool."""
        pass


class PooledBuffer:
    """A buffer that can be pooled and reused."""

    def __init__(self, size: int):
        self.size = size
        self.data = bytearray(size)
        self.in_use = False

    def reset(self):
        """Reset buffer for reuse."""
        self.data = bytearray(self.size)
        self.in_use = False


class BufferPool(MemoryPool):
    """Pool for reusable byte buffers."""

    def __init__(self, buffer_size: int, initial_count: int = 10, max_count: int = 100):
        self.buffer_size = buffer_size
        self.max_count = max_count
        self._lock = threading.Lock()
        self._pool: list[PooledBuffer] = []

        # Pre-allocate initial buffers
        for _ in range(initial_count):
            self._pool.append(PooledBuffer(buffer_size))

    def acquire(self) -> PooledBuffer:
        """Acquire a buffer from the pool."""
        with self._lock:
            if self._pool:
                buffer = self._pool.pop()
                buffer.in_use = True
                return buffer
            else:
                # Create new buffer if pool is empty
                return PooledBuffer(self.buffer_size)

    def release(self, buffer: PooledBuffer):
        """Release a buffer back to the pool."""
        with self._lock:
            if len(self._pool) < self.max_count:
                buffer.reset()
                self._pool.append(buffer)
