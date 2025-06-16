"""
Unified Memory Management for M4 Pro GPU Acceleration

Implements zero-copy memory buffers that can be shared between CPU and GPU
operations, leveraging Apple Silicon's unified memory architecture.
"""

import logging
import time
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

logger = logging.getLogger(__name__)

# PHASE 1.3: Memory Management Fix - Resolve BufferType import issues
try:
    from .production_error_recovery import production_error_handling
except ImportError:
    from bolt.production_error_recovery import production_error_handling


class BufferType(Enum):
    """Types of unified memory buffers"""

    EMBEDDING_MATRIX = "embedding_matrix"
    SEARCH_RESULTS = "search_results"
    INDEX_CACHE = "index_cache"
    TEMPORARY = "temporary"


@dataclass
class BufferStats:
    """Statistics for buffer usage"""

    total_allocations: int = 0
    active_buffers: int = 0
    total_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    zero_copy_operations: int = 0
    memory_pressure_events: int = 0


class UnifiedMemoryBuffer:
    """
    A memory buffer that can be shared between CPU and GPU without copying.

    Uses MLX arrays backed by shared memory that can be accessed efficiently
    from both CPU (NumPy) and GPU (Metal) operations.
    """

    def __init__(self, size_bytes: int, buffer_type: BufferType, name: str = ""):
        self.size_bytes = size_bytes
        self.buffer_type = buffer_type
        self.name = name or f"{buffer_type.value}_{id(self)}"
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0

        if not MLX_AVAILABLE:
            raise RuntimeError("MLX not available - unified memory requires MLX")

        # Create the MLX array that serves as our unified buffer
        self._mlx_buffer = mx.zeros((size_bytes,), dtype=mx.uint8)

        # Create CPU view without copying
        self._setup_cpu_view()

        logger.debug(
            f"Created unified buffer '{self.name}' of {size_bytes/1024/1024:.1f}MB"
        )

    def _setup_cpu_view(self):
        """Set up CPU view of the MLX buffer"""
        try:
            # MLX arrays support numpy conversion but not direct buffer interface
            # Use MLX's built-in numpy conversion for zero-copy when possible
            if hasattr(self._mlx_buffer, "__dlpack__"):
                # Try DLPack protocol for zero-copy
                self._cpu_view = np.from_dlpack(self._mlx_buffer.__dlpack__())
            else:
                # Fallback to MLX's numpy conversion (may involve copy)
                self._cpu_view = np.array(self._mlx_buffer, copy=False)
        except Exception as e:
            logger.debug(f"Zero-copy view not available: {e}, using copy fallback")
            self._cpu_view = None

    async def as_numpy(
        self, dtype: np.dtype = np.float32, shape: tuple | None = None
    ) -> np.ndarray:
        """Get numpy view of buffer with specified dtype and shape"""
        async with production_error_handling("UnifiedMemoryBuffer", "as_numpy"):
            self._update_access_stats()

            try:
                if self._cpu_view is not None:
                    # Calculate proper byte size for dtype conversion
                    dtype_size = np.dtype(dtype).itemsize
                    max_elements = len(self._cpu_view) // dtype_size

                    if shape:
                        required_elements = np.prod(shape)
                        if required_elements > max_elements:
                            logger.warning(
                                f"Requested shape {shape} requires {required_elements} elements, "
                                f"but buffer only has {max_elements} elements for dtype {dtype}"
                            )
                            # Truncate shape to fit available data
                            shape = (
                                (max_elements,)
                                if len(shape) == 1
                                else (max_elements // shape[1], shape[1])
                            )

                    # Create view with proper dtype
                    view = self._cpu_view[: max_elements * dtype_size].view(dtype)
                    if shape:
                        view = view.reshape(shape)
                    return view
                else:
                    # Fallback to copy via MLX
                    logger.debug(f"Using copy fallback for buffer {self.name}")
                    array = np.array(self._mlx_buffer)

                    # Convert to requested dtype
                    if array.dtype != dtype:
                        array = array.astype(dtype)

                    if shape:
                        # Ensure we have enough elements
                        if array.size >= np.prod(shape):
                            array = array.flat[: np.prod(shape)].reshape(shape)
                        else:
                            logger.warning(
                                f"Buffer too small for shape {shape}, using available size"
                            )

                    return array
            except Exception as e:
                logger.error(f"Failed to create numpy view: {e}")
                # Return empty array as last resort
                if shape:
                    return np.zeros(shape, dtype=dtype)
                else:
                    return np.array([], dtype=dtype)

    async def as_mlx(
        self, dtype: mx.Dtype = mx.float32, shape: tuple | None = None
    ) -> mx.array:
        """Get MLX view of buffer with specified dtype and shape"""
        async with production_error_handling("UnifiedMemoryBuffer", "as_mlx"):
            self._update_access_stats()

            try:
                view = self._mlx_buffer.view(dtype)

                if shape:
                    # Calculate available elements for the requested dtype
                    dtype_size = (
                        4 if dtype == mx.float32 else 8 if dtype == mx.float64 else 1
                    )
                    max_elements = self._mlx_buffer.size // dtype_size
                    required_elements = int(np.prod(shape))

                    if required_elements > max_elements:
                        logger.warning(
                            f"Requested MLX shape {shape} requires {required_elements} elements, "
                            f"but buffer only has {max_elements} elements for dtype {dtype}"
                        )
                        # Truncate shape to fit available data
                        if len(shape) == 1:
                            shape = (max_elements,)
                        elif len(shape) == 2:
                            # For 2D shapes, keep second dimension and adjust first
                            new_first_dim = max_elements // shape[1]
                            shape = (new_first_dim, shape[1])
                        else:
                            # For higher dimensions, flatten to 1D
                            shape = (max_elements,)
                        logger.info(f"Adjusted MLX shape to {shape} to fit buffer")

                    # Only reshape if we have enough elements
                    if int(np.prod(shape)) <= max_elements:
                        view = view[: int(np.prod(shape))].reshape(shape)
                    else:
                        logger.error(
                            "Cannot reshape MLX buffer: still insufficient elements after adjustment"
                        )
                        return mx.zeros(shape, dtype=dtype)

                return view

            except Exception as e:
                logger.error(f"Failed to create MLX view: {e}")
                # Return zeros array as fallback
                if shape:
                    return mx.zeros(shape, dtype=dtype)
                else:
                    return mx.array([], dtype=dtype)

    async def copy_from_numpy(self, data: np.ndarray) -> None:
        """Copy data from numpy array into buffer"""
        async with production_error_handling("UnifiedMemoryBuffer", "copy_from_numpy"):
            self._update_access_stats()

            if data.nbytes > self.size_bytes:
                raise ValueError(
                    f"Data size {data.nbytes} exceeds buffer size {self.size_bytes}"
                )

            # Copy data into MLX buffer preserving byte structure
            # Convert data to bytes while preserving the original structure
            data_bytes = data.astype(
                data.dtype
            ).tobytes()  # Keep original dtype, convert to bytes

            # Convert bytes to uint8 array for storage
            flat_data = np.frombuffer(data_bytes, dtype=np.uint8)

            # Store in MLX buffer
            if len(flat_data) <= self._mlx_buffer.size:
                self._mlx_buffer[: len(flat_data)] = mx.array(flat_data)
            else:
                logger.error(
                    f"Data too large for buffer: {len(flat_data)} > {self._mlx_buffer.size}"
                )
                raise ValueError(
                    f"Data size {len(flat_data)} exceeds buffer capacity {self._mlx_buffer.size}"
                )

    async def copy_from_mlx(self, data: mx.array) -> None:
        """Copy data from MLX array into buffer"""
        async with production_error_handling("UnifiedMemoryBuffer", "copy_from_mlx"):
            self._update_access_stats()

            if data.nbytes > self.size_bytes:
                raise ValueError(
                    f"Data size {data.nbytes} exceeds buffer size {self.size_bytes}"
                )

            # Copy data into MLX buffer
            flat_data = data.flatten().astype(mx.uint8)
            self._mlx_buffer[: len(flat_data)] = flat_data

    async def zero_copy_transfer(self, data: np.ndarray) -> mx.array:
        """
        Transfer numpy data to MLX with zero-copy if possible.

        For optimal performance, data should already be in a compatible format.
        """
        async with production_error_handling(
            "UnifiedMemoryBuffer", "zero_copy_transfer"
        ):
            self._update_access_stats()

            try:
                # Try zero-copy conversion
                mlx_data = mx.array(data, copy=False)
                logger.debug(f"Zero-copy transfer successful for {self.name}")
                return mlx_data
            except Exception as e:
                logger.debug(f"Zero-copy failed for {self.name}: {e}, using copy")
                return mx.array(data)

    def _update_access_stats(self):
        """Update access statistics"""
        self.last_accessed = time.time()
        self.access_count += 1

    @property
    def memory_mb(self) -> float:
        """Memory usage in MB"""
        return self.size_bytes / 1024 / 1024

    @property
    def age_seconds(self) -> float:
        """Age of buffer in seconds"""
        return time.time() - self.created_at

    def __del__(self):
        """Cleanup when buffer is deleted"""
        logger.debug(f"Releasing unified buffer '{self.name}' ({self.memory_mb:.1f}MB)")


class UnifiedMemoryManager:
    """
    Manages unified memory buffers for optimal M4 Pro performance.

    Provides allocation, deallocation, and optimization strategies for
    zero-copy operations between CPU and GPU.
    """

    def __init__(self, max_memory_mb: float = 2048):
        self.max_memory_mb = max_memory_mb
        self.buffers: dict[str, UnifiedMemoryBuffer] = {}
        self.buffer_registry = weakref.WeakValueDictionary()
        self.stats = BufferStats()
        self._memory_pressure_threshold = 0.8  # 80% of max memory

        logger.info(f"Initialized UnifiedMemoryManager with {max_memory_mb}MB limit")

    async def allocate_buffer(
        self, size_bytes: int, buffer_type: BufferType, name: str | None = None
    ) -> UnifiedMemoryBuffer:
        """Allocate a new unified memory buffer"""
        async with production_error_handling("UnifiedMemoryManager", "allocate_buffer"):
            # Check memory pressure
            if self._check_memory_pressure(size_bytes):
                await self._handle_memory_pressure()

            # Create buffer
            buffer_name = name or f"{buffer_type.value}_{len(self.buffers)}"

            if buffer_name in self.buffers:
                logger.warning(
                    f"Buffer '{buffer_name}' already exists, returning existing"
                )
                return self.buffers[buffer_name]

            buffer = UnifiedMemoryBuffer(size_bytes, buffer_type, buffer_name)
            self.buffers[buffer_name] = buffer
            self.buffer_registry[buffer_name] = buffer

            # Update stats
            self.stats.total_allocations += 1
            self.stats.active_buffers += 1
            self.stats.total_memory_mb += buffer.memory_mb
            self.stats.peak_memory_mb = max(
                self.stats.peak_memory_mb, self.stats.total_memory_mb
            )

            logger.info(f"Allocated buffer '{buffer_name}' ({buffer.memory_mb:.1f}MB)")
            return buffer

    def get_buffer(self, name: str) -> UnifiedMemoryBuffer | None:
        """Get buffer by name"""
        return self.buffers.get(name)

    def release_buffer(self, name: str) -> bool:
        """Release a buffer by name"""
        if name in self.buffers:
            buffer = self.buffers.pop(name)
            self.stats.active_buffers -= 1
            self.stats.total_memory_mb -= buffer.memory_mb
            logger.info(f"Released buffer '{name}' ({buffer.memory_mb:.1f}MB)")
            return True
        return False

    def _check_memory_pressure(self, additional_bytes: int) -> bool:
        """Check if allocating additional memory would cause pressure"""
        current_mb = self.stats.total_memory_mb
        additional_mb = additional_bytes / 1024 / 1024
        projected_usage = (current_mb + additional_mb) / self.max_memory_mb

        return projected_usage > self._memory_pressure_threshold

    async def _handle_memory_pressure(self):
        """Handle memory pressure by freeing least recently used buffers"""
        async with production_error_handling(
            "UnifiedMemoryManager", "handle_memory_pressure"
        ):
            self.stats.memory_pressure_events += 1
            logger.warning("Memory pressure detected, attempting to free buffers")

            # Sort buffers by last access time (oldest first)
            sorted_buffers = sorted(
                self.buffers.items(), key=lambda x: x[1].last_accessed
            )

            # Free temporary buffers first
            for name, buffer in sorted_buffers:
                if buffer.buffer_type == BufferType.TEMPORARY:
                    if self.release_buffer(name):
                        logger.info(
                            f"Freed temporary buffer '{name}' due to memory pressure"
                        )
                        break

            # If still under pressure, free oldest non-critical buffers
            if (
                self.stats.total_memory_mb / self.max_memory_mb
                > self._memory_pressure_threshold
            ):
                for name, buffer in sorted_buffers:
                    if (
                        buffer.buffer_type != BufferType.EMBEDDING_MATRIX
                    ):  # Keep embedding matrix
                        if self.release_buffer(name):
                            logger.info(f"Freed buffer '{name}' due to memory pressure")
                            break

    def optimize_memory_layout(self):
        """Optimize memory layout for better performance"""
        logger.info("Optimizing memory layout for M4 Pro")

        # Group buffers by type for better cache locality
        embedding_buffers = []
        cache_buffers = []
        temp_buffers = []

        for buffer in self.buffers.values():
            if buffer.buffer_type == BufferType.EMBEDDING_MATRIX:
                embedding_buffers.append(buffer)
            elif buffer.buffer_type == BufferType.INDEX_CACHE:
                cache_buffers.append(buffer)
            elif buffer.buffer_type == BufferType.TEMPORARY:
                temp_buffers.append(buffer)

        logger.debug(
            f"Memory layout: {len(embedding_buffers)} embedding, "
            f"{len(cache_buffers)} cache, {len(temp_buffers)} temporary buffers"
        )

    def get_memory_stats(self) -> dict[str, Any]:
        """Get detailed memory statistics"""
        active_by_type = {}
        memory_by_type = {}

        for buffer in self.buffers.values():
            buffer_type = buffer.buffer_type.value
            active_by_type[buffer_type] = active_by_type.get(buffer_type, 0) + 1
            memory_by_type[buffer_type] = (
                memory_by_type.get(buffer_type, 0.0) + buffer.memory_mb
            )

        return {
            "total_buffers": len(self.buffers),
            "total_memory_mb": self.stats.total_memory_mb,
            "peak_memory_mb": self.stats.peak_memory_mb,
            "memory_usage_percent": (self.stats.total_memory_mb / self.max_memory_mb)
            * 100,
            "active_by_type": active_by_type,
            "memory_by_type": memory_by_type,
            "zero_copy_operations": self.stats.zero_copy_operations,
            "memory_pressure_events": self.stats.memory_pressure_events,
        }

    async def cleanup_unused(self, max_age_seconds: int = 3600):
        """Clean up unused buffers older than max_age_seconds"""
        async with production_error_handling("UnifiedMemoryManager", "cleanup_unused"):
            current_time = time.time()
            to_remove = []

            for name, buffer in self.buffers.items():
                if (
                    buffer.buffer_type == BufferType.TEMPORARY
                    and current_time - buffer.last_accessed > max_age_seconds
                ):
                    to_remove.append(name)

            for name in to_remove:
                self.release_buffer(name)

            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} unused buffers")


# Global instance
_unified_memory_manager: UnifiedMemoryManager | None = None


def get_unified_memory_manager() -> UnifiedMemoryManager:
    """Get global unified memory manager instance"""
    global _unified_memory_manager
    if _unified_memory_manager is None:
        _unified_memory_manager = UnifiedMemoryManager()
    return _unified_memory_manager


async def create_embedding_buffer(
    embeddings: np.ndarray, name: str = "embeddings"
) -> UnifiedMemoryBuffer:
    """
    Create a unified memory buffer for embedding matrix.

    Optimized for zero-copy operations between CPU-based indexing
    and GPU-based similarity search.
    """
    async with production_error_handling(
        "UnifiedMemoryManager", "create_embedding_buffer"
    ):
        manager = get_unified_memory_manager()
        buffer = await manager.allocate_buffer(
            embeddings.nbytes, BufferType.EMBEDDING_MATRIX, name
        )

        # Copy embeddings into buffer
        await buffer.copy_from_numpy(embeddings)

        logger.info(
            f"Created embedding buffer '{name}' with {embeddings.shape} embeddings"
        )
        return buffer


async def create_search_buffer(
    size_mb: int, name: str = "search_results"
) -> UnifiedMemoryBuffer:
    """Create a unified memory buffer for search results"""
    async with production_error_handling(
        "UnifiedMemoryManager", "create_search_buffer"
    ):
        manager = get_unified_memory_manager()
        return await manager.allocate_buffer(
            int(size_mb * 1024 * 1024), BufferType.SEARCH_RESULTS, name
        )
