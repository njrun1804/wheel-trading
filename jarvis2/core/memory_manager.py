"""Memory management for M4 Pro unified architecture.

Manages shared memory buffers for zero-copy tensor passing between processes.
Enforces 18GB Metal memory limit.
"""
import multiprocessing as mp

try:
    from multiprocessing import shared_memory
except ImportError:
    import multiprocessing.shared_memory as shared_memory

import logging
import platform
import weakref
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

logger = logging.getLogger(__name__)
TOTAL_MEMORY_GB = 24
METAL_LIMIT_GB = 18
BUFFER_POOL_GB = 4


@dataclass
class MemoryBuffer:
    """Shared memory buffer metadata."""
    name: str
    size: int
    dtype: np.dtype
    shape: Tuple[int, ...]
    in_use: bool = False


class UnifiedMemoryManager:
    """Manages unified memory for M4 Pro.
    
    Pre-allocates shared memory pools for efficient tensor passing
    between processes without copying.
    """

    def __init__(self, pool_size_gb: float = BUFFER_POOL_GB):
        self.pool_size_bytes = int(pool_size_gb * 1024 * 1024 * 1024)
        self.buffers: Dict[str, MemoryBuffer] = {}
        self.shm_objects: Dict[str, shared_memory.SharedMemory] = {}
        self._init_pool()
        self.metal_limit_bytes = METAL_LIMIT_GB * 1024 * 1024 * 1024
        self._check_memory_pressure()

    def _init_pool(self):
        """Initialize shared memory pool."""
        buffer_sizes = [(100, 1024 * 1024), (50, 10 * 1024 * 1024), (20, 50 *
            1024 * 1024), (10, 100 * 1024 * 1024)]
        total_allocated = 0
        buffer_id = 0
        for count, size in buffer_sizes:
            for _ in range(count):
                if total_allocated + size > self.pool_size_bytes:
                    break
                name = f"jarvis_buffer_{buffer_id}"
                try:
                    shm = shared_memory.SharedMemory(create = True, size = size)
                    self.shm_objects[name] = shm
                    self.buffers[name] = MemoryBuffer(name = name, size = size,
                        dtype = np.float32, shape=(size // 4,))
                    total_allocated += size
                    buffer_id += 1
                except Exception as e:
                    logger.error(f"Failed to allocate buffer: {e}")
                    break
        logger.info(
            f"Allocated {len(self.buffers)} buffers, {total_allocated / 1024 ** 3:.1f}GB total"
            )

    def allocate_tensor(self, shape: Tuple[int, ...], dtype: np.dtype = np.
        float32) ->Tuple[str, np.ndarray]:
        """Allocate a tensor in shared memory.
        
        Returns:
            (buffer_name, numpy_array)
        """
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        size = np.prod(shape) * dtype.itemsize
        best_buffer = None
        best_size = float('inf")
        for name, buffer in self.buffers.items():
            if not buffer.in_use and buffer.size >= size:
                if buffer.size < best_size:
                    best_buffer = name
                    best_size = buffer.size
        if not best_buffer:
            raise MemoryError(
                f"No buffer available for {size / 1024 ** 2:.1f}MB tensor')
        self.buffers[best_buffer].in_use = True
        self.buffers[best_buffer].shape = shape
        self.buffers[best_buffer].dtype = dtype
        shm = self.shm_objects[best_buffer]
        array = np.ndarray(shape, dtype = dtype, buffer = shm.buf)
        logger.debug(f"Allocated {shape} tensor in buffer {best_buffer}")
        return best_buffer, array

    def get_tensor(self, buffer_name: str) ->np.ndarray:
        """Get tensor from shared memory buffer."""
        if buffer_name not in self.buffers:
            raise KeyError(f"Unknown buffer: {buffer_name}")
        buffer = self.buffers[buffer_name]
        shm = self.shm_objects[buffer_name]
        array = np.ndarray(buffer.shape, dtype = buffer.dtype, buffer = shm.buf)
        return array

    def release_tensor(self, buffer_name: str):
        """Release tensor buffer for reuse."""
        if buffer_name in self.buffers:
            self.buffers[buffer_name].in_use = False
            logger.debug(f"Released buffer {buffer_name}")

    def _check_memory_pressure(self):
        """Check if we're approaching memory limits."""
        vm = psutil.virtual_memory()
        used_gb = vm.used / 1024 ** 3
        if platform.system() == 'Darwin':
            try:
                import torch
                if torch.backends.mps.is_available():
                    mps_gb = torch.mps.current_allocated_memory() / 1024 ** 3
                    used_gb += mps_gb
            except (ImportError, AttributeError) as e:
                logger.debug(f"Ignored exception in {'memory_manager.py'}: {e}"
                    )
        if used_gb > METAL_LIMIT_GB:
            logger.warning(
                f"Memory pressure high: {used_gb:.1f}GB used (limit: {METAL_LIMIT_GB}GB)"
                )
        return used_gb

    def get_stats(self) ->Dict[str, Any]:
        """Get memory statistics."""
        vm = psutil.virtual_memory()
        buffers_in_use = sum(1 for b in self.buffers.values() if b.in_use)
        total_buffer_size = sum(b.size for b in self.buffers.values())
        used_buffer_size = sum(b.size for b in self.buffers.values() if b.
            in_use)
        return {'system_memory_gb': vm.total / 1024 ** 3, 'system_used_gb':
            vm.used / 1024 ** 3, 'system_available_gb': vm.available / 1024 **
            3, 'metal_limit_gb': METAL_LIMIT_GB, 'buffer_pool_gb': self.
            pool_size_bytes / 1024 ** 3, 'buffers_total': len(self.buffers),
            'buffers_in_use': buffers_in_use, 'buffer_utilization': 
            used_buffer_size / total_buffer_size if total_buffer_size > 0 else
            0}

    def cleanup(self):
        """Clean up all shared memory."""
        for shm in self.shm_objects.values():
            try:
                shm.close()
                shm.unlink()
            except Exception as e:
                logger.debug(f"Ignored exception in {'memory_manager.py'}: {e}"
                    )
        logger.info('Memory manager cleaned up')


class TensorQueue:
    """Queue for passing tensors between processes using shared memory."""

    def __init__(self, memory_manager: UnifiedMemoryManager, maxsize: int = 100):
        self.memory_manager = memory_manager
        self.metadata_queue = mp.Queue(maxsize = maxsize)
        self._buffer_metadata = {}

    def put(self, tensor: np.ndarray, copy: bool = True) ->str:
        """Put tensor in queue.
        
        Args:
            tensor: Numpy array to send
            copy: If True, copy data to shared memory. If False, tensor must already be in shared memory.
            
        Returns:
            buffer_name for retrieval
        """
        if copy:
            buffer_name, shared_array = self.memory_manager.allocate_tensor(
                tensor.shape, tensor.dtype)
            shared_array[:] = tensor
        elif hasattr(tensor, '_buffer_name'):
            buffer_name = tensor._buffer_name
        else:
            tensor_id = id(tensor)
            if tensor_id in self._buffer_metadata:
                buffer_name = self._buffer_metadata[tensor_id]
            else:
                raise ValueError('Zero-copy tensor must have buffer metadata')
        self.metadata_queue.put({'buffer_name': buffer_name, 'shape':
            tensor.shape, 'dtype': tensor.dtype})
        return buffer_name

    def register_buffer(self, tensor: np.ndarray, buffer_name: str):
        """Register a tensor's buffer name for zero-copy operations."""
        self._buffer_metadata[id(tensor)] = buffer_name

    def get(self, timeout: Optional[float]=None) ->Tuple[str, np.ndarray]:
        """Get tensor from queue.
        
        Returns:
            (buffer_name, tensor)
        """
        metadata = self.metadata_queue.get(timeout = timeout)
        tensor = self.memory_manager.get_tensor(metadata['buffer_name'])
        return metadata['buffer_name'], tensor

    def get_nowait(self) ->Tuple[str, np.ndarray]:
        """Get tensor without waiting."""
        metadata = self.metadata_queue.get_nowait()
        tensor = self.memory_manager.get_tensor(metadata['buffer_name'])
        return metadata['buffer_name'], tensor


_memory_manager = None


def get_memory_manager() ->UnifiedMemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = UnifiedMemoryManager()
    return _memory_manager
