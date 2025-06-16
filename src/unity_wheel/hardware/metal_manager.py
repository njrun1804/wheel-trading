"""Unified Metal GPU Manager.

Consolidates all Metal GPU acceleration implementations into a single,
efficient interface for M4 Pro hardware.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None

try:
    import torch

    if torch.backends.mps.is_available():
        MPS_AVAILABLE = True
    else:
        MPS_AVAILABLE = False
except ImportError:
    MPS_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class MetalStats:
    """Metal GPU performance statistics."""

    operations_executed: int = 0
    total_time_ms: float = 0.0
    memory_peak_mb: float = 0.0
    memory_current_mb: float = 0.0
    mlx_operations: int = 0
    mps_operations: int = 0
    fallback_operations: int = 0

    @property
    def average_latency_ms(self) -> float:
        """Calculate average operation latency."""
        if self.operations_executed == 0:
            return 0.0
        return self.total_time_ms / self.operations_executed


class MetalGPUManager:
    """Unified Metal GPU acceleration manager.

    Provides a single interface to Metal GPU acceleration using both
    MLX and MPS backends with intelligent routing and fallback.
    """

    def __init__(self, gpu_cores: int = 20, memory_coordinator=None):
        """Initialize Metal GPU manager."""
        self.gpu_cores = gpu_cores
        self.memory_coordinator = memory_coordinator
        self.stats = MetalStats()

        # Availability flags
        self.mlx_available = MLX_AVAILABLE
        self.mps_available = MPS_AVAILABLE
        self._initialized = False

        # Operation thresholds
        self._operation_thresholds = {
            "matrix_multiply": 50000,  # 250x200 matrix
            "vector_ops": 10000,  # 10K elements
            "similarity_search": 5000,  # 5K vectors
            "attention": 512,  # 512 tokens
        }

        # Memory management
        self._memory_pool_mb = 0
        self._allocated_memory_mb = 0

        logger.info(
            f"MetalGPUManager initialized: MLX={self.mlx_available}, MPS={self.mps_available}"
        )

    async def initialize(self):
        """Initialize Metal GPU resources."""
        if self._initialized:
            return

        start_time = time.time()

        # Allocate memory pool
        if self.memory_coordinator:
            memory_budget = self.memory_coordinator.get_budget("metal")
            self._memory_pool_mb = memory_budget
        else:
            # Default to 8GB for Metal GPU
            self._memory_pool_mb = 8 * 1024

        # Initialize backends
        await self._initialize_mlx()
        await self._initialize_mps()

        # Warmup
        await self._warmup()

        self._initialized = True
        init_time = (time.time() - start_time) * 1000
        logger.info(
            f"Metal GPU initialized in {init_time:.1f}ms (pool: {self._memory_pool_mb}MB)"
        )

    async def _initialize_mlx(self):
        """Initialize MLX backend."""
        if not self.mlx_available:
            return

        try:
            # Set MLX memory limit
            import os

            os.environ["MLX_GPU_MEMORY_LIMIT"] = str(int(self._memory_pool_mb))

            # Test MLX availability
            test_array = mx.array([1.0, 2.0, 3.0])
            mx.eval(test_array)

            logger.info("MLX backend initialized successfully")

        except Exception as e:
            logger.warning(f"MLX initialization failed: {e}")
            self.mlx_available = False

    async def _initialize_mps(self):
        """Initialize MPS backend."""
        if not self.mps_available:
            return

        try:
            # Set MPS memory fraction
            memory_fraction = min(0.8, self._memory_pool_mb / (24 * 1024))
            torch.mps.set_per_process_memory_fraction(memory_fraction)

            # Test MPS availability
            device = torch.device("mps")
            test_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
            torch.mps.synchronize()

            logger.info("MPS backend initialized successfully")

        except Exception as e:
            logger.warning(f"MPS initialization failed: {e}")
            self.mps_available = False

    async def _warmup(self):
        """Warmup Metal GPU with small operations."""
        warmup_ops = []

        # MLX warmup
        if self.mlx_available:
            try:
                a = mx.random.normal((100, 100))
                b = mx.random.normal((100, 100))
                c = a @ b
                mx.eval(c)
                warmup_ops.append("MLX")
            except Exception as e:
                logger.warning(f"MLX warmup failed: {e}")

        # MPS warmup
        if self.mps_available:
            try:
                device = torch.device("mps")
                a = torch.randn(100, 100, device=device)
                b = torch.randn(100, 100, device=device)
                c = torch.mm(a, b)
                torch.mps.synchronize()
                warmup_ops.append("MPS")
            except Exception as e:
                logger.warning(f"MPS warmup failed: {e}")

        if warmup_ops:
            logger.debug(f"Metal GPU warmup completed: {', '.join(warmup_ops)}")

    def is_available(self) -> bool:
        """Check if Metal GPU is available."""
        return self.mlx_available or self.mps_available

    def _select_backend(self, operation_type: str, *args) -> str:
        """Select the best backend for the operation."""
        if not self.is_available():
            return "cpu"

        # Prefer MLX for most operations (faster, more memory efficient)
        if self.mlx_available:
            return "mlx"
        elif self.mps_available:
            return "mps"
        else:
            return "cpu"

    def _should_use_gpu(self, operation_type: str, *args) -> bool:
        """Determine if operation should use GPU."""
        threshold = self._operation_thresholds.get(operation_type, 10000)

        # Estimate workload size
        total_elements = 0
        for arg in args:
            if isinstance(arg, (np.ndarray, list)):
                if hasattr(arg, "size"):
                    total_elements += arg.size
                else:
                    total_elements += len(arg)

        return total_elements >= threshold

    async def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation on Metal GPU."""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        operation_type = kwargs.pop("operation_type", "generic")

        # Check if we should use GPU
        if not self._should_use_gpu(operation_type, *args):
            self.stats.fallback_operations += 1
            return await self._execute_cpu(operation, *args, **kwargs)

        # Select backend
        backend = self._select_backend(operation_type, *args)

        try:
            if backend == "mlx":
                result = await self._execute_mlx(operation, *args, **kwargs)
                self.stats.mlx_operations += 1
            elif backend == "mps":
                result = await self._execute_mps(operation, *args, **kwargs)
                self.stats.mps_operations += 1
            else:
                result = await self._execute_cpu(operation, *args, **kwargs)
                self.stats.fallback_operations += 1

            # Update stats
            execution_time = (time.time() - start_time) * 1000
            self.stats.operations_executed += 1
            self.stats.total_time_ms += execution_time

            return result

        except Exception as e:
            logger.error(f"Metal execution failed: {e}")
            # Fallback to CPU
            self.stats.fallback_operations += 1
            return await self._execute_cpu(operation, *args, **kwargs)

    async def _execute_mlx(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation using MLX."""
        # Convert numpy arrays to MLX arrays
        mlx_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                mlx_args.append(mx.array(arg))
            else:
                mlx_args.append(arg)

        # Execute operation
        if asyncio.iscoroutinefunction(operation):
            result = await operation(*mlx_args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, operation, *mlx_args, **kwargs)

        # Ensure evaluation
        if isinstance(result, mx.array):
            mx.eval(result)

        # Convert back to numpy if needed
        if isinstance(result, mx.array):
            return np.array(result)

        return result

    async def _execute_mps(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation using MPS."""
        device = torch.device("mps")

        # Convert numpy arrays to PyTorch tensors
        torch_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                torch_args.append(torch.from_numpy(arg).to(device))
            else:
                torch_args.append(arg)

        # Execute operation
        if asyncio.iscoroutinefunction(operation):
            result = await operation(*torch_args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, operation, *torch_args, **kwargs)

        # Synchronize
        torch.mps.synchronize()

        # Convert back to numpy if needed
        if isinstance(result, torch.Tensor):
            return result.cpu().numpy()

        return result

    async def _execute_cpu(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation on CPU."""
        if asyncio.iscoroutinefunction(operation):
            return await operation(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, operation, *args, **kwargs)

    # Specific GPU-accelerated operations

    async def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Accelerated matrix multiplication."""
        return await self.execute(
            lambda x, y: x @ y, a, b, operation_type="matrix_multiply"
        )

    async def similarity_search(
        self, query: np.ndarray, database: np.ndarray, top_k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Accelerated similarity search."""

        def _similarity_search(q, db):
            # Normalize vectors
            q_norm = q / np.linalg.norm(q)
            db_norms = db / np.linalg.norm(db, axis=1, keepdims=True)

            # Compute similarities
            similarities = db_norms @ q_norm

            # Find top-k
            indices = np.argsort(similarities)[::-1][:top_k]
            scores = similarities[indices]

            return indices, scores

        return await self.execute(
            _similarity_search, query, database, operation_type="similarity_search"
        )

    async def attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Accelerated attention computation."""

        def _attention(q, k, v, m=None):
            # Scaled dot-product attention
            d_k = q.shape[-1]
            scores = (q @ k.T) / np.sqrt(d_k)

            # Apply mask if provided
            if m is not None:
                scores += m * -1e9

            # Softmax
            attention_weights = self._softmax(scores)

            # Apply to values
            return attention_weights @ v

        return await self.execute(
            _attention, query, key, value, mask, operation_type="attention"
        )

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        x_exp = np.exp(x - x_max)
        return x_exp / np.sum(x_exp, axis=axis, keepdims=True)

    async def batch_operations(
        self, operations: list[tuple[Callable, tuple, dict]]
    ) -> list[Any]:
        """Execute multiple operations in parallel."""
        tasks = []
        for operation, args, kwargs in operations:
            task = asyncio.create_task(self.execute(operation, *args, **kwargs))
            tasks.append(task)

        return await asyncio.gather(*tasks)

    # Memory management

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage."""
        current_mb = 0

        try:
            if self.mlx_available:
                # MLX memory tracking (if available)
                current_mb += self._allocated_memory_mb

            if self.mps_available:
                # PyTorch MPS memory tracking
                current_mb += torch.mps.current_allocated_memory() / (1024 * 1024)
        except:
            pass

        return {
            "allocated_mb": current_mb,
            "pool_mb": self._memory_pool_mb,
            "usage_percent": (current_mb / self._memory_pool_mb) * 100
            if self._memory_pool_mb > 0
            else 0,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        memory_usage = self.get_memory_usage()

        return {
            "available": self.is_available(),
            "backends": {"mlx": self.mlx_available, "mps": self.mps_available},
            "operations": {
                "total": self.stats.operations_executed,
                "mlx": self.stats.mlx_operations,
                "mps": self.stats.mps_operations,
                "fallback": self.stats.fallback_operations,
                "average_latency_ms": self.stats.average_latency_ms,
            },
            "memory": memory_usage,
            "thresholds": self._operation_thresholds,
        }

    def tune_thresholds(self):
        """Auto-tune operation thresholds based on performance."""
        if self.stats.operations_executed < 50:
            return  # Not enough data

        avg_latency = self.stats.average_latency_ms

        # Adjust thresholds based on performance
        if avg_latency > 20:  # Too slow, increase thresholds
            for op_type in self._operation_thresholds:
                self._operation_thresholds[op_type] = int(
                    self._operation_thresholds[op_type] * 1.2
                )
        elif avg_latency < 5:  # Very fast, decrease thresholds
            for op_type in self._operation_thresholds:
                self._operation_thresholds[op_type] = int(
                    self._operation_thresholds[op_type] * 0.8
                )

        logger.debug(f"Tuned Metal thresholds: {self._operation_thresholds}")

    def shutdown(self):
        """Shutdown Metal GPU manager."""
        logger.info("Shutting down Metal GPU manager")

        # Clear any GPU memory
        try:
            if self.mps_available:
                torch.mps.empty_cache()
        except:
            pass

        self._initialized = False


# Example usage and testing
if __name__ == "__main__":

    async def demo():
        """Demonstrate Metal GPU acceleration."""
        print("=== Metal GPU Manager Demo ===")

        # Initialize
        metal = MetalGPUManager()
        await metal.initialize()

        print(f"Metal available: {metal.is_available()}")
        print(f"MLX: {metal.mlx_available}, MPS: {metal.mps_available}")

        if not metal.is_available():
            print("No Metal acceleration available")
            return

        # Test matrix multiplication
        print("\n--- Matrix Multiplication Test ---")
        a = np.random.randn(1000, 1000).astype(np.float32)
        b = np.random.randn(1000, 1000).astype(np.float32)

        start = time.time()
        result = await metal.matrix_multiply(a, b)
        metal_time = (time.time() - start) * 1000

        # CPU baseline
        start = time.time()
        cpu_result = a @ b
        cpu_time = (time.time() - start) * 1000

        print(f"Metal time: {metal_time:.1f}ms")
        print(f"CPU time: {cpu_time:.1f}ms")
        print(f"Speedup: {cpu_time/metal_time:.1f}x")
        print(f"Result shape: {result.shape}")

        # Test similarity search
        print("\n--- Similarity Search Test ---")
        query = np.random.randn(768).astype(np.float32)
        database = np.random.randn(10000, 768).astype(np.float32)

        start = time.time()
        indices, scores = await metal.similarity_search(query, database, top_k=10)
        search_time = (time.time() - start) * 1000

        print(f"Search time: {search_time:.1f}ms")
        print(f"Top-k indices: {indices[:5]}")
        print(f"Top-k scores: {scores[:5]}")

        # Show statistics
        print("\n--- Performance Statistics ---")
        stats = metal.get_stats()
        for category, data in stats.items():
            print(f"\n{category.upper()}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {data}")

        # Shutdown
        metal.shutdown()

    # Run demo
    asyncio.run(demo())
