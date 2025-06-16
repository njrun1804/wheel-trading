"""Device routing for optimal performance on M4 Pro.

Automatically routes operations to MLX (GPU), PyTorch (MPS/CPU), or CPU
based on operation type and hardware capabilities.
"""
import logging
import os
import platform
import subprocess
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


def _detect_m4_pro() -> bool:
    """Detect if running on M4 Pro using sysctl."""
    if platform.machine() != "arm64":
        return False
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        cpu_brand = result.stdout.strip()
        return "Apple M4" in cpu_brand
    except (subprocess.SubprocessError, FileNotFoundError):
        return "Apple M4" in platform.platform()


IS_M4PRO = _detect_m4_pro()
if IS_M4PRO:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_METAL_WORKSPACE_LIMIT_BYTES"] = str(18 * 1024 * 1024 * 1024)


class Backend(Enum):
    """Available compute backends."""

    MLX_GPU = "mlx_gpu"
    TORCH_MPS = "torch_mps"
    TORCH_CPU = "torch_cpu"
    CPU_NATIVE = "cpu_native"


class OperationType(Enum):
    """Types of operations for routing."""

    NEURAL_FORWARD = "neural_forward"
    NEURAL_BACKWARD = "neural_backward"
    MATRIX_MULTIPLY = "matrix_multiply"
    CONVOLUTION = "convolution"
    GROUPED_CONV = "grouped_conv"
    TRANSFORMER = "transformer"
    EMBEDDING = "embedding"
    TREE_SEARCH = "tree_search"
    VECTOR_SEARCH = "vector_search"


class DeviceRouter:
    """Routes operations to optimal backend on M4 Pro."""

    def __init__(self):
        self.is_m4pro = IS_M4PRO
        self._backends = self._detect_backends()
        self._routing_table = self._build_routing_table()
        logger.info(
            f"Device Router initialized on {'M4 Pro' if self.is_m4pro else platform.machine()}"
        )
        logger.info(f"Available backends: {list(self._backends.keys())}")

    def _detect_backends(self) -> dict[Backend, bool]:
        """Detect available backends."""
        backends = {}
        try:
            import mlx.core as mx

            backends[Backend.MLX_GPU] = mx.metal.is_available()
            if backends[Backend.MLX_GPU]:
                os.environ["MLX_FORCE_CPU"] = "0"
            logger.info("MLX GPU available")
        except ImportError:
            backends[Backend.MLX_GPU] = False
            logger.warning("MLX not installed")
        try:
            import torch

            backends[Backend.TORCH_MPS] = torch.backends.mps.is_available()
            backends[Backend.TORCH_CPU] = True
            if backends[Backend.TORCH_MPS]:
                try:
                    test = torch.randn(10, 10, device="mps")
                    _ = test @ test.T
                    logger.info("PyTorch MPS available and working")
                except Exception:
                    backends[Backend.TORCH_MPS] = False
                    logger.warning("PyTorch MPS available but not working")
        except ImportError:
            backends[Backend.TORCH_MPS] = False
            backends[Backend.TORCH_CPU] = False
            logger.warning("PyTorch not installed")
        backends[Backend.CPU_NATIVE] = True
        return backends

    def _build_routing_table(self) -> dict[OperationType, Backend]:
        """Build optimal routing for each operation type."""
        routing = {}
        for op in [
            OperationType.NEURAL_FORWARD,
            OperationType.MATRIX_MULTIPLY,
            OperationType.EMBEDDING,
        ]:
            if self._backends.get(Backend.MLX_GPU):
                routing[op] = Backend.MLX_GPU
            elif self._backends.get(Backend.TORCH_MPS):
                routing[op] = Backend.TORCH_MPS
            else:
                routing[op] = Backend.TORCH_CPU
        if self._backends.get(Backend.MLX_GPU):
            routing[OperationType.CONVOLUTION] = Backend.MLX_GPU
        else:
            routing[OperationType.CONVOLUTION] = Backend.TORCH_CPU
        routing[OperationType.GROUPED_CONV] = Backend.TORCH_CPU
        if self._backends.get(Backend.TORCH_MPS):
            routing[OperationType.TRANSFORMER] = Backend.TORCH_MPS
        else:
            routing[OperationType.TRANSFORMER] = Backend.TORCH_CPU
        routing[OperationType.TREE_SEARCH] = Backend.CPU_NATIVE
        routing[OperationType.VECTOR_SEARCH] = Backend.CPU_NATIVE
        routing[OperationType.NEURAL_BACKWARD] = Backend.TORCH_CPU
        return routing

    def route(self, operation: OperationType) -> Backend:
        """Get optimal backend for operation."""
        return self._routing_table.get(operation, Backend.CPU_NATIVE)

    def get_device(self, backend: Backend) -> Any:
        """Get device object for backend."""
        if backend == Backend.MLX_GPU:
            import mlx.core as mx

            return mx.gpu
        elif backend == Backend.TORCH_MPS:
            import torch

            return torch.device("mps")
        elif backend == Backend.TORCH_CPU:
            import torch

            return torch.device("cpu")
        else:
            return None

    def benchmark_operation(
        self, operation: OperationType, input_size: tuple, iterations: int = 100
    ) -> dict[Backend, float]:
        """Benchmark operation on available backends."""
        import time

        results = {}
        for backend in self._backends:
            if not self._backends[backend]:
                continue
            try:
                if backend == Backend.MLX_GPU:
                    import mlx.core as mx

                    data = mx.random.normal(input_size)
                    start = time.perf_counter()
                    for _ in range(iterations):
                        _ = data @ data.T
                        mx.eval(data)
                    elapsed = time.perf_counter() - start
                elif backend in [Backend.TORCH_MPS, Backend.TORCH_CPU]:
                    import torch

                    device = self.get_device(backend)
                    data = torch.randn(input_size, device=device)
                    start = time.perf_counter()
                    for _ in range(iterations):
                        _ = data @ data.T
                    if backend == Backend.TORCH_MPS:
                        torch.mps.synchronize()
                    elapsed = time.perf_counter() - start
                else:
                    continue
                results[backend] = elapsed / iterations
                logger.info(f"{backend.value}: {results[backend] * 1000:.2f}ms per op")
            except Exception as e:
                logger.error(f"Benchmark failed for {backend}: {e}")
        return results

    def select_optimal_backend(
        self, operation: OperationType, input_size: tuple
    ) -> Backend:
        """Benchmark and select optimal backend for specific operation."""
        results = self.benchmark_operation(operation, input_size)
        if results:
            optimal = min(results.items(), key=lambda x: x[1])[0]
            logger.info(f"Optimal backend for {operation.value}: {optimal.value}")
            self._routing_table[operation] = optimal
            return optimal
        return self.route(operation)


_router = None


def get_router() -> DeviceRouter:
    """Get global device router instance."""
    global _router
    if _router is None:
        _router = DeviceRouter()
    return _router
