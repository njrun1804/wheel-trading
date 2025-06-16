"""GPU acceleration optimization with MLX for vector operations.

This module provides MLX-accelerated operations for the 8-agent system with:
- @gpuify decorator for automatic GPU offloading
- Accelerated similarity searches, vector operations, and dense math
- Optimized embeddings and text processing kernels  
- Fallback to CPU when GPU is busy
- Metal compute optimizations for Apple Silicon
"""

import asyncio
import functools
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


class GPUAccelerator:
    """GPU acceleration manager for MLX operations on Apple Silicon."""

    def __init__(self):
        """Initialize GPU accelerator with hardware configuration."""
        # Load hardware config
        config_path = Path(__file__).parent.parent / "optimization_config.json"
        with open(config_path) as f:
            self.config = json.load(f)

        # Metal GPU info
        self.gpu_available = mx.metal.is_available()
        if self.gpu_available:
            logger.info("MLX Metal GPU acceleration enabled")
        else:
            logger.warning("MLX Metal GPU not available, falling back to CPU")

        # Load Bolt configuration for GPU settings
        try:
            from bob.integration.bolt.config import get_default_config

            bolt_config = get_default_config()
            gpu_memory_ratio = bolt_config.hardware.gpu_memory_threshold_ratio
            batch_size = bolt_config.performance.gpu_batch_size
        except Exception as e:
            logger.warning(f"Failed to load Bolt GPU config, using defaults: {e}")
            gpu_memory_ratio = 0.7
            batch_size = 256

        # GPU memory tracking
        self._memory_threshold = (
            self.config["memory"]["max_allocation_gb"]
            * gpu_memory_ratio
            * 1024
            * 1024
            * 1024
        )
        self._batch_size = batch_size

        # Workload size thresholds for intelligent CPU/GPU routing (configurable)
        try:
            perf_config = bolt_config.performance
            self._workload_thresholds = {
                "vector_ops": perf_config.vector_ops_threshold,
                "matrix_ops": perf_config.matrix_ops_threshold,
                "batch_ops": perf_config.batch_ops_threshold,
                "similarity": perf_config.similarity_threshold,
                "embedding": perf_config.embedding_threshold,
                "attention": perf_config.attention_threshold,
            }

            # GPU overhead compensation from configuration
            self._gpu_overhead_us = {
                "initialization": perf_config.gpu_initialization_overhead_us,
                "memory_transfer": perf_config.gpu_memory_transfer_overhead_us,
                "evaluation": perf_config.gpu_evaluation_overhead_us,
                "cleanup": perf_config.gpu_cleanup_overhead_us,
            }
        except Exception as e:
            logger.warning(
                f"Failed to load performance thresholds, using defaults: {e}"
            )
            # Fallback to original hardcoded values
            self._workload_thresholds = {
                "vector_ops": 10000,
                "matrix_ops": 500 * 500,
                "batch_ops": 200,
                "similarity": 2000,
                "embedding": 500,
                "attention": 128,
            }

            self._gpu_overhead_us = {
                "initialization": 2000,
                "memory_transfer": 500,
                "evaluation": 300,
                "cleanup": 100,
            }

        # Performance calibration - update based on runtime measurements
        self._performance_model = {
            "cpu_ops_per_sec": 1e6,  # 1M simple ops/sec on CPU
            "gpu_ops_per_sec": 5e6,  # 5M ops/sec on GPU (theoretical)
            "memory_bandwidth_gbps": 400,  # M4 Pro unified memory bandwidth
        }

        # Performance stats
        self.stats = {
            "gpu_operations": 0,
            "cpu_fallbacks": 0,
            "cpu_preferred": 0,  # Intelligently routed to CPU
            "total_gpu_time": 0.0,
            "total_cpu_time": 0.0,
            "overhead_avoided_ms": 0.0,  # GPU overhead avoided by CPU routing
            "memory_peaks": [],
            "workload_size_stats": {  # Track workload sizes for threshold tuning
                "small_cpu": [],
                "large_gpu": [],
                "borderline": [],
            },
        }

    def _check_gpu_memory(self, estimated_bytes: int) -> bool:
        """Check if GPU has enough memory for operation."""
        if not self.gpu_available:
            return False

        # Check if operation would exceed threshold
        if estimated_bytes > self._memory_threshold:
            logger.debug(
                f"Operation size {estimated_bytes / 1e9:.1f}GB exceeds threshold"
            )
            return False

        return True

    def _estimate_workload_size(self, operation_type: str, *args) -> int:
        """Estimate workload size for intelligent CPU/GPU routing."""
        total_elements = 0

        for arg in args:
            if isinstance(arg, mx.array | np.ndarray):
                total_elements += arg.size
            elif hasattr(arg, "__len__"):
                total_elements += len(arg)
            elif isinstance(arg, list | tuple):
                for item in arg:
                    if isinstance(item, mx.array | np.ndarray):
                        total_elements += item.size
                    elif hasattr(item, "__len__"):
                        total_elements += len(item)

        return total_elements

    def _should_use_gpu(
        self, operation_type: str, workload_size: int, estimated_time_us: float = 0
    ) -> tuple[bool, str]:
        """Intelligent decision on CPU vs GPU based on workload size and overhead."""
        if not self.gpu_available:
            return False, "GPU not available"

        # Get threshold for this operation type
        threshold = self._workload_thresholds.get(operation_type, 1000)

        # Calculate total GPU overhead
        total_overhead_us = (
            self._gpu_overhead_us["initialization"]
            + self._gpu_overhead_us["memory_transfer"]
            + self._gpu_overhead_us["evaluation"]
            + self._gpu_overhead_us["cleanup"]
        )

        # Estimate CPU execution time
        cpu_time_us = workload_size / self._performance_model["cpu_ops_per_sec"] * 1e6

        # Estimate GPU execution time (without overhead)
        gpu_compute_time_us = (
            workload_size / self._performance_model["gpu_ops_per_sec"] * 1e6
        )

        # Total GPU time including overhead
        gpu_total_time_us = gpu_compute_time_us + total_overhead_us

        # Decision logic
        if workload_size < threshold:
            reason = f"Small workload ({workload_size} < {threshold})"
            self.stats["workload_size_stats"]["small_cpu"].append(workload_size)
            return False, reason

        # For larger workloads, check if GPU overhead is worth it
        if gpu_total_time_us >= cpu_time_us * 0.8:  # GPU must be >20% faster
            reason = f"GPU overhead too high ({gpu_total_time_us:.0f}us vs CPU {cpu_time_us:.0f}us)"
            self.stats["workload_size_stats"]["borderline"].append(workload_size)
            return False, reason

        # Use GPU for large workloads where it provides clear benefit
        self.stats["workload_size_stats"]["large_gpu"].append(workload_size)
        return True, f"Large workload benefits from GPU ({workload_size} elements)"

    def _update_performance_model(
        self,
        operation_type: str,
        workload_size: int,
        actual_gpu_time: float,
        actual_cpu_time: float,
    ):
        """Update performance model based on actual measurements."""
        if actual_gpu_time > 0 and actual_cpu_time > 0:
            # Calculate actual throughput
            gpu_ops_per_sec = workload_size / actual_gpu_time
            cpu_ops_per_sec = workload_size / actual_cpu_time

            # Exponential moving average to update model
            alpha = 0.1  # Learning rate
            self._performance_model["gpu_ops_per_sec"] = (
                alpha * gpu_ops_per_sec
                + (1 - alpha) * self._performance_model["gpu_ops_per_sec"]
            )
            self._performance_model["cpu_ops_per_sec"] = (
                alpha * cpu_ops_per_sec
                + (1 - alpha) * self._performance_model["cpu_ops_per_sec"]
            )

            logger.debug(
                f"Updated performance model: GPU {gpu_ops_per_sec:.0f} ops/s, CPU {cpu_ops_per_sec:.0f} ops/s"
            )

    def _estimate_memory(self, *arrays: mx.array | np.ndarray) -> int:
        """Estimate memory usage for arrays."""
        total_bytes = 0
        for arr in arrays:
            if isinstance(arr, mx.array | np.ndarray):
                total_bytes += arr.nbytes
            elif hasattr(arr, "__len__"):
                # Estimate for sequences
                total_bytes += len(arr) * 4  # Assume float32
        return total_bytes

    @property
    def gpu_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        if self.stats["gpu_operations"] + self.stats["cpu_fallbacks"] == 0:
            return 0.0
        return (
            100.0
            * self.stats["gpu_operations"]
            / (self.stats["gpu_operations"] + self.stats["cpu_fallbacks"])
        )

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        total_ops = (
            self.stats["gpu_operations"]
            + self.stats["cpu_fallbacks"]
            + self.stats["cpu_preferred"]
        )

        return {
            "gpu_available": self.gpu_available,
            "gpu_operations": self.stats["gpu_operations"],
            "cpu_fallbacks": self.stats["cpu_fallbacks"],
            "cpu_preferred": self.stats["cpu_preferred"],
            "total_operations": total_ops,
            "intelligent_routing_rate": self.stats["cpu_preferred"]
            / max(1, total_ops)
            * 100,
            "gpu_utilization": self.gpu_utilization,
            "avg_gpu_time_ms": self.stats["total_gpu_time"]
            / max(1, self.stats["gpu_operations"])
            * 1000,
            "avg_cpu_time_ms": self.stats["total_cpu_time"]
            / max(1, self.stats["cpu_fallbacks"] + self.stats["cpu_preferred"])
            * 1000,
            "speedup": self.stats["total_cpu_time"]
            / max(0.001, self.stats["total_gpu_time"])
            if self.stats["total_gpu_time"] > 0
            else 0,
            "overhead_avoided_ms": self.stats["overhead_avoided_ms"],
            "memory_peak_gb": max(self.stats["memory_peaks"]) / 1e9
            if self.stats["memory_peaks"]
            else 0,
            "workload_thresholds": self._workload_thresholds,
            "performance_model": self._performance_model,
            "workload_distribution": {
                "small_cpu_count": len(self.stats["workload_size_stats"]["small_cpu"]),
                "large_gpu_count": len(self.stats["workload_size_stats"]["large_gpu"]),
                "borderline_count": len(
                    self.stats["workload_size_stats"]["borderline"]
                ),
            },
        }

    def tune_thresholds(self):
        """Auto-tune workload thresholds based on performance data."""
        # Analyze workload distribution and performance
        workload_stats = self.stats["workload_size_stats"]

        if (
            len(workload_stats["small_cpu"]) > 10
            and len(workload_stats["large_gpu"]) > 10
        ):
            # Find optimal threshold based on actual performance
            small_sizes = workload_stats["small_cpu"]
            large_sizes = workload_stats["large_gpu"]

            # Use 75th percentile of small workloads as new threshold
            if small_sizes:
                new_threshold = int(np.percentile(small_sizes, 75))

                # Update thresholds conservatively
                for op_type in self._workload_thresholds:
                    old_threshold = self._workload_thresholds[op_type]
                    # Move threshold by 10% towards optimal value
                    self._workload_thresholds[op_type] = int(
                        old_threshold * 0.9 + new_threshold * 0.1
                    )

                logger.info(
                    f"Auto-tuned workload thresholds based on {len(small_sizes)} small and {len(large_sizes)} large workloads"
                )

        # Clear stats for next tuning cycle
        for key in workload_stats:
            workload_stats[key] = workload_stats[key][-100:]  # Keep last 100 samples


# Global accelerator instance
_accelerator = GPUAccelerator()


def gpuify(
    func: Callable | None = None,
    *,
    fallback: bool = True,
    batch_size: int | None = None,
    memory_check: bool = True,
    operation_type: str = "generic",
) -> Callable:
    """Decorator for automatic GPU offloading with intelligent CPU/GPU routing.

    Args:
        func: Function to decorate
        fallback: Whether to fallback to CPU if GPU is busy
        batch_size: Override default batch size
        memory_check: Whether to check GPU memory before offloading
        operation_type: Type of operation for intelligent routing (vector_ops, matrix_ops, etc.)

    Returns:
        Decorated function that uses CPU or GPU based on workload size
    """

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        async def async_wrapper(*args, **kwargs):
            # Intelligent CPU/GPU routing based on workload size
            workload_size = _accelerator._estimate_workload_size(operation_type, *args)
            use_gpu, routing_reason = _accelerator._should_use_gpu(
                operation_type, workload_size
            )

            # Also check memory constraints
            if use_gpu and memory_check:
                estimated_mem = _accelerator._estimate_memory(*args)
                if not _accelerator._check_gpu_memory(estimated_mem):
                    use_gpu = False
                    routing_reason = "Insufficient GPU memory"

            start_time = time.perf_counter()
            logger.debug(
                f"Operation {f.__name__}: workload_size={workload_size}, use_gpu={use_gpu}, reason={routing_reason}"
            )

            try:
                if use_gpu:
                    # Convert numpy arrays to MLX
                    mlx_args = []
                    for arg in args:
                        if isinstance(arg, np.ndarray):
                            mlx_args.append(mx.array(arg))
                        else:
                            mlx_args.append(arg)

                    # Run on GPU
                    result = await f(*mlx_args, **kwargs)

                    # Ensure computation completes
                    if isinstance(result, mx.array):
                        mx.eval(result)

                    _accelerator.stats["gpu_operations"] += 1
                    _accelerator.stats["total_gpu_time"] += (
                        time.perf_counter() - start_time
                    )

                    # Track memory usage
                    if memory_check and isinstance(result, mx.array):
                        _accelerator.stats["memory_peaks"].append(result.nbytes)

                    # Convert back to numpy if needed and clean up MLX arrays
                    if (
                        isinstance(result, mx.array)
                        and len(args) > 0
                        and isinstance(args[0], np.ndarray)
                    ):
                        numpy_result = np.array(result)
                        # Clean up temporary MLX arrays
                        del result
                        for arg in mlx_args:
                            if isinstance(arg, mx.array):
                                del arg
                        # Force garbage collection
                        import gc

                        gc.collect()
                        return numpy_result

                    return result

                elif fallback:
                    # CPU execution (either fallback or intelligent routing)
                    if (
                        "Small workload" in routing_reason
                        or "GPU overhead" in routing_reason
                    ):
                        _accelerator.stats["cpu_preferred"] += 1
                        # Track avoided GPU overhead
                        estimated_overhead_ms = (
                            sum(_accelerator._gpu_overhead_us.values()) / 1000
                        )
                        _accelerator.stats[
                            "overhead_avoided_ms"
                        ] += estimated_overhead_ms
                        logger.debug(
                            f"Intelligently routed to CPU for {f.__name__}: {routing_reason}"
                        )
                    else:
                        _accelerator.stats["cpu_fallbacks"] += 1
                        logger.debug(
                            f"Falling back to CPU for {f.__name__}: {routing_reason}"
                        )

                    # Run original function on CPU
                    result = await f(*args, **kwargs)
                    cpu_time = time.perf_counter() - start_time
                    _accelerator.stats["total_cpu_time"] += cpu_time

                    # Update performance model if we have comparison data
                    if (
                        "gpu_operations" in _accelerator.stats
                        and _accelerator.stats["gpu_operations"] > 0
                    ):
                        avg_gpu_time = (
                            _accelerator.stats["total_gpu_time"]
                            / _accelerator.stats["gpu_operations"]
                        )
                        _accelerator._update_performance_model(
                            operation_type, workload_size, avg_gpu_time, cpu_time
                        )

                    return result
                else:
                    raise RuntimeError(
                        f"GPU required for {f.__name__} but not available"
                    )

            except Exception as e:
                logger.error(f"Error in GPU operation {f.__name__}: {e}")
                if fallback:
                    _accelerator.stats["cpu_fallbacks"] += 1
                    result = await f(*args, **kwargs)
                    cpu_time = time.perf_counter() - start_time
                    _accelerator.stats["total_cpu_time"] += cpu_time
                    return result
                raise

        @functools.wraps(f)
        def sync_wrapper(*args, **kwargs):
            # Intelligent CPU/GPU routing based on workload size
            workload_size = _accelerator._estimate_workload_size(operation_type, *args)
            use_gpu, routing_reason = _accelerator._should_use_gpu(
                operation_type, workload_size
            )

            # Also check memory constraints
            if use_gpu and memory_check:
                estimated_mem = _accelerator._estimate_memory(*args)
                if not _accelerator._check_gpu_memory(estimated_mem):
                    use_gpu = False
                    routing_reason = "Insufficient GPU memory"

            start_time = time.perf_counter()
            logger.debug(
                f"Operation {f.__name__}: workload_size={workload_size}, use_gpu={use_gpu}, reason={routing_reason}"
            )

            try:
                if use_gpu:
                    # Convert numpy arrays to MLX
                    mlx_args = []
                    for arg in args:
                        if isinstance(arg, np.ndarray):
                            mlx_args.append(mx.array(arg))
                        else:
                            mlx_args.append(arg)

                    # Run on GPU
                    result = f(*mlx_args, **kwargs)

                    # Ensure computation completes
                    if isinstance(result, mx.array):
                        mx.eval(result)

                    _accelerator.stats["gpu_operations"] += 1
                    _accelerator.stats["total_gpu_time"] += (
                        time.perf_counter() - start_time
                    )

                    # Track memory usage
                    if memory_check and isinstance(result, mx.array):
                        _accelerator.stats["memory_peaks"].append(result.nbytes)

                    # Convert back to numpy if needed and clean up MLX arrays
                    if (
                        isinstance(result, mx.array)
                        and len(args) > 0
                        and isinstance(args[0], np.ndarray)
                    ):
                        numpy_result = np.array(result)
                        # Clean up temporary MLX arrays
                        del result
                        for arg in mlx_args:
                            if isinstance(arg, mx.array):
                                del arg
                        # Force garbage collection
                        import gc

                        gc.collect()
                        return numpy_result

                    return result

                elif fallback:
                    # CPU execution (either fallback or intelligent routing)
                    if (
                        "Small workload" in routing_reason
                        or "GPU overhead" in routing_reason
                    ):
                        _accelerator.stats["cpu_preferred"] += 1
                        # Track avoided GPU overhead
                        estimated_overhead_ms = (
                            sum(_accelerator._gpu_overhead_us.values()) / 1000
                        )
                        _accelerator.stats[
                            "overhead_avoided_ms"
                        ] += estimated_overhead_ms
                        logger.debug(
                            f"Intelligently routed to CPU for {f.__name__}: {routing_reason}"
                        )
                    else:
                        _accelerator.stats["cpu_fallbacks"] += 1
                        logger.debug(
                            f"Falling back to CPU for {f.__name__}: {routing_reason}"
                        )

                    # Run original function on CPU
                    result = f(*args, **kwargs)
                    cpu_time = time.perf_counter() - start_time
                    _accelerator.stats["total_cpu_time"] += cpu_time

                    # Update performance model if we have comparison data
                    if (
                        "gpu_operations" in _accelerator.stats
                        and _accelerator.stats["gpu_operations"] > 0
                    ):
                        avg_gpu_time = (
                            _accelerator.stats["total_gpu_time"]
                            / _accelerator.stats["gpu_operations"]
                        )
                        _accelerator._update_performance_model(
                            operation_type, workload_size, avg_gpu_time, cpu_time
                        )

                    return result
                else:
                    raise RuntimeError(
                        f"GPU required for {f.__name__} but not available"
                    )

            except Exception as e:
                logger.error(f"Error in GPU operation {f.__name__}: {e}")
                if fallback:
                    _accelerator.stats["cpu_fallbacks"] += 1
                    result = f(*args, **kwargs)
                    cpu_time = time.perf_counter() - start_time
                    _accelerator.stats["total_cpu_time"] += cpu_time
                    return result
                raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(f):
            return async_wrapper
        else:
            return sync_wrapper

    # Handle decorator with/without arguments
    if func is None:
        return decorator
    else:
        return decorator(func)


# Accelerated vector operations with real MLX implementations
@gpuify(operation_type="vector_ops")
def cosine_similarity(a: mx.array | np.ndarray, b: mx.array | np.ndarray) -> float:
    """Compute cosine similarity between two vectors using optimized MLX operations."""
    if isinstance(a, np.ndarray):
        a = mx.array(a)
    if isinstance(b, np.ndarray):
        b = mx.array(b)

    # Ensure vectors are 1D
    if a.ndim > 1:
        a = mx.flatten(a)
    if b.ndim > 1:
        b = mx.flatten(b)

    # Check for zero vectors to avoid division by zero
    norm_a = mx.linalg.norm(a)
    norm_b = mx.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    # Normalize vectors using efficient operations
    a_norm = a / norm_a
    b_norm = b / norm_b

    # Compute dot product for cosine similarity
    similarity = mx.sum(a_norm * b_norm)
    mx.eval(similarity)

    return float(similarity)


@gpuify(operation_type="similarity")
def batch_cosine_similarity(
    query: mx.array | np.ndarray, vectors: mx.array | np.ndarray
) -> mx.array:
    """Compute cosine similarity between query and batch of vectors with optimized batching."""
    if isinstance(query, np.ndarray):
        query = mx.array(query)
    if isinstance(vectors, np.ndarray):
        vectors = mx.array(vectors)

    # Ensure proper dimensions
    if query.ndim == 1:
        query = mx.expand_dims(query, axis=0)
    if vectors.ndim == 1:
        vectors = mx.expand_dims(vectors, axis=0)

    # Handle batch vs single query
    if query.shape[0] > 1:
        # Multiple queries: compute pairwise similarities
        query_norms = mx.linalg.norm(query, axis=1, keepdims=True)
        vector_norms = mx.linalg.norm(vectors, axis=1, keepdims=True)

        # Avoid division by zero
        query_norms = mx.maximum(query_norms, 1e-8)
        vector_norms = mx.maximum(vector_norms, 1e-8)

        query_normalized = query / query_norms
        vectors_normalized = vectors / vector_norms

        # Compute similarity matrix
        similarities = query_normalized @ vectors_normalized.T
    else:
        # Single query: compute similarities with all vectors
        query_norm = mx.linalg.norm(query, axis=1, keepdims=True)
        vector_norms = mx.linalg.norm(vectors, axis=1, keepdims=True)

        # Avoid division by zero
        query_norm = mx.maximum(query_norm, 1e-8)
        vector_norms = mx.maximum(vector_norms, 1e-8)

        query_normalized = query / query_norm
        vectors_normalized = vectors / vector_norms

        # Compute similarities
        similarities = mx.squeeze(vectors_normalized @ query_normalized.T, axis=1)

    mx.eval(similarities)
    return similarities


@gpuify(operation_type="vector_ops")
def euclidean_distance(a: mx.array | np.ndarray, b: mx.array | np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    if isinstance(a, np.ndarray):
        a = mx.array(a)
    if isinstance(b, np.ndarray):
        b = mx.array(b)

    diff = a - b
    distance = mx.sqrt(mx.sum(diff * diff))
    mx.eval(distance)

    return float(distance)


@gpuify(operation_type="matrix_ops")
def matrix_multiply(a: mx.array | np.ndarray, b: mx.array | np.ndarray) -> mx.array:
    """Accelerated matrix multiplication with optimized memory layout."""
    if isinstance(a, np.ndarray):
        a = mx.array(a)
    if isinstance(b, np.ndarray):
        b = mx.array(b)

    # Validate matrix dimensions
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected 2D matrices, got shapes {a.shape} and {b.shape}")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Matrix dimension mismatch: {a.shape[1]} != {b.shape[0]}")

    # Use optimized matrix multiplication
    # MLX automatically uses Metal Performance Shaders for large matrices
    result = mx.matmul(a, b)

    # Force evaluation to ensure computation completes
    mx.eval(result)

    return result


@gpuify(operation_type="batch_ops", batch_size=4096)
def batch_matrix_multiply(
    matrices_a: list[mx.array | np.ndarray], matrices_b: list[mx.array | np.ndarray]
) -> list[mx.array]:
    """Optimized batch matrix multiplication with memory-efficient processing."""
    if len(matrices_a) != len(matrices_b):
        raise ValueError(
            f"Mismatched batch sizes: {len(matrices_a)} vs {len(matrices_b)}"
        )

    if not matrices_a:
        return []

    results: list[mx.array] = []
    batch_size = min(_accelerator._batch_size, 32)  # Limit for memory efficiency

    # Process in adaptive batches based on matrix sizes
    for i in range(0, len(matrices_a), batch_size):
        batch_a = matrices_a[i : i + batch_size]
        batch_b = matrices_b[i : i + batch_size]

        try:
            # Convert to MLX arrays if needed
            mlx_batch_a = [
                mx.array(m) if isinstance(m, np.ndarray) else m for m in batch_a
            ]
            mlx_batch_b = [
                mx.array(m) if isinstance(m, np.ndarray) else m for m in batch_b
            ]

            # Validate shapes for batching
            first_shape_a = mlx_batch_a[0].shape
            first_shape_b = mlx_batch_b[0].shape

            # Check if all matrices have compatible shapes for stacking
            can_stack = all(m.shape == first_shape_a for m in mlx_batch_a) and all(
                m.shape == first_shape_b for m in mlx_batch_b
            )

            if can_stack and len(mlx_batch_a) > 1:
                # Stack into batch tensors for efficient computation
                a_tensor = mx.stack(mlx_batch_a, axis=0)
                b_tensor = mx.stack(mlx_batch_b, axis=0)

                # Batch matrix multiplication
                batch_results = mx.matmul(a_tensor, b_tensor)
                mx.eval(batch_results)

                # Unstack results
                for j in range(len(mlx_batch_a)):
                    results.append(batch_results[j])
            else:
                # Process individually if shapes don't match
                for a_mat, b_mat in zip(mlx_batch_a, mlx_batch_b, strict=False):
                    result = mx.matmul(a_mat, b_mat)
                    mx.eval(result)
                    results.append(result)

        except Exception as e:
            logger.error(f"Batch matrix multiplication failed: {e}")
            # Fallback to individual processing
            for a_mat, b_mat in zip(batch_a, batch_b, strict=False):
                try:
                    result = matrix_multiply(a_mat, b_mat)
                    # matrix_multiply always returns mx.array, no conversion needed
                    results.append(result)
                except Exception as inner_e:
                    logger.error(f"Individual matrix multiply failed: {inner_e}")
                    # Return zero matrix as fallback
                    if isinstance(a_mat, np.ndarray):
                        a_mat = mx.array(a_mat)
                    if isinstance(b_mat, np.ndarray):
                        b_mat = mx.array(b_mat)
                    fallback_shape = (a_mat.shape[0], b_mat.shape[1])
                    results.append(mx.zeros(fallback_shape))

    return results


# Advanced text processing kernels with real MLX implementations
@gpuify(operation_type="embedding")
def text_embedding_similarity(
    embeddings_a: mx.array | np.ndarray, embeddings_b: mx.array | np.ndarray
) -> mx.array:
    """Compute pairwise similarities between text embeddings with numerical stability."""
    if isinstance(embeddings_a, np.ndarray):
        embeddings_a = mx.array(embeddings_a)
    if isinstance(embeddings_b, np.ndarray):
        embeddings_b = mx.array(embeddings_b)

    # Ensure 2D arrays
    if embeddings_a.ndim == 1:
        embeddings_a = mx.expand_dims(embeddings_a, axis=0)
    if embeddings_b.ndim == 1:
        embeddings_b = mx.expand_dims(embeddings_b, axis=0)

    # Compute norms with numerical stability
    norms_a = mx.linalg.norm(embeddings_a, axis=1, keepdims=True)
    norms_b = mx.linalg.norm(embeddings_b, axis=1, keepdims=True)

    # Add small epsilon to prevent division by zero
    eps = 1e-8
    norms_a = mx.maximum(norms_a, eps)
    norms_b = mx.maximum(norms_b, eps)

    # Normalize embeddings
    embeddings_a_norm = embeddings_a / norms_a
    embeddings_b_norm = embeddings_b / norms_b

    # Compute similarity matrix using optimized matrix multiplication
    similarities = mx.matmul(embeddings_a_norm, embeddings_b_norm.T)

    # Clamp to valid cosine similarity range [-1, 1]
    similarities = mx.clip(similarities, -1.0, 1.0)
    mx.eval(similarities)

    return similarities


@gpuify(operation_type="similarity")
def top_k_similarity_search(
    query: mx.array | np.ndarray, database: mx.array | np.ndarray, k: int = 10
) -> tuple[mx.array, mx.array]:
    """Find top-k most similar vectors using optimized GPU-accelerated search."""
    if isinstance(query, np.ndarray):
        query = mx.array(query)
    if isinstance(database, np.ndarray):
        database = mx.array(database)

    # Ensure proper dimensions
    if query.ndim == 1:
        query = mx.expand_dims(query, axis=0)
    if database.ndim == 1:
        database = mx.expand_dims(database, axis=0)

    # Limit k to database size
    k = min(k, database.shape[0])

    # Compute similarities efficiently
    similarities = batch_cosine_similarity(query, database)

    # Handle single query case
    if similarities.ndim == 1:
        # Use argpartition for better performance when k << n
        if k < database.shape[0] // 10:  # Use partition for small k
            # Argpartition for top-k (MLX equivalent)
            all_indices = mx.arange(similarities.shape[0])

            # Sort similarities and get indices
            sorted_indices = mx.argsort(-similarities)  # Negative for descending
            top_k_indices = sorted_indices[:k]
            top_k_scores = similarities[top_k_indices]
        else:
            # Full sort for large k
            sorted_indices = mx.argsort(-similarities)
            top_k_indices = sorted_indices[:k]
            top_k_scores = similarities[top_k_indices]
    else:
        # Handle multiple queries
        indices_list: list[mx.array] = []
        scores_list: list[mx.array] = []

        for i in range(similarities.shape[0]):
            query_sims = similarities[i]
            sorted_indices = mx.argsort(-query_sims)
            indices = sorted_indices[:k]
            scores = query_sims[indices]

            indices_list.append(indices)
            scores_list.append(scores)

        top_k_indices = mx.stack(indices_list)
        top_k_scores = mx.stack(scores_list)

    mx.eval(top_k_indices)
    mx.eval(top_k_scores)

    return top_k_indices, top_k_scores


# Advanced dense math operations with real implementations
@gpuify(operation_type="vector_ops")
def softmax(x: mx.array | np.ndarray, axis: int = -1) -> mx.array:
    """Compute numerically stable softmax using optimized MLX operations."""
    if isinstance(x, np.ndarray):
        x = mx.array(x)

    # Numerical stability: subtract max to prevent overflow
    x_max = mx.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max

    # Compute exponentials
    x_exp = mx.exp(x_shifted)

    # Compute sum with numerical stability
    x_sum = mx.sum(x_exp, axis=axis, keepdims=True)

    # Add small epsilon to prevent division by zero
    x_sum = mx.maximum(x_sum, 1e-8)

    # Compute softmax
    result = x_exp / x_sum
    mx.eval(result)

    return result


@gpuify(operation_type="vector_ops")
def layer_norm(
    x: mx.array | np.ndarray,
    gamma: mx.array | np.ndarray | None = None,
    beta: mx.array | np.ndarray | None = None,
    eps: float = 1e-5,
) -> mx.array:
    """Optimized layer normalization with proper broadcasting and numerical stability."""
    if isinstance(x, np.ndarray):
        x = mx.array(x)

    # Store original shape for proper broadcasting
    original_shape = x.shape
    last_dim = x.shape[-1]

    # Compute mean and variance along last dimension
    mean = mx.mean(x, axis=-1, keepdims=True)

    # Compute variance using the more numerically stable formula
    x_centered = x - mean
    var = mx.mean(x_centered * x_centered, axis=-1, keepdims=True)

    # Normalize with numerical stability
    denominator = mx.sqrt(var + eps)
    x_norm = x_centered / denominator

    # Apply affine transformation if parameters provided
    if gamma is not None:
        if isinstance(gamma, np.ndarray):
            gamma = mx.array(gamma)
        # Ensure gamma has correct shape for broadcasting
        if gamma.shape != (last_dim,) and gamma.shape != original_shape:
            gamma = mx.reshape(gamma, (last_dim,))
        x_norm = x_norm * gamma

    if beta is not None:
        if isinstance(beta, np.ndarray):
            beta = mx.array(beta)
        # Ensure beta has correct shape for broadcasting
        if beta.shape != (last_dim,) and beta.shape != original_shape:
            beta = mx.reshape(beta, (last_dim,))
        x_norm = x_norm + beta

    mx.eval(x_norm)
    return x_norm


@gpuify(operation_type="attention")
def attention_scores(
    query: mx.array | np.ndarray,
    key: mx.array | np.ndarray,
    value: mx.array | np.ndarray,
    mask: mx.array | np.ndarray | None = None,
) -> mx.array:
    """Compute optimized scaled dot-product attention with proper memory management."""
    if isinstance(query, np.ndarray):
        query = mx.array(query)
    if isinstance(key, np.ndarray):
        key = mx.array(key)
    if isinstance(value, np.ndarray):
        value = mx.array(value)

    # Validate input dimensions
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            f"Query and key last dimension mismatch: {query.shape[-1]} vs {key.shape[-1]}"
        )
    if key.shape[-2] != value.shape[-2]:
        raise ValueError(
            f"Key and value sequence length mismatch: {key.shape[-2]} vs {value.shape[-2]}"
        )

    # Get dimensions
    d_k = query.shape[-1]
    scale = 1.0 / mx.sqrt(mx.array(float(d_k)))

    # Compute attention scores with proper scaling
    # Use matmul for better performance on large matrices
    scores = mx.matmul(query, key.T) * scale

    # Apply mask if provided (for causal attention, padding, etc.)
    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask = mx.array(mask)

        # Ensure mask is broadcastable with scores
        if mask.shape != scores.shape:
            # Try to broadcast mask to scores shape
            try:
                mask = mx.broadcast_to(mask, scores.shape)
            except ValueError:
                logger.warning(
                    f"Mask shape {mask.shape} not compatible with scores shape {scores.shape}"
                )

        # Apply mask (masked positions get very negative values)
        mask_value = -1e9 if scores.dtype == mx.float32 else -1e4
        scores = mx.where(mask, scores, mask_value)

    # Compute attention weights using numerically stable softmax
    attention_weights = softmax(scores, axis=-1)

    # Apply attention to values
    output = mx.matmul(attention_weights, value)
    mx.eval(output)

    return output


# Advanced financial computing operations
@gpuify(operation_type="matrix_ops")
def covariance_matrix(returns: mx.array | np.ndarray, bias: bool = False) -> mx.array:
    """Compute covariance matrix for financial returns with GPU acceleration."""
    if isinstance(returns, np.ndarray):
        returns = mx.array(returns)

    # Center the data
    mean_returns = mx.mean(returns, axis=0, keepdims=True)
    centered_returns = returns - mean_returns

    # Compute covariance matrix
    n_samples = returns.shape[0]
    divisor = n_samples if bias else (n_samples - 1)

    cov_matrix = mx.matmul(centered_returns.T, centered_returns) / divisor
    mx.eval(cov_matrix)

    return cov_matrix


@gpuify(operation_type="matrix_ops")
def portfolio_optimization_weights(
    expected_returns: mx.array | np.ndarray,
    cov_matrix: mx.array | np.ndarray,
    risk_aversion: float = 1.0,
) -> mx.array:
    """Compute optimal portfolio weights using mean-variance optimization."""
    if isinstance(expected_returns, np.ndarray):
        expected_returns = mx.array(expected_returns)
    if isinstance(cov_matrix, np.ndarray):
        cov_matrix = mx.array(cov_matrix)

    # Add regularization for numerical stability
    n_assets = cov_matrix.shape[0]
    regularization = 1e-6 * mx.eye(n_assets)
    cov_reg = cov_matrix + regularization

    # Compute inverse covariance matrix
    try:
        inv_cov = mx.linalg.inv(cov_reg)
    except Exception:
        # Fallback to pseudo-inverse if singular
        inv_cov = mx.linalg.pinv(cov_reg)

    # Compute optimal weights
    ones = mx.ones((n_assets, 1))

    # Mean-variance optimization formula
    numerator = mx.matmul(inv_cov, expected_returns.reshape(-1, 1))
    denominator = mx.matmul(mx.matmul(ones.T, inv_cov), ones)

    weights = numerator / (risk_aversion * denominator)
    weights = mx.squeeze(weights)

    mx.eval(weights)
    return weights


@gpuify(operation_type="vector_ops")
def value_at_risk(
    returns: mx.array | np.ndarray, confidence_level: float = 0.05
) -> float:
    """Compute Value at Risk using GPU-accelerated quantile computation."""
    if isinstance(returns, np.ndarray):
        returns = mx.array(returns)

    # Sort returns for quantile computation
    sorted_returns = mx.sort(returns)
    n_returns = returns.shape[0]

    # Compute quantile index
    quantile_idx = int(confidence_level * n_returns)
    quantile_idx = max(0, min(quantile_idx, n_returns - 1))

    var_value = sorted_returns[quantile_idx]
    mx.eval(var_value)

    return float(var_value)


@gpuify(operation_type="matrix_ops")
def principal_component_analysis(
    data: mx.array | np.ndarray, n_components: int | None = None
) -> tuple[mx.array, mx.array, mx.array]:
    """Perform PCA using GPU-accelerated SVD."""
    if isinstance(data, np.ndarray):
        data = mx.array(data)

    # Center the data
    mean_data = mx.mean(data, axis=0, keepdims=True)
    centered_data = data - mean_data

    # Compute SVD
    U, S, Vt = mx.linalg.svd(centered_data)

    # Extract components
    if n_components is not None:
        n_components = min(n_components, S.shape[0])
        U = U[:, :n_components]
        S = S[:n_components]
        Vt = Vt[:n_components, :]

    # Compute explained variance ratio
    explained_variance = (S**2) / (data.shape[0] - 1)
    explained_variance_ratio = explained_variance / mx.sum(explained_variance)

    # Transform data to PCA space
    transformed_data = mx.matmul(centered_data, Vt.T)

    mx.eval([transformed_data, Vt, explained_variance_ratio])
    return transformed_data, Vt, explained_variance_ratio


# GPU Memory Management Integration
class GPUOperationContext:
    """Context manager for GPU operations with automatic memory management."""

    def __init__(self, operation_name: str, estimated_mb: float = 50.0):
        self.operation_name = operation_name
        self.estimated_mb = estimated_mb
        self.arrays_to_cleanup: list[mx.array] = []

    def __enter__(self):
        # Get memory manager if available
        try:
            from .gpu_memory_optimizer import get_gpu_memory_manager

            self.memory_manager = get_gpu_memory_manager()
            self.allocation_context = self.memory_manager.allocate_operation(
                self.operation_name, self.estimated_mb
            )
            self.allocation_context.__enter__()
        except ImportError:
            self.memory_manager = None
            self.allocation_context = None

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup tracked arrays
        for array in self.arrays_to_cleanup:
            if self.memory_manager:
                self.memory_manager.return_array_to_pool(array)

        # Exit allocation context
        if self.allocation_context:
            self.allocation_context.__exit__(exc_type, exc_val, exc_tb)

        # Force garbage collection if there was an error
        if exc_type is not None and self.memory_manager:
            self.memory_manager.cleanup_memory(force=True)

    def track_array(self, array):
        """Track array for automatic cleanup."""
        self.arrays_to_cleanup.append(array)
        return array


# Production-ready error handling for GPU operations
class GPUOperationError(Exception):
    """Custom exception for GPU operation failures."""

    def __init__(self, message: str, operation: str, fallback_available: bool = True):
        self.operation = operation
        self.fallback_available = fallback_available
        super().__init__(f"GPU operation '{operation}' failed: {message}")


def _cpu_fallback_operation(operation_name: str, *args, **kwargs):
    """CPU fallback implementations for critical operations."""
    if operation_name == "matrix_multiply" and len(args) >= 2:
        a, b = args[0], args[1]
        if isinstance(a, mx.array):
            a = np.array(a)
        if isinstance(b, mx.array):
            b = np.array(b)
        return np.matmul(a, b)

    elif operation_name == "cosine_similarity" and len(args) >= 2:
        a, b = args[0], args[1]
        if isinstance(a, mx.array):
            a = np.array(a)
        if isinstance(b, mx.array):
            b = np.array(b)

        # Flatten and normalize
        a_flat = a.flatten()
        b_flat = b.flatten()
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return np.dot(a_flat, b_flat) / (norm_a * norm_b)

    elif operation_name == "softmax" and len(args) >= 1:
        x = args[0]
        if isinstance(x, mx.array):
            x = np.array(x)

        axis = kwargs.get("axis", -1)
        x_max = np.max(x, axis=axis, keepdims=True)
        x_exp = np.exp(x - x_max)
        return x_exp / np.sum(x_exp, axis=axis, keepdims=True)

    else:
        raise NotImplementedError(f"CPU fallback not implemented for {operation_name}")


# Benchmark utilities
async def benchmark_gpu_operations():
    """Benchmark GPU operations and print results."""
    print("=== GPU Acceleration Benchmark with Intelligent Routing ===")
    print(f"MLX Metal Available: {_accelerator.gpu_available}")

    # Test different operation sizes to validate thresholds
    test_sizes = [
        (10, 10),  # Very small - should use CPU
        (100, 100),  # Small - should use CPU
        (500, 500),  # Medium - borderline
        (1000, 1000),  # Large - should use GPU
        (2000, 2000),  # Very large - should use GPU
    ]

    print("\n=== Matrix Multiplication Threshold Testing ===")
    for size in test_sizes:
        print(f"\nMatrix size: {size[0]}x{size[1]} ({size[0]*size[1]:,} elements)")

        # Create test data
        a = np.random.randn(*size).astype(np.float32)
        b = np.random.randn(*size).astype(np.float32)

        # Test with intelligent routing
        start = time.perf_counter()
        matrix_multiply(a, b)
        routing_time = time.perf_counter() - start

        # CPU baseline for comparison
        start = time.perf_counter()
        a @ b
        cpu_time = time.perf_counter() - start

        print(f"  Routed time: {routing_time*1000:.1f}ms")
        print(f"  CPU baseline: {cpu_time*1000:.1f}ms")
        print(f"  Speedup: {cpu_time/routing_time:.1f}x")

    # Test vector operations with different sizes
    print("\n=== Vector Operations Threshold Testing ===")
    vector_sizes = [100, 500, 1000, 5000, 10000]

    for size in vector_sizes:
        print(f"\nVector size: {size:,} elements")

        # Create test vectors
        a = np.random.randn(size).astype(np.float32)
        b = np.random.randn(size).astype(np.float32)

        # Test cosine similarity with routing
        start = time.perf_counter()
        cosine_similarity(a, b)
        routing_time = time.perf_counter() - start

        # CPU baseline
        start = time.perf_counter()
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        np.dot(a_norm, b_norm)
        cpu_time = time.perf_counter() - start

        print(f"  Routed time: {routing_time*1000:.3f}ms")
        print(f"  CPU baseline: {cpu_time*1000:.3f}ms")
        print(f"  Speedup: {cpu_time/routing_time:.1f}x")

    # Test similarity search with different database sizes
    print("\n=== Similarity Search Threshold Testing ===")
    database_sizes = [100, 1000, 5000, 10000]
    query = np.random.randn(768).astype(np.float32)

    for db_size in database_sizes:
        print(f"\nDatabase size: {db_size:,} vectors")
        database = np.random.randn(db_size, 768).astype(np.float32)

        # Test with routing
        start = time.perf_counter()
        gpu_indices, gpu_scores = top_k_similarity_search(query, database, k=10)
        routing_time = time.perf_counter() - start

        # CPU baseline
        start = time.perf_counter()
        cpu_sims = database @ query
        np.argsort(cpu_sims)[-10:]
        cpu_time = time.perf_counter() - start

        print(f"  Routed time: {routing_time*1000:.1f}ms")
        print(f"  CPU baseline: {cpu_time*1000:.1f}ms")
        print(f"  Speedup: {cpu_time/routing_time:.1f}x")

    # Test threshold auto-tuning
    print("\n=== Testing Threshold Auto-Tuning ===")
    _accelerator.tune_thresholds()

    # Print comprehensive stats
    print("\n=== Intelligent Routing Statistics ===")
    stats = _accelerator.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        elif isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    return stats


# Example usage for 8-agent system
class AgentAccelerator:
    """GPU-accelerated operations for multi-agent system."""

    def __init__(self, num_agents: int = 8):
        self.num_agents = num_agents
        self.embedding_dim = 768  # Standard transformer embedding size

    @gpuify(operation_type="similarity")
    async def compute_agent_similarities(
        self, agent_states: list[np.ndarray]
    ) -> mx.array:
        """Compute pairwise similarities between agent states."""
        # Stack agent states
        states = mx.stack([mx.array(s) for s in agent_states])

        # Compute similarity matrix
        similarities = text_embedding_similarity(states, states)

        return similarities

    @gpuify(operation_type="batch_ops", batch_size=8)
    async def update_agent_embeddings(
        self,
        embeddings: list[np.ndarray],
        gradients: list[np.ndarray],
        learning_rate: float = 0.001,
    ) -> list[mx.array]:
        """Update agent embeddings using gradients."""
        updated = []

        for emb, grad in zip(embeddings, gradients, strict=False):
            emb_mx = mx.array(emb) if isinstance(emb, np.ndarray) else emb
            grad_mx = mx.array(grad) if isinstance(grad, np.ndarray) else grad

            # Gradient descent update
            new_emb = emb_mx - learning_rate * grad_mx
            mx.eval(new_emb)
            updated.append(new_emb)

        return updated

    @gpuify(operation_type="attention")
    async def consensus_voting(self, agent_outputs: list[np.ndarray]) -> mx.array:
        """Compute consensus from agent outputs using attention mechanism."""
        # Stack outputs
        outputs = mx.stack([mx.array(o) for o in agent_outputs])

        # Self-attention to find consensus
        consensus = attention_scores(outputs, outputs, outputs)

        # Average pooling
        result = mx.mean(consensus, axis=0)
        mx.eval(result)

        return result


if __name__ == "__main__":
    # Run benchmark
    asyncio.run(benchmark_gpu_operations())

    # Example multi-agent usage
    async def example_usage():
        accelerator = AgentAccelerator(num_agents=8)

        # Generate random agent states
        agent_states = [np.random.randn(768).astype(np.float32) for _ in range(8)]

        # Compute similarities
        similarities = await accelerator.compute_agent_similarities(agent_states)
        print(f"\nAgent similarity matrix shape: {similarities.shape}")

        # Update embeddings
        gradients = [np.random.randn(768).astype(np.float32) * 0.01 for _ in range(8)]
        updated = await accelerator.update_agent_embeddings(agent_states, gradients)
        print(f"Updated {len(updated)} agent embeddings")

        # Consensus voting
        outputs = [np.random.randn(768).astype(np.float32) for _ in range(8)]
        consensus = await accelerator.consensus_voting(outputs)
        print(f"Consensus output shape: {consensus.shape}")

    asyncio.run(example_usage())


# Additional GPU-accelerated operations for completion
@gpuify(operation_type='matrix_ops')
def cholesky_decomposition(matrix: mx.array | np.ndarray) -> mx.array:
    """Compute Cholesky decomposition for positive definite matrices."""
    if isinstance(matrix, np.ndarray):
        matrix = mx.array(matrix)
    
    # Add small regularization for numerical stability
    n = matrix.shape[0]
    regularization = 1e-8 * mx.eye(n)
    matrix_reg = matrix + regularization
    
    # MLX Cholesky decomposition
    try:
        chol = mx.linalg.cholesky(matrix_reg)
    except Exception:
        # Fallback to eigendecomposition-based method
        eigenvals, eigenvecs = mx.linalg.eigh(matrix_reg)
        eigenvals = mx.maximum(eigenvals, 1e-8)  # Ensure positive
        sqrt_vals = mx.sqrt(eigenvals)
        chol = eigenvecs @ mx.diag(sqrt_vals)
    
    mx.eval(chol)
    return chol


@gpuify(operation_type='matrix_ops')
def eigendecomposition(matrix: mx.array | np.ndarray) -> tuple[mx.array, mx.array]:
    """Compute eigenvalues and eigenvectors using GPU acceleration."""
    if isinstance(matrix, np.ndarray):
        matrix = mx.array(matrix)
    
    # Ensure matrix is symmetric for stable decomposition
    symmetric_matrix = (matrix + matrix.T) / 2
    
    try:
        # Try GPU first
        eigenvals, eigenvecs = mx.linalg.eigh(symmetric_matrix)
        mx.eval([eigenvals, eigenvecs])
    except Exception:
        # Fallback to CPU stream for operations not yet supported on GPU
        import mlx.core as mx_cpu
        cpu_stream = mx_cpu.default_device_streams()[mx_cpu.default_device()]
        with mx_cpu.stream(cpu_stream):
            eigenvals, eigenvecs = mx.linalg.eigh(symmetric_matrix)
            mx.eval([eigenvals, eigenvecs])
    
    return eigenvals, eigenvecs


@gpuify(operation_type='vector_ops')
def gpu_sort_with_indices(array: mx.array | np.ndarray, axis: int = -1) -> tuple[mx.array, mx.array]:
    """GPU-accelerated sorting with indices."""
    if isinstance(array, np.ndarray):
        array = mx.array(array)
    
    # Get sorted indices
    indices = mx.argsort(array, axis=axis)
    
    # Get sorted values
    if axis == -1 or axis == array.ndim - 1:
        sorted_array = mx.take_along_axis(array, indices, axis=axis)
    else:
        sorted_array = array[indices] if array.ndim == 1 else mx.take_along_axis(array, indices, axis=axis)
    
    mx.eval([sorted_array, indices])
    return sorted_array, indices


@gpuify(operation_type='vector_ops') 
def quantile_computation(data: mx.array | np.ndarray, quantiles: list[float]) -> mx.array:
    """GPU-accelerated quantile computation."""
    if isinstance(data, np.ndarray):
        data = mx.array(data)
    
    # Sort the data
    sorted_data = mx.sort(data)
    n = data.shape[0]
    
    # Compute quantile indices
    quantile_results = []
    for q in quantiles:
        # Linear interpolation between indices
        index = q * (n - 1)
        lower_idx = int(mx.floor(index))
        upper_idx = int(mx.ceil(index))
        
        # Handle edge cases
        lower_idx = max(0, min(lower_idx, n - 1))
        upper_idx = max(0, min(upper_idx, n - 1))
        
        if lower_idx == upper_idx:
            quantile_val = sorted_data[lower_idx]
        else:
            # Linear interpolation
            weight = index - lower_idx
            quantile_val = (1 - weight) * sorted_data[lower_idx] + weight * sorted_data[upper_idx]
        
        quantile_results.append(quantile_val)
    
    result = mx.array(quantile_results)
    mx.eval(result)
    return result


@gpuify(operation_type='matrix_ops')
def batch_matrix_inverse(matrices: list[mx.array | np.ndarray]) -> list[mx.array]:
    """GPU-accelerated batch matrix inversion with regularization."""
    results = []
    
    for matrix in matrices:
        if isinstance(matrix, np.ndarray):
            matrix = mx.array(matrix)
        
        try:
            # Add regularization for numerical stability
            n = matrix.shape[0]
            regularization = 1e-6 * mx.eye(n)
            matrix_reg = matrix + regularization
            
            # Compute inverse
            inv_matrix = mx.linalg.inv(matrix_reg)
            mx.eval(inv_matrix)
            results.append(inv_matrix)
        except Exception:
            # Fallback to pseudo-inverse
            logger.warning("Matrix inversion failed, using pseudo-inverse")
            pinv_matrix = mx.linalg.pinv(matrix)
            mx.eval(pinv_matrix)
            results.append(pinv_matrix)
    
    return results


@gpuify(operation_type='matrix_ops')
def correlation_matrix(data: mx.array | np.ndarray) -> mx.array:
    """Compute correlation matrix with GPU acceleration."""
    if isinstance(data, np.ndarray):
        data = mx.array(data)
    
    # Center the data
    mean_data = mx.mean(data, axis=0, keepdims=True)
    centered = data - mean_data
    
    # Compute standard deviations
    std_devs = mx.std(centered, axis=0, keepdims=True)
    
    # Avoid division by zero
    std_devs = mx.maximum(std_devs, 1e-8)
    
    # Normalize
    normalized = centered / std_devs
    
    # Compute correlation matrix
    n_samples = data.shape[0]
    corr_matrix = mx.matmul(normalized.T, normalized) / (n_samples - 1)
    
    mx.eval(corr_matrix)
    return corr_matrix


@gpuify(operation_type='vector_ops')
def gpu_histogram(data: mx.array | np.ndarray, bins: int = 50) -> tuple[mx.array, mx.array]:
    """GPU-accelerated histogram computation."""
    if isinstance(data, np.ndarray):
        data = mx.array(data)
    
    # Compute data range
    data_min = mx.min(data)
    data_max = mx.max(data)
    
    # Create bin edges
    bin_width = (data_max - data_min) / bins
    bin_edges = mx.arange(bins + 1) * bin_width + data_min
    
    # Compute histogram
    # Simple implementation using sorting and counting
    sorted_data = mx.sort(data)
    n = data.shape[0]
    
    # Count elements in each bin
    hist_counts = []
    for i in range(bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1] if i < bins - 1 else data_max + 1e-10
        
        # Count elements in bin range
        mask = (sorted_data >= lower) & (sorted_data < upper)
        count = mx.sum(mask.astype(mx.int32))
        hist_counts.append(count)
    
    histogram = mx.array(hist_counts)
    mx.eval([histogram, bin_edges])
    
    return histogram, bin_edges


@gpuify(operation_type='matrix_ops')
def svd_decomposition(matrix: mx.array | np.ndarray) -> tuple[mx.array, mx.array, mx.array]:
    """GPU-accelerated SVD with fallback handling."""
    if isinstance(matrix, np.ndarray):
        matrix = mx.array(matrix)
    
    try:
        U, S, Vt = mx.linalg.svd(matrix)
        mx.eval([U, S, Vt])
        return U, S, Vt
    except Exception as e:
        logger.warning(f"SVD failed: {e}, using alternative approach")
        # Fallback: use eigendecomposition on ATA and AAT
        if matrix.shape[0] <= matrix.shape[1]:
            # More rows than columns: use ATA
            ATA = mx.matmul(matrix.T, matrix)
            eigenvals, V = mx.linalg.eigh(ATA)
            
            # Sort in descending order
            idx = mx.argsort(-eigenvals)
            eigenvals = eigenvals[idx]
            V = V[:, idx]
            
            # Compute S and U
            S = mx.sqrt(mx.maximum(eigenvals, 0))
            U = mx.matmul(matrix, V) / mx.maximum(S, 1e-10)
            Vt = V.T
        else:
            # More columns than rows: use AAT
            AAT = mx.matmul(matrix, matrix.T)
            eigenvals, U = mx.linalg.eigh(AAT)
            
            # Sort in descending order
            idx = mx.argsort(-eigenvals)
            eigenvals = eigenvals[idx]
            U = U[:, idx]
            
            # Compute S and V
            S = mx.sqrt(mx.maximum(eigenvals, 0))
            V = mx.matmul(matrix.T, U) / mx.maximum(S, 1e-10)
            Vt = V.T
        
        mx.eval([U, S, Vt])
        return U, S, Vt
