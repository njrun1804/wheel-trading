"""Production-ready GPU acceleration with MLX for the 8-agent system.

Provides MLX-accelerated operations with:
- @gpuify decorator for automatic GPU offloading
- Accelerated similarity searches, vector operations, and dense math
- Optimized embeddings and text processing kernels  
- Fallback to CPU when GPU is busy
- Metal compute optimizations for Apple Silicon (M4 Pro)
"""

import asyncio
import functools
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """Manages GPU memory allocation and prevents overflow."""

    def __init__(self, max_memory_gb: float = 18.0):
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self.allocated_bytes = 0
        self._lock = threading.Lock()

    def can_allocate(self, bytes_needed: int) -> bool:
        """Check if allocation is possible without overflow."""
        with self._lock:
            # Keep 20% buffer for safety
            threshold = self.max_memory_bytes * 0.8
            return (self.allocated_bytes + bytes_needed) < threshold

    def allocate(self, bytes_needed: int):
        """Track allocation."""
        with self._lock:
            self.allocated_bytes += bytes_needed

    def deallocate(self, bytes_freed: int):
        """Track deallocation."""
        with self._lock:
            self.allocated_bytes = max(0, self.allocated_bytes - bytes_freed)


@dataclass
class GPUStats:
    """Performance statistics for GPU operations."""

    gpu_ops: int = 0
    cpu_ops: int = 0
    gpu_time: float = 0.0
    cpu_time: float = 0.0
    memory_peak: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def gpu_utilization(self) -> float:
        total = self.gpu_ops + self.cpu_ops
        return 100.0 * self.gpu_ops / max(1, total)

    @property
    def avg_speedup(self) -> float:
        if self.gpu_time > 0:
            return self.cpu_time / self.gpu_time
        return 0.0


class GPUAccelerator:
    """Main GPU accelerator for MLX operations on M4 Pro."""

    def __init__(self):
        """Initialize GPU accelerator."""
        # Load config
        config_path = Path(__file__).parent.parent / "optimization_config.json"
        with open(config_path) as f:
            self.config = json.load(f)

        # Check GPU availability
        self.gpu_available = mx.metal.is_available()
        if self.gpu_available:
            mx.set_default_device(mx.gpu)
            logger.info("MLX Metal GPU acceleration enabled on M4 Pro")
        else:
            logger.warning("MLX Metal GPU not available")

        # Memory manager
        self.memory_manager = GPUMemoryManager(
            max_memory_gb=self.config["memory"]["max_allocation_gb"]
        )

        # Stats tracking
        self.stats = GPUStats()

        # Operation cache
        self._op_cache = {}

    def estimate_memory(self, *arrays) -> int:
        """Estimate memory usage for arrays."""
        total = 0
        for arr in arrays:
            if isinstance(arr, mx.array | np.ndarray):
                total += arr.nbytes
            elif hasattr(arr, "__len__") and hasattr(arr[0], "shape"):
                # List of arrays
                for a in arr:
                    if isinstance(a, mx.array | np.ndarray):
                        total += a.nbytes
        return total

    def should_use_gpu(self, memory_needed: int) -> bool:
        """Decide whether to use GPU based on memory availability."""
        if not self.gpu_available:
            return False
        return self.memory_manager.can_allocate(memory_needed)


# Global accelerator instance
_accelerator = GPUAccelerator()


def gpuify(
    func: Callable | None = None,
    *,
    fallback: bool = True,
    memory_check: bool = True,
    cache_key: str | None = None,
) -> Callable:
    """Decorator for automatic GPU offloading with MLX.

    Args:
        func: Function to decorate
        fallback: Whether to fallback to CPU if GPU unavailable
        memory_check: Whether to check GPU memory before offloading
        cache_key: Optional key for caching compiled operations
    """

    def decorator(f: Callable) -> Callable:
        # Generate cache key if not provided
        nonlocal cache_key
        if cache_key is None:
            cache_key = f"{f.__module__}.{f.__name__}"

        @functools.wraps(f)
        def sync_wrapper(*args, **kwargs):
            # Memory check
            if memory_check:
                mem_needed = _accelerator.estimate_memory(*args)
                use_gpu = _accelerator.should_use_gpu(mem_needed)
            else:
                use_gpu = _accelerator.gpu_available

            start = time.perf_counter()

            try:
                if use_gpu:
                    # Convert to MLX arrays
                    mlx_args = []
                    for arg in args:
                        if isinstance(arg, np.ndarray):
                            mlx_args.append(mx.array(arg))
                        else:
                            mlx_args.append(arg)

                    # Execute on GPU
                    with mx.stream(mx.gpu):
                        result = f(*mlx_args, **kwargs)

                        # Ensure evaluation
                        if isinstance(result, mx.array):
                            mx.eval(result)
                        elif isinstance(result, tuple) and all(
                            isinstance(r, mx.array) for r in result
                        ):
                            mx.eval(*result)

                    # Update stats
                    elapsed = time.perf_counter() - start
                    _accelerator.stats.gpu_ops += 1
                    _accelerator.stats.gpu_time += elapsed

                    # Track memory
                    if isinstance(result, mx.array):
                        _accelerator.stats.memory_peak = max(
                            _accelerator.stats.memory_peak, result.nbytes
                        )

                    return result

                elif fallback:
                    # CPU fallback
                    result = f(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    _accelerator.stats.cpu_ops += 1
                    _accelerator.stats.cpu_time += elapsed
                    return result
                else:
                    raise RuntimeError(f"GPU required for {f.__name__}")

            except Exception as e:
                if fallback:
                    logger.debug(f"GPU error, falling back to CPU: {e}")
                    result = f(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    _accelerator.stats.cpu_ops += 1
                    _accelerator.stats.cpu_time += elapsed
                    return result
                raise

        @functools.wraps(f)
        async def async_wrapper(*args, **kwargs):
            # Run sync version in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, sync_wrapper, *args, **kwargs)

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(f):
            return async_wrapper
        return sync_wrapper

    if func is None:
        return decorator
    return decorator(func)


# Core accelerated operations
@gpuify(cache_key="cosine_similarity")
def cosine_similarity(a: mx.array | np.ndarray, b: mx.array | np.ndarray) -> float:
    """GPU-accelerated cosine similarity."""
    if not isinstance(a, mx.array):
        a = mx.array(a, dtype=mx.float32)
    if not isinstance(b, mx.array):
        b = mx.array(b, dtype=mx.float32)

    # Normalize
    a_norm = a / (mx.linalg.norm(a) + 1e-8)
    b_norm = b / (mx.linalg.norm(b) + 1e-8)

    similarity = mx.sum(a_norm * b_norm)
    return float(similarity)


@gpuify(cache_key="batch_cosine_similarity")
def batch_cosine_similarity(
    query: mx.array | np.ndarray, vectors: mx.array | np.ndarray, chunk_size: int = 4096
) -> np.ndarray:
    """Batch cosine similarity with chunking for memory efficiency."""
    if not isinstance(query, mx.array):
        query = mx.array(query, dtype=mx.float32)
    if not isinstance(vectors, mx.array):
        vectors = mx.array(vectors, dtype=mx.float32)

    # Normalize query
    query_norm = query / (mx.linalg.norm(query) + 1e-8)

    n_vectors = vectors.shape[0]
    similarities = []

    # Process in chunks
    for i in range(0, n_vectors, chunk_size):
        end = min(i + chunk_size, n_vectors)
        chunk = vectors[i:end]

        # Normalize chunk
        norms = mx.linalg.norm(chunk, axis=1, keepdims=True)
        chunk_norm = chunk / (norms + 1e-8)

        # Compute similarities
        chunk_sims = chunk_norm @ query_norm
        similarities.append(chunk_sims)

    # Concatenate results
    result = mx.concatenate(similarities)
    return np.array(result)


@gpuify(cache_key="top_k_search")
def top_k_similarity_search(
    query: mx.array | np.ndarray, database: mx.array | np.ndarray, k: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Find top-k most similar vectors."""
    # Compute similarities
    similarities = batch_cosine_similarity(query, database)

    # Convert to numpy for sorting (MLX sort can be slower for large arrays)
    if isinstance(similarities, mx.array):
        similarities_np = np.array(similarities)
    else:
        similarities_np = similarities

    # Get top-k indices
    if k < len(similarities_np) // 10:
        # For small k, use partition
        indices = np.argpartition(similarities_np, -k)[-k:]
        indices = indices[np.argsort(similarities_np[indices])][::-1]
    else:
        # For large k, use full sort
        indices = np.argsort(similarities_np)[::-1][:k]

    scores = similarities_np[indices]

    return indices, scores


@gpuify(cache_key="matrix_multiply")
def matrix_multiply(
    a: mx.array | np.ndarray, b: mx.array | np.ndarray
) -> mx.array | np.ndarray:
    """GPU-accelerated matrix multiplication."""
    return_numpy = isinstance(a, np.ndarray)

    if not isinstance(a, mx.array):
        a = mx.array(a, dtype=mx.float32)
    if not isinstance(b, mx.array):
        b = mx.array(b, dtype=mx.float32)

    result = a @ b

    if return_numpy:
        return np.array(result)
    return result


@gpuify(cache_key="attention")
def scaled_dot_product_attention(
    query: mx.array | np.ndarray,
    key: mx.array | np.ndarray,
    value: mx.array | np.ndarray,
    mask: mx.array | np.ndarray | None = None,
    temperature: float = 1.0,
) -> mx.array | np.ndarray:
    """Scaled dot-product attention mechanism."""
    return_numpy = isinstance(query, np.ndarray)

    if not isinstance(query, mx.array):
        query = mx.array(query, dtype=mx.float32)
    if not isinstance(key, mx.array):
        key = mx.array(key, dtype=mx.float32)
    if not isinstance(value, mx.array):
        value = mx.array(value, dtype=mx.float32)

    # Compute attention scores
    d_k = query.shape[-1]
    scores = (query @ key.T) / (mx.sqrt(mx.array(d_k)) * temperature)

    # Apply mask
    if mask is not None:
        if not isinstance(mask, mx.array):
            mask = mx.array(mask)
        scores = scores + mask * -1e9

    # Softmax
    weights = mx.softmax(scores, axis=-1)

    # Apply attention
    output = weights @ value

    if return_numpy:
        return np.array(output)
    return output


# Agent-specific accelerated operations
class AgentGPUAccelerator:
    """GPU acceleration for 8-agent system operations."""

    def __init__(self, num_agents: int = 8, embedding_dim: int = 768):
        self.num_agents = num_agents
        self.embedding_dim = embedding_dim

    @gpuify
    def compute_agent_similarities(self, agent_states: list[np.ndarray]) -> np.ndarray:
        """Compute pairwise similarities between agents."""
        # Stack states
        states = mx.stack([mx.array(s, dtype=mx.float32) for s in agent_states])

        # Normalize
        norms = mx.linalg.norm(states, axis=1, keepdims=True)
        states_norm = states / (norms + 1e-8)

        # Compute similarity matrix
        similarities = states_norm @ states_norm.T

        return np.array(similarities)

    @gpuify
    def parallel_agent_forward(
        self, states: list[np.ndarray], weights: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Parallel forward pass for all agents."""
        # Convert to MLX
        states_mx = [mx.array(s, dtype=mx.float32) for s in states]
        weights_mx = [mx.array(w, dtype=mx.float32) for w in weights]

        # Parallel computation
        outputs = []
        for state, weight in zip(states_mx, weights_mx, strict=False):
            output = state @ weight
            outputs.append(output)

        # Evaluate all at once
        mx.eval(*outputs)

        # Convert back
        return [np.array(o) for o in outputs]

    @gpuify
    def hierarchical_consensus(
        self, agent_outputs: list[np.ndarray], temperature: float = 1.0
    ) -> np.ndarray:
        """Compute consensus using attention mechanism."""
        # Stack outputs
        outputs = mx.stack([mx.array(o, dtype=mx.float32) for o in agent_outputs])

        # Self-attention for consensus
        consensus = scaled_dot_product_attention(
            outputs, outputs, outputs, temperature=temperature
        )

        # Average pooling
        result = mx.mean(consensus, axis=0)

        return np.array(result)

    @gpuify
    def update_embeddings(
        self,
        embeddings: list[np.ndarray],
        gradients: list[np.ndarray],
        learning_rate: float = 0.001,
    ) -> list[np.ndarray]:
        """Update agent embeddings using gradients."""
        updated = []

        for emb, grad in zip(embeddings, gradients, strict=False):
            emb_mx = mx.array(emb, dtype=mx.float32)
            grad_mx = mx.array(grad, dtype=mx.float32)

            # Gradient descent
            new_emb = emb_mx - learning_rate * grad_mx
            updated.append(np.array(new_emb))

        return updated


def get_gpu_stats() -> dict[str, Any]:
    """Get current GPU acceleration statistics."""
    stats = _accelerator.stats
    return {
        "gpu_available": _accelerator.gpu_available,
        "gpu_operations": stats.gpu_ops,
        "cpu_fallbacks": stats.cpu_ops,
        "gpu_utilization": stats.gpu_utilization,
        "average_speedup": stats.avg_speedup,
        "memory_peak_mb": stats.memory_peak / (1024 * 1024),
        "avg_gpu_time_ms": stats.gpu_time / max(1, stats.gpu_ops) * 1000,
        "avg_cpu_time_ms": stats.cpu_time / max(1, stats.cpu_ops) * 1000,
    }


async def benchmark():
    """Comprehensive benchmark of GPU acceleration."""
    print("=== GPU Acceleration Benchmark (M4 Pro) ===")
    print(f"MLX Metal Available: {_accelerator.gpu_available}\n")

    # 1. Matrix multiplication
    print("1. Matrix Multiplication:")
    for size in [(1024, 1024), (4096, 4096), (8192, 2048)]:
        a = np.random.randn(*size).astype(np.float32)
        b = np.random.randn(size[1], size[0]).astype(np.float32)

        # CPU baseline
        start = time.perf_counter()
        np.dot(a, b)
        cpu_time = time.perf_counter() - start

        # GPU accelerated
        start = time.perf_counter()
        matrix_multiply(a, b)
        gpu_time = time.perf_counter() - start

        speedup = cpu_time / gpu_time
        print(
            f"  {size}: CPU={cpu_time*1000:.1f}ms, GPU={gpu_time*1000:.1f}ms, Speedup={speedup:.1f}x"
        )

    # 2. Similarity search
    print("\n2. Similarity Search (50k vectors, 768 dims):")
    query = np.random.randn(768).astype(np.float32)
    database = np.random.randn(50000, 768).astype(np.float32)

    # Normalize
    query = query / np.linalg.norm(query)
    database = database / np.linalg.norm(database, axis=1, keepdims=True)

    start = time.perf_counter()
    indices, scores = top_k_similarity_search(query, database, k=100)
    search_time = time.perf_counter() - start

    print(f"  Time: {search_time*1000:.1f}ms")
    print(f"  Throughput: {50000/search_time:.0f} vectors/sec")

    # 3. 8-Agent system
    print("\n3. 8-Agent System Operations:")
    agent_acc = AgentGPUAccelerator(num_agents=8)

    # Generate test data
    states = [np.random.randn(768).astype(np.float32) for _ in range(8)]
    weights = [np.random.randn(768, 768).astype(np.float32) * 0.01 for _ in range(8)]

    # Agent similarities
    start = time.perf_counter()
    agent_acc.compute_agent_similarities(states)
    sim_time = time.perf_counter() - start

    # Parallel forward
    start = time.perf_counter()
    outputs = agent_acc.parallel_agent_forward(states, weights)
    forward_time = time.perf_counter() - start

    # Consensus
    start = time.perf_counter()
    agent_acc.hierarchical_consensus(outputs)
    consensus_time = time.perf_counter() - start

    print(f"  Similarities: {sim_time*1000:.1f}ms")
    print(f"  Parallel forward: {forward_time*1000:.1f}ms")
    print(f"  Consensus: {consensus_time*1000:.1f}ms")
    print(f"  Total: {(sim_time+forward_time+consensus_time)*1000:.1f}ms")

    # 4. Overall stats
    print("\n=== Overall Performance ===")
    stats = get_gpu_stats()
    print(f"GPU Utilization: {stats['gpu_utilization']:.1f}%")
    print(f"Average Speedup: {stats['average_speedup']:.1f}x")
    print(f"Memory Peak: {stats['memory_peak_mb']:.1f}MB")
    print(
        f"GPU ops: {stats['gpu_operations']}, CPU fallbacks: {stats['cpu_fallbacks']}"
    )


if __name__ == "__main__":
    # Run benchmark
    asyncio.run(benchmark())

    # Example usage
    print("\n=== Example Usage ===")

    # Simple similarity
    a = np.random.randn(768).astype(np.float32)
    b = np.random.randn(768).astype(np.float32)
    similarity = cosine_similarity(a, b)
    print(f"Cosine similarity: {similarity:.4f}")

    # Agent system
    agent_acc = AgentGPUAccelerator()
    states = [np.random.randn(768).astype(np.float32) for _ in range(8)]
    sims = agent_acc.compute_agent_similarities(states)
    print(f"Agent similarity matrix:\n{sims[:3, :3]}")  # Show 3x3 subset
