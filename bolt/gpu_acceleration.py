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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
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
        
        # GPU memory tracking
        self._memory_threshold = self.config["memory"]["max_allocation_gb"] * 0.7 * 1024 * 1024 * 1024  # 70% of max
        self._batch_size = self.config["gpu"]["batch_size"]
        
        # Performance stats
        self.stats = {
            "gpu_operations": 0,
            "cpu_fallbacks": 0,
            "total_gpu_time": 0.0,
            "total_cpu_time": 0.0,
            "memory_peaks": []
        }
        
    def _check_gpu_memory(self, estimated_bytes: int) -> bool:
        """Check if GPU has enough memory for operation."""
        if not self.gpu_available:
            return False
            
        # Check if operation would exceed threshold
        if estimated_bytes > self._memory_threshold:
            logger.debug(f"Operation size {estimated_bytes / 1e9:.1f}GB exceeds threshold")
            return False
            
        return True
    
    def _estimate_memory(self, *arrays: Union[mx.array, np.ndarray]) -> int:
        """Estimate memory usage for arrays."""
        total_bytes = 0
        for arr in arrays:
            if isinstance(arr, mx.array):
                total_bytes += arr.nbytes
            elif isinstance(arr, np.ndarray):
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
        return 100.0 * self.stats["gpu_operations"] / (self.stats["gpu_operations"] + self.stats["cpu_fallbacks"])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_ops = self.stats["gpu_operations"] + self.stats["cpu_fallbacks"]
        
        return {
            "gpu_available": self.gpu_available,
            "gpu_operations": self.stats["gpu_operations"],
            "cpu_fallbacks": self.stats["cpu_fallbacks"],
            "gpu_utilization": self.gpu_utilization,
            "avg_gpu_time_ms": self.stats["total_gpu_time"] / max(1, self.stats["gpu_operations"]) * 1000,
            "avg_cpu_time_ms": self.stats["total_cpu_time"] / max(1, self.stats["cpu_fallbacks"]) * 1000,
            "speedup": self.stats["total_cpu_time"] / max(0.001, self.stats["total_gpu_time"]) if self.stats["total_gpu_time"] > 0 else 0,
            "memory_peak_gb": max(self.stats["memory_peaks"]) / 1e9 if self.stats["memory_peaks"] else 0
        }


# Global accelerator instance
_accelerator = GPUAccelerator()


def gpuify(func: Optional[Callable] = None, *, 
          fallback: bool = True,
          batch_size: Optional[int] = None,
          memory_check: bool = True) -> Callable:
    """Decorator for automatic GPU offloading with MLX.
    
    Args:
        func: Function to decorate
        fallback: Whether to fallback to CPU if GPU is busy
        batch_size: Override default batch size
        memory_check: Whether to check GPU memory before offloading
    
    Returns:
        Decorated function that runs on GPU when possible
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        async def async_wrapper(*args, **kwargs):
            # Estimate memory usage
            if memory_check:
                estimated_mem = _accelerator._estimate_memory(*args)
                use_gpu = _accelerator._check_gpu_memory(estimated_mem)
            else:
                use_gpu = _accelerator.gpu_available
            
            start_time = time.perf_counter()
            
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
                    _accelerator.stats["total_gpu_time"] += time.perf_counter() - start_time
                    
                    # Track memory usage
                    if memory_check and isinstance(result, mx.array):
                        _accelerator.stats["memory_peaks"].append(result.nbytes)
                    
                    # Convert back to numpy if needed
                    if isinstance(result, mx.array) and len(args) > 0 and isinstance(args[0], np.ndarray):
                        return np.array(result)
                    
                    return result
                    
                elif fallback:
                    # Fallback to CPU
                    logger.debug(f"Falling back to CPU for {f.__name__}")
                    _accelerator.stats["cpu_fallbacks"] += 1
                    
                    # Run original function
                    result = await f(*args, **kwargs)
                    _accelerator.stats["total_cpu_time"] += time.perf_counter() - start_time
                    
                    return result
                else:
                    raise RuntimeError(f"GPU required for {f.__name__} but not available")
                    
            except Exception as e:
                logger.error(f"Error in GPU operation {f.__name__}: {e}")
                if fallback:
                    _accelerator.stats["cpu_fallbacks"] += 1
                    result = await f(*args, **kwargs)
                    _accelerator.stats["total_cpu_time"] += time.perf_counter() - start_time
                    return result
                raise
        
        @functools.wraps(f)
        def sync_wrapper(*args, **kwargs):
            # Estimate memory usage
            if memory_check:
                estimated_mem = _accelerator._estimate_memory(*args)
                use_gpu = _accelerator._check_gpu_memory(estimated_mem)
            else:
                use_gpu = _accelerator.gpu_available
            
            start_time = time.perf_counter()
            
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
                    _accelerator.stats["total_gpu_time"] += time.perf_counter() - start_time
                    
                    # Track memory usage
                    if memory_check and isinstance(result, mx.array):
                        _accelerator.stats["memory_peaks"].append(result.nbytes)
                    
                    # Convert back to numpy if needed
                    if isinstance(result, mx.array) and len(args) > 0 and isinstance(args[0], np.ndarray):
                        return np.array(result)
                    
                    return result
                    
                elif fallback:
                    # Fallback to CPU
                    logger.debug(f"Falling back to CPU for {f.__name__}")
                    _accelerator.stats["cpu_fallbacks"] += 1
                    
                    # Run original function
                    result = f(*args, **kwargs)
                    _accelerator.stats["total_cpu_time"] += time.perf_counter() - start_time
                    
                    return result
                else:
                    raise RuntimeError(f"GPU required for {f.__name__} but not available")
                    
            except Exception as e:
                logger.error(f"Error in GPU operation {f.__name__}: {e}")
                if fallback:
                    _accelerator.stats["cpu_fallbacks"] += 1
                    result = f(*args, **kwargs)
                    _accelerator.stats["total_cpu_time"] += time.perf_counter() - start_time
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


# Accelerated vector operations
@gpuify
def cosine_similarity(a: Union[mx.array, np.ndarray], 
                     b: Union[mx.array, np.ndarray]) -> float:
    """Compute cosine similarity between two vectors."""
    if isinstance(a, np.ndarray):
        a = mx.array(a)
    if isinstance(b, np.ndarray):
        b = mx.array(b)
    
    # Normalize vectors
    a_norm = a / mx.linalg.norm(a)
    b_norm = b / mx.linalg.norm(b)
    
    # Compute similarity
    similarity = mx.sum(a_norm * b_norm)
    mx.eval(similarity)
    
    return float(similarity)


@gpuify
def batch_cosine_similarity(query: Union[mx.array, np.ndarray],
                           vectors: Union[mx.array, np.ndarray]) -> mx.array:
    """Compute cosine similarity between query and batch of vectors."""
    if isinstance(query, np.ndarray):
        query = mx.array(query)
    if isinstance(vectors, np.ndarray):
        vectors = mx.array(vectors)
    
    # Normalize query
    query_norm = query / mx.linalg.norm(query)
    
    # Normalize vectors (batch)
    norms = mx.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = vectors / norms
    
    # Compute similarities
    similarities = vectors_norm @ query_norm
    mx.eval(similarities)
    
    return similarities


@gpuify
def euclidean_distance(a: Union[mx.array, np.ndarray],
                      b: Union[mx.array, np.ndarray]) -> float:
    """Compute Euclidean distance between two vectors."""
    if isinstance(a, np.ndarray):
        a = mx.array(a)
    if isinstance(b, np.ndarray):
        b = mx.array(b)
    
    diff = a - b
    distance = mx.sqrt(mx.sum(diff * diff))
    mx.eval(distance)
    
    return float(distance)


@gpuify
def matrix_multiply(a: Union[mx.array, np.ndarray],
                   b: Union[mx.array, np.ndarray]) -> mx.array:
    """Accelerated matrix multiplication."""
    if isinstance(a, np.ndarray):
        a = mx.array(a)
    if isinstance(b, np.ndarray):
        b = mx.array(b)
    
    result = a @ b
    mx.eval(result)
    
    return result


@gpuify(batch_size=4096)
def batch_matrix_multiply(matrices_a: List[Union[mx.array, np.ndarray]],
                         matrices_b: List[Union[mx.array, np.ndarray]]) -> List[mx.array]:
    """Batch matrix multiplication with adaptive sizing."""
    results = []
    batch_size = _accelerator._batch_size
    
    # Process in batches
    for i in range(0, len(matrices_a), batch_size):
        batch_a = matrices_a[i:i + batch_size]
        batch_b = matrices_b[i:i + batch_size]
        
        # Stack into tensors
        a_tensor = mx.stack([mx.array(m) if isinstance(m, np.ndarray) else m for m in batch_a])
        b_tensor = mx.stack([mx.array(m) if isinstance(m, np.ndarray) else m for m in batch_b])
        
        # Batch multiplication
        batch_results = a_tensor @ b_tensor
        mx.eval(batch_results)
        
        # Unstack results
        for j in range(len(batch_a)):
            results.append(batch_results[j])
    
    return results


# Text processing kernels
@gpuify
def text_embedding_similarity(embeddings_a: Union[mx.array, np.ndarray],
                             embeddings_b: Union[mx.array, np.ndarray]) -> mx.array:
    """Compute pairwise similarities between text embeddings."""
    if isinstance(embeddings_a, np.ndarray):
        embeddings_a = mx.array(embeddings_a)
    if isinstance(embeddings_b, np.ndarray):
        embeddings_b = mx.array(embeddings_b)
    
    # Normalize embeddings
    norms_a = mx.linalg.norm(embeddings_a, axis=1, keepdims=True)
    norms_b = mx.linalg.norm(embeddings_b, axis=1, keepdims=True)
    
    embeddings_a_norm = embeddings_a / norms_a
    embeddings_b_norm = embeddings_b / norms_b
    
    # Compute similarity matrix
    similarities = embeddings_a_norm @ embeddings_b_norm.T
    mx.eval(similarities)
    
    return similarities


@gpuify
def top_k_similarity_search(query: Union[mx.array, np.ndarray],
                           database: Union[mx.array, np.ndarray],
                           k: int = 10) -> Tuple[mx.array, mx.array]:
    """Find top-k most similar vectors using GPU acceleration."""
    if isinstance(query, np.ndarray):
        query = mx.array(query)
    if isinstance(database, np.ndarray):
        database = mx.array(database)
    
    # Compute similarities
    similarities = batch_cosine_similarity(query, database)
    
    # Find top-k
    # MLX doesn't have direct topk, so we use argsort
    sorted_indices = mx.argsort(similarities, axis=0)[::-1][:k]
    top_k_scores = similarities[sorted_indices]
    
    mx.eval(sorted_indices)
    mx.eval(top_k_scores)
    
    return sorted_indices, top_k_scores


# Dense math operations
@gpuify
def softmax(x: Union[mx.array, np.ndarray], axis: int = -1) -> mx.array:
    """Compute softmax with numerical stability."""
    if isinstance(x, np.ndarray):
        x = mx.array(x)
    
    # Numerical stability
    x_max = mx.max(x, axis=axis, keepdims=True)
    x_exp = mx.exp(x - x_max)
    x_sum = mx.sum(x_exp, axis=axis, keepdims=True)
    
    result = x_exp / x_sum
    mx.eval(result)
    
    return result


@gpuify
def layer_norm(x: Union[mx.array, np.ndarray], 
              gamma: Optional[Union[mx.array, np.ndarray]] = None,
              beta: Optional[Union[mx.array, np.ndarray]] = None,
              eps: float = 1e-5) -> mx.array:
    """Layer normalization for transformer models."""
    if isinstance(x, np.ndarray):
        x = mx.array(x)
    
    # Compute mean and variance
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    
    # Normalize
    x_norm = (x - mean) / mx.sqrt(var + eps)
    
    # Apply scale and shift if provided
    if gamma is not None:
        if isinstance(gamma, np.ndarray):
            gamma = mx.array(gamma)
        x_norm = x_norm * gamma
    
    if beta is not None:
        if isinstance(beta, np.ndarray):
            beta = mx.array(beta)
        x_norm = x_norm + beta
    
    mx.eval(x_norm)
    return x_norm


@gpuify
def attention_scores(query: Union[mx.array, np.ndarray],
                    key: Union[mx.array, np.ndarray],
                    value: Union[mx.array, np.ndarray],
                    mask: Optional[Union[mx.array, np.ndarray]] = None) -> mx.array:
    """Compute scaled dot-product attention."""
    if isinstance(query, np.ndarray):
        query = mx.array(query)
    if isinstance(key, np.ndarray):
        key = mx.array(key)
    if isinstance(value, np.ndarray):
        value = mx.array(value)
    
    # Compute attention scores
    d_k = query.shape[-1]
    scores = (query @ key.T) / mx.sqrt(mx.array(d_k))
    
    # Apply mask if provided
    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask = mx.array(mask)
        scores = scores + mask * -1e9
    
    # Apply softmax
    attention_weights = softmax(scores, axis=-1)
    
    # Apply attention to values
    output = attention_weights @ value
    mx.eval(output)
    
    return output


# Benchmark utilities
async def benchmark_gpu_operations():
    """Benchmark GPU operations and print results."""
    print("=== GPU Acceleration Benchmark ===")
    print(f"MLX Metal Available: {_accelerator.gpu_available}")
    
    # Test different operation sizes
    sizes = [(100, 100), (1000, 1000), (4096, 4096)]
    
    for size in sizes:
        print(f"\nMatrix size: {size}")
        
        # Create test data
        a = np.random.randn(*size).astype(np.float32)
        b = np.random.randn(*size).astype(np.float32)
        
        # CPU baseline
        start = time.perf_counter()
        for _ in range(10):
            _ = a @ b
        cpu_time = (time.perf_counter() - start) / 10
        
        # GPU accelerated
        start = time.perf_counter()
        for _ in range(10):
            _ = matrix_multiply(a, b)
        gpu_time = (time.perf_counter() - start) / 10
        
        speedup = cpu_time / gpu_time
        print(f"  CPU: {cpu_time*1000:.1f}ms")
        print(f"  GPU: {gpu_time*1000:.1f}ms")
        print(f"  Speedup: {speedup:.1f}x")
    
    # Test similarity search
    print("\nSimilarity Search (10k vectors, 768 dims):")
    query = np.random.randn(768).astype(np.float32)
    database = np.random.randn(10000, 768).astype(np.float32)
    
    # CPU baseline
    start = time.perf_counter()
    cpu_sims = database @ query
    cpu_top_k = np.argsort(cpu_sims)[-10:]
    cpu_time = time.perf_counter() - start
    
    # GPU accelerated
    start = time.perf_counter()
    gpu_indices, gpu_scores = top_k_similarity_search(query, database, k=10)
    gpu_time = time.perf_counter() - start
    
    speedup = cpu_time / gpu_time
    print(f"  CPU: {cpu_time*1000:.1f}ms")
    print(f"  GPU: {gpu_time*1000:.1f}ms")
    print(f"  Speedup: {speedup:.1f}x")
    
    # Print overall stats
    print("\n=== Overall Statistics ===")
    stats = _accelerator.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")


# Example usage for 8-agent system
class AgentAccelerator:
    """GPU-accelerated operations for multi-agent system."""
    
    def __init__(self, num_agents: int = 8):
        self.num_agents = num_agents
        self.embedding_dim = 768  # Standard transformer embedding size
        
    @gpuify
    async def compute_agent_similarities(self, agent_states: List[np.ndarray]) -> mx.array:
        """Compute pairwise similarities between agent states."""
        # Stack agent states
        states = mx.stack([mx.array(s) for s in agent_states])
        
        # Compute similarity matrix
        similarities = text_embedding_similarity(states, states)
        
        return similarities
    
    @gpuify(batch_size=8)
    async def update_agent_embeddings(self, 
                                    embeddings: List[np.ndarray],
                                    gradients: List[np.ndarray],
                                    learning_rate: float = 0.001) -> List[mx.array]:
        """Update agent embeddings using gradients."""
        updated = []
        
        for emb, grad in zip(embeddings, gradients):
            emb_mx = mx.array(emb) if isinstance(emb, np.ndarray) else emb
            grad_mx = mx.array(grad) if isinstance(grad, np.ndarray) else grad
            
            # Gradient descent update
            new_emb = emb_mx - learning_rate * grad_mx
            mx.eval(new_emb)
            updated.append(new_emb)
        
        return updated
    
    @gpuify
    async def consensus_voting(self, agent_outputs: List[np.ndarray]) -> mx.array:
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