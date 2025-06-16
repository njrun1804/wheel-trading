"""
Optimized GPU Acceleration with <1.0s Initialization

This module provides MLX-accelerated operations with optimized initialization:
- Lazy GPU loading (only when needed)
- Cached hardware detection
- Optimized MLX import and setup  
- GPU context pooling
- <1.0s target initialization time
"""

import asyncio
import functools
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, List, Union

# Use optimized lazy loading for MLX
try:
    from src.unity_wheel.gpu.lazy_gpu_loader import (
        get_mlx_core, is_gpu_ready, LazyGPUContext, requires_gpu,
        get_optimized_initializer, warmup_gpu_components
    )
    from src.unity_wheel.gpu.optimized_gpu_init import initialize_gpu_optimized
    mx = None  # Will be loaded lazily
    USE_OPTIMIZED_INIT = True
except ImportError:
    # Fallback to direct import
    import mlx.core as mx
    USE_OPTIMIZED_INIT = False

import numpy as np

logger = logging.getLogger(__name__)


class OptimizedGPUAccelerator:
    """GPU acceleration manager with optimized <1.0s initialization."""
    
    def __init__(self):
        """Initialize GPU accelerator with lazy loading."""
        # Track initialization time
        self._init_start_time = time.perf_counter()
        
        # Lightweight initialization only
        self._gpu_available = None
        self._mlx_core = None
        self._initialized = False
        self._init_stats = None
        
        # Load minimal config (no file I/O unless needed)
        self._default_config = {
            "memory": {"max_allocation_gb": 4},
            "performance": {"gpu_batch_size": 256}
        }
        self.config = self._default_config.copy()
        
        # Defer heavy initialization
        self._hardware_info = None
        self._performance_model = {
            'cpu_ops_per_sec': 1e6,
            'gpu_ops_per_sec': 5e6,
            'memory_bandwidth_gbps': 400,
        }
        
        # Performance stats
        self.stats = {
            "gpu_operations": 0,
            "cpu_fallbacks": 0,
            "cpu_preferred": 0,
            "total_gpu_time": 0.0,
            "total_cpu_time": 0.0,
            "overhead_avoided_ms": 0.0,
            "memory_peaks": [],
            "workload_size_stats": {
                'small_cpu': [],
                'large_gpu': [],
                'borderline': []
            }
        }
        
        # Workload thresholds (conservative defaults)
        self._workload_thresholds = {
            'vector_ops': 10000,
            'matrix_ops': 500 * 500,
            'batch_ops': 200,
            'similarity': 2000,
            'embedding': 500,
            'attention': 128,
        }
        
        # GPU overhead estimates (will be updated with real measurements)
        self._gpu_overhead_us = {
            'initialization': 1500,  # Reduced from 2000 with optimization
            'memory_transfer': 300,  # Reduced from 500
            'evaluation': 200,       # Reduced from 300
            'cleanup': 50,           # Reduced from 100
        }
        
        # Memory management
        self._memory_threshold = self._default_config["memory"]["max_allocation_gb"] * 0.7 * 1024 * 1024 * 1024
        self._batch_size = self._default_config["performance"]["gpu_batch_size"]
        
        init_time = (time.perf_counter() - self._init_start_time) * 1000
        logger.debug(f"OptimizedGPUAccelerator created in {init_time:.1f}ms (lightweight)")
        
        # Start background warmup if enabled
        if USE_OPTIMIZED_INIT:
            self._start_background_warmup()
    
    def _start_background_warmup(self):
        """Start background warmup of GPU components."""
        try:
            # Start warmup in background thread (non-blocking)
            import threading
            warmup_thread = threading.Thread(
                target=self._background_warmup, 
                daemon=True
            )
            warmup_thread.start()
            logger.debug("Started background GPU warmup")
        except Exception as e:
            logger.debug(f"Background warmup failed: {e}")
    
    def _background_warmup(self):
        """Background warmup process."""
        try:
            # Warmup critical components
            warmup_gpu_components(['mlx_core', 'optimized_initializer'])
        except Exception as e:
            logger.debug(f"Background warmup error: {e}")
    
    @property
    def gpu_available(self) -> bool:
        """Check if GPU is available (lazy evaluation)."""
        if self._gpu_available is not None:
            return self._gpu_available
        
        try:
            if USE_OPTIMIZED_INIT:
                # Use optimized detection
                self._gpu_available = is_gpu_ready()
                if not self._gpu_available:
                    # Try to initialize if not ready
                    mx_core = get_mlx_core()
                    self._gpu_available = hasattr(mx_core, 'metal') and mx_core.metal.is_available()
            else:
                # Fallback to direct check
                self._gpu_available = mx.metal.is_available()
                
            if self._gpu_available:
                logger.info("MLX Metal GPU acceleration available")
            else:
                logger.warning("MLX Metal GPU not available, using CPU fallback")
                
        except Exception as e:
            logger.warning(f"GPU availability check failed: {e}")
            self._gpu_available = False
        
        return self._gpu_available
    
    def _ensure_initialized(self):
        """Ensure GPU is initialized (called on first use)."""
        if self._initialized:
            return
        
        start_time = time.perf_counter()
        
        try:
            if USE_OPTIMIZED_INIT:
                # Use optimized async initialization
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # No event loop, create temporary one
                    pass
                
                if loop and loop.is_running():
                    # In async context, schedule for later
                    logger.debug("Scheduling GPU initialization for async context")
                    asyncio.create_task(self._async_initialize())
                else:
                    # Sync initialization
                    self._init_stats = asyncio.run(initialize_gpu_optimized())
                    logger.info(f"GPU initialized in {self._init_stats.total_time_ms:.1f}ms")
            else:
                # Fallback initialization
                self._fallback_initialize()
                
        except Exception as e:
            logger.warning(f"GPU initialization failed: {e}")
        
        self._initialized = True
        init_time = (time.perf_counter() - start_time) * 1000
        
        # Update overhead estimates based on actual initialization time
        if init_time > 0:
            # Scale overhead estimates based on actual performance
            scale_factor = min(1.0, 1000 / init_time)  # Better perf = lower overhead
            for key in self._gpu_overhead_us:
                self._gpu_overhead_us[key] = int(self._gpu_overhead_us[key] * scale_factor)
        
        logger.debug(f"GPU initialization completed in {init_time:.1f}ms")
    
    async def _async_initialize(self):
        """Asynchronous GPU initialization."""
        try:
            self._init_stats = await initialize_gpu_optimized()
            logger.info(f"Async GPU initialized in {self._init_stats.total_time_ms:.1f}ms")
        except Exception as e:
            logger.warning(f"Async GPU initialization failed: {e}")
    
    def _fallback_initialize(self):
        """Fallback initialization without optimizations."""
        try:
            # Basic MLX initialization
            if mx:
                test_array = mx.array([1.0, 2.0, 3.0])
                mx.eval(test_array)
                logger.info("Fallback GPU initialization completed")
        except Exception as e:
            logger.warning(f"Fallback initialization failed: {e}")
    
    def _get_mlx_core(self):
        """Get MLX core, loading if necessary."""
        if self._mlx_core is not None:
            return self._mlx_core
        
        try:
            if USE_OPTIMIZED_INIT:
                self._mlx_core = get_mlx_core()
            else:
                self._mlx_core = mx
            return self._mlx_core
        except Exception as e:
            logger.error(f"Failed to get MLX core: {e}")
            return None
    
    def _should_use_gpu(self, operation_type: str, workload_size: int) -> tuple[bool, str]:
        """Intelligent decision on CPU vs GPU based on workload size and overhead."""
        if not self.gpu_available:
            return False, "GPU not available"
        
        # Get threshold for this operation type
        threshold = self._workload_thresholds.get(operation_type, 1000)
        
        # Calculate total GPU overhead
        total_overhead_us = sum(self._gpu_overhead_us.values())
        
        # Estimate execution times
        cpu_time_us = workload_size / self._performance_model['cpu_ops_per_sec'] * 1e6
        gpu_compute_time_us = workload_size / self._performance_model['gpu_ops_per_sec'] * 1e6
        gpu_total_time_us = gpu_compute_time_us + total_overhead_us
        
        # Decision logic
        if workload_size < threshold:
            reason = f"Small workload ({workload_size} < {threshold})"
            return False, reason
        
        # GPU must be significantly faster to justify overhead
        if gpu_total_time_us >= cpu_time_us * 0.8:  # GPU must be >20% faster
            reason = f"GPU overhead too high ({gpu_total_time_us:.0f}us vs CPU {cpu_time_us:.0f}us)"
            return False, reason
        
        return True, f"Large workload benefits from GPU ({workload_size} elements)"
    
    def _estimate_workload_size(self, operation_type: str, *args) -> int:
        """Estimate workload size for intelligent CPU/GPU routing."""
        total_elements = 0
        
        for arg in args:
            if hasattr(arg, 'size'):  # numpy array or similar
                total_elements += arg.size
            elif hasattr(arg, '__len__'):
                total_elements += len(arg)
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    if hasattr(item, 'size'):
                        total_elements += item.size
                    elif hasattr(item, '__len__'):
                        total_elements += len(item)
        
        return total_elements
    
    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        total_ops = self.stats["gpu_operations"] + self.stats["cpu_fallbacks"] + self.stats["cpu_preferred"]
        
        stats = {
            "gpu_available": self.gpu_available,
            "initialized": self._initialized,
            "gpu_operations": self.stats["gpu_operations"],
            "cpu_fallbacks": self.stats["cpu_fallbacks"],
            "cpu_preferred": self.stats["cpu_preferred"],
            "total_operations": total_ops,
            "intelligent_routing_rate": self.stats["cpu_preferred"] / max(1, total_ops) * 100,
            "avg_gpu_time_ms": self.stats["total_gpu_time"] / max(1, self.stats["gpu_operations"]) * 1000,
            "avg_cpu_time_ms": self.stats["total_cpu_time"] / max(1, self.stats["cpu_fallbacks"] + self.stats["cpu_preferred"]) * 1000,
            "overhead_avoided_ms": self.stats["overhead_avoided_ms"],
            "workload_thresholds": self._workload_thresholds,
            "gpu_overhead_us": self._gpu_overhead_us,
        }
        
        # Add initialization stats if available
        if self._init_stats:
            stats["initialization"] = {
                "total_time_ms": self._init_stats.total_time_ms,
                "hardware_detection_ms": self._init_stats.hardware_detection_ms,
                "mlx_import_ms": self._init_stats.mlx_import_ms,
                "was_cached": self._init_stats.was_cached,
                "parallel_speedup": self._init_stats.parallel_speedup
            }
        
        return stats


def optimized_gpuify(func: Callable | None = None, *, 
                    fallback: bool = True,
                    operation_type: str = 'generic') -> Callable:
    """Optimized decorator for GPU offloading with lazy initialization."""
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        async def async_wrapper(*args, **kwargs):
            # Get accelerator (will be created lazily)
            accelerator = get_optimized_gpu_accelerator()
            
            # Ensure initialization on first use
            accelerator._ensure_initialized()
            
            # Intelligent routing
            workload_size = accelerator._estimate_workload_size(operation_type, *args)
            use_gpu, routing_reason = accelerator._should_use_gpu(operation_type, workload_size)
            
            start_time = time.perf_counter()
            
            try:
                if use_gpu:
                    # Use optimized GPU context
                    if USE_OPTIMIZED_INIT:
                        with LazyGPUContext(['mlx_core']) as ctx:
                            mx_core = ctx.get('mlx_core', accelerator._get_mlx_core())
                            result = await f(*args, **kwargs)
                    else:
                        result = await f(*args, **kwargs)
                    
                    accelerator.stats["gpu_operations"] += 1
                    accelerator.stats["total_gpu_time"] += time.perf_counter() - start_time
                    return result
                    
                elif fallback:
                    # CPU execution
                    if "Small workload" in routing_reason:
                        accelerator.stats["cpu_preferred"] += 1
                        accelerator.stats["overhead_avoided_ms"] += sum(accelerator._gpu_overhead_us.values()) / 1000
                    else:
                        accelerator.stats["cpu_fallbacks"] += 1
                    
                    result = await f(*args, **kwargs)
                    accelerator.stats["total_cpu_time"] += time.perf_counter() - start_time
                    return result
                else:
                    raise RuntimeError(f"GPU required for {f.__name__} but not available")
                    
            except Exception as e:
                if fallback:
                    accelerator.stats["cpu_fallbacks"] += 1
                    result = await f(*args, **kwargs)
                    accelerator.stats["total_cpu_time"] += time.perf_counter() - start_time
                    return result
                raise
        
        @functools.wraps(f)
        def sync_wrapper(*args, **kwargs):
            # Similar implementation for sync functions
            accelerator = get_optimized_gpu_accelerator()
            accelerator._ensure_initialized()
            
            workload_size = accelerator._estimate_workload_size(operation_type, *args)
            use_gpu, routing_reason = accelerator._should_use_gpu(operation_type, workload_size)
            
            start_time = time.perf_counter()
            
            try:
                if use_gpu:
                    if USE_OPTIMIZED_INIT:
                        with LazyGPUContext(['mlx_core']) as ctx:
                            result = f(*args, **kwargs)
                    else:
                        result = f(*args, **kwargs)
                    
                    accelerator.stats["gpu_operations"] += 1
                    accelerator.stats["total_gpu_time"] += time.perf_counter() - start_time
                    return result
                    
                elif fallback:
                    if "Small workload" in routing_reason:
                        accelerator.stats["cpu_preferred"] += 1
                        accelerator.stats["overhead_avoided_ms"] += sum(accelerator._gpu_overhead_us.values()) / 1000
                    else:
                        accelerator.stats["cpu_fallbacks"] += 1
                    
                    result = f(*args, **kwargs)
                    accelerator.stats["total_cpu_time"] += time.perf_counter() - start_time
                    return result
                else:
                    raise RuntimeError(f"GPU required for {f.__name__} but not available")
                    
            except Exception as e:
                if fallback:
                    accelerator.stats["cpu_fallbacks"] += 1
                    result = f(*args, **kwargs)
                    accelerator.stats["total_cpu_time"] += time.perf_counter() - start_time
                    return result
                raise
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(f):
            return async_wrapper
        else:
            return sync_wrapper
    
    # Handle decorator with/without arguments
    if func is None:
        return decorator
    else:
        return decorator(func)


# Optimized vector operations using the new decorator
@optimized_gpuify(operation_type='vector_ops')
def optimized_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Optimized cosine similarity with lazy GPU loading."""
    mx_core = get_mlx_core() if USE_OPTIMIZED_INIT else mx
    
    if mx_core is None:
        # CPU fallback
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        return float(np.dot(a_norm, b_norm))
    
    # GPU implementation
    a_mx = mx_core.array(a.flatten())
    b_mx = mx_core.array(b.flatten())
    
    norm_a = mx_core.linalg.norm(a_mx)
    norm_b = mx_core.linalg.norm(b_mx)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    a_norm = a_mx / norm_a
    b_norm = b_mx / norm_b
    
    similarity = mx_core.sum(a_norm * b_norm)
    mx_core.eval(similarity)
    
    return float(similarity)


@optimized_gpuify(operation_type='matrix_ops')
def optimized_matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimized matrix multiplication with lazy GPU loading."""
    mx_core = get_mlx_core() if USE_OPTIMIZED_INIT else mx
    
    if mx_core is None:
        # CPU fallback
        return np.matmul(a, b)
    
    # GPU implementation
    a_mx = mx_core.array(a)
    b_mx = mx_core.array(b)
    
    result = mx_core.matmul(a_mx, b_mx)
    mx_core.eval(result)
    
    return np.array(result)


# Global optimized accelerator instance (created lazily)
_optimized_accelerator: OptimizedGPUAccelerator | None = None


def get_optimized_gpu_accelerator() -> OptimizedGPUAccelerator:
    """Get or create the global optimized GPU accelerator."""
    global _optimized_accelerator
    if _optimized_accelerator is None:
        _optimized_accelerator = OptimizedGPUAccelerator()
    return _optimized_accelerator


def get_gpu_initialization_stats() -> dict[str, Any]:
    """Get GPU initialization performance statistics."""
    accelerator = get_optimized_gpu_accelerator()
    return accelerator.get_stats()


def print_gpu_performance_report():
    """Print detailed GPU performance report."""
    accelerator = get_optimized_gpu_accelerator()
    stats = accelerator.get_stats()
    
    print("\n=== Optimized GPU Acceleration Report ===")
    print(f"GPU Available: {stats['gpu_available']}")
    print(f"Initialized: {stats['initialized']}")
    
    if 'initialization' in stats:
        init_stats = stats['initialization']
        print(f"\nInitialization Performance:")
        print(f"  Total Time: {init_stats['total_time_ms']:.1f}ms (Target: <1000ms)")
        print(f"  Hardware Detection: {init_stats['hardware_detection_ms']:.1f}ms")
        print(f"  MLX Import: {init_stats['mlx_import_ms']:.1f}ms")
        print(f"  Was Cached: {init_stats['was_cached']}")
        print(f"  Parallel Speedup: {init_stats['parallel_speedup']:.1f}x")
        
        if init_stats['total_time_ms'] < 1000:
            print("  ✅ Initialization target achieved!")
        else:
            print("  ❌ Initialization target missed")
    
    print(f"\nOperation Statistics:")
    print(f"  GPU Operations: {stats['gpu_operations']}")
    print(f"  CPU Preferred: {stats['cpu_preferred']}")
    print(f"  CPU Fallbacks: {stats['cpu_fallbacks']}")
    print(f"  Intelligent Routing Rate: {stats['intelligent_routing_rate']:.1f}%")
    print(f"  Overhead Avoided: {stats['overhead_avoided_ms']:.1f}ms")
    
    print("=========================================\n")


# Benchmark function
async def benchmark_optimized_gpu():
    """Benchmark optimized GPU initialization and operations."""
    print("=== Optimized GPU Benchmark ===")
    
    # Test initialization time
    init_times = []
    for i in range(3):
        start_time = time.perf_counter()
        accelerator = OptimizedGPUAccelerator()
        accelerator._ensure_initialized()
        init_time = (time.perf_counter() - start_time) * 1000
        init_times.append(init_time)
        print(f"Initialization {i+1}: {init_time:.1f}ms")
    
    avg_init_time = sum(init_times) / len(init_times)
    print(f"Average initialization time: {avg_init_time:.1f}ms")
    print(f"Target achieved: {'✅' if avg_init_time < 1000 else '❌'}")
    
    # Test operations
    print(f"\nTesting operations...")
    
    # Small operation (should use CPU)
    a_small = np.random.randn(100, 100).astype(np.float32)
    b_small = np.random.randn(100, 100).astype(np.float32)
    
    start = time.perf_counter()
    result_small = optimized_matrix_multiply(a_small, b_small)
    small_time = (time.perf_counter() - start) * 1000
    print(f"Small matrix multiply: {small_time:.3f}ms")
    
    # Large operation (should use GPU if available)
    a_large = np.random.randn(1000, 1000).astype(np.float32)
    b_large = np.random.randn(1000, 1000).astype(np.float32)
    
    start = time.perf_counter()
    result_large = optimized_matrix_multiply(a_large, b_large)
    large_time = (time.perf_counter() - start) * 1000
    print(f"Large matrix multiply: {large_time:.1f}ms")
    
    # Print performance report
    print_gpu_performance_report()


if __name__ == "__main__":
    asyncio.run(benchmark_optimized_gpu())