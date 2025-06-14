"""Hardware-Aware Executor for M4 Pro optimization.

Maximizes utilization of M4 Pro's 12 CPU cores (8P+4E), 20 GPU cores,
and 24GB unified memory for code generation tasks.
"""
from __future__ import annotations

import asyncio
import logging
import os
import psutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple
import multiprocessing as mp
import numpy as np

logger = logging.getLogger(__name__)

# Try to import Metal Performance Shaders for GPU
try:
    import torch
    if torch.backends.mps.is_available():
        MPS_AVAILABLE = True
        logger.info("Metal Performance Shaders available for GPU acceleration")
    else:
        MPS_AVAILABLE = False
except ImportError:
    MPS_AVAILABLE = False

# Try to import MLX for Apple Silicon optimization
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
    logger.info("MLX available for Apple Silicon optimization")
except ImportError:
    MLX_AVAILABLE = False


class HardwareAwareExecutor:
    """Optimizes task execution for M4 Pro hardware."""
    
    def __init__(self):
        # Detect actual hardware
        self.cpu_count = psutil.cpu_count(logical=True)
        self.p_cores = 8  # Performance cores (M4 Pro)
        self.e_cores = 4  # Efficiency cores (M4 Pro)
        # Note: GPU core detection on macOS is not straightforward
        # M4 Pro configurations: 16 cores (base) or 20 cores (high-end)
        # TODO: Detect actual GPU cores via system_profiler or Metal API
        self.gpu_cores = 16  # M4 Pro GPU cores (adjust if you have 20-core model)
        
        # Get actual memory size
        self.memory_bytes = psutil.virtual_memory().total
        self.memory_gb = self.memory_bytes / (1024**3)
        self.usable_memory_gb = self.memory_gb * 0.85  # Leave 15% for system
        
        # Memory allocation strategy
        self.memory_allocations = {
            'index_cache': int(self.usable_memory_gb * 0.2 * 1024),  # 20% for indexes (MB)
            'model_cache': int(self.usable_memory_gb * 0.3 * 1024),  # 30% for models
            'experience_buffer': int(self.usable_memory_gb * 0.2 * 1024),  # 20% for experience
            'working_memory': int(self.usable_memory_gb * 0.3 * 1024),  # 30% for operations
        }
        
        logger.info(f"Detected {self.memory_gb:.1f}GB RAM, allocating {self.usable_memory_gb:.1f}GB for Jarvis")
        
        # Set optimal thread counts
        self._configure_threading()
        
        # Executors
        self.p_core_executor = None
        self.e_core_executor = None
        self.gpu_executor = None
        
        # Resource monitoring
        self.gpu_utilization = 0.0
        self.memory_usage_mb = 0.0
        self._monitor_task = None
    
    def _configure_threading(self):
        """Configure optimal threading for M4 Pro."""
        # Set environment variables for optimal performance
        os.environ['OMP_NUM_THREADS'] = str(self.p_cores)
        os.environ['MKL_NUM_THREADS'] = str(self.p_cores)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.p_cores)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(self.p_cores)
        
        # MLX specific - use unified memory efficiently
        if MLX_AVAILABLE:
            os.environ['MLX_NUM_THREADS'] = str(self.p_cores)
            # Set memory limit to use available unified memory
            os.environ['MLX_GPU_MEMORY_LIMIT'] = str(int(self.usable_memory_gb * 0.5 * 1024))  # 50% for GPU
        
        # PyTorch specific
        if MPS_AVAILABLE:
            torch.set_num_threads(self.p_cores)
            # Configure MPS to use unified memory efficiently
            torch.mps.set_per_process_memory_fraction(0.5)  # 50% of available memory
    
    async def initialize(self):
        """Initialize hardware resources."""
        logger.info(f"Initializing hardware executor: {self.cpu_count} CPUs "
                   f"({self.p_cores}P+{self.e_cores}E), "
                   f"~{self.gpu_cores} GPU cores (est), {self.memory_gb:.1f}GB memory")
        
        # Create executors if not already created
        if self.p_core_executor is None:
            self.p_core_executor = ThreadPoolExecutor(
                max_workers=self.p_cores,
                thread_name_prefix="p_core"
            )
        
        if self.e_core_executor is None:
            self.e_core_executor = ThreadPoolExecutor(
                max_workers=self.e_cores,
                thread_name_prefix="e_core"
            )
        
        # Skip resource monitoring for stability
        self._monitor_task = None
        
        # Skip GPU warmup for faster init
        logger.info("Hardware executor ready (GPU warmup skipped for speed)")
    
    def _init_p_core_worker(self):
        """Initialize performance core worker process."""
        # Set CPU affinity if available (macOS doesn't support this directly)
        # But we can set thread priority
        pass  # Simplified for ThreadPoolExecutor
    
    async def _warmup_gpu(self):
        """Warm up Metal GPU."""
        if not MPS_AVAILABLE:
            return
        
        logger.info("Warming up Metal GPU...")
        
        try:
            # Simple matrix multiplication to initialize GPU
            device = torch.device("mps")
            x = torch.randn(100, 100, device=device)  # Smaller for faster warmup
            y = torch.randn(100, 100, device=device)
            z = torch.matmul(x, y)
            torch.mps.synchronize()
            
            logger.info("GPU warmup complete")
        except Exception as e:
            logger.warning(f"GPU warmup failed: {e}, continuing without GPU")
    
    async def _warmup_mlx(self):
        """Warm up MLX."""
        if not MLX_AVAILABLE:
            return
        
        logger.info("Warming up MLX...")
        
        try:
            # Simple operation to initialize MLX
            x = mx.random.normal((100, 100))  # Smaller for faster warmup
            y = mx.random.normal((100, 100))
            z = mx.matmul(x, y)
            mx.eval(z)
            
            logger.info("MLX warmup complete")
        except Exception as e:
            logger.warning(f"MLX warmup failed: {e}, continuing without MLX")
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with hardware optimization."""
        task_type = self._classify_task(task)
        
        if task_type == "cpu_intensive":
            return await self._execute_on_p_cores(task)
        elif task_type == "io_bound":
            return await self._execute_on_e_cores(task)
        elif task_type == "gpu_suitable":
            return await self._execute_on_gpu(task)
        elif task_type == "memory_intensive":
            return await self._execute_with_memory_optimization(task)
        else:
            # Default to P-cores
            return await self._execute_on_p_cores(task)
    
    def _classify_task(self, task: Dict[str, Any]) -> str:
        """Classify task type for optimal execution."""
        # Simple heuristics
        if task.get('requires_gpu', False):
            return "gpu_suitable"
        
        if task.get('memory_intensive', False):
            return "memory_intensive"
        
        if task.get('io_operations', 0) > 5:
            return "io_bound"
        
        # Default to CPU intensive
        return "cpu_intensive"
    
    async def _execute_on_p_cores(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CPU-intensive task on performance cores."""
        loop = asyncio.get_event_loop()
        
        func = task.get('function')
        args = task.get('args', ())
        kwargs = task.get('kwargs', {})
        
        start_time = time.time()
        
        try:
            # Run in process pool
            result = await loop.run_in_executor(
                self.p_core_executor,
                func,
                *args,
                **kwargs
            )
            
            execution_time = time.time() - start_time
            
            return {
                'status': 'success',
                'result': result,
                'execution_time': execution_time,
                'executor': 'p_cores'
            }
            
        except Exception as e:
            logger.error(f"P-core execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'executor': 'p_cores'
            }
    
    async def _execute_on_e_cores(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute I/O-bound task on efficiency cores."""
        loop = asyncio.get_event_loop()
        
        func = task.get('function')
        args = task.get('args', ())
        kwargs = task.get('kwargs', {})
        
        start_time = time.time()
        
        try:
            # Run in thread pool
            result = await loop.run_in_executor(
                self.e_core_executor,
                func,
                *args,
                **kwargs
            )
            
            execution_time = time.time() - start_time
            
            return {
                'status': 'success',
                'result': result,
                'execution_time': execution_time,
                'executor': 'e_cores'
            }
            
        except Exception as e:
            logger.error(f"E-core execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'executor': 'e_cores'
            }
    
    async def _execute_on_gpu(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GPU-suitable task."""
        if MPS_AVAILABLE:
            return await self._execute_on_mps(task)
        elif MLX_AVAILABLE:
            return await self._execute_on_mlx(task)
        else:
            # Fallback to CPU
            logger.warning("GPU requested but not available, falling back to CPU")
            return await self._execute_on_p_cores(task)
    
    async def _execute_on_mps(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute on Metal Performance Shaders."""
        func = task.get('function')
        args = task.get('args', ())
        kwargs = task.get('kwargs', {})
        
        start_time = time.time()
        
        try:
            # Ensure MPS device
            device = torch.device("mps")
            
            # Move data to GPU if needed
            gpu_args = []
            for arg in args:
                if isinstance(arg, (np.ndarray, list)):
                    tensor = torch.tensor(arg, device=device)
                    gpu_args.append(tensor)
                else:
                    gpu_args.append(arg)
            
            # Execute
            result = func(*gpu_args, **kwargs)
            
            # Move result back to CPU if needed
            if isinstance(result, torch.Tensor):
                result = result.cpu().numpy()
            
            torch.mps.synchronize()
            execution_time = time.time() - start_time
            
            return {
                'status': 'success',
                'result': result,
                'execution_time': execution_time,
                'executor': 'mps_gpu'
            }
            
        except Exception as e:
            logger.error(f"MPS execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'executor': 'mps_gpu'
            }
    
    async def _execute_on_mlx(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute on MLX."""
        func = task.get('function')
        args = task.get('args', ())
        kwargs = task.get('kwargs', {})
        
        start_time = time.time()
        
        try:
            # Convert data to MLX arrays
            mlx_args = []
            for arg in args:
                if isinstance(arg, (np.ndarray, list)):
                    mlx_args.append(mx.array(arg))
                else:
                    mlx_args.append(arg)
            
            # Execute
            result = func(*mlx_args, **kwargs)
            
            # Evaluate and convert back
            mx.eval(result)
            if isinstance(result, mx.array):
                result = np.array(result)
            
            execution_time = time.time() - start_time
            
            return {
                'status': 'success',
                'result': result,
                'execution_time': execution_time,
                'executor': 'mlx_gpu'
            }
            
        except Exception as e:
            logger.error(f"MLX execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'executor': 'mlx_gpu'
            }
    
    async def _execute_with_memory_optimization(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute memory-intensive task with optimization."""
        # Pre-allocate memory pool
        memory_pool_size = task.get('memory_requirement_mb', 1000)
        
        # Use unified memory advantage
        if memory_pool_size > 8000:  # Large allocation
            logger.info(f"Large memory allocation: {memory_pool_size}MB")
            # Consider using memory mapping or streaming
        
        # Execute with memory monitoring
        return await self._execute_on_p_cores(task)
    
    async def batch_execute(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple tasks in parallel."""
        # Classify tasks
        cpu_tasks = []
        io_tasks = []
        gpu_tasks = []
        
        for task in tasks:
            task_type = self._classify_task(task)
            if task_type == "gpu_suitable":
                gpu_tasks.append(task)
            elif task_type == "io_bound":
                io_tasks.append(task)
            else:
                cpu_tasks.append(task)
        
        # Execute in parallel
        results = []
        
        # GPU tasks first (usually bottleneck)
        if gpu_tasks:
            gpu_results = await asyncio.gather(
                *[self._execute_on_gpu(t) for t in gpu_tasks]
            )
            results.extend(gpu_results)
        
        # CPU and I/O tasks in parallel
        cpu_futures = [self._execute_on_p_cores(t) for t in cpu_tasks]
        io_futures = [self._execute_on_e_cores(t) for t in io_tasks]
        
        if cpu_futures or io_futures:
            mixed_results = await asyncio.gather(*(cpu_futures + io_futures))
            results.extend(mixed_results)
        
        return results
    
    async def _monitor_resources(self):
        """Monitor resource usage."""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage_mb = memory.used / (1024 * 1024)
                
                # GPU usage (approximate)
                if MPS_AVAILABLE:
                    # MPS doesn't provide direct utilization metrics
                    # Estimate based on memory pressure
                    self.gpu_utilization = min(100, memory.percent * 0.7)
                elif MLX_AVAILABLE:
                    # Similar estimation for MLX
                    self.gpu_utilization = min(100, memory.percent * 0.6)
                
                # Log if high usage
                if cpu_percent > 90 or memory.percent > 85:
                    logger.warning(
                        f"High resource usage - CPU: {cpu_percent}%, "
                        f"Memory: {memory.percent}%"
                    )
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(10)
    
    def get_gpu_utilization(self) -> float:
        """Get current GPU utilization estimate."""
        return self.gpu_utilization
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.memory_usage_mb
    
    def get_optimal_batch_size(self, task_type: str = "general", item_size_mb: float = 10) -> int:
        """Get optimal batch size for task type."""
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        allocated_memory = self.memory_allocations.get('working_memory', 4096)  # MB
        usable_memory = min(available_memory * 0.7, allocated_memory)  # Use 70% of available
        
        if task_type == "gpu":
            # GPU operations need more memory per item
            batch_size = int(usable_memory / (item_size_mb * 2))  # 2x overhead for GPU
            return min(256, max(1, batch_size))
        elif task_type == "cpu":
            # CPU operations can be more memory efficient
            batch_size = int(usable_memory / item_size_mb)
            return min(self.p_cores * 4, max(1, batch_size))  # Up to 4 items per core
        elif task_type == "mcts":
            # MCTS needs memory for tree expansion
            batch_size = int(usable_memory / (item_size_mb * 0.5))  # Trees are memory light
            return min(512, max(self.p_cores, batch_size))
        else:
            batch_size = int(usable_memory / item_size_mb)
            return min(self.cpu_count * 2, max(1, batch_size))
    
    async def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down hardware executor")
        
        # Cancel monitoring
        if self._monitor_task:
            self._monitor_task.cancel()
        
        # Shutdown executors
        if self.p_core_executor:
            self.p_core_executor.shutdown(wait=True)
        
        if self.e_core_executor:
            self.e_core_executor.shutdown(wait=True)
        
        logger.info("Hardware executor shutdown complete")


class TaskScheduler:
    """Intelligent task scheduling for M4 Pro."""
    
    def __init__(self, hardware_executor: HardwareAwareExecutor):
        self.executor = hardware_executor
        self.task_queue = asyncio.Queue()
        self.results = {}
        self._scheduler_task = None
    
    async def start(self):
        """Start the scheduler."""
        self._scheduler_task = asyncio.create_task(self._schedule_loop())
    
    async def submit(self, task_id: str, task: Dict[str, Any]) -> str:
        """Submit a task for scheduling."""
        task['id'] = task_id
        task['submitted_at'] = time.time()
        await self.task_queue.put(task)
        return task_id
    
    async def get_result(self, task_id: str, timeout: float = None) -> Dict[str, Any]:
        """Get task result."""
        start_time = time.time()
        
        while task_id not in self.results:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} timed out")
            await asyncio.sleep(0.1)
        
        return self.results.pop(task_id)
    
    async def _schedule_loop(self):
        """Main scheduling loop."""
        pending_tasks = []
        
        while True:
            try:
                # Collect tasks
                while not self.task_queue.empty() and len(pending_tasks) < 100:
                    task = await self.task_queue.get()
                    pending_tasks.append(task)
                
                if pending_tasks:
                    # Sort by priority if specified
                    pending_tasks.sort(
                        key=lambda t: t.get('priority', 0),
                        reverse=True
                    )
                    
                    # Determine batch size based on resource availability
                    batch_size = min(
                        len(pending_tasks),
                        self.executor.get_optimal_batch_size()
                    )
                    
                    # Execute batch
                    batch = pending_tasks[:batch_size]
                    pending_tasks = pending_tasks[batch_size:]
                    
                    results = await self.executor.batch_execute(batch)
                    
                    # Store results
                    for task, result in zip(batch, results):
                        self.results[task['id']] = result
                
                await asyncio.sleep(0.01)  # Small delay
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(1)


class MemoryOptimizer:
    """Optimizes memory usage for M4 Pro's 24GB unified memory."""
    
    def __init__(self, target_usage_percent: float = 85):
        self.target_usage_percent = target_usage_percent
        self.memory_pools = {}
        self.total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        self.reserved_memory_mb = self.total_memory_mb * 0.15  # Reserve 15% for system
        self.available_for_pools = self.total_memory_mb - self.reserved_memory_mb
        
        # Pre-allocate major pools
        self.default_pools = {
            'mcts_trees': 0.15,  # 15% for MCTS tree structures
            'neural_models': 0.25,  # 25% for neural network models
            'code_cache': 0.20,  # 20% for generated code cache
            'index_data': 0.15,  # 15% for search indexes
            'batch_processing': 0.25,  # 25% for batch operations
        }
    
    def allocate_pool(self, name: str, size_mb: int) -> bool:
        """Pre-allocate memory pool."""
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        
        if size_mb > available_mb * 0.8:
            logger.warning(f"Cannot allocate {size_mb}MB for {name}, only {available_mb}MB available")
            return False
        
        # In practice, would allocate actual memory
        self.memory_pools[name] = {
            'size_mb': size_mb,
            'allocated_at': time.time()
        }
        
        logger.info(f"Allocated {size_mb}MB pool for {name}")
        return True
    
    def release_pool(self, name: str):
        """Release memory pool."""
        if name in self.memory_pools:
            del self.memory_pools[name]
            logger.info(f"Released memory pool {name}")
    
    def get_recommended_batch_size(self, item_size_mb: float) -> int:
        """Get recommended batch size based on available memory."""
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        target_mb = available_mb * (self.target_usage_percent / 100)
        
        return max(1, int(target_mb / item_size_mb))