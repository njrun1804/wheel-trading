"""Hardware-Aware Executor for M4 Pro optimization.

Maximizes utilization of M4 Pro's 12 CPU cores (8P+4E), 20 GPU cores,
and 24GB unified memory for code generation tasks.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import numpy as np
import psutil

logger = logging.getLogger(__name__)
try:
    import torch
    if torch.backends.mps.is_available():
        MPS_AVAILABLE = True
        logger.info('Metal Performance Shaders available for GPU acceleration')
    else:
        MPS_AVAILABLE = False
except ImportError:
    MPS_AVAILABLE = False
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
    logger.info('MLX available for Apple Silicon optimization')
except ImportError:
    MLX_AVAILABLE = False


class HardwareAwareExecutor:
    """Optimizes task execution for M4 Pro hardware."""

    def __init__(self):
        from .m4_detector import get_detector
        detector = get_detector()
        hw_info = detector.detect_hardware()
        self.cpu_count = hw_info['cpu']['logical_cores']
        self.p_cores = hw_info['cpu']['p_cores']
        self.e_cores = hw_info['cpu']['e_cores']
        self.gpu_cores = hw_info['gpu']['cores']
        self.is_apple_silicon = hw_info['apple_silicon']
        self.metal_supported = hw_info['gpu']['metal_supported']
        logger.info(
            f"Detected {hw_info['model'].get('chip', 'CPU')} with {self.p_cores}P+{self.e_cores}E cores, {self.gpu_cores} GPU cores"
            )
        self.memory_bytes = psutil.virtual_memory().total
        self.memory_gb = self.memory_bytes / 1024 ** 3
        self.usable_memory_gb = self.memory_gb * 0.85
        self.memory_allocations = {'index_cache': int(self.usable_memory_gb *
            0.2 * 1024), 'model_cache': int(self.usable_memory_gb * 0.3 * 
            1024), 'experience_buffer': int(self.usable_memory_gb * 0.2 * 
            1024), 'working_memory': int(self.usable_memory_gb * 0.3 * 1024)}
        logger.info(
            f'Detected {self.memory_gb:.1f}GB RAM, allocating {self.usable_memory_gb:.1f}GB for Jarvis'
            )
        self._configure_threading()
        self.p_core_executor = None
        self.e_core_executor = None
        self.gpu_executor = None
        self.gpu_utilization = 0.0
        self.memory_usage_mb = 0.0
        self._monitor_task = None

    def _configure_threading(self):
        """Configure optimal threading for M4 Pro."""
        os.environ['OMP_NUM_THREADS'] = str(self.p_cores)
        os.environ['MKL_NUM_THREADS'] = str(self.p_cores)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.p_cores)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(self.p_cores)
        if MLX_AVAILABLE:
            os.environ['MLX_NUM_THREADS'] = str(self.p_cores)
            os.environ['MLX_GPU_MEMORY_LIMIT'] = str(int(self.
                usable_memory_gb * 0.5 * 1024))
        if MPS_AVAILABLE:
            torch.set_num_threads(self.p_cores)
            torch.mps.set_per_process_memory_fraction(0.5)

    async def initialize(self):
        """Initialize hardware resources."""
        await asyncio.sleep(0)
        logger.info(
            f'Initializing hardware executor: {self.cpu_count} CPUs ({self.p_cores}P+{self.e_cores}E), ~{self.gpu_cores} GPU cores (est), {self.memory_gb:.1f}GB memory'
            )
        if self.p_core_executor is None:
            self.p_core_executor = ThreadPoolExecutor(max_workers=self.
                p_cores, thread_name_prefix='p_core')
        if self.e_core_executor is None:
            self.e_core_executor = ThreadPoolExecutor(max_workers=self.
                e_cores, thread_name_prefix='e_core')
        self._monitor_task = None
        logger.info('Hardware executor ready (GPU warmup skipped for speed)')

    def _init_p_core_worker(self):
        """Initialize performance core worker process."""

    def _warmup_gpu(self):
        """Warm up Metal GPU."""
        if not MPS_AVAILABLE:
            return
        logger.info('Warming up Metal GPU...')
        try:
            device = torch.device('mps')
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.matmul(x, y)
            torch.mps.synchronize()
            logger.info('GPU warmup complete')
        except Exception as e:
            logger.warning(f'GPU warmup failed: {e}, continuing without GPU')

    def _warmup_mlx(self):
        """Warm up MLX."""
        if not MLX_AVAILABLE:
            return
        logger.info('Warming up MLX...')
        try:
            x = mx.random.normal((100, 100))
            y = mx.random.normal((100, 100))
            z = mx.matmul(x, y)
            mx.eval(z)
            logger.info('MLX warmup complete')
        except Exception as e:
            logger.warning(f'MLX warmup failed: {e}, continuing without MLX')

    async def execute(self, task: Dict[str, Any]) ->Dict[str, Any]:
        """Execute task with hardware optimization."""
        task_type = self._classify_task(task)
        if task_type == 'cpu_intensive':
            return await self._execute_on_p_cores(task)
        elif task_type == 'io_bound':
            return await self._execute_on_e_cores(task)
        elif task_type == 'gpu_suitable':
            return await self._execute_on_gpu(task)
        elif task_type == 'memory_intensive':
            return await self._execute_with_memory_optimization(task)
        else:
            return await self._execute_on_p_cores(task)

    def _classify_task(self, task: Dict[str, Any]) ->str:
        """Classify task type for optimal execution."""
        if task.get('requires_gpu', False):
            return 'gpu_suitable'
        if task.get('memory_intensive', False):
            return 'memory_intensive'
        if task.get('io_operations', 0) > 5:
            return 'io_bound'
        return 'cpu_intensive'

    async def _execute_on_p_cores(self, task: Dict[str, Any]) ->Dict[str, Any]:
        """Execute CPU-intensive task on performance cores."""
        loop = asyncio.get_event_loop()
        func = task.get('function')
        args = task.get('args', ())
        kwargs = task.get('kwargs', {})
        start_time = time.time()
        try:
            result = await loop.run_in_executor(self.p_core_executor, func,
                *args, **kwargs)
            execution_time = time.time() - start_time
            return {'status': 'success', 'result': result, 'execution_time':
                execution_time, 'executor': 'p_cores'}
        except Exception as e:
            logger.error(f'P-core execution failed: {e}')
            return {'status': 'error', 'error': str(e), 'executor': 'p_cores'}

    async def _execute_on_e_cores(self, task: Dict[str, Any]) ->Dict[str, Any]:
        """Execute I/O-bound task on efficiency cores."""
        loop = asyncio.get_event_loop()
        func = task.get('function')
        args = task.get('args', ())
        kwargs = task.get('kwargs', {})
        start_time = time.time()
        try:
            result = await loop.run_in_executor(self.e_core_executor, func,
                *args, **kwargs)
            execution_time = time.time() - start_time
            return {'status': 'success', 'result': result, 'execution_time':
                execution_time, 'executor': 'e_cores'}
        except Exception as e:
            logger.error(f'E-core execution failed: {e}')
            return {'status': 'error', 'error': str(e), 'executor': 'e_cores'}

    async def _execute_on_gpu(self, task: Dict[str, Any]) ->Dict[str, Any]:
        """Execute GPU-suitable task."""
        if MPS_AVAILABLE:
            return await self._execute_on_mps(task)
        elif MLX_AVAILABLE:
            return await self._execute_on_mlx(task)
        else:
            logger.warning(
                'GPU requested but not available, falling back to CPU')
            return await self._execute_on_p_cores(task)

    async def _execute_on_mps(self, task: Dict[str, Any]) ->Dict[str, Any]:
        """Execute on Metal Performance Shaders."""
        await asyncio.sleep(0)
        func = task.get('function')
        args = task.get('args', ())
        kwargs = task.get('kwargs', {})
        start_time = time.time()
        try:
            device = torch.device('mps')
            gpu_args = []
            for arg in args:
                if isinstance(arg, (np.ndarray, list)):
                    tensor = torch.tensor(arg, device=device)
                    gpu_args.append(tensor)
                else:
                    gpu_args.append(arg)
            result = func(*gpu_args, **kwargs)
            if isinstance(result, torch.Tensor):
                result = result.cpu().numpy()
            torch.mps.synchronize()
            execution_time = time.time() - start_time
            return {'status': 'success', 'result': result, 'execution_time':
                execution_time, 'executor': 'mps_gpu'}
        except Exception as e:
            logger.error(f'MPS execution failed: {e}')
            return {'status': 'error', 'error': str(e), 'executor': 'mps_gpu'}

    async def _execute_on_mlx(self, task: Dict[str, Any]) ->Dict[str, Any]:
        """Execute on MLX."""
        await asyncio.sleep(0)
        func = task.get('function')
        args = task.get('args', ())
        kwargs = task.get('kwargs', {})
        start_time = time.time()
        try:
            mlx_args = []
            for arg in args:
                if isinstance(arg, (np.ndarray, list)):
                    mlx_args.append(mx.array(arg))
                else:
                    mlx_args.append(arg)
            result = func(*mlx_args, **kwargs)
            mx.eval(result)
            if isinstance(result, mx.array):
                result = np.array(result)
            execution_time = time.time() - start_time
            return {'status': 'success', 'result': result, 'execution_time':
                execution_time, 'executor': 'mlx_gpu'}
        except Exception as e:
            logger.error(f'MLX execution failed: {e}')
            return {'status': 'error', 'error': str(e), 'executor': 'mlx_gpu'}

    async def _execute_with_memory_optimization(self, task: Dict[str, Any]
        ) ->Dict[str, Any]:
        """Execute memory-intensive task with optimization."""
        memory_pool_size = task.get('memory_requirement_mb', 1000)
        if memory_pool_size > 8000:
            logger.info(f'Large memory allocation: {memory_pool_size}MB')
        return await self._execute_on_p_cores(task)

    async def batch_execute(self, tasks: List[Dict[str, Any]]) ->List[Dict[
        str, Any]]:
        """Execute multiple tasks in parallel."""
        cpu_tasks = []
        io_tasks = []
        gpu_tasks = []
        for task in tasks:
            task_type = self._classify_task(task)
            if task_type == 'gpu_suitable':
                gpu_tasks.append(task)
            elif task_type == 'io_bound':
                io_tasks.append(task)
            else:
                cpu_tasks.append(task)
        results = []
        if gpu_tasks:
            gpu_results = await asyncio.gather(*[self._execute_on_gpu(t) for
                t in gpu_tasks])
            results.extend(gpu_results)
        cpu_futures = [self._execute_on_p_cores(t) for t in cpu_tasks]
        io_futures = [self._execute_on_e_cores(t) for t in io_tasks]
        if cpu_futures or io_futures:
            mixed_results = await asyncio.gather(*(cpu_futures + io_futures))
            results.extend(mixed_results)
        return results

    async def _monitor_resources(self):
        """Monitor resource usage with real GPU metrics."""
        from .metal_monitor import EnhancedResourceMonitor
        monitor = EnhancedResourceMonitor()
        await monitor.start()
        try:
            while True:
                try:
                    metrics = monitor.get_system_metrics()
                    self.memory_usage_mb = metrics['memory']['used_gb'] * 1024
                    self.gpu_utilization = metrics['gpu']['utilization_percent'
                        ]
                    cpu_avg = metrics['cpu']['percent_avg']
                    memory_percent = metrics['memory']['percent']
                    if (cpu_avg > 90 or memory_percent > 85 or self.
                        gpu_utilization > 90):
                        logger.warning(
                            f'High resource usage - CPU: {cpu_avg:.1f}%, Memory: {memory_percent:.1f}%, GPU: {self.gpu_utilization:.1f}%'
                            )
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f'Resource monitoring error: {e}')
                    await asyncio.sleep(10)
        finally:
            await monitor.stop()

    def get_gpu_utilization(self) ->float:
        """Get current GPU utilization estimate."""
        return self.gpu_utilization

    def get_memory_usage_mb(self) ->float:
        """Get current memory usage in MB."""
        return self.memory_usage_mb

    def get_optimal_batch_size(self, task_type: str='general', item_size_mb:
        float=10) ->int:
        """Get optimal batch size for task type."""
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        allocated_memory = self.memory_allocations.get('working_memory', 4096)
        usable_memory = min(available_memory * 0.7, allocated_memory)
        if task_type == 'gpu':
            batch_size = int(usable_memory / (item_size_mb * 2))
            return min(256, max(1, batch_size))
        elif task_type == 'cpu':
            batch_size = int(usable_memory / item_size_mb)
            return min(self.p_cores * 4, max(1, batch_size))
        elif task_type == 'mcts':
            batch_size = int(usable_memory / (item_size_mb * 0.5))
            return min(512, max(self.p_cores, batch_size))
        else:
            batch_size = int(usable_memory / item_size_mb)
            return min(self.cpu_count * 2, max(1, batch_size))

    def shutdown(self):
        """Clean shutdown."""
        logger.info('Shutting down hardware executor')
        if self._monitor_task:
            self._monitor_task.cancel()
        if self.p_core_executor:
            self.p_core_executor.shutdown(wait=True)
        if self.e_core_executor:
            self.e_core_executor.shutdown(wait=True)
        logger.info('Hardware executor shutdown complete')


class TaskScheduler:
    """Intelligent task scheduling for M4 Pro."""

    def __init__(self, hardware_executor: HardwareAwareExecutor):
        self.executor = hardware_executor
        self.task_queue = asyncio.Queue()
        self.results = {}
        self._scheduler_task = None

    async def start(self):
        """Start the scheduler."""
        await asyncio.sleep(0)
        self._scheduler_task = asyncio.create_task(self._schedule_loop())

    async def submit(self, task_id: str, task: Dict[str, Any]) ->str:
        """Submit a task for scheduling."""
        task['id'] = task_id
        task['submitted_at'] = time.time()
        await self.task_queue.put(task)
        return task_id

    async def get_result(self, task_id: str, timeout: float=None) ->Dict[
        str, Any]:
        """Get task result."""
        start_time = time.time()
        while task_id not in self.results:
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f'Task {task_id} timed out')
            await asyncio.sleep(0.1)
        return self.results.pop(task_id)

    async def _schedule_loop(self):
        """Main scheduling loop."""
        pending_tasks = []
        while True:
            try:
                while not self.task_queue.empty() and len(pending_tasks) < 100:
                    task = await self.task_queue.get()
                    pending_tasks.append(task)
                if pending_tasks:
                    pending_tasks.sort(key=lambda t: t.get('priority', 0),
                        reverse=True)
                    batch_size = min(len(pending_tasks), self.executor.
                        get_optimal_batch_size())
                    batch = pending_tasks[:batch_size]
                    pending_tasks = pending_tasks[batch_size:]
                    results = await self.executor.batch_execute(batch)
                    for task, result in zip(batch, results):
                        self.results[task['id']] = result
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f'Scheduler error: {e}')
                await asyncio.sleep(1)


class MemoryOptimizer:
    """Optimizes memory usage for M4 Pro's 24GB unified memory."""

    def __init__(self, target_usage_percent: float=85):
        self.target_usage_percent = target_usage_percent
        self.memory_pools = {}
        self.total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        self.reserved_memory_mb = self.total_memory_mb * 0.15
        self.available_for_pools = (self.total_memory_mb - self.
            reserved_memory_mb)
        self.default_pools = {'mcts_trees': 0.15, 'neural_models': 0.25,
            'code_cache': 0.2, 'index_data': 0.15, 'batch_processing': 0.25}

    def allocate_pool(self, name: str, size_mb: int) ->bool:
        """Pre-allocate memory pool."""
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        if size_mb > available_mb * 0.8:
            logger.warning(
                f'Cannot allocate {size_mb}MB for {name}, only {available_mb}MB available'
                )
            return False
        self.memory_pools[name] = {'size_mb': size_mb, 'allocated_at': time
            .time()}
        logger.info(f'Allocated {size_mb}MB pool for {name}')
        return True

    def release_pool(self, name: str):
        """Release memory pool."""
        if name in self.memory_pools:
            del self.memory_pools[name]
            logger.info(f'Released memory pool {name}')

    def get_recommended_batch_size(self, item_size_mb: float) ->int:
        """Get recommended batch size based on available memory."""
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        target_mb = available_mb * (self.target_usage_percent / 100)
        return max(1, int(target_mb / item_size_mb))
