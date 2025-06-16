"""
M4 Pro-Aware Adaptive Concurrency Management

Implements intelligent concurrency control that dynamically adapts to M4 Pro's
heterogeneous core architecture (8 P-cores + 4 E-cores + 20 Metal cores).
"""

import asyncio
import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from queue import PriorityQueue
from typing import Any

import psutil

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task types optimized for different M4 Pro compute units"""

    CPU_INTENSIVE = "cpu_intensive"  # P-cores only
    IO_BOUND = "io_bound"  # E-cores preferred
    GPU_ACCELERATED = "gpu_accelerated"  # Metal cores
    MEMORY_BOUND = "memory_bound"  # Unified memory optimized
    MIXED_WORKLOAD = "mixed_workload"  # Dynamic allocation


class CoreType(Enum):
    """M4 Pro core types"""

    PERFORMANCE = "performance"  # P-cores (8 cores)
    EFFICIENCY = "efficiency"  # E-cores (4 cores)
    METAL = "metal"  # GPU cores (20 cores)
    NEURAL = "neural"  # ANE (16 cores)


@dataclass
class CoreAllocation:
    """Core allocation strategy for hardware (configurable)"""

    p_cores: int = 8  # Performance cores (configurable)
    e_cores: int = 4  # Efficiency cores (configurable)
    metal_cores: int = 20  # Metal GPU cores (configurable)
    ane_cores: int = 16  # Apple Neural Engine cores (configurable)

    @classmethod
    def from_config(cls):
        """Create CoreAllocation from Bolt configuration."""
        try:
            from bob.integration.bolt.config import get_default_config

            config = get_default_config()
            return cls(
                p_cores=config.hardware.performance_cores,
                e_cores=config.hardware.efficiency_cores,
                metal_cores=config.hardware.metal_cores,
                ane_cores=config.hardware.ane_cores,
            )
        except Exception as e:
            logger.warning(
                f"Failed to load core allocation from config, using defaults: {e}"
            )
            return cls()  # Use default values

    @property
    def total_cpu_cores(self) -> int:
        return self.p_cores + self.e_cores

    @property
    def total_compute_units(self) -> int:
        return self.p_cores + self.e_cores + self.metal_cores + self.ane_cores


@dataclass
class ConcurrencyMetrics:
    """Metrics for adaptive concurrency management"""

    active_tasks: dict[TaskType, int] = field(default_factory=dict)
    completed_tasks: dict[TaskType, int] = field(default_factory=dict)
    failed_tasks: dict[TaskType, int] = field(default_factory=dict)
    average_latency: dict[TaskType, float] = field(default_factory=dict)
    peak_concurrency: dict[TaskType, int] = field(default_factory=dict)
    resource_utilization: dict[CoreType, float] = field(default_factory=dict)
    adaptive_adjustments: int = 0
    pressure_events: int = 0


class AdaptiveConcurrencyManager:
    """
    Intelligent concurrency manager that adapts to M4 Pro's heterogeneous architecture.

    Key features:
    - Dynamic task routing based on core availability
    - Resource pressure detection and mitigation
    - Workload-aware concurrency limits
    - Performance-based adaptation
    """

    def __init__(self, enable_monitoring: bool = True):
        self.core_allocation = (
            CoreAllocation.from_config()
        )  # Use configurable allocation
        self.metrics = ConcurrencyMetrics()
        self.enable_monitoring = enable_monitoring

        # Initialize concurrency limits based on M4 Pro capabilities
        self._initialize_concurrency_limits()

        # Task queues for different core types
        self._task_queues: dict[CoreType, PriorityQueue] = {
            CoreType.PERFORMANCE: PriorityQueue(),
            CoreType.EFFICIENCY: PriorityQueue(),
            CoreType.METAL: PriorityQueue(),
            CoreType.NEURAL: PriorityQueue(),
        }

        # Thread pools for CPU tasks
        self._p_core_executor = ThreadPoolExecutor(
            max_workers=self.core_allocation.p_cores, thread_name_prefix="p-core"
        )
        self._e_core_executor = ThreadPoolExecutor(
            max_workers=self.core_allocation.e_cores, thread_name_prefix="e-core"
        )

        # Semaphores for resource management
        self._semaphores = self._create_resource_semaphores()

        # Monitoring
        self._monitoring_active = False
        self._monitor_thread: threading.Thread | None = None
        if enable_monitoring:
            self._start_monitoring()

        # Performance tracking
        self._task_history: list[
            tuple[TaskType, float, float]
        ] = []  # (type, start, duration)
        self._adaptation_history: list[
            tuple[str, float, dict]
        ] = []  # (reason, timestamp, adjustments)

        logger.info("Initialized AdaptiveConcurrencyManager for M4 Pro")
        logger.info(
            f"Core allocation: {self.core_allocation.p_cores}P + {self.core_allocation.e_cores}E + "
            f"{self.core_allocation.metal_cores}M + {self.core_allocation.ane_cores}ANE"
        )

    def _initialize_concurrency_limits(self):
        """Initialize concurrency limits based on M4 Pro characteristics"""
        self._base_limits = {
            TaskType.CPU_INTENSIVE: self.core_allocation.p_cores,
            TaskType.IO_BOUND: self.core_allocation.total_cpu_cores,
            TaskType.GPU_ACCELERATED: min(4, self.core_allocation.metal_cores // 4),
            TaskType.MEMORY_BOUND: self.core_allocation.total_cpu_cores // 2,
            TaskType.MIXED_WORKLOAD: self.core_allocation.total_cpu_cores,
        }

        # Dynamic limits (will be adjusted based on performance)
        self._current_limits = self._base_limits.copy()

        logger.debug(f"Base concurrency limits: {self._base_limits}")

    def _create_resource_semaphores(self) -> dict[TaskType, asyncio.Semaphore]:
        """Create semaphores for resource management"""
        return {
            task_type: asyncio.Semaphore(limit)
            for task_type, limit in self._current_limits.items()
        }

    def _start_monitoring(self):
        """Start background monitoring thread"""
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="concurrency-monitor", daemon=True
        )
        self._monitor_thread.start()
        logger.debug("Started concurrency monitoring")

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                self._update_system_metrics()
                self._adapt_concurrency_limits()
                self._cleanup_task_history()
                time.sleep(5)  # Monitor every 5 seconds
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(10)  # Back off on error

    def _update_system_metrics(self):
        """Update system resource utilization metrics"""
        try:
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory utilization
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Update metrics
            self.metrics.resource_utilization[CoreType.PERFORMANCE] = min(
                100.0, cpu_percent * 1.2
            )  # Estimate P-core usage
            self.metrics.resource_utilization[CoreType.EFFICIENCY] = min(
                100.0, cpu_percent * 0.8
            )  # Estimate E-core usage

            # Metal GPU utilization (approximate)
            if MLX_AVAILABLE:
                try:
                    # Estimate GPU usage based on active GPU tasks
                    gpu_tasks = self.metrics.active_tasks.get(
                        TaskType.GPU_ACCELERATED, 0
                    )
                    max_gpu_tasks = self._current_limits[TaskType.GPU_ACCELERATED]
                    gpu_utilization = (gpu_tasks / max(max_gpu_tasks, 1)) * 100
                    self.metrics.resource_utilization[CoreType.METAL] = gpu_utilization
                except Exception:
                    self.metrics.resource_utilization[CoreType.METAL] = 0.0

            # Detect resource pressure
            if cpu_percent > 80 or memory_percent > 85:
                self.metrics.pressure_events += 1
                logger.warning(
                    f"Resource pressure detected: CPU {cpu_percent}%, Memory {memory_percent}%"
                )

        except Exception as e:
            logger.debug(f"Metrics update error: {e}")

    def _adapt_concurrency_limits(self):
        """Adapt concurrency limits based on current performance"""
        adaptations_made = False
        adaptation_reasons = []

        # Check for resource pressure
        cpu_utilization = self.metrics.resource_utilization.get(CoreType.PERFORMANCE, 0)

        if cpu_utilization > 90:
            # High CPU pressure - reduce CPU-intensive limits
            old_limit = self._current_limits[TaskType.CPU_INTENSIVE]
            self._current_limits[TaskType.CPU_INTENSIVE] = max(1, old_limit - 1)
            if old_limit != self._current_limits[TaskType.CPU_INTENSIVE]:
                adaptations_made = True
                adaptation_reasons.append(
                    f"Reduced CPU_INTENSIVE: {old_limit} -> {self._current_limits[TaskType.CPU_INTENSIVE]}"
                )

        elif cpu_utilization < 50:
            # Low CPU pressure - potentially increase limits
            old_limit = self._current_limits[TaskType.CPU_INTENSIVE]
            max_limit = self._base_limits[TaskType.CPU_INTENSIVE]
            if old_limit < max_limit:
                self._current_limits[TaskType.CPU_INTENSIVE] = min(
                    max_limit, old_limit + 1
                )
                adaptations_made = True
                adaptation_reasons.append(
                    f"Increased CPU_INTENSIVE: {old_limit} -> {self._current_limits[TaskType.CPU_INTENSIVE]}"
                )

        # Adapt GPU limits based on success rate
        gpu_completed = self.metrics.completed_tasks.get(TaskType.GPU_ACCELERATED, 0)
        gpu_failed = self.metrics.failed_tasks.get(TaskType.GPU_ACCELERATED, 0)

        if gpu_completed > 0:
            gpu_success_rate = gpu_completed / (gpu_completed + gpu_failed)
            if gpu_success_rate < 0.8:  # Less than 80% success rate
                old_limit = self._current_limits[TaskType.GPU_ACCELERATED]
                self._current_limits[TaskType.GPU_ACCELERATED] = max(1, old_limit - 1)
                if old_limit != self._current_limits[TaskType.GPU_ACCELERATED]:
                    adaptations_made = True
                    adaptation_reasons.append(
                        f"Reduced GPU_ACCELERATED due to failures: {old_limit} -> {self._current_limits[TaskType.GPU_ACCELERATED]}"
                    )

        # Update semaphores if adaptations were made
        if adaptations_made:
            self._update_semaphores()
            self.metrics.adaptive_adjustments += 1

            # Record adaptation
            self._adaptation_history.append(
                (
                    "; ".join(adaptation_reasons),
                    time.time(),
                    self._current_limits.copy(),
                )
            )

            logger.info(f"Adapted concurrency limits: {'; '.join(adaptation_reasons)}")

    def _update_semaphores(self):
        """Update semaphores with new limits"""
        for task_type, new_limit in self._current_limits.items():
            if task_type in self._semaphores:
                # Note: asyncio.Semaphore doesn't support dynamic limit changes
                # We'll recreate the semaphore (this is acceptable for our use case)
                self._semaphores[task_type] = asyncio.Semaphore(new_limit)

    def _cleanup_task_history(self):
        """Clean up old task history to prevent memory leaks"""
        current_time = time.time()
        cutoff_time = current_time - 3600  # Keep last hour

        self._task_history = [
            (task_type, start, duration)
            for task_type, start, duration in self._task_history
            if start > cutoff_time
        ]

        # Keep only recent adaptations
        self._adaptation_history = self._adaptation_history[
            -50:
        ]  # Keep last 50 adaptations

    async def execute_task(
        self,
        task_func: Callable,
        task_type: TaskType,
        priority: int = 0,
        timeout: float | None = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute a task with adaptive concurrency control.

        Routes tasks to appropriate compute units based on task type and system state.
        """
        start_time = time.time()

        # Acquire semaphore for resource management
        semaphore = self._semaphores.get(task_type)
        if semaphore:
            await semaphore.acquire()

        try:
            # Update metrics
            self.metrics.active_tasks[task_type] = (
                self.metrics.active_tasks.get(task_type, 0) + 1
            )
            self.metrics.peak_concurrency[task_type] = max(
                self.metrics.peak_concurrency.get(task_type, 0),
                self.metrics.active_tasks[task_type],
            )

            # Route task to appropriate executor
            result = await self._route_task(
                task_func, task_type, timeout, *args, **kwargs
            )

            # Update success metrics
            self.metrics.completed_tasks[task_type] = (
                self.metrics.completed_tasks.get(task_type, 0) + 1
            )

            return result

        except Exception as e:
            # Update failure metrics
            self.metrics.failed_tasks[task_type] = (
                self.metrics.failed_tasks.get(task_type, 0) + 1
            )
            logger.error(f"Task execution failed ({task_type.value}): {e}")
            raise

        finally:
            # Clean up
            if semaphore:
                semaphore.release()

            # Update metrics
            self.metrics.active_tasks[task_type] -= 1

            # Record task completion
            duration = time.time() - start_time
            self._task_history.append((task_type, start_time, duration))

            # Update average latency
            if task_type not in self.metrics.average_latency:
                self.metrics.average_latency[task_type] = duration
            else:
                # Exponential moving average
                alpha = 0.1
                self.metrics.average_latency[task_type] = (
                    alpha * duration
                    + (1 - alpha) * self.metrics.average_latency[task_type]
                )

    async def _route_task(
        self,
        task_func: Callable,
        task_type: TaskType,
        timeout: float | None,
        *args,
        **kwargs,
    ) -> Any:
        """Route task to appropriate compute unit based on task type"""

        if task_type == TaskType.CPU_INTENSIVE:
            # Use P-cores for CPU-intensive tasks
            return await self._execute_on_p_cores(task_func, timeout, *args, **kwargs)

        elif task_type == TaskType.IO_BOUND:
            # Use E-cores for I/O-bound tasks
            return await self._execute_on_e_cores(task_func, timeout, *args, **kwargs)

        elif task_type == TaskType.GPU_ACCELERATED:
            # Use Metal cores for GPU tasks
            return await self._execute_on_gpu(task_func, timeout, *args, **kwargs)

        elif task_type == TaskType.MEMORY_BOUND:
            # Use unified memory optimization
            return await self._execute_memory_optimized(
                task_func, timeout, *args, **kwargs
            )

        else:  # MIXED_WORKLOAD
            # Dynamic routing based on current load
            return await self._execute_adaptive(task_func, timeout, *args, **kwargs)

    async def _execute_on_p_cores(
        self, task_func: Callable, timeout: float | None, *args, **kwargs
    ) -> Any:
        """Execute task on P-cores with high-performance threading"""
        loop = asyncio.get_event_loop()

        # Set high QoS for P-core tasks
        def wrapped_task():
            try:
                # Set thread priority (macOS-specific)
                if hasattr(os, "nice"):
                    os.nice(-5)  # Higher priority
            except (OSError, PermissionError) as e:
                logger.debug(f"Could not set thread priority: {e}")
                # Continue execution without priority adjustment

            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(task_func):
                # Run async function in new event loop
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(task_func(*args, **kwargs))
                finally:
                    new_loop.close()
            else:
                return task_func(*args, **kwargs)

        future = loop.run_in_executor(self._p_core_executor, wrapped_task)

        try:
            if timeout:
                return await asyncio.wait_for(future, timeout=timeout)
            else:
                return await future
        except TimeoutError as e:
            logger.warning(f"P-core task timeout after {timeout}s")
            raise TimeoutError("P-core task execution timeout") from e

    async def _execute_on_e_cores(
        self, task_func: Callable, timeout: float | None, *args, **kwargs
    ) -> Any:
        """Execute task on E-cores for efficiency"""
        loop = asyncio.get_event_loop()

        future = loop.run_in_executor(self._e_core_executor, task_func, *args, **kwargs)

        try:
            if timeout:
                return await asyncio.wait_for(future, timeout=timeout)
            else:
                return await future
        except TimeoutError as e:
            logger.warning(f"E-core task timeout after {timeout}s")
            raise TimeoutError("E-core task execution timeout") from e

    async def _execute_on_gpu(
        self, task_func: Callable, timeout: float | None, *args, **kwargs
    ) -> Any:
        """Execute task on Metal GPU cores"""
        if not MLX_AVAILABLE:
            logger.warning("MLX not available, falling back to CPU")
            return await self._execute_on_p_cores(task_func, timeout, *args, **kwargs)

        loop = asyncio.get_event_loop()

        def gpu_wrapper():
            try:
                # Ensure MLX uses GPU
                if hasattr(mx, "set_default_device"):
                    mx.set_default_device("gpu")
                return task_func(*args, **kwargs)
            except Exception as e:
                logger.debug(f"GPU execution failed, details: {e}")
                raise

        future = loop.run_in_executor(None, gpu_wrapper)

        try:
            if timeout:
                return await asyncio.wait_for(future, timeout=timeout)
            else:
                return await future
        except TimeoutError as e:
            logger.warning(f"GPU task timeout after {timeout}s")
            raise TimeoutError("GPU task execution timeout") from e

    async def _execute_memory_optimized(
        self, task_func: Callable, timeout: float | None, *args, **kwargs
    ) -> Any:
        """Execute task with unified memory optimization"""
        # Use P-cores but with memory-aware scheduling
        return await self._execute_on_p_cores(task_func, timeout, *args, **kwargs)

    async def _execute_adaptive(
        self, task_func: Callable, timeout: float | None, *args, **kwargs
    ) -> Any:
        """Execute task with adaptive routing based on current load"""
        # Choose executor based on current utilization
        p_core_load = self.metrics.resource_utilization.get(CoreType.PERFORMANCE, 0)
        e_core_load = self.metrics.resource_utilization.get(CoreType.EFFICIENCY, 0)

        if p_core_load < e_core_load:
            return await self._execute_on_p_cores(task_func, timeout, *args, **kwargs)
        else:
            return await self._execute_on_e_cores(task_func, timeout, *args, **kwargs)

    async def batch_execute(
        self,
        tasks: list[tuple[Callable, TaskType, int]],  # (func, type, priority)
        max_concurrent: int | None = None,
    ) -> list[Any]:
        """Execute multiple tasks concurrently with adaptive scheduling"""

        if max_concurrent is None:
            max_concurrent = self.core_allocation.total_cpu_cores

        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(task_func, task_type, priority):
            async with semaphore:
                return await self.execute_task(task_func, task_type, priority)

        # Create coroutines
        coroutines = [
            execute_with_semaphore(task_func, task_type, priority)
            for task_func, task_type, priority in tasks
        ]

        # Execute all tasks
        return await asyncio.gather(*coroutines, return_exceptions=True)

    def get_optimal_concurrency(self, task_type: TaskType) -> int:
        """Get optimal concurrency limit for task type"""
        return self._current_limits.get(task_type, 1)

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            "core_allocation": {
                "p_cores": self.core_allocation.p_cores,
                "e_cores": self.core_allocation.e_cores,
                "metal_cores": self.core_allocation.metal_cores,
                "ane_cores": self.core_allocation.ane_cores,
            },
            "current_limits": self._current_limits,
            "base_limits": self._base_limits,
            "active_tasks": dict(self.metrics.active_tasks),
            "completed_tasks": dict(self.metrics.completed_tasks),
            "failed_tasks": dict(self.metrics.failed_tasks),
            "average_latency": dict(self.metrics.average_latency),
            "peak_concurrency": dict(self.metrics.peak_concurrency),
            "resource_utilization": {
                k.value: v for k, v in self.metrics.resource_utilization.items()
            },
            "adaptive_adjustments": self.metrics.adaptive_adjustments,
            "pressure_events": self.metrics.pressure_events,
            "recent_adaptations": self._adaptation_history[-10:],  # Last 10 adaptations
        }

    def shutdown(self):
        """Shutdown the concurrency manager"""
        logger.info("Shutting down AdaptiveConcurrencyManager")

        # Stop monitoring
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)

        # Shutdown executors
        self._p_core_executor.shutdown(wait=True)
        self._e_core_executor.shutdown(wait=True)

        logger.info("AdaptiveConcurrencyManager shutdown complete")


# Global instance
_concurrency_manager: AdaptiveConcurrencyManager | None = None


def get_adaptive_concurrency_manager() -> AdaptiveConcurrencyManager:
    """Get global adaptive concurrency manager instance"""
    global _concurrency_manager
    if _concurrency_manager is None:
        _concurrency_manager = AdaptiveConcurrencyManager()
    return _concurrency_manager


async def execute_with_adaptive_concurrency(
    task_func: Callable,
    task_type: TaskType,
    priority: int = 0,
    timeout: float | None = None,
    *args,
    **kwargs,
) -> Any:
    """Convenience function for executing tasks with adaptive concurrency"""
    manager = get_adaptive_concurrency_manager()
    return await manager.execute_task(
        task_func, task_type, priority, timeout, *args, **kwargs
    )
