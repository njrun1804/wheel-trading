"""
Memory-Aware Resource Scheduler - Intelligent task scheduling based on memory constraints

Provides memory-aware scheduling for trading system operations, ML training,
and data processing tasks with dynamic resource allocation and priority management.
"""

import heapq
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

from .unified_manager import get_memory_manager

logger = logging.getLogger(__name__)


class TaskPriority(IntEnum):
    """Task priority levels (higher number = higher priority)"""

    CRITICAL = 10  # System critical tasks
    HIGH = 8  # High priority trading operations
    NORMAL = 5  # Standard operations
    LOW = 3  # Background tasks
    DEFERRED = 1  # Can be delayed significantly


class TaskState(Enum):
    """Task execution states"""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    MEMORY_WAIT = "memory_wait"


class SchedulingStrategy(Enum):
    """Scheduling strategy types"""

    FIFO = "fifo"  # First in, first out
    PRIORITY = "priority"  # Priority-based
    MEMORY_AWARE = "memory_aware"  # Memory-constrained scheduling
    ADAPTIVE = "adaptive"  # Adaptive based on system state


@dataclass
class ResourceRequirements:
    """Resource requirements for a task"""

    memory_mb: float
    cpu_cores: int = 1
    estimated_duration_seconds: float = 60
    component: str = "trading_data"
    can_preempt: bool = True
    memory_priority: int = 5
    min_memory_mb: float | None = None  # Minimum memory to run
    max_memory_mb: float | None = None  # Maximum beneficial memory


@dataclass
class ScheduledTask:
    """Represents a scheduled task"""

    task_id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    requirements: ResourceRequirements = None
    priority: TaskPriority = TaskPriority.NORMAL
    submitted_at: float = field(default_factory=time.time)
    scheduled_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None
    state: TaskState = TaskState.PENDING
    result: Any = None
    error: Exception | None = None
    memory_allocation_id: str | None = None
    estimated_finish_time: float | None = None
    retry_count: int = 0
    max_retries: int = 3
    tags: list[str] = field(default_factory=list)

    def __lt__(self, other):
        """For priority queue ordering"""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.submitted_at < other.submitted_at  # FIFO for same priority


@dataclass
class SchedulerStats:
    """Scheduler performance statistics"""

    tasks_submitted: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_cancelled: int = 0
    tasks_memory_wait: int = 0
    total_wait_time: float = 0
    total_execution_time: float = 0
    memory_allocation_failures: int = 0
    preemptions: int = 0
    average_queue_size: float = 0
    peak_queue_size: int = 0


class ResourceScheduler:
    """
    Memory-aware resource scheduler for trading system operations

    Features:
    - Memory-constrained task scheduling
    - Priority-based execution
    - Adaptive resource allocation
    - Preemption support for high-priority tasks
    - Resource usage tracking and optimization
    """

    def __init__(
        self,
        max_concurrent_tasks: int = 4,
        strategy: SchedulingStrategy = SchedulingStrategy.MEMORY_AWARE,
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.strategy = strategy
        self.memory_manager = get_memory_manager()

        # Task management
        self.task_queue = []  # Priority queue of pending tasks
        self.running_tasks: dict[str, ScheduledTask] = {}
        self.completed_tasks: deque = deque(maxlen=1000)  # Keep last 1000
        self.task_futures: dict[str, Future] = {}

        # Thread management
        self.executor = ThreadPoolExecutor(
            max_workers=max_concurrent_tasks, thread_name_prefix="ResourceScheduler"
        )

        # Scheduler thread
        self.scheduler_thread: threading.Thread | None = None
        self.running = False
        self.lock = threading.RLock()

        # Statistics
        self.stats = SchedulerStats()

        # Resource tracking
        self.memory_usage_history: deque = deque(
            maxlen=300
        )  # 5 minutes at 1s intervals
        self.task_performance_history: dict[str, list[float]] = defaultdict(list)

        # Scheduling configuration
        self.memory_pressure_threshold = 0.80
        self.preemption_enabled = True
        self.adaptive_batch_sizing = True

        logger.info(
            f"ResourceScheduler initialized: max_tasks={max_concurrent_tasks}, "
            f"strategy={strategy.value}"
        )

    def start(self):
        """Start the scheduler"""
        if not self.running:
            self.running = True
            self.scheduler_thread = threading.Thread(
                target=self._scheduler_loop, daemon=True, name="ResourceSchedulerLoop"
            )
            self.scheduler_thread.start()
            logger.info("Resource scheduler started")

    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=10)

        # Cancel pending tasks
        with self.lock:
            while self.task_queue:
                task = heapq.heappop(self.task_queue)
                task.state = TaskState.CANCELLED
                self.stats.tasks_cancelled += 1

        # Shutdown executor
        self.executor.shutdown(wait=True)
        logger.info("Resource scheduler stopped")

    def submit_task(
        self,
        name: str,
        func: Callable,
        *args,
        requirements: ResourceRequirements | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        tags: list[str] | None = None,
        **kwargs,
    ) -> str:
        """
        Submit a task for execution

        Args:
            name: Task name
            func: Function to execute
            args: Function arguments
            requirements: Resource requirements
            priority: Task priority
            tags: Optional tags for categorization
            kwargs: Function keyword arguments

        Returns:
            task_id: Unique task identifier
        """
        task_id = f"task_{int(time.time() * 1000000)}"

        # Default requirements if not specified
        if requirements is None:
            requirements = ResourceRequirements(
                memory_mb=100, component="trading_data"  # Default 100MB
            )

        task = ScheduledTask(
            task_id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            requirements=requirements,
            priority=priority,
            tags=tags or [],
        )

        with self.lock:
            heapq.heappush(self.task_queue, task)
            self.stats.tasks_submitted += 1

            # Update peak queue size
            queue_size = len(self.task_queue)
            if queue_size > self.stats.peak_queue_size:
                self.stats.peak_queue_size = queue_size

        logger.debug(
            f"Task submitted: {name} (ID: {task_id}, Priority: {priority.name})"
        )
        return task_id

    def get_task_status(self, task_id: str) -> ScheduledTask | None:
        """Get task status by ID"""
        with self.lock:
            # Check running tasks
            if task_id in self.running_tasks:
                return self.running_tasks[task_id]

            # Check pending tasks
            for task in self.task_queue:
                if task.task_id == task_id:
                    return task

            # Check completed tasks
            for task in self.completed_tasks:
                if task.task_id == task_id:
                    return task

            return None

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        with self.lock:
            # Cancel if running
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                if task_id in self.task_futures:
                    future = self.task_futures[task_id]
                    if future.cancel():
                        task.state = TaskState.CANCELLED
                        self._cleanup_task(task_id)
                        self.stats.tasks_cancelled += 1
                        return True
                return False

            # Cancel if pending
            new_queue = []
            cancelled = False
            while self.task_queue:
                task = heapq.heappop(self.task_queue)
                if task.task_id == task_id:
                    task.state = TaskState.CANCELLED
                    self.completed_tasks.append(task)
                    self.stats.tasks_cancelled += 1
                    cancelled = True
                else:
                    new_queue.append(task)

            # Rebuild queue
            self.task_queue = new_queue
            heapq.heapify(self.task_queue)

            return cancelled

    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                self._schedule_tasks()
                self._update_statistics()
                self._cleanup_completed_tasks()
                time.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(5)

    def _schedule_tasks(self):
        """Schedule pending tasks based on available resources"""
        with self.lock:
            if (
                not self.task_queue
                or len(self.running_tasks) >= self.max_concurrent_tasks
            ):
                return

            # Get current memory pressure
            memory_pressure = self.memory_manager.pressure_monitor.get_pressure_level()

            # Apply scheduling strategy
            if self.strategy == SchedulingStrategy.MEMORY_AWARE:
                self._schedule_memory_aware(memory_pressure)
            elif self.strategy == SchedulingStrategy.PRIORITY:
                self._schedule_priority_based()
            elif self.strategy == SchedulingStrategy.ADAPTIVE:
                self._schedule_adaptive(memory_pressure)
            else:  # FIFO
                self._schedule_fifo()

    def _schedule_memory_aware(self, memory_pressure: float):
        """Memory-aware scheduling"""
        available_slots = self.max_concurrent_tasks - len(self.running_tasks)

        # If under memory pressure, be more conservative
        if memory_pressure > self.memory_pressure_threshold:
            available_slots = min(available_slots, 1)  # Only one task at a time

        scheduled_count = 0
        temp_queue = []

        while self.task_queue and scheduled_count < available_slots:
            task = heapq.heappop(self.task_queue)

            # Check if we can allocate memory for this task
            if self._can_allocate_memory(task, memory_pressure):
                if self._start_task(task):
                    scheduled_count += 1
                else:
                    # Failed to start, put back in queue
                    temp_queue.append(task)
            else:
                # Memory not available, mark as waiting
                task.state = TaskState.MEMORY_WAIT
                temp_queue.append(task)
                self.stats.tasks_memory_wait += 1

        # Put unscheduled tasks back
        for task in temp_queue:
            heapq.heappush(self.task_queue, task)

    def _schedule_priority_based(self):
        """Priority-based scheduling (ignoring memory constraints)"""
        available_slots = self.max_concurrent_tasks - len(self.running_tasks)
        scheduled_count = 0

        while self.task_queue and scheduled_count < available_slots:
            task = heapq.heappop(self.task_queue)
            if self._start_task(task):
                scheduled_count += 1

    def _schedule_adaptive(self, memory_pressure: float):
        """Adaptive scheduling based on system state"""
        # Use memory-aware scheduling under pressure, otherwise priority-based
        if memory_pressure > self.memory_pressure_threshold:
            self._schedule_memory_aware(memory_pressure)
        else:
            self._schedule_priority_based()

    def _schedule_fifo(self):
        """First-in-first-out scheduling"""
        available_slots = self.max_concurrent_tasks - len(self.running_tasks)

        # Sort by submission time instead of priority
        queue_copy = list(self.task_queue)
        queue_copy.sort(key=lambda t: t.submitted_at)
        self.task_queue = queue_copy
        heapq.heapify(self.task_queue)

        scheduled_count = 0
        while self.task_queue and scheduled_count < available_slots:
            task = heapq.heappop(self.task_queue)
            if self._start_task(task):
                scheduled_count += 1

    def _can_allocate_memory(self, task: ScheduledTask, memory_pressure: float) -> bool:
        """Check if memory can be allocated for task"""
        requirements = task.requirements

        # Check component budget
        component_usage = self.memory_manager.get_component_usage(
            requirements.component
        )
        available_mb = component_usage["budget_mb"] - component_usage["allocated_mb"]

        # Require more free memory under pressure
        memory_multiplier = 1.0
        if memory_pressure > 0.85:
            memory_multiplier = 1.5  # Need 50% more free memory
        elif memory_pressure > 0.75:
            memory_multiplier = 1.2  # Need 20% more free memory

        required_mb = requirements.memory_mb * memory_multiplier

        # For high priority tasks, be more lenient
        if task.priority >= TaskPriority.HIGH:
            memory_multiplier *= 0.8

        return available_mb >= required_mb

    def _start_task(self, task: ScheduledTask) -> bool:
        """Start executing a task"""
        try:
            # Allocate memory
            alloc_id = self.memory_manager.allocate(
                component=task.requirements.component,
                size_mb=task.requirements.memory_mb,
                description=f"Task: {task.name}",
                priority=task.requirements.memory_priority,
                can_evict=task.requirements.can_preempt,
                tags=["scheduled_task"] + task.tags,
            )

            if not alloc_id:
                self.stats.memory_allocation_failures += 1
                return False

            task.memory_allocation_id = alloc_id
            task.state = TaskState.SCHEDULED
            task.scheduled_at = time.time()

            # Submit to executor
            future = self.executor.submit(self._execute_task, task)

            self.running_tasks[task.task_id] = task
            self.task_futures[task.task_id] = future

            # Estimate finish time
            task.estimated_finish_time = (
                time.time() + task.requirements.estimated_duration_seconds
            )

            logger.debug(
                f"Started task: {task.name} (Memory: {task.requirements.memory_mb}MB)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start task {task.name}: {e}")
            if task.memory_allocation_id:
                self.memory_manager.deallocate(task.memory_allocation_id)
            return False

    def _execute_task(self, task: ScheduledTask) -> Any:
        """Execute a task (runs in thread pool)"""
        task.state = TaskState.RUNNING
        task.started_at = time.time()

        try:
            logger.info(f"Executing task: {task.name}")

            # Execute the task function
            result = task.func(*task.args, **task.kwargs)

            task.result = result
            task.state = TaskState.COMPLETED
            task.completed_at = time.time()

            # Record performance
            execution_time = task.completed_at - task.started_at
            self.task_performance_history[task.name].append(execution_time)

            logger.info(f"Task completed: {task.name} ({execution_time:.2f}s)")
            return result

        except Exception as e:
            task.error = e
            task.state = TaskState.FAILED
            task.completed_at = time.time()

            logger.error(f"Task failed: {task.name} - {e}")
            raise

        finally:
            # Clean up in main thread (scheduled by _cleanup_completed_tasks)
            pass

    def _cleanup_task(self, task_id: str):
        """Clean up task resources"""
        with self.lock:
            if task_id in self.running_tasks:
                task = self.running_tasks.pop(task_id)

                # Deallocate memory
                if task.memory_allocation_id:
                    self.memory_manager.deallocate(task.memory_allocation_id)

                # Remove future
                if task_id in self.task_futures:
                    self.task_futures.pop(task_id)

                # Add to completed tasks
                self.completed_tasks.append(task)

                # Update statistics
                if task.state == TaskState.COMPLETED:
                    self.stats.tasks_completed += 1

                    # Update timing statistics
                    wait_time = (
                        task.started_at - task.submitted_at if task.started_at else 0
                    )
                    execution_time = (
                        task.completed_at - task.started_at
                        if task.completed_at and task.started_at
                        else 0
                    )

                    self.stats.total_wait_time += wait_time
                    self.stats.total_execution_time += execution_time

                elif task.state == TaskState.FAILED:
                    self.stats.tasks_failed += 1

    def _cleanup_completed_tasks(self):
        """Clean up completed tasks"""
        with self.lock:
            completed_task_ids = []

            for task_id, future in self.task_futures.items():
                if future.done():
                    completed_task_ids.append(task_id)

            for task_id in completed_task_ids:
                self._cleanup_task(task_id)

    def _update_statistics(self):
        """Update scheduler statistics"""
        with self.lock:
            # Update average queue size
            current_queue_size = len(self.task_queue)
            if self.stats.tasks_submitted > 0:
                self.stats.average_queue_size = (
                    self.stats.average_queue_size * (self.stats.tasks_submitted - 1)
                    + current_queue_size
                ) / self.stats.tasks_submitted

            # Record memory usage
            memory_usage = self.memory_manager.get_system_usage()["allocated_mb"]
            self.memory_usage_history.append(memory_usage)

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive scheduler statistics"""
        with self.lock:
            total_tasks = self.stats.tasks_submitted
            avg_wait_time = self.stats.total_wait_time / max(
                1, self.stats.tasks_completed
            )
            avg_execution_time = self.stats.total_execution_time / max(
                1, self.stats.tasks_completed
            )

            return {
                "tasks": {
                    "submitted": self.stats.tasks_submitted,
                    "completed": self.stats.tasks_completed,
                    "failed": self.stats.tasks_failed,
                    "cancelled": self.stats.tasks_cancelled,
                    "running": len(self.running_tasks),
                    "pending": len(self.task_queue),
                    "memory_wait": self.stats.tasks_memory_wait,
                },
                "performance": {
                    "completion_rate": self.stats.tasks_completed / max(1, total_tasks),
                    "failure_rate": self.stats.tasks_failed / max(1, total_tasks),
                    "average_wait_time": avg_wait_time,
                    "average_execution_time": avg_execution_time,
                    "memory_allocation_failures": self.stats.memory_allocation_failures,
                    "preemptions": self.stats.preemptions,
                },
                "queue": {
                    "current_size": len(self.task_queue),
                    "peak_size": self.stats.peak_queue_size,
                    "average_size": self.stats.average_queue_size,
                },
                "resources": {
                    "max_concurrent_tasks": self.max_concurrent_tasks,
                    "current_memory_mb": self.memory_usage_history[-1]
                    if self.memory_usage_history
                    else 0,
                    "strategy": self.strategy.value,
                },
            }

    def get_task_performance_summary(self, task_name: str) -> dict[str, float]:
        """Get performance summary for a specific task type"""
        if task_name not in self.task_performance_history:
            return {}

        times = self.task_performance_history[task_name]
        return {
            "count": len(times),
            "average_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "total_time": sum(times),
        }

    def optimize_configuration(self) -> dict[str, Any]:
        """Analyze performance and suggest configuration optimizations"""
        stats = self.get_statistics()
        recommendations = []

        # Analyze completion rate
        if stats["performance"]["completion_rate"] < 0.9:
            recommendations.append(
                "Low completion rate - consider increasing max_concurrent_tasks"
            )

        # Analyze memory allocation failures
        if (
            stats["performance"]["memory_allocation_failures"]
            > stats["tasks"]["submitted"] * 0.1
        ):
            recommendations.append(
                "High memory allocation failures - review memory budgets"
            )

        # Analyze queue size
        if stats["queue"]["average_size"] > self.max_concurrent_tasks * 2:
            recommendations.append(
                "Large queue size - consider increasing concurrency or task prioritization"
            )

        # Analyze wait times
        if stats["performance"]["average_wait_time"] > 60:  # More than 1 minute
            recommendations.append(
                "High wait times - optimize scheduling strategy or increase resources"
            )

        return {
            "current_stats": stats,
            "recommendations": recommendations,
            "optimal_settings": {
                "suggested_max_concurrent": min(
                    8, max(2, int(stats["queue"]["average_size"] / 2))
                ),
                "suggested_strategy": SchedulingStrategy.ADAPTIVE.value
                if len(recommendations) > 2
                else self.strategy.value,
            },
        }


# Global scheduler instance
_resource_scheduler: ResourceScheduler | None = None


def get_resource_scheduler() -> ResourceScheduler:
    """Get or create the global resource scheduler"""
    global _resource_scheduler
    if _resource_scheduler is None:
        _resource_scheduler = ResourceScheduler()
        _resource_scheduler.start()
    return _resource_scheduler


# Convenience functions for common task types


def schedule_trading_task(
    name: str,
    func: Callable,
    memory_mb: float = 100,
    priority: TaskPriority = TaskPriority.NORMAL,
    *args,
    **kwargs,
) -> str:
    """Schedule a trading-related task"""
    requirements = ResourceRequirements(
        memory_mb=memory_mb, component="trading_data", memory_priority=6
    )
    scheduler = get_resource_scheduler()
    return scheduler.submit_task(
        name,
        func,
        *args,
        requirements=requirements,
        priority=priority,
        tags=["trading"],
        **kwargs,
    )


def schedule_ml_task(
    name: str,
    func: Callable,
    memory_mb: float = 500,
    priority: TaskPriority = TaskPriority.NORMAL,
    *args,
    **kwargs,
) -> str:
    """Schedule an ML-related task"""
    requirements = ResourceRequirements(
        memory_mb=memory_mb,
        component="ml_models",
        memory_priority=7,
        estimated_duration_seconds=300,  # ML tasks typically longer
    )
    scheduler = get_resource_scheduler()
    return scheduler.submit_task(
        name,
        func,
        *args,
        requirements=requirements,
        priority=priority,
        tags=["ml"],
        **kwargs,
    )


def schedule_database_task(
    name: str,
    func: Callable,
    memory_mb: float = 200,
    priority: TaskPriority = TaskPriority.NORMAL,
    *args,
    **kwargs,
) -> str:
    """Schedule a database-related task"""
    requirements = ResourceRequirements(
        memory_mb=memory_mb,
        component="database",
        memory_priority=6,
        estimated_duration_seconds=120,
    )
    scheduler = get_resource_scheduler()
    return scheduler.submit_task(
        name,
        func,
        *args,
        requirements=requirements,
        priority=priority,
        tags=["database"],
        **kwargs,
    )
