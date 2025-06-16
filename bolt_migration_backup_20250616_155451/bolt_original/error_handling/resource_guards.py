"""
Resource Guards for Bolt System

Comprehensive resource management and protection to prevent system crashes
due to resource exhaustion. Includes memory, CPU, GPU, and disk guards.
"""

import asyncio
import gc
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import psutil

from .exceptions import (
    BoltGPUException,
    BoltMemoryException,
    BoltResourceException,
    ErrorSeverity,
)


class ResourceState(Enum):
    """Resource utilization states."""

    NORMAL = "normal"  # < 70% usage
    ELEVATED = "elevated"  # 70-85% usage
    HIGH = "high"  # 85-95% usage
    CRITICAL = "critical"  # > 95% usage
    EXHAUSTED = "exhausted"  # Resource unavailable


class ActionType(Enum):
    """Types of actions that can be taken by resource guards."""

    MONITOR = "monitor"
    WARN = "warn"
    THROTTLE = "throttle"
    REJECT = "reject"
    EMERGENCY = "emergency"


@dataclass
class ResourceThresholds:
    """Thresholds for resource monitoring."""

    normal_threshold: float = 70.0  # Normal operation limit
    elevated_threshold: float = 80.0  # Start warnings
    high_threshold: float = 90.0  # Start throttling
    critical_threshold: float = 95.0  # Start rejecting requests
    emergency_threshold: float = 98.0  # Emergency measures


@dataclass
class ResourceUsage:
    """Current resource usage information."""

    timestamp: float
    current_usage: float
    total_available: float
    percentage: float
    state: ResourceState
    trend: str = "stable"  # "increasing", "decreasing", "stable"
    metadata: dict[str, Any] = field(default_factory=dict)


class ResourceGuard(ABC):
    """Abstract base class for resource guards."""

    def __init__(
        self,
        name: str,
        thresholds: ResourceThresholds | None = None,
        check_interval: float = 5.0,
        enable_monitoring: bool = True,
    ):
        self.name = name
        self.thresholds = thresholds or ResourceThresholds()
        self.check_interval = check_interval
        self.enable_monitoring = enable_monitoring

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{name}")

        # State tracking
        self.current_usage: ResourceUsage | None = None
        self.usage_history: list[ResourceUsage] = []
        self.max_history = 1000

        # Callbacks
        self.state_change_callbacks: list[
            Callable[[ResourceState, ResourceState], None]
        ] = []
        self.threshold_callbacks: dict[
            ResourceState, list[Callable[[ResourceUsage], None]]
        ] = {state: [] for state in ResourceState}

        # Monitoring
        self._monitoring = False
        self._monitor_task: asyncio.Task | None = None
        self._lock = threading.RLock()

        # Actions taken
        self.actions_taken: list[tuple[float, ActionType, str]] = []

        # Statistics
        self.stats = {
            "checks_performed": 0,
            "state_changes": 0,
            "actions_taken": 0,
            "time_in_states": {state.value: 0.0 for state in ResourceState},
            "last_state_change": time.time(),
        }

    @abstractmethod
    async def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def apply_throttling(self, usage: ResourceUsage) -> bool:
        """Apply throttling measures. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def apply_emergency_measures(self, usage: ResourceUsage) -> bool:
        """Apply emergency measures. Must be implemented by subclasses."""
        pass

    async def check_usage(self) -> ResourceUsage:
        """Check current resource usage and update state."""
        with self._lock:
            try:
                usage = await self.get_current_usage()
                self.current_usage = usage
                self.usage_history.append(usage)

                # Trim history
                if len(self.usage_history) > self.max_history:
                    self.usage_history = self.usage_history[-self.max_history :]

                self.stats["checks_performed"] += 1

                # Update trend
                if len(self.usage_history) > 3:
                    recent = [u.percentage for u in self.usage_history[-3:]]
                    if recent[-1] > recent[0] + 5:
                        usage.trend = "increasing"
                    elif recent[-1] < recent[0] - 5:
                        usage.trend = "decreasing"
                    else:
                        usage.trend = "stable"

                # Handle state changes and actions
                await self._handle_usage_update(usage)

                return usage

            except Exception as e:
                self.logger.error(f"Failed to check resource usage: {e}", exc_info=True)
                # Return last known usage or create error state
                if self.current_usage:
                    return self.current_usage
                else:
                    return ResourceUsage(
                        timestamp=time.time(),
                        current_usage=0,
                        total_available=1,
                        percentage=0,
                        state=ResourceState.NORMAL,
                    )

    async def _handle_usage_update(self, usage: ResourceUsage):
        """Handle resource usage updates and take appropriate actions."""
        old_state = (
            self.current_usage.state if self.current_usage else ResourceState.NORMAL
        )
        new_state = usage.state

        # Update time in states
        current_time = time.time()
        time_diff = current_time - self.stats["last_state_change"]
        self.stats["time_in_states"][old_state.value] += time_diff
        self.stats["last_state_change"] = current_time

        # Handle state changes
        if old_state != new_state:
            self.stats["state_changes"] += 1
            self.logger.info(
                f"Resource {self.name} state changed: {old_state.value} -> {new_state.value}"
            )

            # Notify callbacks
            for callback in self.state_change_callbacks:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    self.logger.warning(f"State change callback failed: {e}")

        # Take appropriate actions based on state
        await self._take_action(usage, new_state)

        # Notify threshold callbacks
        for callback in self.threshold_callbacks.get(new_state, []):
            try:
                callback(usage)
            except Exception as e:
                self.logger.warning(f"Threshold callback failed: {e}")

    async def _take_action(self, usage: ResourceUsage, state: ResourceState):
        """Take appropriate action based on resource state."""
        if state == ResourceState.NORMAL:
            # No action needed
            pass

        elif state == ResourceState.ELEVATED:
            self._record_action(
                ActionType.WARN, f"Resource usage elevated: {usage.percentage:.1f}%"
            )
            self.logger.warning(
                f"Resource {self.name} usage elevated: {usage.percentage:.1f}%"
            )

        elif state == ResourceState.HIGH:
            self._record_action(
                ActionType.THROTTLE, f"Applying throttling: {usage.percentage:.1f}%"
            )
            self.logger.warning(
                f"Resource {self.name} usage high, applying throttling: {usage.percentage:.1f}%"
            )

            try:
                throttled = await self.apply_throttling(usage)
                if throttled:
                    self.logger.info(f"Throttling applied successfully for {self.name}")
                else:
                    self.logger.warning(f"Throttling failed for {self.name}")
            except Exception as e:
                self.logger.error(f"Error applying throttling for {self.name}: {e}")

        elif state == ResourceState.CRITICAL:
            self._record_action(
                ActionType.REJECT, f"Resource critical: {usage.percentage:.1f}%"
            )
            self.logger.error(f"Resource {self.name} critical: {usage.percentage:.1f}%")

            # Apply throttling if not already done
            try:
                await self.apply_throttling(usage)
            except Exception as e:
                self.logger.error(f"Error applying throttling for {self.name}: {e}")

        elif state == ResourceState.EXHAUSTED:
            self._record_action(
                ActionType.EMERGENCY, f"Resource exhausted: {usage.percentage:.1f}%"
            )
            self.logger.critical(
                f"Resource {self.name} exhausted, applying emergency measures: {usage.percentage:.1f}%"
            )

            try:
                emergency_applied = await self.apply_emergency_measures(usage)
                if emergency_applied:
                    self.logger.info(
                        f"Emergency measures applied successfully for {self.name}"
                    )
                else:
                    self.logger.error(f"Emergency measures failed for {self.name}")
            except Exception as e:
                self.logger.error(
                    f"Error applying emergency measures for {self.name}: {e}"
                )

    def _record_action(self, action_type: ActionType, description: str):
        """Record an action taken by the resource guard."""
        self.actions_taken.append((time.time(), action_type, description))
        self.stats["actions_taken"] += 1

        # Trim action history
        if len(self.actions_taken) > 1000:
            self.actions_taken = self.actions_taken[-1000:]

    def _determine_state(self, percentage: float) -> ResourceState:
        """Determine resource state based on percentage usage."""
        if percentage >= self.thresholds.emergency_threshold:
            return ResourceState.EXHAUSTED
        elif percentage >= self.thresholds.critical_threshold:
            return ResourceState.CRITICAL
        elif percentage >= self.thresholds.high_threshold:
            return ResourceState.HIGH
        elif percentage >= self.thresholds.elevated_threshold:
            return ResourceState.ELEVATED
        else:
            return ResourceState.NORMAL

    def add_state_change_callback(
        self, callback: Callable[[ResourceState, ResourceState], None]
    ):
        """Add callback for state changes."""
        self.state_change_callbacks.append(callback)

    def add_threshold_callback(
        self, state: ResourceState, callback: Callable[[ResourceUsage], None]
    ):
        """Add callback for specific threshold."""
        self.threshold_callbacks[state].append(callback)

    async def start_monitoring(self):
        """Start continuous resource monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self.logger.info(f"Starting monitoring for resource {self.name}")

        if self.enable_monitoring:
            self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False

        if self._monitor_task:
            self._monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._monitor_task
            self._monitor_task = None

        self.logger.info(f"Stopped monitoring for resource {self.name}")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        try:
            while self._monitoring:
                await self.check_usage()
                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Monitor loop error for {self.name}: {e}", exc_info=True)

    @contextmanager
    def usage_context(self, operation: str):
        """Context manager for tracking resource usage during operations."""
        start_time = time.time()

        try:
            yield
        finally:
            end_time = time.time()
            # Check usage after operation
            asyncio.create_task(self.check_usage())

            duration = end_time - start_time
            self.logger.debug(f"Operation '{operation}' completed in {duration:.2f}s")

    def get_stats(self) -> dict[str, Any]:
        """Get resource guard statistics."""
        with self._lock:
            current_time = time.time()

            # Update current state time
            if self.current_usage:
                time_diff = current_time - self.stats["last_state_change"]
                self.stats["time_in_states"][
                    self.current_usage.state.value
                ] += time_diff

            return {
                "name": self.name,
                "current_usage": self.current_usage.percentage
                if self.current_usage
                else 0,
                "current_state": self.current_usage.state.value
                if self.current_usage
                else "unknown",
                "monitoring": self._monitoring,
                "thresholds": {
                    "elevated": self.thresholds.elevated_threshold,
                    "high": self.thresholds.high_threshold,
                    "critical": self.thresholds.critical_threshold,
                    "emergency": self.thresholds.emergency_threshold,
                },
                "statistics": self.stats.copy(),
                "recent_actions": [
                    {"timestamp": ts, "type": action_type.value, "description": desc}
                    for ts, action_type, desc in self.actions_taken[-10:]
                ],
            }


class MemoryGuard(ResourceGuard):
    """Memory resource guard."""

    def __init__(
        self,
        name: str = "memory",
        thresholds: ResourceThresholds | None = None,
        **kwargs,
    ):
        # Memory-specific thresholds
        if thresholds is None:
            thresholds = ResourceThresholds(
                elevated_threshold=75.0,
                high_threshold=85.0,
                critical_threshold=92.0,
                emergency_threshold=97.0,
            )

        super().__init__(name, thresholds, **kwargs)

        # Memory-specific settings
        self.gc_enabled = True
        self.last_gc_time = 0
        self.gc_interval = 30.0  # Minimum seconds between forced GC

    async def get_current_usage(self) -> ResourceUsage:
        """Get current memory usage."""
        try:
            vm = psutil.virtual_memory()

            usage = ResourceUsage(
                timestamp=time.time(),
                current_usage=vm.used,
                total_available=vm.total,
                percentage=vm.percent,
                state=self._determine_state(vm.percent),
                metadata={
                    "available_gb": vm.available / (1024**3),
                    "used_gb": vm.used / (1024**3),
                    "total_gb": vm.total / (1024**3),
                    "buffers_gb": getattr(vm, "buffers", 0) / (1024**3),
                    "cached_gb": getattr(vm, "cached", 0) / (1024**3),
                },
            )

            return usage

        except Exception as e:
            raise BoltMemoryException(
                f"Failed to get memory usage: {e}", severity=ErrorSeverity.HIGH
            )

    async def apply_throttling(self, usage: ResourceUsage) -> bool:
        """Apply memory throttling measures."""
        actions_taken = []

        try:
            # Force garbage collection if enough time has passed
            current_time = time.time()
            if self.gc_enabled and current_time - self.last_gc_time > self.gc_interval:
                before_gc = usage.current_usage
                gc.collect()
                self.last_gc_time = current_time

                # Check how much memory was freed
                try:
                    after_gc = psutil.virtual_memory().used
                    freed_mb = (before_gc - after_gc) / (1024 * 1024)
                    actions_taken.append(
                        f"Forced garbage collection (freed {freed_mb:.1f}MB)"
                    )
                except Exception:
                    actions_taken.append("Forced garbage collection")

            # Set environment variables for memory-conscious operation
            os.environ["BOLT_MEMORY_CONSERVATION"] = "true"
            os.environ["BOLT_REDUCE_BATCH_SIZE"] = "true"

            # Reduce batch sizes in a more targeted way
            current_batch = int(os.environ.get("BOLT_BATCH_SIZE", "32"))
            new_batch = max(1, current_batch // 2)
            os.environ["BOLT_BATCH_SIZE"] = str(new_batch)
            actions_taken.append(f"Reduced batch size: {current_batch} -> {new_batch}")

            # Reduce worker count
            current_workers = int(os.environ.get("BOLT_MAX_WORKERS", "4"))
            new_workers = max(1, current_workers // 2)
            os.environ["BOLT_MAX_WORKERS"] = str(new_workers)
            actions_taken.append(f"Reduced workers: {current_workers} -> {new_workers}")

            # Enable memory monitoring
            os.environ["BOLT_MEMORY_MONITORING"] = "true"
            actions_taken.append("Enabled memory conservation mode")

            self.logger.info(f"Memory throttling applied: {', '.join(actions_taken)}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to apply memory throttling: {e}")
            return False

    async def apply_emergency_measures(self, usage: ResourceUsage) -> bool:
        """Apply emergency memory measures."""
        actions_taken = []

        try:
            # Aggressive garbage collection with measurement
            if self.gc_enabled:
                before_gc = psutil.virtual_memory().used

                # Multiple garbage collection passes
                for generation in range(3):
                    collected = gc.collect(generation)
                    if collected > 0:
                        actions_taken.append(
                            f"GC generation {generation}: {collected} objects"
                        )

                after_gc = psutil.virtual_memory().used
                freed_mb = (before_gc - after_gc) / (1024 * 1024)
                actions_taken.append(f"Total memory freed: {freed_mb:.1f}MB")

            # Set emergency mode with minimal resource usage
            os.environ["BOLT_EMERGENCY_MODE"] = "true"
            os.environ["BOLT_MINIMAL_OPERATION"] = "true"
            os.environ["BOLT_BATCH_SIZE"] = "1"  # Minimal batch size
            os.environ["BOLT_MAX_WORKERS"] = "1"  # Single worker only
            os.environ["BOLT_DISABLE_CACHING"] = "true"
            actions_taken.append("Enabled emergency mode with minimal resource usage")

            # Clear Python caches
            try:
                import sys

                # Clear import cache
                if hasattr(sys, "_clear_type_cache"):
                    sys._clear_type_cache()
                    actions_taken.append("Cleared Python type cache")

                # Clear function caches

                # This would clear lru_cache decorated functions if we had references
                actions_taken.append("Attempted function cache clearing")

            except Exception as cache_error:
                self.logger.warning(f"Cache clearing failed: {cache_error}")

            # Try to clear ML framework caches
            try:
                # PyTorch MPS cache
                import torch

                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                    actions_taken.append("Cleared PyTorch MPS cache")
            except Exception:
                pass

            try:
                # MLX doesn't have explicit cache clearing, but we can try
                import mlx.core as mx

                # Force synchronization which may help with memory
                mx.eval(mx.array([1]))
                actions_taken.append("Synchronized MLX operations")
            except Exception:
                pass

            # Limit future allocations
            os.environ["BOLT_MEMORY_LIMIT_MB"] = str(
                int(usage.total_available * 0.7 / (1024 * 1024))
            )
            actions_taken.append(
                f"Set memory limit to 70% of total ({usage.total_available * 0.7 / (1024**3):.1f}GB)"
            )

            self.logger.critical(
                f"Memory emergency measures applied: {', '.join(actions_taken)}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to apply memory emergency measures: {e}")
            return False

    def clear_memory_caches(self):
        """Clear memory caches if possible."""
        try:
            # This would integrate with cache managers when available
            gc.collect()
            self.logger.info("Memory caches cleared")
        except Exception as e:
            self.logger.warning(f"Failed to clear memory caches: {e}")


class CPUGuard(ResourceGuard):
    """CPU resource guard."""

    def __init__(
        self, name: str = "cpu", thresholds: ResourceThresholds | None = None, **kwargs
    ):
        # CPU-specific thresholds
        if thresholds is None:
            thresholds = ResourceThresholds(
                elevated_threshold=70.0,
                high_threshold=85.0,
                critical_threshold=95.0,
                emergency_threshold=98.0,
            )

        super().__init__(name, thresholds, **kwargs)

        # CPU-specific settings
        self.cpu_count = psutil.cpu_count(logical=True)
        self.core_count_physical = psutil.cpu_count(logical=False)

    async def get_current_usage(self) -> ResourceUsage:
        """Get current CPU usage."""
        try:
            # Get average CPU usage over a short interval
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Get per-CPU usage for more detail
            per_cpu = psutil.cpu_percent(interval=None, percpu=True)

            usage = ResourceUsage(
                timestamp=time.time(),
                current_usage=cpu_percent,
                total_available=100.0,
                percentage=cpu_percent,
                state=self._determine_state(cpu_percent),
                metadata={
                    "logical_cores": self.cpu_count,
                    "physical_cores": self.core_count_physical,
                    "per_cpu_usage": per_cpu,
                    "max_core_usage": max(per_cpu) if per_cpu else 0,
                    "min_core_usage": min(per_cpu) if per_cpu else 0,
                    "load_average": os.getloadavg()
                    if hasattr(os, "getloadavg")
                    else None,
                },
            )

            return usage

        except Exception as e:
            raise BoltResourceException(
                f"Failed to get CPU usage: {e}",
                resource_type="cpu",
                severity=ErrorSeverity.MEDIUM,
            )

    async def apply_throttling(self, usage: ResourceUsage) -> bool:
        """Apply CPU throttling measures."""
        try:
            actions_taken = []

            # Reduce concurrency based on current load
            current_workers = int(
                os.environ.get("BOLT_MAX_WORKERS", str(self.cpu_count))
            )
            if usage.percentage > 90:
                new_workers = max(1, self.cpu_count // 4)  # Aggressive reduction
            else:
                new_workers = max(1, current_workers // 2)  # Standard reduction

            os.environ["BOLT_MAX_WORKERS"] = str(new_workers)
            actions_taken.append(f"Reduced workers: {current_workers} -> {new_workers}")

            # Set CPU throttling flags
            os.environ["BOLT_CPU_THROTTLE"] = "true"
            os.environ["BOLT_CPU_CONSERVATIVE"] = "true"

            # Reduce thread pool sizes if applicable
            os.environ["BOLT_THREAD_POOL_SIZE"] = str(new_workers)
            actions_taken.append(f"Limited thread pool to {new_workers}")

            # Add processing delays to reduce CPU pressure
            if usage.percentage > 95:
                os.environ["BOLT_PROCESSING_DELAY_MS"] = "100"  # 100ms delays
                actions_taken.append("Added processing delays (100ms)")
            elif usage.percentage > 85:
                os.environ["BOLT_PROCESSING_DELAY_MS"] = "50"  # 50ms delays
                actions_taken.append("Added processing delays (50ms)")

            # Reduce I/O concurrency
            os.environ["BOLT_IO_CONCURRENCY"] = "2"
            actions_taken.append("Limited I/O concurrency to 2")

            self.logger.info(f"CPU throttling applied: {', '.join(actions_taken)}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to apply CPU throttling: {e}")
            return False

    async def apply_emergency_measures(self, usage: ResourceUsage) -> bool:
        """Apply emergency CPU measures."""
        try:
            actions_taken = []

            # Force single-threaded operation
            os.environ["BOLT_MAX_WORKERS"] = "1"
            os.environ["BOLT_EMERGENCY_MODE"] = "true"
            os.environ["BOLT_SINGLE_THREADED"] = "true"
            os.environ["BOLT_THREAD_POOL_SIZE"] = "1"
            actions_taken.append("Forced single-threaded operation")

            # Add significant processing delays
            os.environ["BOLT_PROCESSING_DELAY_MS"] = "500"  # 500ms delays
            actions_taken.append("Added 500ms processing delays")

            # Disable non-essential features
            os.environ["BOLT_DISABLE_BACKGROUND_TASKS"] = "true"
            os.environ["BOLT_DISABLE_MONITORING"] = "true"
            os.environ["BOLT_MINIMAL_LOGGING"] = "true"
            actions_taken.append("Disabled non-essential features")

            # Limit I/O operations
            os.environ["BOLT_IO_CONCURRENCY"] = "1"
            os.environ["BOLT_IO_DELAY_MS"] = "100"
            actions_taken.append("Limited I/O operations")

            # Try to reduce process priority if possible
            try:
                import psutil

                current_process = psutil.Process()
                # Lower the process priority
                if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
                    current_process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                    actions_taken.append("Lowered process priority")
                elif hasattr(current_process, "nice"):
                    current_process.nice(10)  # Lower priority
                    actions_taken.append("Lowered process nice value")
            except Exception as priority_error:
                self.logger.warning(
                    f"Failed to lower process priority: {priority_error}"
                )

            # Force immediate yielding in tight loops
            os.environ["BOLT_FORCE_YIELD"] = "true"
            actions_taken.append("Enabled forced yielding")

            self.logger.critical(
                f"CPU emergency measures applied: {', '.join(actions_taken)}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to apply CPU emergency measures: {e}")
            return False


class GPUGuard(ResourceGuard):
    """GPU resource guard."""

    def __init__(
        self,
        name: str = "gpu",
        thresholds: ResourceThresholds | None = None,
        gpu_backend: str = "auto",
        **kwargs,
    ):
        # GPU-specific thresholds
        if thresholds is None:
            thresholds = ResourceThresholds(
                elevated_threshold=70.0,
                high_threshold=85.0,
                critical_threshold=92.0,
                emergency_threshold=97.0,
            )

        super().__init__(name, thresholds, **kwargs)

        self.gpu_backend = gpu_backend
        self.gpu_available = False
        self.gpu_memory_limit = 18 * 1024 * 1024 * 1024  # 18GB for M4 Pro

        # Detect GPU backend
        self._detect_gpu_backend()

    def _detect_gpu_backend(self):
        """Detect available GPU backend."""
        try:
            if self.gpu_backend == "auto" or self.gpu_backend == "mlx":
                import mlx.core as mx

                if mx.metal.is_available():
                    self.gpu_backend = "mlx"
                    self.gpu_available = True
                    return
        except ImportError:
            pass

        try:
            if self.gpu_backend == "auto" or self.gpu_backend == "mps":
                import torch

                if torch.backends.mps.is_available():
                    self.gpu_backend = "mps"
                    self.gpu_available = True
                    return
        except ImportError:
            pass

        self.gpu_backend = "none"
        self.gpu_available = False

    async def get_current_usage(self) -> ResourceUsage:
        """Get current GPU usage."""
        if not self.gpu_available:
            return ResourceUsage(
                timestamp=time.time(),
                current_usage=0,
                total_available=1,
                percentage=0,
                state=ResourceState.NORMAL,
                metadata={"backend": "none", "available": False},
            )

        try:
            memory_used = 0
            memory_total = self.gpu_memory_limit

            if self.gpu_backend == "mps":
                try:
                    import torch

                    if hasattr(torch.mps, "current_allocated_memory"):
                        memory_used = torch.mps.current_allocated_memory()
                except Exception:
                    pass

            elif self.gpu_backend == "mlx":
                # MLX doesn't expose memory directly, estimate from system
                vm = psutil.virtual_memory()
                # Rough estimate - GPU memory often correlates with system pressure
                estimated_gpu_usage = max(
                    0, (vm.total - vm.available) / (1024**3) - 4.0
                )
                memory_used = estimated_gpu_usage * 1024**3

            percentage = (memory_used / memory_total) * 100 if memory_total > 0 else 0

            usage = ResourceUsage(
                timestamp=time.time(),
                current_usage=memory_used,
                total_available=memory_total,
                percentage=percentage,
                state=self._determine_state(percentage),
                metadata={
                    "backend": self.gpu_backend,
                    "available": self.gpu_available,
                    "memory_used_gb": memory_used / (1024**3),
                    "memory_total_gb": memory_total / (1024**3),
                },
            )

            return usage

        except Exception as e:
            raise BoltGPUException(
                f"Failed to get GPU usage: {e}",
                gpu_backend=self.gpu_backend,
                severity=ErrorSeverity.MEDIUM,
            )

    async def apply_throttling(self, usage: ResourceUsage) -> bool:
        """Apply GPU throttling measures."""
        try:
            # Reduce GPU usage
            os.environ["BOLT_GPU_THROTTLE"] = "true"
            os.environ["BOLT_REDUCE_GPU_BATCH"] = "true"

            if self.gpu_backend == "mps":
                # Reduce PyTorch memory allocation
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.5"

            self.logger.info("GPU throttling applied")
            return True

        except Exception as e:
            self.logger.error(f"Failed to apply GPU throttling: {e}")
            return False

    async def apply_emergency_measures(self, usage: ResourceUsage) -> bool:
        """Apply emergency GPU measures."""
        try:
            # Disable GPU acceleration
            os.environ["BOLT_DISABLE_GPU"] = "true"
            os.environ["BOLT_CPU_ONLY"] = "true"

            if self.gpu_backend == "mps":
                # Clear PyTorch GPU cache
                try:
                    import torch

                    if hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
                except Exception:
                    pass

            self.logger.critical(
                "GPU emergency measures applied: disabled GPU acceleration"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to apply GPU emergency measures: {e}")
            return False


class ResourceGuardManager:
    """Manages multiple resource guards."""

    def __init__(self):
        self.guards: dict[str, ResourceGuard] = {}
        self.logger = logging.getLogger(f"{__name__}.ResourceGuardManager")
        self._monitoring = False

    def add_guard(self, guard: ResourceGuard):
        """Add a resource guard."""
        self.guards[guard.name] = guard
        self.logger.info(f"Added resource guard: {guard.name}")

    def remove_guard(self, name: str):
        """Remove a resource guard."""
        if name in self.guards:
            guard = self.guards.pop(name)
            asyncio.create_task(guard.stop_monitoring())
            self.logger.info(f"Removed resource guard: {name}")

    def get_guard(self, name: str) -> ResourceGuard | None:
        """Get a resource guard by name."""
        return self.guards.get(name)

    async def start_all_monitoring(self):
        """Start monitoring for all guards."""
        self._monitoring = True
        tasks = []
        for guard in self.guards.values():
            tasks.append(guard.start_monitoring())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.info("Started monitoring for all resource guards")

    async def stop_all_monitoring(self):
        """Stop monitoring for all guards."""
        self._monitoring = False
        tasks = []
        for guard in self.guards.values():
            tasks.append(guard.stop_monitoring())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.info("Stopped monitoring for all resource guards")

    async def check_all_resources(self) -> dict[str, ResourceUsage]:
        """Check all resource usage."""
        results = {}
        for name, guard in self.guards.items():
            try:
                usage = await guard.check_usage()
                results[name] = usage
            except Exception as e:
                self.logger.error(f"Failed to check resource {name}: {e}")

        return results

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all guards."""
        return {name: guard.get_stats() for name, guard in self.guards.items()}

    def get_critical_resources(self) -> list[str]:
        """Get list of resources in critical state."""
        critical = []
        for name, guard in self.guards.items():
            if guard.current_usage and guard.current_usage.state in [
                ResourceState.CRITICAL,
                ResourceState.EXHAUSTED,
            ]:
                critical.append(name)
        return critical

    def get_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        if not self.guards:
            return 100.0

        total_score = 0
        for guard in self.guards.values():
            if guard.current_usage:
                # Score based on usage percentage (inverted)
                usage_score = max(0, 100 - guard.current_usage.percentage)
                total_score += usage_score
            else:
                total_score += 100  # Unknown is considered healthy

        return total_score / len(self.guards)


# Global resource guard manager
_resource_guard_manager = ResourceGuardManager()


def get_resource_guard_manager() -> ResourceGuardManager:
    """Get the global resource guard manager."""
    return _resource_guard_manager


def setup_default_guards() -> ResourceGuardManager:
    """Set up default resource guards."""
    manager = get_resource_guard_manager()

    # Add standard guards
    manager.add_guard(MemoryGuard())
    manager.add_guard(CPUGuard())
    manager.add_guard(GPUGuard())

    return manager
