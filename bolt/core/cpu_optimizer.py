"""
CPU Optimization for M4 Pro - Phase 2 Implementation

Implements CPU scheduling optimization with core affinity for maximum utilization.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum

import psutil

logger = logging.getLogger(__name__)


class CoreType(Enum):
    """M4 Pro core types."""

    P_CORE = "performance"  # Cores 0-7
    E_CORE = "efficiency"  # Cores 8-11


@dataclass
class CPUMetrics:
    """CPU performance metrics."""

    utilization_percent: float
    p_core_usage: float
    e_core_usage: float
    context_switches: int
    load_average: float
    thread_count: int


class M4ProCPUOptimizer:
    """
    PHASE 2.1 & 2.2: CPU Scheduling Optimization

    Implements intelligent core affinity assignment and load balancing
    to increase CPU utilization from 15% to 80%+.
    """

    def __init__(self):
        # M4 Pro core configuration
        self.p_cores = list(range(8))  # Performance cores 0-7
        self.e_cores = list(range(8, 12))  # Efficiency cores 8-11
        self.total_cores = 12

        # Core assignment tracking
        self.core_assignments: dict[int, str] = {}  # core_id -> task_type
        self.thread_assignments: dict[int, int] = {}  # thread_id -> core_id

        # Performance monitoring
        self.metrics = CPUMetrics(0, 0, 0, 0, 0, 0)
        self._monitoring = False
        self._monitor_thread: threading.Thread | None = None

        # Load balancing
        self.load_threshold = 0.8  # 80% load triggers rebalancing
        self.rebalance_interval = 1.0  # Check every second

    def assign_task_to_core(self, task_type: str, thread_id: int | None = None) -> int:
        """
        Assign task to optimal core based on workload type.

        Args:
            task_type: Type of task ('compute', 'io', 'coordination', 'memory')
            thread_id: Thread ID to assign (current thread if None)

        Returns:
            Assigned core ID
        """
        if thread_id is None:
            thread_id = threading.get_ident()

        # Determine optimal core based on task type
        if task_type in ["compute", "analysis", "optimization"]:
            # Heavy compute tasks go to P-cores
            available_p_cores = [
                c
                for c in self.p_cores
                if c not in self.core_assignments
                or self.core_assignments[c] == task_type
            ]

            if available_p_cores:
                core_id = available_p_cores[0]
            else:
                # Fallback to least loaded P-core
                core_id = min(self.p_cores, key=lambda c: self._get_core_load(c))
        else:
            # I/O, coordination, and memory tasks go to E-cores
            available_e_cores = [
                c
                for c in self.e_cores
                if c not in self.core_assignments
                or self.core_assignments[c] == task_type
            ]

            if available_e_cores:
                core_id = available_e_cores[0]
            else:
                # Fallback to least loaded E-core
                core_id = min(self.e_cores, key=lambda c: self._get_core_load(c))

        # Apply CPU affinity
        try:
            os.sched_setaffinity(0, {core_id})
            self.core_assignments[core_id] = task_type
            self.thread_assignments[thread_id] = core_id
            return core_id
        except OSError as e:
            # Fallback if affinity setting fails
            print(f"Failed to set CPU affinity to core {core_id}: {e}")
            return -1

    def assign_agent_pool_cores(self, num_agents: int) -> dict[int, int]:
        """
        Assign agent pool to optimal cores for maximum throughput.

        Args:
            num_agents: Number of agents in pool

        Returns:
            Dict mapping agent_id -> core_id
        """
        assignments = {}

        # Distribute agents across both P and E cores
        # Use P-cores for primary agents (heavy lifting)
        # Use E-cores for coordination and I/O

        for agent_id in range(num_agents):
            if agent_id < 6:  # First 6 agents on P-cores
                core_id = self.p_cores[agent_id % len(self.p_cores)]
            else:  # Remaining agents on E-cores
                e_core_idx = (agent_id - 6) % len(self.e_cores)
                core_id = self.e_cores[e_core_idx]

            assignments[agent_id] = core_id
            self.core_assignments[core_id] = f"agent_{agent_id}"

        return assignments

    def optimize_for_throughput(self) -> None:
        """
        Apply system-wide optimizations for maximum throughput.

        PHASE 2.2 Implementation: Fix CPU underutilization
        """
        try:
            # Set process priority to high for better scheduling
            current_process = psutil.Process()
            if hasattr(current_process, "nice"):
                current_process.nice(-5)  # Higher priority

            # Optimize scheduler settings for macOS
            if hasattr(os, "sched_setscheduler"):
                # Use round-robin scheduling for better distribution
                os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(1))

            # Enable all CPU cores
            available_cores = set(range(self.total_cores))
            os.sched_setaffinity(0, available_cores)

            print(f"✅ CPU optimization applied: {self.total_cores} cores enabled")

        except Exception as e:
            print(f"⚠️ CPU optimization partially failed: {e}")

    def start_monitoring(self) -> None:
        """Start CPU performance monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop CPU performance monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def get_metrics(self) -> CPUMetrics:
        """Get current CPU performance metrics."""
        return self.metrics

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            try:
                # Overall CPU utilization
                cpu_percent = psutil.cpu_percent(interval=0.1)

                # Per-core utilization
                per_core = psutil.cpu_percent(interval=0.1, percpu=True)

                # Calculate P-core and E-core averages
                p_core_usage = sum(per_core[i] for i in self.p_cores) / len(
                    self.p_cores
                )
                e_core_usage = sum(per_core[i] for i in self.e_cores) / len(
                    self.e_cores
                )

                # System metrics
                load_avg = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0
                thread_count = threading.active_count()

                # Update metrics
                self.metrics = CPUMetrics(
                    utilization_percent=cpu_percent,
                    p_core_usage=p_core_usage,
                    e_core_usage=e_core_usage,
                    context_switches=0,  # Would need more complex tracking
                    load_average=load_avg,
                    thread_count=thread_count,
                )

                # Trigger rebalancing if needed
                if cpu_percent < 50:  # Low utilization detected
                    self._suggest_load_increase()

                time.sleep(self.rebalance_interval)

            except Exception as e:
                print(f"CPU monitoring error: {e}")
                time.sleep(1.0)

    def _get_core_load(self, core_id: int) -> float:
        """Get load for a specific core (simplified)."""
        try:
            per_core = psutil.cpu_percent(interval=0.01, percpu=True)
            return per_core[core_id] if core_id < len(per_core) else 0.0
        except (ImportError, AttributeError, IndexError) as e:
            logger.debug(f"Could not get CPU usage for core {core_id}: {e}")
            return 0.0

    def _suggest_load_increase(self) -> None:
        """Suggest increasing load when CPU is underutilized."""
        if self.metrics.utilization_percent < 30:
            print(
                f"⚡ CPU underutilized ({self.metrics.utilization_percent:.1f}%) - "
                "consider increasing agent count or batch size"
            )


# Global instance for easy access
_cpu_optimizer: M4ProCPUOptimizer | None = None


def get_cpu_optimizer() -> M4ProCPUOptimizer:
    """Get global CPU optimizer instance."""
    global _cpu_optimizer
    if _cpu_optimizer is None:
        _cpu_optimizer = M4ProCPUOptimizer()
    return _cpu_optimizer


def assign_current_thread_to_core(task_type: str) -> int:
    """
    Convenience function to assign current thread to optimal core.

    Args:
        task_type: Type of task ('compute', 'io', 'coordination', 'memory')

    Returns:
        Assigned core ID
    """
    optimizer = get_cpu_optimizer()
    return optimizer.assign_task_to_core(task_type)


def optimize_system_for_throughput() -> None:
    """Apply system-wide CPU optimizations."""
    optimizer = get_cpu_optimizer()
    optimizer.optimize_for_throughput()
    optimizer.start_monitoring()
