#!/usr/bin/env python3
"""
Performance Monitoring for 8-Agent Bolt System
Real-time hardware usage tracking and bottleneck detection
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from .hardware_state import get_hardware_state

    HAS_HARDWARE_STATE = True
except ImportError:
    HAS_HARDWARE_STATE = False

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Metrics for individual agent performance"""

    agent_id: str
    cpu_core: int | None
    cpu_percent: float
    memory_mb: float
    gpu_percent: float
    tasks_completed: int
    tasks_failed: int
    average_task_time_ms: float
    current_task: str | None
    timestamp: float


@dataclass
class SystemMetrics:
    """Overall system performance metrics"""

    timestamp: float

    # CPU metrics
    cpu_p_cores_used: int
    cpu_e_cores_used: int
    cpu_overall_percent: float
    cpu_frequency_mhz: float

    # Memory metrics
    memory_total_gb: float
    memory_used_gb: float
    memory_available_gb: float
    memory_percent: float

    # GPU metrics
    gpu_utilization: float
    gpu_memory_mb: float
    gpu_cores_active: int

    # Agent metrics
    agents_active: int
    agents_idle: int
    total_tasks_queued: int
    total_tasks_completed: int

    # Performance indicators
    bottleneck_detected: str | None
    health_status: str  # 'healthy', 'warning', 'critical'


class PerformanceMonitor:
    """Real-time performance monitoring for Bolt system"""

    def __init__(self, max_history: int = 300):  # 5 minutes at 1s intervals
        self.max_history = max_history
        self.agent_metrics = {}
        self.system_metrics_history = deque(maxlen=max_history)
        self.agent_metrics_history = defaultdict(lambda: deque(maxlen=max_history))

        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()

        # Performance thresholds
        self.thresholds = {
            "cpu_warning": 85.0,
            "cpu_critical": 95.0,
            "memory_warning": 85.0,
            "memory_critical": 95.0,
            "gpu_warning": 90.0,
            "gpu_critical": 98.0,
            "task_time_warning": 30000,  # 30s
            "task_time_critical": 60000,  # 60s
        }

        # Hardware state integration
        self.hardware_state = None
        if HAS_HARDWARE_STATE:
            try:
                self.hardware_state = get_hardware_state()
            except Exception as e:
                logger.warning(f"Hardware state unavailable: {e}")

        logger.info("📊 Performance Monitor initialized")

    def start_monitoring(self, interval: float = 1.0):
        """Start background performance monitoring"""

        if self._monitoring:
            logger.warning("Performance monitoring already running")
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval,), daemon=True
        )
        self._monitor_thread.start()

        logger.info(f"📈 Performance monitoring started (interval: {interval}s)")

    def stop_monitoring(self):
        """Stop background monitoring"""

        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

        logger.info("📊 Performance monitoring stopped")

    def _monitoring_loop(self, interval: float):
        """Background monitoring loop"""

        while self._monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()

                with self._lock:
                    self.system_metrics_history.append(system_metrics)

                # Detect bottlenecks
                bottleneck = self._detect_bottlenecks(system_metrics)
                if bottleneck:
                    logger.warning(f"🚨 Performance bottleneck detected: {bottleneck}")

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)  # Longer sleep on error

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics"""

        timestamp = time.time()

        # Default values
        cpu_percent = 0.0
        memory_total_gb = 0.0
        memory_used_gb = 0.0
        memory_available_gb = 0.0
        memory_percent = 0.0
        cpu_freq = 0.0

        # Collect system metrics if psutil available
        if HAS_PSUTIL:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                memory_total_gb = memory.total / (1024**3)
                memory_used_gb = memory.used / (1024**3)
                memory_available_gb = memory.available / (1024**3)
                memory_percent = memory.percent

                cpu_freq_info = psutil.cpu_freq()
                cpu_freq = cpu_freq_info.current if cpu_freq_info else 0.0

            except Exception as e:
                logger.debug(f"System metrics collection error: {e}")

        # Hardware-specific metrics
        cpu_p_cores_used = 0
        cpu_e_cores_used = 0
        gpu_utilization = 0.0
        gpu_memory_mb = 0.0
        gpu_cores_active = 0

        if self.hardware_state:
            try:
                # Get CPU core usage
                cpu_info = self.hardware_state.cpu
                cpu_p_cores_used = min(
                    cpu_info.p_cores,
                    len([a for a in self.agent_metrics.values() if a.cpu_percent > 10]),
                )

                # Get GPU metrics
                gpu_info = self.hardware_state.gpu
                if hasattr(gpu_info, "utilization"):
                    gpu_utilization = gpu_info.utilization
                if hasattr(gpu_info, "memory_used_mb"):
                    gpu_memory_mb = gpu_info.memory_used_mb
                if hasattr(gpu_info, "cores"):
                    gpu_cores_active = gpu_info.cores

            except Exception as e:
                logger.debug(f"Hardware state metrics error: {e}")

        # Agent metrics
        agents_active = len([a for a in self.agent_metrics.values() if a.current_task])
        agents_idle = len(self.agent_metrics) - agents_active
        total_tasks_completed = sum(
            a.tasks_completed for a in self.agent_metrics.values()
        )
        total_tasks_queued = 0  # Would need queue reference

        # Health assessment
        health_status = self._assess_health(
            cpu_percent, memory_percent, gpu_utilization
        )
        bottleneck = self._detect_current_bottleneck(
            cpu_percent, memory_percent, gpu_utilization
        )

        return SystemMetrics(
            timestamp=timestamp,
            cpu_p_cores_used=cpu_p_cores_used,
            cpu_e_cores_used=cpu_e_cores_used,
            cpu_overall_percent=cpu_percent,
            cpu_frequency_mhz=cpu_freq,
            memory_total_gb=memory_total_gb,
            memory_used_gb=memory_used_gb,
            memory_available_gb=memory_available_gb,
            memory_percent=memory_percent,
            gpu_utilization=gpu_utilization,
            gpu_memory_mb=gpu_memory_mb,
            gpu_cores_active=gpu_cores_active,
            agents_active=agents_active,
            agents_idle=agents_idle,
            total_tasks_queued=total_tasks_queued,
            total_tasks_completed=total_tasks_completed,
            bottleneck_detected=bottleneck,
            health_status=health_status,
        )

    def register_agent(self, agent_id: str, cpu_core: int | None = None):
        """Register an agent for monitoring"""

        with self._lock:
            self.agent_metrics[agent_id] = AgentMetrics(
                agent_id=agent_id,
                cpu_core=cpu_core,
                cpu_percent=0.0,
                memory_mb=0.0,
                gpu_percent=0.0,
                tasks_completed=0,
                tasks_failed=0,
                average_task_time_ms=0.0,
                current_task=None,
                timestamp=time.time(),
            )

        logger.debug(f"📊 Agent {agent_id} registered for monitoring")

    def update_agent_metrics(self, agent_id: str, **metrics):
        """Update metrics for a specific agent"""

        with self._lock:
            if agent_id in self.agent_metrics:
                agent = self.agent_metrics[agent_id]

                # Update provided metrics
                for key, value in metrics.items():
                    if hasattr(agent, key):
                        setattr(agent, key, value)

                agent.timestamp = time.time()

                # Store in history
                self.agent_metrics_history[agent_id].append(agent)

    def _assess_health(
        self, cpu_percent: float, memory_percent: float, gpu_percent: float
    ) -> str:
        """Assess overall system health"""

        if (
            cpu_percent > self.thresholds["cpu_critical"]
            or memory_percent > self.thresholds["memory_critical"]
            or gpu_percent > self.thresholds["gpu_critical"]
        ):
            return "critical"

        if (
            cpu_percent > self.thresholds["cpu_warning"]
            or memory_percent > self.thresholds["memory_warning"]
            or gpu_percent > self.thresholds["gpu_warning"]
        ):
            return "warning"

        return "healthy"

    def _detect_current_bottleneck(
        self, cpu_percent: float, memory_percent: float, gpu_percent: float
    ) -> str | None:
        """Detect current performance bottleneck"""

        bottlenecks = []

        if cpu_percent > self.thresholds["cpu_warning"]:
            bottlenecks.append(f"CPU ({cpu_percent:.1f}%)")

        if memory_percent > self.thresholds["memory_warning"]:
            bottlenecks.append(f"Memory ({memory_percent:.1f}%)")

        if gpu_percent > self.thresholds["gpu_warning"]:
            bottlenecks.append(f"GPU ({gpu_percent:.1f}%)")

        return ", ".join(bottlenecks) if bottlenecks else None

    def _detect_bottlenecks(self, current_metrics: SystemMetrics) -> str | None:
        """Detect performance bottlenecks with trend analysis"""

        if len(self.system_metrics_history) < 5:
            return None

        # Get recent metrics for trend analysis
        recent_metrics = list(self.system_metrics_history)[-5:]

        # Check for sustained high usage
        avg_cpu = sum(m.cpu_overall_percent for m in recent_metrics) / len(
            recent_metrics
        )
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_gpu = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)

        bottlenecks = []

        if avg_cpu > self.thresholds["cpu_warning"]:
            bottlenecks.append(f"Sustained CPU load ({avg_cpu:.1f}%)")

        if avg_memory > self.thresholds["memory_warning"]:
            bottlenecks.append(f"Sustained memory usage ({avg_memory:.1f}%)")

        if avg_gpu > self.thresholds["gpu_warning"]:
            bottlenecks.append(f"Sustained GPU usage ({avg_gpu:.1f}%)")

        # Check for stuck agents
        stuck_agents = [
            agent_id
            for agent_id, agent in self.agent_metrics.items()
            if agent.current_task
            and agent.average_task_time_ms > self.thresholds["task_time_warning"]
        ]

        if stuck_agents:
            bottlenecks.append(f"Slow agents: {', '.join(stuck_agents)}")

        return "; ".join(bottlenecks) if bottlenecks else None

    def get_real_time_dashboard(self) -> dict[str, Any]:
        """Get real-time performance dashboard"""

        with self._lock:
            current_system = (
                self.system_metrics_history[-1] if self.system_metrics_history else None
            )
            current_agents = dict(self.agent_metrics)

        if not current_system:
            return {"error": "No system metrics available"}

        # Calculate performance trends
        trends = self._calculate_trends()

        return {
            "timestamp": current_system.timestamp,
            "system": asdict(current_system),
            "agents": {
                agent_id: asdict(metrics)
                for agent_id, metrics in current_agents.items()
            },
            "trends": trends,
            "summary": {
                "health": current_system.health_status,
                "bottleneck": current_system.bottleneck_detected,
                "agents_working": current_system.agents_active,
                "total_agents": len(current_agents),
                "tasks_completed": current_system.total_tasks_completed,
            },
        }

    def _calculate_trends(self) -> dict[str, str]:
        """Calculate performance trends"""

        if len(self.system_metrics_history) < 2:
            return {}

        current = self.system_metrics_history[-1]
        previous = self.system_metrics_history[-2]

        def trend_indicator(current_val: float, previous_val: float) -> str:
            if current_val > previous_val * 1.1:
                return "📈 increasing"
            elif current_val < previous_val * 0.9:
                return "📉 decreasing"
            else:
                return "➡️ stable"

        return {
            "cpu": trend_indicator(
                current.cpu_overall_percent, previous.cpu_overall_percent
            ),
            "memory": trend_indicator(current.memory_percent, previous.memory_percent),
            "gpu": trend_indicator(current.gpu_utilization, previous.gpu_utilization),
            "agents": trend_indicator(current.agents_active, previous.agents_active),
        }

    def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report"""

        with self._lock:
            if not self.system_metrics_history:
                return {"error": "No performance data available"}

            # Calculate statistics from history
            metrics_list = list(self.system_metrics_history)

            cpu_values = [m.cpu_overall_percent for m in metrics_list]
            memory_values = [m.memory_percent for m in metrics_list]
            gpu_values = [m.gpu_utilization for m in metrics_list]

            return {
                "monitoring_duration_minutes": len(metrics_list) / 60,
                "total_samples": len(metrics_list),
                "cpu": {
                    "average": sum(cpu_values) / len(cpu_values),
                    "peak": max(cpu_values),
                    "p95": sorted(cpu_values)[int(len(cpu_values) * 0.95)]
                    if cpu_values
                    else 0,
                },
                "memory": {
                    "average": sum(memory_values) / len(memory_values),
                    "peak": max(memory_values),
                    "p95": sorted(memory_values)[int(len(memory_values) * 0.95)]
                    if memory_values
                    else 0,
                },
                "gpu": {
                    "average": sum(gpu_values) / len(gpu_values),
                    "peak": max(gpu_values),
                    "p95": sorted(gpu_values)[int(len(gpu_values) * 0.95)]
                    if gpu_values
                    else 0,
                },
                "agents": {
                    "total_registered": len(self.agent_metrics),
                    "tasks_completed": sum(
                        a.tasks_completed for a in self.agent_metrics.values()
                    ),
                    "tasks_failed": sum(
                        a.tasks_failed for a in self.agent_metrics.values()
                    ),
                },
                "health_summary": {
                    "healthy_samples": len(
                        [m for m in metrics_list if m.health_status == "healthy"]
                    ),
                    "warning_samples": len(
                        [m for m in metrics_list if m.health_status == "warning"]
                    ),
                    "critical_samples": len(
                        [m for m in metrics_list if m.health_status == "critical"]
                    ),
                },
            }


# Global monitor instance
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


class AgentPerformanceTracker:
    """Performance tracking context manager for agents"""

    def __init__(self, agent_id: str, task_name: str):
        self.agent_id = agent_id
        self.task_name = task_name
        self.monitor = get_performance_monitor()
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.monitor.update_agent_metrics(self.agent_id, current_task=self.task_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            task_time_ms = (time.time() - self.start_time) * 1000

            # Update metrics based on success/failure
            if exc_type is None:
                self.monitor.update_agent_metrics(
                    self.agent_id,
                    tasks_completed=self.monitor.agent_metrics[
                        self.agent_id
                    ].tasks_completed
                    + 1,
                    average_task_time_ms=task_time_ms,
                    current_task=None,
                )
            else:
                self.monitor.update_agent_metrics(
                    self.agent_id,
                    tasks_failed=self.monitor.agent_metrics[self.agent_id].tasks_failed
                    + 1,
                    current_task=None,
                )

    async def __aenter__(self):
        self.start_time = time.time()
        self.monitor.update_agent_metrics(self.agent_id, current_task=self.task_name)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            task_time_ms = (time.time() - self.start_time) * 1000

            # Update metrics based on success/failure
            if exc_type is None:
                self.monitor.update_agent_metrics(
                    self.agent_id,
                    tasks_completed=self.monitor.agent_metrics[
                        self.agent_id
                    ].tasks_completed
                    + 1,
                    average_task_time_ms=task_time_ms,
                    current_task=None,
                )
            else:
                self.monitor.update_agent_metrics(
                    self.agent_id,
                    tasks_failed=self.monitor.agent_metrics[self.agent_id].tasks_failed
                    + 1,
                    current_task=None,
                )


if __name__ == "__main__":

    async def test_performance_monitoring():
        """Test performance monitoring system"""

        print("📊 Testing Performance Monitor")
        print("=" * 50)

        monitor = get_performance_monitor()
        monitor.start_monitoring(interval=0.5)

        # Register some test agents
        for i in range(3):
            monitor.register_agent(f"agent_{i}", cpu_core=i)

        # Simulate some agent activity
        for i in range(10):
            agent_id = f"agent_{i % 3}"

            with AgentPerformanceTracker(agent_id, f"test_task_{i}"):
                await asyncio.sleep(0.1)  # Simulate work

            # Show dashboard every few iterations
            if i % 3 == 0:
                dashboard = monitor.get_real_time_dashboard()
                print(f"\n📈 Iteration {i}:")
                print(f"  System Health: {dashboard['summary']['health']}")
                print(f"  Active Agents: {dashboard['summary']['agents_working']}")
                print(f"  Completed Tasks: {dashboard['summary']['tasks_completed']}")
                if dashboard["summary"]["bottleneck"]:
                    print(f"  Bottleneck: {dashboard['summary']['bottleneck']}")

        # Generate final report
        print("\n📊 Final Performance Report:")
        report = monitor.get_performance_report()
        print(
            f"  Monitoring Duration: {report['monitoring_duration_minutes']:.1f} minutes"
        )
        print(f"  CPU Average: {report['cpu']['average']:.1f}%")
        print(f"  Memory Average: {report['memory']['average']:.1f}%")
        print(f"  Tasks Completed: {report['agents']['tasks_completed']}")

        monitor.stop_monitoring()

    asyncio.run(test_performance_monitoring())
