#!/usr/bin/env python3
"""
Unified Performance Monitoring System for Consolidation Tracking
Monitors all subsystems and tracks consolidation benefits
"""

import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HardwareMetrics:
    """Real-time hardware utilization metrics"""

    timestamp: float

    # CPU metrics
    cpu_percent_overall: float
    cpu_percent_per_core: list[float]
    cpu_frequency_mhz: float
    p_cores_active: int
    e_cores_active: int

    # Memory metrics
    memory_total_gb: float
    memory_used_gb: float
    memory_available_gb: float
    memory_percent: float
    memory_pressure: str  # 'low', 'medium', 'high'

    # GPU metrics
    gpu_utilization_percent: float
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    gpu_cores_active: int

    # ANE metrics (Apple Neural Engine)
    ane_utilization_percent: float
    ane_power_watts: float


@dataclass
class SubsystemMetrics:
    """Performance metrics for individual subsystems"""

    name: str
    timestamp: float

    # Query performance
    query_count: int = 0
    query_latency_p50_ms: float = 0.0
    query_latency_p95_ms: float = 0.0
    query_latency_p99_ms: float = 0.0

    # Cache performance
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    cache_size_mb: float = 0.0

    # Throughput
    operations_per_second: float = 0.0
    bytes_processed: int = 0

    # Resource usage
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage_percent: float = 0.0

    # Health
    errors: int = 0
    warnings: int = 0
    status: str = "healthy"  # 'healthy', 'degraded', 'critical'


@dataclass
class ConsolidationMetrics:
    """Metrics tracking consolidation progress and benefits"""

    timestamp: float

    # Before consolidation (baseline)
    baseline_subsystems: int = 0
    baseline_total_memory_mb: float = 0.0
    baseline_total_cpu_percent: float = 0.0
    baseline_avg_latency_ms: float = 0.0
    baseline_duplicate_operations: int = 0

    # After consolidation
    unified_subsystems: int = 0
    unified_memory_mb: float = 0.0
    unified_cpu_percent: float = 0.0
    unified_avg_latency_ms: float = 0.0
    unified_cache_efficiency: float = 0.0

    # Improvements
    memory_reduction_percent: float = 0.0
    cpu_reduction_percent: float = 0.0
    latency_improvement_percent: float = 0.0
    duplicate_elimination_percent: float = 0.0

    # Progress tracking
    consolidation_phase: str = (
        "planning"  # 'planning', 'implementing', 'testing', 'complete'
    )
    components_migrated: list[str] = field(default_factory=list)
    components_remaining: list[str] = field(default_factory=list)
    estimated_completion_hours: float = 0.0


class UnifiedMonitor:
    """Central monitoring system for all performance metrics"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

        # Monitoring configuration
        self.sample_interval = self.config.get("sample_interval", 1.0)
        self.history_size = self.config.get(
            "history_size", 3600
        )  # 1 hour at 1s intervals
        self.alert_thresholds = self.config.get(
            "alert_thresholds",
            {
                "cpu_percent": 85.0,
                "memory_percent": 85.0,
                "gpu_percent": 90.0,
                "latency_p95_ms": 100.0,
                "error_rate": 0.01,
            },
        )

        # Metrics storage
        self.hardware_history = deque(maxlen=self.history_size)
        self.subsystem_metrics = defaultdict(lambda: deque(maxlen=self.history_size))
        self.consolidation_history = deque(maxlen=self.history_size)

        # Current state
        self.subsystems = {}
        self.baseline_captured = False
        self.baseline_metrics = None

        # Monitoring state
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()

        # Alert callbacks
        self.alert_callbacks = []

        # Performance counters
        self._operation_counters = defaultdict(int)
        self._latency_samples = defaultdict(list)

        logger.info("🔍 Unified Monitor initialized")

    def start_monitoring(self):
        """Start background monitoring"""
        if self._monitoring:
            logger.warning("Monitoring already active")
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitor_thread.start()

        logger.info(f"📊 Monitoring started (interval: {self.sample_interval}s)")

    def stop_monitoring(self):
        """Stop monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

        logger.info("📊 Monitoring stopped")

    def register_subsystem(self, name: str, metrics_provider: Any | None = None):
        """Register a subsystem for monitoring"""
        with self._lock:
            self.subsystems[name] = {
                "registered": datetime.now(),
                "provider": metrics_provider,
                "status": "active",
            }

        logger.info(f"✅ Subsystem registered: {name}")

    def capture_baseline(self):
        """Capture baseline metrics before consolidation"""
        with self._lock:
            # Collect current metrics as baseline
            hardware = self._collect_hardware_metrics()
            subsystem_stats = self._aggregate_subsystem_metrics()

            self.baseline_metrics = {
                "timestamp": time.time(),
                "hardware": hardware,
                "subsystems": subsystem_stats,
                "total_subsystems": len(self.subsystems),
                "total_memory_mb": sum(
                    s.memory_usage_mb for s in subsystem_stats.values()
                ),
                "total_cpu_percent": sum(
                    s.cpu_usage_percent for s in subsystem_stats.values()
                ),
                "avg_latency_ms": self._calculate_average_latency(),
            }

            self.baseline_captured = True

        logger.info("📸 Baseline metrics captured for consolidation tracking")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                # Collect metrics
                hardware_metrics = self._collect_hardware_metrics()
                subsystem_metrics = self._collect_subsystem_metrics()
                consolidation_metrics = self._calculate_consolidation_metrics()

                # Store metrics
                with self._lock:
                    self.hardware_history.append(hardware_metrics)

                    for name, metrics in subsystem_metrics.items():
                        self.subsystem_metrics[name].append(metrics)

                    if consolidation_metrics:
                        self.consolidation_history.append(consolidation_metrics)

                # Check for alerts
                self._check_alerts(hardware_metrics, subsystem_metrics)

                # Sleep until next sample
                time.sleep(self.sample_interval)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5.0)

    def _collect_hardware_metrics(self) -> HardwareMetrics:
        """Collect current hardware metrics"""
        timestamp = time.time()

        # Default values
        cpu_percent = 0.0
        cpu_per_core = []
        cpu_freq = 0.0
        memory_total = 0.0
        memory_used = 0.0
        memory_available = 0.0
        memory_percent = 0.0

        if HAS_PSUTIL:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
                cpu_freq_info = psutil.cpu_freq()
                cpu_freq = cpu_freq_info.current if cpu_freq_info else 0.0

                # Memory metrics
                memory = psutil.virtual_memory()
                memory_total = memory.total / (1024**3)
                memory_used = memory.used / (1024**3)
                memory_available = memory.available / (1024**3)
                memory_percent = memory.percent

            except Exception as e:
                logger.debug(f"psutil metrics error: {e}")

        # Estimate core types (M4 Pro specific)
        p_cores_active = sum(
            1 for i, usage in enumerate(cpu_per_core[:8]) if usage > 10
        )
        e_cores_active = sum(
            1 for i, usage in enumerate(cpu_per_core[8:]) if usage > 10
        )

        # Memory pressure assessment
        if memory_percent < 70:
            memory_pressure = "low"
        elif memory_percent < 85:
            memory_pressure = "medium"
        else:
            memory_pressure = "high"

        # GPU metrics (placeholder - would need Metal API)
        gpu_utilization = self._estimate_gpu_utilization()
        gpu_memory_used = 0.0
        gpu_memory_total = 16384.0  # M4 Pro has 16GB shared
        gpu_cores_active = 0

        # ANE metrics (placeholder - would need CoreML API)
        ane_utilization = 0.0
        ane_power = 0.0

        return HardwareMetrics(
            timestamp=timestamp,
            cpu_percent_overall=cpu_percent,
            cpu_percent_per_core=cpu_per_core,
            cpu_frequency_mhz=cpu_freq,
            p_cores_active=p_cores_active,
            e_cores_active=e_cores_active,
            memory_total_gb=memory_total,
            memory_used_gb=memory_used,
            memory_available_gb=memory_available,
            memory_percent=memory_percent,
            memory_pressure=memory_pressure,
            gpu_utilization_percent=gpu_utilization,
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_total_mb=gpu_memory_total,
            gpu_cores_active=gpu_cores_active,
            ane_utilization_percent=ane_utilization,
            ane_power_watts=ane_power,
        )

    def _collect_subsystem_metrics(self) -> dict[str, SubsystemMetrics]:
        """Collect metrics from all registered subsystems"""
        metrics = {}

        for name, info in self.subsystems.items():
            try:
                # Get metrics from provider if available
                if info["provider"] and hasattr(info["provider"], "get_metrics"):
                    raw_metrics = info["provider"].get_metrics()
                    metrics[name] = self._parse_subsystem_metrics(name, raw_metrics)
                else:
                    # Use tracked metrics
                    metrics[name] = self._calculate_subsystem_metrics(name)

            except Exception as e:
                logger.debug(f"Metrics collection error for {name}: {e}")
                metrics[name] = SubsystemMetrics(
                    name=name, timestamp=time.time(), status="error"
                )

        return metrics

    def _calculate_subsystem_metrics(self, name: str) -> SubsystemMetrics:
        """Calculate metrics for a subsystem from tracked data"""
        timestamp = time.time()

        # Get recent operations
        ops_key = f"{name}_operations"
        ops_count = self._operation_counters.get(ops_key, 0)

        # Calculate latencies
        latency_key = f"{name}_latency"
        latencies = self._latency_samples.get(latency_key, [])

        if latencies:
            latencies_sorted = sorted(latencies)
            p50 = latencies_sorted[int(len(latencies) * 0.5)]
            p95 = latencies_sorted[int(len(latencies) * 0.95)]
            p99 = latencies_sorted[int(len(latencies) * 0.99)]
        else:
            p50 = p95 = p99 = 0.0

        # Cache metrics
        cache_hits = self._operation_counters.get(f"{name}_cache_hits", 0)
        cache_misses = self._operation_counters.get(f"{name}_cache_misses", 0)
        cache_total = cache_hits + cache_misses
        cache_hit_rate = cache_hits / cache_total if cache_total > 0 else 0.0

        return SubsystemMetrics(
            name=name,
            timestamp=timestamp,
            query_count=ops_count,
            query_latency_p50_ms=p50,
            query_latency_p95_ms=p95,
            query_latency_p99_ms=p99,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_hit_rate=cache_hit_rate,
            operations_per_second=ops_count / self.sample_interval,
            status="healthy"
            if p95 < self.alert_thresholds["latency_p95_ms"]
            else "degraded",
        )

    def _calculate_consolidation_metrics(self) -> ConsolidationMetrics | None:
        """Calculate consolidation progress and benefits"""
        if not self.baseline_captured or not self.baseline_metrics:
            return None

        timestamp = time.time()

        # Current state
        current_subsystems = len(self.subsystems)
        current_metrics = self._aggregate_subsystem_metrics()

        # Calculate current totals
        current_memory = sum(m.memory_usage_mb for m in current_metrics.values())
        current_cpu = sum(m.cpu_usage_percent for m in current_metrics.values())
        current_latency = self._calculate_average_latency()

        # Calculate improvements
        baseline = self.baseline_metrics
        memory_reduction = (
            (
                (baseline["total_memory_mb"] - current_memory)
                / baseline["total_memory_mb"]
                * 100
            )
            if baseline["total_memory_mb"] > 0
            else 0
        )

        cpu_reduction = (
            (
                (baseline["total_cpu_percent"] - current_cpu)
                / baseline["total_cpu_percent"]
                * 100
            )
            if baseline["total_cpu_percent"] > 0
            else 0
        )

        latency_improvement = (
            (
                (baseline["avg_latency_ms"] - current_latency)
                / baseline["avg_latency_ms"]
                * 100
            )
            if baseline["avg_latency_ms"] > 0
            else 0
        )

        # Estimate phase and progress
        phase = self._estimate_consolidation_phase()
        migrated, remaining = self._get_migration_status()

        return ConsolidationMetrics(
            timestamp=timestamp,
            baseline_subsystems=baseline["total_subsystems"],
            baseline_total_memory_mb=baseline["total_memory_mb"],
            baseline_total_cpu_percent=baseline["total_cpu_percent"],
            baseline_avg_latency_ms=baseline["avg_latency_ms"],
            unified_subsystems=current_subsystems,
            unified_memory_mb=current_memory,
            unified_cpu_percent=current_cpu,
            unified_avg_latency_ms=current_latency,
            memory_reduction_percent=memory_reduction,
            cpu_reduction_percent=cpu_reduction,
            latency_improvement_percent=latency_improvement,
            consolidation_phase=phase,
            components_migrated=migrated,
            components_remaining=remaining,
        )

    def _aggregate_subsystem_metrics(self) -> dict[str, SubsystemMetrics]:
        """Get latest metrics for all subsystems"""
        latest = {}

        for name, history in self.subsystem_metrics.items():
            if history:
                latest[name] = history[-1]

        return latest

    def _calculate_average_latency(self) -> float:
        """Calculate average latency across all subsystems"""
        latencies = []

        for samples in self._latency_samples.values():
            latencies.extend(samples)

        return sum(latencies) / len(latencies) if latencies else 0.0

    def _estimate_gpu_utilization(self) -> float:
        """Estimate GPU utilization from MLX if available"""
        if HAS_MLX:
            try:
                # This is a placeholder - actual implementation would need
                # to track MLX operations or use Metal Performance Shaders
                return 0.0
            except Exception:
                pass

        return 0.0

    def _estimate_consolidation_phase(self) -> str:
        """Estimate current consolidation phase"""
        # This would be updated based on actual migration progress
        if not self.baseline_captured:
            return "planning"

        # Check migration progress
        migrated = len(
            [s for s in self.subsystems.values() if s.get("migrated", False)]
        )
        total = len(self.subsystems)

        if migrated == 0:
            return "planning"
        elif migrated < total * 0.3:
            return "implementing"
        elif migrated < total:
            return "testing"
        else:
            return "complete"

    def _get_migration_status(self) -> tuple[list[str], list[str]]:
        """Get lists of migrated and remaining components"""
        migrated = []
        remaining = []

        for name, info in self.subsystems.items():
            if info.get("migrated", False):
                migrated.append(name)
            else:
                remaining.append(name)

        return migrated, remaining

    def _check_alerts(
        self, hardware: HardwareMetrics, subsystems: dict[str, SubsystemMetrics]
    ):
        """Check for alert conditions"""
        alerts = []

        # Hardware alerts
        if hardware.cpu_percent_overall > self.alert_thresholds["cpu_percent"]:
            alerts.append(f"High CPU usage: {hardware.cpu_percent_overall:.1f}%")

        if hardware.memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append(f"High memory usage: {hardware.memory_percent:.1f}%")

        if hardware.gpu_utilization_percent > self.alert_thresholds["gpu_percent"]:
            alerts.append(f"High GPU usage: {hardware.gpu_utilization_percent:.1f}%")

        # Subsystem alerts
        for name, metrics in subsystems.items():
            if metrics.query_latency_p95_ms > self.alert_thresholds["latency_p95_ms"]:
                alerts.append(
                    f"{name}: High latency P95={metrics.query_latency_p95_ms:.1f}ms"
                )

            if metrics.errors > 0:
                error_rate = metrics.errors / max(metrics.query_count, 1)
                if error_rate > self.alert_thresholds["error_rate"]:
                    alerts.append(f"{name}: High error rate {error_rate:.1%}")

        # Trigger callbacks
        for alert in alerts:
            logger.warning(f"⚠️ Alert: {alert}")
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

    def record_operation(
        self,
        subsystem: str,
        operation: str,
        latency_ms: float,
        success: bool = True,
        cached: bool = False,
    ):
        """Record an operation for a subsystem"""
        with self._lock:
            # Update counters
            self._operation_counters[f"{subsystem}_operations"] += 1

            if cached:
                self._operation_counters[f"{subsystem}_cache_hits"] += 1
            else:
                self._operation_counters[f"{subsystem}_cache_misses"] += 1

            if not success:
                self._operation_counters[f"{subsystem}_errors"] += 1

            # Store latency sample
            latency_key = f"{subsystem}_latency"
            if latency_key not in self._latency_samples:
                self._latency_samples[latency_key] = deque(maxlen=1000)

            self._latency_samples[latency_key].append(latency_ms)

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get real-time dashboard data"""
        with self._lock:
            # Latest metrics
            latest_hardware = (
                self.hardware_history[-1] if self.hardware_history else None
            )
            latest_consolidation = (
                self.consolidation_history[-1] if self.consolidation_history else None
            )

            # Subsystem summaries
            subsystem_summary = {}
            for name, history in self.subsystem_metrics.items():
                if history:
                    latest = history[-1]
                    subsystem_summary[name] = {
                        "status": latest.status,
                        "latency_p95": latest.query_latency_p95_ms,
                        "cache_hit_rate": latest.cache_hit_rate,
                        "ops_per_sec": latest.operations_per_second,
                    }

            return {
                "timestamp": time.time(),
                "hardware": asdict(latest_hardware) if latest_hardware else None,
                "consolidation": asdict(latest_consolidation)
                if latest_consolidation
                else None,
                "subsystems": subsystem_summary,
                "alerts": self._get_active_alerts(),
            }

    def _get_active_alerts(self) -> list[str]:
        """Get currently active alerts"""
        alerts = []

        if self.hardware_history:
            latest = self.hardware_history[-1]

            if latest.cpu_percent_overall > self.alert_thresholds["cpu_percent"]:
                alerts.append(f"CPU: {latest.cpu_percent_overall:.1f}%")

            if latest.memory_percent > self.alert_thresholds["memory_percent"]:
                alerts.append(f"Memory: {latest.memory_percent:.1f}%")

        return alerts

    def generate_report(self, period_minutes: int = 60) -> dict[str, Any]:
        """Generate performance report for specified period"""
        with self._lock:
            cutoff_time = time.time() - (period_minutes * 60)

            # Filter metrics by time
            hardware_in_period = [
                m for m in self.hardware_history if m.timestamp > cutoff_time
            ]
            consolidation_in_period = [
                m for m in self.consolidation_history if m.timestamp > cutoff_time
            ]

            if not hardware_in_period:
                return {"error": "No data available for period"}

            # Hardware statistics
            cpu_values = [m.cpu_percent_overall for m in hardware_in_period]
            memory_values = [m.memory_percent for m in hardware_in_period]
            gpu_values = [m.gpu_utilization_percent for m in hardware_in_period]

            # Consolidation progress
            consolidation_summary = None
            if consolidation_in_period:
                latest_consolidation = consolidation_in_period[-1]
                consolidation_summary = {
                    "phase": latest_consolidation.consolidation_phase,
                    "memory_saved": f"{latest_consolidation.memory_reduction_percent:.1f}%",
                    "cpu_saved": f"{latest_consolidation.cpu_reduction_percent:.1f}%",
                    "latency_improved": f"{latest_consolidation.latency_improvement_percent:.1f}%",
                    "components_migrated": len(
                        latest_consolidation.components_migrated
                    ),
                    "components_remaining": len(
                        latest_consolidation.components_remaining
                    ),
                }

            return {
                "period_minutes": period_minutes,
                "samples": len(hardware_in_period),
                "hardware": {
                    "cpu": {
                        "average": sum(cpu_values) / len(cpu_values),
                        "peak": max(cpu_values),
                        "p95": sorted(cpu_values)[int(len(cpu_values) * 0.95)],
                    },
                    "memory": {
                        "average": sum(memory_values) / len(memory_values),
                        "peak": max(memory_values),
                        "p95": sorted(memory_values)[int(len(memory_values) * 0.95)],
                    },
                    "gpu": {
                        "average": sum(gpu_values) / len(gpu_values),
                        "peak": max(gpu_values),
                        "p95": sorted(gpu_values)[int(len(gpu_values) * 0.95)],
                    },
                },
                "consolidation": consolidation_summary,
                "subsystems": self._summarize_subsystems(period_minutes),
            }

    def _summarize_subsystems(self, period_minutes: int) -> dict[str, Any]:
        """Summarize subsystem performance"""
        cutoff_time = time.time() - (period_minutes * 60)
        summary = {}

        for name, history in self.subsystem_metrics.items():
            metrics_in_period = [m for m in history if m.timestamp > cutoff_time]

            if metrics_in_period:
                latencies = [m.query_latency_p95_ms for m in metrics_in_period]
                cache_rates = [m.cache_hit_rate for m in metrics_in_period]

                summary[name] = {
                    "samples": len(metrics_in_period),
                    "avg_latency_p95": sum(latencies) / len(latencies),
                    "avg_cache_hit_rate": sum(cache_rates) / len(cache_rates),
                    "total_operations": sum(m.query_count for m in metrics_in_period),
                    "total_errors": sum(m.errors for m in metrics_in_period),
                }

        return summary


# Global monitor instance
_monitor = None


def get_unified_monitor() -> UnifiedMonitor:
    """Get global unified monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = UnifiedMonitor()
    return _monitor


# Context manager for operation tracking
class MonitoredOperation:
    """Context manager for tracking operations"""

    def __init__(self, subsystem: str, operation: str):
        self.subsystem = subsystem
        self.operation = operation
        self.monitor = get_unified_monitor()
        self.start_time = None
        self.cached = False
        self.success = True

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            latency_ms = (time.time() - self.start_time) * 1000
            self.success = exc_type is None

            self.monitor.record_operation(
                self.subsystem,
                self.operation,
                latency_ms,
                success=self.success,
                cached=self.cached,
            )

    def mark_cached(self):
        """Mark this operation as served from cache"""
        self.cached = True
