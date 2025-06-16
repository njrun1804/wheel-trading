#!/usr/bin/env python3
"""
Bolt Throughput Monitor
Real-time monitoring dashboard for production throughput metrics
"""

import asyncio
import json
import logging
import statistics
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from bolt_throughput_optimizer import ThroughputMetrics, ThroughputProfiler

logger = logging.getLogger(__name__)


@dataclass
class ThroughputAlert:
    """Alert for throughput issues"""

    timestamp: float
    alert_type: str
    severity: str  # 'warning', 'critical'
    message: str
    current_value: float
    threshold: float
    suggested_action: str


@dataclass
class PerformanceTrend:
    """Performance trend analysis"""

    metric_name: str
    current_value: float
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_magnitude: float  # percentage change
    prediction_5min: float
    prediction_15min: float


class ThroughputAlerting:
    """Alert system for throughput issues"""

    def __init__(self):
        self.thresholds = {
            "ops_per_sec_warning": 80.0,
            "ops_per_sec_critical": 50.0,
            "latency_p95_warning": 150.0,
            "latency_p95_critical": 300.0,
            "cpu_utilization_warning": 85.0,
            "cpu_utilization_critical": 95.0,
            "memory_usage_warning": 85.0,
            "memory_usage_critical": 95.0,
            "database_pool_warning": 80.0,
            "database_pool_critical": 95.0,
        }
        self.alert_history = deque(maxlen=1000)
        self.alert_cooldowns = {}  # Prevent spam
        self.cooldown_period = 300  # 5 minutes

    def check_alerts(self, metrics: ThroughputMetrics) -> list[ThroughputAlert]:
        """Check for alert conditions"""
        alerts = []
        current_time = time.time()

        # Throughput alerts
        if metrics.ops_per_sec < self.thresholds["ops_per_sec_critical"]:
            alert = self._create_alert(
                "throughput_critical",
                "critical",
                f"Throughput critically low: {metrics.ops_per_sec:.1f} ops/sec",
                metrics.ops_per_sec,
                self.thresholds["ops_per_sec_critical"],
                "Scale up resources, check for bottlenecks",
            )
            if self._should_send_alert("throughput_critical", current_time):
                alerts.append(alert)
        elif metrics.ops_per_sec < self.thresholds["ops_per_sec_warning"]:
            alert = self._create_alert(
                "throughput_warning",
                "warning",
                f"Throughput below target: {metrics.ops_per_sec:.1f} ops/sec",
                metrics.ops_per_sec,
                self.thresholds["ops_per_sec_warning"],
                "Monitor closely, prepare to scale",
            )
            if self._should_send_alert("throughput_warning", current_time):
                alerts.append(alert)

        # Latency alerts
        if metrics.latency_p95_ms > self.thresholds["latency_p95_critical"]:
            alert = self._create_alert(
                "latency_critical",
                "critical",
                f"Latency critically high: {metrics.latency_p95_ms:.1f}ms",
                metrics.latency_p95_ms,
                self.thresholds["latency_p95_critical"],
                "Investigate slow operations, optimize queries",
            )
            if self._should_send_alert("latency_critical", current_time):
                alerts.append(alert)
        elif metrics.latency_p95_ms > self.thresholds["latency_p95_warning"]:
            alert = self._create_alert(
                "latency_warning",
                "warning",
                f"Latency elevated: {metrics.latency_p95_ms:.1f}ms",
                metrics.latency_p95_ms,
                self.thresholds["latency_p95_warning"],
                "Review performance metrics",
            )
            if self._should_send_alert("latency_warning", current_time):
                alerts.append(alert)

        # CPU alerts
        if metrics.cpu_utilization > self.thresholds["cpu_utilization_critical"]:
            alert = self._create_alert(
                "cpu_critical",
                "critical",
                f"CPU critically overloaded: {metrics.cpu_utilization:.1f}%",
                metrics.cpu_utilization,
                self.thresholds["cpu_utilization_critical"],
                "Scale horizontally or reduce load",
            )
            if self._should_send_alert("cpu_critical", current_time):
                alerts.append(alert)
        elif metrics.cpu_utilization > self.thresholds["cpu_utilization_warning"]:
            alert = self._create_alert(
                "cpu_warning",
                "warning",
                f"CPU utilization high: {metrics.cpu_utilization:.1f}%",
                metrics.cpu_utilization,
                self.thresholds["cpu_utilization_warning"],
                "Monitor CPU trends",
            )
            if self._should_send_alert("cpu_warning", current_time):
                alerts.append(alert)

        # Store alerts in history
        for alert in alerts:
            self.alert_history.append(alert)

        return alerts

    def _create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        current_value: float,
        threshold: float,
        suggested_action: str,
    ) -> ThroughputAlert:
        """Create a throughput alert"""
        return ThroughputAlert(
            timestamp=time.time(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            current_value=current_value,
            threshold=threshold,
            suggested_action=suggested_action,
        )

    def _should_send_alert(self, alert_type: str, current_time: float) -> bool:
        """Check if alert should be sent (cooldown logic)"""
        if alert_type not in self.alert_cooldowns:
            self.alert_cooldowns[alert_type] = current_time
            return True

        time_since_last = current_time - self.alert_cooldowns[alert_type]
        if time_since_last >= self.cooldown_period:
            self.alert_cooldowns[alert_type] = current_time
            return True

        return False


class ThroughputTrendAnalyzer:
    """Analyze throughput trends and predictions"""

    def __init__(self, history_size: int = 300):
        self.metrics_history = deque(maxlen=history_size)

    def add_metrics(self, metrics: ThroughputMetrics):
        """Add metrics to history for trend analysis"""
        self.metrics_history.append(metrics)

    def analyze_trends(self) -> list[PerformanceTrend]:
        """Analyze performance trends"""
        if len(self.metrics_history) < 10:
            return []

        trends = []

        # Analyze ops/sec trend
        trends.append(
            self._analyze_metric_trend(
                "ops_per_sec", [m.ops_per_sec for m in self.metrics_history]
            )
        )

        # Analyze latency trend
        trends.append(
            self._analyze_metric_trend(
                "latency_p95_ms", [m.latency_p95_ms for m in self.metrics_history]
            )
        )

        # Analyze CPU trend
        trends.append(
            self._analyze_metric_trend(
                "cpu_utilization", [m.cpu_utilization for m in self.metrics_history]
            )
        )

        return trends

    def _analyze_metric_trend(
        self, metric_name: str, values: list[float]
    ) -> PerformanceTrend:
        """Analyze trend for a specific metric"""
        if len(values) < 5:
            return PerformanceTrend(
                metric_name=metric_name,
                current_value=values[-1] if values else 0.0,
                trend_direction="unknown",
                trend_magnitude=0.0,
                prediction_5min=values[-1] if values else 0.0,
                prediction_15min=values[-1] if values else 0.0,
            )

        current_value = values[-1]

        # Calculate trend using recent values
        recent_values = values[-10:]  # Last 10 measurements
        older_values = values[-20:-10] if len(values) >= 20 else values[:-10]

        if older_values:
            recent_avg = statistics.mean(recent_values)
            older_avg = statistics.mean(older_values)

            if recent_avg > older_avg * 1.05:
                trend_direction = "increasing"
                trend_magnitude = ((recent_avg - older_avg) / older_avg) * 100
            elif recent_avg < older_avg * 0.95:
                trend_direction = "decreasing"
                trend_magnitude = ((older_avg - recent_avg) / older_avg) * 100
            else:
                trend_direction = "stable"
                trend_magnitude = 0.0
        else:
            trend_direction = "stable"
            trend_magnitude = 0.0

        # Simple linear prediction (for demonstration)
        if len(values) >= 5:
            # Calculate slope of recent trend
            x = list(range(len(recent_values)))
            y = recent_values

            # Simple linear regression
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))

            if n * sum_x2 - sum_x**2 != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

                # Predict future values (assuming 1 measurement per minute)
                prediction_5min = current_value + slope * 5
                prediction_15min = current_value + slope * 15
            else:
                prediction_5min = current_value
                prediction_15min = current_value
        else:
            prediction_5min = current_value
            prediction_15min = current_value

        return PerformanceTrend(
            metric_name=metric_name,
            current_value=current_value,
            trend_direction=trend_direction,
            trend_magnitude=abs(trend_magnitude),
            prediction_5min=max(0, prediction_5min),
            prediction_15min=max(0, prediction_15min),
        )


class ThroughputDashboard:
    """Real-time throughput monitoring dashboard"""

    def __init__(self):
        self.profiler = ThroughputProfiler()
        self.alerting = ThroughputAlerting()
        self.trend_analyzer = ThroughputTrendAnalyzer()
        self.monitoring = False
        self.monitor_thread = None
        self.dashboard_data = {
            "current_metrics": None,
            "alerts": [],
            "trends": [],
            "system_health": "unknown",
            "uptime_seconds": 0,
            "total_operations": 0,
        }
        self.start_time = time.time()

    def start_monitoring(self, interval: float = 10.0):
        """Start real-time monitoring"""
        if self.monitoring:
            logger.warning("Monitoring already running")
            return

        self.monitoring = True
        self.start_time = time.time()

        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval,), daemon=True
        )
        self.monitor_thread.start()

        logger.info(f"🖥️  Throughput monitoring started (interval: {interval}s)")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        logger.info("🖥️  Throughput monitoring stopped")

    def _monitoring_loop(self, interval: float):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Calculate current metrics
                metrics = self.profiler.calculate_throughput(window_seconds=interval)

                # Update dashboard data
                self.dashboard_data["current_metrics"] = metrics
                self.dashboard_data["uptime_seconds"] = time.time() - self.start_time
                self.dashboard_data["total_operations"] += len(
                    self.profiler.operation_times
                )

                # Add to trend analysis
                self.trend_analyzer.add_metrics(metrics)

                # Check for alerts
                alerts = self.alerting.check_alerts(metrics)
                if alerts:
                    self.dashboard_data["alerts"].extend(alerts)
                    # Keep only recent alerts
                    self.dashboard_data["alerts"] = self.dashboard_data["alerts"][-50:]

                # Update trends
                self.dashboard_data["trends"] = self.trend_analyzer.analyze_trends()

                # Update system health
                self.dashboard_data["system_health"] = self._assess_system_health(
                    metrics, alerts
                )

                # Log critical alerts
                for alert in alerts:
                    if alert.severity == "critical":
                        logger.error(f"🚨 CRITICAL ALERT: {alert.message}")
                    elif alert.severity == "warning":
                        logger.warning(f"⚠️  WARNING: {alert.message}")

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)  # Longer sleep on error

    def record_operation(self, duration_ms: float, metadata: dict[str, Any] = None):
        """Record an operation for monitoring"""
        self.profiler.record_operation(duration_ms, metadata)

    def _assess_system_health(
        self, metrics: ThroughputMetrics, recent_alerts: list[ThroughputAlert]
    ) -> str:
        """Assess overall system health"""

        # Check for critical alerts
        critical_alerts = [a for a in recent_alerts if a.severity == "critical"]
        if critical_alerts:
            return "critical"

        # Check for warning alerts
        warning_alerts = [a for a in recent_alerts if a.severity == "warning"]
        if warning_alerts:
            return "warning"

        # Check key metrics
        if (
            metrics.ops_per_sec >= 100
            and metrics.latency_p95_ms <= 150
            and metrics.cpu_utilization <= 80
        ):
            return "healthy"
        elif (
            metrics.ops_per_sec >= 80
            and metrics.latency_p95_ms <= 200
            and metrics.cpu_utilization <= 90
        ):
            return "degraded"
        else:
            return "unhealthy"

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get current dashboard data"""
        dashboard = self.dashboard_data.copy()

        # Convert dataclasses to dicts for JSON serialization
        if dashboard["current_metrics"]:
            dashboard["current_metrics"] = asdict(dashboard["current_metrics"])

        dashboard["alerts"] = [asdict(alert) for alert in dashboard["alerts"]]
        dashboard["trends"] = [asdict(trend) for trend in dashboard["trends"]]

        return dashboard

    def get_health_summary(self) -> dict[str, Any]:
        """Get concise health summary"""
        metrics = self.dashboard_data["current_metrics"]

        if not metrics:
            return {"status": "unknown", "message": "No metrics available"}

        health = self.dashboard_data["system_health"]
        recent_alerts = len(
            [
                a
                for a in self.dashboard_data["alerts"]
                if time.time() - a.timestamp < 300
            ]
        )  # Last 5 minutes

        return {
            "status": health,
            "ops_per_sec": metrics.ops_per_sec,
            "latency_p95_ms": metrics.latency_p95_ms,
            "cpu_utilization": metrics.cpu_utilization,
            "target_100_ops_met": metrics.ops_per_sec >= 100,
            "recent_alerts": recent_alerts,
            "uptime_hours": self.dashboard_data["uptime_seconds"] / 3600,
            "total_operations": self.dashboard_data["total_operations"],
        }

    def generate_performance_report(self) -> str:
        """Generate human-readable performance report"""
        summary = self.get_health_summary()
        dashboard = self.get_dashboard_data()

        report = []
        report.append("🖥️  Bolt Throughput Monitor - Performance Report")
        report.append("=" * 60)

        # System Health
        health_emoji = {
            "healthy": "🟢",
            "degraded": "🟡",
            "warning": "🟠",
            "unhealthy": "🔴",
            "critical": "🚨",
            "unknown": "⚪",
        }

        report.append(
            f"System Health: {health_emoji.get(summary['status'], '⚪')} {summary['status'].upper()}"
        )

        # Key Metrics
        report.append("\n📊 Key Metrics:")
        report.append(
            f"  Throughput: {summary['ops_per_sec']:.1f} ops/sec {'✅' if summary['target_100_ops_met'] else '❌'}"
        )
        report.append(f"  Latency (P95): {summary['latency_p95_ms']:.1f}ms")
        report.append(f"  CPU Usage: {summary['cpu_utilization']:.1f}%")

        # Uptime and Operations
        report.append("\n⏱️  System Status:")
        report.append(f"  Uptime: {summary['uptime_hours']:.1f} hours")
        report.append(f"  Total Operations: {summary['total_operations']:,}")
        report.append(f"  Recent Alerts: {summary['recent_alerts']}")

        # Recent Trends
        if dashboard["trends"]:
            report.append("\n📈 Performance Trends:")
            for trend in dashboard["trends"]:
                direction_emoji = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}
                report.append(
                    f"  {trend['metric_name']}: {direction_emoji.get(trend['trend_direction'], '➡️')} "
                    f"{trend['trend_direction']} ({trend['trend_magnitude']:.1f}%)"
                )

        # Recent Alerts
        recent_alerts = [
            a for a in dashboard["alerts"] if time.time() - a["timestamp"] < 1800
        ]  # Last 30 minutes

        if recent_alerts:
            report.append("\n🚨 Recent Alerts (Last 30 min):")
            for alert in recent_alerts[-5:]:  # Show last 5
                severity_emoji = {"critical": "🚨", "warning": "⚠️"}
                report.append(
                    f"  {severity_emoji.get(alert['severity'], '⚠️')} {alert['message']}"
                )

        return "\n".join(report)


# Global dashboard instance
_throughput_dashboard = None


def get_throughput_dashboard() -> ThroughputDashboard:
    """Get global throughput dashboard instance"""
    global _throughput_dashboard
    if _throughput_dashboard is None:
        _throughput_dashboard = ThroughputDashboard()
    return _throughput_dashboard


# Context manager for operation tracking
class OperationTracker:
    """Context manager to track operation performance"""

    def __init__(self, operation_name: str, metadata: dict[str, Any] = None):
        self.operation_name = operation_name
        self.metadata = metadata or {}
        self.start_time = None
        self.dashboard = get_throughput_dashboard()

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            metadata = self.metadata.copy()
            metadata.update(
                {"operation_name": self.operation_name, "success": exc_type is None}
            )
            self.dashboard.record_operation(duration_ms, metadata)

    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            metadata = self.metadata.copy()
            metadata.update(
                {"operation_name": self.operation_name, "success": exc_type is None}
            )
            self.dashboard.record_operation(duration_ms, metadata)


if __name__ == "__main__":

    async def demo_monitoring():
        """Demo the throughput monitoring system"""
        print("🖥️  Bolt Throughput Monitor Demo")
        print("=" * 50)

        dashboard = get_throughput_dashboard()
        dashboard.start_monitoring(interval=2.0)

        print("Simulating operations...")

        # Simulate various operations
        for i in range(100):
            # Mix of fast and slow operations
            if i % 10 == 0:
                # Slow operation
                async with OperationTracker("slow_operation", {"iteration": i}):
                    await asyncio.sleep(0.1)
            else:
                # Fast operation
                async with OperationTracker("fast_operation", {"iteration": i}):
                    await asyncio.sleep(0.01)

            # Show dashboard every 20 operations
            if i % 20 == 0 and i > 0:
                print(f"\n📊 After {i} operations:")
                summary = dashboard.get_health_summary()
                print(f"  Throughput: {summary['ops_per_sec']:.1f} ops/sec")
                print(f"  Health: {summary['status']}")
                print(f"  Alerts: {summary['recent_alerts']}")

        # Final report
        print("\n" + dashboard.generate_performance_report())

        # Save dashboard data
        with open("throughput_monitor_demo.json", "w") as f:
            json.dump(dashboard.get_dashboard_data(), f, indent=2, default=str)

        dashboard.stop_monitoring()
        print("\n📄 Demo data saved to: throughput_monitor_demo.json")

    asyncio.run(demo_monitoring())
