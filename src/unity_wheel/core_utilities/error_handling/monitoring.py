"""
Error Monitoring and Health Checking System

Provides comprehensive error tracking, pattern detection, health checking,
and alerting capabilities for system stability monitoring.
"""

import functools
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any
from uuid import uuid4

from .exceptions import ErrorSeverity, UnityWheelError
from .logging_enhanced import get_enhanced_logger


class SystemStatus(Enum):
    """Overall system health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DOWN = "down"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ErrorMetrics:
    """Error metrics for monitoring."""

    error_count: int = 0
    error_rate_per_minute: float = 0.0
    unique_error_types: int = 0
    most_frequent_error: str | None = None
    most_frequent_component: str | None = None
    last_error_time: float | None = None
    average_error_interval: float = 0.0
    critical_errors: int = 0
    retryable_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HealthMetrics:
    """System health metrics."""

    status: SystemStatus
    uptime_seconds: float
    error_rate: float
    success_rate: float
    average_response_time_ms: float
    active_connections: int
    memory_usage_percent: float
    cpu_usage_percent: float
    disk_usage_percent: float
    last_check_time: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["status"] = self.status.value
        return result


@dataclass
class Alert:
    """System alert."""

    id: str
    severity: AlertSeverity
    title: str
    message: str
    component: str
    timestamp: float
    resolved: bool = False
    resolved_at: float | None = None
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["severity"] = self.severity.value
        return result


class ErrorPattern:
    """Detected error pattern."""

    def __init__(self, pattern_id: str, description: str):
        self.pattern_id = pattern_id
        self.description = description
        self.occurrences = 0
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.components_affected: set[str] = set()
        self.error_types: set[str] = set()

    def add_occurrence(self, error: UnityWheelError) -> None:
        """Add an error occurrence to this pattern."""
        self.occurrences += 1
        self.last_seen = time.time()
        if error.component:
            self.components_affected.add(error.component)
        self.error_types.add(type(error).__name__)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "occurrences": self.occurrences,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration_minutes": (self.last_seen - self.first_seen) / 60,
            "components_affected": list(self.components_affected),
            "error_types": list(self.error_types),
        }


class ErrorMonitor:
    """Monitors errors and detects patterns."""

    def __init__(self, max_error_history: int = 1000):
        self.max_error_history = max_error_history
        self.error_history: deque = deque(maxlen=max_error_history)
        self.error_counts = defaultdict(int)
        self.error_by_component = defaultdict(int)
        self.error_by_type = defaultdict(int)
        self.error_patterns: dict[str, ErrorPattern] = {}
        self.start_time = time.time()
        self.logger = get_enhanced_logger("error_monitor")

        # Alerting
        self.alerts: list[Alert] = []
        self.alert_callbacks: list[Callable[[Alert], None]] = []

        # Thresholds
        self.error_rate_threshold = 10  # errors per minute
        self.critical_error_threshold = 5  # critical errors per hour
        self.pattern_threshold = 5  # occurrences to detect pattern

        # Lock for thread safety
        self._lock = threading.Lock()

    def record_error(self, error: UnityWheelError) -> None:
        """Record an error for monitoring."""
        with self._lock:
            timestamp = time.time()

            # Add to history
            error_record = {
                "timestamp": timestamp,
                "error_code": error.error_code,
                "error_type": type(error).__name__,
                "category": error.category.value,
                "severity": error.severity.value,
                "component": error.component,
                "operation": error.operation,
                "message": error.message,
                "is_retryable": error.is_retryable(),
            }
            self.error_history.append(error_record)

            # Update counters
            self.error_counts[error.error_code] += 1
            if error.component:
                self.error_by_component[error.component] += 1
            self.error_by_type[type(error).__name__] += 1

            # Check for patterns
            self._detect_patterns(error)

            # Check for alerts
            self._check_alert_conditions(error)

    def _detect_patterns(self, error: UnityWheelError) -> None:
        """Detect error patterns."""
        # Pattern 1: Repeated same error in short time
        pattern_key = f"repeated_{error.error_code}"
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = ErrorPattern(
                pattern_key, f"Repeated error: {error.error_code}"
            )

        pattern = self.error_patterns[pattern_key]
        pattern.add_occurrence(error)

        # Alert if pattern threshold reached
        if pattern.occurrences >= self.pattern_threshold:
            self._create_alert(
                AlertSeverity.WARNING,
                f"Error Pattern Detected: {error.error_code}",
                f"Error {error.error_code} has occurred {pattern.occurrences} times",
                error.component or "unknown",
            )

        # Pattern 2: Component failing frequently
        if error.component:
            component_pattern_key = f"component_{error.component}"
            if component_pattern_key not in self.error_patterns:
                self.error_patterns[component_pattern_key] = ErrorPattern(
                    component_pattern_key, f"Component failures: {error.component}"
                )

            comp_pattern = self.error_patterns[component_pattern_key]
            comp_pattern.add_occurrence(error)

            if comp_pattern.occurrences >= self.pattern_threshold * 2:
                self._create_alert(
                    AlertSeverity.CRITICAL,
                    f"Component Instability: {error.component}",
                    f"Component {error.component} has {comp_pattern.occurrences} failures",
                    error.component,
                )

    def _check_alert_conditions(self, error: UnityWheelError) -> None:
        """Check if error conditions warrant alerts."""
        # Critical error alert
        if error.severity == ErrorSeverity.CRITICAL:
            self._create_alert(
                AlertSeverity.CRITICAL,
                f"Critical Error: {error.error_code}",
                f"Critical error in {error.component}: {error.message}",
                error.component or "unknown",
            )

        # High error rate alert
        current_time = time.time()
        recent_errors = [
            e
            for e in self.error_history
            if current_time - e["timestamp"] < 60  # Last minute
        ]

        if len(recent_errors) > self.error_rate_threshold:
            self._create_alert(
                AlertSeverity.WARNING,
                "High Error Rate",
                f"{len(recent_errors)} errors in the last minute",
                "system",
            )

    def _create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        component: str,
        context: dict[str, Any] | None = None,
    ) -> Alert:
        """Create and dispatch an alert."""
        alert = Alert(
            id=str(uuid4())[:8],
            severity=severity,
            title=title,
            message=message,
            component=component,
            timestamp=time.time(),
            context=context or {},
        )

        self.alerts.append(alert)
        self.logger.warning(f"Alert created: {title} - {message}")

        # Dispatch to callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")

        return alert

    def get_metrics(self) -> ErrorMetrics:
        """Get current error metrics."""
        with self._lock:
            if not self.error_history:
                return ErrorMetrics()

            current_time = time.time()

            # Calculate error rate (last minute)
            recent_errors = [
                e for e in self.error_history if current_time - e["timestamp"] < 60
            ]
            error_rate = len(recent_errors)

            # Find most frequent error
            most_frequent_error = (
                max(self.error_counts.items(), key=lambda x: x[1])[0]
                if self.error_counts
                else None
            )
            most_frequent_component = (
                max(self.error_by_component.items(), key=lambda x: x[1])[0]
                if self.error_by_component
                else None
            )

            # Calculate average interval between errors
            timestamps = [e["timestamp"] for e in self.error_history]
            if len(timestamps) > 1:
                intervals = [
                    timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))
                ]
                avg_interval = sum(intervals) / len(intervals)
            else:
                avg_interval = 0.0

            # Count critical and retryable errors
            critical_errors = sum(
                1 for e in self.error_history if e["severity"] == "critical"
            )
            retryable_errors = sum(1 for e in self.error_history if e["is_retryable"])

            return ErrorMetrics(
                error_count=len(self.error_history),
                error_rate_per_minute=error_rate,
                unique_error_types=len(self.error_by_type),
                most_frequent_error=most_frequent_error,
                most_frequent_component=most_frequent_component,
                last_error_time=self.error_history[-1]["timestamp"]
                if self.error_history
                else None,
                average_error_interval=avg_interval,
                critical_errors=critical_errors,
                retryable_errors=retryable_errors,
            )

    def get_patterns(self) -> list[dict[str, Any]]:
        """Get detected error patterns."""
        with self._lock:
            return [pattern.to_dict() for pattern in self.error_patterns.values()]

    def get_alerts(self, unresolved_only: bool = True) -> list[dict[str, Any]]:
        """Get system alerts."""
        alerts = [a for a in self.alerts if not unresolved_only or not a.resolved]
        return [alert.to_dict() for alert in alerts]

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_at = time.time()
                return True
        return False

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)

    def clear_old_data(self, hours: int = 24) -> None:
        """Clear old error data."""
        cutoff_time = time.time() - (hours * 3600)

        with self._lock:
            # Clean error history (deque handles this automatically with maxlen)
            # Clean patterns
            old_patterns = [
                pid
                for pid, pattern in self.error_patterns.items()
                if pattern.last_seen < cutoff_time
            ]
            for pid in old_patterns:
                del self.error_patterns[pid]

            # Clean old alerts
            self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]


class HealthChecker:
    """Monitors overall system health."""

    def __init__(self):
        self.start_time = time.time()
        self.health_history: deque = deque(maxlen=100)
        self.operation_times: deque = deque(maxlen=1000)
        self.success_count = 0
        self.failure_count = 0
        self.logger = get_enhanced_logger("health_checker")

        # Health check callbacks
        self.health_checks: list[Callable[[], dict[str, Any]]] = []

        # Thresholds
        self.degraded_error_rate = 0.05  # 5% error rate
        self.critical_error_rate = 0.20  # 20% error rate
        self.slow_response_threshold = 5000  # 5 seconds in ms

    def record_operation(self, success: bool, duration_ms: float) -> None:
        """Record an operation result."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        self.operation_times.append(duration_ms)

    def add_health_check(self, check_func: Callable[[], dict[str, Any]]) -> None:
        """Add a custom health check function."""
        self.health_checks.append(check_func)

    def get_health_status(self) -> HealthMetrics:
        """Get current health status."""
        current_time = time.time()
        uptime = current_time - self.start_time

        # Calculate rates
        total_operations = self.success_count + self.failure_count
        if total_operations > 0:
            error_rate = self.failure_count / total_operations
            success_rate = self.success_count / total_operations
        else:
            error_rate = 0.0
            success_rate = 1.0

        # Calculate average response time
        avg_response_time = (
            sum(self.operation_times) / len(self.operation_times)
            if self.operation_times
            else 0.0
        )

        # Determine status
        if error_rate >= self.critical_error_rate:
            status = SystemStatus.CRITICAL
        elif error_rate >= self.degraded_error_rate or avg_response_time > self.slow_response_threshold:
            status = SystemStatus.DEGRADED
        else:
            status = SystemStatus.HEALTHY

        # Run custom health checks
        custom_metrics = {}
        for check_func in self.health_checks:
            try:
                result = check_func()
                custom_metrics.update(result)
            except Exception as e:
                self.logger.warning(f"Health check failed: {e}")
                status = SystemStatus.DEGRADED

        metrics = HealthMetrics(
            status=status,
            uptime_seconds=uptime,
            error_rate=error_rate,
            success_rate=success_rate,
            average_response_time_ms=avg_response_time,
            active_connections=custom_metrics.get("active_connections", 0),
            memory_usage_percent=custom_metrics.get("memory_usage_percent", 0.0),
            cpu_usage_percent=custom_metrics.get("cpu_usage_percent", 0.0),
            disk_usage_percent=custom_metrics.get("disk_usage_percent", 0.0),
            last_check_time=current_time,
        )

        self.health_history.append(metrics)
        return metrics

    def get_health_trend(self, minutes: int = 30) -> list[dict[str, Any]]:
        """Get health trend over time."""
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [
            m for m in self.health_history if m.last_check_time > cutoff_time
        ]
        return [m.to_dict() for m in recent_metrics]


# Global instances
_error_monitor = ErrorMonitor()
_health_checker = HealthChecker()


def track_error_patterns(error: UnityWheelError) -> None:
    """Track error in global monitor."""
    _error_monitor.record_error(error)


def alert_on_error(error: UnityWheelError) -> None:
    """Check error for alerting conditions."""
    _error_monitor._check_alert_conditions(error)


def get_error_monitor() -> ErrorMonitor:
    """Get global error monitor."""
    return _error_monitor


def get_health_checker() -> HealthChecker:
    """Get global health checker."""
    return _health_checker


def health_check_decorator(func: Callable) -> Callable:
    """Decorator to automatically track function health."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000
            _health_checker.record_operation(True, duration_ms)
            return result
        except Exception:
            duration_ms = (time.perf_counter() - start_time) * 1000
            _health_checker.record_operation(False, duration_ms)
            raise

    return wrapper


def async_health_check_decorator(func: Callable) -> Callable:
    """Async decorator to automatically track function health."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000
            _health_checker.record_operation(True, duration_ms)
            return result
        except Exception:
            duration_ms = (time.perf_counter() - start_time) * 1000
            _health_checker.record_operation(False, duration_ms)
            raise

    return wrapper


# Example alert handlers
def console_alert_handler(alert: Alert) -> None:
    """Simple console alert handler."""
    print(f"🚨 ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}")


def log_alert_handler(alert: Alert) -> None:
    """Log-based alert handler."""
    logger = get_enhanced_logger("alerts")
    logger.warning(f"Alert: {alert.title}", extra=alert.to_dict())


# Set up default alert handlers
_error_monitor.add_alert_callback(console_alert_handler)
_error_monitor.add_alert_callback(log_alert_handler)
