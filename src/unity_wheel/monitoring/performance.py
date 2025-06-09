"""Performance monitoring and tracking for critical operations."""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    operation: str
    duration_ms: float
    timestamp: datetime
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation": self.operation,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "metadata": self.metadata,
        }


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    operation: str
    count: int
    success_rate: float
    avg_duration_ms: float
    p50_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "operation": self.operation,
            "count": self.count,
            "success_rate": self.success_rate,
            "avg_duration_ms": round(self.avg_duration_ms, 2),
            "p50_duration_ms": round(self.p50_duration_ms, 2),
            "p95_duration_ms": round(self.p95_duration_ms, 2),
            "p99_duration_ms": round(self.p99_duration_ms, 2),
            "min_duration_ms": round(self.min_duration_ms, 2),
            "max_duration_ms": round(self.max_duration_ms, 2),
        }


class PerformanceMonitor:
    """
    Centralized performance monitoring system.
    
    Tracks operation latencies, success rates, and performance trends.
    """
    
    # Performance thresholds (ms)
    THRESHOLDS = {
        "black_scholes_price": 0.2,
        "calculate_all_greeks": 0.3,
        "find_optimal_put_strike": 100.0,
        "advise_position": 200.0,
        "calculate_var": 10.0,
        "calculate_cvar": 10.0,
        "risk_analysis": 50.0,
        "decision_engine": 200.0,
        "api_call": 1000.0,
        "default": 100.0,
    }
    
    def __init__(self, max_history: int = 10000):
        """Initialize performance monitor."""
        self.max_history = max_history
        self.metrics: Dict[str, deque[PerformanceMetric]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self.lock = Lock()
        self.start_time = datetime.now(timezone.utc)
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[str, PerformanceMetric], None]] = []
        
        # SLA violations
        self.sla_violations: List[Dict[str, Any]] = []
        
    def record(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            operation=operation,
            duration_ms=duration_ms,
            timestamp=datetime.now(timezone.utc),
            success=success,
            metadata=metadata or {},
        )
        
        with self.lock:
            self.metrics[operation].append(metric)
        
        # Check for SLA violation
        threshold = self.THRESHOLDS.get(operation, self.THRESHOLDS["default"])
        if duration_ms > threshold:
            self._handle_sla_violation(metric, threshold)
    
    def _handle_sla_violation(self, metric: PerformanceMetric, threshold: float) -> None:
        """Handle SLA violation."""
        violation = {
            "operation": metric.operation,
            "duration_ms": metric.duration_ms,
            "threshold_ms": threshold,
            "timestamp": metric.timestamp,
            "severity": self._calculate_severity(metric.duration_ms, threshold),
        }
        
        self.sla_violations.append(violation)
        
        # Log warning
        logger.warning(
            f"SLA violation: {metric.operation} took {metric.duration_ms:.1f}ms "
            f"(threshold: {threshold:.1f}ms)",
            extra=violation,
        )
        
        # Trigger alerts
        for callback in self.alert_callbacks:
            try:
                callback(f"SLA violation: {metric.operation}", metric)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _calculate_severity(self, actual: float, threshold: float) -> str:
        """Calculate violation severity."""
        ratio = actual / threshold
        if ratio < 1.5:
            return "low"
        elif ratio < 2.0:
            return "medium"
        elif ratio < 3.0:
            return "high"
        else:
            return "critical"
    
    def get_stats(self, operation: str, window_minutes: int = 60) -> Optional[PerformanceStats]:
        """Get performance statistics for an operation."""
        with self.lock:
            if operation not in self.metrics:
                return None
            
            # Filter by time window
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
            recent_metrics = [
                m for m in self.metrics[operation]
                if m.timestamp > cutoff
            ]
            
            if not recent_metrics:
                return None
            
            # Calculate statistics
            durations = [m.duration_ms for m in recent_metrics]
            successes = sum(1 for m in recent_metrics if m.success)
            
            return PerformanceStats(
                operation=operation,
                count=len(recent_metrics),
                success_rate=successes / len(recent_metrics),
                avg_duration_ms=np.mean(durations),
                p50_duration_ms=np.percentile(durations, 50),
                p95_duration_ms=np.percentile(durations, 95),
                p99_duration_ms=np.percentile(durations, 99),
                min_duration_ms=min(durations),
                max_duration_ms=max(durations),
            )
    
    def get_all_stats(self, window_minutes: int = 60) -> Dict[str, PerformanceStats]:
        """Get statistics for all tracked operations."""
        stats = {}
        for operation in list(self.metrics.keys()):
            op_stats = self.get_stats(operation, window_minutes)
            if op_stats:
                stats[operation] = op_stats
        return stats
    
    def get_slow_operations(
        self,
        window_minutes: int = 60,
        min_count: int = 10,
    ) -> List[Tuple[str, PerformanceStats]]:
        """Get operations that are consistently slow."""
        stats = self.get_all_stats(window_minutes)
        slow_ops = []
        
        for operation, op_stats in stats.items():
            if op_stats.count < min_count:
                continue
            
            threshold = self.THRESHOLDS.get(operation, self.THRESHOLDS["default"])
            if op_stats.p95_duration_ms > threshold:
                slow_ops.append((operation, op_stats))
        
        # Sort by severity (ratio of p95 to threshold)
        slow_ops.sort(
            key=lambda x: x[1].p95_duration_ms / self.THRESHOLDS.get(x[0], self.THRESHOLDS["default"]),
            reverse=True,
        )
        
        return slow_ops
    
    def get_performance_trends(
        self,
        operation: str,
        bucket_minutes: int = 5,
        window_hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """Get performance trends over time."""
        with self.lock:
            if operation not in self.metrics:
                return []
            
            # Group metrics by time bucket
            cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
            buckets = defaultdict(list)
            
            for metric in self.metrics[operation]:
                if metric.timestamp < cutoff:
                    continue
                
                # Round to bucket
                bucket_time = metric.timestamp.replace(
                    minute=(metric.timestamp.minute // bucket_minutes) * bucket_minutes,
                    second=0,
                    microsecond=0,
                )
                buckets[bucket_time].append(metric)
            
            # Calculate stats per bucket
            trends = []
            for bucket_time, metrics in sorted(buckets.items()):
                durations = [m.duration_ms for m in metrics]
                successes = sum(1 for m in metrics if m.success)
                
                trends.append({
                    "timestamp": bucket_time.isoformat(),
                    "count": len(metrics),
                    "success_rate": successes / len(metrics) if metrics else 0,
                    "avg_duration_ms": np.mean(durations) if durations else 0,
                    "p95_duration_ms": np.percentile(durations, 95) if durations else 0,
                })
            
            return trends
    
    def generate_report(self, format: str = "json") -> str:
        """Generate performance report."""
        report_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_hours": (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600,
            "operations_tracked": len(self.metrics),
            "total_measurements": sum(len(metrics) for metrics in self.metrics.values()),
            "statistics": {
                op: stats.to_dict()
                for op, stats in self.get_all_stats(60).items()
            },
            "slow_operations": [
                {
                    "operation": op,
                    "stats": stats.to_dict(),
                    "threshold_ms": self.THRESHOLDS.get(op, self.THRESHOLDS["default"]),
                }
                for op, stats in self.get_slow_operations()
            ],
            "recent_sla_violations": self.sla_violations[-10:],  # Last 10 violations
        }
        
        if format == "json":
            return json.dumps(report_data, indent=2)
        else:
            # Text format
            lines = [
                "=" * 60,
                "PERFORMANCE MONITORING REPORT",
                f"Generated: {report_data['timestamp']}",
                f"Uptime: {report_data['uptime_hours']:.1f} hours",
                "=" * 60,
                "",
                f"Operations Tracked: {report_data['operations_tracked']}",
                f"Total Measurements: {report_data['total_measurements']}",
                "",
            ]
            
            # Statistics table
            if report_data["statistics"]:
                lines.extend([
                    "OPERATION STATISTICS (last 60 minutes):",
                    "-" * 60,
                    f"{'Operation':<30} {'Count':>8} {'Avg(ms)':>10} {'P95(ms)':>10}",
                    "-" * 60,
                ])
                
                for op, stats in report_data["statistics"].items():
                    lines.append(
                        f"{op:<30} {stats['count']:>8} "
                        f"{stats['avg_duration_ms']:>10.1f} "
                        f"{stats['p95_duration_ms']:>10.1f}"
                    )
                
                lines.append("")
            
            # Slow operations
            if report_data["slow_operations"]:
                lines.extend([
                    "SLOW OPERATIONS:",
                    "-" * 60,
                ])
                
                for op_data in report_data["slow_operations"]:
                    op = op_data["operation"]
                    stats = op_data["stats"]
                    threshold = op_data["threshold_ms"]
                    lines.append(
                        f"  {op}: P95={stats['p95_duration_ms']:.1f}ms "
                        f"(threshold={threshold:.1f}ms)"
                    )
                
                lines.append("")
            
            # Recent violations
            if report_data["recent_sla_violations"]:
                lines.extend([
                    "RECENT SLA VIOLATIONS:",
                    "-" * 60,
                ])
                
                for violation in report_data["recent_sla_violations"][-5:]:
                    lines.append(
                        f"  {violation['operation']}: {violation['duration_ms']:.1f}ms "
                        f"at {violation['timestamp']}"
                    )
            
            lines.append("=" * 60)
            
            return "\n".join(lines)
    
    def export_metrics(self, file_path: Path) -> None:
        """Export all metrics to file for analysis."""
        with self.lock:
            all_metrics = []
            for operation, metrics in self.metrics.items():
                for metric in metrics:
                    all_metrics.append({
                        **metric.to_dict(),
                        "operation": operation,
                    })
            
            with open(file_path, "w") as f:
                json.dump(all_metrics, f, indent=2)
    
    def register_alert_callback(self, callback: Callable[[str, PerformanceMetric], None]) -> None:
        """Register callback for performance alerts."""
        self.alert_callbacks.append(callback)
    
    def clear_old_metrics(self, days: int = 7) -> int:
        """Clear metrics older than specified days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cleared = 0
        
        with self.lock:
            for operation in list(self.metrics.keys()):
                old_size = len(self.metrics[operation])
                self.metrics[operation] = deque(
                    (m for m in self.metrics[operation] if m.timestamp > cutoff),
                    maxlen=self.max_history,
                )
                cleared += old_size - len(self.metrics[operation])
        
        return cleared


# Global singleton instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def performance_monitored(operation: Optional[str] = None):
    """
    Decorator to monitor function performance.
    
    Parameters
    ----------
    operation : str, optional
        Operation name. If not provided, uses function name.
    
    Examples
    --------
    @performance_monitored("critical_calculation")
    def calculate_something():
        ...
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__
        
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                monitor.record(op_name, duration_ms, success)
        
        return wrapper
    
    return decorator