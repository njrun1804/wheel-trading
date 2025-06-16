#!/usr/bin/env python3
"""
Metrics Collection and Aggregation System
Collects metrics from all subsystems and provides unified views
"""

import asyncio
import json
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point"""

    name: str
    value: float
    timestamp: float
    tags: dict[str, str]
    metadata: dict[str, Any]


@dataclass
class AggregatedMetric:
    """Aggregated metric statistics"""

    name: str
    count: int
    sum: float
    min: float
    max: float
    avg: float
    p50: float
    p95: float
    p99: float
    std_dev: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "count": self.count,
            "sum": round(self.sum, 2),
            "min": round(self.min, 2),
            "max": round(self.max, 2),
            "avg": round(self.avg, 2),
            "p50": round(self.p50, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
            "std_dev": round(self.std_dev, 2),
        }


class MetricsCollector:
    """Central metrics collection and aggregation system"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

        # Configuration
        self.retention_minutes = self.config.get("retention_minutes", 60)
        self.aggregation_intervals = self.config.get(
            "aggregation_intervals", [1, 5, 15, 60]
        )
        self.max_metrics_per_name = self.config.get("max_metrics_per_name", 10000)

        # Storage
        self.metrics = defaultdict(lambda: deque(maxlen=self.max_metrics_per_name))
        self.aggregations = defaultdict(dict)  # name -> interval -> AggregatedMetric

        # Metric providers
        self.providers = {}
        self.provider_intervals = {}

        # State
        self._collecting = False
        self._aggregation_thread = None
        self._collection_threads = {}
        self._lock = threading.Lock()

        # Callbacks for metric events
        self.metric_callbacks = defaultdict(list)

        logger.info("📊 Metrics Collector initialized")

    def register_provider(
        self, name: str, provider: Callable[[], dict[str, float]], interval: float = 1.0
    ):
        """Register a metric provider function"""
        with self._lock:
            self.providers[name] = provider
            self.provider_intervals[name] = interval

        logger.info(f"📈 Registered metric provider: {name} (interval: {interval}s)")

    def start_collection(self):
        """Start metric collection"""
        if self._collecting:
            logger.warning("Collection already active")
            return

        self._collecting = True

        # Start aggregation thread
        self._aggregation_thread = threading.Thread(
            target=self._aggregation_loop, daemon=True
        )
        self._aggregation_thread.start()

        # Start collection threads for each provider
        for name in self.providers:
            thread = threading.Thread(
                target=self._collection_loop, args=(name,), daemon=True
            )
            thread.start()
            self._collection_threads[name] = thread

        logger.info("📊 Metrics collection started")

    def stop_collection(self):
        """Stop metric collection"""
        self._collecting = False

        # Wait for threads
        if self._aggregation_thread:
            self._aggregation_thread.join(timeout=5.0)

        for thread in self._collection_threads.values():
            thread.join(timeout=5.0)

        self._collection_threads.clear()

        logger.info("📊 Metrics collection stopped")

    def record_metric(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Record a single metric"""
        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            metadata=metadata or {},
        )

        with self._lock:
            self.metrics[name].append(metric)

        # Trigger callbacks
        for callback in self.metric_callbacks[name]:
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"Metric callback error: {e}")

    def batch_record(
        self, metrics: dict[str, float], tags: dict[str, str] | None = None
    ):
        """Record multiple metrics at once"""
        timestamp = time.time()

        with self._lock:
            for name, value in metrics.items():
                metric = MetricPoint(
                    name=name,
                    value=value,
                    timestamp=timestamp,
                    tags=tags or {},
                    metadata={},
                )
                self.metrics[name].append(metric)

    def _collection_loop(self, provider_name: str):
        """Collection loop for a provider"""
        provider = self.providers[provider_name]
        interval = self.provider_intervals[provider_name]

        while self._collecting:
            try:
                # Collect metrics from provider
                metrics = provider()

                if metrics:
                    # Add provider tag
                    tags = {"provider": provider_name}
                    self.batch_record(metrics, tags)

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Provider {provider_name} error: {e}")
                time.sleep(interval * 5)  # Back off on error

    def _aggregation_loop(self):
        """Aggregation loop for computing statistics"""
        next_runs = {
            interval: time.time() + interval for interval in self.aggregation_intervals
        }

        while self._collecting:
            try:
                current_time = time.time()

                # Check which intervals need aggregation
                for interval in self.aggregation_intervals:
                    if current_time >= next_runs[interval]:
                        self._aggregate_metrics(interval)
                        next_runs[interval] = current_time + interval

                # Clean old metrics
                self._clean_old_metrics()

                # Sleep until next aggregation
                next_run = min(next_runs.values())
                sleep_time = max(0.1, next_run - time.time())
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Aggregation error: {e}")
                time.sleep(5.0)

    def _aggregate_metrics(self, interval_minutes: int):
        """Aggregate metrics for a specific interval"""
        cutoff_time = time.time() - (interval_minutes * 60)

        with self._lock:
            for name, metric_deque in self.metrics.items():
                # Get metrics in interval
                metrics_in_interval = [
                    m for m in metric_deque if m.timestamp > cutoff_time
                ]

                if metrics_in_interval:
                    values = [m.value for m in metrics_in_interval]
                    values_sorted = sorted(values)

                    # Calculate statistics
                    count = len(values)
                    sum_val = sum(values)
                    avg = sum_val / count

                    # Calculate standard deviation
                    variance = sum((x - avg) ** 2 for x in values) / count
                    std_dev = variance**0.5

                    # Create aggregation
                    agg = AggregatedMetric(
                        name=name,
                        count=count,
                        sum=sum_val,
                        min=min(values),
                        max=max(values),
                        avg=avg,
                        p50=values_sorted[int(count * 0.5)],
                        p95=values_sorted[int(count * 0.95)],
                        p99=values_sorted[int(count * 0.99)],
                        std_dev=std_dev,
                    )

                    self.aggregations[name][interval_minutes] = agg

    def _clean_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff_time = time.time() - (self.retention_minutes * 60)

        with self._lock:
            for name, metric_deque in list(self.metrics.items()):
                # Remove old metrics
                while metric_deque and metric_deque[0].timestamp < cutoff_time:
                    metric_deque.popleft()

                # Remove empty metric series
                if not metric_deque:
                    del self.metrics[name]

    def get_metric_value(
        self, name: str, lookback_seconds: int = 60
    ) -> float | None:
        """Get latest metric value within lookback period"""
        cutoff_time = time.time() - lookback_seconds

        with self._lock:
            if name in self.metrics:
                recent_metrics = [
                    m for m in self.metrics[name] if m.timestamp > cutoff_time
                ]
                if recent_metrics:
                    return recent_metrics[-1].value

        return None

    def get_metric_series(
        self, name: str, lookback_minutes: int = 60
    ) -> list[Tuple[float, float]]:
        """Get time series data for a metric"""
        cutoff_time = time.time() - (lookback_minutes * 60)

        with self._lock:
            if name in self.metrics:
                series = [
                    (m.timestamp, m.value)
                    for m in self.metrics[name]
                    if m.timestamp > cutoff_time
                ]
                return series

        return []

    def get_aggregated_metrics(
        self, interval_minutes: int = 5
    ) -> dict[str, AggregatedMetric]:
        """Get aggregated metrics for an interval"""
        with self._lock:
            return {
                name: agg
                for name, intervals in self.aggregations.items()
                if interval_minutes in intervals
                for agg in [intervals[interval_minutes]]
            }

    def get_all_metrics_summary(self) -> dict[str, Any]:
        """Get summary of all metrics"""
        with self._lock:
            summary = {
                "total_metrics": len(self.metrics),
                "total_data_points": sum(len(deque) for deque in self.metrics.values()),
                "metrics": {},
            }

            # Add per-metric summaries
            for name, metric_deque in self.metrics.items():
                if metric_deque:
                    latest = metric_deque[-1]
                    values = [m.value for m in metric_deque]

                    summary["metrics"][name] = {
                        "latest_value": latest.value,
                        "latest_timestamp": latest.timestamp,
                        "data_points": len(metric_deque),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                    }

            return summary

    def register_callback(
        self, metric_name: str, callback: Callable[[MetricPoint], None]
    ):
        """Register callback for metric updates"""
        self.metric_callbacks[metric_name].append(callback)

    def export_metrics(self, output_path: Path, format: str = "json"):
        """Export metrics to file"""
        with self._lock:
            if format == "json":
                data = {"timestamp": datetime.now().isoformat(), "metrics": {}}

                for name, metric_deque in self.metrics.items():
                    data["metrics"][name] = [
                        {"value": m.value, "timestamp": m.timestamp, "tags": m.tags}
                        for m in metric_deque
                    ]

                with open(output_path, "w") as f:
                    json.dump(data, f, indent=2)

            elif format == "csv":
                # Simple CSV export
                lines = ["metric_name,timestamp,value\n"]

                for name, metric_deque in self.metrics.items():
                    for m in metric_deque:
                        lines.append(f"{name},{m.timestamp},{m.value}\n")

                with open(output_path, "w") as f:
                    f.writelines(lines)

        logger.info(f"📤 Exported metrics to {output_path}")


# Built-in metric providers
def cpu_metrics_provider() -> dict[str, float]:
    """Provide CPU metrics"""
    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()

        return {
            "cpu.percent": cpu_percent,
            "cpu.frequency_mhz": cpu_freq.current if cpu_freq else 0.0,
            "cpu.cores_physical": psutil.cpu_count(logical=False),
            "cpu.cores_logical": psutil.cpu_count(logical=True),
        }
    except ImportError:
        return {}


def memory_metrics_provider() -> dict[str, float]:
    """Provide memory metrics"""
    try:
        import psutil

        memory = psutil.virtual_memory()

        return {
            "memory.total_gb": memory.total / (1024**3),
            "memory.used_gb": memory.used / (1024**3),
            "memory.available_gb": memory.available / (1024**3),
            "memory.percent": memory.percent,
        }
    except ImportError:
        return {}


def gpu_metrics_provider() -> dict[str, float]:
    """Provide GPU metrics (placeholder)"""
    # This would need actual GPU monitoring implementation
    return {"gpu.utilization": 0.0, "gpu.memory_used_mb": 0.0, "gpu.temperature_c": 0.0}


# Global collector instance
_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()

        # Register built-in providers
        _collector.register_provider("cpu", cpu_metrics_provider, interval=1.0)
        _collector.register_provider("memory", memory_metrics_provider, interval=5.0)
        _collector.register_provider("gpu", gpu_metrics_provider, interval=10.0)

    return _collector


# Decorator for timing functions
def timed_metric(metric_name: str | None = None):
    """Decorator to time function execution"""

    def decorator(func):
        name = metric_name or f"function.{func.__name__}.duration_ms"

        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception:
                success = False
                raise
            finally:
                duration_ms = (time.time() - start) * 1000
                collector = get_metrics_collector()
                collector.record_metric(
                    name, duration_ms, tags={"success": str(success)}
                )

        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception:
                success = False
                raise
            finally:
                duration_ms = (time.time() - start) * 1000
                collector = get_metrics_collector()
                collector.record_metric(
                    name, duration_ms, tags={"success": str(success)}
                )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    return decorator
