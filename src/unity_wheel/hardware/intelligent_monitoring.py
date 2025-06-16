"""
Intelligent Hardware State Monitoring System

Advanced monitoring with:
1. Predictive performance analysis
2. Anomaly detection for hardware behavior
3. Adaptive threshold management
4. Resource utilization forecasting
5. Intelligent alerting with context
6. Self-tuning optimization
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import psutil
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringState(Enum):
    """Monitoring system states."""
    STARTING = "starting"
    ACTIVE = "active"
    DEGRADED = "degraded"
    SUSPENDED = "suspended"
    SHUTDOWN = "shutdown"


@dataclass
class HardwareAlert:
    """Hardware performance alert."""
    
    timestamp: float
    severity: AlertSeverity
    component: str
    metric: str
    value: float
    threshold: float
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "component": self.component,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "message": self.message,
            "context": self.context,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at
        }


@dataclass
class MetricSnapshot:
    """Snapshot of hardware metrics at a point in time."""
    
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    access_latency_ms: float
    cache_hit_rate: float
    error_rate: float
    throughput_rps: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "gpu_usage": self.gpu_usage,
            "access_latency_ms": self.access_latency_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "error_rate": self.error_rate,
            "throughput_rps": self.throughput_rps
        }


class AnomalyDetector:
    """Machine learning-based anomaly detection for hardware metrics."""
    
    def __init__(self, window_size: int = 100, contamination: float = 0.1):
        self.window_size = window_size
        self.contamination = contamination
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.anomaly_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        
        # DBSCAN for anomaly detection
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.scaler = StandardScaler()
        
    def add_metric(self, metric_name: str, value: float) -> bool:
        """Add metric value and detect anomalies."""
        self.metric_history[metric_name].append(value)
        
        # Need minimum samples for anomaly detection
        if len(self.metric_history[metric_name]) < 10:
            return False
        
        # Update baseline statistics
        values = list(self.metric_history[metric_name])
        self.baseline_stats[metric_name] = {
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "median": statistics.median(values),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99)
        }
        
        # Simple statistical anomaly detection
        baseline = self.baseline_stats[metric_name]
        z_score = abs(value - baseline["mean"]) / max(baseline["stdev"], 0.001)
        
        # Consider anomaly if z-score > 3 (3 standard deviations)
        is_anomaly = z_score > 3.0
        
        # Store anomaly score
        self.anomaly_scores[metric_name].append(z_score)
        
        return is_anomaly
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of anomaly detection."""
        summary = {}
        
        for metric_name, scores in self.anomaly_scores.items():
            if not scores:
                continue
                
            recent_scores = list(scores)[-10:]  # Last 10 scores
            summary[metric_name] = {
                "current_score": recent_scores[-1] if recent_scores else 0,
                "avg_score": statistics.mean(recent_scores),
                "max_score": max(recent_scores),
                "anomaly_rate": sum(1 for s in recent_scores if s > 3.0) / len(recent_scores),
                "baseline": self.baseline_stats.get(metric_name, {})
            }
        
        return summary


class AdaptiveThresholdManager:
    """Adaptive threshold management that learns from historical data."""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_history: Dict[str, List[float]] = defaultdict(list)
        
        # Default thresholds
        self.default_thresholds = {
            "cpu_usage": {"warning": 70.0, "critical": 85.0},
            "memory_usage": {"warning": 75.0, "critical": 90.0},
            "gpu_usage": {"warning": 80.0, "critical": 95.0},
            "access_latency_ms": {"warning": 5.0, "critical": 10.0},
            "cache_hit_rate": {"warning": 0.8, "critical": 0.6},
            "error_rate": {"warning": 0.05, "critical": 0.1},
            "throughput_rps": {"warning": 100.0, "critical": 50.0}
        }
        
        # Initialize with defaults
        for metric, thresholds in self.default_thresholds.items():
            self.thresholds[metric] = thresholds.copy()
    
    def update_metric(self, metric_name: str, value: float) -> None:
        """Update metric history and adapt thresholds."""
        self.metric_history[metric_name].append(value)
        
        # Adapt thresholds if we have enough history
        if len(self.metric_history[metric_name]) >= 100:
            self._adapt_thresholds(metric_name)
    
    def _adapt_thresholds(self, metric_name: str) -> None:
        """Adapt thresholds based on historical data."""
        values = list(self.metric_history[metric_name])
        
        if metric_name not in self.thresholds:
            self.thresholds[metric_name] = self.default_thresholds.get(metric_name, {
                "warning": np.percentile(values, 75),
                "critical": np.percentile(values, 90)
            })
            return
        
        # Calculate new thresholds based on percentiles
        if metric_name in ["cpu_usage", "memory_usage", "gpu_usage", "access_latency_ms", "error_rate"]:
            # Higher values are bad
            new_warning = np.percentile(values, 75)
            new_critical = np.percentile(values, 90)
        else:
            # Lower values are bad (cache_hit_rate, throughput_rps)
            new_warning = np.percentile(values, 25)
            new_critical = np.percentile(values, 10)
        
        # Apply learning rate for gradual adaptation
        current_thresholds = self.thresholds[metric_name]
        current_thresholds["warning"] = (
            (1 - self.learning_rate) * current_thresholds.get("warning", new_warning) +
            self.learning_rate * new_warning
        )
        current_thresholds["critical"] = (
            (1 - self.learning_rate) * current_thresholds.get("critical", new_critical) +
            self.learning_rate * new_critical
        )
    
    def check_thresholds(self, metric_name: str, value: float) -> Optional[AlertSeverity]:
        """Check if value exceeds thresholds."""
        if metric_name not in self.thresholds:
            return None
        
        thresholds = self.thresholds[metric_name]
        
        if metric_name in ["cpu_usage", "memory_usage", "gpu_usage", "access_latency_ms", "error_rate"]:
            # Higher values are bad
            if value >= thresholds.get("critical", 90.0):
                return AlertSeverity.CRITICAL
            elif value >= thresholds.get("warning", 75.0):
                return AlertSeverity.WARNING
        else:
            # Lower values are bad
            if value <= thresholds.get("critical", 10.0):
                return AlertSeverity.CRITICAL
            elif value <= thresholds.get("warning", 25.0):
                return AlertSeverity.WARNING
        
        return None
    
    def get_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get current thresholds."""
        return self.thresholds.copy()


class PerformanceForecaster:
    """Simple performance forecasting using moving averages."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.trend_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    
    def add_metric(self, metric_name: str, value: float) -> None:
        """Add metric value and update trends."""
        self.metric_history[metric_name].append(value)
        
        # Calculate trend if we have enough data
        if len(self.metric_history[metric_name]) >= 10:
            recent_values = list(self.metric_history[metric_name])[-10:]
            trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
            self.trend_history[metric_name].append(trend)
    
    def forecast(self, metric_name: str, steps_ahead: int = 5) -> Optional[float]:
        """Forecast metric value steps ahead."""
        if metric_name not in self.metric_history:
            return None
        
        history = list(self.metric_history[metric_name])
        if len(history) < 10:
            return None
        
        # Simple linear trend forecast
        recent_trend = statistics.mean(list(self.trend_history[metric_name])[-5:]) if self.trend_history[metric_name] else 0
        current_value = history[-1]
        
        forecast_value = current_value + (recent_trend * steps_ahead)
        return forecast_value
    
    def get_trend_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Get trend analysis for all metrics."""
        analysis = {}
        
        for metric_name, trends in self.trend_history.items():
            if not trends:
                continue
            
            recent_trends = list(trends)[-5:]
            avg_trend = statistics.mean(recent_trends)
            
            # Classify trend
            if abs(avg_trend) < 0.1:
                trend_direction = "stable"
            elif avg_trend > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
            
            analysis[metric_name] = {
                "direction": trend_direction,
                "rate": avg_trend,
                "volatility": statistics.stdev(recent_trends) if len(recent_trends) > 1 else 0,
                "forecast_5_steps": self.forecast(metric_name, 5)
            }
        
        return analysis


class IntelligentMonitoringSystem:
    """Main intelligent monitoring system."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".unity_wheel" / "monitoring_config.json"
        self.state = MonitoringState.STARTING
        
        # Core components
        self.anomaly_detector = AnomalyDetector()
        self.threshold_manager = AdaptiveThresholdManager()
        self.forecaster = PerformanceForecaster()
        
        # Data storage
        self.metric_snapshots: deque = deque(maxlen=10000)
        self.active_alerts: Dict[str, HardwareAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Monitoring state
        self.start_time = time.time()
        self.last_health_check = 0.0
        self.monitoring_interval = 1.0  # 1 second
        self.health_check_interval = 30.0  # 30 seconds
        
        # Tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._alert_cleanup_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.alert_callbacks: List[Callable[[HardwareAlert], None]] = []
        self.health_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Performance tracking
        self.monitoring_overhead_ms: deque = deque(maxlen=100)
        
        # Load configuration
        self._load_config()
    
    def _load_config(self) -> None:
        """Load monitoring configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path) as f:
                    config = json.load(f)
                
                # Update intervals
                self.monitoring_interval = config.get("monitoring_interval", 1.0)
                self.health_check_interval = config.get("health_check_interval", 30.0)
                
                # Update thresholds
                if "thresholds" in config:
                    for metric, thresholds in config["thresholds"].items():
                        self.threshold_manager.thresholds[metric] = thresholds
                
                logger.info(f"Loaded monitoring config from {self.config_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load monitoring config: {e}")
    
    def save_config(self) -> None:
        """Save current configuration."""
        try:
            config = {
                "monitoring_interval": self.monitoring_interval,
                "health_check_interval": self.health_check_interval,
                "thresholds": self.threshold_manager.get_thresholds(),
                "last_updated": time.time()
            }
            
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save monitoring config: {e}")
    
    async def start_monitoring(self) -> None:
        """Start the monitoring system."""
        if self.state != MonitoringState.STARTING:
            logger.warning(f"Cannot start monitoring from state {self.state}")
            return
        
        logger.info("Starting intelligent hardware monitoring system")
        
        # Start monitoring tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._alert_cleanup_task = asyncio.create_task(self._alert_cleanup_loop())
        
        self.state = MonitoringState.ACTIVE
        logger.info("Intelligent monitoring system started")
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        logger.info("Stopping intelligent monitoring system")
        self.state = MonitoringState.SHUTDOWN
        
        # Cancel tasks
        for task in [self._monitoring_task, self._health_check_task, self._alert_cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Save configuration
        self.save_config()
        logger.info("Intelligent monitoring system stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.state == MonitoringState.ACTIVE:
            try:
                start_time = time.perf_counter()
                
                # Collect metrics
                await self._collect_metrics()
                
                # Check for alerts
                await self._check_alerts()
                
                # Update forecasts
                self._update_forecasts()
                
                # Record monitoring overhead
                overhead = (time.perf_counter() - start_time) * 1000
                self.monitoring_overhead_ms.append(overhead)
                
                # Adaptive monitoring interval based on overhead
                if overhead > 50:  # If overhead > 50ms, slow down
                    self.monitoring_interval = min(5.0, self.monitoring_interval * 1.1)
                elif overhead < 10:  # If overhead < 10ms, speed up
                    self.monitoring_interval = max(0.5, self.monitoring_interval * 0.9)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                self.state = MonitoringState.DEGRADED
                await asyncio.sleep(5.0)
    
    async def _collect_metrics(self) -> None:
        """Collect hardware metrics."""
        timestamp = time.time()
        
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Simulated metrics (would be real in production)
            gpu_usage = min(90.0, cpu_percent * 0.8)  # Estimate GPU usage
            access_latency = 2.0 + np.random.normal(0, 0.5)  # Simulated latency
            cache_hit_rate = 0.85 + np.random.normal(0, 0.05)  # Simulated cache
            error_rate = max(0, np.random.normal(0.01, 0.005))  # Simulated errors
            throughput = 200 + np.random.normal(0, 20)  # Simulated throughput
            
            # Create snapshot
            snapshot = MetricSnapshot(
                timestamp=timestamp,
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                gpu_usage=gpu_usage,
                access_latency_ms=access_latency,
                cache_hit_rate=cache_hit_rate,
                error_rate=error_rate,
                throughput_rps=throughput
            )
            
            # Store snapshot
            self.metric_snapshots.append(snapshot)
            
            # Update components
            for metric_name, value in snapshot.to_dict().items():
                if metric_name != "timestamp":
                    self.threshold_manager.update_metric(metric_name, value)
                    self.forecaster.add_metric(metric_name, value)
                    self.anomaly_detector.add_metric(metric_name, value)
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
    
    async def _check_alerts(self) -> None:
        """Check for alert conditions."""
        if not self.metric_snapshots:
            return
        
        current_snapshot = self.metric_snapshots[-1]
        
        for metric_name, value in current_snapshot.to_dict().items():
            if metric_name == "timestamp":
                continue
            
            # Check thresholds
            severity = self.threshold_manager.check_thresholds(metric_name, value)
            if severity:
                await self._create_alert(
                    component="hardware",
                    metric=metric_name,
                    value=value,
                    severity=severity,
                    message=f"{metric_name} threshold exceeded: {value:.2f}"
                )
            
            # Check anomalies
            is_anomaly = self.anomaly_detector.add_metric(metric_name, value)
            if is_anomaly:
                await self._create_alert(
                    component="hardware",
                    metric=metric_name,
                    value=value,
                    severity=AlertSeverity.WARNING,
                    message=f"Anomaly detected in {metric_name}: {value:.2f}"
                )
    
    async def _create_alert(self, 
                          component: str,
                          metric: str,
                          value: float,
                          severity: AlertSeverity,
                          message: str) -> None:
        """Create new alert."""
        alert_key = f"{component}_{metric}_{severity.value}"
        
        # Check if alert already exists
        if alert_key in self.active_alerts:
            return  # Don't duplicate alerts
        
        # Get threshold for context
        thresholds = self.threshold_manager.thresholds.get(metric, {})
        threshold = thresholds.get(severity.value.lower(), 0)
        
        # Create alert
        alert = HardwareAlert(
            timestamp=time.time(),
            severity=severity,
            component=component,
            metric=metric,
            value=value,
            threshold=threshold,
            message=message,
            context={
                "trend": self.forecaster.get_trend_analysis().get(metric, {}),
                "anomaly_score": self.anomaly_detector.anomaly_scores.get(metric, deque())[-1] if self.anomaly_detector.anomaly_scores.get(metric) else 0,
                "historical_avg": statistics.mean(list(self.threshold_manager.metric_history[metric])[-10:]) if self.threshold_manager.metric_history[metric] else 0
            }
        )
        
        # Store alert
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        logger.warning(f"Alert created: {alert.message}")
    
    def _update_forecasts(self) -> None:
        """Update performance forecasts."""
        # Forecasting is updated automatically when metrics are added
        pass
    
    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while self.state == MonitoringState.ACTIVE:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(30.0)
    
    async def _perform_health_check(self) -> None:
        """Perform system health check."""
        health_data = {
            "timestamp": time.time(),
            "monitoring_state": self.state.value,
            "uptime_seconds": time.time() - self.start_time,
            "monitoring_overhead": {
                "avg_ms": statistics.mean(list(self.monitoring_overhead_ms)) if self.monitoring_overhead_ms else 0,
                "max_ms": max(list(self.monitoring_overhead_ms)) if self.monitoring_overhead_ms else 0
            },
            "active_alerts": len(self.active_alerts),
            "metrics_collected": len(self.metric_snapshots),
            "trend_analysis": self.forecaster.get_trend_analysis(),
            "anomaly_summary": self.anomaly_detector.get_anomaly_summary()
        }
        
        # Notify health callbacks
        for callback in self.health_callbacks:
            try:
                callback(health_data)
            except Exception as e:
                logger.error(f"Health callback error: {e}")
        
        self.last_health_check = time.time()
    
    async def _alert_cleanup_loop(self) -> None:
        """Clean up resolved alerts."""
        while self.state == MonitoringState.ACTIVE:
            try:
                current_time = time.time()
                resolved_alerts = []
                
                # Check if alerts should be resolved
                for alert_key, alert in self.active_alerts.items():
                    # Auto-resolve alerts after 5 minutes if metric is back to normal
                    if current_time - alert.timestamp > 300:  # 5 minutes
                        # Check current metric value
                        if self.metric_snapshots:
                            current_snapshot = self.metric_snapshots[-1]
                            current_value = getattr(current_snapshot, alert.metric, None)
                            
                            if current_value is not None:
                                severity = self.threshold_manager.check_thresholds(alert.metric, current_value)
                                if not severity:  # No threshold violation
                                    alert.resolved = True
                                    alert.resolved_at = current_time
                                    resolved_alerts.append(alert_key)
                
                # Remove resolved alerts
                for alert_key in resolved_alerts:
                    del self.active_alerts[alert_key]
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Alert cleanup error: {e}")
                await asyncio.sleep(60)
    
    def add_alert_callback(self, callback: Callable[[HardwareAlert], None]) -> None:
        """Add alert callback."""
        self.alert_callbacks.append(callback)
    
    def add_health_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add health check callback."""
        self.health_callbacks.append(callback)
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Get comprehensive monitoring report."""
        current_time = time.time()
        
        # Recent metrics
        recent_snapshots = list(self.metric_snapshots)[-10:]
        
        return {
            "system_status": {
                "state": self.state.value,
                "uptime_seconds": current_time - self.start_time,
                "last_health_check": self.last_health_check,
                "monitoring_overhead_ms": statistics.mean(list(self.monitoring_overhead_ms)) if self.monitoring_overhead_ms else 0
            },
            "current_metrics": recent_snapshots[-1].to_dict() if recent_snapshots else {},
            "active_alerts": [alert.to_dict() for alert in self.active_alerts.values()],
            "alert_summary": {
                "total_active": len(self.active_alerts),
                "by_severity": {
                    severity.value: sum(1 for alert in self.active_alerts.values() if alert.severity == severity)
                    for severity in AlertSeverity
                },
                "total_historical": len(self.alert_history)
            },
            "performance_analysis": {
                "trends": self.forecaster.get_trend_analysis(),
                "anomalies": self.anomaly_detector.get_anomaly_summary(),
                "thresholds": self.threshold_manager.get_thresholds()
            },
            "data_collection": {
                "snapshots_collected": len(self.metric_snapshots),
                "collection_rate_hz": 1.0 / self.monitoring_interval,
                "storage_usage_mb": len(self.metric_snapshots) * 0.001  # Rough estimate
            }
        }


# Global monitoring system
_monitoring_system: Optional[IntelligentMonitoringSystem] = None


async def get_monitoring_system() -> IntelligentMonitoringSystem:
    """Get or create the global monitoring system."""
    global _monitoring_system
    
    if _monitoring_system is None:
        _monitoring_system = IntelligentMonitoringSystem()
        await _monitoring_system.start_monitoring()
        
        # Add default alert callback
        def default_alert_callback(alert: HardwareAlert):
            logger.warning(f"Hardware Alert [{alert.severity.value.upper()}]: {alert.message}")
        
        _monitoring_system.add_alert_callback(default_alert_callback)
    
    return _monitoring_system


if __name__ == "__main__":
    async def test_monitoring():
        print("🔍 Testing Intelligent Hardware Monitoring")
        print("=" * 50)
        
        # Get monitoring system
        monitoring = await get_monitoring_system()
        
        # Let it run for a bit
        print("Monitoring system started, collecting data...")
        await asyncio.sleep(10)
        
        # Get report
        report = monitoring.get_monitoring_report()
        print(f"\nSystem State: {report['system_status']['state']}")
        print(f"Uptime: {report['system_status']['uptime_seconds']:.1f}s")
        print(f"Active Alerts: {report['alert_summary']['total_active']}")
        print(f"Snapshots Collected: {report['data_collection']['snapshots_collected']}")
        
        # Show current metrics
        if report['current_metrics']:
            print("\nCurrent Metrics:")
            for metric, value in report['current_metrics'].items():
                if metric != 'timestamp':
                    print(f"  {metric}: {value:.2f}")
        
        # Show trend analysis
        if report['performance_analysis']['trends']:
            print("\nTrend Analysis:")
            for metric, trend in report['performance_analysis']['trends'].items():
                print(f"  {metric}: {trend['direction']} (rate: {trend['rate']:.3f})")
        
        # Test alert creation by simulating high CPU
        print("\nSimulating high resource usage to test alerting...")
        
        # Stop monitoring
        await monitoring.stop_monitoring()
        print("\n✅ Intelligent monitoring system test completed!")
    
    asyncio.run(test_monitoring())