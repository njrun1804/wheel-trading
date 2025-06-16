import re
#!/usr/bin/env python3
"""
Einstein Search Performance Monitor

Real-time monitoring and alerting for search performance with:
1. Real-time performance tracking
2. Automated alerting for performance degradation
3. Memory usage monitoring
4. Load balancing recommendations
5. Performance trend analysis
"""

import asyncio
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import psutil

from einstein.einstein_config import get_einstein_config

logger = logging.getLogger(__name__)


@dataclass
class PerformanceAlert:
    """Performance alert data structure."""
    
    alert_type: str  # 'latency', 'memory', 'error_rate', 'throughput'
    severity: str    # 'warning', 'critical', 'info'
    message: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False


@dataclass
class SearchPerformanceSnapshot:
    """Snapshot of search performance metrics."""
    
    timestamp: float
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    queries_per_second: float
    cache_hit_rate_percent: float
    error_rate_percent: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_searches: int
    total_searches: int


class PerformanceMetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.response_times = deque(maxlen=history_size)
        self.cache_hits = deque(maxlen=history_size)
        self.errors = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        self.query_counts = defaultdict(int)
        
        # System metrics
        self.memory_snapshots = deque(maxlen=100)
        self.cpu_snapshots = deque(maxlen=100)
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def record_search(
        self, 
        response_time_ms: float, 
        cache_hit: bool, 
        error: bool = False,
        query_type: Optional[str] = None
    ):
        """Record a search operation."""
        with self._lock:
            current_time = time.time()
            
            self.response_times.append(response_time_ms)
            self.cache_hits.append(cache_hit)
            self.errors.append(error)
            self.timestamps.append(current_time)
            
            if query_type:
                self.query_counts[query_type] += 1
    
    def record_system_metrics(self):
        """Record system-level metrics."""
        try:
            # Memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            
            with self._lock:
                self.memory_snapshots.append(memory_mb)
                self.cpu_snapshots.append(cpu_percent)
                
        except Exception as e:
            logger.debug(f"Failed to collect system metrics: {e}")
    
    def get_snapshot(self) -> SearchPerformanceSnapshot:
        """Get current performance snapshot."""
        with self._lock:
            if not self.response_times:
                return SearchPerformanceSnapshot(
                    timestamp=time.time(),
                    avg_response_time_ms=0,
                    p50_response_time_ms=0,
                    p95_response_time_ms=0,
                    p99_response_time_ms=0,
                    queries_per_second=0,
                    cache_hit_rate_percent=0,
                    error_rate_percent=0,
                    memory_usage_mb=0,
                    cpu_usage_percent=0,
                    active_searches=0,
                    total_searches=0,
                )
            
            # Calculate response time statistics
            response_times_array = np.array(list(self.response_times))
            avg_response = np.mean(response_times_array)
            p50_response = np.percentile(response_times_array, 50)
            p95_response = np.percentile(response_times_array, 95)
            p99_response = np.percentile(response_times_array, 99)
            
            # Calculate throughput (last minute)
            current_time = time.time()
            minute_ago = current_time - 60
            recent_searches = sum(1 for t in self.timestamps if t > minute_ago)
            qps = recent_searches / 60.0
            
            # Calculate cache hit rate
            cache_hit_rate = (sum(self.cache_hits) / len(self.cache_hits) * 100) if self.cache_hits else 0
            
            # Calculate error rate
            error_rate = (sum(self.errors) / len(self.errors) * 100) if self.errors else 0
            
            # System metrics
            avg_memory = np.mean(self.memory_snapshots) if self.memory_snapshots else 0
            avg_cpu = np.mean(self.cpu_snapshots) if self.cpu_snapshots else 0
            
            return SearchPerformanceSnapshot(
                timestamp=current_time,
                avg_response_time_ms=avg_response,
                p50_response_time_ms=p50_response,
                p95_response_time_ms=p95_response,
                p99_response_time_ms=p99_response,
                queries_per_second=qps,
                cache_hit_rate_percent=cache_hit_rate,
                error_rate_percent=error_rate,
                memory_usage_mb=avg_memory,
                cpu_usage_percent=avg_cpu,
                active_searches=0,  # Would be tracked by search engine
                total_searches=len(self.response_times),
            )


class PerformanceAlerting:
    """Handles performance alerting and notifications."""
    
    def __init__(self, config):
        self.config = config
        self.alert_handlers = []
        self.active_alerts = {}
        self.alert_history = deque(maxlen=500)
        
        # Alert thresholds
        self.thresholds = {
            'p99_latency_ms': 50,
            'avg_latency_ms': 25,
            'error_rate_percent': 5,
            'memory_usage_mb': 1000,
            'cpu_usage_percent': 80,
            'cache_hit_rate_percent': 70,  # Minimum acceptable cache hit rate
        }
    
    def add_alert_handler(self, handler: Callable[[PerformanceAlert], None]):
        """Add alert handler function."""
        self.alert_handlers.append(handler)
    
    def check_performance_thresholds(self, snapshot: SearchPerformanceSnapshot):
        """Check performance against thresholds and generate alerts."""
        current_time = time.time()
        
        # Check P99 latency
        if snapshot.p99_response_time_ms > self.thresholds['p99_latency_ms']:
            self._create_alert(
                'latency',
                'critical' if snapshot.p99_response_time_ms > 75 else 'warning',
                f"P99 latency {snapshot.p99_response_time_ms:.1f}ms exceeds threshold {self.thresholds['p99_latency_ms']}ms",
                snapshot.p99_response_time_ms,
                self.thresholds['p99_latency_ms']
            )
        
        # Check average latency
        if snapshot.avg_response_time_ms > self.thresholds['avg_latency_ms']:
            self._create_alert(
                'latency',
                'warning',
                f"Average latency {snapshot.avg_response_time_ms:.1f}ms exceeds threshold {self.thresholds['avg_latency_ms']}ms",
                snapshot.avg_response_time_ms,
                self.thresholds['avg_latency_ms']
            )
        
        # Check error rate
        if snapshot.error_rate_percent > self.thresholds['error_rate_percent']:
            self._create_alert(
                'error_rate',
                'critical',
                f"Error rate {snapshot.error_rate_percent:.1f}% exceeds threshold {self.thresholds['error_rate_percent']}%",
                snapshot.error_rate_percent,
                self.thresholds['error_rate_percent']
            )
        
        # Check memory usage
        if snapshot.memory_usage_mb > self.thresholds['memory_usage_mb']:
            self._create_alert(
                'memory',
                'warning',
                f"Memory usage {snapshot.memory_usage_mb:.1f}MB exceeds threshold {self.thresholds['memory_usage_mb']}MB",
                snapshot.memory_usage_mb,
                self.thresholds['memory_usage_mb']
            )
        
        # Check CPU usage
        if snapshot.cpu_usage_percent > self.thresholds['cpu_usage_percent']:
            self._create_alert(
                'cpu',
                'warning',
                f"CPU usage {snapshot.cpu_usage_percent:.1f}% exceeds threshold {self.thresholds['cpu_usage_percent']}%",
                snapshot.cpu_usage_percent,
                self.thresholds['cpu_usage_percent']
            )
        
        # Check cache hit rate (low is bad)
        if snapshot.cache_hit_rate_percent < self.thresholds['cache_hit_rate_percent']:
            self._create_alert(
                'cache',
                'warning',
                f"Cache hit rate {snapshot.cache_hit_rate_percent:.1f}% below threshold {self.thresholds['cache_hit_rate_percent']}%",
                snapshot.cache_hit_rate_percent,
                self.thresholds['cache_hit_rate_percent']
            )
    
    def _create_alert(
        self, 
        alert_type: str, 
        severity: str, 
        message: str, 
        value: float, 
        threshold: float
    ):
        """Create and dispatch alert."""
        alert_key = f"{alert_type}_{severity}"
        
        # Avoid duplicate alerts
        if alert_key in self.active_alerts:
            last_alert_time = self.active_alerts[alert_key]
            if time.time() - last_alert_time < 300:  # 5 minute cooldown
                return
        
        alert = PerformanceAlert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            value=value,
            threshold=threshold,
        )
        
        self.active_alerts[alert_key] = time.time()
        self.alert_history.append(alert)
        
        # Dispatch to handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def get_active_alerts(self) -> list[PerformanceAlert]:
        """Get currently active alerts."""
        return [alert for alert in self.alert_history if not alert.resolved]


class SearchPerformanceMonitor:
    """Real-time search performance monitor."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.config = get_einstein_config()
        
        # Core components
        self.metrics_collector = PerformanceMetricsCollector()
        self.alerting = PerformanceAlerting(self.config)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_task = None
        self.monitor_interval_seconds = 10
        
        # Performance history
        self.performance_snapshots = deque(maxlen=1000)
        
        # Set up default alert handler
        self.alerting.add_alert_handler(self._default_alert_handler)
    
    def _default_alert_handler(self, alert: PerformanceAlert):
        """Default alert handler that logs alerts."""
        level = logging.ERROR if alert.severity == 'critical' else logging.WARNING
        logger.log(level, f"PERFORMANCE ALERT: {alert.message}")
    
    async def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
        
        logger.info("Starting Einstein search performance monitoring...")
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.monitoring_active:
            return
        
        logger.info("Stopping Einstein search performance monitoring...")
        self.monitoring_active = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.monitoring_active:
                # Collect system metrics
                self.metrics_collector.record_system_metrics()
                
                # Get performance snapshot
                snapshot = self.metrics_collector.get_snapshot()
                self.performance_snapshots.append(snapshot)
                
                # Check for performance issues
                self.alerting.check_performance_thresholds(snapshot)
                
                # Wait for next iteration
                await asyncio.sleep(self.monitor_interval_seconds)
                
        except asyncio.CancelledError:
            logger.info("Performance monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Performance monitoring loop failed: {e}")
    
    def record_search_operation(
        self,
        response_time_ms: float,
        cache_hit: bool,
        error: bool = False,
        query_type: Optional[str] = None
    ):
        """Record a search operation for monitoring."""
        self.metrics_collector.record_search(
            response_time_ms, cache_hit, error, query_type
        )
    
    def get_current_performance(self) -> SearchPerformanceSnapshot:
        """Get current performance snapshot."""
        return self.metrics_collector.get_snapshot()
    
    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        current_snapshot = self.get_current_performance()
        active_alerts = self.alerting.get_active_alerts()
        
        # Calculate trends
        trends = self._calculate_trends()
        
        return {
            "current_performance": {
                "timestamp": current_snapshot.timestamp,
                "avg_response_time_ms": current_snapshot.avg_response_time_ms,
                "p50_response_time_ms": current_snapshot.p50_response_time_ms,
                "p95_response_time_ms": current_snapshot.p95_response_time_ms,
                "p99_response_time_ms": current_snapshot.p99_response_time_ms,
                "queries_per_second": current_snapshot.queries_per_second,
                "cache_hit_rate_percent": current_snapshot.cache_hit_rate_percent,
                "error_rate_percent": current_snapshot.error_rate_percent,
                "memory_usage_mb": current_snapshot.memory_usage_mb,
                "cpu_usage_percent": current_snapshot.cpu_usage_percent,
                "total_searches": current_snapshot.total_searches,
            },
            "performance_status": {
                "health_score": self._calculate_health_score(current_snapshot),
                "target_met": current_snapshot.p99_response_time_ms < 50,
                "performance_grade": self._get_performance_grade(current_snapshot),
            },
            "alerts": {
                "active_count": len(active_alerts),
                "active_alerts": [
                    {
                        "type": alert.alert_type,
                        "severity": alert.severity,
                        "message": alert.message,
                        "timestamp": alert.timestamp,
                    }
                    for alert in active_alerts
                ],
                "total_alerts": len(self.alerting.alert_history),
            },
            "trends": trends,
            "recommendations": self._get_performance_recommendations(current_snapshot),
        }
    
    def _calculate_trends(self) -> dict[str, Any]:
        """Calculate performance trends."""
        if len(self.performance_snapshots) < 10:
            return {"status": "insufficient_data"}
        
        # Get recent snapshots for trend analysis
        recent_snapshots = list(self.performance_snapshots)[-50:]
        response_times = [s.avg_response_time_ms for s in recent_snapshots]
        cache_hit_rates = [s.cache_hit_rate_percent for s in recent_snapshots]
        
        # Calculate trends (simple linear regression)
        x = np.arange(len(response_times))
        response_time_trend = np.polyfit(x, response_times, 1)[0]  # Slope
        cache_hit_trend = np.polyfit(x, cache_hit_rates, 1)[0]
        
        return {
            "response_time_trend": {
                "direction": "improving" if response_time_trend < 0 else "degrading",
                "slope_ms_per_sample": response_time_trend,
                "significance": "high" if abs(response_time_trend) > 1 else "low",
            },
            "cache_hit_trend": {
                "direction": "improving" if cache_hit_trend > 0 else "degrading", 
                "slope_percent_per_sample": cache_hit_trend,
                "significance": "high" if abs(cache_hit_trend) > 0.5 else "low",
            },
        }
    
    def _calculate_health_score(self, snapshot: SearchPerformanceSnapshot) -> float:
        """Calculate overall health score (0-100)."""
        score = 100.0
        
        # Latency penalty
        if snapshot.p99_response_time_ms > 50:
            score -= min(30, (snapshot.p99_response_time_ms - 50) / 2)
        
        # Error rate penalty
        if snapshot.error_rate_percent > 0:
            score -= min(20, snapshot.error_rate_percent * 4)
        
        # Cache hit rate bonus/penalty
        if snapshot.cache_hit_rate_percent > 80:
            score += 5
        elif snapshot.cache_hit_rate_percent < 50:
            score -= 15
        
        # Memory usage penalty
        if snapshot.memory_usage_mb > 1000:
            score -= min(10, (snapshot.memory_usage_mb - 1000) / 100)
        
        return max(0, score)
    
    def _get_performance_grade(self, snapshot: SearchPerformanceSnapshot) -> str:
        """Get performance grade."""
        health_score = self._calculate_health_score(snapshot)
        
        if health_score >= 90:
            return "A"
        elif health_score >= 80:
            return "B" 
        elif health_score >= 70:
            return "C"
        elif health_score >= 60:
            return "D"
        else:
            return "F"
    
    def _get_performance_recommendations(self, snapshot: SearchPerformanceSnapshot) -> list[str]:
        """Get performance improvement recommendations."""
        recommendations = []
        
        if snapshot.p99_response_time_ms > 50:
            recommendations.append("Consider increasing cache size or implementing query optimization")
        
        if snapshot.cache_hit_rate_percent < 70:
            recommendations.append("Cache hit rate is low - review cache policies and pre-warming strategies")
        
        if snapshot.error_rate_percent > 2:
            recommendations.append("Error rate is elevated - investigate search failures and timeouts")
        
        if snapshot.memory_usage_mb > 800:
            recommendations.append("Memory usage is high - consider implementing memory optimization")
        
        if snapshot.queries_per_second < 10:
            recommendations.append("Low throughput detected - review concurrency settings and bottlenecks")
        
        if not recommendations:
            recommendations.append("Performance is optimal - continue monitoring")
        
        return recommendations


# Global monitor instance
_performance_monitor = None


def get_search_performance_monitor(project_root: Optional[Path] = None) -> SearchPerformanceMonitor:
    """Get global search performance monitor instance."""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = SearchPerformanceMonitor(project_root)
    
    return _performance_monitor


if __name__ == "__main__":
    async def demo_performance_monitoring():
        """Demonstrate performance monitoring capabilities."""
        print("🔍 Einstein Search Performance Monitor Demo")
        print("=" * 50)
        
        monitor = get_search_performance_monitor()
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Simulate search operations
        print("\n1️⃣ Simulating search operations...")
        
        for i in range(50):
            # Simulate varying performance
            response_time = 20 + np.random.normal(0, 10)  # Base 20ms with variation
            cache_hit = np.random.random() > 0.3  # 70% cache hit rate
            error = np.random.random() > 0.95     # 5% error rate
            
            # Add some high latency spikes
            if i % 10 == 0:
                response_time += 30  # Spike to ~50ms
            
            monitor.record_search_operation(
                response_time_ms=response_time,
                cache_hit=cache_hit,
                error=error,
                query_type="test"
            )
            
            # Small delay to simulate real operations
            await asyncio.sleep(0.1)
        
        # Wait for monitoring to collect data
        await asyncio.sleep(2)
        
        # Get performance report
        print("\n2️⃣ Performance Report:")
        report = monitor.get_performance_report()
        
        current = report["current_performance"]
        status = report["performance_status"]
        alerts = report["alerts"]
        
        print(f"   Average response time: {current['avg_response_time_ms']:.1f}ms")
        print(f"   P99 response time: {current['p99_response_time_ms']:.1f}ms")
        print(f"   Cache hit rate: {current['cache_hit_rate_percent']:.1f}%")
        print(f"   Error rate: {current['error_rate_percent']:.1f}%")
        print(f"   Health score: {status['health_score']:.1f}/100 (Grade: {status['performance_grade']})")
        print(f"   Target met: {'✅' if status['target_met'] else '❌'}")
        
        # Show alerts
        if alerts["active_count"] > 0:
            print(f"\n⚠️ Active Alerts ({alerts['active_count']}):")
            for alert in alerts["active_alerts"]:
                print(f"   - {alert['severity'].upper()}: {alert['message']}")
        else:
            print("\n✅ No active performance alerts")
        
        # Show recommendations
        print(f"\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"   - {rec}")
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        print("\n✅ Performance monitoring demo complete!")
    
    asyncio.run(demo_performance_monitoring())