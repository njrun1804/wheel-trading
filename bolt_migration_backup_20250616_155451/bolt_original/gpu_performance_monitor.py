#!/usr/bin/env python3
"""
GPU Performance Monitoring System for Bolt
Real-time monitoring and optimization of GPU acceleration performance
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GPUPerformanceMetrics:
    """Performance metrics for GPU operations."""
    timestamp: float
    operation_name: str
    execution_time_ms: float
    gpu_memory_mb: float
    gpu_utilization_percent: float
    cpu_usage_percent: float
    memory_usage_mb: float
    throughput_ops_per_sec: float
    batch_size: int
    data_size_mb: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    timestamp: float
    total_gpu_memory_mb: float
    available_gpu_memory_mb: float
    gpu_temperature_celsius: Optional[float]
    cpu_usage_percent: float
    ram_usage_mb: float
    ram_available_mb: float
    disk_io_read_mb_per_sec: float
    disk_io_write_mb_per_sec: float
    network_io_mb_per_sec: float


@dataclass
class PerformanceAlert:
    """Performance alert for issues."""
    timestamp: float
    severity: str  # 'info', 'warning', 'error', 'critical'
    category: str  # 'memory', 'performance', 'error', 'utilization'
    message: str
    metrics: Dict[str, Any]
    recommendations: List[str]


class PerformanceAnalyzer:
    """Analyze performance metrics and generate recommendations."""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.performance_thresholds = {
            'gpu_memory_warning': 0.8,  # 80% memory usage
            'gpu_memory_critical': 0.9,  # 90% memory usage
            'cpu_warning': 80.0,  # 80% CPU usage
            'latency_warning_ms': 1000.0,  # 1 second
            'throughput_degradation': 0.5,  # 50% of baseline
            'error_rate_warning': 0.05,  # 5% error rate
        }
    
    def analyze_operation_performance(
        self, 
        operation_metrics: List[GPUPerformanceMetrics]
    ) -> List[PerformanceAlert]:
        """Analyze operation performance and generate alerts."""
        alerts = []
        
        if not operation_metrics:
            return alerts
        
        # Group by operation name
        ops_by_name = defaultdict(list)
        for metric in operation_metrics:
            ops_by_name[metric.operation_name].append(metric)
        
        for op_name, metrics in ops_by_name.items():
            recent_metrics = metrics[-20:]  # Last 20 operations
            
            # Calculate statistics
            avg_latency = sum(m.execution_time_ms for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.gpu_memory_mb for m in recent_metrics) / len(recent_metrics)
            error_rate = sum(1 for m in recent_metrics if not m.success) / len(recent_metrics)
            avg_throughput = sum(m.throughput_ops_per_sec for m in recent_metrics) / len(recent_metrics)
            
            # Check for performance issues
            if avg_latency > self.performance_thresholds['latency_warning_ms']:
                alerts.append(PerformanceAlert(
                    timestamp=time.time(),
                    severity='warning',
                    category='performance',
                    message=f"High latency detected in {op_name}: {avg_latency:.1f}ms",
                    metrics={'operation': op_name, 'avg_latency_ms': avg_latency},
                    recommendations=[
                        "Consider reducing batch size",
                        "Check for memory pressure",
                        "Optimize algorithm complexity"
                    ]
                ))
            
            if error_rate > self.performance_thresholds['error_rate_warning']:
                alerts.append(PerformanceAlert(
                    timestamp=time.time(),
                    severity='error',
                    category='error',
                    message=f"High error rate in {op_name}: {error_rate:.1%}",
                    metrics={'operation': op_name, 'error_rate': error_rate},
                    recommendations=[
                        "Review error logs for root cause",
                        "Check GPU memory availability",
                        "Verify input data validity"
                    ]
                ))
            
            # Check for throughput degradation
            baseline_throughput = self.baseline_metrics.get(f"{op_name}_throughput", avg_throughput)
            if avg_throughput < baseline_throughput * self.performance_thresholds['throughput_degradation']:
                alerts.append(PerformanceAlert(
                    timestamp=time.time(),
                    severity='warning',
                    category='performance',
                    message=f"Throughput degradation in {op_name}: {avg_throughput:.1f} ops/s (baseline: {baseline_throughput:.1f})",
                    metrics={'operation': op_name, 'throughput': avg_throughput, 'baseline': baseline_throughput},
                    recommendations=[
                        "Check for resource contention",
                        "Verify GPU utilization",
                        "Consider workload rebalancing"
                    ]
                ))
        
        return alerts
    
    def analyze_system_metrics(self, system_metrics: List[SystemMetrics]) -> List[PerformanceAlert]:
        """Analyze system metrics and generate alerts."""
        alerts = []
        
        if not system_metrics:
            return alerts
        
        latest = system_metrics[-1]
        
        # GPU memory analysis
        gpu_memory_usage = latest.total_gpu_memory_mb - latest.available_gpu_memory_mb
        gpu_memory_ratio = gpu_memory_usage / latest.total_gpu_memory_mb if latest.total_gpu_memory_mb > 0 else 0
        
        if gpu_memory_ratio > self.performance_thresholds['gpu_memory_critical']:
            alerts.append(PerformanceAlert(
                timestamp=time.time(),
                severity='critical',
                category='memory',
                message=f"Critical GPU memory usage: {gpu_memory_ratio:.1%} ({gpu_memory_usage:.1f}MB/{latest.total_gpu_memory_mb:.1f}MB)",
                metrics={'gpu_memory_usage_mb': gpu_memory_usage, 'gpu_memory_ratio': gpu_memory_ratio},
                recommendations=[
                    "Reduce batch sizes immediately",
                    "Clear GPU memory caches",
                    "Defer non-critical GPU operations"
                ]
            ))
        elif gpu_memory_ratio > self.performance_thresholds['gpu_memory_warning']:
            alerts.append(PerformanceAlert(
                timestamp=time.time(),
                severity='warning',
                category='memory',
                message=f"High GPU memory usage: {gpu_memory_ratio:.1%}",
                metrics={'gpu_memory_ratio': gpu_memory_ratio},
                recommendations=[
                    "Monitor memory usage closely",
                    "Consider reducing batch sizes",
                    "Clear unnecessary cached data"
                ]
            ))
        
        # CPU usage analysis
        if latest.cpu_usage_percent > self.performance_thresholds['cpu_warning']:
            alerts.append(PerformanceAlert(
                timestamp=time.time(),
                severity='warning',
                category='utilization',
                message=f"High CPU usage: {latest.cpu_usage_percent:.1f}%",
                metrics={'cpu_usage_percent': latest.cpu_usage_percent},
                recommendations=[
                    "Check for CPU-bound operations",
                    "Consider GPU acceleration for compute tasks",
                    "Optimize parallel processing"
                ]
            ))
        
        return alerts
    
    def generate_optimization_recommendations(
        self, 
        operation_metrics: List[GPUPerformanceMetrics],
        system_metrics: List[SystemMetrics]
    ) -> List[str]:
        """Generate optimization recommendations based on performance analysis."""
        recommendations = []
        
        if not operation_metrics or not system_metrics:
            return ["Insufficient data for optimization analysis"]
        
        # Analyze GPU utilization patterns
        avg_gpu_util = sum(m.gpu_utilization_percent for m in operation_metrics[-50:]) / min(50, len(operation_metrics))
        
        if avg_gpu_util < 50:
            recommendations.append("GPU underutilized - consider increasing batch sizes or workload complexity")
        elif avg_gpu_util > 95:
            recommendations.append("GPU highly utilized - monitor for bottlenecks and consider load balancing")
        
        # Analyze memory usage patterns
        memory_metrics = [m.gpu_memory_mb for m in operation_metrics[-20:]]
        if memory_metrics:
            memory_variance = max(memory_metrics) - min(memory_metrics)
            if memory_variance > 200:  # More than 200MB variance
                recommendations.append("High memory usage variance - implement memory pooling")
        
        # Analyze operation performance patterns
        ops_by_name = defaultdict(list)
        for metric in operation_metrics[-100:]:
            ops_by_name[metric.operation_name].append(metric.execution_time_ms)
        
        for op_name, latencies in ops_by_name.items():
            if len(latencies) > 10:
                avg_latency = sum(latencies) / len(latencies)
                if avg_latency > 500:  # More than 500ms
                    recommendations.append(f"Optimize {op_name} operation - average latency {avg_latency:.1f}ms")
        
        return recommendations


class GPUPerformanceMonitor:
    """Comprehensive GPU performance monitoring system."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.operation_metrics: deque[GPUPerformanceMetrics] = deque(maxlen=max_history)
        self.system_metrics: deque[SystemMetrics] = deque(maxlen=max_history // 10)
        self.alerts: deque[PerformanceAlert] = deque(maxlen=1000)
        
        self.analyzer = PerformanceAnalyzer()
        self.monitoring_active = False
        self.monitoring_task = None
        self.metrics_lock = asyncio.Lock()
        
        # Performance tracking
        self.operation_start_times = {}
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get static system information."""
        info = {
            "mlx_available": MLX_AVAILABLE,
            "psutil_available": PSUTIL_AVAILABLE,
            "gpu_cores": 20,  # M4 Pro GPU cores
            "cpu_cores": 12,  # M4 Pro CPU cores (8P + 4E)
        }
        
        if PSUTIL_AVAILABLE:
            info.update({
                "total_ram_gb": psutil.virtual_memory().total / (1024**3),
                "cpu_count": psutil.cpu_count(),
            })
        
        return info
    
    async def start_monitoring(self, interval_seconds: float = 5.0):
        """Start background performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval_seconds))
        logger.info("Started GPU performance monitoring")
    
    async def stop_monitoring(self):
        """Stop background performance monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped GPU performance monitoring")
    
    async def _monitoring_loop(self, interval_seconds: float):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                if system_metrics:
                    async with self.metrics_lock:
                        self.system_metrics.append(system_metrics)
                
                # Analyze for alerts
                alerts = self.analyzer.analyze_system_metrics(list(self.system_metrics)[-10:])
                for alert in alerts:
                    await self._handle_alert(alert)
                
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def _collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Collect current system performance metrics."""
        try:
            current_time = time.time()
            
            # GPU metrics
            gpu_memory_total = 600.0  # M4 Pro estimated GPU memory in MB
            gpu_memory_available = gpu_memory_total
            
            if MLX_AVAILABLE:
                try:
                    gpu_memory_used = mx.metal.get_active_memory() / (1024 * 1024)
                    gpu_memory_available = gpu_memory_total - gpu_memory_used
                except Exception:
                    pass
            
            # System metrics
            cpu_usage = 0.0
            ram_usage = 0.0
            ram_available = 0.0
            disk_read = 0.0
            disk_write = 0.0
            network_io = 0.0
            
            if PSUTIL_AVAILABLE:
                cpu_usage = psutil.cpu_percent(interval=0.1)
                ram = psutil.virtual_memory()
                ram_usage = ram.used / (1024 * 1024)
                ram_available = ram.available / (1024 * 1024)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if hasattr(self, '_last_disk_io'):
                    time_delta = current_time - self._last_disk_time
                    if time_delta > 0:
                        disk_read = (disk_io.read_bytes - self._last_disk_io.read_bytes) / (1024 * 1024) / time_delta
                        disk_write = (disk_io.write_bytes - self._last_disk_io.write_bytes) / (1024 * 1024) / time_delta
                
                self._last_disk_io = disk_io
                self._last_disk_time = current_time
                
                # Network I/O
                net_io = psutil.net_io_counters()
                if hasattr(self, '_last_net_io'):
                    time_delta = current_time - self._last_net_time
                    if time_delta > 0:
                        bytes_sent = net_io.bytes_sent - self._last_net_io.bytes_sent
                        bytes_recv = net_io.bytes_recv - self._last_net_io.bytes_recv
                        network_io = (bytes_sent + bytes_recv) / (1024 * 1024) / time_delta
                
                self._last_net_io = net_io
                self._last_net_time = current_time
            
            return SystemMetrics(
                timestamp=current_time,
                total_gpu_memory_mb=gpu_memory_total,
                available_gpu_memory_mb=gpu_memory_available,
                gpu_temperature_celsius=None,  # Not easily available on M4 Pro
                cpu_usage_percent=cpu_usage,
                ram_usage_mb=ram_usage,
                ram_available_mb=ram_available,
                disk_io_read_mb_per_sec=disk_read,
                disk_io_write_mb_per_sec=disk_write,
                network_io_mb_per_sec=network_io
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return None
    
    async def record_operation_start(self, operation_name: str, operation_id: str = None):
        """Record the start of a GPU operation."""
        if operation_id is None:
            operation_id = f"{operation_name}_{int(time.time() * 1000000)}"
        
        self.operation_start_times[operation_id] = {
            'name': operation_name,
            'start_time': time.time(),
            'start_gpu_memory': self._get_gpu_memory_usage()
        }
        
        return operation_id
    
    async def record_operation_end(
        self, 
        operation_id: str,
        success: bool = True,
        error_message: str = None,
        batch_size: int = 1,
        data_size_mb: float = 0.0
    ):
        """Record the end of a GPU operation."""
        if operation_id not in self.operation_start_times:
            logger.warning(f"No start record found for operation {operation_id}")
            return
        
        start_info = self.operation_start_times.pop(operation_id)
        end_time = time.time()
        execution_time_ms = (end_time - start_info['start_time']) * 1000
        
        # Calculate throughput
        throughput = 0.0
        if execution_time_ms > 0:
            throughput = (batch_size / execution_time_ms) * 1000  # ops per second
        
        # Create performance metric
        metric = GPUPerformanceMetrics(
            timestamp=end_time,
            operation_name=start_info['name'],
            execution_time_ms=execution_time_ms,
            gpu_memory_mb=self._get_gpu_memory_usage(),
            gpu_utilization_percent=self._estimate_gpu_utilization(execution_time_ms),
            cpu_usage_percent=psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0,
            memory_usage_mb=psutil.virtual_memory().used / (1024 * 1024) if PSUTIL_AVAILABLE else 0.0,
            throughput_ops_per_sec=throughput,
            batch_size=batch_size,
            data_size_mb=data_size_mb,
            success=success,
            error_message=error_message
        )
        
        async with self.metrics_lock:
            self.operation_metrics.append(metric)
        
        # Analyze for alerts
        recent_metrics = list(self.operation_metrics)[-20:]
        alerts = self.analyzer.analyze_operation_performance(recent_metrics)
        for alert in alerts:
            await self._handle_alert(alert)
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if MLX_AVAILABLE:
            try:
                return mx.metal.get_active_memory() / (1024 * 1024)
            except Exception:
                pass
        return 0.0
    
    def _estimate_gpu_utilization(self, execution_time_ms: float) -> float:
        """Estimate GPU utilization based on execution time."""
        # Simple heuristic: longer operations generally indicate higher utilization
        if execution_time_ms < 10:
            return min(execution_time_ms * 10, 100)
        elif execution_time_ms < 100:
            return min(execution_time_ms, 100)
        else:
            return 95.0  # Assume high utilization for long operations
    
    async def _handle_alert(self, alert: PerformanceAlert):
        """Handle a performance alert."""
        async with self.metrics_lock:
            self.alerts.append(alert)
        
        # Log the alert
        log_func = {
            'info': logger.info,
            'warning': logger.warning,
            'error': logger.error,
            'critical': logger.critical
        }.get(alert.severity, logger.info)
        
        log_func(f"Performance Alert [{alert.severity.upper()}]: {alert.message}")
        
        # Log recommendations
        if alert.recommendations:
            logger.info(f"Recommendations: {'; '.join(alert.recommendations)}")
    
    async def get_performance_report(self, include_history: bool = False) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        async with self.metrics_lock:
            recent_operations = list(self.operation_metrics)[-100:]
            recent_system = list(self.system_metrics)[-20:]
            recent_alerts = list(self.alerts)[-50:]
        
        # Calculate summary statistics
        summary = {
            "monitoring_active": self.monitoring_active,
            "total_operations_recorded": len(self.operation_metrics),
            "system_info": self.system_info,
            "timestamp": time.time()
        }
        
        if recent_operations:
            avg_latency = sum(m.execution_time_ms for m in recent_operations) / len(recent_operations)
            avg_throughput = sum(m.throughput_ops_per_sec for m in recent_operations) / len(recent_operations)
            success_rate = sum(1 for m in recent_operations if m.success) / len(recent_operations)
            
            summary.update({
                "average_latency_ms": avg_latency,
                "average_throughput_ops_per_sec": avg_throughput,
                "success_rate": success_rate,
                "operations_analyzed": len(recent_operations)
            })
        
        if recent_system:
            latest_system = recent_system[-1]
            summary.update({
                "current_gpu_memory_mb": latest_system.total_gpu_memory_mb - latest_system.available_gpu_memory_mb,
                "current_cpu_usage_percent": latest_system.cpu_usage_percent,
                "current_ram_usage_mb": latest_system.ram_usage_mb
            })
        
        # Generate recommendations
        recommendations = self.analyzer.generate_optimization_recommendations(recent_operations, recent_system)
        
        report = {
            "summary": summary,
            "recommendations": recommendations,
            "alert_count": len(recent_alerts),
            "critical_alerts": len([a for a in recent_alerts if a.severity == 'critical']),
            "recent_alerts": [asdict(alert) for alert in recent_alerts[-10:]]
        }
        
        if include_history:
            report.update({
                "operation_history": [asdict(m) for m in recent_operations],
                "system_history": [asdict(m) for m in recent_system]
            })
        
        return report
    
    async def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        async with self.metrics_lock:
            data = {
                "system_info": self.system_info,
                "export_timestamp": time.time(),
                "operation_metrics": [asdict(m) for m in self.operation_metrics],
                "system_metrics": [asdict(m) for m in self.system_metrics],
                "alerts": [asdict(a) for a in self.alerts]
            }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported performance metrics to {filepath}")


# Global monitor instance
_performance_monitor: Optional[GPUPerformanceMonitor] = None


def get_performance_monitor() -> GPUPerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = GPUPerformanceMonitor()
    return _performance_monitor


# Context manager for operation monitoring
class gpu_operation_monitor:
    """Context manager for monitoring GPU operations."""
    
    def __init__(self, operation_name: str, batch_size: int = 1, data_size_mb: float = 0.0):
        self.operation_name = operation_name
        self.batch_size = batch_size
        self.data_size_mb = data_size_mb
        self.operation_id = None
        self.monitor = get_performance_monitor()
    
    async def __aenter__(self):
        self.operation_id = await self.monitor.record_operation_start(self.operation_name)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        error_message = str(exc_val) if exc_val else None
        
        await self.monitor.record_operation_end(
            self.operation_id,
            success=success,
            error_message=error_message,
            batch_size=self.batch_size,
            data_size_mb=self.data_size_mb
        )


if __name__ == "__main__":
    # Test the performance monitoring system
    async def test_performance_monitoring():
        print("Testing GPU Performance Monitoring System")
        print("=" * 50)
        
        monitor = get_performance_monitor()
        
        # Start monitoring
        await monitor.start_monitoring(interval_seconds=1.0)
        
        # Simulate some operations
        for i in range(5):
            async with gpu_operation_monitor(f"test_operation_{i}", batch_size=32):
                await asyncio.sleep(0.1)  # Simulate work
        
        # Get performance report
        report = await monitor.get_performance_report()
        print(f"Operations recorded: {report['summary']['total_operations_recorded']}")
        print(f"Average latency: {report['summary'].get('average_latency_ms', 0):.1f}ms")
        print(f"Success rate: {report['summary'].get('success_rate', 0):.1%}")
        
        # Export metrics
        await monitor.export_metrics("performance_test.json")
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        print("Performance monitoring test completed")
    
    asyncio.run(test_performance_monitoring())