"""
Parallel Processing Performance Monitor
Real-time monitoring and reporting of parallel processing optimization achievements.
"""

import asyncio
import json
import logging
import psutil
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import weakref

from ..config.hardware_config import get_hardware_config

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    
    timestamp: float
    metric_name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SpeedupMeasurement:
    """Speedup measurement for a specific operation."""
    
    operation_name: str
    serial_time: float
    parallel_time: float
    speedup: float
    efficiency: float
    target_met: bool
    timestamp: float = field(default_factory=time.time)

class ParallelPerformanceMonitor:
    """
    Real-time performance monitor for parallel processing optimizations.
    
    Features:
    - Continuous performance tracking
    - Speedup measurement and reporting
    - Resource utilization monitoring
    - Performance regression detection
    - Optimization recommendations
    """
    
    def __init__(self, history_size: int = 1000):
        self.hw_config = get_hardware_config()
        self.history_size = history_size
        
        # Performance metrics storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.speedup_measurements: deque = deque(maxlen=history_size)
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = 1.0  # seconds
        
        # Performance targets
        self.targets = {
            'cpu_intensive_speedup': 4.0,
            'memory_intensive_speedup': 2.5,
            'io_bound_speedup': 10.0,
            'mixed_workload_speedup': 4.0,
            'overall_speedup': 4.0,
            'cpu_utilization': 0.85,
            'memory_efficiency': 0.90,
            'cache_hit_rate': 0.85
        }
        
        # Current state tracking
        self.current_operations = {}
        self.resource_stats = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("📊 Parallel Performance Monitor initialized")
    
    def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ParallelPerfMonitor"
        )
        self.monitor_thread.start()
        
        logger.info("🔍 Started real-time performance monitoring")
    
    def stop_monitoring(self):
        """Stop real-time performance monitoring."""
        self.monitoring_active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("⏹️ Stopped performance monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}", exc_info=True)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        timestamp = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Store metrics
        with self._lock:
            self.metrics_history['cpu_total'].append(
                PerformanceMetric(timestamp, 'cpu_total', cpu_percent)
            )
            
            self.metrics_history['memory_usage'].append(
                PerformanceMetric(timestamp, 'memory_usage', memory.percent)
            )
            
            # Per-core CPU usage
            if len(cpu_per_core) >= self.hw_config.cpu_cores:
                # P-cores average
                p_core_usage = sum(cpu_per_core[:self.hw_config.cpu_performance_cores]) / self.hw_config.cpu_performance_cores
                self.metrics_history['p_core_usage'].append(
                    PerformanceMetric(timestamp, 'p_core_usage', p_core_usage)
                )
                
                # E-cores average (if present)
                if self.hw_config.cpu_efficiency_cores > 0:
                    e_core_usage = sum(cpu_per_core[self.hw_config.cpu_performance_cores:]) / self.hw_config.cpu_efficiency_cores
                    self.metrics_history['e_core_usage'].append(
                        PerformanceMetric(timestamp, 'e_core_usage', e_core_usage)
                    )
    
    def record_speedup(self, 
                      operation_name: str,
                      serial_time: float,
                      parallel_time: float,
                      metadata: Optional[Dict[str, Any]] = None) -> SpeedupMeasurement:
        """
        Record a speedup measurement.
        
        Args:
            operation_name: Name of the operation
            serial_time: Time for serial execution
            parallel_time: Time for parallel execution
            metadata: Additional metadata
            
        Returns:
            SpeedupMeasurement object
        """
        if parallel_time <= 0:
            logger.warning(f"Invalid parallel time for {operation_name}: {parallel_time}")
            parallel_time = 0.001  # Avoid division by zero
        
        speedup = serial_time / parallel_time
        efficiency = speedup / self.hw_config.cpu_cores
        
        # Determine target based on operation type
        target = self._get_target_for_operation(operation_name)
        target_met = speedup >= target
        
        measurement = SpeedupMeasurement(
            operation_name=operation_name,
            serial_time=serial_time,
            parallel_time=parallel_time,
            speedup=speedup,
            efficiency=efficiency,
            target_met=target_met
        )
        
        with self._lock:
            self.speedup_measurements.append(measurement)
        
        # Log significant measurements
        if speedup >= target:
            logger.info(f"🎯 {operation_name}: {speedup:.2f}x speedup (target: {target:.1f}x) ✅")
        else:
            logger.warning(f"⚠️ {operation_name}: {speedup:.2f}x speedup (target: {target:.1f}x) ❌")
        
        return measurement
    
    def _get_target_for_operation(self, operation_name: str) -> float:
        """Get performance target for an operation type."""
        operation_lower = operation_name.lower()
        
        if 'cpu' in operation_lower or 'compute' in operation_lower:
            return self.targets['cpu_intensive_speedup']
        elif 'memory' in operation_lower or 'mem' in operation_lower:
            return self.targets['memory_intensive_speedup']
        elif 'io' in operation_lower or 'file' in operation_lower:
            return self.targets['io_bound_speedup']
        elif 'mixed' in operation_lower:
            return self.targets['mixed_workload_speedup']
        else:
            return self.targets['overall_speedup']
    
    def get_performance_summary(self, 
                              window_minutes: int = 10) -> Dict[str, Any]:
        """
        Get performance summary for the specified time window.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Performance summary dictionary
        """
        window_start = time.time() - (window_minutes * 60)
        
        with self._lock:
            # Filter measurements within window
            recent_measurements = [
                m for m in self.speedup_measurements
                if m.timestamp >= window_start
            ]
            
            if not recent_measurements:
                return {
                    'window_minutes': window_minutes,
                    'measurements': 0,
                    'message': 'No measurements in time window'
                }
            
            # Calculate statistics
            speedups = [m.speedup for m in recent_measurements]
            efficiencies = [m.efficiency for m in recent_measurements]
            targets_met = sum(1 for m in recent_measurements if m.target_met)
            
            # Group by operation type
            by_operation = defaultdict(list)
            for m in recent_measurements:
                by_operation[m.operation_name].append(m)
            
            operation_stats = {}
            for op_name, measurements in by_operation.items():
                op_speedups = [m.speedup for m in measurements]
                operation_stats[op_name] = {
                    'count': len(measurements),
                    'avg_speedup': statistics.mean(op_speedups),
                    'max_speedup': max(op_speedups),
                    'min_speedup': min(op_speedups),
                    'targets_met': sum(1 for m in measurements if m.target_met),
                    'success_rate': sum(1 for m in measurements if m.target_met) / len(measurements) * 100
                }
            
            summary = {
                'window_minutes': window_minutes,
                'measurements': len(recent_measurements),
                'overall_stats': {
                    'avg_speedup': statistics.mean(speedups),
                    'max_speedup': max(speedups),
                    'min_speedup': min(speedups),
                    'avg_efficiency': statistics.mean(efficiencies),
                    'targets_met': targets_met,
                    'success_rate': (targets_met / len(recent_measurements)) * 100
                },
                'operation_stats': operation_stats,
                'target_achievement': {
                    '4x_speedup_achieved': statistics.mean(speedups) >= 4.0,
                    'high_efficiency': statistics.mean(efficiencies) >= 0.75,
                    'consistent_performance': (max(speedups) - min(speedups)) < 2.0
                }
            }
            
            return summary
    
    def get_resource_utilization_report(self) -> Dict[str, Any]:
        """Get current resource utilization report."""
        with self._lock:
            recent_cpu = list(self.metrics_history['cpu_total'])[-10:]  # Last 10 measurements
            recent_memory = list(self.metrics_history['memory_usage'])[-10:]
            
            if not recent_cpu or not recent_memory:
                return {'error': 'Insufficient monitoring data'}
            
            cpu_values = [m.value for m in recent_cpu]
            memory_values = [m.value for m in recent_memory]
            
            report = {
                'cpu': {
                    'current_usage': cpu_values[-1] if cpu_values else 0,
                    'avg_usage': statistics.mean(cpu_values),
                    'max_usage': max(cpu_values),
                    'target_met': statistics.mean(cpu_values) >= self.targets['cpu_utilization'] * 100
                },
                'memory': {
                    'current_usage': memory_values[-1] if memory_values else 0,
                    'avg_usage': statistics.mean(memory_values),
                    'max_usage': max(memory_values),
                    'efficiency_score': (100 - statistics.mean(memory_values)) / 100  # Lower usage = higher efficiency
                }
            }
            
            # P-core vs E-core utilization
            if 'p_core_usage' in self.metrics_history and 'e_core_usage' in self.metrics_history:
                p_core_recent = list(self.metrics_history['p_core_usage'])[-10:]
                e_core_recent = list(self.metrics_history['e_core_usage'])[-10:]
                
                if p_core_recent and e_core_recent:
                    p_core_avg = statistics.mean([m.value for m in p_core_recent])
                    e_core_avg = statistics.mean([m.value for m in e_core_recent])
                    
                    report['core_distribution'] = {
                        'p_core_utilization': p_core_avg,
                        'e_core_utilization': e_core_avg,
                        'load_balance_ratio': p_core_avg / max(e_core_avg, 1.0),
                        'optimal_distribution': abs(p_core_avg - e_core_avg) < 20  # Within 20% difference
                    }
            
            return report
    
    def detect_performance_regressions(self, 
                                     comparison_window_minutes: int = 30) -> List[Dict[str, Any]]:
        """
        Detect performance regressions by comparing recent vs historical performance.
        
        Args:
            comparison_window_minutes: Minutes to compare against
            
        Returns:
            List of detected regressions
        """
        current_time = time.time()
        comparison_cutoff = current_time - (comparison_window_minutes * 60)
        recent_cutoff = current_time - (comparison_window_minutes * 60 / 3)  # Last 1/3 of window
        
        with self._lock:
            # Split measurements into historical and recent
            historical = [m for m in self.speedup_measurements 
                         if m.timestamp < comparison_cutoff]
            recent = [m for m in self.speedup_measurements 
                     if m.timestamp >= recent_cutoff]
            
            if len(historical) < 5 or len(recent) < 3:
                return []  # Insufficient data
            
            regressions = []
            
            # Group by operation type
            historical_by_op = defaultdict(list)
            recent_by_op = defaultdict(list)
            
            for m in historical:
                historical_by_op[m.operation_name].append(m.speedup)
            
            for m in recent:
                recent_by_op[m.operation_name].append(m.speedup)
            
            # Compare performance for each operation type
            for op_name in recent_by_op.keys():
                if op_name in historical_by_op:
                    hist_avg = statistics.mean(historical_by_op[op_name])
                    recent_avg = statistics.mean(recent_by_op[op_name])
                    
                    # Detect significant regression (>10% degradation)
                    degradation = (hist_avg - recent_avg) / hist_avg
                    if degradation > 0.1:  # 10% regression threshold
                        regressions.append({
                            'operation': op_name,
                            'historical_avg_speedup': hist_avg,
                            'recent_avg_speedup': recent_avg,
                            'degradation_percent': degradation * 100,
                            'severity': 'high' if degradation > 0.25 else 'medium'
                        })
            
            return regressions
    
    def generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on current performance."""
        recommendations = []
        
        # Get current performance summary
        summary = self.get_performance_summary()
        if 'overall_stats' not in summary:
            return recommendations
        
        overall_stats = summary['overall_stats']
        
        # CPU utilization recommendations
        utilization_report = self.get_resource_utilization_report()
        if 'cpu' in utilization_report:
            cpu_usage = utilization_report['cpu']['avg_usage']
            if cpu_usage < 60:
                recommendations.append({
                    'category': 'CPU Utilization',
                    'priority': 'high',
                    'issue': f'Low CPU utilization ({cpu_usage:.1f}%)',
                    'recommendation': 'Increase parallelism or reduce I/O wait times',
                    'expected_impact': 'Up to 30% speedup improvement'
                })
            elif cpu_usage > 95:
                recommendations.append({
                    'category': 'CPU Utilization', 
                    'priority': 'medium',
                    'issue': f'CPU over-utilization ({cpu_usage:.1f}%)',
                    'recommendation': 'Reduce thread contention or optimize task scheduling',
                    'expected_impact': 'Improved stability and efficiency'
                })
        
        # Speedup recommendations
        avg_speedup = overall_stats.get('avg_speedup', 0)
        if avg_speedup < 3.0:
            recommendations.append({
                'category': 'Parallel Efficiency',
                'priority': 'high',
                'issue': f'Low average speedup ({avg_speedup:.2f}x)',
                'recommendation': 'Review task decomposition and reduce synchronization overhead',
                'expected_impact': 'Achieve 4.0x target speedup'
            })
        
        # Core distribution recommendations
        if 'core_distribution' in utilization_report:
            core_dist = utilization_report['core_distribution']
            if not core_dist.get('optimal_distribution', True):
                recommendations.append({
                    'category': 'Load Balancing',
                    'priority': 'medium',
                    'issue': 'Uneven P-core/E-core distribution',
                    'recommendation': 'Implement better task-to-core affinity mapping',
                    'expected_impact': '10-15% efficiency improvement'
                })
        
        # Performance consistency recommendations
        if overall_stats.get('max_speedup', 0) - overall_stats.get('min_speedup', 0) > 3.0:
            recommendations.append({
                'category': 'Performance Consistency',
                'priority': 'low',
                'issue': 'High performance variance across operations',
                'recommendation': 'Standardize optimization techniques across operation types',
                'expected_impact': 'More predictable performance'
            })
        
        return recommendations
    
    def export_performance_data(self, 
                              filename: str = "parallel_performance_data.json") -> str:
        """Export performance data to JSON file."""
        with self._lock:
            export_data = {
                'export_timestamp': time.time(),
                'hardware_config': {
                    'cpu_cores': self.hw_config.cpu_cores,
                    'p_cores': self.hw_config.cpu_performance_cores,
                    'e_cores': self.hw_config.cpu_efficiency_cores,
                    'memory_gb': self.hw_config.memory_total_gb,
                    'model': self.hw_config.model_name
                },
                'performance_targets': self.targets,
                'speedup_measurements': [
                    {
                        'operation': m.operation_name,
                        'speedup': m.speedup,
                        'efficiency': m.efficiency,
                        'target_met': m.target_met,
                        'timestamp': m.timestamp
                    }
                    for m in self.speedup_measurements
                ],
                'performance_summary': self.get_performance_summary(),
                'resource_utilization': self.get_resource_utilization_report(),
                'recommendations': self.generate_optimization_recommendations()
            }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"📊 Performance data exported to {filename}")
        return filename
    
    def print_real_time_dashboard(self):
        """Print a real-time performance dashboard."""
        summary = self.get_performance_summary(window_minutes=5)
        utilization = self.get_resource_utilization_report()
        
        print("\n" + "="*60)
        print("🚀 M4 Pro Parallel Processing Performance Dashboard")
        print("="*60)
        
        if 'overall_stats' in summary:
            stats = summary['overall_stats']
            print(f"📊 Performance (Last 5 minutes):")
            print(f"   Average Speedup: {stats['avg_speedup']:.2f}x")
            print(f"   Max Speedup: {stats['max_speedup']:.2f}x")
            print(f"   Success Rate: {stats['success_rate']:.1f}%")
            print(f"   Measurements: {summary['measurements']}")
            
            # Target achievement
            target_achieved = stats['avg_speedup'] >= 4.0
            print(f"   4.0x Target: {'✅ ACHIEVED' if target_achieved else '❌ NOT MET'}")
        
        if 'cpu' in utilization:
            print(f"\n💻 Resource Utilization:")
            print(f"   CPU Usage: {utilization['cpu']['current_usage']:.1f}%")
            print(f"   Memory Usage: {utilization['memory']['current_usage']:.1f}%")
            
            if 'core_distribution' in utilization:
                core_dist = utilization['core_distribution']
                print(f"   P-Core Usage: {core_dist['p_core_utilization']:.1f}%")
                print(f"   E-Core Usage: {core_dist['e_core_utilization']:.1f}%")
        
        print("="*60)


# Global monitor instance
_monitor_instance: Optional[ParallelPerformanceMonitor] = None
_monitor_lock = threading.Lock()

def get_performance_monitor() -> ParallelPerformanceMonitor:
    """Get or create the global performance monitor."""
    global _monitor_instance
    
    if _monitor_instance is None:
        with _monitor_lock:
            if _monitor_instance is None:
                _monitor_instance = ParallelPerformanceMonitor()
    
    return _monitor_instance

# Context manager for automatic speedup measurement
class measure_speedup:
    """Context manager for automatic speedup measurement."""
    
    def __init__(self, operation_name: str, enable_serial: bool = False):
        self.operation_name = operation_name
        self.enable_serial = enable_serial
        self.monitor = get_performance_monitor()
        self.start_time = None
        self.serial_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            parallel_time = time.perf_counter() - self.start_time
            
            # For automatic measurement, estimate serial time as parallel_time * cores
            if self.serial_time is None:
                hw_config = get_hardware_config()
                estimated_serial_time = parallel_time * hw_config.cpu_cores
                self.monitor.record_speedup(
                    self.operation_name,
                    estimated_serial_time,
                    parallel_time
                )
    
    def record_serial_time(self, serial_time: float):
        """Record actual serial execution time."""
        self.serial_time = serial_time