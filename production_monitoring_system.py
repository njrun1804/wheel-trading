#!/usr/bin/env python3
"""
Production Monitoring System for Wheel Trading Platform
Provides comprehensive 24-hour post-deployment monitoring and validation.
"""

import asyncio
import json
import platform
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from src.unity_wheel.monitoring.performance import get_performance_monitor
from src.unity_wheel.utils.logging import get_logger

logger = get_logger(__name__)


class ProductionMonitor:
    """Comprehensive production monitoring system."""
    
    def __init__(self):
        """Initialize production monitor."""
        self.start_time = datetime.now()
        self.performance_monitor = get_performance_monitor()
        self.alerts = deque(maxlen=1000)
        self.metrics_history = defaultdict(deque)
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'response_time_ms': 500.0,
            'error_rate': 0.05,  # 5%
            'trading_latency_ms': 100.0,
            'data_latency_ms': 50.0
        }
        self.is_monitoring = False
        
    async def start_monitoring(self):
        """Start comprehensive monitoring."""
        logger.info("Starting production monitoring system...")
        self.is_monitoring = True
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitor_system_resources()),
            asyncio.create_task(self._monitor_trading_performance()),
            asyncio.create_task(self._monitor_data_quality()),
            asyncio.create_task(self._monitor_error_rates()),
            asyncio.create_task(self._monitor_hardware_optimization()),
            asyncio.create_task(self._generate_periodic_reports()),
        ]
        
        logger.info("Production monitoring active - tracking all KPIs")
        await asyncio.gather(*tasks)
        
    async def stop_monitoring(self):
        """Stop monitoring and generate final report."""
        self.is_monitoring = False
        logger.info("Stopping production monitoring...")
        
        # Generate final report
        final_report = await self._generate_comprehensive_report()
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"production_monitoring_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
            
        logger.info(f"Final production report saved to {report_file}")
        return final_report
        
    async def _monitor_system_resources(self):
        """Monitor CPU, memory, disk usage."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                metrics = {
                    'timestamp': datetime.now(),
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_usage': disk.percent,
                    'disk_free_gb': disk.free / (1024**3),
                    'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
                }
                
                # Store metrics
                self.metrics_history['system_resources'].append(metrics)
                
                # Check thresholds
                if cpu_percent > self.thresholds['cpu_usage']:
                    await self._alert(f"High CPU usage: {cpu_percent:.1f}%", 'warning')
                    
                if memory.percent > self.thresholds['memory_usage']:
                    await self._alert(f"High memory usage: {memory.percent:.1f}%", 'warning')
                    
                if disk.percent > self.thresholds['disk_usage']:
                    await self._alert(f"High disk usage: {disk.percent:.1f}%", 'critical')
                    
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring system resources: {e}")
                await asyncio.sleep(60)
                
    async def _monitor_trading_performance(self):
        """Monitor trading system performance."""
        while self.is_monitoring:
            try:
                # Get trading-specific performance stats
                trading_stats = self.performance_monitor.get_stats('advise_position', 5)
                decision_stats = self.performance_monitor.get_stats('decision_engine', 5)
                
                if trading_stats:
                    metrics = {
                        'timestamp': datetime.now(),
                        'avg_response_time': trading_stats.avg_duration_ms,
                        'p95_response_time': trading_stats.p95_duration_ms,
                        'success_rate': trading_stats.success_rate,
                        'operation_count': trading_stats.count
                    }
                    
                    self.metrics_history['trading_performance'].append(metrics)
                    
                    # Check thresholds
                    if trading_stats.p95_duration_ms > self.thresholds['response_time_ms']:
                        await self._alert(
                            f"Slow trading response: {trading_stats.p95_duration_ms:.1f}ms", 
                            'warning'
                        )
                        
                    if trading_stats.success_rate < (1 - self.thresholds['error_rate']):
                        await self._alert(
                            f"High trading error rate: {(1-trading_stats.success_rate)*100:.1f}%", 
                            'critical'
                        )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring trading performance: {e}")
                await asyncio.sleep(60)
                
    async def _monitor_data_quality(self):
        """Monitor data ingestion and quality."""
        while self.is_monitoring:
            try:
                # Check data freshness and quality
                from src.unity_wheel.storage.storage import get_storage
                storage = get_storage()
                
                # Check latest data timestamps
                latest_prices = await storage.get_latest_price_timestamp()
                latest_options = await storage.get_latest_options_timestamp()
                
                now = datetime.now()
                price_age = (now - latest_prices).total_seconds() if latest_prices else float('inf')
                options_age = (now - latest_options).total_seconds() if latest_options else float('inf')
                
                metrics = {
                    'timestamp': now,
                    'price_data_age_seconds': price_age,
                    'options_data_age_seconds': options_age,
                    'data_quality_score': self._calculate_data_quality_score(price_age, options_age)
                }
                
                self.metrics_history['data_quality'].append(metrics)
                
                # Check data freshness
                if price_age > 300:  # 5 minutes
                    await self._alert(f"Stale price data: {price_age/60:.1f} minutes old", 'warning')
                    
                if options_age > 900:  # 15 minutes
                    await self._alert(f"Stale options data: {options_age/60:.1f} minutes old", 'warning')
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring data quality: {e}")
                await asyncio.sleep(300)
                
    async def _monitor_error_rates(self):
        """Monitor error rates across all operations."""
        while self.is_monitoring:
            try:
                all_stats = self.performance_monitor.get_all_stats(15)  # Last 15 minutes
                
                error_metrics = {}
                for operation, stats in all_stats.items():
                    error_rate = 1 - stats.success_rate
                    error_metrics[operation] = {
                        'error_rate': error_rate,
                        'operation_count': stats.count,
                        'avg_duration': stats.avg_duration_ms
                    }
                    
                    # Alert on high error rates
                    if error_rate > self.thresholds['error_rate'] and stats.count > 5:
                        await self._alert(
                            f"High error rate in {operation}: {error_rate*100:.1f}%", 
                            'critical'
                        )
                
                self.metrics_history['error_rates'].append({
                    'timestamp': datetime.now(),
                    'operations': error_metrics
                })
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring error rates: {e}")
                await asyncio.sleep(300)
                
    async def _monitor_hardware_optimization(self):
        """Monitor M4 Pro hardware optimization effectiveness."""
        while self.is_monitoring:
            try:
                # Check CPU core utilization
                cpu_percent_per_core = psutil.cpu_percent(percpu=True)
                
                # Check if we're effectively using M4 Pro cores
                p_cores = cpu_percent_per_core[:8]  # Performance cores
                e_cores = cpu_percent_per_core[8:12] if len(cpu_percent_per_core) > 8 else []
                
                metrics = {
                    'timestamp': datetime.now(),
                    'p_core_avg_usage': sum(p_cores) / len(p_cores) if p_cores else 0,
                    'e_core_avg_usage': sum(e_cores) / len(e_cores) if e_cores else 0,
                    'total_cores_active': sum(1 for usage in cpu_percent_per_core if usage > 5),
                    'parallel_efficiency': self._calculate_parallel_efficiency(),
                    'memory_bandwidth_usage': self._estimate_memory_bandwidth()
                }
                
                self.metrics_history['hardware_optimization'].append(metrics)
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring hardware optimization: {e}")
                await asyncio.sleep(120)
                
    async def _generate_periodic_reports(self):
        """Generate periodic status reports."""
        while self.is_monitoring:
            try:
                # Generate hourly reports
                await asyncio.sleep(3600)  # 1 hour
                
                report = await self._generate_status_report()
                
                # Log key metrics
                logger.info("=== HOURLY PRODUCTION STATUS ===")
                logger.info(f"Uptime: {report['uptime_hours']:.1f} hours")
                logger.info(f"System Health: {report['overall_health']}")
                logger.info(f"Trading Performance: {report['trading_summary']['avg_response_time']:.1f}ms avg")
                logger.info(f"Alerts Generated: {len(self.alerts)} total")
                
                # Save report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                report_file = f"hourly_status_{timestamp}.json"
                
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                    
            except Exception as e:
                logger.error(f"Error generating periodic report: {e}")
                
    async def _alert(self, message: str, severity: str = 'info'):
        """Generate alert."""
        alert = {
            'timestamp': datetime.now(),
            'message': message,
            'severity': severity
        }
        
        self.alerts.append(alert)
        
        # Log based on severity
        if severity == 'critical':
            logger.critical(f"ALERT: {message}")
        elif severity == 'warning':
            logger.warning(f"ALERT: {message}")
        else:
            logger.info(f"ALERT: {message}")
            
    def _calculate_data_quality_score(self, price_age: float, options_age: float) -> float:
        """Calculate data quality score (0-1)."""
        price_score = max(0, 1 - price_age / 300)  # Degrade over 5 minutes
        options_score = max(0, 1 - options_age / 900)  # Degrade over 15 minutes
        return (price_score + options_score) / 2
        
    def _calculate_parallel_efficiency(self) -> float:
        """Calculate parallel processing efficiency."""
        # Simple metric based on core utilization distribution
        try:
            cpu_percents = psutil.cpu_percent(percpu=True)
            if not cpu_percents:
                return 0.0
                
            # Higher efficiency when load is distributed
            variance = sum((x - sum(cpu_percents)/len(cpu_percents))**2 for x in cpu_percents) / len(cpu_percents)
            efficiency = max(0, 1 - variance / 1000)  # Normalized variance penalty
            return min(1.0, efficiency)
            
        except Exception:
            return 0.0
            
    def _estimate_memory_bandwidth(self) -> float:
        """Estimate memory bandwidth utilization."""
        try:
            memory = psutil.virtual_memory()
            # Simple heuristic based on memory activity
            # Real implementation would need platform-specific tools
            return min(1.0, memory.percent / 100 * 0.8)  # Conservative estimate
            
        except Exception:
            return 0.0
            
    async def _generate_status_report(self) -> Dict[str, Any]:
        """Generate current status report."""
        uptime = datetime.now() - self.start_time
        
        # Get latest metrics
        latest_system = self.metrics_history['system_resources'][-1] if self.metrics_history['system_resources'] else {}
        latest_trading = self.metrics_history['trading_performance'][-1] if self.metrics_history['trading_performance'] else {}
        latest_data = self.metrics_history['data_quality'][-1] if self.metrics_history['data_quality'] else {}
        
        # Calculate overall health
        health_factors = []
        if latest_system:
            health_factors.extend([
                1.0 - latest_system['cpu_usage'] / 100,
                1.0 - latest_system['memory_usage'] / 100,
                1.0 - latest_system['disk_usage'] / 100
            ])
        
        if latest_trading:
            health_factors.append(latest_trading['success_rate'])
            
        if latest_data:
            health_factors.append(latest_data['data_quality_score'])
            
        overall_health = sum(health_factors) / len(health_factors) if health_factors else 0.5
        
        return {
            'timestamp': datetime.now(),
            'uptime_hours': uptime.total_seconds() / 3600,
            'overall_health': overall_health,
            'system_status': latest_system,
            'trading_summary': latest_trading,
            'data_quality': latest_data,
            'alert_count': len(self.alerts),
            'recent_alerts': list(self.alerts)[-5:],  # Last 5 alerts
            'performance_stats': self.performance_monitor.get_all_stats(60)
        }
        
    async def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        uptime = datetime.now() - self.start_time
        
        # Aggregate all collected metrics
        report = {
            'monitoring_session': {
                'start_time': self.start_time,
                'end_time': datetime.now(),
                'duration_hours': uptime.total_seconds() / 3600,
                'platform': platform.platform(),
                'python_version': platform.python_version()
            },
            'system_performance': self._analyze_system_performance(),
            'trading_performance': self._analyze_trading_performance(),
            'data_quality_analysis': self._analyze_data_quality(),
            'hardware_optimization': self._analyze_hardware_optimization(),
            'alerts_summary': self._analyze_alerts(),
            'recommendations': self._generate_recommendations(),
            'kpi_summary': self._generate_kpi_summary()
        }
        
        return report
        
    def _analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze system performance over monitoring period."""
        if not self.metrics_history['system_resources']:
            return {'status': 'no_data'}
            
        metrics = list(self.metrics_history['system_resources'])
        
        cpu_values = [m['cpu_usage'] for m in metrics]
        memory_values = [m['memory_usage'] for m in metrics]
        
        return {
            'cpu_usage': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'threshold_violations': sum(1 for v in cpu_values if v > self.thresholds['cpu_usage'])
            },
            'memory_usage': {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'threshold_violations': sum(1 for v in memory_values if v > self.thresholds['memory_usage'])
            },
            'stability_score': self._calculate_stability_score(cpu_values, memory_values)
        }
        
    def _analyze_trading_performance(self) -> Dict[str, Any]:
        """Analyze trading performance over monitoring period."""
        if not self.metrics_history['trading_performance']:
            return {'status': 'no_data'}
            
        metrics = list(self.metrics_history['trading_performance'])
        
        response_times = [m['avg_response_time'] for m in metrics]
        success_rates = [m['success_rate'] for m in metrics]
        
        return {
            'response_time': {
                'avg': sum(response_times) / len(response_times),
                'max': max(response_times),
                'min': min(response_times)
            },
            'success_rate': {
                'avg': sum(success_rates) / len(success_rates),
                'min': min(success_rates)
            },
            'total_operations': sum(m['operation_count'] for m in metrics),
            'performance_score': sum(success_rates) / len(success_rates) if success_rates else 0
        }
        
    def _analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze data quality over monitoring period."""
        if not self.metrics_history['data_quality']:
            return {'status': 'no_data'}
            
        metrics = list(self.metrics_history['data_quality'])
        quality_scores = [m['data_quality_score'] for m in metrics]
        
        return {
            'average_quality_score': sum(quality_scores) / len(quality_scores),
            'min_quality_score': min(quality_scores),
            'data_freshness_issues': sum(1 for m in metrics 
                                       if m['price_data_age_seconds'] > 300 or 
                                          m['options_data_age_seconds'] > 900)
        }
        
    def _analyze_hardware_optimization(self) -> Dict[str, Any]:
        """Analyze hardware optimization effectiveness."""
        if not self.metrics_history['hardware_optimization']:
            return {'status': 'no_data'}
            
        metrics = list(self.metrics_history['hardware_optimization'])
        
        p_core_usage = [m['p_core_avg_usage'] for m in metrics]
        efficiency_scores = [m['parallel_efficiency'] for m in metrics]
        
        return {
            'p_core_utilization': {
                'avg': sum(p_core_usage) / len(p_core_usage),
                'max': max(p_core_usage)
            },
            'parallel_efficiency': {
                'avg': sum(efficiency_scores) / len(efficiency_scores),
                'min': min(efficiency_scores)
            },
            'optimization_effectiveness': 'excellent' if sum(efficiency_scores) / len(efficiency_scores) > 0.8 else 'good'
        }
        
    def _analyze_alerts(self) -> Dict[str, Any]:
        """Analyze alerts generated during monitoring."""
        if not self.alerts:
            return {'total': 0, 'by_severity': {}}
            
        by_severity = defaultdict(int)
        for alert in self.alerts:
            by_severity[alert['severity']] += 1
            
        return {
            'total': len(self.alerts),
            'by_severity': dict(by_severity),
            'alerts_per_hour': len(self.alerts) / ((datetime.now() - self.start_time).total_seconds() / 3600)
        }
        
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Analyze patterns and suggest improvements
        if self.metrics_history['system_resources']:
            avg_cpu = sum(m['cpu_usage'] for m in self.metrics_history['system_resources']) / len(self.metrics_history['system_resources'])
            if avg_cpu > 70:
                recommendations.append("Consider optimizing CPU-intensive operations or scaling resources")
                
        if self.metrics_history['trading_performance']:
            avg_response = sum(m['avg_response_time'] for m in self.metrics_history['trading_performance']) / len(self.metrics_history['trading_performance'])
            if avg_response > 200:
                recommendations.append("Trading response times could be optimized")
                
        # Hardware-specific recommendations
        if self.metrics_history['hardware_optimization']:
            avg_efficiency = sum(m['parallel_efficiency'] for m in self.metrics_history['hardware_optimization']) / len(self.metrics_history['hardware_optimization'])
            if avg_efficiency < 0.7:
                recommendations.append("Parallel processing efficiency could be improved")
                
        if not recommendations:
            recommendations.append("System performing optimally - no major recommendations")
            
        return recommendations
        
    def _generate_kpi_summary(self) -> Dict[str, Any]:
        """Generate KPI summary against targets."""
        return {
            'availability': self._calculate_availability(),
            'performance_sla': self._calculate_performance_sla(),
            'data_quality_sla': self._calculate_data_quality_sla(),
            'error_rate_sla': self._calculate_error_rate_sla()
        }
        
    def _calculate_stability_score(self, cpu_values: List[float], memory_values: List[float]) -> float:
        """Calculate system stability score."""
        cpu_variance = sum((x - sum(cpu_values)/len(cpu_values))**2 for x in cpu_values) / len(cpu_values)
        memory_variance = sum((x - sum(memory_values)/len(memory_values))**2 for x in memory_values) / len(memory_values)
        
        # Lower variance = higher stability
        cpu_stability = max(0, 1 - cpu_variance / 100)
        memory_stability = max(0, 1 - memory_variance / 100)
        
        return (cpu_stability + memory_stability) / 2
        
    def _calculate_availability(self) -> float:
        """Calculate system availability percentage."""
        # Based on successful operations vs total operations
        all_stats = self.performance_monitor.get_all_stats(1440)  # Last 24 hours
        if not all_stats:
            return 1.0
            
        total_ops = sum(stats.count for stats in all_stats.values())
        successful_ops = sum(stats.count * stats.success_rate for stats in all_stats.values())
        
        return successful_ops / total_ops if total_ops > 0 else 1.0
        
    def _calculate_performance_sla(self) -> float:
        """Calculate performance SLA compliance."""
        trading_stats = self.performance_monitor.get_stats('advise_position', 1440)
        if not trading_stats:
            return 1.0
            
        # SLA: 95% of requests under 500ms
        return 1.0 if trading_stats.p95_duration_ms <= 500 else 0.8
        
    def _calculate_data_quality_sla(self) -> float:
        """Calculate data quality SLA compliance."""
        if not self.metrics_history['data_quality']:
            return 1.0
            
        quality_scores = [m['data_quality_score'] for m in self.metrics_history['data_quality']]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        return avg_quality
        
    def _calculate_error_rate_sla(self) -> float:
        """Calculate error rate SLA compliance."""
        all_stats = self.performance_monitor.get_all_stats(1440)
        if not all_stats:
            return 1.0
            
        total_ops = sum(stats.count for stats in all_stats.values())
        failed_ops = sum(stats.count * (1 - stats.success_rate) for stats in all_stats.values())
        
        error_rate = failed_ops / total_ops if total_ops > 0 else 0
        
        # SLA: < 5% error rate
        return 1.0 if error_rate < 0.05 else 0.5


async def main():
    """Main monitoring function."""
    monitor = ProductionMonitor()
    
    try:
        print("🚀 Starting Production Monitoring System")
        print("📊 Monitoring will run for 24 hours unless interrupted")
        print("🔍 Tracking: System Resources, Trading Performance, Data Quality, Hardware Optimization")
        print("⚠️  Alerts will be logged for threshold violations")
        print()
        
        await monitor.start_monitoring()
        
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring interrupted by user")
    except Exception as e:
        print(f"❌ Monitoring error: {e}")
    finally:
        final_report = await monitor.stop_monitoring()
        print("📋 Final production report generated")
        print(f"📈 Overall Health Score: {final_report.get('kpi_summary', {}).get('availability', 0):.2%}")


if __name__ == "__main__":
    asyncio.run(main())