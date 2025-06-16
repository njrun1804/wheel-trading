"""Advanced resource monitoring and alerting system.

This module provides real-time monitoring of system resources with 
intelligent alerting and automatic remediation capabilities.
"""

from __future__ import annotations

import asyncio
import json
import logging
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MimeText
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil

from .resource_manager import ResourceMetrics, get_resource_tracker
from .logging import get_logger

logger = get_logger(__name__)


class AlertLevel:
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


class ResourceAlert:
    """Represents a resource usage alert."""
    
    def __init__(self, level: str, message: str, metrics: ResourceMetrics, 
                 resource_type: str = "system"):
        self.level = level
        self.message = message
        self.metrics = metrics
        self.resource_type = resource_type
        self.timestamp = datetime.now()
        self.resolved = False
        self.resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "level": self.level,
            "message": self.message,
            "resource_type": self.resource_type,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None,
            "metrics": {
                "open_files": self.metrics.open_files,
                "memory_mb": self.metrics.memory_mb,
                "processes": self.metrics.processes,
                "threads": self.metrics.threads,
            }
        }


class ResourceThresholds:
    """Configurable thresholds for resource monitoring."""
    
    def __init__(self):
        # File descriptor thresholds (percentage of max)
        self.fd_warning = 0.7   # 70%
        self.fd_error = 0.85    # 85%
        self.fd_critical = 0.95 # 95%
        
        # Memory thresholds (MB)
        self.memory_warning = 1500
        self.memory_error = 2000
        self.memory_critical = 2500
        
        # Process/thread thresholds
        self.process_warning = 50
        self.process_error = 100
        self.thread_warning = 200
        self.thread_error = 500
        
        # Network connection thresholds
        self.network_warning = 100
        self.network_error = 200
        
        # Rate-based thresholds
        self.memory_growth_rate_mb_per_min = 100  # MB/min
        self.fd_growth_rate_per_min = 50         # FDs/min
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> ResourceThresholds:
        """Create thresholds from configuration."""
        thresholds = cls()
        for key, value in config.items():
            if hasattr(thresholds, key):
                setattr(thresholds, key, value)
        return thresholds


class ResourceMonitor:
    """Advanced resource monitor with alerting and remediation."""
    
    def __init__(self, thresholds: Optional[ResourceThresholds] = None,
                 alert_cooldown: int = 300):  # 5 minutes
        self.thresholds = thresholds or ResourceThresholds()
        self.alert_cooldown = alert_cooldown
        
        # Resource tracker
        self.tracker = get_resource_tracker()
        
        # Alert management
        self.active_alerts: List[ResourceAlert] = []
        self.alert_history: List[ResourceAlert] = []
        self.last_alert_time: Dict[str, datetime] = {}
        
        # Metrics history for trend analysis
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history_size = 1000
        
        # Alert handlers
        self.alert_handlers: List[Callable[[ResourceAlert], None]] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Report generation
        self.report_path = Path("resource_monitoring_report.json")
        
        # Setup default handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default alert handlers."""
        self.add_alert_handler(self._log_alert)
        self.add_alert_handler(self._save_alert_to_file)
    
    def add_alert_handler(self, handler: Callable[[ResourceAlert], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def _log_alert(self, alert: ResourceAlert):
        """Default handler: log the alert."""
        log_func = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical,
        }.get(alert.level, logger.info)
        
        log_func(f"Resource Alert [{alert.level.upper()}]: {alert.message}")
    
    def _save_alert_to_file(self, alert: ResourceAlert):
        """Default handler: save alert to file."""
        try:
            alerts_file = Path("resource_alerts.jsonl")
            with open(alerts_file, "a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to save alert to file: {e}")
    
    def _should_alert(self, alert_key: str) -> bool:
        """Check if enough time has passed since last alert of this type."""
        if alert_key not in self.last_alert_time:
            return True
        
        time_since_last = datetime.now() - self.last_alert_time[alert_key]
        return time_since_last.total_seconds() >= self.alert_cooldown
    
    def _trigger_alert(self, level: str, message: str, metrics: ResourceMetrics, 
                      resource_type: str = "system"):
        """Trigger an alert and call all handlers."""
        alert_key = f"{resource_type}_{level}"
        
        if not self._should_alert(alert_key):
            return
        
        alert = ResourceAlert(level, message, metrics, resource_type)
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        self.last_alert_time[alert_key] = datetime.now()
        
        # Call all alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def _check_file_descriptors(self, metrics: ResourceMetrics):
        """Check file descriptor usage against thresholds."""
        if metrics.max_files == 0:
            return
        
        fd_ratio = metrics.open_files / metrics.max_files
        
        if fd_ratio >= self.thresholds.fd_critical:
            self._trigger_alert(
                AlertLevel.CRITICAL,
                f"Critical file descriptor usage: {metrics.open_files}/{metrics.max_files} ({fd_ratio:.1%})",
                metrics,
                "file_descriptors"
            )
        elif fd_ratio >= self.thresholds.fd_error:
            self._trigger_alert(
                AlertLevel.ERROR,
                f"High file descriptor usage: {metrics.open_files}/{metrics.max_files} ({fd_ratio:.1%})",
                metrics,
                "file_descriptors"
            )
        elif fd_ratio >= self.thresholds.fd_warning:
            self._trigger_alert(
                AlertLevel.WARNING,
                f"Elevated file descriptor usage: {metrics.open_files}/{metrics.max_files} ({fd_ratio:.1%})",
                metrics,
                "file_descriptors"
            )
    
    def _check_memory_usage(self, metrics: ResourceMetrics):
        """Check memory usage against thresholds."""
        if metrics.memory_mb >= self.thresholds.memory_critical:
            self._trigger_alert(
                AlertLevel.CRITICAL,
                f"Critical memory usage: {metrics.memory_mb:.1f}MB",
                metrics,
                "memory"
            )
        elif metrics.memory_mb >= self.thresholds.memory_error:
            self._trigger_alert(
                AlertLevel.ERROR,
                f"High memory usage: {metrics.memory_mb:.1f}MB",
                metrics,
                "memory"
            )
        elif metrics.memory_mb >= self.thresholds.memory_warning:
            self._trigger_alert(
                AlertLevel.WARNING,
                f"Elevated memory usage: {metrics.memory_mb:.1f}MB",
                metrics,
                "memory"
            )
    
    def _check_process_threads(self, metrics: ResourceMetrics):
        """Check process and thread counts."""
        if metrics.processes >= self.thresholds.process_error:
            self._trigger_alert(
                AlertLevel.ERROR,
                f"High process count: {metrics.processes}",
                metrics,
                "processes"
            )
        elif metrics.processes >= self.thresholds.process_warning:
            self._trigger_alert(
                AlertLevel.WARNING,
                f"Elevated process count: {metrics.processes}",
                metrics,
                "processes"
            )
        
        if metrics.threads >= self.thresholds.thread_error:
            self._trigger_alert(
                AlertLevel.ERROR,
                f"High thread count: {metrics.threads}",
                metrics,
                "threads"
            )
        elif metrics.threads >= self.thresholds.thread_warning:
            self._trigger_alert(
                AlertLevel.WARNING,
                f"Elevated thread count: {metrics.threads}",
                metrics,
                "threads"
            )
    
    def _check_growth_rates(self, metrics: ResourceMetrics):
        """Check resource growth rates for trends."""
        if len(self.metrics_history) < 2:
            return
        
        # Get metrics from 1 minute ago
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        past_metrics = None
        for m in reversed(self.metrics_history):
            if m.timestamp <= one_minute_ago:
                past_metrics = m
                break
        
        if not past_metrics:
            return
        
        time_diff_minutes = (now - past_metrics.timestamp).total_seconds() / 60
        if time_diff_minutes == 0:
            return
        
        # Check memory growth rate
        memory_growth = (metrics.memory_mb - past_metrics.memory_mb) / time_diff_minutes
        if memory_growth > self.thresholds.memory_growth_rate_mb_per_min:
            self._trigger_alert(
                AlertLevel.WARNING,
                f"Rapid memory growth: {memory_growth:.1f}MB/min",
                metrics,
                "memory_growth"
            )
        
        # Check file descriptor growth rate
        fd_growth = (metrics.open_files - past_metrics.open_files) / time_diff_minutes
        if fd_growth > self.thresholds.fd_growth_rate_per_min:
            self._trigger_alert(
                AlertLevel.WARNING,
                f"Rapid file descriptor growth: {fd_growth:.1f}/min",
                metrics,
                "fd_growth"
            )
    
    def _resolve_alerts(self, metrics: ResourceMetrics):
        """Check if any active alerts can be resolved."""
        for alert in self.active_alerts[:]:  # Copy list to modify during iteration
            should_resolve = False
            
            if alert.resource_type == "file_descriptors":
                fd_ratio = metrics.open_files / metrics.max_files if metrics.max_files > 0 else 0
                should_resolve = fd_ratio < self.thresholds.fd_warning
                
            elif alert.resource_type == "memory":
                should_resolve = metrics.memory_mb < self.thresholds.memory_warning
                
            elif alert.resource_type == "processes":
                should_resolve = metrics.processes < self.thresholds.process_warning
                
            elif alert.resource_type == "threads":
                should_resolve = metrics.threads < self.thresholds.thread_warning
            
            if should_resolve:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                self.active_alerts.remove(alert)
                logger.info(f"Alert resolved: {alert.message}")
    
    def check_resources(self) -> ResourceMetrics:
        """Check all resources and trigger alerts as needed."""
        metrics = self.tracker.get_current_metrics()
        
        # Add to history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)
        
        # Check all thresholds
        self._check_file_descriptors(metrics)
        self._check_memory_usage(metrics)
        self._check_process_threads(metrics)
        self._check_growth_rates(metrics)
        
        # Check for alert resolution
        self._resolve_alerts(metrics)
        
        return metrics
    
    async def start_monitoring(self, interval: float = 30.0):
        """Start continuous resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info(f"Started resource monitoring (interval: {interval}s)")
    
    async def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped resource monitoring")
    
    async def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self.check_resources()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive resource monitoring report."""
        current_metrics = self.tracker.get_current_metrics()
        
        # Calculate statistics from history
        if self.metrics_history:
            memory_values = [m.memory_mb for m in self.metrics_history]
            fd_values = [m.open_files for m in self.metrics_history]
            
            memory_stats = {
                "current": current_metrics.memory_mb,
                "avg": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
            }
            
            fd_stats = {
                "current": current_metrics.open_files,
                "avg": sum(fd_values) / len(fd_values),
                "max": max(fd_values),
                "min": min(fd_values),
            }
        else:
            memory_stats = {"current": current_metrics.memory_mb}
            fd_stats = {"current": current_metrics.open_files}
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": {
                "open_files": current_metrics.open_files,
                "max_files": current_metrics.max_files,
                "memory_mb": current_metrics.memory_mb,
                "memory_percent": current_metrics.memory_percent,
                "processes": current_metrics.processes,
                "threads": current_metrics.threads,
                "network_connections": current_metrics.network_connections,
            },
            "statistics": {
                "memory": memory_stats,
                "file_descriptors": fd_stats,
            },
            "active_alerts": [alert.to_dict() for alert in self.active_alerts],
            "alert_history_count": len(self.alert_history),
            "recent_alerts": [alert.to_dict() for alert in self.alert_history[-10:]],
            "thresholds": {
                "fd_warning": self.thresholds.fd_warning,
                "fd_error": self.thresholds.fd_error,
                "fd_critical": self.thresholds.fd_critical,
                "memory_warning": self.thresholds.memory_warning,
                "memory_error": self.thresholds.memory_error,
                "memory_critical": self.thresholds.memory_critical,
            },
        }
        
        return report
    
    def save_report(self, file_path: Optional[Path] = None) -> Path:
        """Save monitoring report to file."""
        if file_path is None:
            file_path = self.report_path
        
        report = self.generate_report()
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Resource monitoring report saved to {file_path}")
        return file_path
    
    def cleanup(self):
        """Clean up monitoring resources."""
        if self.monitoring_active:
            asyncio.create_task(self.stop_monitoring())
        
        # Save final report
        try:
            self.save_report()
        except Exception as e:
            logger.error(f"Failed to save final report: {e}")


# Email alert handler
class EmailAlertHandler:
    """Send alerts via email."""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, 
                 password: str, from_email: str, to_emails: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
    
    def __call__(self, alert: ResourceAlert):
        """Send alert via email."""
        if alert.level not in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            return  # Only send emails for serious alerts
        
        try:
            subject = f"Resource Alert [{alert.level.upper()}]: {alert.resource_type}"
            body = f"""
Resource Alert Notification

Level: {alert.level.upper()}
Resource Type: {alert.resource_type}
Time: {alert.timestamp}
Message: {alert.message}

Current Metrics:
- Open Files: {alert.metrics.open_files}
- Memory: {alert.metrics.memory_mb:.1f}MB
- Processes: {alert.metrics.processes}
- Threads: {alert.metrics.threads}

This is an automated alert from the resource monitoring system.
"""
            
            msg = MimeText(body)
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Alert email sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")


# Global monitor instance
_resource_monitor: Optional[ResourceMonitor] = None


def get_resource_monitor() -> ResourceMonitor:
    """Get or create global resource monitor."""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
    return _resource_monitor


# Convenience functions
async def start_monitoring(interval: float = 30.0):
    """Start resource monitoring."""
    monitor = get_resource_monitor()
    await monitor.start_monitoring(interval)


async def stop_monitoring():
    """Stop resource monitoring."""
    monitor = get_resource_monitor()
    await monitor.stop_monitoring()


def generate_report() -> Dict[str, Any]:
    """Generate resource monitoring report."""
    monitor = get_resource_monitor()
    return monitor.generate_report()


def save_report(file_path: Optional[Path] = None) -> Path:
    """Save monitoring report to file."""
    monitor = get_resource_monitor()
    return monitor.save_report(file_path)


def add_email_alerts(smtp_server: str, smtp_port: int, username: str,
                    password: str, from_email: str, to_emails: List[str]):
    """Add email alert handler."""
    monitor = get_resource_monitor()
    email_handler = EmailAlertHandler(smtp_server, smtp_port, username, 
                                    password, from_email, to_emails)
    monitor.add_alert_handler(email_handler)