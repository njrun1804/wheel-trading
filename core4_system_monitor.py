#!/usr/bin/env python3
"""
Core 4 System Monitor - Comprehensive system monitoring and alerting
Integrates with process monitor and resource manager for complete system oversight
"""

import asyncio
import contextlib
import json
import logging
import os
import platform
import smtplib
import subprocess
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("core4_system_monitor.log"), logging.StreamHandler()],
)
logger = logging.getLogger("Core4SystemMonitor")


@dataclass
class SystemMetrics:
    """System metrics snapshot"""

    timestamp: datetime
    cpu_percent: float
    cpu_count: int
    cpu_freq: float
    load_avg: tuple[float, float, float]
    memory_percent: float
    memory_total_gb: float
    memory_used_gb: float
    memory_available_gb: float
    swap_percent: float
    disk_percent: float
    disk_total_gb: float
    disk_used_gb: float
    disk_free_gb: float
    process_count: int
    thread_count: int
    network_bytes_sent: int
    network_bytes_recv: int
    boot_time: datetime
    uptime_seconds: int
    temperature: float | None = None
    fan_speed: int | None = None


@dataclass
class ProcessMetrics:
    """Process metrics snapshot"""

    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    threads: int
    open_files: int
    connections: int
    status: str
    create_time: datetime
    cmdline: str


@dataclass
class Alert:
    """System alert"""

    id: str
    timestamp: datetime
    severity: str  # low, medium, high, critical
    category: str  # cpu, memory, disk, process, network
    message: str
    value: float
    threshold: float
    resolved: bool = False
    resolved_at: datetime | None = None
    actions_taken: list[str] = None


class Core4SystemMonitor:
    """Comprehensive system monitoring with alerting"""

    def __init__(self, config_file: str | None = None):
        self.config = self._load_config(config_file)
        self.metrics_history = deque(maxlen=1000)
        self.process_history = defaultdict(lambda: deque(maxlen=100))
        self.alerts = deque(maxlen=1000)
        self.active_alerts = {}
        self.monitoring = True
        self.last_alert_times = defaultdict(lambda: datetime.min)

        # Initialize components
        self._setup_monitoring()

    def _load_config(self, config_file: str | None) -> dict:
        """Load monitoring configuration"""
        default_config = {
            "monitoring": {
                "interval": 5,
                "process_monitoring": True,
                "network_monitoring": True,
                "disk_monitoring": True,
                "temperature_monitoring": True,
            },
            "thresholds": {
                "cpu_warning": 75.0,
                "cpu_critical": 90.0,
                "memory_warning": 80.0,
                "memory_critical": 95.0,
                "disk_warning": 80.0,
                "disk_critical": 95.0,
                "load_avg_warning": 8.0,
                "load_avg_critical": 12.0,
                "process_cpu_warning": 50.0,
                "process_memory_warning": 1024,  # MB
                "temperature_warning": 80.0,
                "temperature_critical": 90.0,
            },
            "alerts": {
                "email_enabled": False,
                "email_recipients": [],
                "email_smtp_server": "localhost",
                "email_smtp_port": 587,
                "cooldown_period": 300,  # 5 minutes
                "max_alerts_per_hour": 20,
            },
            "actions": {
                "auto_cleanup_enabled": True,
                "auto_restart_services": False,
                "emergency_shutdown": False,
            },
            "logging": {"level": "INFO", "max_log_size_mb": 100, "backup_count": 5},
        }

        if config_file and Path(config_file).exists():
            try:
                with open(config_file) as f:
                    user_config = json.load(f)
                self._deep_update(default_config, user_config)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")

        return default_config

    def _deep_update(self, base_dict: dict, update_dict: dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def _setup_monitoring(self):
        """Setup monitoring components"""
        # Create directories
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Setup log rotation
        try:
            from logging.handlers import RotatingFileHandler

            handler = RotatingFileHandler(
                "logs/core4_system_monitor.log",
                maxBytes=self.config["logging"]["max_log_size_mb"] * 1024 * 1024,
                backupCount=self.config["logging"]["backup_count"],
            )
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(handler)
        except ImportError:
            logger.warning("Log rotation not available")

    def get_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()

        # Load average
        try:
            load_avg = os.getloadavg()
        except (AttributeError, OSError):
            load_avg = (0.0, 0.0, 0.0)

        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Disk metrics
        disk = psutil.disk_usage("/")

        # Process metrics
        process_count = len(psutil.pids())
        thread_count = 0
        try:
            for p in psutil.process_iter():
                try:
                    if p.is_running():
                        thread_count += p.num_threads()
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
        except Exception:
            thread_count = 0

        # Network metrics
        network = psutil.net_io_counters()

        # System info
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time

        # Temperature (if available)
        temperature = self._get_cpu_temperature()

        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            cpu_freq=cpu_freq.current if cpu_freq else 0,
            load_avg=load_avg,
            memory_percent=memory.percent,
            memory_total_gb=memory.total / 1024**3,
            memory_used_gb=memory.used / 1024**3,
            memory_available_gb=memory.available / 1024**3,
            swap_percent=swap.percent,
            disk_percent=disk.percent,
            disk_total_gb=disk.total / 1024**3,
            disk_used_gb=disk.used / 1024**3,
            disk_free_gb=disk.free / 1024**3,
            process_count=process_count,
            thread_count=thread_count,
            network_bytes_sent=network.bytes_sent if network else 0,
            network_bytes_recv=network.bytes_recv if network else 0,
            boot_time=boot_time,
            uptime_seconds=int(uptime.total_seconds()),
            temperature=temperature,
        )

    def _get_cpu_temperature(self) -> float | None:
        """Get CPU temperature (macOS specific)"""
        try:
            # Try to get temperature from system
            if platform.system() == "Darwin":
                result = subprocess.run(
                    ["sudo", "powermetrics", "--samplers", "smc", "-n", "1", "-i", "1"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    # Parse temperature from output
                    for line in result.stdout.split("\n"):
                        if "CPU die temperature" in line:
                            temp_str = line.split(":")[-1].strip()
                            if temp_str.endswith("C"):
                                return float(temp_str[:-1])
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
            pass

        return None

    def get_process_metrics(self, limit: int = 20) -> list[ProcessMetrics]:
        """Get metrics for top processes"""
        processes = []

        for proc in psutil.process_iter(
            [
                "pid",
                "name",
                "cpu_percent",
                "memory_info",
                "memory_percent",
                "num_threads",
                "status",
                "create_time",
                "cmdline",
            ]
        ):
            try:
                info = proc.info

                # Get additional metrics
                open_files = 0
                connections = 0

                with contextlib.suppress(psutil.AccessDenied, psutil.NoSuchProcess):
                    open_files = len(proc.open_files())

                with contextlib.suppress(psutil.AccessDenied, psutil.NoSuchProcess):
                    connections = len(proc.connections())

                process_metrics = ProcessMetrics(
                    pid=info["pid"],
                    name=info["name"],
                    cpu_percent=info["cpu_percent"] or 0,
                    memory_percent=info["memory_percent"] or 0,
                    memory_mb=info["memory_info"].rss / 1024**2
                    if info["memory_info"]
                    else 0,
                    threads=info["num_threads"] or 0,
                    open_files=open_files,
                    connections=connections,
                    status=info["status"],
                    create_time=datetime.fromtimestamp(info["create_time"]),
                    cmdline=" ".join(info["cmdline"] or []),
                )

                processes.append(process_metrics)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Sort by CPU usage and return top processes
        processes.sort(key=lambda p: p.cpu_percent, reverse=True)
        return processes[:limit]

    def check_thresholds(self, metrics: SystemMetrics) -> list[Alert]:
        """Check metrics against thresholds and generate alerts"""
        alerts = []
        thresholds = self.config["thresholds"]
        datetime.now()

        # CPU alerts
        if metrics.cpu_percent >= thresholds["cpu_critical"]:
            alerts.append(
                self._create_alert(
                    "cpu_critical",
                    "critical",
                    "cpu",
                    f"Critical CPU usage: {metrics.cpu_percent:.1f}%",
                    metrics.cpu_percent,
                    thresholds["cpu_critical"],
                )
            )
        elif metrics.cpu_percent >= thresholds["cpu_warning"]:
            alerts.append(
                self._create_alert(
                    "cpu_warning",
                    "high",
                    "cpu",
                    f"High CPU usage: {metrics.cpu_percent:.1f}%",
                    metrics.cpu_percent,
                    thresholds["cpu_warning"],
                )
            )

        # Memory alerts
        if metrics.memory_percent >= thresholds["memory_critical"]:
            alerts.append(
                self._create_alert(
                    "memory_critical",
                    "critical",
                    "memory",
                    f"Critical memory usage: {metrics.memory_percent:.1f}%",
                    metrics.memory_percent,
                    thresholds["memory_critical"],
                )
            )
        elif metrics.memory_percent >= thresholds["memory_warning"]:
            alerts.append(
                self._create_alert(
                    "memory_warning",
                    "high",
                    "memory",
                    f"High memory usage: {metrics.memory_percent:.1f}%",
                    metrics.memory_percent,
                    thresholds["memory_warning"],
                )
            )

        # Disk alerts
        if metrics.disk_percent >= thresholds["disk_critical"]:
            alerts.append(
                self._create_alert(
                    "disk_critical",
                    "critical",
                    "disk",
                    f"Critical disk usage: {metrics.disk_percent:.1f}%",
                    metrics.disk_percent,
                    thresholds["disk_critical"],
                )
            )
        elif metrics.disk_percent >= thresholds["disk_warning"]:
            alerts.append(
                self._create_alert(
                    "disk_warning",
                    "high",
                    "disk",
                    f"High disk usage: {metrics.disk_percent:.1f}%",
                    metrics.disk_percent,
                    thresholds["disk_warning"],
                )
            )

        # Load average alerts
        if metrics.load_avg[0] >= thresholds["load_avg_critical"]:
            alerts.append(
                self._create_alert(
                    "load_avg_critical",
                    "critical",
                    "cpu",
                    f"Critical load average: {metrics.load_avg[0]:.2f}",
                    metrics.load_avg[0],
                    thresholds["load_avg_critical"],
                )
            )
        elif metrics.load_avg[0] >= thresholds["load_avg_warning"]:
            alerts.append(
                self._create_alert(
                    "load_avg_warning",
                    "high",
                    "cpu",
                    f"High load average: {metrics.load_avg[0]:.2f}",
                    metrics.load_avg[0],
                    thresholds["load_avg_warning"],
                )
            )

        # Temperature alerts
        if metrics.temperature:
            if metrics.temperature >= thresholds["temperature_critical"]:
                alerts.append(
                    self._create_alert(
                        "temperature_critical",
                        "critical",
                        "temperature",
                        f"Critical CPU temperature: {metrics.temperature:.1f}°C",
                        metrics.temperature,
                        thresholds["temperature_critical"],
                    )
                )
            elif metrics.temperature >= thresholds["temperature_warning"]:
                alerts.append(
                    self._create_alert(
                        "temperature_warning",
                        "high",
                        "temperature",
                        f"High CPU temperature: {metrics.temperature:.1f}°C",
                        metrics.temperature,
                        thresholds["temperature_warning"],
                    )
                )

        return alerts

    def check_process_thresholds(self, processes: list[ProcessMetrics]) -> list[Alert]:
        """Check process metrics against thresholds"""
        alerts = []
        thresholds = self.config["thresholds"]

        for proc in processes:
            # CPU alerts
            if proc.cpu_percent >= thresholds["process_cpu_warning"]:
                alerts.append(
                    self._create_alert(
                        f"process_cpu_{proc.pid}",
                        "medium",
                        "process",
                        f"High CPU usage by process {proc.name} (PID {proc.pid}): {proc.cpu_percent:.1f}%",
                        proc.cpu_percent,
                        thresholds["process_cpu_warning"],
                    )
                )

            # Memory alerts
            if proc.memory_mb >= thresholds["process_memory_warning"]:
                alerts.append(
                    self._create_alert(
                        f"process_memory_{proc.pid}",
                        "medium",
                        "process",
                        f"High memory usage by process {proc.name} (PID {proc.pid}): {proc.memory_mb:.1f}MB",
                        proc.memory_mb,
                        thresholds["process_memory_warning"],
                    )
                )

        return alerts

    def _create_alert(
        self,
        alert_id: str,
        severity: str,
        category: str,
        message: str,
        value: float,
        threshold: float,
    ) -> Alert:
        """Create a new alert"""
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            message=message,
            value=value,
            threshold=threshold,
            actions_taken=[],
        )

        return alert

    def handle_alerts(self, alerts: list[Alert]) -> dict:
        """Handle generated alerts"""
        actions_taken = {
            "notifications_sent": 0,
            "processes_killed": 0,
            "services_restarted": 0,
            "cleanup_performed": False,
        }

        for alert in alerts:
            # Check cooldown period
            if self._is_alert_in_cooldown(alert):
                continue

            # Add to active alerts
            self.active_alerts[alert.id] = alert
            self.alerts.append(alert)

            # Log alert
            logger.warning(f"ALERT [{alert.severity.upper()}] {alert.message}")

            # Send notifications
            if self.config["alerts"]["email_enabled"] and self._send_email_alert(alert):
                actions_taken["notifications_sent"] += 1

            # Take automated actions based on severity
            if alert.severity == "critical":
                if self.config["actions"]["auto_cleanup_enabled"]:
                    cleanup_result = self._perform_emergency_cleanup(alert)
                    if cleanup_result:
                        actions_taken["cleanup_performed"] = True
                        actions_taken["processes_killed"] += cleanup_result.get(
                            "processes_killed", 0
                        )
                        alert.actions_taken.append("emergency_cleanup")

            # Update last alert time
            self.last_alert_times[alert.id] = alert.timestamp

        return actions_taken

    def _is_alert_in_cooldown(self, alert: Alert) -> bool:
        """Check if alert is in cooldown period"""
        last_time = self.last_alert_times.get(alert.id, datetime.min)
        cooldown = timedelta(seconds=self.config["alerts"]["cooldown_period"])
        return datetime.now() - last_time < cooldown

    def _send_email_alert(self, alert: Alert) -> bool:
        """Send email notification for alert"""
        try:
            msg = MIMEMultipart()
            msg["From"] = "core4-monitor@system"
            msg["To"] = ", ".join(self.config["alerts"]["email_recipients"])
            msg[
                "Subject"
            ] = f"[{alert.severity.upper()}] System Alert: {alert.category}"

            body = f"""
System Alert Notification

Time: {alert.timestamp}
Severity: {alert.severity.upper()}
Category: {alert.category}
Message: {alert.message}
Value: {alert.value}
Threshold: {alert.threshold}

This is an automated message from Core 4 System Monitor.
            """

            msg.attach(MIMEText(body, "plain"))

            server = smtplib.SMTP(
                self.config["alerts"]["email_smtp_server"],
                self.config["alerts"]["email_smtp_port"],
            )
            server.sendmail(
                "core4-monitor@system",
                self.config["alerts"]["email_recipients"],
                msg.as_string(),
            )
            server.quit()

            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _perform_emergency_cleanup(self, alert: Alert) -> dict:
        """Perform emergency cleanup based on alert"""
        cleanup_result = {"processes_killed": 0, "memory_freed_mb": 0}

        try:
            if alert.category in ["memory", "cpu"]:
                # Import process monitor for cleanup
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "core4_process_monitor", "core4_process_monitor.py"
                )
                if spec and spec.loader:
                    monitor_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(monitor_module)

                    monitor = monitor_module.Core4ProcessMonitor()

                    # Clean up excessive processes
                    cleaned = monitor.auto_cleanup_excessive_processes()
                    cleanup_result["processes_killed"] = len(cleaned)
                    cleanup_result["memory_freed_mb"] = sum(
                        p.memory_mb for p in cleaned
                    )

                    logger.info(
                        f"Emergency cleanup: killed {len(cleaned)} processes, "
                        f"freed {cleanup_result['memory_freed_mb']:.1f}MB"
                    )

        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")

        return cleanup_result

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            del self.active_alerts[alert_id]
            logger.info(f"Alert resolved: {alert_id}")
            return True
        return False

    def get_system_health_score(self, metrics: SystemMetrics) -> dict:
        """Calculate overall system health score"""
        scores = {}

        # CPU health (0-100)
        cpu_score = max(0, 100 - metrics.cpu_percent)
        scores["cpu"] = cpu_score

        # Memory health (0-100)
        memory_score = max(0, 100 - metrics.memory_percent)
        scores["memory"] = memory_score

        # Disk health (0-100)
        disk_score = max(0, 100 - metrics.disk_percent)
        scores["disk"] = disk_score

        # Load average health (0-100)
        load_score = max(0, 100 - (metrics.load_avg[0] / metrics.cpu_count * 100))
        scores["load"] = load_score

        # Overall health score
        overall_score = (cpu_score + memory_score + disk_score + load_score) / 4
        scores["overall"] = overall_score

        # Health status
        if overall_score >= 90:
            status = "excellent"
        elif overall_score >= 75:
            status = "good"
        elif overall_score >= 60:
            status = "fair"
        elif overall_score >= 40:
            status = "poor"
        else:
            status = "critical"

        return {"scores": scores, "status": status, "timestamp": metrics.timestamp}

    def generate_monitoring_report(self) -> dict:
        """Generate comprehensive monitoring report"""
        current_metrics = self.get_system_metrics()
        process_metrics = self.get_process_metrics(10)
        health_score = self.get_system_health_score(current_metrics)

        # Get recent alerts
        recent_alerts = [
            alert
            for alert in self.alerts
            if datetime.now() - alert.timestamp < timedelta(hours=24)
        ]

        # Calculate uptime
        uptime_str = str(timedelta(seconds=current_metrics.uptime_seconds))

        return {
            "report_timestamp": datetime.now().isoformat(),
            "system_info": {
                "platform": platform.system(),
                "architecture": platform.machine(),
                "python_version": platform.python_version(),
                "uptime": uptime_str,
            },
            "current_metrics": asdict(current_metrics),
            "health_score": health_score,
            "top_processes": [asdict(p) for p in process_metrics],
            "active_alerts": len(self.active_alerts),
            "recent_alerts_24h": len(recent_alerts),
            "alert_summary": {
                "critical": len([a for a in recent_alerts if a.severity == "critical"]),
                "high": len([a for a in recent_alerts if a.severity == "high"]),
                "medium": len([a for a in recent_alerts if a.severity == "medium"]),
                "low": len([a for a in recent_alerts if a.severity == "low"]),
            },
        }

    async def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Starting Core 4 system monitoring loop")

        while self.monitoring:
            try:
                # Collect metrics
                system_metrics = self.get_system_metrics()
                self.metrics_history.append(system_metrics)

                # Collect process metrics
                if self.config["monitoring"]["process_monitoring"]:
                    process_metrics = self.get_process_metrics()
                    for proc in process_metrics:
                        self.process_history[proc.pid].append(proc)

                # Check thresholds
                system_alerts = self.check_thresholds(system_metrics)
                process_alerts = (
                    self.check_process_thresholds(process_metrics)
                    if process_metrics
                    else []
                )

                all_alerts = system_alerts + process_alerts

                # Handle alerts
                if all_alerts:
                    actions = self.handle_alerts(all_alerts)
                    if any(actions.values()):
                        logger.info(f"Alert actions taken: {actions}")

                # Auto-resolve alerts
                self._auto_resolve_alerts(system_metrics)

                await asyncio.sleep(self.config["monitoring"]["interval"])

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)

    def _auto_resolve_alerts(self, current_metrics: SystemMetrics):
        """Auto-resolve alerts when conditions improve"""
        resolved_alerts = []

        for alert_id, alert in list(self.active_alerts.items()):
            should_resolve = False

            if (
                alert.category == "cpu"
                and current_metrics.cpu_percent < alert.threshold * 0.8
                or alert.category == "memory"
                and current_metrics.memory_percent < alert.threshold * 0.8
                or alert.category == "disk"
                and current_metrics.disk_percent < alert.threshold * 0.8
            ):
                should_resolve = True

            if should_resolve:
                self.resolve_alert(alert_id)
                resolved_alerts.append(alert_id)

        if resolved_alerts:
            logger.info(f"Auto-resolved alerts: {resolved_alerts}")

    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.monitoring = False
        logger.info("Stopping Core 4 system monitoring")


def main():
    """Main function for standalone execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Core 4 System Monitor")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--monitor", action="store_true", help="Start monitoring loop")
    parser.add_argument(
        "--report", action="store_true", help="Generate monitoring report"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show current system status"
    )
    parser.add_argument(
        "--health", action="store_true", help="Show system health score"
    )
    parser.add_argument("--alerts", action="store_true", help="Show active alerts")

    args = parser.parse_args()

    monitor = Core4SystemMonitor(args.config)

    if args.report:
        report = monitor.generate_monitoring_report()
        print(json.dumps(report, indent=2, default=str))
        return

    if args.health:
        metrics = monitor.get_system_metrics()
        health = monitor.get_system_health_score(metrics)
        print(
            f"System Health Score: {health['scores']['overall']:.1f} ({health['status']})"
        )
        print(f"  CPU: {health['scores']['cpu']:.1f}")
        print(f"  Memory: {health['scores']['memory']:.1f}")
        print(f"  Disk: {health['scores']['disk']:.1f}")
        print(f"  Load: {health['scores']['load']:.1f}")
        return

    if args.alerts:
        if monitor.active_alerts:
            print("Active Alerts:")
            for _alert_id, alert in monitor.active_alerts.items():
                print(f"  [{alert.severity.upper()}] {alert.message}")
        else:
            print("No active alerts")
        return

    if args.status:
        metrics = monitor.get_system_metrics()
        print("System Status:")
        print(f"  CPU: {metrics.cpu_percent:.1f}% ({metrics.cpu_count} cores)")
        print(
            f"  Memory: {metrics.memory_percent:.1f}% ({metrics.memory_used_gb:.1f}GB / {metrics.memory_total_gb:.1f}GB)"
        )
        print(
            f"  Disk: {metrics.disk_percent:.1f}% ({metrics.disk_used_gb:.1f}GB / {metrics.disk_total_gb:.1f}GB)"
        )
        print(f"  Load: {metrics.load_avg[0]:.2f}")
        print(f"  Processes: {metrics.process_count}")
        print(f"  Uptime: {timedelta(seconds=metrics.uptime_seconds)}")
        if metrics.temperature:
            print(f"  Temperature: {metrics.temperature:.1f}°C")
        return

    if args.monitor:
        print("Starting system monitoring (Press Ctrl+C to stop)...")
        try:
            asyncio.run(monitor.monitor_loop())
        except KeyboardInterrupt:
            print("\nSystem monitoring stopped by user")
    else:
        # Default: show status
        args.status = True
        main()


if __name__ == "__main__":
    main()
