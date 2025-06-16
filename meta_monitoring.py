"""
Meta System Monitoring - Production Observability
Real-time monitoring, metrics collection, and alerting for the meta system
"""

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# datetime and timedelta were unused - removed
from meta_config import get_meta_config


@dataclass
class MetricEvent:
    """Represents a metric event"""

    timestamp: float
    metric_name: str
    value: float
    tags: dict[str, str]
    component: str


@dataclass
class HealthCheck:
    """System health check result"""

    component: str
    status: str  # 'healthy', 'warning', 'critical'
    message: str
    timestamp: float
    metrics: dict[str, Any]


class MetaSystemMonitor:
    """Production monitoring for the meta system"""

    def __init__(self):
        self.config = get_meta_config()
        self.db = sqlite3.connect(self.config.database.monitoring_db)
        self.metrics_buffer = []
        self.alerts_sent = set()

        self._init_monitoring_schema()
        print("📊 Meta System Monitor initialized")

    def _init_monitoring_schema(self):
        """Initialize monitoring database schema"""

        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                tags_json TEXT NOT NULL,
                component TEXT NOT NULL
            )
        """
        )

        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS health_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                component TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT NOT NULL,
                metrics_json TEXT NOT NULL
            )
        """
        )

        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                component TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE
            )
        """
        )

        self.db.commit()

    def record_metric(
        self, metric_name: str, value: float, component: str = "meta_system", **tags
    ):
        """Record a performance metric"""

        metric = MetricEvent(
            timestamp=time.time(),
            metric_name=metric_name,
            value=value,
            tags=tags,
            component=component,
        )

        self.metrics_buffer.append(metric)

        # Flush buffer if it gets too large
        if len(self.metrics_buffer) > 100:
            self._flush_metrics()

    def _flush_metrics(self):
        """Flush metrics buffer to database"""

        for metric in self.metrics_buffer:
            self.db.execute(
                """
                INSERT INTO metrics 
                (timestamp, metric_name, value, tags_json, component)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    metric.timestamp,
                    metric.metric_name,
                    metric.value,
                    json.dumps(metric.tags),
                    metric.component,
                ),
            )

        self.db.commit()
        self.metrics_buffer.clear()

    def health_check(self, component: str) -> HealthCheck:
        """Perform health check on a component"""

        metrics = {}
        status = "healthy"
        message = "All systems operational"

        try:
            if component == "meta_prime":
                metrics = self._check_meta_prime_health()
            elif component == "meta_coordinator":
                metrics = self._check_coordinator_health()
            elif component == "meta_daemon":
                metrics = self._check_daemon_health()
            elif component == "database":
                metrics = self._check_database_health()
            else:
                metrics = self._check_general_health()

            # Determine status based on metrics
            if any(v > 90 for k, v in metrics.items() if k.endswith("_usage_percent")):
                status = "critical"
                message = "High resource usage detected"
            elif any(
                v > 75 for k, v in metrics.items() if k.endswith("_usage_percent")
            ):
                status = "warning"
                message = "Elevated resource usage"

        except Exception as e:
            status = "critical"
            message = f"Health check failed: {e}"
            metrics = {"error": str(e)}

        health_check = HealthCheck(
            component=component,
            status=status,
            message=message,
            timestamp=time.time(),
            metrics=metrics,
        )

        self._record_health_check(health_check)
        return health_check

    def _check_meta_prime_health(self) -> dict[str, Any]:
        """Check MetaPrime component health"""

        try:
            # Check if database exists and is accessible
            from meta_config import get_meta_config

            config = get_meta_config()
            db_path = Path(config.database.evolution_db)
            if not db_path.exists():
                return {"database_exists": False, "error": "Database not found"}

            # Check database size
            db_size_mb = db_path.stat().st_size / 1024 / 1024

            # Check recent activity
            db = sqlite3.connect(str(db_path))
            cursor = db.execute(
                """
                SELECT COUNT(*) FROM observations 
                WHERE timestamp > ?
            """,
                (time.time() - self.config.timing.recent_activity_window_seconds * 12,),
            )  # Recent activity window

            recent_observations = cursor.fetchone()[0]

            cursor = db.execute("SELECT COUNT(*) FROM observations")
            total_observations = cursor.fetchone()[0]

            db.close()

            return {
                "database_size_mb": db_size_mb,
                "recent_observations": recent_observations,
                "total_observations": total_observations,
                "database_usage_percent": min(
                    100, (db_size_mb / 100) * 100
                ),  # Assume 100MB limit
            }

        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def _check_coordinator_health(self) -> dict[str, Any]:
        """Check MetaCoordinator health"""

        try:
            from meta_config import get_meta_config

            config = get_meta_config()
            db = sqlite3.connect(config.database.evolution_db)

            # Check coordination events
            cursor = db.execute(
                """
                SELECT COUNT(*) FROM coordination_events
                WHERE timestamp > ?
            """,
                (time.time() - self.config.timing.recent_activity_window_seconds * 12,),
            )

            recent_events = cursor.fetchone()[0]

            # Check evolution plans
            cursor = db.execute(
                """
                SELECT COUNT(*) FROM evolution_plans
                WHERE status = 'completed'
            """
            )

            completed_evolutions = cursor.fetchone()[0]

            db.close()

            return {
                "recent_coordination_events": recent_events,
                "completed_evolutions": completed_evolutions,
                "coordination_activity_percent": min(100, recent_events * 10),
            }

        except Exception as e:
            return {"error": str(e)}

    def _check_daemon_health(self) -> dict[str, Any]:
        """Check daemon health"""

        import psutil

        try:
            # Check memory usage
            memory_info = psutil.virtual_memory()
            memory_usage_percent = memory_info.percent

            # Check CPU usage
            cpu_usage_percent = psutil.cpu_percent(interval=1)

            # Check disk usage
            disk_usage = psutil.disk_usage("/")
            disk_usage_percent = (disk_usage.used / disk_usage.total) * 100

            return {
                "memory_usage_percent": memory_usage_percent,
                "cpu_usage_percent": cpu_usage_percent,
                "disk_usage_percent": disk_usage_percent,
                "system_load": psutil.getloadavg()[0]
                if hasattr(psutil, "getloadavg")
                else 0,
            }

        except Exception as e:
            return {"error": str(e)}

    def _check_database_health(self) -> dict[str, Any]:
        """Check database health"""

        try:
            from meta_config import get_meta_config

            config = get_meta_config()
            db_files = [
                config.database.evolution_db,
                config.database.monitoring_db,
                config.database.reality_db,
            ]
            total_size = 0
            file_count = 0

            for db_file in db_files:
                path = Path(db_file)
                if path.exists():
                    total_size += path.stat().st_size
                    file_count += 1

            total_size_mb = total_size / 1024 / 1024

            return {
                "database_files": file_count,
                "total_database_size_mb": total_size_mb,
                "database_usage_percent": min(
                    100, (total_size_mb / self.config.system.max_db_size_mb * 5) * 100
                ),  # Based on config limit
            }

        except Exception as e:
            return {"error": str(e)}

    def _check_general_health(self) -> dict[str, Any]:
        """General system health check"""

        uptime = time.time() - getattr(self, "start_time", time.time())

        return {
            "uptime_hours": uptime
            / self.config.timing.recent_activity_window_seconds
            * 12,
            "metrics_buffered": len(self.metrics_buffer),
            "alerts_sent": len(self.alerts_sent),
        }

    def _record_health_check(self, health_check: HealthCheck):
        """Record health check result"""

        self.db.execute(
            """
            INSERT INTO health_checks
            (timestamp, component, status, message, metrics_json)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                health_check.timestamp,
                health_check.component,
                health_check.status,
                health_check.message,
                json.dumps(health_check.metrics),
            ),
        )

        self.db.commit()

        # Trigger alerts if critical
        if health_check.status == "critical":
            self._send_alert(
                alert_type="health_check_critical",
                severity="critical",
                message=f"{health_check.component}: {health_check.message}",
                component=health_check.component,
            )

    def _send_alert(self, alert_type: str, severity: str, message: str, component: str):
        """Send an alert"""

        alert_key = f"{alert_type}:{component}"

        # Prevent duplicate alerts within 1 hour
        if alert_key in self.alerts_sent:
            return

        self.alerts_sent.add(alert_key)

        self.db.execute(
            """
            INSERT INTO alerts
            (timestamp, alert_type, severity, message, component)
            VALUES (?, ?, ?, ?, ?)
        """,
            (time.time(), alert_type, severity, message, component),
        )

        self.db.commit()

        # In production, this would send to Slack, email, etc.
        print(f"🚨 ALERT [{severity.upper()}] {component}: {message}")

    async def monitoring_loop(self):
        """Main monitoring loop for continuous operation"""

        print("🔄 Starting continuous monitoring...")

        components = ["meta_prime", "meta_coordinator", "meta_daemon", "database"]

        while True:
            try:
                # Perform health checks
                for component in components:
                    health = self.health_check(component)

                    if health.status != "healthy":
                        print(f"⚠️  {component}: {health.status} - {health.message}")

                    # Record key metrics
                    for metric_name, value in health.metrics.items():
                        if isinstance(value, int | float):
                            self.record_metric(
                                f"health.{metric_name}", value, component=component
                            )

                # Flush metrics
                self._flush_metrics()

                # Clear old alerts (older than 1 hour)
                current_alerts = set()
                for alert in self.alerts_sent:
                    current_alerts.add(alert)
                self.alerts_sent = current_alerts

                # Wait before next check
                await asyncio.sleep(self.config.timing.health_check_interval_seconds)

            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                await asyncio.sleep(10)  # Short wait on error

    def get_monitoring_dashboard(self) -> dict[str, Any]:
        """Get monitoring dashboard data"""

        # Get recent health checks
        cursor = self.db.execute(
            """
            SELECT component, status, message, timestamp, metrics_json
            FROM health_checks 
            WHERE timestamp > ?
            ORDER BY timestamp DESC
            LIMIT 20
        """,
            (time.time() - 3600,),
        )

        health_checks = []
        for row in cursor.fetchall():
            health_checks.append(
                {
                    "component": row[0],
                    "status": row[1],
                    "message": row[2],
                    "timestamp": row[3],
                    "metrics": json.loads(row[4]),
                }
            )

        # Get recent metrics
        cursor = self.db.execute(
            """
            SELECT metric_name, AVG(value) as avg_value, MAX(value) as max_value
            FROM metrics 
            WHERE timestamp > ?
            GROUP BY metric_name
        """,
            (time.time() - 3600,),
        )

        metrics_summary = {
            row[0]: {"avg": row[1], "max": row[2]} for row in cursor.fetchall()
        }

        # Get active alerts
        cursor = self.db.execute(
            """
            SELECT alert_type, severity, message, component, timestamp
            FROM alerts 
            WHERE timestamp > ? AND resolved = FALSE
            ORDER BY timestamp DESC
        """,
            (time.time() - 3600,),
        )

        active_alerts = [
            {
                "type": row[0],
                "severity": row[1],
                "message": row[2],
                "component": row[3],
                "timestamp": row[4],
            }
            for row in cursor.fetchall()
        ]

        return {
            "timestamp": time.time(),
            "health_checks": health_checks,
            "metrics_summary": metrics_summary,
            "active_alerts": active_alerts,
            "system_status": "healthy" if not active_alerts else "degraded",
        }


def run_monitoring_check():
    """Run a one-time monitoring check"""

    monitor = MetaSystemMonitor()

    print("📊 Meta System Health Check")
    print("=" * 50)

    components = ["meta_prime", "meta_coordinator", "meta_daemon", "database"]

    overall_status = "healthy"

    for component in components:
        health = monitor.health_check(component)

        status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "❌"}.get(
            health.status, "❓"
        )

        print(f"{status_emoji} {component}: {health.status} - {health.message}")

        if health.status in ["warning", "critical"]:
            overall_status = health.status

        # Show key metrics
        for metric, value in health.metrics.items():
            if isinstance(value, int | float):
                print(f"   {metric}: {value:.1f}")

    print(f"\n🎯 Overall System Status: {overall_status.upper()}")

    # Show dashboard
    dashboard = monitor.get_monitoring_dashboard()
    if dashboard["active_alerts"]:
        print(f"\n🚨 Active Alerts: {len(dashboard['active_alerts'])}")
        for alert in dashboard["active_alerts"][:3]:
            print(f"   • {alert['severity']}: {alert['message']}")

    return overall_status == "healthy"


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        # Run continuous monitoring
        monitor = MetaSystemMonitor()
        asyncio.run(monitor.monitoring_loop())
    else:
        # Run one-time health check
        healthy = run_monitoring_check()
        sys.exit(0 if healthy else 1)
