#!/usr/bin/env python3
"""
System Health Monitor and Alerting Validation
Monitors system health metrics and validates alerting functionality.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any

import psutil

# Add paths
sys.path.append(".")
sys.path.append("src")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SystemHealthMonitor:
    """Real-time system health monitoring and alerting."""

    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        self.start_time = datetime.now()

        # Health thresholds
        self.thresholds = {
            "cpu_percent_warning": 80.0,
            "cpu_percent_critical": 95.0,
            "memory_percent_warning": 80.0,
            "memory_percent_critical": 95.0,
            "disk_percent_warning": 85.0,
            "disk_percent_critical": 95.0,
            "response_time_warning": 1000.0,  # ms
            "response_time_critical": 5000.0,  # ms
        }

    async def collect_metrics(self) -> dict[str, Any]:
        """Collect comprehensive system metrics."""
        try:
            # Basic system metrics
            cpu_percent = psutil.cpu_percent(interval=1.0)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(".")

            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                network_metrics = {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv,
                }
            except:
                network_metrics = {}

            # Process metrics
            try:
                python_processes = [
                    p
                    for p in psutil.process_iter(
                        ["pid", "name", "cpu_percent", "memory_percent"]
                    )
                    if "python" in p.info["name"].lower()
                ]
                process_count = len(python_processes)
                total_python_cpu = sum(
                    p.info["cpu_percent"] or 0 for p in python_processes
                )
                total_python_memory = sum(
                    p.info["memory_percent"] or 0 for p in python_processes
                )
            except:
                process_count = 0
                total_python_cpu = 0
                total_python_memory = 0

            # Application-specific metrics
            app_metrics = await self.collect_application_metrics()

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_total_gb": memory.total / (1024**3),
                    "memory_available_gb": memory.available / (1024**3),
                    "memory_percent": memory.percent,
                    "disk_total_gb": disk.total / (1024**3),
                    "disk_free_gb": disk.free / (1024**3),
                    "disk_percent": (disk.used / disk.total) * 100,
                    "load_average": os.getloadavg()
                    if hasattr(os, "getloadavg")
                    else [0, 0, 0],
                },
                "network": network_metrics,
                "processes": {
                    "python_process_count": process_count,
                    "total_python_cpu": total_python_cpu,
                    "total_python_memory": total_python_memory,
                },
                "application": app_metrics,
            }

            # Store metrics
            self.metrics_history.append(metrics)

            # Keep only last 100 metrics
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]

            return metrics

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    async def collect_application_metrics(self) -> dict[str, Any]:
        """Collect application-specific health metrics."""
        app_metrics = {}

        try:
            # Test Einstein responsiveness
            start_time = time.time()
            try:
                import psutil

                from einstein.einstein_config import EinsteinConfig, HardwareConfig

                hw = HardwareConfig(
                    cpu_cores=psutil.cpu_count(),
                    cpu_performance_cores=8,
                    cpu_efficiency_cores=4,
                    memory_total_gb=24.0,
                    memory_available_gb=20.0,
                    has_gpu=True,
                    gpu_cores=20,
                    platform_type="apple_silicon",
                    architecture="arm64",
                )
                EinsteinConfig(hardware=hw)
                einstein_response_time = (time.time() - start_time) * 1000
                einstein_healthy = True
            except Exception:
                einstein_response_time = None
                einstein_healthy = False

            app_metrics["einstein"] = {
                "healthy": einstein_healthy,
                "response_time_ms": einstein_response_time,
            }

            # Test database connectivity
            start_time = time.time()
            try:
                import duckdb

                conn = duckdb.connect(":memory:")
                conn.execute("SELECT 1").fetchone()
                conn.close()
                db_response_time = (time.time() - start_time) * 1000
                db_healthy = True
            except Exception:
                db_response_time = None
                db_healthy = False

            app_metrics["database"] = {
                "healthy": db_healthy,
                "response_time_ms": db_response_time,
            }

            # Test accelerated tools
            start_time = time.time()
            try:
                from src.unity_wheel.accelerated_tools.ripgrep_turbo import (
                    get_ripgrep_turbo,
                )

                get_ripgrep_turbo()
                tools_response_time = (time.time() - start_time) * 1000
                tools_healthy = True
            except Exception:
                tools_response_time = None
                tools_healthy = False

            app_metrics["accelerated_tools"] = {
                "healthy": tools_healthy,
                "response_time_ms": tools_response_time,
            }

        except Exception as e:
            app_metrics["error"] = str(e)

        return app_metrics

    def check_thresholds(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """Check metrics against thresholds and generate alerts."""
        alerts = []

        if "system" not in metrics:
            return alerts

        system = metrics["system"]
        timestamp = metrics["timestamp"]

        # CPU alerts
        cpu_percent = system.get("cpu_percent", 0)
        if cpu_percent >= self.thresholds["cpu_percent_critical"]:
            alerts.append(
                {
                    "level": "CRITICAL",
                    "component": "CPU",
                    "message": f"CPU usage critical: {cpu_percent:.1f}%",
                    "value": cpu_percent,
                    "threshold": self.thresholds["cpu_percent_critical"],
                    "timestamp": timestamp,
                }
            )
        elif cpu_percent >= self.thresholds["cpu_percent_warning"]:
            alerts.append(
                {
                    "level": "WARNING",
                    "component": "CPU",
                    "message": f"CPU usage high: {cpu_percent:.1f}%",
                    "value": cpu_percent,
                    "threshold": self.thresholds["cpu_percent_warning"],
                    "timestamp": timestamp,
                }
            )

        # Memory alerts
        memory_percent = system.get("memory_percent", 0)
        if memory_percent >= self.thresholds["memory_percent_critical"]:
            alerts.append(
                {
                    "level": "CRITICAL",
                    "component": "MEMORY",
                    "message": f"Memory usage critical: {memory_percent:.1f}%",
                    "value": memory_percent,
                    "threshold": self.thresholds["memory_percent_critical"],
                    "timestamp": timestamp,
                }
            )
        elif memory_percent >= self.thresholds["memory_percent_warning"]:
            alerts.append(
                {
                    "level": "WARNING",
                    "component": "MEMORY",
                    "message": f"Memory usage high: {memory_percent:.1f}%",
                    "value": memory_percent,
                    "threshold": self.thresholds["memory_percent_warning"],
                    "timestamp": timestamp,
                }
            )

        # Disk alerts
        disk_percent = system.get("disk_percent", 0)
        if disk_percent >= self.thresholds["disk_percent_critical"]:
            alerts.append(
                {
                    "level": "CRITICAL",
                    "component": "DISK",
                    "message": f"Disk usage critical: {disk_percent:.1f}%",
                    "value": disk_percent,
                    "threshold": self.thresholds["disk_percent_critical"],
                    "timestamp": timestamp,
                }
            )
        elif disk_percent >= self.thresholds["disk_percent_warning"]:
            alerts.append(
                {
                    "level": "WARNING",
                    "component": "DISK",
                    "message": f"Disk usage high: {disk_percent:.1f}%",
                    "value": disk_percent,
                    "threshold": self.thresholds["disk_percent_warning"],
                    "timestamp": timestamp,
                }
            )

        # Application response time alerts
        if "application" in metrics:
            for component, app_metrics in metrics["application"].items():
                if isinstance(app_metrics, dict) and "response_time_ms" in app_metrics:
                    response_time = app_metrics["response_time_ms"]
                    if (
                        response_time
                        and response_time >= self.thresholds["response_time_critical"]
                    ):
                        alerts.append(
                            {
                                "level": "CRITICAL",
                                "component": component.upper(),
                                "message": f"{component} response time critical: {response_time:.1f}ms",
                                "value": response_time,
                                "threshold": self.thresholds["response_time_critical"],
                                "timestamp": timestamp,
                            }
                        )
                    elif (
                        response_time
                        and response_time >= self.thresholds["response_time_warning"]
                    ):
                        alerts.append(
                            {
                                "level": "WARNING",
                                "component": component.upper(),
                                "message": f"{component} response time high: {response_time:.1f}ms",
                                "value": response_time,
                                "threshold": self.thresholds["response_time_warning"],
                                "timestamp": timestamp,
                            }
                        )

        return alerts

    def generate_health_report(self) -> dict[str, Any]:
        """Generate comprehensive health report."""
        if not self.metrics_history:
            return {"error": "No metrics collected yet"}

        latest_metrics = self.metrics_history[-1]

        # Calculate averages over last 10 measurements
        recent_metrics = (
            self.metrics_history[-10:]
            if len(self.metrics_history) >= 10
            else self.metrics_history
        )

        avg_cpu = sum(
            m.get("system", {}).get("cpu_percent", 0) for m in recent_metrics
        ) / len(recent_metrics)
        avg_memory = sum(
            m.get("system", {}).get("memory_percent", 0) for m in recent_metrics
        ) / len(recent_metrics)
        avg_disk = sum(
            m.get("system", {}).get("disk_percent", 0) for m in recent_metrics
        ) / len(recent_metrics)

        # Health status determination
        critical_alerts = [a for a in self.alerts if a.get("level") == "CRITICAL"]
        warning_alerts = [a for a in self.alerts if a.get("level") == "WARNING"]

        if critical_alerts:
            health_status = "CRITICAL"
        elif warning_alerts:
            health_status = "WARNING"
        elif avg_cpu > 50 or avg_memory > 70:
            health_status = "MODERATE"
        else:
            health_status = "HEALTHY"

        uptime = datetime.now() - self.start_time

        report = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime.total_seconds(),
            "health_status": health_status,
            "current_metrics": latest_metrics,
            "averages": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory,
                "disk_percent": avg_disk,
                "measurement_count": len(recent_metrics),
            },
            "alerts": {
                "total": len(self.alerts),
                "critical": len(critical_alerts),
                "warning": len(warning_alerts),
                "recent_alerts": self.alerts[-5:] if self.alerts else [],
            },
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            },
        }

        return report

    async def run_monitoring_cycle(
        self, duration_seconds: int = 60, interval_seconds: int = 5
    ):
        """Run monitoring for specified duration."""
        logger.info(f"Starting health monitoring for {duration_seconds} seconds...")

        end_time = time.time() + duration_seconds
        cycle_count = 0

        while time.time() < end_time:
            cycle_count += 1
            logger.info(f"Monitoring cycle {cycle_count}")

            # Collect metrics
            metrics = await self.collect_metrics()

            # Check thresholds and generate alerts
            new_alerts = self.check_thresholds(metrics)
            self.alerts.extend(new_alerts)

            # Log any new alerts
            for alert in new_alerts:
                if alert["level"] == "CRITICAL":
                    logger.critical(alert["message"])
                elif alert["level"] == "WARNING":
                    logger.warning(alert["message"])

            # Print current status
            if "system" in metrics:
                system = metrics["system"]
                print(
                    f"Cycle {cycle_count}: CPU={system.get('cpu_percent', 0):.1f}% "
                    f"Memory={system.get('memory_percent', 0):.1f}% "
                    f"Disk={system.get('disk_percent', 0):.1f}% "
                    f"Alerts={len(new_alerts)}"
                )

            # Wait for next cycle
            await asyncio.sleep(interval_seconds)

        logger.info(f"Monitoring completed. Total cycles: {cycle_count}")
        return self.generate_health_report()

    async def validate_alerting_system(self) -> dict[str, Any]:
        """Validate that alerting system works correctly."""
        logger.info("Validating alerting system...")

        validation_results = {"timestamp": datetime.now().isoformat(), "tests": {}}

        # Test 1: Threshold detection
        test_metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": 90.0,  # Should trigger warning
                "memory_percent": 96.0,  # Should trigger critical
                "disk_percent": 50.0,  # Should be fine
            },
            "application": {
                "test_component": {
                    "healthy": True,
                    "response_time_ms": 2000.0,  # Should trigger warning
                }
            },
        }

        alerts = self.check_thresholds(test_metrics)

        validation_results["tests"]["threshold_detection"] = {
            "status": "PASSED"
            if len(alerts) >= 3
            else "FAILED",  # Expecting CPU warning, memory critical, response time warning
            "alerts_generated": len(alerts),
            "alert_details": alerts,
        }

        # Test 2: Health status calculation
        original_alerts = self.alerts.copy()
        self.alerts = alerts  # Temporarily set test alerts

        health_report = self.generate_health_report()
        expected_status = "CRITICAL"  # Due to memory critical alert

        validation_results["tests"]["health_status_calculation"] = {
            "status": "PASSED"
            if health_report.get("health_status") == expected_status
            else "FAILED",
            "expected_status": expected_status,
            "actual_status": health_report.get("health_status", "UNKNOWN"),
        }

        # Restore original alerts
        self.alerts = original_alerts

        # Test 3: Metrics collection
        try:
            metrics = await self.collect_metrics()
            metrics_valid = "system" in metrics and "timestamp" in metrics

            validation_results["tests"]["metrics_collection"] = {
                "status": "PASSED" if metrics_valid else "FAILED",
                "metrics_collected": len(metrics.keys())
                if isinstance(metrics, dict)
                else 0,
            }
        except Exception as e:
            validation_results["tests"]["metrics_collection"] = {
                "status": "FAILED",
                "error": str(e),
            }

        # Calculate overall validation status
        passed_tests = sum(
            1
            for test in validation_results["tests"].values()
            if test.get("status") == "PASSED"
        )
        total_tests = len(validation_results["tests"])

        validation_results["summary"] = {
            "overall_status": "PASSED" if passed_tests == total_tests else "FAILED",
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
        }

        logger.info(
            f"Alerting validation completed: {passed_tests}/{total_tests} tests passed"
        )
        return validation_results


async def main():
    """Main health monitoring validation."""
    print("🏥 System Health Monitor & Alerting Validation")
    print("=" * 60)

    monitor = SystemHealthMonitor()

    # Run validation tests
    print("\n🧪 Validating alerting system...")
    alerting_validation = await monitor.validate_alerting_system()

    print(f"Alerting System: {alerting_validation['summary']['overall_status']}")
    print(
        f"Tests Passed: {alerting_validation['summary']['passed_tests']}/{alerting_validation['summary']['total_tests']}"
    )

    # Run short monitoring cycle
    print("\n📊 Running health monitoring cycle...")
    health_report = await monitor.run_monitoring_cycle(
        duration_seconds=30, interval_seconds=5
    )

    print(f"\nHealth Status: {health_report['health_status']}")
    print(f"Uptime: {health_report['uptime_seconds']:.1f} seconds")
    print(f"Total Alerts: {health_report['alerts']['total']}")

    # Print current system metrics
    if (
        "current_metrics" in health_report
        and "system" in health_report["current_metrics"]
    ):
        system = health_report["current_metrics"]["system"]
        print(f"CPU: {system.get('cpu_percent', 0):.1f}%")
        print(f"Memory: {system.get('memory_percent', 0):.1f}%")
        print(f"Disk: {system.get('disk_percent', 0):.1f}%")

    # Save reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    alerting_file = f"alerting_validation_{timestamp}.json"
    with open(alerting_file, "w") as f:
        json.dump(alerting_validation, f, indent=2, default=str)

    health_file = f"health_report_{timestamp}.json"
    with open(health_file, "w") as f:
        json.dump(health_report, f, indent=2, default=str)

    print("\n💾 Reports saved:")
    print(f"   Alerting validation: {alerting_file}")
    print(f"   Health report: {health_file}")

    # Final validation status
    overall_success = alerting_validation["summary"][
        "overall_status"
    ] == "PASSED" and health_report["health_status"] in ["HEALTHY", "MODERATE"]

    if overall_success:
        print("\n✅ System health monitoring and alerting validation PASSED")
    else:
        print("\n❌ System health monitoring and alerting validation FAILED")

    return {
        "alerting_validation": alerting_validation,
        "health_report": health_report,
        "overall_success": overall_success,
    }


if __name__ == "__main__":
    result = asyncio.run(main())
