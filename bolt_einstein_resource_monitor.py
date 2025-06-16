#!/usr/bin/env python3
"""
Resource Monitor for Bolt-Einstein Integration

Monitors resource usage and coordinates between systems to prevent conflicts.
"""

import asyncio
import logging
import time
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor and coordinate resource usage between Bolt and Einstein"""

    def __init__(self):
        self.monitoring = False
        self.resource_history = []
        self.alerts = []
        self.thresholds = {
            "memory_warning": 85.0,  # %
            "memory_critical": 95.0,  # %
            "cpu_warning": 80.0,  # %
            "cpu_critical": 95.0,  # %
        }

    async def start_monitoring(self, interval: float = 5.0):
        """Start resource monitoring"""
        self.monitoring = True
        logger.info("Resource monitoring started")

        while self.monitoring:
            try:
                # Collect resource stats
                stats = self._collect_stats()
                self.resource_history.append(stats)

                # Keep only last 100 readings
                if len(self.resource_history) > 100:
                    self.resource_history.pop(0)

                # Check for alerts
                self._check_alerts(stats)

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(interval)

    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        logger.info("Resource monitoring stopped")

    def _collect_stats(self) -> dict[str, Any]:
        """Collect current resource statistics"""
        return {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "disk_usage_percent": psutil.disk_usage("/").percent,
            "process_count": len(psutil.pids()),
        }

    def _check_alerts(self, stats: dict[str, Any]):
        """Check for resource alerts"""
        alerts = []

        # Memory alerts
        if stats["memory_percent"] > self.thresholds["memory_critical"]:
            alerts.append(f"CRITICAL: Memory usage {stats['memory_percent']:.1f}%")
        elif stats["memory_percent"] > self.thresholds["memory_warning"]:
            alerts.append(f"WARNING: Memory usage {stats['memory_percent']:.1f}%")

        # CPU alerts
        if stats["cpu_percent"] > self.thresholds["cpu_critical"]:
            alerts.append(f"CRITICAL: CPU usage {stats['cpu_percent']:.1f}%")
        elif stats["cpu_percent"] > self.thresholds["cpu_warning"]:
            alerts.append(f"WARNING: CPU usage {stats['cpu_percent']:.1f}%")

        # Log new alerts
        for alert in alerts:
            if alert not in self.alerts:
                logger.warning(alert)
                self.alerts.append(alert)

        # Clear old alerts
        self.alerts = alerts

    def get_current_stats(self) -> dict[str, Any] | None:
        """Get current resource statistics"""
        if self.resource_history:
            return self.resource_history[-1]
        return None

    def get_resource_trend(self, minutes: int = 5) -> dict[str, Any]:
        """Get resource usage trend over specified minutes"""
        if not self.resource_history:
            return {}

        cutoff_time = time.time() - (minutes * 60)
        recent_stats = [
            s for s in self.resource_history if s["timestamp"] > cutoff_time
        ]

        if not recent_stats:
            return {}

        # Calculate trends
        memory_values = [s["memory_percent"] for s in recent_stats]
        cpu_values = [s["cpu_percent"] for s in recent_stats]

        return {
            "memory_trend": {
                "min": min(memory_values),
                "max": max(memory_values),
                "avg": sum(memory_values) / len(memory_values),
                "current": memory_values[-1] if memory_values else 0,
            },
            "cpu_trend": {
                "min": min(cpu_values),
                "max": max(cpu_values),
                "avg": sum(cpu_values) / len(cpu_values),
                "current": cpu_values[-1] if cpu_values else 0,
            },
            "sample_count": len(recent_stats),
            "time_range_minutes": minutes,
        }


async def main():
    """Test the resource monitor"""
    monitor = ResourceMonitor()

    # Start monitoring
    monitor_task = asyncio.create_task(monitor.start_monitoring(interval=2.0))

    # Let it run for a bit
    await asyncio.sleep(10)

    # Check stats
    current = monitor.get_current_stats()
    if current:
        print(f"Current stats: {current}")

    trend = monitor.get_resource_trend(minutes=1)
    if trend:
        print(f"Trend: {trend}")

    # Stop monitoring
    monitor.stop_monitoring()
    await monitor_task


if __name__ == "__main__":
    asyncio.run(main())
