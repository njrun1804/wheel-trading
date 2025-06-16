#!/usr/bin/env python3
"""
CPU Monitor for M4 Pro Performance Debugging
Monitors CPU usage and provides emergency throttling recommendations.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime

import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CPUSnapshot:
    """Single CPU usage snapshot."""

    timestamp: float
    cpu_percent: float
    cpu_per_core: list[float]
    memory_percent: float
    memory_available_gb: float
    active_processes: int
    top_processes: list[tuple]  # (name, cpu_percent, memory_mb)


class CPUMonitor:
    """Real-time CPU monitoring for M4 Pro debugging."""

    def __init__(
        self, warning_threshold: float = 80.0, emergency_threshold: float = 90.0
    ):
        self.warning_threshold = warning_threshold
        self.emergency_threshold = emergency_threshold
        self.snapshots = []
        self.max_snapshots = 300  # 5 minutes at 1-second intervals
        self.monitoring = False
        self.last_warning = 0
        self.last_emergency = 0

    async def start_monitoring(self, interval: float = 1.0):
        """Start continuous CPU monitoring."""
        self.monitoring = True
        logger.info("🔍 Starting CPU monitoring (M4 Pro optimized)")
        logger.info(f"   Warning threshold: {self.warning_threshold}%")
        logger.info(f"   Emergency threshold: {self.emergency_threshold}%")

        try:
            while self.monitoring:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)

                # Keep only recent snapshots
                if len(self.snapshots) > self.max_snapshots:
                    self.snapshots.pop(0)

                # Check for issues
                await self._check_cpu_health(snapshot)

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            logger.info("CPU monitoring stopped by user")
        except Exception as e:
            logger.error(f"CPU monitoring error: {e}")
        finally:
            self.monitoring = False

    def _take_snapshot(self) -> CPUSnapshot:
        """Take a CPU usage snapshot."""
        # Get CPU information
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)

        # Get memory information
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)

        # Get process information
        processes = []
        try:
            for proc in psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_info"]
            ):
                try:
                    proc_info = proc.info
                    if proc_info["cpu_percent"] and proc_info["cpu_percent"] > 1.0:
                        memory_mb = (
                            proc_info["memory_info"].rss / (1024**2)
                            if proc_info["memory_info"]
                            else 0
                        )
                        processes.append(
                            (proc_info["name"], proc_info["cpu_percent"], memory_mb)
                        )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.debug(f"Error getting process info: {e}")
            processes = []

        # Sort by CPU usage and take top 5
        processes.sort(key=lambda x: x[1], reverse=True)
        top_processes = processes[:5]

        return CPUSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            cpu_per_core=cpu_per_core,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            active_processes=len(processes),
            top_processes=top_processes,
        )

    async def _check_cpu_health(self, snapshot: CPUSnapshot):
        """Check CPU health and issue warnings."""
        now = time.time()

        # Emergency threshold
        if snapshot.cpu_percent >= self.emergency_threshold:
            if now - self.last_emergency > 30:  # Throttle emergency messages
                logger.error(f"🚨 EMERGENCY: CPU at {snapshot.cpu_percent:.1f}%!")
                logger.error(f"   Memory: {snapshot.memory_percent:.1f}%")
                logger.error(f"   Available RAM: {snapshot.memory_available_gb:.1f}GB")
                logger.error("   Top CPU processes:")
                for name, cpu, mem in snapshot.top_processes[:3]:
                    logger.error(f"     • {name}: {cpu:.1f}% CPU, {mem:.1f}MB RAM")

                # Emergency recommendations
                logger.error("🆘 EMERGENCY ACTIONS:")
                logger.error("   1. Kill non-essential processes")
                logger.error("   2. Reduce Einstein concurrency limits")
                logger.error("   3. Stop background file scanning")
                logger.error("   4. Consider system restart if persistent")

                self.last_emergency = now

        # Warning threshold
        elif snapshot.cpu_percent >= self.warning_threshold:
            if now - self.last_warning > 10:  # Throttle warning messages
                logger.warning(f"⚠️ HIGH CPU: {snapshot.cpu_percent:.1f}%")
                logger.warning(f"   Memory: {snapshot.memory_percent:.1f}%")
                logger.warning("   Top CPU processes:")
                for name, cpu, mem in snapshot.top_processes[:3]:
                    logger.warning(f"     • {name}: {cpu:.1f}% CPU, {mem:.1f}MB RAM")

                self.last_warning = now

    def get_cpu_summary(self) -> dict:
        """Get CPU usage summary."""
        if not self.snapshots:
            return {}

        recent_snapshots = self.snapshots[-60:]  # Last minute

        avg_cpu = sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots)
        max_cpu = max(s.cpu_percent for s in recent_snapshots)
        min_cpu = min(s.cpu_percent for s in recent_snapshots)

        avg_memory = sum(s.memory_percent for s in recent_snapshots) / len(
            recent_snapshots
        )

        # M4 Pro specific core analysis
        p_cores_usage = []
        e_cores_usage = []

        if recent_snapshots and recent_snapshots[-1].cpu_per_core:
            cores = recent_snapshots[-1].cpu_per_core
            # M4 Pro: first 8 cores are P-cores, last 4 are E-cores
            p_cores_usage = cores[:8] if len(cores) >= 8 else cores
            e_cores_usage = cores[8:] if len(cores) > 8 else []

        return {
            "monitoring_duration_minutes": (time.time() - self.snapshots[0].timestamp)
            / 60,
            "samples_collected": len(self.snapshots),
            "cpu_average": avg_cpu,
            "cpu_maximum": max_cpu,
            "cpu_minimum": min_cpu,
            "memory_average": avg_memory,
            "p_cores_usage": p_cores_usage,
            "e_cores_usage": e_cores_usage,
            "p_cores_average": sum(p_cores_usage) / len(p_cores_usage)
            if p_cores_usage
            else 0,
            "e_cores_average": sum(e_cores_usage) / len(e_cores_usage)
            if e_cores_usage
            else 0,
            "emergency_events": sum(
                1 for s in recent_snapshots if s.cpu_percent >= self.emergency_threshold
            ),
            "warning_events": sum(
                1 for s in recent_snapshots if s.cpu_percent >= self.warning_threshold
            ),
        }

    def stop_monitoring(self):
        """Stop CPU monitoring."""
        self.monitoring = False
        logger.info("CPU monitoring stopped")

    def save_report(self, filename: str = None):
        """Save detailed CPU report."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cpu_report_{timestamp}.json"

        summary = self.get_cpu_summary()

        import json

        report = {
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "platform": "M4 Pro" if psutil.cpu_count() == 12 else "Unknown",
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            },
            "monitoring_summary": summary,
            "recommendations": self._generate_recommendations(summary),
        }

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"📊 CPU report saved to {filename}")
        return filename

    def _generate_recommendations(self, summary: dict) -> list[str]:
        """Generate performance recommendations."""
        recommendations = []

        if summary.get("cpu_average", 0) > 70:
            recommendations.append("🔧 Reduce Einstein concurrency limits")
            recommendations.append(
                "🔧 Disable background file scanning during peak usage"
            )

        if summary.get("p_cores_average", 0) > 80:
            recommendations.append("⚡ P-cores overloaded - reduce thread pool workers")

        if summary.get("emergency_events", 0) > 0:
            recommendations.append(
                "🚨 Emergency events detected - system needs immediate tuning"
            )

        if summary.get("warning_events", 0) > 10:
            recommendations.append(
                "⚠️ Frequent high CPU - consider permanent limit reduction"
            )

        return recommendations


async def main():
    """Main monitoring function."""
    monitor = CPUMonitor(warning_threshold=75.0, emergency_threshold=90.0)

    try:
        await monitor.start_monitoring(interval=1.0)
    except KeyboardInterrupt:
        logger.info("Stopping CPU monitor...")
    finally:
        summary = monitor.get_cpu_summary()
        logger.info("📊 Final CPU Summary:")
        for key, value in summary.items():
            logger.info(f"   {key}: {value}")

        # Save report
        monitor.save_report()


if __name__ == "__main__":
    asyncio.run(main())
