"""
MLX GPU Memory Monitor and Leak Detection System

Real-time monitoring of MLX memory usage with:
- Memory leak detection
- Performance alerts
- Automatic cleanup triggers
- Memory usage trends
- GPU utilization tracking
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import psutil

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

import contextlib

from .mlx_memory_manager import get_mlx_memory_manager

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""

    timestamp: float
    mlx_allocated_mb: float
    system_memory_mb: float
    system_memory_percent: float
    gpu_arrays_count: int
    pooled_arrays_count: int
    gc_runs: int
    metal_clears: int
    operation_count: int


@dataclass
class MemoryAlert:
    """Memory usage alert."""

    timestamp: float
    alert_type: str
    severity: str  # "warning", "critical"
    message: str
    memory_mb: float
    threshold_mb: float
    recommended_action: str


class MLXMemoryMonitor:
    """Real-time memory monitoring and leak detection for MLX."""

    def __init__(
        self,
        monitoring_interval: float = 5.0,
        memory_threshold_mb: float = 3072,  # 3GB warning threshold
        critical_threshold_mb: float = 4096,  # 4GB critical threshold
        history_duration_hours: int = 24,
    ):
        self.monitoring_interval = monitoring_interval
        self.memory_threshold_mb = memory_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.history_duration = timedelta(hours=history_duration_hours)

        # Monitoring state
        self.monitoring_active = False
        self.monitor_task = None

        # Data storage
        self.memory_history: list[MemorySnapshot] = []
        self.alerts: list[MemoryAlert] = []
        self.performance_metrics = {
            "peak_memory_mb": 0.0,
            "average_memory_mb": 0.0,
            "memory_growth_rate_mb_per_hour": 0.0,
            "gc_efficiency": 0.0,
            "leak_score": 0.0,
        }

        # Alert thresholds
        self.alert_thresholds = {
            "memory_growth_rate_mb_per_hour": 100.0,  # 100MB/hour growth
            "gc_efficiency_threshold": 0.3,  # 30% minimum efficiency
            "leak_score_threshold": 0.7,  # 70% leak probability
        }

        logger.info(
            f"MLX Memory Monitor initialized: {memory_threshold_mb}MB warning, {critical_threshold_mb}MB critical"
        )

    async def start_monitoring(self):
        """Start continuous memory monitoring."""
        if not MLX_AVAILABLE:
            logger.warning("MLX not available - memory monitoring disabled")
            return

        if self.monitoring_active:
            logger.warning("Memory monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started MLX memory monitoring")

    async def stop_monitoring(self):
        """Stop memory monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitor_task

        logger.info("Stopped MLX memory monitoring")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Take memory snapshot
                snapshot = await self._take_memory_snapshot()
                self.memory_history.append(snapshot)

                # Clean old history
                self._cleanup_old_history()

                # Check for alerts
                await self._check_for_alerts(snapshot)

                # Update performance metrics
                self._update_performance_metrics()

                # Log status periodically
                if len(self.memory_history) % 12 == 0:  # Every minute with 5s interval
                    self._log_status(snapshot)

                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)

    async def _take_memory_snapshot(self) -> MemorySnapshot:
        """Take a snapshot of current memory usage."""
        memory_manager = get_mlx_memory_manager()
        stats = memory_manager.get_memory_stats()

        # System memory
        system_memory = psutil.virtual_memory()

        return MemorySnapshot(
            timestamp=time.time(),
            mlx_allocated_mb=stats.total_allocated_mb,
            system_memory_mb=system_memory.used / (1024 * 1024),
            system_memory_percent=system_memory.percent,
            gpu_arrays_count=stats.current_arrays,
            pooled_arrays_count=stats.pooled_arrays,
            gc_runs=stats.gc_runs,
            metal_clears=stats.metal_clears,
            operation_count=memory_manager.operation_count,
        )

    def _cleanup_old_history(self):
        """Remove old history entries."""
        cutoff_time = time.time() - self.history_duration.total_seconds()
        self.memory_history = [
            s for s in self.memory_history if s.timestamp >= cutoff_time
        ]
        self.alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]

    async def _check_for_alerts(self, snapshot: MemorySnapshot):
        """Check for memory-related alerts."""
        alerts = []

        # Memory threshold alerts
        if snapshot.mlx_allocated_mb >= self.critical_threshold_mb:
            alerts.append(
                MemoryAlert(
                    timestamp=snapshot.timestamp,
                    alert_type="critical_memory",
                    severity="critical",
                    message=f"MLX memory usage critical: {snapshot.mlx_allocated_mb:.1f}MB",
                    memory_mb=snapshot.mlx_allocated_mb,
                    threshold_mb=self.critical_threshold_mb,
                    recommended_action="Force cleanup and reduce batch sizes",
                )
            )
        elif snapshot.mlx_allocated_mb >= self.memory_threshold_mb:
            alerts.append(
                MemoryAlert(
                    timestamp=snapshot.timestamp,
                    alert_type="high_memory",
                    severity="warning",
                    message=f"MLX memory usage high: {snapshot.mlx_allocated_mb:.1f}MB",
                    memory_mb=snapshot.mlx_allocated_mb,
                    threshold_mb=self.memory_threshold_mb,
                    recommended_action="Consider cleanup or reduce operations",
                )
            )

        # Memory leak detection
        leak_score = self._calculate_leak_score()
        if leak_score >= self.alert_thresholds["leak_score_threshold"]:
            alerts.append(
                MemoryAlert(
                    timestamp=snapshot.timestamp,
                    alert_type="memory_leak",
                    severity="warning" if leak_score < 0.9 else "critical",
                    message=f"Potential memory leak detected (score: {leak_score:.2f})",
                    memory_mb=snapshot.mlx_allocated_mb,
                    threshold_mb=0,
                    recommended_action="Investigate memory usage patterns and force cleanup",
                )
            )

        # Memory growth rate alert
        growth_rate = self._calculate_memory_growth_rate()
        if growth_rate >= self.alert_thresholds["memory_growth_rate_mb_per_hour"]:
            alerts.append(
                MemoryAlert(
                    timestamp=snapshot.timestamp,
                    alert_type="memory_growth",
                    severity="warning",
                    message=f"High memory growth rate: {growth_rate:.1f}MB/hour",
                    memory_mb=snapshot.mlx_allocated_mb,
                    threshold_mb=0,
                    recommended_action="Monitor for leaks and optimize memory usage",
                )
            )

        # GC efficiency alert
        gc_efficiency = self._calculate_gc_efficiency()
        if gc_efficiency < self.alert_thresholds["gc_efficiency_threshold"]:
            alerts.append(
                MemoryAlert(
                    timestamp=snapshot.timestamp,
                    alert_type="gc_inefficiency",
                    severity="warning",
                    message=f"Low GC efficiency: {gc_efficiency:.1%}",
                    memory_mb=snapshot.mlx_allocated_mb,
                    threshold_mb=0,
                    recommended_action="Force manual cleanup and check for reference cycles",
                )
            )

        # Process alerts
        for alert in alerts:
            self.alerts.append(alert)
            logger.log(
                logging.CRITICAL if alert.severity == "critical" else logging.WARNING,
                f"MEMORY ALERT [{alert.alert_type}]: {alert.message} - {alert.recommended_action}",
            )

            # Trigger automatic cleanup for critical alerts
            if alert.severity == "critical":
                await self._trigger_emergency_cleanup()

    async def _trigger_emergency_cleanup(self):
        """Trigger emergency memory cleanup."""
        logger.critical("Triggering emergency memory cleanup")

        try:
            memory_manager = get_mlx_memory_manager()
            await memory_manager.cleanup_memory(force=True)

            # Wait a bit and take another snapshot
            await asyncio.sleep(2)
            snapshot = await self._take_memory_snapshot()

            logger.info(
                f"Emergency cleanup complete. Memory usage: {snapshot.mlx_allocated_mb:.1f}MB"
            )

        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")

    def _calculate_leak_score(self) -> float:
        """Calculate memory leak probability score (0-1)."""
        if len(self.memory_history) < 10:
            return 0.0

        # Look at recent memory trends
        recent_snapshots = self.memory_history[-10:]

        # Calculate memory growth without corresponding GC
        memory_values = [s.mlx_allocated_mb for s in recent_snapshots]
        gc_counts = [s.gc_runs for s in recent_snapshots]

        # Memory trend
        memory_trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]

        # GC frequency
        gc_frequency = (
            (gc_counts[-1] - gc_counts[0]) / len(gc_counts) if len(gc_counts) > 1 else 0
        )

        # High memory growth with low GC frequency indicates potential leak
        leak_score = min(
            1.0, max(0.0, (memory_trend / 100) * (1 - min(1.0, gc_frequency)))
        )

        return leak_score

    def _calculate_memory_growth_rate(self) -> float:
        """Calculate memory growth rate in MB/hour."""
        if len(self.memory_history) < 2:
            return 0.0

        # Use recent history (last hour or available data)
        hour_ago = time.time() - 3600
        recent_snapshots = [s for s in self.memory_history if s.timestamp >= hour_ago]

        if len(recent_snapshots) < 2:
            recent_snapshots = self.memory_history[
                -min(12, len(self.memory_history)) :
            ]  # Last minute

        if len(recent_snapshots) < 2:
            return 0.0

        # Calculate growth rate
        time_span_hours = (
            recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp
        ) / 3600
        memory_growth = (
            recent_snapshots[-1].mlx_allocated_mb - recent_snapshots[0].mlx_allocated_mb
        )

        return memory_growth / max(0.1, time_span_hours)  # Avoid division by zero

    def _calculate_gc_efficiency(self) -> float:
        """Calculate garbage collection efficiency."""
        if len(self.memory_history) < 5:
            return 1.0

        recent_snapshots = self.memory_history[-5:]

        # Look for memory reductions after GC runs
        total_gc_runs = 0
        effective_cleanups = 0

        for i in range(1, len(recent_snapshots)):
            prev_snapshot = recent_snapshots[i - 1]
            curr_snapshot = recent_snapshots[i]

            if curr_snapshot.gc_runs > prev_snapshot.gc_runs:
                total_gc_runs += 1
                # Check if memory was reduced
                if curr_snapshot.mlx_allocated_mb < prev_snapshot.mlx_allocated_mb:
                    effective_cleanups += 1

        return effective_cleanups / max(1, total_gc_runs)

    def _update_performance_metrics(self):
        """Update performance metrics based on history."""
        if not self.memory_history:
            return

        memory_values = [s.mlx_allocated_mb for s in self.memory_history]

        self.performance_metrics.update(
            {
                "peak_memory_mb": max(memory_values),
                "average_memory_mb": np.mean(memory_values),
                "memory_growth_rate_mb_per_hour": self._calculate_memory_growth_rate(),
                "gc_efficiency": self._calculate_gc_efficiency(),
                "leak_score": self._calculate_leak_score(),
            }
        )

    def _log_status(self, snapshot: MemorySnapshot):
        """Log current monitoring status."""
        logger.info(
            f"MLX Memory Status: {snapshot.mlx_allocated_mb:.1f}MB allocated, "
            f"{snapshot.gpu_arrays_count} arrays, "
            f"Growth: {self.performance_metrics['memory_growth_rate_mb_per_hour']:.1f}MB/h, "
            f"Leak Score: {self.performance_metrics['leak_score']:.2f}"
        )

    def get_current_stats(self) -> dict[str, Any]:
        """Get current monitoring statistics."""
        latest_snapshot = self.memory_history[-1] if self.memory_history else None

        return {
            "monitoring_active": self.monitoring_active,
            "latest_snapshot": asdict(latest_snapshot) if latest_snapshot else None,
            "performance_metrics": self.performance_metrics,
            "total_alerts": len(self.alerts),
            "critical_alerts": len(
                [a for a in self.alerts if a.severity == "critical"]
            ),
            "warning_alerts": len([a for a in self.alerts if a.severity == "warning"]),
            "history_points": len(self.memory_history),
        }

    def save_report(self, filepath: Path):
        """Save monitoring report to file."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_config": {
                "interval_seconds": self.monitoring_interval,
                "memory_threshold_mb": self.memory_threshold_mb,
                "critical_threshold_mb": self.critical_threshold_mb,
            },
            "performance_metrics": self.performance_metrics,
            "recent_alerts": [asdict(alert) for alert in self.alerts[-10:]],
            "memory_history": [
                asdict(snapshot) for snapshot in self.memory_history[-100:]
            ],
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Memory monitoring report saved to {filepath}")

    def plot_memory_usage(self, save_path: Path | None = None):
        """Plot memory usage over time."""
        if not self.memory_history:
            logger.warning("No memory history to plot")
            return

        timestamps = [datetime.fromtimestamp(s.timestamp) for s in self.memory_history]
        memory_values = [s.mlx_allocated_mb for s in self.memory_history]
        array_counts = [s.gpu_arrays_count for s in self.memory_history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Memory usage plot
        ax1.plot(timestamps, memory_values, "b-", linewidth=2, label="MLX Memory")
        ax1.axhline(
            y=self.memory_threshold_mb,
            color="orange",
            linestyle="--",
            label="Warning Threshold",
        )
        ax1.axhline(
            y=self.critical_threshold_mb,
            color="red",
            linestyle="--",
            label="Critical Threshold",
        )
        ax1.set_ylabel("Memory (MB)")
        ax1.set_title("MLX GPU Memory Usage Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Array count plot
        ax2.plot(timestamps, array_counts, "g-", linewidth=2, label="GPU Arrays")
        ax2.set_ylabel("Array Count")
        ax2.set_xlabel("Time")
        ax2.set_title("GPU Array Count Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Mark alerts
        for alert in self.alerts:
            alert_time = datetime.fromtimestamp(alert.timestamp)
            color = "red" if alert.severity == "critical" else "orange"
            ax1.axvline(x=alert_time, color=color, alpha=0.5, linestyle=":")
            ax2.axvline(x=alert_time, color=color, alpha=0.5, linestyle=":")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Memory usage plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_monitoring()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_monitoring()


# Global monitor instance
_memory_monitor: MLXMemoryMonitor | None = None


def get_memory_monitor() -> MLXMemoryMonitor:
    """Get or create the global memory monitor."""
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = MLXMemoryMonitor()
    return _memory_monitor


async def start_global_monitoring():
    """Start global memory monitoring."""
    monitor = get_memory_monitor()
    await monitor.start_monitoring()
    return monitor


async def stop_global_monitoring():
    """Stop global memory monitoring."""
    monitor = get_memory_monitor()
    await monitor.stop_monitoring()


if __name__ == "__main__":
    # Test the memory monitor
    async def test_monitor():
        print("Testing MLX Memory Monitor...")

        async with MLXMemoryMonitor(monitoring_interval=1.0) as monitor:
            # Simulate some memory usage
            if MLX_AVAILABLE:
                arrays = []
                for i in range(20):
                    # Create some arrays
                    array = mx.random.normal([1000, 1000])
                    mx.eval(array)
                    arrays.append(array)

                    await asyncio.sleep(0.5)

                    # Clean up some arrays periodically
                    if i % 5 == 0 and arrays:
                        del arrays[:2]
                        arrays = arrays[2:]
                        import gc

                        gc.collect()

                # Generate report
                monitor.save_report(Path("memory_monitor_test_report.json"))
                print("Test completed - check memory_monitor_test_report.json")
            else:
                print("MLX not available - monitoring test skipped")
                await asyncio.sleep(5)

    asyncio.run(test_monitor())
