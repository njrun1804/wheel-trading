"""
Production-ready fallback MetalMonitor implementation.

This module provides a comprehensive fallback implementation for when
the main MetalMonitor cannot be imported.
"""

import asyncio
import logging
import time
from dataclasses import dataclass

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FallbackMetalStats:
    """Fallback Metal GPU statistics using system metrics."""

    memory_used_gb: float
    memory_total_gb: float
    utilization_percent: float
    cores_active: int
    cores_total: int
    temperature_c: float
    timestamp: float

    @property
    def memory_available_gb(self) -> float:
        return self.memory_total_gb - self.memory_used_gb

    @property
    def memory_utilization_percent(self) -> float:
        if self.memory_total_gb == 0:
            return 0.0
        return (self.memory_used_gb / self.memory_total_gb) * 100


class FallbackMetalMonitor:
    """Production fallback MetalMonitor using system metrics."""

    def __init__(self):
        self.started = False
        self.stats_history = []
        self.max_history = 300
        self.gpu_cores = 20  # M4 Pro estimate
        self.memory_limit_gb = 18.0  # Conservative unified memory estimate
        self._monitor_task = None
        logger.info("🔥 MetalMonitor initialized (fallback mode)")

    async def start(self):
        """Start monitoring using system metrics."""
        if self.started:
            logger.warning("MetalMonitor already started")
            return

        self.started = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("📊 MetalMonitor started (fallback mode)")

    async def stop(self):
        """Stop monitoring."""
        self.started = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("📊 MetalMonitor stopped")

    async def _monitor_loop(self):
        """Background monitoring loop using system metrics."""
        while self.started:
            try:
                stats = await self._collect_fallback_stats()
                self.stats_history.append(stats)

                if len(self.stats_history) > self.max_history:
                    self.stats_history.pop(0)

                if stats.memory_utilization_percent > 85:
                    logger.warning(
                        f"High memory usage: {stats.memory_utilization_percent:.1f}%"
                    )

                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Fallback monitoring error: {e}")
                await asyncio.sleep(5.0)

    async def _collect_fallback_stats(self) -> FallbackMetalStats:
        """Collect system metrics as fallback for Metal stats."""
        timestamp = time.time()

        # Default values
        memory_used_gb = 0.0
        memory_total_gb = self.memory_limit_gb
        utilization_percent = 0.0
        cores_active = 0
        temperature_c = 0.0

        if PSUTIL_AVAILABLE:
            try:
                # Get system memory info
                memory = psutil.virtual_memory()

                # Estimate GPU memory usage from system metrics
                memory_used_gb = min(
                    memory.used / (1024**3) * 0.2, self.memory_limit_gb * 0.8
                )

                # Estimate GPU utilization from CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                utilization_percent = min(
                    cpu_percent * 0.6, 100
                )  # Conservative estimate
                cores_active = int(utilization_percent / 100 * self.gpu_cores)

                # Get system temperature if available
                try:
                    if hasattr(psutil, "sensors_temperatures"):
                        temps = psutil.sensors_temperatures()
                        if temps:
                            temp_values = [
                                temp.current
                                for sensors in temps.values()
                                for temp in sensors
                            ]
                            temperature_c = (
                                sum(temp_values) / len(temp_values)
                                if temp_values
                                else 0.0
                            )
                except Exception:
                    pass

            except Exception as e:
                logger.debug(f"Failed to collect psutil stats: {e}")
        else:
            # Minimal fallback without psutil
            import os

            try:
                # Try to get load average as CPU estimate
                load_avg = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.5
                utilization_percent = min(load_avg * 20, 100)  # Very rough estimate
                cores_active = int(utilization_percent / 100 * self.gpu_cores)
            except Exception:
                pass

        return FallbackMetalStats(
            memory_used_gb=round(memory_used_gb, 2),
            memory_total_gb=memory_total_gb,
            utilization_percent=round(utilization_percent, 1),
            cores_active=cores_active,
            cores_total=self.gpu_cores,
            temperature_c=round(temperature_c, 1),
            timestamp=timestamp,
        )

    async def get_stats(self):
        """Get latest monitoring stats."""
        if not self.stats_history:
            return await self._collect_fallback_stats()
        return self.stats_history[-1]

    async def get_average_stats(self, seconds: int = 30):
        """Get average statistics over specified time period."""
        if not self.stats_history:
            return None

        cutoff_time = time.time() - seconds
        recent_stats = [s for s in self.stats_history if s.timestamp > cutoff_time]

        if not recent_stats:
            return self.stats_history[-1] if self.stats_history else None

        avg_memory_used = sum(s.memory_used_gb for s in recent_stats) / len(
            recent_stats
        )
        avg_utilization = sum(s.utilization_percent for s in recent_stats) / len(
            recent_stats
        )
        avg_cores_active = sum(s.cores_active for s in recent_stats) / len(recent_stats)
        avg_temperature = sum(s.temperature_c for s in recent_stats) / len(recent_stats)

        return FallbackMetalStats(
            memory_used_gb=round(avg_memory_used, 2),
            memory_total_gb=self.memory_limit_gb,
            utilization_percent=round(avg_utilization, 1),
            cores_active=int(avg_cores_active),
            cores_total=self.gpu_cores,
            temperature_c=round(avg_temperature, 1),
            timestamp=time.time(),
        )

    def get_performance_summary(self):
        """Get performance summary."""
        if not self.stats_history:
            return {"error": "No statistics available (fallback mode)"}

        recent_stats = self.stats_history[-60:]

        return {
            "gpu_model": f"Apple M4 Pro (fallback estimation, {self.gpu_cores}-core)",
            "memory_limit_gb": self.memory_limit_gb,
            "current_stats": self.stats_history[-1].__dict__,
            "averages": {
                "memory_used_gb": sum(s.memory_used_gb for s in recent_stats)
                / len(recent_stats),
                "utilization_percent": sum(s.utilization_percent for s in recent_stats)
                / len(recent_stats),
                "cores_active": sum(s.cores_active for s in recent_stats)
                / len(recent_stats),
                "temperature_c": sum(s.temperature_c for s in recent_stats)
                / len(recent_stats),
            },
            "peaks": {
                "max_memory_gb": max(s.memory_used_gb for s in recent_stats),
                "max_utilization": max(s.utilization_percent for s in recent_stats),
                "max_temperature": max(s.temperature_c for s in recent_stats),
            },
            "samples": len(recent_stats),
            "monitoring_duration": recent_stats[-1].timestamp
            - recent_stats[0].timestamp
            if len(recent_stats) > 1
            else 0,
            "mode": "fallback",
        }
