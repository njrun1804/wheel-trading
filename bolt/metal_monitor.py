#!/usr/bin/env python3
"""
Metal GPU Monitor for M4 Pro
Real-time GPU utilization tracking and memory management
"""

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetalStats:
    """Metal GPU statistics."""

    memory_used_gb: float
    memory_total_gb: float
    utilization_percent: float
    cores_active: int
    cores_total: int
    temperature_c: float
    timestamp: float

    @property
    def memory_available_gb(self) -> float:
        """Available GPU memory in GB."""
        return self.memory_total_gb - self.memory_used_gb

    @property
    def memory_utilization_percent(self) -> float:
        """Memory utilization percentage."""
        if self.memory_total_gb == 0:
            return 0.0
        return (self.memory_used_gb / self.memory_total_gb) * 100


class MetalMonitor:
    """Real-time Metal GPU monitoring for M4 Pro."""

    def __init__(self):
        self.is_running = False
        self.stats_history = []
        self.max_history = 300  # 5 minutes at 1s intervals
        self._monitor_task: asyncio.Task | None = None

        # M4 Pro GPU specifications
        self.gpu_cores = 20  # M4 Pro default
        self.memory_limit_gb = 18.0  # Conservative estimate for unified memory

        logger.info("🔥 Metal Monitor initialized for M4 Pro")

    async def start(self):
        """Start GPU monitoring."""
        if self.is_running:
            logger.warning("Metal monitor already running")
            return

        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("📊 Metal GPU monitoring started")

    async def stop(self):
        """Stop GPU monitoring."""
        self.is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task
        logger.info("📊 Metal GPU monitoring stopped")

    async def _monitor_loop(self):
        """Background monitoring loop."""
        while self.is_running:
            try:
                stats = await self._collect_stats()
                self.stats_history.append(stats)

                # Limit history size
                if len(self.stats_history) > self.max_history:
                    self.stats_history.pop(0)

                # Log warnings for high utilization
                if stats.memory_utilization_percent > 85:
                    logger.warning(
                        f"High GPU memory usage: {stats.memory_utilization_percent:.1f}%"
                    )

                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Metal monitoring error: {e}")
                await asyncio.sleep(5.0)

    async def _collect_stats(self) -> MetalStats:
        """Collect current Metal GPU statistics."""
        timestamp = time.time()

        # Default values
        memory_used_gb = 0.0
        memory_total_gb = self.memory_limit_gb
        utilization_percent = 0.0
        cores_active = 0
        temperature_c = 0.0

        try:
            # Try to get system GPU info using system_profiler
            proc = await asyncio.create_subprocess_exec(
                "system_profiler",
                "SPDisplaysDataType",
                "-json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                data = json.loads(stdout.decode())
                displays = data.get("SPDisplaysDataType", [])

                if displays:
                    gpu_info = displays[0]

                    # Extract memory info if available
                    if "sppci_memory_unified" in gpu_info:
                        # This is unified memory system
                        memory_total_gb = self.memory_limit_gb

                    # Extract GPU name for core count detection
                    gpu_name = gpu_info.get("sppci_model", "")
                    if "M4" in gpu_name:
                        if "20-core" in gpu_name or "Max" in gpu_name:
                            self.gpu_cores = 20
                        else:
                            self.gpu_cores = 16

        except Exception as e:
            logger.debug(f"system_profiler GPU info failed: {e}")

        try:
            # Estimate GPU memory usage from system memory pressure
            # This is a heuristic since Metal doesn't expose direct memory usage
            proc = await asyncio.create_subprocess_exec(
                "vm_stat",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                vm_output = stdout.decode()

                # Parse memory pressure indicators
                for line in vm_output.split("\n"):
                    if "Pages wired down:" in line:
                        # Wired memory often indicates GPU usage
                        wired_pages = int(line.split()[-1].replace(".", ""))
                        # Assume 16KB pages and estimate GPU portion
                        wired_gb = (wired_pages * 16384) / (1024**3)
                        memory_used_gb = min(wired_gb * 0.3, self.memory_limit_gb * 0.8)
                        break

        except Exception as e:
            logger.debug(f"vm_stat memory estimation failed: {e}")

        try:
            # Estimate utilization from process activity
            # Look for GPU-intensive processes
            proc = await asyncio.create_subprocess_exec(
                "ps",
                "aux",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                ps_output = stdout.decode()
                gpu_processes = 0

                for line in ps_output.split("\n"):
                    if any(
                        name in line.lower()
                        for name in ["python", "mlx", "metal", "gpu"]
                    ):
                        if "%CPU" not in line and len(line.split()) > 2:
                            try:
                                cpu_percent = float(line.split()[2])
                                if cpu_percent > 5:  # Active process
                                    gpu_processes += 1
                            except (ValueError, IndexError):
                                pass

                # Estimate GPU utilization based on active processes
                utilization_percent = min(gpu_processes * 15, 100)
                cores_active = min(
                    int(utilization_percent / 100 * self.gpu_cores), self.gpu_cores
                )

        except Exception as e:
            logger.debug(f"Process-based GPU estimation failed: {e}")

        try:
            # Get temperature if available (M1/M2/M3/M4 thermal sensors)
            proc = await asyncio.create_subprocess_exec(
                "sudo",
                "powermetrics",
                "--samplers",
                "smc",
                "-n",
                "1",
                "-i",
                "100",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                output = stdout.decode()
                for line in output.split("\n"):
                    if "GPU" in line and "temp" in line.lower():
                        # Extract temperature value
                        parts = line.split()
                        for part in parts:
                            if "C" in part:
                                try:
                                    temperature_c = float(part.replace("C", ""))
                                    break
                                except ValueError:
                                    pass
                        break

        except Exception as e:
            logger.debug(f"Temperature reading failed: {e}")

        return MetalStats(
            memory_used_gb=round(memory_used_gb, 2),
            memory_total_gb=memory_total_gb,
            utilization_percent=round(utilization_percent, 1),
            cores_active=cores_active,
            cores_total=self.gpu_cores,
            temperature_c=round(temperature_c, 1),
            timestamp=timestamp,
        )

    async def get_stats(self) -> MetalStats | None:
        """Get latest GPU statistics."""
        if not self.stats_history:
            # Return current stats if no history
            return await self._collect_stats()
        return self.stats_history[-1]

    async def get_average_stats(self, seconds: int = 30) -> MetalStats | None:
        """Get average statistics over specified time period."""
        if not self.stats_history:
            return None

        # Get stats from last N seconds
        cutoff_time = time.time() - seconds
        recent_stats = [s for s in self.stats_history if s.timestamp > cutoff_time]

        if not recent_stats:
            return self.stats_history[-1] if self.stats_history else None

        # Calculate averages
        avg_memory_used = sum(s.memory_used_gb for s in recent_stats) / len(
            recent_stats
        )
        avg_utilization = sum(s.utilization_percent for s in recent_stats) / len(
            recent_stats
        )
        avg_cores_active = sum(s.cores_active for s in recent_stats) / len(recent_stats)
        avg_temperature = sum(s.temperature_c for s in recent_stats) / len(recent_stats)

        return MetalStats(
            memory_used_gb=round(avg_memory_used, 2),
            memory_total_gb=self.memory_limit_gb,
            utilization_percent=round(avg_utilization, 1),
            cores_active=int(avg_cores_active),
            cores_total=self.gpu_cores,
            temperature_c=round(avg_temperature, 1),
            timestamp=time.time(),
        )

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        if not self.stats_history:
            return {"error": "No statistics available"}

        recent_stats = self.stats_history[-60:]  # Last minute

        return {
            "gpu_model": f"Apple M4 Pro ({self.gpu_cores}-core)",
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
        }


# Example usage
if __name__ == "__main__":

    async def demo():
        """Demonstrate Metal monitoring."""
        print("🔥 Metal GPU Monitor Demo")
        print("=" * 40)

        monitor = MetalMonitor()
        await monitor.start()

        try:
            # Monitor for 10 seconds
            for i in range(10):
                stats = await monitor.get_stats()
                print(f"\nSample {i+1}:")
                print(
                    f"  Memory: {stats.memory_used_gb:.1f}GB / {stats.memory_total_gb:.1f}GB ({stats.memory_utilization_percent:.1f}%)"
                )
                print(f"  Utilization: {stats.utilization_percent:.1f}%")
                print(f"  Active cores: {stats.cores_active} / {stats.cores_total}")
                if stats.temperature_c > 0:
                    print(f"  Temperature: {stats.temperature_c:.1f}°C")

                await asyncio.sleep(1)

            # Show summary
            print("\n📊 Performance Summary:")
            summary = monitor.get_performance_summary()
            print(f"  GPU Model: {summary['gpu_model']}")
            print(f"  Average Memory: {summary['averages']['memory_used_gb']:.1f}GB")
            print(
                f"  Average Utilization: {summary['averages']['utilization_percent']:.1f}%"
            )
            print(f"  Peak Memory: {summary['peaks']['max_memory_gb']:.1f}GB")
            print(f"  Peak Utilization: {summary['peaks']['max_utilization']:.1f}%")

        finally:
            await monitor.stop()

    asyncio.run(demo())
