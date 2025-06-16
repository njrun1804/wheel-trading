"""Real Metal GPU monitoring for macOS."""
import asyncio
import logging
import subprocess
import time

import psutil

logger = logging.getLogger(__name__)


class MetalGPUMonitor:
    """Monitor Metal GPU usage on Apple Silicon."""

    def __init__(self):
        self._last_sample = None
        self._sample_interval = 1.0
        self._gpu_memory_used = 0
        self._gpu_utilization = 0.0
        self._monitoring = False

    async def start_monitoring(self):
        """Start background GPU monitoring."""
        await asyncio.sleep(0)
        self._monitoring = True
        asyncio.create_task(self._monitor_loop())

    def stop_monitoring(self):
        """Stop GPU monitoring."""
        self._monitoring = False

    async def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                self._update_metrics()
                await asyncio.sleep(self._sample_interval)
            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")
                await asyncio.sleep(5)

    def _update_metrics(self):
        """Update GPU metrics using available methods."""
        gpu_stats = self._get_ioreg_stats()
        if gpu_stats:
            self._gpu_utilization = gpu_stats.get("utilization", 0)
            self._gpu_memory_used = gpu_stats.get("memory_used_mb", 0)
            return
        self._estimate_gpu_usage()

    def _get_ioreg_stats(self) -> dict[str, float] | None:
        """Try to get GPU stats from ioreg."""
        try:
            output = subprocess.check_output(
                ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            stats = {}
            for line in output.split("\n"):
                line = line.strip()
                if '"PercentUtilization"' in line:
                    try:
                        value = int(line.split("=")[1].strip())
                        stats["utilization"] = value
                    except (ValueError, IndexError) as e:
                        logger.debug(
                            f"Failed to parse utilization from line: {line.strip()}, error: {e}"
                        )
                if '"AllocatedSize"' in line:
                    try:
                        value = int(line.split("=")[1].strip())
                        stats["memory_used_mb"] = value / (1024 * 1024)
                    except (ValueError, IndexError) as e:
                        logger.debug(
                            f"Failed to parse memory size from line: {line.strip()}, error: {e}"
                        )
            return stats if stats else None
        except subprocess.CalledProcessError:
            return None
        except Exception as e:
            logger.debug(f"ioreg query failed: {e}")
            return None

    def _estimate_gpu_usage(self):
        """Estimate GPU usage from system metrics."""
        try:
            vm = psutil.virtual_memory()
            gpu_processes = self._find_gpu_processes()
            if gpu_processes:
                total_gpu_memory = sum(p["memory_mb"] for p in gpu_processes)
                max_gpu_memory = vm.total * 0.5 / (1024 * 1024)
                utilization = min(100, total_gpu_memory / max_gpu_memory * 100)
                self._gpu_utilization = utilization
                self._gpu_memory_used = total_gpu_memory
            else:
                self._gpu_utilization = 0.0
                self._gpu_memory_used = 0
        except Exception as e:
            logger.debug(f"GPU estimation failed: {e}")
            self._gpu_utilization = 0.0
            self._gpu_memory_used = 0

    def _find_gpu_processes(self) -> list:
        """Find processes likely using GPU."""
        gpu_processes = []
        try:
            for proc in psutil.process_iter(["pid", "name", "memory_info"]):
                try:
                    pinfo = proc.info
                    name = pinfo["name"].lower()
                    gpu_keywords = [
                        "python",
                        "julia",
                        "metalperformanceshaders",
                        "tensorflow",
                        "pytorch",
                        "mlx",
                    ]
                    if any(keyword in name for keyword in gpu_keywords):
                        memory_mb = pinfo["memory_info"].rss / (1024 * 1024)
                        if memory_mb > 100:
                            gpu_processes.append(
                                {
                                    "pid": pinfo["pid"],
                                    "name": pinfo["name"],
                                    "memory_mb": memory_mb,
                                }
                            )
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.debug(f"Ignored exception in {'metal_monitor.py'}: {e}")
        except Exception as e:
            logger.debug(f"Process enumeration failed: {e}")
        return gpu_processes

    def get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        return self._gpu_utilization

    def get_gpu_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        return self._gpu_memory_used

    def get_metrics(self) -> dict[str, float]:
        """Get all GPU metrics."""
        return {
            "utilization_percent": self._gpu_utilization,
            "memory_used_mb": self._gpu_memory_used,
            "monitoring": self._monitoring,
        }


class EnhancedResourceMonitor:
    """Enhanced resource monitoring with real GPU metrics."""

    def __init__(self):
        self.gpu_monitor = MetalGPUMonitor()
        self._start_time = time.time()

    async def start(self):
        """Start monitoring."""
        await self.gpu_monitor.start_monitoring()

    async def stop(self):
        """Stop monitoring."""
        await self.gpu_monitor.stop_monitoring()

    def get_system_metrics(self) -> dict[str, any]:
        """Get comprehensive system metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        gpu_metrics = self.gpu_monitor.get_metrics()
        disk_io = psutil.disk_io_counters()
        net_io = psutil.net_io_counters()
        temps = self._get_temperatures()
        return {
            "timestamp": time.time(),
            "uptime": time.time() - self._start_time,
            "cpu": {
                "percent_per_core": cpu_percent,
                "percent_avg": sum(cpu_percent) / len(cpu_percent),
                "frequency_mhz": cpu_freq.current if cpu_freq else 0,
                "frequency_max_mhz": cpu_freq.max if cpu_freq else 0,
            },
            "memory": {
                "total_gb": vm.total / 1024**3,
                "used_gb": vm.used / 1024**3,
                "available_gb": vm.available / 1024**3,
                "percent": vm.percent,
                "swap_used_gb": swap.used / 1024**3,
                "swap_percent": swap.percent,
            },
            "gpu": gpu_metrics,
            "io": {
                "disk_read_mb": disk_io.read_bytes / 1024**2 if disk_io else 0,
                "disk_write_mb": disk_io.write_bytes / 1024**2 if disk_io else 0,
                "network_sent_mb": net_io.bytes_sent / 1024**2 if net_io else 0,
                "network_recv_mb": net_io.bytes_recv / 1024**2 if net_io else 0,
            },
            "temperature": temps,
        }

    def _get_temperatures(self) -> dict[str, float]:
        """Get system temperatures if available."""
        temps = {}
        try:
            if hasattr(psutil, "sensors_temperatures"):
                sensor_temps = psutil.sensors_temperatures()
                for name, entries in sensor_temps.items():
                    if entries:
                        temps[name] = entries[0].current
            if not temps:
                try:
                    output = subprocess.check_output(
                        [
                            "sudo",
                            "-n",
                            "powermetrics",
                            "-i",
                            "1",
                            "-n",
                            "1",
                            "-s",
                            "thermal",
                        ],
                        text=True,
                        stderr=subprocess.DEVNULL,
                        timeout=2,
                    )
                    for line in output.split("\n"):
                        if "temperature" in line.lower():
                            parts = line.split(":")
                            if len(parts) == 2:
                                try:
                                    temp = float(parts[1].strip().replace("C", ""))
                                    temps["system"] = temp
                                    break
                                except Exception as e:
                                    logger.debug(
                                        f"Ignored exception in {'metal_monitor.py'}: {e}"
                                    )
                except Exception as e:
                    logger.debug(f"Ignored exception in {'metal_monitor.py'}: {e}")
        except Exception as e:
            logger.debug(f"Temperature reading failed: {e}")
        return temps


if __name__ == "__main__":

    async def test_monitor():
        monitor = EnhancedResourceMonitor()
        await monitor.start()
        for i in range(10):
            metrics = monitor.get_system_metrics()
            print(f"\nIteration {i + 1}:")
            print(f"CPU: {metrics['cpu']['percent_avg']:.1f}%")
            print(f"Memory: {metrics['memory']['percent']:.1f}%")
            print(f"GPU: {metrics['gpu']['utilization_percent']:.1f}%")
            print(f"GPU Memory: {metrics['gpu']['memory_used_mb']:.0f}MB")
            await asyncio.sleep(1)
        await monitor.stop()

    asyncio.run(test_monitor())
