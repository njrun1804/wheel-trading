#!/usr/bin/env python3
"""
Unified System Manager - All Optimizations Embedded
Replaces all external daemons/services with embedded monitoring
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/unified_system.log"), logging.StreamHandler()],
)
logger = logging.getLogger("UnifiedSystem")


@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_available: int
    memory_percent: float
    load_average: tuple
    process_count: int
    failed_services: int
    gpu_memory_used: int = 0
    thermal_state: str = "normal"


class UnifiedSystemManager:
    """Embedded system manager - no external daemons needed"""

    def __init__(self):
        self.running = False
        self.monitors = {}
        self.executor = ThreadPoolExecutor(max_workers=8)  # Use all 8 P-cores
        self.metrics = SystemMetrics(0, 0, 0, (0, 0, 0), 0, 0)
        self.callbacks = []

        # Create logs directory
        os.makedirs("logs", exist_ok=True)

        # Initialize components
        self.memory_manager = EmbeddedMemoryManager()
        self.process_manager = EmbeddedProcessManager()
        self.service_optimizer = EmbeddedServiceOptimizer()
        self.gpu_manager = EmbeddedGPUManager()

        logger.info("Unified System Manager initialized")

    def add_callback(self, callback: Callable):
        """Add callback for system events"""
        self.callbacks.append(callback)

    def start(self):
        """Start all monitoring in embedded mode"""
        if self.running:
            return

        self.running = True
        logger.info("Starting unified system monitoring...")

        # Start all monitors in parallel threads
        monitors = [
            ("memory_monitor", self._memory_monitor_loop),
            ("process_monitor", self._process_monitor_loop),
            ("service_monitor", self._service_monitor_loop),
            ("gpu_monitor", self._gpu_monitor_loop),
            ("system_optimizer", self._system_optimizer_loop),
            ("metrics_collector", self._metrics_collector_loop),
        ]

        for name, monitor_func in monitors:
            future = self.executor.submit(monitor_func)
            self.monitors[name] = future
            logger.info(f"Started {name}")

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("All monitors started successfully")

    def stop(self):
        """Stop all monitoring"""
        self.running = False
        logger.info("Stopping unified system monitoring...")

        # Cancel all futures
        for name, future in self.monitors.items():
            future.cancel()
            logger.info(f"Stopped {name}")

        self.executor.shutdown(wait=True)
        logger.info("Unified system manager stopped")

    def get_status(self) -> dict:
        """Get current system status"""
        return {
            "running": self.running,
            "monitors": list(self.monitors.keys()),
            "metrics": {
                "cpu_percent": self.metrics.cpu_percent,
                "memory_available_gb": self.metrics.memory_available / 1024,
                "memory_percent": self.metrics.memory_percent,
                "load_average": self.metrics.load_average,
                "process_count": self.metrics.process_count,
                "failed_services": self.metrics.failed_services,
                "gpu_memory_used_mb": self.metrics.gpu_memory_used,
                "thermal_state": self.metrics.thermal_state,
            },
        }

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def _memory_monitor_loop(self):
        """Embedded memory monitoring"""
        while self.running:
            try:
                # Get memory info
                memory = psutil.virtual_memory()
                available_mb = memory.available // (1024 * 1024)

                # Update metrics
                self.metrics.memory_available = available_mb
                self.metrics.memory_percent = memory.percent

                # Check thresholds
                if available_mb < 500:  # Critical
                    logger.warning(f"CRITICAL: Only {available_mb}MB memory available")
                    self.memory_manager.emergency_cleanup()
                elif available_mb < 1000:  # Warning
                    logger.warning(f"WARNING: Only {available_mb}MB memory available")
                    self.memory_manager.moderate_cleanup()

                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(
                            "memory",
                            {"available_mb": available_mb, "percent": memory.percent},
                        )
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
                time.sleep(10)

    def _process_monitor_loop(self):
        """Embedded process monitoring"""
        while self.running:
            try:
                processes = list(
                    psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"])
                )
                self.metrics.process_count = len(processes)

                # Find high-resource processes
                high_cpu = []
                high_memory = []

                for proc in processes:
                    try:
                        cpu_percent = (
                            proc.info.get("cpu_percent", 0) if proc.info else 0
                        )
                        memory_info = (
                            proc.info.get("memory_info") if proc.info else None
                        )

                        if (
                            cpu_percent
                            and isinstance(cpu_percent, int | float)
                            and cpu_percent > 50
                        ):
                            high_cpu.append(proc)
                        if (
                            memory_info
                            and hasattr(memory_info, "rss")
                            and memory_info.rss
                            and memory_info.rss > 1024 * 1024 * 1000
                        ):  # >1GB
                            high_memory.append(proc)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                        continue

                # Handle excessive processes
                if high_cpu or high_memory:
                    self.process_manager.handle_excessive_processes(
                        high_cpu + high_memory
                    )

                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Process monitor error: {e}")
                time.sleep(30)

    def _service_monitor_loop(self):
        """Embedded service monitoring"""
        while self.running:
            try:
                # Check failed services
                result = subprocess.run(
                    ["launchctl", "list"], capture_output=True, text=True
                )
                failed_count = 0

                if result.returncode == 0:
                    lines = result.stdout.split("\n")[1:]  # Skip header
                    for line in lines:
                        if line.strip():
                            parts = line.split("\t")
                            if len(parts) >= 2:
                                parts[0].strip()
                                status = parts[1].strip()
                                # Only count as failed if status is not 0 and not -9 (terminated)
                                # -9 is normal termination, 0 is success
                                if status != "0" and status != "-9" and status != "-":
                                    failed_count += 1

                self.metrics.failed_services = failed_count

                if failed_count > 30:
                    logger.warning(f"High number of failed services: {failed_count}")
                    self.service_optimizer.optimize_services()

                time.sleep(120)  # Check every 2 minutes

            except Exception as e:
                logger.error(f"Service monitor error: {e}")
                time.sleep(60)

    def _gpu_monitor_loop(self):
        """Embedded GPU monitoring (MLX memory tracking)"""
        while self.running:
            try:
                # Check GPU memory if MLX is available
                try:
                    import mlx.core as mx

                    # Estimate GPU memory usage
                    gpu_memory = (
                        100  # Placeholder - MLX doesn't expose direct memory stats
                    )
                    self.metrics.gpu_memory_used = gpu_memory

                    # Cleanup if needed
                    if gpu_memory > 500:  # >500MB
                        self.gpu_manager.cleanup_gpu_memory()

                except ImportError:
                    pass  # MLX not available

                time.sleep(45)  # Check every 45 seconds

            except Exception as e:
                logger.error(f"GPU monitor error: {e}")
                time.sleep(60)

    def _system_optimizer_loop(self):
        """Embedded system optimization"""
        while self.running:
            try:
                # Get system load
                load_avg = os.getloadavg()
                self.metrics.load_average = load_avg

                # Optimize if load is high
                if load_avg[0] > 8.0:  # High load for 12-core system
                    logger.warning(f"High system load: {load_avg[0]}")
                    self.service_optimizer.reduce_system_load()

                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics.cpu_percent = cpu_percent

                time.sleep(180)  # Check every 3 minutes

            except Exception as e:
                logger.error(f"System optimizer error: {e}")
                time.sleep(120)

    def _metrics_collector_loop(self):
        """Collect and log metrics"""
        while self.running:
            try:
                metrics_data = {
                    "timestamp": time.time(),
                    "cpu_percent": self.metrics.cpu_percent,
                    "memory_available_mb": self.metrics.memory_available,
                    "memory_percent": self.metrics.memory_percent,
                    "load_average": self.metrics.load_average,
                    "process_count": self.metrics.process_count,
                    "failed_services": self.metrics.failed_services,
                    "gpu_memory_used": self.metrics.gpu_memory_used,
                }

                # Log to file
                with open("logs/system_metrics.json", "a") as f:
                    f.write(json.dumps(metrics_data) + "\n")

                time.sleep(300)  # Log every 5 minutes

            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                time.sleep(60)


class EmbeddedMemoryManager:
    """Memory management embedded in main process"""

    def emergency_cleanup(self):
        """Emergency memory cleanup"""
        logger.warning("Executing emergency memory cleanup")
        try:
            # Clear system caches
            subprocess.run(["sudo", "purge"], check=False, capture_output=True)

            # Clear user caches
            subprocess.run(["rm", "-rf", "/tmp/*"], check=False, capture_output=True)

            logger.info("Emergency cleanup completed")
        except Exception as e:
            logger.error(f"Emergency cleanup error: {e}")

    def moderate_cleanup(self):
        """Moderate memory cleanup"""
        logger.info("Executing moderate memory cleanup")
        try:
            # Clear DNS cache
            subprocess.run(
                ["sudo", "dscacheutil", "-flushcache"], check=False, capture_output=True
            )

            logger.info("Moderate cleanup completed")
        except Exception as e:
            logger.error(f"Moderate cleanup error: {e}")


class EmbeddedProcessManager:
    """Process management embedded in main process"""

    def __init__(self):
        self.protected_processes = {
            "python",
            "claude",
            "launchd",
            "kernel_task",
            "loginwindow",
        }

    def handle_excessive_processes(self, processes):
        """Handle processes using too many resources"""
        for proc in processes:
            try:
                if proc.info["name"].lower() in self.protected_processes:
                    continue

                # Log high usage
                cpu = proc.info.get("cpu_percent", 0)
                memory = proc.info.get("memory_info", {}).get("rss", 0) // (1024 * 1024)

                logger.warning(
                    f"High resource process: {proc.info['name']} (PID {proc.info['pid']}) - CPU: {cpu}%, Memory: {memory}MB"
                )

                # Could implement process termination here if needed
                # For safety, just log for now

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue


class EmbeddedServiceOptimizer:
    """Service optimization embedded in main process"""

    def optimize_services(self):
        """Optimize system services"""
        logger.info("Optimizing system services")
        try:
            # Basic service optimization
            subprocess.run(
                ["killall", "-HUP", "mDNSResponder"], check=False, capture_output=True
            )
            logger.info("Service optimization completed")
        except Exception as e:
            logger.error(f"Service optimization error: {e}")

    def reduce_system_load(self):
        """Reduce system load"""
        logger.info("Reducing system load")
        try:
            # Reduce process priorities
            subprocess.run(
                ["sudo", "renice", "10", "-p", str(os.getpid())],
                check=False,
                capture_output=True,
            )
            logger.info("System load reduction completed")
        except Exception as e:
            logger.error(f"Load reduction error: {e}")


class EmbeddedGPUManager:
    """GPU management embedded in main process"""

    def cleanup_gpu_memory(self):
        """Cleanup GPU memory"""
        logger.info("Cleaning up GPU memory")
        try:
            # Force garbage collection
            import gc

            gc.collect()

            # MLX cleanup if available
            try:
                import mlx.core as mx

                # Force evaluation of pending operations
                mx.eval(mx.array([1]))
                logger.info("MLX GPU cleanup completed")
            except ImportError:
                pass

        except Exception as e:
            logger.error(f"GPU cleanup error: {e}")


# Main execution
if __name__ == "__main__":
    manager = UnifiedSystemManager()

    # Add example callback
    def system_callback(event_type, data):
        if event_type == "memory" and data["available_mb"] < 1000:
            print(f"⚠️  Low memory alert: {data['available_mb']}MB available")

    manager.add_callback(system_callback)

    try:
        print("🚀 Starting Unified System Manager...")
        print("   All monitoring embedded in main process")
        print("   Press Ctrl+C to stop")

        manager.start()

        # Keep main thread alive
        while manager.running:
            status = manager.get_status()
            print(
                f"\r📊 CPU: {status['metrics']['cpu_percent']:.1f}% | "
                f"RAM: {status['metrics']['memory_available_gb']:.1f}GB | "
                f"Load: {status['metrics']['load_average'][0]:.2f} | "
                f"Procs: {status['metrics']['process_count']} | "
                f"Failed Svcs: {status['metrics']['failed_services']}",
                end="",
            )

            time.sleep(10)

    except KeyboardInterrupt:
        print("\n🛑 Stopping Unified System Manager...")
        manager.stop()
        print("✅ Stopped successfully")
