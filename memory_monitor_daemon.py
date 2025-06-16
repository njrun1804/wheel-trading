#!/usr/bin/env python3
"""
Critical Memory Monitoring Daemon for Trading System
Provides real-time memory monitoring with automatic cleanup and alerts.
"""

import contextlib
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import psutil


class MemoryMonitorDaemon:
    def __init__(self, config_path: str | None = None):
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.running = True
        self.alert_queue = queue.Queue()

        # Memory thresholds (in bytes)
        self.thresholds = {
            "critical": self.config.get(
                "critical_threshold", 500 * 1024 * 1024
            ),  # 500MB
            "warning": self.config.get("warning_threshold", 1024 * 1024 * 1024),  # 1GB
            "caution": self.config.get("caution_threshold", 2048 * 1024 * 1024),  # 2GB
        }

        # Protected processes (never kill these)
        self.protected_patterns = [
            "python",
            "claude",
            "wheel",
            "trading",
            "databento",
            "postgres",
            "System",
            "kernel",
            "launchd",
            "WindowServer",
        ]

        # Process memory limits
        self.process_limits = {
            "browser": 800 * 1024 * 1024,  # 800MB per browser process
            "node": 1200 * 1024 * 1024,  # 1.2GB per Node.js process
            "webkit": 500 * 1024 * 1024,  # 500MB per WebKit process
            "trading": 2048 * 1024 * 1024,  # 2GB per trading process (protected)
        }

        self.stats = {
            "cleanups_performed": 0,
            "processes_restarted": 0,
            "memory_freed": 0,
            "alerts_sent": 0,
            "uptime_start": datetime.now(),
        }

    def _load_config(self, config_path: str | None) -> dict:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return json.load(f)

        return {
            "monitor_interval": 30,  # seconds
            "log_level": "INFO",
            "max_log_size": 10 * 1024 * 1024,  # 10MB
            "enable_auto_cleanup": True,
            "enable_process_management": True,
            "alert_cooldown": 300,  # 5 minutes
        }

    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.get("log_level", "INFO"))

        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/memory_monitor.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("MemoryMonitor")
        self.logger.info("Memory Monitor Daemon initialized")

    def get_memory_info(self) -> dict:
        """Get detailed memory information."""
        try:
            # Get system memory info
            vm_info = psutil.virtual_memory()
            swap_info = psutil.swap_memory()

            # Calculate free memory including buffers/cache
            available_memory = vm_info.available

            memory_info = {
                "timestamp": datetime.now().isoformat(),
                "total": vm_info.total,
                "available": available_memory,
                "used": vm_info.used,
                "percentage": vm_info.percent,
                "free": vm_info.free,
                "buffers": getattr(vm_info, "buffers", 0),
                "cached": getattr(vm_info, "cached", 0),
                "swap_total": swap_info.total,
                "swap_used": swap_info.used,
                "swap_free": swap_info.free,
                "swap_percentage": swap_info.percent,
            }

            # Determine memory status
            if available_memory < self.thresholds["critical"]:
                memory_info["status"] = "CRITICAL"
                memory_info["color"] = "red"
            elif available_memory < self.thresholds["warning"]:
                memory_info["status"] = "WARNING"
                memory_info["color"] = "yellow"
            elif available_memory < self.thresholds["caution"]:
                memory_info["status"] = "CAUTION"
                memory_info["color"] = "orange"
            else:
                memory_info["status"] = "OPTIMAL"
                memory_info["color"] = "green"

            return memory_info

        except Exception as e:
            self.logger.error(f"Error getting memory info: {e}")
            return {}

    def get_top_memory_processes(self, limit: int = 20) -> list[dict]:
        """Get top memory consuming processes."""
        try:
            processes = []
            for proc in psutil.process_iter(
                ["pid", "name", "memory_info", "cpu_percent", "cmdline"]
            ):
                try:
                    proc_info = proc.info
                    memory_info = proc_info.get("memory_info")
                    if not memory_info or not hasattr(memory_info, "rss"):
                        continue
                    memory_mb = memory_info.rss / 1024 / 1024

                    processes.append(
                        {
                            "pid": proc_info["pid"],
                            "name": proc_info["name"],
                            "memory_mb": round(memory_mb, 2),
                            "memory_bytes": proc_info["memory_info"].rss,
                            "cpu_percent": proc_info["cpu_percent"],
                            "cmdline": " ".join(proc_info["cmdline"][:3])
                            if proc_info["cmdline"]
                            else proc_info["name"],
                            "protected": self._is_protected_process(
                                proc_info["name"], proc_info["cmdline"]
                            ),
                        }
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by memory usage
            processes.sort(key=lambda x: x["memory_bytes"], reverse=True)
            return processes[:limit]

        except Exception as e:
            self.logger.error(f"Error getting process info: {e}")
            return []

    def _is_protected_process(self, name: str, cmdline: list[str]) -> bool:
        """Check if process should be protected from termination."""
        if not name:
            return True

        name_lower = name.lower()
        cmdline_str = " ".join(cmdline).lower() if cmdline else ""

        for pattern in self.protected_patterns:
            if pattern.lower() in name_lower or pattern.lower() in cmdline_str:
                return True

        return False

    def categorize_process(self, name: str, cmdline: list[str]) -> str:
        """Categorize process for memory management."""
        if not name:
            return "unknown"

        name_lower = name.lower()
        cmdline_str = " ".join(cmdline).lower() if cmdline else ""

        if any(
            pattern in name_lower
            for pattern in ["chrome", "firefox", "safari", "browser"]
        ):
            return "browser"
        elif "node" in name_lower or "nodejs" in name_lower:
            return "node"
        elif "webkit" in name_lower or "webprocess" in name_lower:
            return "webkit"
        elif any(
            pattern in cmdline_str for pattern in ["trading", "wheel", "databento"]
        ):
            return "trading"
        else:
            return "other"

    def cleanup_system_caches(self) -> int:
        """Clean system caches and temporary files."""
        freed_bytes = 0

        try:
            # Clean system caches (macOS)
            if sys.platform == "darwin":
                commands = [
                    ["sudo", "purge"],  # Force memory cleanup
                    ["sudo", "dscacheutil", "-flushcache"],  # DNS cache
                ]

                for cmd in commands:
                    try:
                        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
                        self.logger.info(f"Executed: {' '.join(cmd)}")
                    except (
                        subprocess.CalledProcessError,
                        subprocess.TimeoutExpired,
                    ) as e:
                        self.logger.warning(f"Cache cleanup command failed: {e}")

            # Clean temporary directories
            temp_dirs = [
                "/tmp",
                os.path.expanduser("~/Library/Caches"),
                "/var/folders",
            ]

            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    try:
                        # Remove old temporary files (older than 1 day)
                        result = subprocess.run(
                            ["find", temp_dir, "-type", "f", "-mtime", "+1", "-delete"],
                            capture_output=True,
                            timeout=60,
                        )

                        if result.returncode == 0:
                            self.logger.info(f"Cleaned temp files in {temp_dir}")
                    except subprocess.TimeoutExpired:
                        self.logger.warning(f"Timeout cleaning {temp_dir}")

            self.logger.info("System cache cleanup completed")
            return freed_bytes

        except Exception as e:
            self.logger.error(f"Error during system cleanup: {e}")
            return 0

    def manage_memory_processes(self, memory_info: dict, processes: list[dict]) -> int:
        """Manage processes based on memory usage and limits."""
        if not self.config.get("enable_process_management", True):
            return 0

        actions_taken = 0
        memory_freed = 0

        # Group processes by category
        process_groups = {}
        for proc in processes:
            if proc["protected"]:
                continue

            category = self.categorize_process(
                proc["name"], proc.get("cmdline", "").split()
            )
            if category not in process_groups:
                process_groups[category] = []
            process_groups[category].append(proc)

        # Apply memory limits per category
        for category, category_processes in process_groups.items():
            if category not in self.process_limits:
                continue

            limit_bytes = self.process_limits[category]

            for proc in category_processes:
                if proc["memory_bytes"] > limit_bytes:
                    try:
                        pid = proc["pid"]
                        self.logger.warning(
                            f"Process {proc['name']} (PID: {pid}) using {proc['memory_mb']:.1f}MB "
                            f"exceeds {category} limit of {limit_bytes/1024/1024:.1f}MB"
                        )

                        # Try graceful termination first
                        process = psutil.Process(pid)
                        process.terminate()

                        # Wait for graceful termination
                        try:
                            process.wait(timeout=10)
                            memory_freed += proc["memory_bytes"]
                            actions_taken += 1
                            self.logger.info(
                                f"Successfully terminated {proc['name']} (PID: {pid})"
                            )
                        except psutil.TimeoutExpired:
                            # Force kill if necessary
                            process.kill()
                            memory_freed += proc["memory_bytes"]
                            actions_taken += 1
                            self.logger.warning(
                                f"Force killed {proc['name']} (PID: {pid})"
                            )

                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        self.logger.warning(
                            f"Could not manage process {proc['name']}: {e}"
                        )

        if actions_taken > 0:
            self.stats["processes_restarted"] += actions_taken
            self.stats["memory_freed"] += memory_freed
            self.logger.info(
                f"Process management: {actions_taken} actions, {memory_freed/1024/1024:.1f}MB freed"
            )

        return memory_freed

    def emergency_cleanup(self, memory_info: dict) -> bool:
        """Perform emergency memory cleanup."""
        self.logger.critical("EMERGENCY MEMORY CLEANUP INITIATED")

        total_freed = 0

        # 1. Clean system caches
        freed = self.cleanup_system_caches()
        total_freed += freed

        # 2. Get current processes and manage them
        processes = self.get_top_memory_processes(50)
        freed = self.manage_memory_processes(memory_info, processes)
        total_freed += freed

        # 3. Force garbage collection if Python processes are running
        with contextlib.suppress(Exception):
            subprocess.run(["python3", "-c", "import gc; gc.collect()"], timeout=10)

        self.stats["cleanups_performed"] += 1
        self.stats["memory_freed"] += total_freed

        self.logger.critical(
            f"Emergency cleanup completed. Freed {total_freed/1024/1024:.1f}MB"
        )

        # Wait and re-check memory
        time.sleep(5)
        new_memory_info = self.get_memory_info()

        if new_memory_info["available"] > self.thresholds["critical"]:
            self.logger.info("Emergency cleanup successful - memory pressure relieved")
            return True
        else:
            self.logger.critical(
                "Emergency cleanup insufficient - critical memory pressure persists"
            )
            return False

    def send_alert(self, memory_info: dict, processes: list[dict]):
        """Send memory pressure alert."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "status": memory_info["status"],
            "available_mb": memory_info["available"] / 1024 / 1024,
            "percentage_used": memory_info["percentage"],
            "top_processes": processes[:10],
            "stats": self.stats.copy(),
        }

        # Log alert
        self.logger.warning(
            f"MEMORY ALERT: {memory_info['status']} - {alert['available_mb']:.1f}MB available"
        )

        # Save alert to file
        alert_file = (
            Path("logs")
            / f"memory_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(alert_file, "w") as f:
            json.dump(alert, f, indent=2, default=str)

        self.stats["alerts_sent"] += 1

    def monitor_loop(self):
        """Main monitoring loop."""
        self.logger.info("Starting memory monitoring loop")
        last_alert_time = {}

        while self.running:
            try:
                # Get current memory status
                memory_info = self.get_memory_info()
                if not memory_info:
                    time.sleep(self.config["monitor_interval"])
                    continue

                processes = self.get_top_memory_processes()

                # Log current status
                status = memory_info["status"]
                available_mb = memory_info["available"] / 1024 / 1024

                if status != "OPTIMAL":
                    self.logger.info(
                        f"Memory Status: {status} - {available_mb:.1f}MB available "
                        f"({memory_info['percentage']:.1f}% used)"
                    )

                # Handle different memory pressure levels
                current_time = datetime.now()

                if status == "CRITICAL":
                    # Always perform emergency cleanup for critical status
                    self.emergency_cleanup(memory_info)
                    self.send_alert(memory_info, processes)

                elif status == "WARNING":
                    # Send alert if cooldown period has passed
                    if (
                        "warning" not in last_alert_time
                        or current_time - last_alert_time["warning"]
                        > timedelta(seconds=self.config["alert_cooldown"])
                    ):
                        self.send_alert(memory_info, processes)
                        last_alert_time["warning"] = current_time

                        # Perform preventive cleanup
                        if self.config.get("enable_auto_cleanup", True):
                            self.manage_memory_processes(memory_info, processes)

                elif status == "CAUTION":
                    # Increased monitoring frequency for caution status
                    if (
                        "caution" not in last_alert_time
                        or current_time - last_alert_time["caution"]
                        > timedelta(seconds=self.config["alert_cooldown"] * 2)
                    ):
                        self.logger.info(
                            f"Memory caution: {available_mb:.1f}MB available"
                        )
                        last_alert_time["caution"] = current_time

                # Save periodic statistics
                if current_time.minute % 15 == 0:  # Every 15 minutes
                    self.save_statistics(memory_info, processes)

                time.sleep(self.config["monitor_interval"])

            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config["monitor_interval"])

    def save_statistics(self, memory_info: dict, processes: list[dict]):
        """Save monitoring statistics to file."""
        stats_data = {
            "timestamp": datetime.now().isoformat(),
            "memory_info": memory_info,
            "top_processes": processes[:10],
            "daemon_stats": self.stats.copy(),
            "uptime_hours": (
                datetime.now() - self.stats["uptime_start"]
            ).total_seconds()
            / 3600,
        }

        # Append to daily log file
        log_file = (
            Path("logs") / f"memory_analysis_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        with open(log_file, "a") as f:
            f.write(json.dumps(stats_data, default=str) + "\n")

    def get_status(self) -> dict:
        """Get current daemon status."""
        memory_info = self.get_memory_info()
        processes = self.get_top_memory_processes(10)

        return {
            "running": self.running,
            "uptime_hours": (
                datetime.now() - self.stats["uptime_start"]
            ).total_seconds()
            / 3600,
            "memory_info": memory_info,
            "top_processes": processes,
            "stats": self.stats.copy(),
            "thresholds": {k: v / 1024 / 1024 for k, v in self.thresholds.items()},
            "config": self.config,
        }

    def signal_handler(self, signum, frame):
        """Handle termination signals."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully")
        self.running = False

    def start(self):
        """Start the memory monitoring daemon."""
        # Register signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        try:
            self.monitor_loop()
        except Exception as e:
            self.logger.critical(f"Fatal error in daemon: {e}")
        finally:
            self.logger.info("Memory monitoring daemon stopped")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Memory Monitor Daemon")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument(
        "--status", action="store_true", help="Show current status and exit"
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="Perform emergency cleanup and exit"
    )

    args = parser.parse_args()

    daemon = MemoryMonitorDaemon(args.config)

    if args.status:
        status = daemon.get_status()
        print(json.dumps(status, indent=2))
        return

    if args.cleanup:
        memory_info = daemon.get_memory_info()
        success = daemon.emergency_cleanup(memory_info)
        print(f"Emergency cleanup {'successful' if success else 'failed'}")
        return

    # Start daemon
    daemon.start()


if __name__ == "__main__":
    main()
