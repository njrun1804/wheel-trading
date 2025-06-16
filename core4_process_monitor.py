#!/usr/bin/env python3
"""
Core 4 Process Monitor - Resource-heavy process management and monitoring
Handles CPU/memory intensive processes, cleanup, and automated management
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("core4_process_monitor.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("Core4ProcessMonitor")


@dataclass
class ProcessInfo:
    """Process information container"""

    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    status: str
    create_time: float
    cmdline: list[str]
    parent_pid: int
    children_count: int
    open_files: int
    connections: int
    threads: int


@dataclass
class ProcessAlert:
    """Process alert information"""

    pid: int
    name: str
    alert_type: str
    value: float
    threshold: float
    timestamp: datetime
    action_taken: str


class Core4ProcessMonitor:
    """Core 4 Process Monitor - Handles resource-heavy processes"""

    def __init__(self, config_file: str | None = None):
        self.config = self._load_config(config_file)
        self.process_history = defaultdict(lambda: deque(maxlen=100))
        self.alerts = deque(maxlen=1000)
        self.cleanup_stats = {
            "processes_killed": 0,
            "zombies_cleaned": 0,
            "memory_freed_mb": 0,
            "cpu_recovered": 0.0,
        }
        self.monitoring = True
        self.critical_processes = set()
        self._load_critical_processes()

    def _load_config(self, config_file: str | None) -> dict:
        """Load configuration settings"""
        default_config = {
            "cpu_threshold": 80.0,  # CPU % threshold for alerts
            "memory_threshold": 85.0,  # Memory % threshold for alerts
            "process_memory_threshold": 1024,  # MB per process
            "max_processes_per_user": 200,
            "zombie_cleanup_interval": 60,
            "monitoring_interval": 5,
            "alert_cooldown": 300,  # 5 minutes
            "auto_kill_enabled": True,
            "auto_kill_cpu_threshold": 95.0,
            "auto_kill_memory_threshold": 90.0,
            "protected_processes": [
                "kernel_task",
                "launchd",
                "WindowServer",
                "loginwindow",
                "SystemUIServer",
                "Dock",
                "Finder",
                "Activity Monitor",
            ],
        }

        if config_file and Path(config_file).exists():
            try:
                import json

                with open(config_file) as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")

        return default_config

    def _load_critical_processes(self):
        """Load critical system processes that should never be killed"""
        critical_patterns = [
            "kernel",
            "launchd",
            "WindowServer",
            "loginwindow",
            "SystemUIServer",
            "Dock",
            "Finder",
            "ssh",
            "Activity Monitor",
            "claude",
            "python",
            "bash",
            "zsh",
            "Terminal",
            "WezTerm",
        ]

        for proc in psutil.process_iter(["pid", "name"]):
            try:
                name = proc.info["name"].lower()
                if any(pattern.lower() in name for pattern in critical_patterns):
                    self.critical_processes.add(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def get_process_info(self, pid: int) -> ProcessInfo | None:
        """Get detailed process information"""
        try:
            proc = psutil.Process(pid)

            # Get process metrics
            cpu_percent = proc.cpu_percent()
            memory_info = proc.memory_info()
            memory_percent = proc.memory_percent()
            memory_mb = memory_info.rss / 1024 / 1024

            # Get additional info
            try:
                open_files = len(proc.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = 0

            try:
                connections = len(proc.connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                connections = 0

            try:
                children_count = len(proc.children())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                children_count = 0

            try:
                threads = proc.num_threads()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                threads = 0

            return ProcessInfo(
                pid=pid,
                name=proc.name(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_mb=memory_mb,
                status=proc.status(),
                create_time=proc.create_time(),
                cmdline=proc.cmdline(),
                parent_pid=proc.ppid(),
                children_count=children_count,
                open_files=open_files,
                connections=connections,
                threads=threads,
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.debug(f"Cannot access process {pid}: {e}")
            return None

    def get_top_processes(
        self, sort_by: str = "cpu", limit: int = 20
    ) -> list[ProcessInfo]:
        """Get top processes by CPU or memory usage"""
        processes = []

        for proc in psutil.process_iter():
            try:
                info = self.get_process_info(proc.pid)
                if info:
                    processes.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Sort by requested metric
        if sort_by == "cpu":
            processes.sort(key=lambda x: x.cpu_percent, reverse=True)
        elif sort_by == "memory":
            processes.sort(key=lambda x: x.memory_percent, reverse=True)
        elif sort_by == "memory_mb":
            processes.sort(key=lambda x: x.memory_mb, reverse=True)

        return processes[:limit]

    def find_zombie_processes(self) -> list[ProcessInfo]:
        """Find zombie processes"""
        zombies = []

        for proc in psutil.process_iter():
            try:
                if proc.status() == psutil.STATUS_ZOMBIE:
                    info = self.get_process_info(proc.pid)
                    if info:
                        zombies.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return zombies

    def find_stuck_processes(self) -> list[ProcessInfo]:
        """Find processes that appear to be stuck (high CPU, no progress)"""
        stuck = []
        current_time = time.time()

        # Look for processes with sustained high CPU usage
        for proc in psutil.process_iter():
            try:
                info = self.get_process_info(proc.pid)
                if not info:
                    continue

                # Check if process has been running for a while with high CPU
                if (
                    info.cpu_percent > 80
                    and current_time - info.create_time > 300
                    and info.status  # Running for 5+ minutes
                    not in ["sleeping", "idle"]
                ):
                    # Check process history for sustained high usage
                    history = self.process_history[proc.pid]
                    if len(history) >= 5:
                        avg_cpu = sum(h.cpu_percent for h in history) / len(history)
                        if avg_cpu > 70:
                            stuck.append(info)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return stuck

    def cleanup_zombie_processes(self) -> int:
        """Clean up zombie processes"""
        zombies = self.find_zombie_processes()
        cleaned = 0

        for zombie in zombies:
            try:
                # Try to kill the parent process if it's not critical
                if zombie.parent_pid not in self.critical_processes:
                    parent = psutil.Process(zombie.parent_pid)
                    logger.info(
                        f"Terminating parent process {zombie.parent_pid} to clean zombie {zombie.pid}"
                    )
                    parent.terminate()

                    # Wait a bit and then force kill if needed
                    time.sleep(2)
                    if parent.is_running():
                        parent.kill()

                    cleaned += 1
                    self.cleanup_stats["zombies_cleaned"] += 1

            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError) as e:
                logger.warning(f"Cannot clean zombie process {zombie.pid}: {e}")

        return cleaned

    def terminate_process(self, pid: int, force: bool = False) -> bool:
        """Terminate a process safely"""
        if pid in self.critical_processes:
            logger.warning(f"Refusing to terminate critical process {pid}")
            return False

        try:
            proc = psutil.Process(pid)
            info = self.get_process_info(pid)

            if not info:
                return False

            # Check if process name is in protected list
            if any(
                protected in info.name.lower()
                for protected in self.config["protected_processes"]
            ):
                logger.warning(f"Process {pid} ({info.name}) is protected")
                return False

            logger.info(
                f"Terminating process {pid} ({info.name}) - "
                f"CPU: {info.cpu_percent}%, Memory: {info.memory_mb:.1f}MB"
            )

            # Try graceful termination first
            proc.terminate()

            # Wait for process to terminate
            try:
                proc.wait(timeout=10)
            except psutil.TimeoutExpired:
                if force:
                    logger.info(f"Force killing process {pid}")
                    proc.kill()
                    proc.wait(timeout=5)
                else:
                    logger.warning(f"Process {pid} did not terminate gracefully")
                    return False

            # Update cleanup stats
            self.cleanup_stats["processes_killed"] += 1
            self.cleanup_stats["memory_freed_mb"] += info.memory_mb
            self.cleanup_stats["cpu_recovered"] += info.cpu_percent

            return True

        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError) as e:
            logger.error(f"Cannot terminate process {pid}: {e}")
            return False

    def auto_cleanup_excessive_processes(self) -> list[ProcessInfo]:
        """Automatically clean up processes using excessive resources"""
        cleaned_processes = []

        if not self.config["auto_kill_enabled"]:
            return cleaned_processes

        # Find processes exceeding thresholds
        top_cpu = self.get_top_processes("cpu", 10)
        top_memory = self.get_top_processes("memory", 10)

        candidates = set()

        # Add high CPU processes
        for proc in top_cpu:
            if proc.cpu_percent > self.config["auto_kill_cpu_threshold"]:
                candidates.add(proc.pid)

        # Add high memory processes
        for proc in top_memory:
            if (
                proc.memory_percent > self.config["auto_kill_memory_threshold"]
                or proc.memory_mb > self.config["process_memory_threshold"]
            ):
                candidates.add(proc.pid)

        # Clean up candidates
        for pid in candidates:
            info = self.get_process_info(pid)
            if info and self.terminate_process(pid, force=True):
                cleaned_processes.append(info)

                # Create alert
                alert = ProcessAlert(
                    pid=pid,
                    name=info.name,
                    alert_type="auto_cleanup",
                    value=max(info.cpu_percent, info.memory_percent),
                    threshold=min(
                        self.config["auto_kill_cpu_threshold"],
                        self.config["auto_kill_memory_threshold"],
                    ),
                    timestamp=datetime.now(),
                    action_taken="terminated",
                )
                self.alerts.append(alert)

        return cleaned_processes

    def set_process_priority(self, pid: int, priority: int) -> bool:
        """Set process priority (nice value)"""
        try:
            proc = psutil.Process(pid)
            current_nice = proc.nice()
            proc.nice(priority)
            logger.info(
                f"Changed process {pid} priority from {current_nice} to {priority}"
            )
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError) as e:
            logger.error(f"Cannot set priority for process {pid}: {e}")
            return False

    def optimize_process_scheduling(self) -> dict:
        """Optimize process scheduling and priorities"""
        optimizations = {
            "high_priority_set": 0,
            "low_priority_set": 0,
            "cpu_affinity_set": 0,
        }

        # Get current top processes
        top_processes = self.get_top_processes("cpu", 20)

        for proc_info in top_processes:
            try:
                psutil.Process(proc_info.pid)

                # Skip critical processes
                if proc_info.pid in self.critical_processes:
                    continue

                # Lower priority for high CPU processes that aren't critical
                if proc_info.cpu_percent > 50 and proc_info.name not in [
                    "claude",
                    "python",
                ]:
                    if self.set_process_priority(proc_info.pid, 10):  # Lower priority
                        optimizations["low_priority_set"] += 1

                # Increase priority for important processes
                elif proc_info.name in ["claude", "python", "Terminal", "WezTerm"]:
                    if self.set_process_priority(proc_info.pid, -5):  # Higher priority
                        optimizations["high_priority_set"] += 1

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return optimizations

    def get_system_resource_usage(self) -> dict:
        """Get overall system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / 1024 / 1024 / 1024,
            "memory_total_gb": memory.total / 1024 / 1024 / 1024,
            "disk_percent": disk.percent,
            "disk_used_gb": disk.used / 1024 / 1024 / 1024,
            "disk_total_gb": disk.total / 1024 / 1024 / 1024,
            "process_count": len(psutil.pids()),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()),
        }

    def create_resource_alert(
        self, alert_type: str, value: float, threshold: float
    ) -> ProcessAlert:
        """Create a resource usage alert"""
        alert = ProcessAlert(
            pid=0,
            name="System",
            alert_type=alert_type,
            value=value,
            threshold=threshold,
            timestamp=datetime.now(),
            action_taken="monitoring",
        )
        self.alerts.append(alert)
        return alert

    def monitor_system_resources(self) -> list[ProcessAlert]:
        """Monitor system resources and create alerts"""
        alerts = []
        system_info = self.get_system_resource_usage()

        # Check CPU usage
        if system_info["cpu_percent"] > self.config["cpu_threshold"]:
            alert = self.create_resource_alert(
                "high_cpu", system_info["cpu_percent"], self.config["cpu_threshold"]
            )
            alerts.append(alert)
            logger.warning(f"High CPU usage: {system_info['cpu_percent']:.1f}%")

        # Check memory usage
        if system_info["memory_percent"] > self.config["memory_threshold"]:
            alert = self.create_resource_alert(
                "high_memory",
                system_info["memory_percent"],
                self.config["memory_threshold"],
            )
            alerts.append(alert)
            logger.warning(f"High memory usage: {system_info['memory_percent']:.1f}%")

        return alerts

    async def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Starting Core 4 process monitoring loop")

        while self.monitoring:
            try:
                # Update process history
                top_processes = self.get_top_processes("cpu", 50)
                for proc in top_processes:
                    self.process_history[proc.pid].append(proc)

                # Monitor system resources
                self.monitor_system_resources()

                # Clean up zombies periodically
                if int(time.time()) % self.config["zombie_cleanup_interval"] == 0:
                    zombies_cleaned = self.cleanup_zombie_processes()
                    if zombies_cleaned > 0:
                        logger.info(f"Cleaned up {zombies_cleaned} zombie processes")

                # Auto cleanup if enabled
                if self.config["auto_kill_enabled"]:
                    cleaned = self.auto_cleanup_excessive_processes()
                    if cleaned:
                        logger.info(f"Auto-cleaned {len(cleaned)} excessive processes")

                # Optimize process scheduling
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    opts = self.optimize_process_scheduling()
                    if any(opts.values()):
                        logger.info(f"Process optimization: {opts}")

                await asyncio.sleep(self.config["monitoring_interval"])

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)

    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.monitoring = False
        logger.info("Stopping Core 4 process monitoring")

    def get_monitoring_report(self) -> dict:
        """Get comprehensive monitoring report"""
        system_info = self.get_system_resource_usage()
        top_cpu = self.get_top_processes("cpu", 10)
        top_memory = self.get_top_processes("memory", 10)
        zombies = self.find_zombie_processes()
        stuck = self.find_stuck_processes()

        return {
            "timestamp": datetime.now().isoformat(),
            "system_resources": system_info,
            "top_cpu_processes": [
                {
                    "pid": p.pid,
                    "name": p.name,
                    "cpu_percent": p.cpu_percent,
                    "memory_mb": p.memory_mb,
                }
                for p in top_cpu
            ],
            "top_memory_processes": [
                {
                    "pid": p.pid,
                    "name": p.name,
                    "memory_percent": p.memory_percent,
                    "memory_mb": p.memory_mb,
                }
                for p in top_memory
            ],
            "zombie_processes": len(zombies),
            "stuck_processes": len(stuck),
            "cleanup_stats": self.cleanup_stats,
            "recent_alerts": len(
                [
                    a
                    for a in self.alerts
                    if datetime.now() - a.timestamp < timedelta(hours=1)
                ]
            ),
        }


def main():
    """Main function for standalone execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Core 4 Process Monitor")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--monitor", action="store_true", help="Start monitoring loop")
    parser.add_argument(
        "--report", action="store_true", help="Generate monitoring report"
    )
    parser.add_argument("--cleanup", action="store_true", help="Run cleanup once")
    parser.add_argument(
        "--top-cpu", type=int, default=10, help="Show top CPU processes"
    )
    parser.add_argument(
        "--top-memory", type=int, default=10, help="Show top memory processes"
    )
    parser.add_argument("--kill-pid", type=int, help="Kill specific process")
    parser.add_argument(
        "--optimize", action="store_true", help="Optimize process scheduling"
    )

    args = parser.parse_args()

    monitor = Core4ProcessMonitor(args.config)

    if args.kill_pid:
        success = monitor.terminate_process(args.kill_pid, force=True)
        print(
            f"Process {args.kill_pid} termination: {'Success' if success else 'Failed'}"
        )
        return

    if args.cleanup:
        zombies = monitor.cleanup_zombie_processes()
        excessive = monitor.auto_cleanup_excessive_processes()
        print(f"Cleaned up {zombies} zombies and {len(excessive)} excessive processes")
        return

    if args.optimize:
        opts = monitor.optimize_process_scheduling()
        print(f"Process optimization results: {opts}")
        return

    if args.report:
        report = monitor.get_monitoring_report()
        import json

        print(json.dumps(report, indent=2, default=str))
        return

    # Show top processes
    print("Top CPU Processes:")
    for i, proc in enumerate(monitor.get_top_processes("cpu", args.top_cpu), 1):
        print(
            f"{i:2d}. PID {proc.pid:5d} - {proc.name:20s} - "
            f"CPU: {proc.cpu_percent:5.1f}% - Memory: {proc.memory_mb:6.1f}MB"
        )

    print("\nTop Memory Processes:")
    for i, proc in enumerate(monitor.get_top_processes("memory", args.top_memory), 1):
        print(
            f"{i:2d}. PID {proc.pid:5d} - {proc.name:20s} - "
            f"Memory: {proc.memory_percent:5.1f}% ({proc.memory_mb:6.1f}MB)"
        )

    # Show system resources
    system_info = monitor.get_system_resource_usage()
    print("\nSystem Resources:")
    print(f"CPU: {system_info['cpu_percent']:.1f}%")
    print(
        f"Memory: {system_info['memory_percent']:.1f}% "
        f"({system_info['memory_used_gb']:.1f}GB / {system_info['memory_total_gb']:.1f}GB)"
    )
    print(f"Processes: {system_info['process_count']}")

    # Show zombies and stuck processes
    zombies = monitor.find_zombie_processes()
    stuck = monitor.find_stuck_processes()

    if zombies:
        print(f"\nZombie Processes: {len(zombies)}")
        for z in zombies:
            print(f"  PID {z.pid} - {z.name}")

    if stuck:
        print(f"\nStuck Processes: {len(stuck)}")
        for s in stuck:
            print(f"  PID {s.pid} - {s.name} - CPU: {s.cpu_percent:.1f}%")

    if args.monitor:
        print("\nStarting monitoring loop (Press Ctrl+C to stop)...")
        try:
            asyncio.run(monitor.monitor_loop())
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")


if __name__ == "__main__":
    main()
