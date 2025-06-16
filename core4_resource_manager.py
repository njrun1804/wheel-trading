#!/usr/bin/env python3
"""
Core 4 Resource Manager - Advanced resource limits, alerts, and automated management
Complements the process monitor with resource constraints and intelligent management
"""

import asyncio
import json
import logging
import os
import resource
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("core4_resource_manager.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("Core4ResourceManager")


@dataclass
class ResourceLimit:
    """Resource limit configuration"""

    name: str
    resource_type: str  # cpu, memory, disk, network, files
    threshold: float
    action: str  # alert, throttle, kill, restart
    cooldown: int  # seconds
    enabled: bool = True


@dataclass
class ResourceAlert:
    """Resource alert with context"""

    timestamp: datetime
    resource_type: str
    current_value: float
    threshold: float
    severity: str  # low, medium, high, critical
    affected_processes: list[int]
    action_taken: str
    resolved: bool = False


@dataclass
class ProcessConstraint:
    """Process-specific resource constraints"""

    pid: int
    name: str
    max_cpu_percent: float
    max_memory_mb: float
    max_open_files: int
    priority: int
    cpu_affinity: list[int] | None
    enforced: bool = True


class Core4ResourceManager:
    """Advanced resource management and constraint enforcement"""

    def __init__(self, config_file: str | None = None):
        self.config = self._load_config(config_file)
        self.limits = self._setup_resource_limits()
        self.alerts = deque(maxlen=1000)
        self.process_constraints = {}
        self.resource_history = defaultdict(lambda: deque(maxlen=200))
        self.enforcement_stats = {
            "processes_throttled": 0,
            "processes_killed": 0,
            "alerts_generated": 0,
            "constraints_applied": 0,
            "memory_reclaimed_mb": 0,
        }
        self.active = True
        self._setup_resource_monitoring()

    def _load_config(self, config_file: str | None) -> dict:
        """Load configuration"""
        default_config = {
            "resource_limits": {
                "system_cpu_threshold": 85.0,
                "system_memory_threshold": 90.0,
                "process_cpu_threshold": 75.0,
                "process_memory_threshold": 1024,  # MB
                "disk_usage_threshold": 85.0,
                "open_files_threshold": 1000,
                "network_connections_threshold": 500,
            },
            "enforcement": {
                "enabled": True,
                "kill_runaway_processes": True,
                "throttle_high_cpu": True,
                "memory_pressure_response": True,
                "automatic_nice_adjustment": True,
            },
            "monitoring": {"interval": 5, "history_size": 200, "alert_cooldown": 300},
            "protection": {
                "critical_processes": [
                    "kernel_task",
                    "launchd",
                    "WindowServer",
                    "loginwindow",
                    "claude",
                    "python",
                    "bash",
                    "zsh",
                    "ssh",
                ],
                "never_kill": ["kernel_task", "launchd", "WindowServer", "loginwindow"],
            },
        }

        if config_file and Path(config_file).exists():
            try:
                with open(config_file) as f:
                    user_config = json.load(f)
                self._deep_update(default_config, user_config)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")

        return default_config

    def _deep_update(self, base_dict: dict, update_dict: dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def _setup_resource_limits(self) -> list[ResourceLimit]:
        """Setup resource limits from config"""
        limits = []
        cfg = self.config["resource_limits"]

        limits.extend(
            [
                ResourceLimit(
                    "system_cpu", "cpu", cfg["system_cpu_threshold"], "alert", 60
                ),
                ResourceLimit(
                    "system_memory",
                    "memory",
                    cfg["system_memory_threshold"],
                    "throttle",
                    30,
                ),
                ResourceLimit(
                    "process_cpu", "cpu", cfg["process_cpu_threshold"], "throttle", 120
                ),
                ResourceLimit(
                    "process_memory",
                    "memory",
                    cfg["process_memory_threshold"],
                    "kill",
                    180,
                ),
                ResourceLimit(
                    "disk_usage", "disk", cfg["disk_usage_threshold"], "alert", 300
                ),
                ResourceLimit(
                    "open_files", "files", cfg["open_files_threshold"], "alert", 60
                ),
                ResourceLimit(
                    "network_connections",
                    "network",
                    cfg["network_connections_threshold"],
                    "alert",
                    60,
                ),
            ]
        )

        return limits

    def _setup_resource_monitoring(self):
        """Setup system resource monitoring"""
        try:
            # Set system resource limits
            resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 8192))  # File descriptors
            resource.setrlimit(resource.RLIMIT_NPROC, (2048, 4096))  # Processes
            logger.info("System resource limits configured")
        except Exception as e:
            logger.warning(f"Could not set system limits: {e}")

    def get_system_resources(self) -> dict:
        """Get comprehensive system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()

        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        disk = psutil.disk_usage("/")
        disk_io = psutil.disk_io_counters()

        network = psutil.net_io_counters()

        # Process counts
        process_count = len(psutil.pids())

        # Load average (macOS)
        try:
            load_avg = os.getloadavg()
        except AttributeError:
            load_avg = (0, 0, 0)

        return {
            "timestamp": datetime.now(),
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count,
                "frequency": cpu_freq.current if cpu_freq else 0,
                "load_avg_1m": load_avg[0],
                "load_avg_5m": load_avg[1],
                "load_avg_15m": load_avg[2],
            },
            "memory": {
                "percent": memory.percent,
                "total_gb": memory.total / 1024**3,
                "used_gb": memory.used / 1024**3,
                "available_gb": memory.available / 1024**3,
                "swap_percent": swap.percent,
                "swap_used_gb": swap.used / 1024**3,
            },
            "disk": {
                "percent": disk.percent,
                "total_gb": disk.total / 1024**3,
                "used_gb": disk.used / 1024**3,
                "free_gb": disk.free / 1024**3,
                "read_mb": disk_io.read_bytes / 1024**2 if disk_io else 0,
                "write_mb": disk_io.write_bytes / 1024**2 if disk_io else 0,
            },
            "network": {
                "bytes_sent_mb": network.bytes_sent / 1024**2 if network else 0,
                "bytes_recv_mb": network.bytes_recv / 1024**2 if network else 0,
                "packets_sent": network.packets_sent if network else 0,
                "packets_recv": network.packets_recv if network else 0,
            },
            "processes": {
                "count": process_count,
                "limit": resource.getrlimit(resource.RLIMIT_NPROC)[0],
            },
        }

    def check_resource_limits(self, resources: dict) -> list[ResourceAlert]:
        """Check if resources exceed configured limits"""
        alerts = []
        current_time = datetime.now()

        for limit in self.limits:
            if not limit.enabled:
                continue

            current_value = 0
            affected_processes = []

            if limit.resource_type == "cpu":
                if limit.name == "system_cpu":
                    current_value = resources["cpu"]["percent"]
                elif limit.name == "process_cpu":
                    # Check individual processes
                    for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
                        try:
                            if proc.info["cpu_percent"] > limit.threshold:
                                affected_processes.append(proc.info["pid"])
                                current_value = max(
                                    current_value, proc.info["cpu_percent"]
                                )
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue

            elif limit.resource_type == "memory":
                if limit.name == "system_memory":
                    current_value = resources["memory"]["percent"]
                elif limit.name == "process_memory":
                    # Check individual processes
                    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
                        try:
                            memory_mb = proc.info["memory_info"].rss / 1024**2
                            if memory_mb > limit.threshold:
                                affected_processes.append(proc.info["pid"])
                                current_value = max(current_value, memory_mb)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue

            elif limit.resource_type == "disk":
                current_value = resources["disk"]["percent"]

            # Create alert if threshold exceeded
            if current_value is not None and current_value > limit.threshold:
                severity = self._determine_severity(current_value, limit.threshold)

                alert = ResourceAlert(
                    timestamp=current_time,
                    resource_type=limit.resource_type,
                    current_value=current_value,
                    threshold=limit.threshold,
                    severity=severity,
                    affected_processes=affected_processes,
                    action_taken=limit.action,
                )

                alerts.append(alert)
                self.alerts.append(alert)

        return alerts

    def _determine_severity(self, current: float, threshold: float) -> str:
        """Determine alert severity based on how much threshold is exceeded"""
        if current is None or threshold is None or threshold == 0:
            return "low"
        ratio = current / threshold
        if ratio >= 1.5:
            return "critical"
        elif ratio >= 1.25:
            return "high"
        elif ratio >= 1.1:
            return "medium"
        else:
            return "low"

    def apply_process_constraint(self, pid: int, constraint: ProcessConstraint) -> bool:
        """Apply resource constraints to a specific process"""
        try:
            proc = psutil.Process(pid)

            # Set CPU affinity if specified
            if constraint.cpu_affinity:
                try:
                    proc.cpu_affinity(constraint.cpu_affinity)
                    logger.info(
                        f"Set CPU affinity for PID {pid} to {constraint.cpu_affinity}"
                    )
                except AttributeError:
                    logger.debug("CPU affinity not supported on this platform")
                except Exception as e:
                    logger.warning(f"Could not set CPU affinity for PID {pid}: {e}")

            # Set process priority
            try:
                proc.nice(constraint.priority)
                logger.info(f"Set priority for PID {pid} to {constraint.priority}")
            except Exception as e:
                logger.warning(f"Could not set priority for PID {pid}: {e}")

            # Store constraint for monitoring
            self.process_constraints[pid] = constraint
            self.enforcement_stats["constraints_applied"] += 1

            return True

        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError) as e:
            logger.error(f"Cannot apply constraints to PID {pid}: {e}")
            return False

    def enforce_resource_limits(self, alerts: list[ResourceAlert]) -> dict:
        """Enforce resource limits based on alerts"""
        enforcement_actions = {
            "throttled": [],
            "killed": [],
            "alerted": [],
            "constrained": [],
        }

        if not self.config["enforcement"]["enabled"]:
            return enforcement_actions

        for alert in alerts:
            if (
                alert.action_taken == "kill"
                and self.config["enforcement"]["kill_runaway_processes"]
            ):
                # Kill processes exceeding memory limits
                for pid in alert.affected_processes:
                    if self._can_kill_process(pid):
                        try:
                            proc = psutil.Process(pid)
                            proc_info = proc.as_dict(["name", "memory_info"])

                            proc.terminate()
                            time.sleep(2)

                            if proc.is_running():
                                proc.kill()

                            memory_freed = proc_info["memory_info"].rss / 1024**2
                            self.enforcement_stats[
                                "memory_reclaimed_mb"
                            ] += memory_freed
                            self.enforcement_stats["processes_killed"] += 1

                            enforcement_actions["killed"].append(
                                {
                                    "pid": pid,
                                    "name": proc_info["name"],
                                    "memory_freed_mb": memory_freed,
                                }
                            )

                            logger.info(
                                f"Killed process {pid} ({proc_info['name']}) - "
                                f"freed {memory_freed:.1f}MB"
                            )

                        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                            logger.warning(f"Could not kill process {pid}: {e}")

            elif (
                alert.action_taken == "throttle"
                and self.config["enforcement"]["throttle_high_cpu"]
            ):
                # Throttle high CPU processes
                for pid in alert.affected_processes:
                    if self._can_throttle_process(pid):
                        constraint = ProcessConstraint(
                            pid=pid,
                            name=psutil.Process(pid).name(),
                            max_cpu_percent=50,
                            max_memory_mb=2048,
                            max_open_files=512,
                            priority=10,  # Lower priority
                        )

                        if self.apply_process_constraint(pid, constraint):
                            self.enforcement_stats["processes_throttled"] += 1
                            enforcement_actions["throttled"].append(
                                {"pid": pid, "name": constraint.name}
                            )

            elif alert.action_taken == "alert":
                enforcement_actions["alerted"].append(
                    {
                        "type": alert.resource_type,
                        "value": alert.current_value,
                        "threshold": alert.threshold,
                    }
                )

        return enforcement_actions

    def _can_kill_process(self, pid: int) -> bool:
        """Check if a process can be safely killed"""
        try:
            proc = psutil.Process(pid)
            name = proc.name().lower()

            # Never kill critical system processes
            never_kill = [p.lower() for p in self.config["protection"]["never_kill"]]
            if any(critical in name for critical in never_kill):
                return False

            # Don't kill our own process or parent
            if pid == os.getpid() or pid == os.getppid():
                return False

            # Don't kill process group leaders of critical processes
            if proc.pid == proc.pgid:
                children = proc.children(recursive=True)
                for child in children:
                    child_name = child.name().lower()
                    if any(critical in child_name for critical in never_kill):
                        return False

            return True

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def _can_throttle_process(self, pid: int) -> bool:
        """Check if a process can be safely throttled"""
        try:
            proc = psutil.Process(pid)
            name = proc.name().lower()

            # Don't throttle critical processes too aggressively
            critical = [
                p.lower() for p in self.config["protection"]["critical_processes"]
            ]
            return not any(crit in name for crit in critical)

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def optimize_memory_usage(self) -> dict:
        """Optimize system memory usage"""
        optimization_results = {
            "memory_freed_mb": 0,
            "processes_optimized": 0,
            "swap_cleared": False,
        }

        # Get current memory usage
        memory = psutil.virtual_memory()

        if memory.percent > 80:  # Only optimize if memory usage is high
            # Find memory-hungry processes
            processes = []
            for proc in psutil.process_iter(
                ["pid", "name", "memory_info", "memory_percent"]
            ):
                try:
                    if proc.info["memory_percent"] > 5:  # Processes using >5% memory
                        processes.append(
                            (proc.info["pid"], proc.info["memory_info"].rss)
                        )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by memory usage
            processes.sort(key=lambda x: x[1], reverse=True)

            # Optimize top memory consumers
            for pid, _memory_bytes in processes[:10]:
                try:
                    proc = psutil.Process(pid)

                    # Skip critical processes
                    if not self._can_throttle_process(pid):
                        continue

                    # Lower priority to encourage swapping
                    original_nice = proc.nice()
                    if original_nice < 15:
                        proc.nice(min(original_nice + 5, 19))
                        optimization_results["processes_optimized"] += 1

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Force garbage collection if possible
            try:
                import gc

                gc.collect()
            except:
                pass

        return optimization_results

    def create_resource_usage_report(self) -> dict:
        """Create comprehensive resource usage report"""
        resources = self.get_system_resources()

        # Get top processes by different metrics
        top_cpu = []
        top_memory = []

        for proc in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_info", "memory_percent"]
        ):
            try:
                top_cpu.append(
                    {
                        "pid": proc.info["pid"],
                        "name": proc.info["name"],
                        "cpu_percent": proc.info["cpu_percent"],
                    }
                )
                top_memory.append(
                    {
                        "pid": proc.info["pid"],
                        "name": proc.info["name"],
                        "memory_percent": proc.info["memory_percent"],
                        "memory_mb": proc.info["memory_info"].rss / 1024**2,
                    }
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        top_cpu.sort(key=lambda x: x["cpu_percent"], reverse=True)
        top_memory.sort(key=lambda x: x["memory_percent"], reverse=True)

        return {
            "timestamp": datetime.now().isoformat(),
            "system_resources": resources,
            "top_cpu_processes": top_cpu[:10],
            "top_memory_processes": top_memory[:10],
            "active_constraints": len(self.process_constraints),
            "recent_alerts": len(
                [
                    a
                    for a in self.alerts
                    if datetime.now() - a.timestamp < timedelta(hours=1)
                ]
            ),
            "enforcement_stats": self.enforcement_stats,
            "resource_limits": [asdict(limit) for limit in self.limits],
        }

    async def monitor_and_enforce(self):
        """Main monitoring and enforcement loop"""
        logger.info("Starting Core 4 resource management loop")

        while self.active:
            try:
                # Get system resources
                resources = self.get_system_resources()

                # Store in history
                self.resource_history["system"].append(resources)

                # Check limits and generate alerts
                alerts = self.check_resource_limits(resources)

                # Enforce limits if there are alerts
                if alerts:
                    enforcement_actions = self.enforce_resource_limits(alerts)

                    if any(enforcement_actions.values()):
                        logger.info(f"Enforcement actions taken: {enforcement_actions}")
                        self.enforcement_stats["alerts_generated"] += len(alerts)

                # Periodic memory optimization
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    memory_opt = self.optimize_memory_usage()
                    if memory_opt["processes_optimized"] > 0:
                        logger.info(f"Memory optimization: {memory_opt}")

                await asyncio.sleep(self.config["monitoring"]["interval"])

            except Exception as e:
                logger.error(f"Error in resource management loop: {e}")
                await asyncio.sleep(10)

    def stop(self):
        """Stop the resource manager"""
        self.active = False
        logger.info("Stopping Core 4 resource manager")


def main():
    """Main function for standalone execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Core 4 Resource Manager")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--monitor", action="store_true", help="Start monitoring loop")
    parser.add_argument(
        "--report", action="store_true", help="Generate resource report"
    )
    parser.add_argument("--optimize", action="store_true", help="Optimize memory usage")
    parser.add_argument("--kill-pid", type=int, help="Kill specific process")
    parser.add_argument(
        "--constrain-pid", type=int, help="Apply constraints to process"
    )
    parser.add_argument(
        "--priority", type=int, default=10, help="Priority for constraint"
    )

    args = parser.parse_args()

    manager = Core4ResourceManager(args.config)

    if args.kill_pid:
        if manager._can_kill_process(args.kill_pid):
            try:
                proc = psutil.Process(args.kill_pid)
                proc.terminate()
                print(f"Terminated process {args.kill_pid}")
            except Exception as e:
                print(f"Failed to kill process {args.kill_pid}: {e}")
        else:
            print(f"Cannot kill process {args.kill_pid} - protected")
        return

    if args.constrain_pid:
        constraint = ProcessConstraint(
            pid=args.constrain_pid,
            name=psutil.Process(args.constrain_pid).name(),
            max_cpu_percent=50,
            max_memory_mb=1024,
            max_open_files=512,
            priority=args.priority,
        )
        success = manager.apply_process_constraint(args.constrain_pid, constraint)
        print(f"Constraint application: {'Success' if success else 'Failed'}")
        return

    if args.optimize:
        results = manager.optimize_memory_usage()
        print(f"Memory optimization results: {results}")
        return

    if args.report:
        report = manager.create_resource_usage_report()
        print(json.dumps(report, indent=2, default=str))
        return

    # Show current status
    resources = manager.get_system_resources()
    print("System Resource Usage:")
    print(
        f"CPU: {resources['cpu']['percent']:.1f}% (Load: {resources['cpu']['load_avg_1m']:.2f})"
    )
    print(
        f"Memory: {resources['memory']['percent']:.1f}% "
        f"({resources['memory']['used_gb']:.1f}GB / {resources['memory']['total_gb']:.1f}GB)"
    )
    print(
        f"Disk: {resources['disk']['percent']:.1f}% "
        f"({resources['disk']['used_gb']:.1f}GB / {resources['disk']['total_gb']:.1f}GB)"
    )
    print(f"Processes: {resources['processes']['count']}")

    if args.monitor:
        print("\nStarting resource monitoring (Press Ctrl+C to stop)...")
        try:
            asyncio.run(manager.monitor_and_enforce())
        except KeyboardInterrupt:
            print("\nResource monitoring stopped by user")


if __name__ == "__main__":
    main()
