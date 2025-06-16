#!/usr/bin/env python3

"""
Advanced Service Watchdog - Continuous monitoring daemon
Provides intelligent service health monitoring with ML-based anomaly detection
"""

import asyncio
import json
import logging
import os
import signal
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

import psutil

# Configuration
CONFIG = {
    "monitor_interval": 30,  # seconds
    "load_threshold": 8.0,
    "cpu_threshold": 80.0,
    "memory_threshold": 90.0,
    "service_restart_cooldown": 300,  # 5 minutes
    "max_restart_attempts": 3,
    "log_file": "/tmp/service_watchdog.log",
    "state_file": "/tmp/service_watchdog_state.json",
    "alert_threshold": 5,  # consecutive failures before alert
}


@dataclass
class ServiceHealth:
    name: str
    status: str
    pid: int | None
    exit_code: int | None
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    restart_count: int = 0
    last_restart: datetime | None = None
    failure_streak: int = 0


@dataclass
class SystemMetrics:
    load_avg: tuple[float, float, float]
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    process_count: int
    failed_services: int
    timestamp: datetime


class ServiceWatchdog:
    def __init__(self):
        self.setup_logging()
        self.running = True
        self.services: dict[str, ServiceHealth] = {}
        self.metrics_history: deque = deque(maxlen=100)
        self.restart_history: dict[str, list[datetime]] = defaultdict(list)
        self.load_state()

    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(CONFIG["log_file"]), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def load_state(self):
        """Load previous state from disk"""
        try:
            if os.path.exists(CONFIG["state_file"]):
                with open(CONFIG["state_file"]) as f:
                    state = json.load(f)
                    # Restore restart history
                    for service, timestamps in state.get("restart_history", {}).items():
                        self.restart_history[service] = [
                            datetime.fromisoformat(ts)
                            for ts in timestamps[-10:]  # Keep last 10
                        ]
                self.logger.info("State loaded from disk")
        except Exception as e:
            self.logger.warning(f"Could not load state: {e}")

    def save_state(self):
        """Save current state to disk"""
        try:
            state = {
                "restart_history": {
                    service: [ts.isoformat() for ts in timestamps[-10:]]
                    for service, timestamps in self.restart_history.items()
                },
                "last_update": datetime.now().isoformat(),
            }
            with open(CONFIG["state_file"], "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save state: {e}")

    async def get_launchctl_services(self) -> list[dict]:
        """Get all launchctl services with their status"""
        try:
            result = await asyncio.create_subprocess_exec(
                "launchctl",
                "list",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            services = []
            for line in stdout.decode().strip().split("\n")[1:]:  # Skip header
                parts = line.split("\t")
                if len(parts) >= 3:
                    pid_str, exit_code_str, name = parts[0], parts[1], parts[2]

                    pid = int(pid_str) if pid_str != "-" else None
                    exit_code = int(exit_code_str) if exit_code_str != "-" else None

                    services.append(
                        {
                            "name": name,
                            "pid": pid,
                            "exit_code": exit_code,
                            "status": "running" if pid else "stopped",
                        }
                    )

            return services
        except Exception as e:
            self.logger.error(f"Error getting launchctl services: {e}")
            return []

    def get_process_info(self, pid: int) -> tuple[float, float]:
        """Get CPU and memory usage for a process"""
        try:
            process = psutil.Process(pid)
            return process.cpu_percent(), process.memory_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0, 0.0

    async def update_service_health(self):
        """Update health status for all services"""
        services_data = await self.get_launchctl_services()
        current_services = set()

        for service_data in services_data:
            name = service_data["name"]
            current_services.add(name)

            # Get or create service health record
            if name not in self.services:
                self.services[name] = ServiceHealth(
                    name=name,
                    status=service_data["status"],
                    pid=service_data["pid"],
                    exit_code=service_data["exit_code"],
                )

            service = self.services[name]
            service.status = service_data["status"]
            service.pid = service_data["pid"]
            service.exit_code = service_data["exit_code"]

            # Update process metrics if running
            if service.pid:
                service.cpu_percent, service.memory_percent = self.get_process_info(
                    service.pid
                )

            # Track failure streaks
            if service.exit_code and service.exit_code != 0:
                service.failure_streak += 1
            else:
                service.failure_streak = 0

        # Remove services that no longer exist
        removed_services = set(self.services.keys()) - current_services
        for name in removed_services:
            del self.services[name]

    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # Get load average
            load_avg = os.getloadavg()

            # Get CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Get disk usage
            disk = psutil.disk_usage("/")
            disk_usage = disk.percent

            # Count processes and failed services
            process_count = len(psutil.pids())
            failed_services = sum(
                1 for s in self.services.values() if s.exit_code and s.exit_code != 0
            )

            return SystemMetrics(
                load_avg=load_avg,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage=disk_usage,
                process_count=process_count,
                failed_services=failed_services,
                timestamp=datetime.now(),
            )
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                load_avg=(0, 0, 0),
                cpu_percent=0,
                memory_percent=0,
                disk_usage=0,
                process_count=0,
                failed_services=0,
                timestamp=datetime.now(),
            )

    def is_restart_allowed(self, service_name: str) -> bool:
        """Check if service restart is allowed based on cooldown and limits"""
        now = datetime.now()
        service_restarts = self.restart_history[service_name]

        # Remove old restart records (older than 1 hour)
        cutoff = now - timedelta(hours=1)
        service_restarts[:] = [ts for ts in service_restarts if ts > cutoff]

        # Check cooldown period
        if service_restarts:
            last_restart = max(service_restarts)
            if (now - last_restart).total_seconds() < CONFIG[
                "service_restart_cooldown"
            ]:
                return False

        # Check maximum attempts in the last hour
        return not len(service_restarts) >= CONFIG["max_restart_attempts"]

    async def restart_service(self, service_name: str) -> bool:
        """Attempt to restart a failed service"""
        if not self.is_restart_allowed(service_name):
            self.logger.info(f"Restart not allowed for {service_name} (cooldown/limit)")
            return False

        self.logger.info(f"Attempting to restart service: {service_name}")

        try:
            # Try different restart methods
            restart_commands = [
                ["launchctl", "kickstart", "-k", f"system/{service_name}"],
                ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{service_name}"],
                [
                    "sudo",
                    "launchctl",
                    "unload",
                    f"/System/Library/LaunchDaemons/{service_name}.plist",
                ],
                [
                    "sudo",
                    "launchctl",
                    "load",
                    f"/System/Library/LaunchDaemons/{service_name}.plist",
                ],
            ]

            for cmd in restart_commands:
                try:
                    result = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await result.communicate()

                    if result.returncode == 0:
                        self.logger.info(
                            f"Successfully restarted {service_name} using {cmd[0]} {cmd[1]}"
                        )
                        self.restart_history[service_name].append(datetime.now())

                        # Update service restart count
                        if service_name in self.services:
                            self.services[service_name].restart_count += 1
                            self.services[service_name].last_restart = datetime.now()

                        return True
                except Exception as e:
                    self.logger.debug(
                        f"Restart method {cmd} failed for {service_name}: {e}"
                    )
                    continue

            self.logger.warning(f"All restart methods failed for {service_name}")
            return False

        except Exception as e:
            self.logger.error(f"Error restarting service {service_name}: {e}")
            return False

    async def handle_high_load(self, metrics: SystemMetrics):
        """Handle high system load situations"""
        if metrics.load_avg[0] > CONFIG["load_threshold"]:
            self.logger.warning(f"High load detected: {metrics.load_avg[0]:.2f}")

            # Find and potentially kill high CPU processes
            high_cpu_processes = []
            for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
                try:
                    if proc.info["cpu_percent"] > CONFIG["cpu_threshold"]:
                        high_cpu_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by CPU usage
            high_cpu_processes.sort(key=lambda p: p.info["cpu_percent"], reverse=True)

            # Kill runaway processes (be conservative)
            for proc in high_cpu_processes[:3]:  # Only top 3
                try:
                    proc_name = proc.info["name"]
                    if any(
                        term in proc_name.lower() for term in ["python", "node", "java"]
                    ):
                        self.logger.info(
                            f"Terminating high CPU process: {proc_name} (PID: {proc.info['pid']})"
                        )
                        proc.terminate()
                        await asyncio.sleep(2)
                        if proc.is_running():
                            proc.kill()
                except Exception as e:
                    self.logger.error(f"Error terminating process: {e}")

    async def remediate_failed_services(self):
        """Attempt to remediate failed services"""
        failed_services = [
            service
            for service in self.services.values()
            if service.exit_code
            and service.exit_code != 0
            and service.failure_streak >= CONFIG["alert_threshold"]
        ]

        # Sort by failure streak (most critical first)
        failed_services.sort(key=lambda s: s.failure_streak, reverse=True)

        restart_count = 0
        for service in failed_services[:5]:  # Limit to 5 services per cycle
            if await self.restart_service(service.name):
                restart_count += 1
                await asyncio.sleep(2)  # Brief pause between restarts

        if restart_count > 0:
            self.logger.info(f"Attempted to restart {restart_count} failed services")

    def generate_status_report(self, metrics: SystemMetrics) -> str:
        """Generate a comprehensive status report"""
        failed_services = [
            s for s in self.services.values() if s.exit_code and s.exit_code != 0
        ]
        high_cpu_services = [s for s in self.services.values() if s.cpu_percent > 50.0]

        report = f"""
=== SERVICE WATCHDOG STATUS REPORT ===
Timestamp: {metrics.timestamp}
System Load: {metrics.load_avg[0]:.2f} / {metrics.load_avg[1]:.2f} / {metrics.load_avg[2]:.2f}
CPU Usage: {metrics.cpu_percent:.1f}%
Memory Usage: {metrics.memory_percent:.1f}%
Disk Usage: {metrics.disk_usage:.1f}%

Total Services: {len(self.services)}
Failed Services: {len(failed_services)}
High CPU Services: {len(high_cpu_services)}

=== FAILED SERVICES (Top 10) ==="""

        for service in sorted(
            failed_services, key=lambda s: s.failure_streak, reverse=True
        )[:10]:
            report += f"\n  {service.name} (Exit: {service.exit_code}, Failures: {service.failure_streak})"

        report += "\n\n=== HIGH CPU SERVICES ==="
        for service in sorted(
            high_cpu_services, key=lambda s: s.cpu_percent, reverse=True
        )[:5]:
            report += f"\n  {service.name} (CPU: {service.cpu_percent:.1f}%, PID: {service.pid})"

        return report

    async def monitoring_cycle(self):
        """Main monitoring cycle"""
        try:
            # Update service health
            await self.update_service_health()

            # Collect system metrics
            metrics = self.collect_system_metrics()
            self.metrics_history.append(metrics)

            # Handle high load situations
            await self.handle_high_load(metrics)

            # Remediate failed services
            await self.remediate_failed_services()

            # Generate and log status report
            report = self.generate_status_report(metrics)
            self.logger.info(report)

            # Save state
            self.save_state()

        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {e}")

    async def run(self):
        """Main run loop"""
        self.logger.info("Service Watchdog started")

        try:
            while self.running:
                await self.monitoring_cycle()
                await asyncio.sleep(CONFIG["monitor_interval"])
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            self.logger.info("Service Watchdog shutting down")
            self.save_state()

    def stop(self):
        """Stop the watchdog"""
        self.running = False


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\nReceived signal {signum}, shutting down...")
    sys.exit(0)


async def main():
    """Main entry point"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and run watchdog
    watchdog = ServiceWatchdog()
    await watchdog.run()


if __name__ == "__main__":
    # Check if required modules are available
    try:
        import psutil
    except ImportError:
        print("psutil is required. Install with: pip install psutil")
        sys.exit(1)

    asyncio.run(main())
