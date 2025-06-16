#!/usr/bin/env python3
"""
Core 4 Manager - Unified management of all Core 4 process monitoring components
Orchestrates process monitor, resource manager, and system monitor
"""

import asyncio
import json
import logging
import signal
from datetime import datetime, timedelta
from pathlib import Path

import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("core4_manager.log"), logging.StreamHandler()],
)
logger = logging.getLogger("Core4Manager")


class Core4Manager:
    """Unified Core 4 process and resource management system"""

    def __init__(self, config_file: str | None = None):
        self.config = self._load_config(config_file)
        self.components = {}
        self.running = False
        self.tasks = []
        self.stats = {
            "start_time": datetime.now(),
            "processes_monitored": 0,
            "processes_cleaned": 0,
            "alerts_generated": 0,
            "actions_taken": 0,
            "uptime_seconds": 0,
        }

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self, config_file: str | None) -> dict:
        """Load configuration"""
        default_config = {
            "components": {
                "process_monitor": {
                    "enabled": True,
                    "interval": 5,
                    "auto_cleanup": True,
                    "cpu_threshold": 80.0,
                    "memory_threshold": 85.0,
                },
                "resource_manager": {
                    "enabled": True,
                    "interval": 10,
                    "enforcement": True,
                    "optimization": True,
                },
                "system_monitor": {
                    "enabled": True,
                    "interval": 15,
                    "alerting": True,
                    "email_notifications": False,
                },
            },
            "global_settings": {
                "log_level": "INFO",
                "data_retention_days": 7,
                "emergency_mode": False,
                "protected_processes": [
                    "kernel_task",
                    "launchd",
                    "WindowServer",
                    "loginwindow",
                    "claude",
                    "python",
                    "bash",
                    "zsh",
                    "Terminal",
                    "WezTerm",
                ],
            },
            "thresholds": {
                "cpu_critical": 95.0,
                "memory_critical": 90.0,
                "disk_critical": 95.0,
                "process_cpu_limit": 75.0,
                "process_memory_limit_mb": 2048,
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

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    async def _setup_components(self):
        """Setup monitoring components"""
        try:
            # Import and setup process monitor
            if self.config["components"]["process_monitor"]["enabled"]:
                from core4_process_monitor import Core4ProcessMonitor

                self.components["process_monitor"] = Core4ProcessMonitor()
                logger.info("Process monitor initialized")

            # Import and setup resource manager
            if self.config["components"]["resource_manager"]["enabled"]:
                from core4_resource_manager import Core4ResourceManager

                self.components["resource_manager"] = Core4ResourceManager()
                logger.info("Resource manager initialized")

            # Import and setup system monitor
            if self.config["components"]["system_monitor"]["enabled"]:
                from core4_system_monitor import Core4SystemMonitor

                self.components["system_monitor"] = Core4SystemMonitor()
                logger.info("System monitor initialized")

        except ImportError as e:
            logger.error(f"Failed to import monitoring component: {e}")
            raise

    async def start(self):
        """Start all monitoring components"""
        logger.info("Starting Core 4 Manager")

        # Setup components
        await self._setup_components()

        self.running = True
        self.stats["start_time"] = datetime.now()

        # Start component tasks
        if "process_monitor" in self.components:
            task = asyncio.create_task(
                self.components["process_monitor"].monitor_loop(),
                name="process_monitor",
            )
            self.tasks.append(task)

        if "resource_manager" in self.components:
            task = asyncio.create_task(
                self.components["resource_manager"].monitor_and_enforce(),
                name="resource_manager",
            )
            self.tasks.append(task)

        if "system_monitor" in self.components:
            task = asyncio.create_task(
                self.components["system_monitor"].monitor_loop(), name="system_monitor"
            )
            self.tasks.append(task)

        # Start management loop
        management_task = asyncio.create_task(
            self._management_loop(), name="management_loop"
        )
        self.tasks.append(management_task)

        logger.info(f"Started {len(self.tasks)} monitoring tasks")

        # Wait for all tasks
        try:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in monitoring tasks: {e}")
        finally:
            await self._cleanup()

    async def _management_loop(self):
        """Main management coordination loop"""
        logger.info("Starting Core 4 management loop")

        while self.running:
            try:
                # Update statistics
                self.stats["uptime_seconds"] = int(
                    (datetime.now() - self.stats["start_time"]).total_seconds()
                )

                # Coordinate between components
                await self._coordinate_components()

                # Perform periodic maintenance
                if self.stats["uptime_seconds"] % 300 == 0:  # Every 5 minutes
                    await self._periodic_maintenance()

                # Generate status report
                if self.stats["uptime_seconds"] % 3600 == 0:  # Every hour
                    await self._generate_status_report()

                await asyncio.sleep(30)  # Management loop interval

            except Exception as e:
                logger.error(f"Error in management loop: {e}")
                await asyncio.sleep(60)

    async def _coordinate_components(self):
        """Coordinate actions between monitoring components"""
        try:
            # Get system resources
            system_resources = self._get_system_resources()

            # Check for critical conditions
            if (
                system_resources["cpu_percent"]
                > self.config["thresholds"]["cpu_critical"]
            ):
                await self._handle_critical_cpu()

            if (
                system_resources["memory_percent"]
                > self.config["thresholds"]["memory_critical"]
            ):
                await self._handle_critical_memory()

            # Update component statistics
            self._update_component_stats()

        except Exception as e:
            logger.error(f"Error coordinating components: {e}")

    def _get_system_resources(self) -> dict:
        """Get current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
            "process_count": len(psutil.pids()),
            "timestamp": datetime.now(),
        }

    async def _handle_critical_cpu(self):
        """Handle critical CPU usage"""
        logger.warning("Critical CPU usage detected - coordinating response")

        # Get process monitor to clean up high CPU processes
        if "process_monitor" in self.components:
            monitor = self.components["process_monitor"]
            cleaned = monitor.auto_cleanup_excessive_processes()
            if cleaned:
                self.stats["processes_cleaned"] += len(cleaned)
                self.stats["actions_taken"] += 1
                logger.info(f"Cleaned {len(cleaned)} high CPU processes")

    async def _handle_critical_memory(self):
        """Handle critical memory usage"""
        logger.warning("Critical memory usage detected - coordinating response")

        # Get resource manager to optimize memory
        if "resource_manager" in self.components:
            manager = self.components["resource_manager"]
            result = manager.optimize_memory_usage()
            if result["processes_optimized"] > 0:
                self.stats["actions_taken"] += 1
                logger.info(
                    f"Optimized {result['processes_optimized']} processes for memory"
                )

    def _update_component_stats(self):
        """Update statistics from components"""
        try:
            if "process_monitor" in self.components:
                monitor = self.components["process_monitor"]
                self.stats["processes_cleaned"] += monitor.cleanup_stats.get(
                    "processes_killed", 0
                )

            if "resource_manager" in self.components:
                manager = self.components["resource_manager"]
                self.stats["actions_taken"] += manager.enforcement_stats.get(
                    "processes_throttled", 0
                )

            if "system_monitor" in self.components:
                monitor = self.components["system_monitor"]
                self.stats["alerts_generated"] += len(monitor.active_alerts)

        except Exception as e:
            logger.debug(f"Error updating component stats: {e}")

    async def _periodic_maintenance(self):
        """Perform periodic maintenance tasks"""
        logger.info("Performing periodic maintenance")

        try:
            # Clean up old log files
            await self._cleanup_old_logs()

            # Optimize process priorities
            await self._optimize_system_performance()

            # Check component health
            await self._check_component_health()

        except Exception as e:
            logger.error(f"Error in periodic maintenance: {e}")

    async def _cleanup_old_logs(self):
        """Clean up old log files"""
        log_dir = Path("logs")
        if log_dir.exists():
            cutoff_date = datetime.now() - timedelta(
                days=self.config["global_settings"]["data_retention_days"]
            )

            for log_file in log_dir.glob("*.log*"):
                try:
                    if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                        log_file.unlink()
                        logger.debug(f"Cleaned up old log file: {log_file}")
                except Exception as e:
                    logger.debug(f"Could not clean up log file {log_file}: {e}")

    async def _optimize_system_performance(self):
        """Optimize system performance"""
        if "process_monitor" in self.components:
            monitor = self.components["process_monitor"]
            optimizations = monitor.optimize_process_scheduling()
            if any(optimizations.values()):
                logger.info(f"System optimization: {optimizations}")

    async def _check_component_health(self):
        """Check health of monitoring components"""
        unhealthy_components = []

        for name, component in self.components.items():
            try:
                # Check if component is responsive
                if hasattr(component, "get_monitoring_report"):
                    await asyncio.wait_for(
                        asyncio.to_thread(component.get_monitoring_report), timeout=10
                    )
            except Exception as e:
                logger.warning(f"Component {name} appears unhealthy: {e}")
                unhealthy_components.append(name)

        if unhealthy_components:
            logger.warning(f"Unhealthy components: {unhealthy_components}")

    async def _generate_status_report(self):
        """Generate comprehensive status report"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "manager_stats": self.stats,
                "system_resources": self._get_system_resources(),
                "components": {},
            }

            # Get component reports
            for name, component in self.components.items():
                try:
                    if hasattr(component, "get_monitoring_report"):
                        report["components"][name] = component.get_monitoring_report()
                    elif hasattr(component, "create_resource_usage_report"):
                        report["components"][
                            name
                        ] = component.create_resource_usage_report()
                except Exception as e:
                    logger.warning(f"Could not get report from {name}: {e}")
                    report["components"][name] = {"error": str(e)}

            # Save report
            report_file = (
                f"core4_status_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Generated status report: {report_file}")

        except Exception as e:
            logger.error(f"Failed to generate status report: {e}")

    def get_status(self) -> dict:
        """Get current manager status"""
        system_resources = self._get_system_resources()

        return {
            "running": self.running,
            "uptime_seconds": self.stats["uptime_seconds"],
            "active_components": list(self.components.keys()),
            "active_tasks": len([t for t in self.tasks if not t.done()]),
            "system_resources": system_resources,
            "statistics": self.stats,
            "health_status": self._get_health_status(system_resources),
        }

    def _get_health_status(self, resources: dict) -> str:
        """Determine overall health status"""
        cpu_ok = resources["cpu_percent"] < self.config["thresholds"]["cpu_critical"]
        memory_ok = (
            resources["memory_percent"] < self.config["thresholds"]["memory_critical"]
        )
        disk_ok = resources["disk_percent"] < self.config["thresholds"]["disk_critical"]

        if cpu_ok and memory_ok and disk_ok:
            return "healthy"
        elif resources["cpu_percent"] > 95 or resources["memory_percent"] > 95:
            return "critical"
        else:
            return "warning"

    async def emergency_cleanup(self):
        """Perform emergency cleanup of system resources"""
        logger.warning("Performing emergency cleanup")

        actions_taken = {
            "processes_killed": 0,
            "memory_freed_mb": 0,
            "priority_adjustments": 0,
        }

        try:
            # Use process monitor for aggressive cleanup
            if "process_monitor" in self.components:
                monitor = self.components["process_monitor"]

                # Kill high resource processes
                cleaned = monitor.auto_cleanup_excessive_processes()
                actions_taken["processes_killed"] = len(cleaned)
                actions_taken["memory_freed_mb"] = sum(p.memory_mb for p in cleaned)

                # Optimize priorities
                optimizations = monitor.optimize_process_scheduling()
                actions_taken["priority_adjustments"] = sum(optimizations.values())

            # Use resource manager for memory optimization
            if "resource_manager" in self.components:
                manager = self.components["resource_manager"]
                memory_result = manager.optimize_memory_usage()
                actions_taken["memory_freed_mb"] += memory_result.get(
                    "memory_freed_mb", 0
                )

            logger.info(f"Emergency cleanup completed: {actions_taken}")

        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")

        return actions_taken

    def stop(self):
        """Stop all monitoring components"""
        logger.info("Stopping Core 4 Manager")
        self.running = False

        # Stop component monitoring
        for name, component in self.components.items():
            try:
                if hasattr(component, "stop_monitoring"):
                    component.stop_monitoring()
                elif hasattr(component, "stop"):
                    component.stop()
                logger.info(f"Stopped {name}")
            except Exception as e:
                logger.warning(f"Error stopping {name}: {e}")

        # Cancel tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

    async def _cleanup(self):
        """Final cleanup"""
        logger.info("Performing final cleanup")

        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        # Generate final report
        try:
            final_stats = self.get_status()
            with open("core4_final_report.json", "w") as f:
                json.dump(final_stats, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save final report: {e}")

        logger.info("Core 4 Manager stopped")


async def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Core 4 Manager")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument(
        "--emergency-cleanup", action="store_true", help="Perform emergency cleanup"
    )
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")

    args = parser.parse_args()

    manager = Core4Manager(args.config)

    if args.status:
        status = manager.get_status()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.emergency_cleanup:
        await manager._setup_components()
        result = await manager.emergency_cleanup()
        print(f"Emergency cleanup result: {result}")
        return

    try:
        await manager.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Manager failed: {e}")
        raise
    finally:
        manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
