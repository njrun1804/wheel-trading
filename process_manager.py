#!/usr/bin/env python3
"""
Process Manager - Automated Resource Management System
Identifies and manages high-CPU/memory processes with focus on fileproviderd and Claude instances.
"""

import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("process_manager.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessInfo:
    """Process information container"""

    pid: int
    ppid: int
    cpu_percent: float
    memory_percent: float
    vsz: int  # Virtual memory size
    rss: int  # Resident set size
    runtime: str
    command: str
    status: str = "running"


class ProcessManager:
    """Advanced process management and monitoring system"""

    def __init__(self):
        self.config = {
            "cpu_threshold": 80.0,  # CPU % threshold for intervention
            "memory_threshold": 70.0,  # Memory % threshold
            "fileproviderd_cpu_limit": 50.0,  # Special limit for fileproviderd
            "claude_memory_limit": 8.0,  # Memory % limit per Claude instance
            "max_claude_instances": 2,  # Maximum allowed Claude instances
            "monitoring_interval": 30,  # Seconds between checks
            "grace_period": 300,  # Seconds before killing process
            "log_retention_days": 7,
        }

        self.process_history = []
        self.intervention_log = []
        self.blocked_processes = set()

    def get_system_info(self) -> dict:
        """Get comprehensive system resource information"""
        try:
            # Get memory stats
            vm_stat = subprocess.run(["vm_stat"], capture_output=True, text=True)
            memory_pressure = subprocess.run(
                ["memory_pressure"], capture_output=True, text=True
            )

            # Parse memory info
            vm_lines = vm_stat.stdout.split("\n")
            memory_info = {}
            for line in vm_lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().replace('"', "").replace(" ", "_").lower()
                    value = value.strip().rstrip(".")
                    if value.isdigit():
                        memory_info[key] = int(value)

            # Calculate memory usage
            page_size = 16384  # macOS page size
            total_pages = (
                memory_info.get("pages_free", 0)
                + memory_info.get("pages_active", 0)
                + memory_info.get("pages_inactive", 0)
                + memory_info.get("pages_wired_down", 0)
            )

            free_memory_mb = (memory_info.get("pages_free", 0) * page_size) / (
                1024 * 1024
            )
            total_memory_mb = (total_pages * page_size) / (1024 * 1024)
            used_memory_percent = (
                ((total_memory_mb - free_memory_mb) / total_memory_mb) * 100
                if total_memory_mb > 0
                else 0
            )

            # Get CPU info
            cpu_info = subprocess.run(
                ["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True
            )
            cpu_count = (
                int(cpu_info.stdout.strip()) if cpu_info.stdout.strip().isdigit() else 8
            )

            return {
                "timestamp": datetime.now().isoformat(),
                "cpu_count": cpu_count,
                "total_memory_mb": total_memory_mb,
                "free_memory_mb": free_memory_mb,
                "used_memory_percent": used_memory_percent,
                "memory_pressure_available": "system has"
                in memory_pressure.stdout.lower(),
                "vm_stats": memory_info,
            }

        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def get_process_list(self) -> list[ProcessInfo]:
        """Get detailed process information"""
        processes = []
        try:
            # Use system ps command to avoid alias issues
            cmd = ["/bin/ps", "-axo", "pid,ppid,pcpu,pmem,vsz,rss,time,comm,args"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            lines = result.stdout.strip().split("\n")[1:]  # Skip header

            for line in lines:
                parts = line.strip().split(None, 8)  # Split into 9 parts max
                if len(parts) >= 8:
                    try:
                        pid = int(parts[0])
                        ppid = int(parts[1])
                        cpu_percent = float(parts[2])
                        memory_percent = float(parts[3])
                        vsz = int(parts[4])
                        rss = int(parts[5])
                        runtime = parts[6]
                        command = parts[7] if len(parts) > 7 else ""
                        full_args = parts[8] if len(parts) > 8 else command

                        processes.append(
                            ProcessInfo(
                                pid=pid,
                                ppid=ppid,
                                cpu_percent=cpu_percent,
                                memory_percent=memory_percent,
                                vsz=vsz,
                                rss=rss,
                                runtime=runtime,
                                command=full_args,
                            )
                        )
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error parsing process line: {line}, error: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error getting process list: {e}")

        return processes

    def identify_problem_processes(
        self, processes: list[ProcessInfo]
    ) -> dict[str, list[ProcessInfo]]:
        """Identify processes that need intervention"""
        problem_processes = {
            "high_cpu": [],
            "high_memory": [],
            "fileproviderd": [],
            "claude_instances": [],
            "zombies": [],
        }

        claude_count = 0

        for proc in processes:
            # High CPU processes
            if proc.cpu_percent > self.config["cpu_threshold"]:
                problem_processes["high_cpu"].append(proc)

            # High memory processes
            if proc.memory_percent > self.config["memory_threshold"]:
                problem_processes["high_memory"].append(proc)

            # fileproviderd specific check
            if "fileproviderd" in proc.command.lower():
                if proc.cpu_percent > self.config["fileproviderd_cpu_limit"]:
                    problem_processes["fileproviderd"].append(proc)

            # Claude instances
            if "claude" in proc.command.lower():
                claude_count += 1
                if (
                    proc.memory_percent > self.config["claude_memory_limit"]
                    or claude_count > self.config["max_claude_instances"]
                ):
                    problem_processes["claude_instances"].append(proc)

        return problem_processes

    def handle_fileproviderd(self, proc: ProcessInfo) -> bool:
        """Special handling for fileproviderd process"""
        logger.warning(f"fileproviderd (PID: {proc.pid}) using {proc.cpu_percent}% CPU")

        # Log the intervention
        intervention = {
            "timestamp": datetime.now().isoformat(),
            "process": "fileproviderd",
            "pid": proc.pid,
            "cpu_percent": proc.cpu_percent,
            "action": "renice",
            "details": "Reducing priority to limit CPU usage",
        }

        try:
            # First, try to renice the process to lower priority
            subprocess.run(["sudo", "renice", "10", str(proc.pid)], check=True)
            logger.info(f"Successfully reniced fileproviderd PID {proc.pid}")
            intervention["status"] = "success"

            # If CPU is still extremely high, try more aggressive measures
            if proc.cpu_percent > 95.0:
                logger.warning(
                    "fileproviderd still using excessive CPU, attempting restart"
                )
                # Note: fileproviderd usually restarts automatically when killed
                os.kill(proc.pid, signal.SIGTERM)
                intervention["action"] = "restart"
                intervention[
                    "details"
                ] = "Terminated for automatic restart due to extreme CPU usage"

        except Exception as e:
            logger.error(f"Error handling fileproviderd: {e}")
            intervention["status"] = "failed"
            intervention["error"] = str(e)

        self.intervention_log.append(intervention)
        return intervention.get("status") == "success"

    def manage_claude_instances(self, claude_processes: list[ProcessInfo]) -> bool:
        """Manage Claude instances - limit count and memory usage"""
        if len(claude_processes) <= self.config["max_claude_instances"]:
            return True

        logger.warning(
            f"Found {len(claude_processes)} Claude instances, max allowed: {self.config['max_claude_instances']}"
        )

        # Sort by memory usage (highest first) and runtime (oldest first)
        claude_processes.sort(key=lambda p: (p.memory_percent, p.runtime), reverse=True)

        # Kill excess instances
        excess_count = len(claude_processes) - self.config["max_claude_instances"]
        killed_count = 0

        for proc in claude_processes[:excess_count]:
            try:
                logger.info(
                    f"Terminating excess Claude instance PID {proc.pid} (Memory: {proc.memory_percent}%)"
                )
                os.kill(proc.pid, signal.SIGTERM)

                # Give it a chance to shut down gracefully
                time.sleep(5)

                # Check if it's still running
                try:
                    os.kill(proc.pid, 0)  # Check if process exists
                    logger.warning(
                        f"Claude PID {proc.pid} didn't terminate gracefully, using SIGKILL"
                    )
                    os.kill(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Process terminated successfully

                killed_count += 1

                intervention = {
                    "timestamp": datetime.now().isoformat(),
                    "process": "claude",
                    "pid": proc.pid,
                    "memory_percent": proc.memory_percent,
                    "action": "terminate",
                    "reason": "excess_instances",
                    "status": "success",
                }
                self.intervention_log.append(intervention)

            except Exception as e:
                logger.error(f"Error terminating Claude PID {proc.pid}: {e}")

        return killed_count > 0

    def create_resource_monitor(self) -> str:
        """Create a continuous resource monitoring script"""
        monitor_script = """#!/bin/bash
# Resource Monitor - Continuous system monitoring
# Generated by ProcessManager

LOG_FILE="resource_monitor.log"
ALERT_THRESHOLD_CPU=80
ALERT_THRESHOLD_MEM=70

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

check_resources() {
    # Get top CPU processes
    TOP_CPU=$(ps -axo pid,pcpu,pmem,comm --sort=-pcpu | head -6 | tail -5)
    
    # Get memory usage
    MEM_USAGE=$(vm_stat | grep -E "(Pages free|Pages active)" | awk '{print $3}' | tr -d '.' | paste -sd+ | bc)
    
    # Check for high CPU processes
    HIGH_CPU=$(ps -axo pid,pcpu,comm --sort=-pcpu | awk -v threshold=$ALERT_THRESHOLD_CPU '$2 > threshold {print $0}' | head -3)
    
    if [ ! -z "$HIGH_CPU" ]; then
        log_message "HIGH CPU ALERT:"
        echo "$HIGH_CPU" | while read line; do
            log_message "  $line"
        done
    fi
    
    # Log current top processes
    log_message "Top CPU processes:"
    echo "$TOP_CPU" | while read line; do
        log_message "  $line"
    done
}

# Main monitoring loop
log_message "Resource monitor started"
while true; do
    check_resources
    sleep 60  # Check every minute
done
"""

        script_path = Path("resource_monitor.sh")
        script_path.write_text(monitor_script)
        script_path.chmod(0o755)

        logger.info(f"Created resource monitor script: {script_path}")
        return str(script_path)

    def cleanup_old_logs(self):
        """Clean up old log files"""
        try:
            cutoff_date = datetime.now() - timedelta(
                days=self.config["log_retention_days"]
            )

            for log_file in ["process_manager.log", "resource_monitor.log"]:
                if os.path.exists(log_file):
                    file_time = datetime.fromtimestamp(os.path.getmtime(log_file))
                    if file_time < cutoff_date:
                        os.remove(log_file)
                        logger.info(f"Removed old log file: {log_file}")

        except Exception as e:
            logger.error(f"Error cleaning up logs: {e}")

    def generate_report(self, system_info: dict, problem_processes: dict) -> str:
        """Generate a comprehensive system report"""
        report = f"""
=== SYSTEM RESOURCE ANALYSIS REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYSTEM OVERVIEW:
- Total Memory: {system_info.get('total_memory_mb', 0):.1f} MB
- Free Memory: {system_info.get('free_memory_mb', 0):.1f} MB
- Memory Usage: {system_info.get('used_memory_percent', 0):.1f}%
- CPU Cores: {system_info.get('cpu_count', 'Unknown')}

PROBLEM PROCESSES IDENTIFIED:

High CPU Processes ({len(problem_processes['high_cpu'])}):
"""

        for proc in problem_processes["high_cpu"]:
            report += (
                f"  PID {proc.pid}: {proc.cpu_percent}% CPU - {proc.command[:60]}\n"
            )

        report += (
            f"\nHigh Memory Processes ({len(problem_processes['high_memory'])}):\n"
        )
        for proc in problem_processes["high_memory"]:
            report += f"  PID {proc.pid}: {proc.memory_percent}% Memory - {proc.command[:60]}\n"

        if problem_processes["fileproviderd"]:
            report += (
                f"\nfileproviderd Issues ({len(problem_processes['fileproviderd'])}):\n"
            )
            for proc in problem_processes["fileproviderd"]:
                report += f"  PID {proc.pid}: {proc.cpu_percent}% CPU, Runtime: {proc.runtime}\n"

        if problem_processes["claude_instances"]:
            report += f"\nClaude Instance Issues ({len(problem_processes['claude_instances'])}):\n"
            for proc in problem_processes["claude_instances"]:
                report += f"  PID {proc.pid}: {proc.memory_percent}% Memory - {proc.command[:40]}\n"

        report += f"\nRECENT INTERVENTIONS ({len(self.intervention_log)}):\n"
        for intervention in self.intervention_log[-5:]:  # Last 5 interventions
            report += f"  {intervention['timestamp']}: {intervention['action']} on {intervention['process']} (PID {intervention['pid']})\n"

        report += "\n=== END REPORT ===\n"
        return report

    def run_single_check(self) -> str:
        """Run a single system check and return report"""
        logger.info("Starting system resource check...")

        # Get system information
        system_info = self.get_system_info()

        # Get process list
        processes = self.get_process_list()
        logger.info(f"Found {len(processes)} processes")

        # Identify problem processes
        problem_processes = self.identify_problem_processes(processes)

        # Handle fileproviderd specifically
        for proc in problem_processes["fileproviderd"]:
            self.handle_fileproviderd(proc)

        # Manage Claude instances
        if problem_processes["claude_instances"]:
            self.manage_claude_instances(problem_processes["claude_instances"])

        # Generate and save report
        report = self.generate_report(system_info, problem_processes)

        # Save detailed data
        detailed_data = {
            "system_info": system_info,
            "problem_processes": {
                key: [
                    {
                        "pid": p.pid,
                        "cpu": p.cpu_percent,
                        "memory": p.memory_percent,
                        "command": p.command,
                    }
                    for p in procs
                ]
                for key, procs in problem_processes.items()
            },
            "intervention_log": self.intervention_log,
            "timestamp": datetime.now().isoformat(),
        }

        with open("process_analysis.json", "w") as f:
            json.dump(detailed_data, f, indent=2)

        return report

    def start_monitoring(self):
        """Start continuous monitoring"""
        logger.info("Starting continuous process monitoring...")

        # Create resource monitor script
        self.create_resource_monitor()

        try:
            while True:
                self.run_single_check()
                logger.info("Check completed")

                # Clean up old logs periodically
                if datetime.now().minute == 0:  # Every hour
                    self.cleanup_old_logs()

                time.sleep(self.config["monitoring_interval"])

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")


def main():
    """Main function with command line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process Manager - Resource Control System"
    )
    parser.add_argument("--check", action="store_true", help="Run single check")
    parser.add_argument(
        "--monitor", action="store_true", help="Start continuous monitoring"
    )
    parser.add_argument(
        "--kill-claude", action="store_true", help="Kill excess Claude instances"
    )
    parser.add_argument(
        "--fix-fileproviderd", action="store_true", help="Fix fileproviderd CPU usage"
    )
    parser.add_argument("--report", action="store_true", help="Generate system report")
    parser.add_argument("--config", help="Custom config file path")

    args = parser.parse_args()

    manager = ProcessManager()

    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            custom_config = json.load(f)
            manager.config.update(custom_config)

    if args.check or not any(
        [args.monitor, args.kill_claude, args.fix_fileproviderd, args.report]
    ):
        report = manager.run_single_check()
        print(report)

    elif args.monitor:
        manager.start_monitoring()

    elif args.kill_claude:
        processes = manager.get_process_list()
        claude_processes = [p for p in processes if "claude" in p.command.lower()]
        if claude_processes:
            manager.manage_claude_instances(claude_processes)
            print(f"Managed {len(claude_processes)} Claude instances")
        else:
            print("No Claude instances found")

    elif args.fix_fileproviderd:
        processes = manager.get_process_list()
        fileproviderd_processes = [
            p for p in processes if "fileproviderd" in p.command.lower()
        ]
        for proc in fileproviderd_processes:
            manager.handle_fileproviderd(proc)
        print(f"Handled {len(fileproviderd_processes)} fileproviderd processes")

    elif args.report:
        system_info = manager.get_system_info()
        processes = manager.get_process_list()
        problem_processes = manager.identify_problem_processes(processes)
        report = manager.generate_report(system_info, problem_processes)
        print(report)


if __name__ == "__main__":
    main()
