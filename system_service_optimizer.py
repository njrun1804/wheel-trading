#!/usr/bin/env python3
"""
System Service Optimizer
Optimizes system services and background processes to reduce memory and CPU usage.
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class SystemServiceOptimizer:
    def __init__(self):
        self.setup_logging()

        # Services that can be safely optimized
        self.optimizable_services = {
            # Background processes that can be throttled
            "backgroundTaskManagement": {"action": "throttle", "priority": "low"},
            "cloudpaird": {"action": "throttle", "priority": "low"},
            "ContactsAgent": {"action": "throttle", "priority": "low"},
            "fileproviderd": {"action": "renice", "priority": "high"},
            "nsurlsessiond": {"action": "throttle", "priority": "medium"},
            "photoanalysisd": {"action": "throttle", "priority": "low"},
            "suggestd": {"action": "throttle", "priority": "low"},
            "remindd": {"action": "throttle", "priority": "low"},
            "callservicesd": {"action": "throttle", "priority": "low"},
            "proactiveeventtrackerd": {"action": "throttle", "priority": "low"},
            "intelligenceplatformd": {"action": "throttle", "priority": "low"},
            "biomed": {"action": "throttle", "priority": "low"},
            "coreduetd": {"action": "throttle", "priority": "low"},
            "rapportd": {"action": "throttle", "priority": "low"},
            "findmydeviced": {"action": "throttle", "priority": "low"},
            "sharingd": {"action": "throttle", "priority": "low"},
            "assistantd": {"action": "throttle", "priority": "low"},
            "aned": {"action": "throttle", "priority": "low"},
            "avconferenced": {"action": "throttle", "priority": "low"},
            "coreaudiod": {"action": "renice", "priority": "medium"},
            "CommCenter": {"action": "throttle", "priority": "low"},
            "bluetoothd": {"action": "renice", "priority": "medium"},
            "wifid": {"action": "renice", "priority": "medium"},
            "networkd": {"action": "renice", "priority": "high"},
            "mDNSResponder": {"action": "renice", "priority": "high"},
            "WindowServer": {"action": "protect", "priority": "critical"},
            "loginwindow": {"action": "protect", "priority": "critical"},
            "SystemUIServer": {"action": "protect", "priority": "critical"},
            "Dock": {"action": "protect", "priority": "critical"},
            "Finder": {"action": "protect", "priority": "critical"},
        }

        # Services that should never be touched
        self.protected_services = {
            "kernel_task",
            "launchd",
            "kextd",
            "UserEventAgent",
            "cfprefsd",
            "distnoted",
            "notifyd",
            "syslogd",
            "powerd",
            "configd",
        }

        self.optimization_results = []

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/service_optimization.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("ServiceOptimizer")

    def get_system_load(self) -> dict:
        """Get current system load information."""
        try:
            # Get load averages
            load_result = subprocess.run(["uptime"], capture_output=True, text=True)
            load_line = load_result.stdout.strip()

            # Parse load averages
            load_info = {}
            if "load average" in load_line:
                load_part = load_line.split("load average:")[1].strip()
                loads = [float(x.strip()) for x in load_part.split(",")]
                load_info = {
                    "load_1m": loads[0] if len(loads) > 0 else 0,
                    "load_5m": loads[1] if len(loads) > 1 else 0,
                    "load_15m": loads[2] if len(loads) > 2 else 0,
                }

            # Get process count
            ps_result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
            process_count = (
                len(ps_result.stdout.strip().split("\n")) - 1
            )  # Subtract header

            # Get memory pressure
            try:
                memory_result = subprocess.run(
                    ["memory_pressure"], capture_output=True, text=True
                )
                memory_pressure = "normal"
                if "critical" in memory_result.stdout.lower():
                    memory_pressure = "critical"
                elif "warning" in memory_result.stdout.lower():
                    memory_pressure = "warning"
            except:
                memory_pressure = "unknown"

            return {
                "timestamp": datetime.now().isoformat(),
                "load_averages": load_info,
                "process_count": process_count,
                "memory_pressure": memory_pressure,
                "uptime": load_line,
            }

        except Exception as e:
            self.logger.error(f"Error getting system load: {e}")
            return {"error": str(e)}

    def get_service_status(self) -> list[dict]:
        """Get status of system services and processes."""
        services = []

        try:
            # Get all processes with detailed info
            ps_result = subprocess.run(
                ["ps", "axo", "pid,ppid,pcpu,pmem,vsz,rss,time,comm,args"],
                capture_output=True,
                text=True,
            )

            lines = ps_result.stdout.strip().split("\n")[1:]  # Skip header

            for line in lines:
                parts = line.strip().split(None, 8)
                if len(parts) >= 8:
                    try:
                        service_info = {
                            "pid": int(parts[0]),
                            "ppid": int(parts[1]),
                            "cpu_percent": float(parts[2]),
                            "memory_percent": float(parts[3]),
                            "vsz": int(parts[4]),
                            "rss": int(parts[5]),
                            "time": parts[6],
                            "command": parts[7],
                            "full_command": parts[8] if len(parts) > 8 else parts[7],
                            "service_name": self._extract_service_name(parts[7]),
                        }
                        services.append(service_info)
                    except (ValueError, IndexError):
                        continue

        except Exception as e:
            self.logger.error(f"Error getting service status: {e}")

        return services

    def _extract_service_name(self, command: str) -> str:
        """Extract service name from command."""
        # Remove path and get just the executable name
        service_name = command.split("/")[-1]
        # Remove common extensions
        service_name = service_name.replace(".app", "").replace(".framework", "")
        return service_name

    def analyze_services(self, services: list[dict]) -> dict:
        """Analyze services for optimization opportunities."""
        analysis = {
            "high_cpu_services": [],
            "high_memory_services": [],
            "optimizable_services": [],
            "protected_services": [],
            "total_services": len(services),
            "total_cpu_usage": 0,
            "total_memory_usage": 0,
        }

        for service in services:
            service_name = service["service_name"]

            # Calculate totals
            analysis["total_cpu_usage"] += service["cpu_percent"]
            analysis["total_memory_usage"] += service["memory_percent"]

            # Identify high resource usage
            if service["cpu_percent"] > 10:
                analysis["high_cpu_services"].append(service)

            if service["memory_percent"] > 5:
                analysis["high_memory_services"].append(service)

            # Check if service is optimizable
            if service_name in self.optimizable_services:
                service["optimization"] = self.optimizable_services[service_name]
                analysis["optimizable_services"].append(service)

            # Check if service is protected
            if service_name in self.protected_services:
                analysis["protected_services"].append(service)

        # Sort by resource usage
        analysis["high_cpu_services"].sort(key=lambda x: x["cpu_percent"], reverse=True)
        analysis["high_memory_services"].sort(
            key=lambda x: x["memory_percent"], reverse=True
        )

        return analysis

    def optimize_service(self, service: dict) -> dict:
        """Optimize a specific service."""
        service_name = service["service_name"]
        pid = service["pid"]
        optimization = service.get("optimization", {})
        action = optimization.get("action", "none")

        result = {
            "service": service_name,
            "pid": pid,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "details": "",
        }

        try:
            if action == "renice":
                # Adjust process priority
                nice_value = 10 if optimization.get("priority") == "low" else 5
                subprocess.run(
                    ["sudo", "renice", str(nice_value), str(pid)], check=True
                )
                result["success"] = True
                result["details"] = f"Reniced to {nice_value}"
                self.logger.info(f"Reniced {service_name} (PID: {pid}) to {nice_value}")

            elif action == "throttle":
                # For background processes, we can try to limit their CPU usage
                # This is a more complex operation that might require system-level controls
                result["details"] = "Throttling not implemented yet"
                self.logger.info(f"Would throttle {service_name} (PID: {pid})")

            elif action == "protect":
                result["details"] = "Service is protected from optimization"
                result["success"] = True

            else:
                result["details"] = "No optimization action defined"

        except subprocess.CalledProcessError as e:
            result["details"] = f"Command failed: {e}"
            self.logger.error(f"Failed to optimize {service_name}: {e}")
        except Exception as e:
            result["details"] = f"Error: {e}"
            self.logger.error(f"Error optimizing {service_name}: {e}")

        return result

    def run_optimization(self) -> dict:
        """Run comprehensive system optimization."""
        self.logger.info("Starting system service optimization")

        # Get current system state
        system_load = self.get_system_load()
        services = self.get_service_status()
        analysis = self.analyze_services(services)

        # Perform optimizations
        optimization_results = []
        optimized_count = 0

        for service in analysis["optimizable_services"]:
            # Only optimize services that are using significant resources
            if service["cpu_percent"] > 1 or service["memory_percent"] > 1:
                result = self.optimize_service(service)
                optimization_results.append(result)
                if result["success"]:
                    optimized_count += 1

        # Generate summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "system_load_before": system_load,
            "services_analyzed": len(services),
            "optimizations_attempted": len(optimization_results),
            "optimizations_successful": optimized_count,
            "high_cpu_services": len(analysis["high_cpu_services"]),
            "high_memory_services": len(analysis["high_memory_services"]),
            "total_cpu_usage": analysis["total_cpu_usage"],
            "total_memory_usage": analysis["total_memory_usage"],
            "optimization_results": optimization_results,
        }

        self.optimization_results.append(summary)

        # Save results
        self.save_optimization_results(summary)

        self.logger.info(
            f"Optimization complete: {optimized_count}/{len(optimization_results)} successful"
        )

        return summary

    def save_optimization_results(self, results: dict):
        """Save optimization results to file."""
        results_file = (
            Path("logs")
            / f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Optimization results saved to: {results_file}")

    def create_optimization_script(self) -> str:
        """Create a script for ongoing optimization."""
        script_content = """#!/bin/bash
# System Service Optimization Script
# Generated by SystemServiceOptimizer

LOG_FILE="logs/service_optimization.log"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

optimize_fileproviderd() {
    # Find fileproviderd processes using high CPU
    FILEPROVIDERD_PIDS=$(ps aux | grep fileproviderd | grep -v grep | awk '$3 > 50 {print $2}')
    
    for pid in $FILEPROVIDERD_PIDS; do
        if [ ! -z "$pid" ]; then
            log_message "Renicing fileproviderd PID $pid"
            sudo renice 10 $pid 2>/dev/null
        fi
    done
}

optimize_background_processes() {
    # List of background processes to optimize
    PROCESSES=("photoanalysisd" "suggestd" "remindd" "callservicesd" "proactiveeventtrackerd")
    
    for process in "${PROCESSES[@]}"; do
        PIDS=$(ps aux | grep "$process" | grep -v grep | awk '$3 > 5 {print $2}')
        for pid in $PIDS; do
            if [ ! -z "$pid" ]; then
                log_message "Renicing $process PID $pid"
                sudo renice 15 $pid 2>/dev/null
            fi
        done
    done
}

check_system_load() {
    LOAD_1M=$(uptime | awk -F'load average:' '{print $2}' | awk -F, '{print $1}' | xargs)
    
    if (( $(echo "$LOAD_1M > 8" | bc -l) )); then
        log_message "High system load detected: $LOAD_1M"
        optimize_fileproviderd
        optimize_background_processes
    fi
}

# Main execution
log_message "Starting system optimization check"
check_system_load
log_message "Optimization check completed"
"""

        script_path = Path("system_optimization.sh")
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        self.logger.info(f"Created optimization script: {script_path}")
        return str(script_path)

    def get_optimization_recommendations(self, analysis: dict) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # System load recommendations
        system_load = self.get_system_load()
        load_1m = system_load.get("load_averages", {}).get("load_1m", 0)

        if load_1m > 8:
            recommendations.append(
                "CRITICAL: System load is very high - immediate optimization needed"
            )
        elif load_1m > 4:
            recommendations.append(
                "WARNING: System load is elevated - consider optimization"
            )

        # CPU usage recommendations
        if analysis["total_cpu_usage"] > 200:
            recommendations.append(
                "High total CPU usage - optimize background processes"
            )

        # Memory usage recommendations
        if analysis["total_memory_usage"] > 80:
            recommendations.append(
                "High memory usage - consider terminating unnecessary processes"
            )

        # Service-specific recommendations
        if len(analysis["high_cpu_services"]) > 5:
            recommendations.append(
                f"Multiple high-CPU services detected: {len(analysis['high_cpu_services'])}"
            )

        if len(analysis["high_memory_services"]) > 10:
            recommendations.append(
                f"Multiple high-memory services detected: {len(analysis['high_memory_services'])}"
            )

        # fileproviderd specific
        fileproviderd_services = [
            s
            for s in analysis["high_cpu_services"]
            if "fileproviderd" in s["service_name"]
        ]
        if fileproviderd_services:
            recommendations.append(
                "fileproviderd is using high CPU - consider optimization"
            )

        return recommendations

    def generate_report(self) -> str:
        """Generate comprehensive optimization report."""
        system_load = self.get_system_load()
        services = self.get_service_status()
        analysis = self.analyze_services(services)
        recommendations = self.get_optimization_recommendations(analysis)

        report = f"""
=== SYSTEM SERVICE OPTIMIZATION REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYSTEM LOAD:
- 1-minute load: {system_load.get('load_averages', {}).get('load_1m', 'N/A')}
- 5-minute load: {system_load.get('load_averages', {}).get('load_5m', 'N/A')}
- 15-minute load: {system_load.get('load_averages', {}).get('load_15m', 'N/A')}
- Process count: {system_load.get('process_count', 'N/A')}
- Memory pressure: {system_load.get('memory_pressure', 'N/A')}

SERVICE ANALYSIS:
- Total services: {analysis['total_services']}
- High CPU services: {len(analysis['high_cpu_services'])}
- High memory services: {len(analysis['high_memory_services'])}
- Optimizable services: {len(analysis['optimizable_services'])}
- Total CPU usage: {analysis['total_cpu_usage']:.1f}%
- Total memory usage: {analysis['total_memory_usage']:.1f}%

TOP HIGH-CPU SERVICES:
"""

        for service in analysis["high_cpu_services"][:10]:
            report += f"  {service['service_name']} (PID: {service['pid']}): {service['cpu_percent']:.1f}% CPU\n"

        report += "\nTOP HIGH-MEMORY SERVICES:\n"
        for service in analysis["high_memory_services"][:10]:
            report += f"  {service['service_name']} (PID: {service['pid']}): {service['memory_percent']:.1f}% Memory\n"

        if analysis["optimizable_services"]:
            report += (
                f"\nOPTIMIZABLE SERVICES ({len(analysis['optimizable_services'])}):\n"
            )
            for service in analysis["optimizable_services"][:10]:
                opt = service.get("optimization", {})
                report += f"  {service['service_name']}: {opt.get('action', 'N/A')} ({opt.get('priority', 'N/A')} priority)\n"

        if recommendations:
            report += "\nRECOMMENDATIONS:\n"
            for rec in recommendations:
                report += f"  • {rec}\n"

        report += "\n=== END REPORT ===\n"
        return report


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="System Service Optimizer")
    parser.add_argument("--optimize", action="store_true", help="Run optimization")
    parser.add_argument("--report", action="store_true", help="Generate report only")
    parser.add_argument(
        "--create-script", action="store_true", help="Create optimization script"
    )

    args = parser.parse_args()

    optimizer = SystemServiceOptimizer()

    if args.optimize:
        result = optimizer.run_optimization()
        print(
            f"Optimization completed: {result['optimizations_successful']}/{result['optimizations_attempted']} successful"
        )

    elif args.create_script:
        script_path = optimizer.create_optimization_script()
        print(f"Optimization script created: {script_path}")

    else:
        # Generate report (default)
        report = optimizer.generate_report()
        print(report)


if __name__ == "__main__":
    main()
