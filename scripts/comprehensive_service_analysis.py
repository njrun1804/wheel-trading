#!/usr/bin/env python3

"""
Comprehensive Service Analysis Tool
Provides detailed analysis of system services, processes, and performance issues
"""

import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime


class ServiceAnalyzer:
    def __init__(self):
        self.services = {}
        self.processes = {}
        self.system_metrics = {}

    def get_launchctl_services(self) -> list[dict]:
        """Get all launchctl services with detailed information"""
        try:
            result = subprocess.run(
                ["launchctl", "list"], capture_output=True, text=True
            )
            services = []

            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
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
                            "failed": exit_code is not None and exit_code != 0,
                        }
                    )

            return services
        except Exception as e:
            print(f"Error getting launchctl services: {e}")
            return []

    def get_high_cpu_processes(self, threshold: float = 10.0) -> list[dict]:
        """Get processes with high CPU usage"""
        try:
            # Use ps to get process information
            result = subprocess.run(
                ["ps", "-Ao", "pid,pcpu,pmem,comm"], capture_output=True, text=True
            )

            processes = []
            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                parts = line.split(None, 3)  # Split into max 4 parts
                if len(parts) >= 4:
                    pid, cpu, mem, command = parts[0], parts[1], parts[2], parts[3]
                    try:
                        cpu_percent = float(cpu)
                        if cpu_percent > threshold:
                            processes.append(
                                {
                                    "pid": int(pid),
                                    "cpu_percent": cpu_percent,
                                    "memory_percent": float(mem),
                                    "command": command,
                                }
                            )
                    except ValueError:
                        continue

            return sorted(processes, key=lambda x: x["cpu_percent"], reverse=True)
        except Exception as e:
            print(f"Error getting high CPU processes: {e}")
            return []

    def get_system_metrics(self) -> dict:
        """Collect comprehensive system metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "load_avg": [0, 0, 0],
            "memory": {},
            "disk": {},
            "network": {},
            "processes": {},
        }

        try:
            # Load average
            with open("/proc/loadavg") as f:
                load_parts = f.read().split()[:3]
                metrics["load_avg"] = [float(x) for x in load_parts]
        except FileNotFoundError:
            # macOS doesn't have /proc/loadavg, use uptime
            try:
                result = subprocess.run(["uptime"], capture_output=True, text=True)
                load_avg_str = result.stdout.split("load averages: ")[1]
                metrics["load_avg"] = [float(x) for x in load_avg_str.split()]
            except:
                pass

        try:
            # Memory statistics using vm_stat
            result = subprocess.run(["vm_stat"], capture_output=True, text=True)
            for line in result.stdout.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().replace(" ", "_").replace('"', "").lower()
                    try:
                        # Extract number from value
                        value = "".join(filter(str.isdigit, value))
                        if value:
                            metrics["memory"][key] = int(value)
                    except:
                        continue
        except Exception as e:
            print(f"Error getting memory stats: {e}")

        try:
            # Process count
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
            metrics["processes"]["total"] = len(result.stdout.strip().split("\n")) - 1
        except:
            pass

        return metrics

    def analyze_failing_services(self) -> dict:
        """Analyze patterns in failing services"""
        services = self.get_launchctl_services()
        failed_services = [s for s in services if s["failed"]]

        analysis = {
            "total_services": len(services),
            "failed_services": len(failed_services),
            "failure_rate": len(failed_services) / len(services) if services else 0,
            "failed_by_category": defaultdict(list),
            "exit_codes": defaultdict(int),
            "recommendations": [],
        }

        # Categorize failed services
        for service in failed_services:
            name = service["name"]
            exit_code = service["exit_code"]

            if exit_code:
                analysis["exit_codes"][exit_code] += 1

            # Categorize by service type
            if "apple" in name.lower():
                if "cloud" in name.lower() or "icloud" in name.lower():
                    analysis["failed_by_category"]["cloud"].append(service)
                elif "network" in name.lower() or "wifi" in name.lower():
                    analysis["failed_by_category"]["network"].append(service)
                elif "security" in name.lower() or "auth" in name.lower():
                    analysis["failed_by_category"]["security"].append(service)
                elif "media" in name.lower() or "photo" in name.lower():
                    analysis["failed_by_category"]["media"].append(service)
                else:
                    analysis["failed_by_category"]["system"].append(service)
            else:
                analysis["failed_by_category"]["third_party"].append(service)

        # Generate recommendations
        if analysis["failed_by_category"]["cloud"]:
            analysis["recommendations"].append(
                "High number of cloud-related service failures. Consider checking network connectivity and iCloud settings."
            )

        if analysis["exit_codes"].get(-9, 0) > 5:
            analysis["recommendations"].append(
                "Many services terminated with SIGKILL (-9). This indicates forced termination, possibly due to resource constraints."
            )

        if analysis["failure_rate"] > 0.1:
            analysis["recommendations"].append(
                f"High service failure rate ({analysis['failure_rate']:.1%}). System may be under stress."
            )

        return analysis

    def identify_problematic_processes(self) -> dict:
        """Identify processes causing high system load"""
        high_cpu_processes = self.get_high_cpu_processes(threshold=5.0)

        analysis = {
            "high_cpu_processes": high_cpu_processes,
            "problematic_patterns": [],
            "remediation_suggestions": [],
        }

        # Analyze patterns
        process_names = [p["command"] for p in high_cpu_processes]

        # Check for common problematic processes
        if any("bird" in name for name in process_names):
            analysis["problematic_patterns"].append(
                {
                    "pattern": "iCloud Drive (bird) high CPU",
                    "description": "iCloud Drive sync process consuming excessive CPU",
                    "processes": [
                        p for p in high_cpu_processes if "bird" in p["command"]
                    ],
                }
            )
            analysis["remediation_suggestions"].append(
                "Consider pausing iCloud Drive sync or checking for sync conflicts"
            )

        if any("python" in name for name in process_names):
            python_processes = [
                p for p in high_cpu_processes if "python" in p["command"]
            ]
            analysis["problematic_patterns"].append(
                {
                    "pattern": "High CPU Python processes",
                    "description": "Python processes using excessive CPU",
                    "processes": python_processes,
                }
            )
            analysis["remediation_suggestions"].append(
                "Review Python processes for runaway scripts or infinite loops"
            )

        if any("claude" in name.lower() for name in process_names):
            analysis["problematic_patterns"].append(
                {
                    "pattern": "Claude process high CPU",
                    "description": "Claude AI tool consuming high CPU",
                    "processes": [
                        p
                        for p in high_cpu_processes
                        if "claude" in p["command"].lower()
                    ],
                }
            )
            analysis["remediation_suggestions"].append(
                "Claude process may be performing intensive operations. Monitor for completion."
            )

        return analysis

    def generate_remediation_plan(self) -> dict:
        """Generate a comprehensive remediation plan"""
        services_analysis = self.analyze_failing_services()
        processes_analysis = self.identify_problematic_processes()
        system_metrics = self.get_system_metrics()

        plan = {
            "priority_actions": [],
            "immediate_fixes": [],
            "monitoring_recommendations": [],
            "long_term_strategies": [],
        }

        # Priority actions based on system load
        current_load = (
            system_metrics["load_avg"][0] if system_metrics["load_avg"] else 0
        )
        if current_load > 8:
            plan["priority_actions"].append(
                {
                    "action": "Reduce system load immediately",
                    "steps": [
                        "Identify and terminate runaway processes",
                        "Clear system caches",
                        "Restart problematic services",
                        "Apply resource limits to high-CPU processes",
                    ],
                }
            )

        # Immediate fixes for high-CPU processes
        for pattern in processes_analysis["problematic_patterns"]:
            if pattern["pattern"] == "iCloud Drive (bird) high CPU":
                plan["immediate_fixes"].append(
                    {
                        "issue": "iCloud Drive high CPU usage",
                        "fix": "killall bird && sleep 2 && open /System/Library/CoreServices/CloudDocs.app",
                        "description": "Restart iCloud Drive service",
                    }
                )

            if "Python" in pattern["pattern"]:
                for proc in pattern["processes"]:
                    if proc["cpu_percent"] > 50:
                        plan["immediate_fixes"].append(
                            {
                                "issue": f'High-CPU Python process (PID {proc["pid"]})',
                                "fix": f'kill -TERM {proc["pid"]}',
                                "description": f'Terminate Python process using {proc["cpu_percent"]}% CPU',
                            }
                        )

        # Service restart recommendations
        if services_analysis["failed_services"] > 10:
            plan["immediate_fixes"].append(
                {
                    "issue": "Multiple failed services",
                    "fix": "Restart critical failed services in phases",
                    "description": "Avoid restarting all services simultaneously",
                }
            )

        # Monitoring recommendations
        plan["monitoring_recommendations"] = [
            "Set up continuous service health monitoring",
            "Implement CPU/memory usage alerts",
            "Monitor system load trends",
            "Track service failure patterns",
            "Set up automated cache clearing",
        ]

        # Long-term strategies
        plan["long_term_strategies"] = [
            "Implement service resource limits",
            "Optimize service startup order",
            "Configure service dependencies",
            "Set up automated remediation scripts",
            "Regular system maintenance schedule",
        ]

        return plan

    def create_dashboard_data(self) -> dict:
        """Create data for a simple dashboard"""
        services = self.get_launchctl_services()
        processes = self.get_high_cpu_processes(threshold=1.0)
        metrics = self.get_system_metrics()

        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "load_avg": metrics["load_avg"],
                "memory_free_mb": metrics["memory"].get("pages_free", 0)
                * 16384
                // 1024
                // 1024,
                "total_processes": metrics["processes"].get("total", 0),
            },
            "services": {
                "total": len(services),
                "running": len([s for s in services if s["status"] == "running"]),
                "failed": len([s for s in services if s["failed"]]),
                "failure_rate": len([s for s in services if s["failed"]])
                / len(services)
                if services
                else 0,
            },
            "processes": {
                "high_cpu_count": len(processes),
                "top_cpu_processes": processes[:5],
            },
        }

    def run_comprehensive_analysis(self) -> None:
        """Run comprehensive analysis and generate report"""
        print("=== COMPREHENSIVE SERVICE ANALYSIS ===")
        print(f"Analysis started: {datetime.now()}")
        print()

        # System overview
        print("=== SYSTEM OVERVIEW ===")
        metrics = self.get_system_metrics()
        print(f"Load Average: {' / '.join(map(str, metrics['load_avg']))}")
        print(
            f"Memory Free: {metrics['memory'].get('pages_free', 0) * 16384 // 1024 // 1024} MB"
        )
        print(f"Total Processes: {metrics['processes'].get('total', 'Unknown')}")
        print()

        # Service analysis
        print("=== SERVICE ANALYSIS ===")
        services_analysis = self.analyze_failing_services()
        print(f"Total Services: {services_analysis['total_services']}")
        print(f"Failed Services: {services_analysis['failed_services']}")
        print(f"Failure Rate: {services_analysis['failure_rate']:.1%}")
        print()

        print("Failed Services by Category:")
        for category, services in services_analysis["failed_by_category"].items():
            print(f"  {category.title()}: {len(services)} services")
        print()

        # Process analysis
        print("=== PROCESS ANALYSIS ===")
        processes_analysis = self.identify_problematic_processes()
        print(f"High CPU Processes: {len(processes_analysis['high_cpu_processes'])}")

        for pattern in processes_analysis["problematic_patterns"]:
            print(f"\nPattern: {pattern['pattern']}")
            print(f"Description: {pattern['description']}")
            print(f"Affected Processes: {len(pattern['processes'])}")
        print()

        # Remediation plan
        print("=== REMEDIATION PLAN ===")
        plan = self.generate_remediation_plan()

        if plan["priority_actions"]:
            print("PRIORITY ACTIONS:")
            for action in plan["priority_actions"]:
                print(f"  • {action['action']}")
                for step in action["steps"]:
                    print(f"    - {step}")

        if plan["immediate_fixes"]:
            print("\nIMMEDIATE FIXES:")
            for fix in plan["immediate_fixes"]:
                print(f"  • {fix['issue']}")
                print(f"    Command: {fix['fix']}")
                print(f"    Description: {fix['description']}")

        print("\nMONITORING RECOMMENDATIONS:")
        for rec in plan["monitoring_recommendations"]:
            print(f"  • {rec}")

        print("\nLONG-TERM STRATEGIES:")
        for strategy in plan["long_term_strategies"]:
            print(f"  • {strategy}")

        # Save detailed data
        dashboard_data = self.create_dashboard_data()
        with open("/tmp/service_analysis_data.json", "w") as f:
            json.dump(dashboard_data, f, indent=2)

        print("\nDetailed analysis data saved to: /tmp/service_analysis_data.json")


def main():
    """Main entry point"""
    analyzer = ServiceAnalyzer()

    if len(sys.argv) > 1 and sys.argv[1] == "dashboard":
        # Generate dashboard data only
        dashboard_data = analyzer.create_dashboard_data()
        print(json.dumps(dashboard_data, indent=2))
    else:
        # Run comprehensive analysis
        analyzer.run_comprehensive_analysis()


if __name__ == "__main__":
    main()
