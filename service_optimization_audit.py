#!/usr/bin/env python3
"""
Core 5 Service Optimization Audit System
Comprehensive analysis and optimization of 500+ background services
"""

import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime

import psutil


class ServiceAuditor:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.audit_data = {
            "timestamp": self.timestamp,
            "system_info": self._get_system_info(),
            "processes": [],
            "launchd_services": [],
            "optimization_targets": [],
            "performance_metrics": {},
        }

    def _get_system_info(self):
        """Gather system information for M4 Pro optimization"""
        try:
            # M4 Pro specs
            cpu_info = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            memory_total = psutil.virtual_memory().total / (1024**3)  # GB

            return {
                "cpu": cpu_info,
                "memory_gb": round(memory_total, 1),
                "cores": os.cpu_count(),
                "platform": "M4 Pro" if "M4" in cpu_info else "Unknown",
            }
        except:
            return {"cpu": "Unknown", "memory_gb": 0, "cores": 0, "platform": "Unknown"}

    def audit_processes(self):
        """Comprehensive process audit with resource usage"""
        print("🔍 Auditing running processes...")

        processes = []
        process_groups = defaultdict(list)

        for proc in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_info", "cmdline"]
        ):
            try:
                pinfo = proc.info
                if pinfo["name"]:
                    # Convert memory_info to dict
                    if pinfo["memory_info"]:
                        pinfo["memory_info"] = {
                            "rss": pinfo["memory_info"].rss,
                            "vms": pinfo["memory_info"].vms,
                        }
                    # Group similar processes
                    base_name = (
                        pinfo["name"].split(".")[0]
                        if "." in pinfo["name"]
                        else pinfo["name"]
                    )
                    process_groups[base_name].append(pinfo)
                    processes.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Identify redundant processes
        redundant_groups = []
        for name, group in process_groups.items():
            if len(group) > 3 and name not in ["kernel_task", "launchd"]:
                redundant_groups.append(
                    {
                        "name": name,
                        "count": len(group),
                        "total_memory": sum(
                            p.get("memory_info", {}).get("rss", 0)
                            if p.get("memory_info")
                            else 0
                            for p in group
                        ),
                        "processes": group,
                    }
                )

        self.audit_data["processes"] = processes
        self.audit_data["process_groups"] = dict(process_groups)
        self.audit_data["redundant_groups"] = redundant_groups

        print(f"✅ Found {len(processes)} processes in {len(process_groups)} groups")
        print(f"🎯 Identified {len(redundant_groups)} redundant process groups")

        return processes, redundant_groups

    def audit_launchd_services(self):
        """Audit LaunchD services for optimization"""
        print("🔍 Auditing LaunchD services...")

        try:
            result = subprocess.run(
                ["launchctl", "list"], capture_output=True, text=True
            )
            services = []
            apple_services = 0
            third_party_services = 0

            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                parts = line.split("\t")
                if len(parts) >= 3:
                    pid = parts[0] if parts[0] != "-" else None
                    status = parts[1]
                    label = parts[2]

                    is_apple = "com.apple." in label
                    if is_apple:
                        apple_services += 1
                    else:
                        third_party_services += 1

                    services.append(
                        {
                            "pid": pid,
                            "status": status,
                            "label": label,
                            "is_apple": is_apple,
                            "running": pid is not None,
                        }
                    )

            self.audit_data["launchd_services"] = services
            self.audit_data["service_counts"] = {
                "total": len(services),
                "apple": apple_services,
                "third_party": third_party_services,
                "running": len([s for s in services if s["running"]]),
            }

            print(f"✅ Found {len(services)} LaunchD services")
            print(f"📊 Apple: {apple_services}, Third-party: {third_party_services}")

            return services

        except Exception as e:
            print(f"❌ Error auditing LaunchD services: {e}")
            return []

    def identify_optimization_targets(self):
        """Identify services that can be optimized or disabled"""
        print("🎯 Identifying optimization targets...")

        optimization_targets = []

        # High memory usage processes
        [
            p
            for p in self.audit_data["processes"]
            if p.get("memory_info")
            and p.get("memory_info", {}).get("rss", 0) > 100 * 1024 * 1024  # >100MB
        ]

        # Non-essential services that can be deferred
        deferrable_services = [
            "com.adobe.",
            "com.google.",
            "com.microsoft.",
            "com.spotify.",
            "com.docker.",
            "com.jetbrains.",
        ]

        for service in self.audit_data["launchd_services"]:
            label = service["label"]

            # Trading system critical services (DO NOT TOUCH)
            if any(
                critical in label.lower()
                for critical in [
                    "python",
                    "duckdb",
                    "einstein",
                    "jarvis",
                    "claude",
                    "trading",
                ]
            ):
                continue

            # Services that can be optimized
            if any(defer in label for defer in deferrable_services):
                optimization_targets.append(
                    {
                        "type": "defer_startup",
                        "service": label,
                        "reason": "Non-essential third-party service",
                        "impact": "low",
                    }
                )

        # Redundant process cleanup
        for group in self.audit_data["redundant_groups"]:
            if group["count"] > 5:
                optimization_targets.append(
                    {
                        "type": "process_cleanup",
                        "process": group["name"],
                        "count": group["count"],
                        "memory_savings": group["total_memory"] // (1024 * 1024),  # MB
                        "reason": f"Excessive {group['name']} processes",
                        "impact": "medium",
                    }
                )

        self.audit_data["optimization_targets"] = optimization_targets
        print(f"✅ Identified {len(optimization_targets)} optimization targets")

        return optimization_targets

    def generate_performance_metrics(self):
        """Calculate current performance metrics"""
        print("📊 Generating performance metrics...")

        total_memory_usage = sum(
            p.get("memory_info", {}).get("rss", 0) if p.get("memory_info") else 0
            for p in self.audit_data["processes"]
        ) / (
            1024**3
        )  # GB

        cpu_usage = sum(
            p.get("cpu_percent", 0) or 0 for p in self.audit_data["processes"]
        )

        metrics = {
            "total_processes": len(self.audit_data["processes"]),
            "total_services": self.audit_data["service_counts"]["total"],
            "memory_usage_gb": round(total_memory_usage, 2),
            "cpu_usage_percent": round(cpu_usage, 2),
            "redundant_processes": len(self.audit_data["redundant_groups"]),
            "optimization_opportunities": len(self.audit_data["optimization_targets"]),
        }

        # Calculate potential savings
        potential_memory_savings = sum(
            target.get("memory_savings", 0)
            for target in self.audit_data["optimization_targets"]
            if target["type"] == "process_cleanup"
        )

        metrics["potential_memory_savings_mb"] = potential_memory_savings
        metrics["potential_process_reduction"] = sum(
            target.get("count", 1) - 1  # Keep one instance
            for target in self.audit_data["optimization_targets"]
            if target["type"] == "process_cleanup"
        )

        self.audit_data["performance_metrics"] = metrics
        print("✅ Performance analysis complete")

        return metrics

    def generate_optimization_report(self):
        """Generate comprehensive optimization recommendations"""
        print("📝 Generating optimization report...")

        report = {
            "executive_summary": {
                "total_processes": self.audit_data["performance_metrics"][
                    "total_processes"
                ],
                "total_services": self.audit_data["performance_metrics"][
                    "total_services"
                ],
                "memory_usage": f"{self.audit_data['performance_metrics']['memory_usage_gb']}GB",
                "optimization_opportunities": len(
                    self.audit_data["optimization_targets"]
                ),
                "potential_savings": {
                    "processes": self.audit_data["performance_metrics"][
                        "potential_process_reduction"
                    ],
                    "memory_mb": self.audit_data["performance_metrics"][
                        "potential_memory_savings_mb"
                    ],
                },
            },
            "recommendations": {
                "immediate_actions": [],
                "startup_optimizations": [],
                "monitoring_recommendations": [],
            },
            "trading_system_optimizations": {
                "cpu_allocation": "Allocate P-cores 0-7 for trading processes",
                "memory_quotas": {
                    "claude": "4GB",
                    "einstein": "2GB",
                    "duckdb": "1GB",
                    "jarvis": "512MB",
                },
                "process_priorities": "Set trading processes to nice -10",
            },
        }

        # Generate specific recommendations
        for target in self.audit_data["optimization_targets"]:
            if target["type"] == "process_cleanup":
                report["recommendations"]["immediate_actions"].append(
                    f"Clean up {target['count']} {target['process']} processes "
                    f"(saves {target['memory_savings']}MB)"
                )
            elif target["type"] == "defer_startup":
                report["recommendations"]["startup_optimizations"].append(
                    f"Defer {target['service']} startup by 60 seconds"
                )

        # Monitoring recommendations
        report["recommendations"]["monitoring_recommendations"] = [
            "Monitor trading process CPU usage hourly",
            "Alert if system memory usage exceeds 20GB",
            "Track process count growth over time",
            "Monitor LaunchD service failures",
        ]

        return report

    def save_audit_data(self):
        """Save complete audit data to JSON"""
        filename = f"service_audit_report_{self.timestamp}.json"
        filepath = f"/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/{filename}"

        with open(filepath, "w") as f:
            json.dump(self.audit_data, f, indent=2, default=str)

        print(f"💾 Audit data saved to {filename}")
        return filepath

    def run_complete_audit(self):
        """Execute complete service audit"""
        print("🚀 Starting Core 5 Service Optimization Audit")
        print("=" * 50)

        # Step 1: Process audit
        processes, redundant_groups = self.audit_processes()

        # Step 2: LaunchD service audit
        self.audit_launchd_services()

        # Step 3: Identify optimization targets
        targets = self.identify_optimization_targets()

        # Step 4: Performance metrics
        metrics = self.generate_performance_metrics()

        # Step 5: Generate report
        report = self.generate_optimization_report()

        # Step 6: Save data
        filepath = self.save_audit_data()

        print("\n" + "=" * 50)
        print("📊 AUDIT SUMMARY")
        print("=" * 50)
        print(f"Total Processes: {metrics['total_processes']}")
        print(f"LaunchD Services: {metrics['total_services']}")
        print(f"Memory Usage: {metrics['memory_usage_gb']}GB")
        print(f"Optimization Targets: {len(targets)}")
        print(f"Potential Process Reduction: {metrics['potential_process_reduction']}")
        print(f"Potential Memory Savings: {metrics['potential_memory_savings_mb']}MB")

        return report, filepath


if __name__ == "__main__":
    auditor = ServiceAuditor()
    report, filepath = auditor.run_complete_audit()

    print("\n✅ Service audit complete!")
    print(f"📄 Report saved to: {filepath}")
