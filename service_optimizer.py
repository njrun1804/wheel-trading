#!/usr/bin/env python3
"""
M4 Pro Service Optimization Script
Reduces 507 background services to optimal levels for trading performance
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("service_optimization.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ServiceOptimizer:
    def __init__(self):
        self.dry_run = True
        self.backup_created = False
        self.disabled_services = []

        # Service categories for optimization
        self.non_essential_patterns = {
            "development": [
                "com.apple.dt.",  # Developer tools
                "com.apple.xcode.",  # Xcode services
                "com.apple.instruments.",  # Instruments
            ],
            "media_sharing": [
                "com.apple.mediaremote",
                "com.apple.mediastream",
                "com.apple.airplay",
                "com.apple.AirPortAssistant",
            ],
            "cloud_sync_heavy": [
                "com.apple.cloudphotosd",
                "com.apple.cloudd",
                "com.apple.syncservices",
                "com.apple.icloud.fmip",
            ],
            "accessibility_unused": [
                "com.apple.assistivetouchd",
                "com.apple.universalaccessd",
                "com.apple.voiceover",
            ],
            "siri_speech": [
                "com.apple.siri.",
                "com.apple.speech.",
                "com.apple.dictation",
                "com.apple.assistantd",
            ],
            "spotlight_heavy": [
                "com.apple.metadata.mds",
                "com.apple.metadata.mdwrite",
                "com.apple.metadata.mdworker",
            ],
            "gaming": [
                "com.apple.GameController",
                "com.apple.GameCenter",
                "com.apple.gamepolicyd",
            ],
            "unused_network": [
                "com.apple.airport",
                "com.apple.bluetooth",
                "com.apple.netbiosd",
            ],
        }

        # Critical services that must never be disabled
        self.critical_services = {
            "com.apple.loginwindow",
            "com.apple.WindowServer",
            "com.apple.Dock",
            "com.apple.Finder",
            "com.apple.SystemUIServer",
            "com.apple.kernel_task",
            "com.apple.launchd",
            "com.apple.logind",
            "com.apple.securityd",
            "com.apple.trustd",
            "com.apple.networkd",
            "com.apple.wirelessproxd",
            "com.apple.wifiproxyd",
            "com.apple.audio",
            "com.apple.coreaudiod",
        }

        # Trading-specific services to keep
        self.trading_essential = {
            "com.apple.nsurlsessiond",  # Network sessions
            "com.apple.cfnetwork",  # Core networking
            "com.apple.networkserviceproxy",  # Network proxy
            "com.apple.systemstats",  # System monitoring
            "com.apple.activitymonitor",  # Activity monitoring
        }

    def get_all_services(self) -> list[dict]:
        """Get all launchd services with their status"""
        try:
            result = subprocess.run(
                ["launchctl", "list"], capture_output=True, text=True, check=True
            )
            services = []

            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                parts = line.split("\t")
                if len(parts) >= 3:
                    pid = parts[0] if parts[0] != "-" else None
                    status = parts[1]
                    label = parts[2]

                    services.append(
                        {
                            "pid": pid,
                            "status": status,
                            "label": label,
                            "active": pid is not None,
                        }
                    )

            return services
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get services: {e}")
            return []

    def analyze_services(self) -> dict:
        """Analyze current service status and categorize them"""
        services = self.get_all_services()

        analysis = {
            "total_services": len(services),
            "active_services": len([s for s in services if s["active"]]),
            "apple_services": len([s for s in services if "com.apple." in s["label"]]),
            "third_party_services": len(
                [s for s in services if "com.apple." not in s["label"]]
            ),
            "categories": {},
            "optimization_candidates": [],
            "critical_services": [],
            "resource_heavy": [],
        }

        for service in services:
            label = service["label"]

            # Check if critical
            if any(critical in label for critical in self.critical_services):
                analysis["critical_services"].append(service)
                continue

            # Check if trading essential
            if any(essential in label for essential in self.trading_essential):
                analysis["critical_services"].append(service)
                continue

            # Categorize for optimization
            for category, patterns in self.non_essential_patterns.items():
                if any(pattern in label for pattern in patterns):
                    if category not in analysis["categories"]:
                        analysis["categories"][category] = []
                    analysis["categories"][category].append(service)

                    if service["active"]:
                        analysis["optimization_candidates"].append(service)
                    break

        return analysis

    def create_backup(self) -> bool:
        """Create backup of current service state"""
        try:
            services = self.get_all_services()
            backup_data = {
                "timestamp": datetime.now().isoformat(),
                "services": services,
                "system_info": {
                    "hostname": subprocess.run(
                        ["hostname"], capture_output=True, text=True
                    ).stdout.strip(),
                    "uptime": subprocess.run(
                        ["uptime"], capture_output=True, text=True
                    ).stdout.strip(),
                },
            }

            backup_file = (
                f"service_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(backup_file, "w") as f:
                json.dump(backup_data, f, indent=2)

            logger.info(f"Backup created: {backup_file}")
            self.backup_created = True
            return True

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False

    def disable_service(self, service_label: str) -> bool:
        """Disable a specific service"""
        if service_label in self.critical_services:
            logger.warning(f"Refusing to disable critical service: {service_label}")
            return False

        try:
            if not self.dry_run:
                subprocess.run(
                    ["launchctl", "unload", "-w", service_label],
                    check=True,
                    capture_output=True,
                )
                logger.info(f"Disabled service: {service_label}")
            else:
                logger.info(f"DRY RUN: Would disable service: {service_label}")

            self.disabled_services.append(service_label)
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to disable {service_label}: {e}")
            return False

    def optimize_services(self, analysis: dict) -> dict:
        """Perform service optimization based on analysis"""
        optimization_results = {
            "disabled_count": 0,
            "disabled_services": [],
            "skipped_services": [],
            "errors": [],
        }

        # Prioritize by category impact
        category_priority = [
            "siri_speech",  # High impact, rarely used for trading
            "accessibility_unused",  # High impact if not needed
            "gaming",  # Not needed for trading
            "media_sharing",  # Not needed for trading
            "cloud_sync_heavy",  # Can be throttled
            "spotlight_heavy",  # Can be reduced
            "development",  # If not developing
            "unused_network",  # If not using specific network features
        ]

        for category in category_priority:
            if category in analysis["categories"]:
                logger.info(f"Optimizing category: {category}")

                for service in analysis["categories"][category]:
                    if service["active"]:
                        if self.disable_service(service["label"]):
                            optimization_results["disabled_count"] += 1
                            optimization_results["disabled_services"].append(
                                service["label"]
                            )
                        else:
                            optimization_results["skipped_services"].append(
                                service["label"]
                            )

        return optimization_results

    def monitor_system_impact(self) -> dict:
        """Monitor system performance after optimization"""
        try:
            # Get current system stats
            uptime_result = subprocess.run(["uptime"], capture_output=True, text=True)

            # Get memory info
            vm_stat_result = subprocess.run(["vm_stat"], capture_output=True, text=True)

            # Get process count
            service_count = len(self.get_all_services())
            active_count = len([s for s in self.get_all_services() if s["active"]])

            return {
                "uptime": uptime_result.stdout.strip(),
                "vm_stat": vm_stat_result.stdout,
                "service_count": service_count,
                "active_service_count": active_count,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to monitor system: {e}")
            return {}

    def create_monitoring_script(self):
        """Create ongoing monitoring script"""
        monitoring_script = """#!/bin/bash
# M4 Pro Service Monitoring Script
# Runs every 5 minutes to monitor system performance

LOG_FILE="service_monitoring.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$DATE] Starting service monitoring..." >> $LOG_FILE

# System load
LOAD=$(uptime | awk '{print $10,$11,$12}')
echo "[$DATE] Load averages: $LOAD" >> $LOG_FILE

# Active services
ACTIVE_SERVICES=$(launchctl list | grep -v "^-" | wc -l)
echo "[$DATE] Active services: $ACTIVE_SERVICES" >> $LOG_FILE

# Memory pressure
MEMORY_PRESSURE=$(memory_pressure 2>/dev/null || echo "N/A")
echo "[$DATE] Memory pressure: $MEMORY_PRESSURE" >> $LOG_FILE

# Check if load is too high
LOAD_1MIN=$(uptime | awk '{print $10}' | cut -d',' -f1)
if (( $(echo "$LOAD_1MIN > 8.0" | bc -l) )); then
    echo "[$DATE] WARNING: High system load detected: $LOAD_1MIN" >> $LOG_FILE
fi

echo "[$DATE] Monitoring complete" >> $LOG_FILE
"""

        with open("service_monitor.sh", "w") as f:
            f.write(monitoring_script)

        os.chmod("service_monitor.sh", 0o755)
        logger.info("Created monitoring script: service_monitor.sh")

    def restore_services(self, backup_file: str) -> bool:
        """Restore services from backup"""
        try:
            with open(backup_file) as f:
                json.load(f)

            restored_count = 0
            for service_label in self.disabled_services:
                try:
                    subprocess.run(
                        ["launchctl", "load", "-w", service_label],
                        check=True,
                        capture_output=True,
                    )
                    logger.info(f"Restored service: {service_label}")
                    restored_count += 1
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to restore {service_label}: {e}")

            logger.info(f"Restored {restored_count} services")
            return True

        except Exception as e:
            logger.error(f"Failed to restore services: {e}")
            return False

    def run_optimization(self, dry_run: bool = True) -> dict:
        """Run complete service optimization"""
        self.dry_run = dry_run

        logger.info("Starting M4 Pro Service Optimization")
        logger.info(f"Dry run mode: {dry_run}")

        # Create backup
        if not self.create_backup():
            logger.error("Failed to create backup, aborting")
            return {"success": False, "error": "Backup failed"}

        # Analyze current state
        logger.info("Analyzing current service state...")
        analysis = self.analyze_services()

        logger.info("Current state:")
        logger.info(f"  Total services: {analysis['total_services']}")
        logger.info(f"  Active services: {analysis['active_services']}")
        logger.info(f"  Apple services: {analysis['apple_services']}")
        logger.info(f"  Third-party services: {analysis['third_party_services']}")
        logger.info(
            f"  Optimization candidates: {len(analysis['optimization_candidates'])}"
        )

        # Perform optimization
        logger.info("Starting optimization...")
        optimization_results = self.optimize_services(analysis)

        # Monitor impact
        post_optimization_stats = self.monitor_system_impact()

        # Create monitoring script
        self.create_monitoring_script()

        results = {
            "success": True,
            "pre_optimization": analysis,
            "optimization_results": optimization_results,
            "post_optimization_stats": post_optimization_stats,
            "dry_run": dry_run,
        }

        logger.info("Optimization complete!")
        logger.info(f"Services disabled: {optimization_results['disabled_count']}")

        return results


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="M4 Pro Service Optimizer")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in dry-run mode (default)",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Actually disable services"
    )
    parser.add_argument(
        "--restore",
        type=str,
        metavar="BACKUP_FILE",
        help="Restore services from backup file",
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Just monitor current state"
    )

    args = parser.parse_args()

    optimizer = ServiceOptimizer()

    if args.restore:
        logger.info(f"Restoring services from {args.restore}")
        optimizer.restore_services(args.restore)
        return

    if args.monitor:
        logger.info("Monitoring current state...")
        analysis = optimizer.analyze_services()
        stats = optimizer.monitor_system_impact()

        print("\n=== Current System State ===")
        print(f"Total services: {analysis['total_services']}")
        print(f"Active services: {analysis['active_services']}")
        print(f"System load: {stats.get('uptime', 'N/A')}")
        return

    # Run optimization
    dry_run = not args.execute
    results = optimizer.run_optimization(dry_run=dry_run)

    if results["success"]:
        print("\n=== Optimization Results ===")
        print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTED'}")
        print(f"Services analyzed: {results['pre_optimization']['total_services']}")
        print(f"Services disabled: {results['optimization_results']['disabled_count']}")
        print(
            f"Current active services: {results['post_optimization_stats'].get('active_service_count', 'N/A')}"
        )

        if dry_run:
            print(f"\nTo execute changes, run: python {sys.argv[0]} --execute")
        else:
            print("\nMonitoring script created: service_monitor.sh")
            print("Run './service_monitor.sh' to check system status")
    else:
        print(f"Optimization failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
