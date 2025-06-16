"""
Memory Management Dashboard
Real-time monitoring and visualization of memory usage
"""

import curses
import json
import time
from datetime import datetime

from bolt.hardware.memory_manager import COMPONENT_BUDGETS, get_memory_manager


class MemoryDashboard:
    """Interactive memory monitoring dashboard"""

    def __init__(self):
        self.manager = get_memory_manager()
        self.running = False
        self.history = {component: [] for component in COMPONENT_BUDGETS}
        self.max_history = 60  # Keep 60 data points

    def collect_metrics(self):
        """Collect current metrics"""
        timestamp = time.time()
        report = self.manager.get_status_report()

        # Store component metrics
        for component, stats in report["components"].items():
            self.history[component].append(
                {
                    "timestamp": timestamp,
                    "allocated_mb": stats["allocated_mb"],
                    "usage_percent": stats["usage_percent"],
                }
            )

            # Trim history
            if len(self.history[component]) > self.max_history:
                self.history[component].pop(0)

        return report

    def format_bar(self, value: float, max_value: float, width: int = 40) -> str:
        """Create a progress bar"""
        if max_value == 0:
            return "[" + " " * width + "]"

        filled = int((value / max_value) * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def get_status_color(self, usage_percent: float) -> int:
        """Get color based on usage percentage"""
        if usage_percent >= 90:
            return curses.COLOR_RED
        elif usage_percent >= 75:
            return curses.COLOR_YELLOW
        else:
            return curses.COLOR_GREEN

    def run_dashboard(self, stdscr):
        """Run the interactive dashboard"""
        # Setup colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)

        # Configure screen
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)  # Non-blocking input
        stdscr.clear()

        self.running = True

        while self.running:
            try:
                # Collect metrics
                report = self.collect_metrics()

                # Clear screen
                stdscr.clear()
                height, width = stdscr.getmaxyx()

                # Header
                row = 0
                header = "═══ BOLT MEMORY MANAGEMENT DASHBOARD ═══"
                stdscr.addstr(
                    row,
                    (width - len(header)) // 2,
                    header,
                    curses.A_BOLD | curses.color_pair(4),
                )
                row += 2

                # System overview
                system = report["system"]
                stdscr.addstr(row, 0, "SYSTEM OVERVIEW", curses.A_BOLD)
                row += 1

                stdscr.addstr(row, 2, f"Total Memory: {system['total_memory_gb']}GB")
                stdscr.addstr(
                    row, 30, f"Max Allocation: {system['max_allocation_gb']}GB"
                )
                row += 1

                total_mb = system["total_allocated_mb"]
                max_mb = system["max_allocation_gb"] * 1024
                usage_pct = (total_mb / max_mb) * 100

                stdscr.addstr(
                    row, 2, f"Total Allocated: {total_mb:.1f}MB ({usage_pct:.1f}%)"
                )
                row += 1

                # System memory bar
                sys_usage = system["system_usage_percent"]
                color = 1 if sys_usage < 75 else (2 if sys_usage < 85 else 3)
                stdscr.addstr(row, 2, "System RAM: ", curses.color_pair(5))
                stdscr.addstr(
                    row, 14, self.format_bar(sys_usage, 100), curses.color_pair(color)
                )
                stdscr.addstr(row, 56, f"{sys_usage:5.1f}%", curses.color_pair(color))
                row += 2

                # Component details
                stdscr.addstr(row, 0, "COMPONENT MEMORY USAGE", curses.A_BOLD)
                row += 1
                stdscr.addstr(row, 2, "Component", curses.A_UNDERLINE)
                stdscr.addstr(row, 15, "Allocated", curses.A_UNDERLINE)
                stdscr.addstr(row, 27, "Limit", curses.A_UNDERLINE)
                stdscr.addstr(row, 35, "Usage", curses.A_UNDERLINE)
                stdscr.addstr(row, 44, "Progress", curses.A_UNDERLINE)
                row += 1

                # Component rows
                for component, stats in report["components"].items():
                    allocated = stats["allocated_mb"]
                    limit = stats["max_mb"]
                    usage = stats["usage_percent"]

                    # Determine color
                    if usage >= 90:
                        color_pair = 3  # Red
                    elif usage >= 75:
                        color_pair = 2  # Yellow
                    else:
                        color_pair = 1  # Green

                    # Component name
                    stdscr.addstr(row, 2, f"{component:12}", curses.color_pair(5))

                    # Allocated
                    stdscr.addstr(
                        row, 15, f"{allocated:7.1f}MB", curses.color_pair(color_pair)
                    )

                    # Limit
                    stdscr.addstr(row, 27, f"{limit:6.0f}MB", curses.color_pair(5))

                    # Usage %
                    stdscr.addstr(
                        row, 35, f"{usage:6.1f}%", curses.color_pair(color_pair)
                    )

                    # Progress bar
                    stdscr.addstr(
                        row,
                        44,
                        self.format_bar(allocated, limit, 30),
                        curses.color_pair(color_pair),
                    )

                    row += 1

                row += 1

                # Allocation statistics
                stdscr.addstr(row, 0, "ALLOCATION STATISTICS", curses.A_BOLD)
                row += 1

                for component, stats in report["components"].items():
                    if stats["allocation_count"] > 0:
                        stdscr.addstr(
                            row,
                            2,
                            f"{component}: {stats['allocation_count']} allocations, "
                            f"{stats['eviction_count']} evictions, "
                            f"peak {stats['peak_mb']:.1f}MB",
                            curses.color_pair(5),
                        )
                        row += 1

                # Instructions
                row = height - 3
                stdscr.addstr(
                    row,
                    2,
                    "Press 'q' to quit, 'r' to reset stats, 'g' to force GC",
                    curses.color_pair(4),
                )

                # Timestamp
                row = height - 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                stdscr.addstr(row, 2, f"Last Update: {timestamp}", curses.color_pair(5))

                # Refresh
                stdscr.refresh()

                # Handle input
                key = stdscr.getch()
                if key == ord("q"):
                    self.running = False
                elif key == ord("r"):
                    # Reset stats
                    for pool in self.manager.pools.values():
                        pool.stats.allocation_count = 0
                        pool.stats.eviction_count = 0
                        pool.stats.pressure_events = 0
                elif key == ord("g"):
                    # Force garbage collection
                    import gc

                    gc.collect()

                # Update every second
                time.sleep(1)

            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                # Log error and continue
                stdscr.addstr(
                    height - 2, 2, f"Error: {str(e)[:width-4]}", curses.color_pair(3)
                )

    def export_metrics(self, filename: str = "memory_metrics.json"):
        """Export current metrics to file"""
        report = self.manager.get_status_report()
        report["history"] = self.history

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Metrics exported to {filename}")


class MemoryAlertMonitor:
    """Background monitor that sends alerts"""

    def __init__(self, alert_threshold: float = 85.0):
        self.manager = get_memory_manager()
        self.alert_threshold = alert_threshold
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minutes between alerts

    def check_alerts(self) -> list[str]:
        """Check for alert conditions"""
        alerts = []
        current_time = time.time()
        report = self.manager.get_status_report()

        # Check system memory
        sys_usage = report["system"]["system_usage_percent"]
        if sys_usage > self.alert_threshold:
            alert_key = "system_memory"
            if self._should_alert(alert_key, current_time):
                alerts.append(f"CRITICAL: System memory usage at {sys_usage:.1f}%")
                self.last_alert_time[alert_key] = current_time

        # Check component usage
        for component, stats in report["components"].items():
            if stats["usage_percent"] > 90:
                alert_key = f"component_{component}"
                if self._should_alert(alert_key, current_time):
                    alerts.append(
                        f"WARNING: {component} memory usage at {stats['usage_percent']:.1f}% "
                        f"({stats['allocated_mb']:.1f}MB / {stats['max_mb']:.1f}MB)"
                    )
                    self.last_alert_time[alert_key] = current_time

        return alerts

    def _should_alert(self, alert_key: str, current_time: float) -> bool:
        """Check if we should send this alert"""
        if alert_key not in self.last_alert_time:
            return True
        return current_time - self.last_alert_time[alert_key] > self.alert_cooldown

    def run_monitor(self):
        """Run continuous monitoring"""
        print("Starting memory alert monitor...")

        while True:
            try:
                alerts = self.check_alerts()
                for alert in alerts:
                    print(f"\n🚨 {alert}")
                    # In production, send to Slack, email, etc.

                time.sleep(30)  # Check every 30 seconds

            except KeyboardInterrupt:
                print("\nMonitor stopped")
                break
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(60)


def print_memory_summary():
    """Print a simple memory summary (non-interactive)"""
    manager = get_memory_manager()
    report = manager.get_status_report()

    print("\n" + "=" * 60)
    print("BOLT MEMORY MANAGER - STATUS SUMMARY")
    print("=" * 60)

    system = report["system"]
    print(f"\nSystem Memory: {system['system_usage_percent']:.1f}% used")
    print(
        f"Total Allocated: {system['total_allocated_mb']:.1f}MB / {system['max_allocation_gb']*1024:.0f}MB"
    )
    print(
        f"Allocation Usage: {(system['total_allocated_mb'] / (system['max_allocation_gb']*1024)) * 100:.1f}%"
    )

    print("\nComponent Usage:")
    print("-" * 50)
    print(f"{'Component':<12} {'Allocated':>10} {'Limit':>10} {'Usage':>8} {'Status'}")
    print("-" * 50)

    for component, stats in report["components"].items():
        allocated = stats["allocated_mb"]
        limit = stats["max_mb"]
        usage = stats["usage_percent"]

        if usage >= 90:
            status = "CRITICAL"
        elif usage >= 75:
            status = "WARNING"
        elif usage > 0:
            status = "OK"
        else:
            status = "IDLE"

        print(
            f"{component:<12} {allocated:>9.1f}MB {limit:>9.0f}MB {usage:>7.1f}% {status}"
        )

    print("\nAllocation Activity:")
    print("-" * 50)

    total_allocs = sum(s["allocation_count"] for s in report["components"].values())
    total_evictions = sum(s["eviction_count"] for s in report["components"].values())

    print(f"Total Allocations: {total_allocs}")
    print(f"Total Evictions: {total_evictions}")

    if total_evictions > 0:
        print(f"Eviction Rate: {(total_evictions / total_allocs) * 100:.1f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Memory Management Dashboard")
    parser.add_argument(
        "--mode",
        choices=["dashboard", "monitor", "summary"],
        default="summary",
        help="Display mode",
    )
    parser.add_argument("--export", help="Export metrics to file")

    args = parser.parse_args()

    if args.mode == "dashboard":
        # Run interactive dashboard
        dashboard = MemoryDashboard()
        curses.wrapper(dashboard.run_dashboard)

        if args.export:
            dashboard.export_metrics(args.export)

    elif args.mode == "monitor":
        # Run alert monitor
        monitor = MemoryAlertMonitor()
        monitor.run_monitor()

    else:
        # Print summary
        print_memory_summary()

        if args.export:
            manager = get_memory_manager()
            report = manager.get_status_report()
            with open(args.export, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nMetrics exported to {args.export}")
