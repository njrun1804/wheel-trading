#!/usr/bin/env python3
"""
Core 4 Demo - Demonstrate process monitoring and cleanup capabilities
Shows how the system identifies and manages resource-intensive processes
"""

import logging
import multiprocessing
import time

# Import Core 4 components
from core4_process_monitor import Core4ProcessMonitor
from core4_resource_manager import Core4ResourceManager
from core4_system_monitor import Core4SystemMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Core4Demo")


class ResourceIntensiveProcess:
    """Create a resource-intensive process for demonstration"""

    def __init__(self, process_type: str = "cpu"):
        self.process_type = process_type
        self.running = False
        self.process = None

    def cpu_intensive_work(self):
        """CPU-intensive work"""
        while True:
            # Consume CPU cycles
            for i in range(1000000):
                _ = i**2

    def memory_intensive_work(self):
        """Memory-intensive work"""
        data = []
        while True:
            # Consume memory
            data.append(bytearray(1024 * 1024))  # 1MB chunks
            time.sleep(0.1)

    def start(self):
        """Start the resource-intensive process"""
        if self.process_type == "cpu":
            target = self.cpu_intensive_work
        else:
            target = self.memory_intensive_work

        self.process = multiprocessing.Process(target=target)
        self.process.start()
        self.running = True
        return self.process.pid

    def stop(self):
        """Stop the process"""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
        self.running = False


class Core4Demo:
    """Demonstrate Core 4 monitoring and cleanup capabilities"""

    def __init__(self):
        self.demo_processes = []
        self.monitor = Core4ProcessMonitor("core4_config.json")
        self.resource_manager = Core4ResourceManager("core4_config.json")
        self.system_monitor = Core4SystemMonitor("core4_config.json")

    def show_initial_status(self):
        """Show system status before demo"""
        print("=" * 60)
        print("CORE 4 PROCESS MONITORING DEMO")
        print("=" * 60)
        print()

        # System resources
        resources = self.system_monitor.get_system_metrics()
        print("Initial System Status:")
        print(f"  CPU Usage: {resources.cpu_percent:.1f}%")
        print(
            f"  Memory Usage: {resources.memory_percent:.1f}% ({resources.memory_used_gb:.1f}GB)"
        )
        print(f"  Total Processes: {resources.process_count}")
        print(f"  Load Average: {resources.load_avg[0]:.2f}")
        print()

        # Top processes
        top_processes = self.monitor.get_top_processes("cpu", 5)
        print("Top CPU Processes:")
        for i, proc in enumerate(top_processes, 1):
            print(
                f"  {i}. PID {proc.pid} - {proc.name} - CPU: {proc.cpu_percent:.1f}% - Memory: {proc.memory_mb:.1f}MB"
            )
        print()

    def create_demo_processes(self, num_cpu: int = 2, num_memory: int = 1):
        """Create resource-intensive processes for demonstration"""
        print(
            f"Creating {num_cpu} CPU-intensive and {num_memory} memory-intensive processes..."
        )

        # Create CPU-intensive processes
        for _i in range(num_cpu):
            proc = ResourceIntensiveProcess("cpu")
            pid = proc.start()
            self.demo_processes.append(proc)
            print(f"  Started CPU-intensive process: PID {pid}")

        # Create memory-intensive processes
        for _i in range(num_memory):
            proc = ResourceIntensiveProcess("memory")
            pid = proc.start()
            self.demo_processes.append(proc)
            print(f"  Started memory-intensive process: PID {pid}")

        print(f"Created {len(self.demo_processes)} demo processes")
        print()

    def show_system_under_load(self):
        """Show system status under load"""
        print("System Status Under Load:")
        print("-" * 40)

        # Wait for processes to ramp up
        time.sleep(5)

        # Show updated system metrics
        resources = self.system_monitor.get_system_metrics()
        print(f"  CPU Usage: {resources.cpu_percent:.1f}%")
        print(
            f"  Memory Usage: {resources.memory_percent:.1f}% ({resources.memory_used_gb:.1f}GB)"
        )
        print(f"  Load Average: {resources.load_avg[0]:.2f}")

        # Show top processes
        top_processes = self.monitor.get_top_processes("cpu", 10)
        print("\nTop CPU Processes:")
        for i, proc in enumerate(top_processes, 1):
            print(
                f"  {i}. PID {proc.pid} - {proc.name} - CPU: {proc.cpu_percent:.1f}% - Memory: {proc.memory_mb:.1f}MB"
            )

        # Check for stuck processes
        stuck_processes = self.monitor.find_stuck_processes()
        if stuck_processes:
            print(f"\nStuck/Runaway Processes Detected: {len(stuck_processes)}")
            for proc in stuck_processes:
                print(f"  PID {proc.pid} - {proc.name} - CPU: {proc.cpu_percent:.1f}%")

        print()

    def demonstrate_monitoring(self):
        """Demonstrate monitoring capabilities"""
        print("Monitoring Capabilities:")
        print("-" * 40)

        # Generate monitoring report
        report = self.monitor.get_monitoring_report()

        print("System Resources:")
        sys_res = report["system_resources"]
        print(f"  CPU: {sys_res['cpu']['percent']:.1f}%")
        print(f"  Memory: {sys_res['memory']['percent']:.1f}%")
        print(f"  Processes: {sys_res['processes']['count']}")

        print("\nTop Processes by CPU:")
        for i, proc in enumerate(report["top_cpu_processes"][:5], 1):
            print(
                f"  {i}. PID {proc['pid']} - {proc['name']} - CPU: {proc['cpu_percent']:.1f}%"
            )

        print("\nCleanup Statistics:")
        stats = report["cleanup_stats"]
        print(f"  Processes Killed: {stats['processes_killed']}")
        print(f"  Zombies Cleaned: {stats['zombies_cleaned']}")
        print(f"  Memory Freed: {stats['memory_freed_mb']:.1f}MB")

        print()

    def demonstrate_alerts(self):
        """Demonstrate alerting system"""
        print("Alert System:")
        print("-" * 40)

        # Get current metrics
        metrics = self.system_monitor.get_system_metrics()

        # Check thresholds
        alerts = self.system_monitor.check_thresholds(metrics)

        if alerts:
            print(f"Active Alerts: {len(alerts)}")
            for alert in alerts:
                print(f"  [{alert.severity.upper()}] {alert.message}")
        else:
            print("No active alerts")

        # Get process alerts
        process_metrics = self.system_monitor.get_process_metrics(20)
        process_alerts = self.system_monitor.check_process_thresholds(process_metrics)

        if process_alerts:
            print(f"\nProcess Alerts: {len(process_alerts)}")
            for alert in process_alerts[:5]:  # Show first 5
                print(f"  [{alert.severity.upper()}] {alert.message}")

        print()

    def demonstrate_cleanup(self):
        """Demonstrate automatic cleanup"""
        print("Automatic Cleanup:")
        print("-" * 40)

        # Perform cleanup
        print("Performing automatic cleanup of excessive processes...")

        # Use process monitor cleanup
        cleaned_processes = self.monitor.auto_cleanup_excessive_processes()

        if cleaned_processes:
            print(f"Cleaned up {len(cleaned_processes)} processes:")
            for proc in cleaned_processes:
                print(
                    f"  PID {proc.pid} - {proc.name} - CPU: {proc.cpu_percent:.1f}% - Memory: {proc.memory_mb:.1f}MB"
                )
        else:
            print("No processes required cleanup")

        # Use resource manager optimization
        print("\nOptimizing memory usage...")
        memory_result = self.resource_manager.optimize_memory_usage()
        print(f"Memory optimization result: {memory_result}")

        # Clean up zombies
        print("\nCleaning up zombie processes...")
        zombies_cleaned = self.monitor.cleanup_zombie_processes()
        print(f"Cleaned up {zombies_cleaned} zombie processes")

        print()

    def demonstrate_resource_limits(self):
        """Demonstrate resource limit enforcement"""
        print("Resource Limit Enforcement:")
        print("-" * 40)

        # Get current resources
        resources = self.resource_manager.get_system_resources()

        # Check limits
        alerts = self.resource_manager.check_resource_limits(resources)

        if alerts:
            print(f"Resource limit violations: {len(alerts)}")
            for alert in alerts:
                print(
                    f"  {alert.resource_type}: {alert.current_value:.1f} > {alert.threshold:.1f}"
                )

            # Enforce limits
            enforcement_actions = self.resource_manager.enforce_resource_limits(alerts)
            print(f"\nEnforcement actions: {enforcement_actions}")
        else:
            print("All resources within limits")

        print()

    def show_final_status(self):
        """Show final system status"""
        print("Final System Status:")
        print("-" * 40)

        # Clean up our demo processes first
        self.cleanup_demo_processes()

        # Wait for cleanup to take effect
        time.sleep(2)

        # Show final metrics
        resources = self.system_monitor.get_system_metrics()
        print(f"  CPU Usage: {resources.cpu_percent:.1f}%")
        print(
            f"  Memory Usage: {resources.memory_percent:.1f}% ({resources.memory_used_gb:.1f}GB)"
        )
        print(f"  Total Processes: {resources.process_count}")
        print(f"  Load Average: {resources.load_avg[0]:.2f}")

        # Show health score
        health = self.system_monitor.get_system_health_score(resources)
        print(
            f"\nSystem Health Score: {health['scores']['overall']:.1f} ({health['status']})"
        )

    def cleanup_demo_processes(self):
        """Clean up demo processes"""
        print("Cleaning up demo processes...")
        for proc in self.demo_processes:
            proc.stop()
        self.demo_processes.clear()

    def run_demo(self):
        """Run the complete demonstration"""
        try:
            # Show initial status
            self.show_initial_status()

            # Create resource-intensive processes
            self.create_demo_processes(2, 1)

            # Show system under load
            self.show_system_under_load()

            # Demonstrate monitoring
            self.demonstrate_monitoring()

            # Demonstrate alerts
            self.demonstrate_alerts()

            # Demonstrate cleanup
            self.demonstrate_cleanup()

            # Demonstrate resource limits
            self.demonstrate_resource_limits()

            # Show final status
            self.show_final_status()

            print("=" * 60)
            print("DEMO COMPLETED SUCCESSFULLY")
            print("=" * 60)

        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        except Exception as e:
            print(f"Demo failed: {e}")
            logger.exception("Demo error")
        finally:
            self.cleanup_demo_processes()


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Core 4 Process Monitoring Demo")
    parser.add_argument("--quick", action="store_true", help="Run quick demo")
    parser.add_argument(
        "--cpu-processes", type=int, default=2, help="Number of CPU-intensive processes"
    )
    parser.add_argument(
        "--memory-processes",
        type=int,
        default=1,
        help="Number of memory-intensive processes",
    )

    args = parser.parse_args()

    demo = Core4Demo()

    if args.quick:
        # Quick demo
        demo.show_initial_status()
        demo.demonstrate_monitoring()
        demo.show_final_status()
    else:
        # Full demo
        demo.run_demo()


if __name__ == "__main__":
    main()
