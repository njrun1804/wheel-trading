#!/usr/bin/env python3
"""
Thermal Monitoring for Production Assessment
Monitors CPU temperature, throttling, and thermal management under load.
"""

import json
import logging
import subprocess
import threading
import time
from dataclasses import asdict, dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ThermalSnapshot:
    """Thermal state snapshot."""

    timestamp: float
    cpu_temp_celsius: float | None = None
    gpu_temp_celsius: float | None = None
    fan_speed_rpm: int | None = None
    cpu_frequency_ghz: float | None = None
    thermal_state: str | None = None
    power_consumption_watts: float | None = None


class ThermalMonitor:
    """Monitors thermal state using macOS system tools."""

    def __init__(self):
        self.snapshots: list[ThermalSnapshot] = []
        self.monitoring = False
        self.monitor_thread: threading.Thread | None = None

    def start_monitoring(self, duration_seconds: int = 900):  # 15 minutes default
        """Start thermal monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(duration_seconds,), daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Thermal monitoring started for {duration_seconds} seconds")

    def stop_monitoring(self):
        """Stop thermal monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info(
            f"Thermal monitoring stopped. Collected {len(self.snapshots)} snapshots"
        )

    def _monitor_loop(self, duration_seconds: int):
        """Main monitoring loop."""
        start_time = time.time()

        while self.monitoring and (time.time() - start_time) < duration_seconds:
            try:
                snapshot = self._capture_thermal_state()
                self.snapshots.append(snapshot)

                # Log thermal warnings
                if snapshot.cpu_temp_celsius and snapshot.cpu_temp_celsius > 80:
                    logger.warning(
                        f"High CPU temperature: {snapshot.cpu_temp_celsius:.1f}°C"
                    )

                time.sleep(2)  # Sample every 2 seconds

            except Exception as e:
                logger.error(f"Error in thermal monitoring: {e}")
                time.sleep(5)  # Wait longer on error

    def _capture_thermal_state(self) -> ThermalSnapshot:
        """Capture current thermal state."""
        snapshot = ThermalSnapshot(timestamp=time.time())

        try:
            # Try to get thermal data using system_profiler
            result = subprocess.run(
                ["system_profiler", "SPPowerDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                json.loads(result.stdout)
                # Parse thermal data if available
                # This is simplified - actual parsing depends on system_profiler output

        except Exception:
            pass

        try:
            # Try powermetrics for more detailed thermal data
            result = subprocess.run(
                ["sudo", "-n", "powermetrics", "--sample-count", "1", "-n", "0"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                output = result.stdout
                # Parse powermetrics output
                for line in output.split("\n"):
                    if "CPU die temperature" in line:
                        try:
                            temp_str = line.split(":")[1].strip().replace("C", "")
                            snapshot.cpu_temp_celsius = float(temp_str)
                        except:
                            pass
                    elif "Package Power" in line:
                        try:
                            power_str = line.split(":")[1].strip().replace("mW", "")
                            snapshot.power_consumption_watts = float(power_str) / 1000
                        except:
                            pass

        except Exception:
            # powermetrics requires sudo, may not be available
            pass

        try:
            # Get CPU frequency information
            result = subprocess.run(
                ["sysctl", "-n", "hw.cpufrequency_max"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                freq_hz = int(result.stdout.strip())
                snapshot.cpu_frequency_ghz = freq_hz / 1_000_000_000

        except Exception:
            pass

        return snapshot

    def get_thermal_analysis(self) -> dict:
        """Analyze thermal data and return summary."""
        if not self.snapshots:
            return {"error": "No thermal data collected"}

        temperatures = [
            s.cpu_temp_celsius for s in self.snapshots if s.cpu_temp_celsius
        ]
        power_consumption = [
            s.power_consumption_watts
            for s in self.snapshots
            if s.power_consumption_watts
        ]

        analysis = {
            "sample_count": len(self.snapshots),
            "duration_minutes": (
                self.snapshots[-1].timestamp - self.snapshots[0].timestamp
            )
            / 60,
            "thermal_data_available": len(temperatures) > 0,
        }

        if temperatures:
            analysis.update(
                {
                    "max_cpu_temp_celsius": max(temperatures),
                    "min_cpu_temp_celsius": min(temperatures),
                    "avg_cpu_temp_celsius": sum(temperatures) / len(temperatures),
                    "thermal_throttling_risk": max(temperatures) > 90,
                    "thermal_warnings": sum(1 for t in temperatures if t > 80),
                }
            )

        if power_consumption:
            analysis.update(
                {
                    "max_power_watts": max(power_consumption),
                    "avg_power_watts": sum(power_consumption) / len(power_consumption),
                }
            )

        return analysis


def run_thermal_stress_test():
    """Run thermal stress test with CPU load."""
    logger.info("Starting thermal stress test")

    # Start thermal monitoring
    monitor = ThermalMonitor()
    monitor.start_monitoring(duration_seconds=300)  # 5 minutes

    # Generate CPU load to increase temperature
    import multiprocessing as mp

    def cpu_load_worker():
        """Generate CPU load."""
        end_time = time.time() + 240  # 4 minutes of load
        while time.time() < end_time:
            # CPU intensive calculation
            sum(i * i for i in range(100000))

    # Start CPU load on all cores
    processes = []
    for _i in range(mp.cpu_count()):
        p = mp.Process(target=cpu_load_worker)
        p.start()
        processes.append(p)

    logger.info(f"Started {len(processes)} CPU load processes")

    # Wait for load test to complete
    for p in processes:
        p.join()

    # Wait a bit more for thermal monitoring
    time.sleep(60)

    # Stop monitoring
    monitor.stop_monitoring()

    # Analyze results
    analysis = monitor.get_thermal_analysis()

    # Save results
    results = {
        "thermal_analysis": analysis,
        "raw_snapshots": [asdict(s) for s in monitor.snapshots],
    }

    with open("thermal_stress_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return analysis


if __name__ == "__main__":
    print("🌡️  Thermal Monitoring Test")
    print("=" * 40)

    try:
        analysis = run_thermal_stress_test()

        print("📊 Thermal Analysis Results:")
        print(f"  Sample Count: {analysis.get('sample_count', 0)}")
        print(f"  Duration: {analysis.get('duration_minutes', 0):.1f} minutes")
        print(
            f"  Thermal Data Available: {analysis.get('thermal_data_available', False)}"
        )

        if analysis.get("thermal_data_available"):
            print(
                f"  Max CPU Temperature: {analysis.get('max_cpu_temp_celsius', 0):.1f}°C"
            )
            print(
                f"  Avg CPU Temperature: {analysis.get('avg_cpu_temp_celsius', 0):.1f}°C"
            )
            print(
                f"  Thermal Throttling Risk: {analysis.get('thermal_throttling_risk', False)}"
            )
            print(f"  Thermal Warnings: {analysis.get('thermal_warnings', 0)}")

            if analysis.get("max_power_watts"):
                print(
                    f"  Max Power Consumption: {analysis.get('max_power_watts', 0):.1f}W"
                )
                print(
                    f"  Avg Power Consumption: {analysis.get('avg_power_watts', 0):.1f}W"
                )
        else:
            print(
                "  ⚠️  Limited thermal data available (requires sudo for detailed metrics)"
            )

    except KeyboardInterrupt:
        print("\n⏹️  Thermal test interrupted by user")
    except Exception as e:
        print(f"\n❌ Thermal test failed: {e}")
