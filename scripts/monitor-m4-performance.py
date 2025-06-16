#!/usr/bin/env python3
"""
M4 Pro Performance Monitor for Wheel Trading
Real-time monitoring of CPU, GPU, memory, and thermal performance
"""

import os
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

import psutil

try:
    import blessed

    TERM = blessed.Terminal()
except ImportError:
    print("Installing blessed for terminal UI...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "blessed"])
    import blessed

    TERM = blessed.Terminal()


class M4ProMonitor:
    def __init__(self):
        self.cpu_history = deque(maxlen=60)
        self.memory_history = deque(maxlen=60)
        self.gpu_history = deque(maxlen=60)
        self.temp_history = deque(maxlen=60)
        self.running = True
        self.data_lock = threading.Lock()

    def get_cpu_temp(self):
        """Get CPU temperature using powermetrics"""
        try:
            result = subprocess.run(
                ["sudo", "powermetrics", "--samplers", "thermal", "-i", "1", "-n", "1"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            for line in result.stdout.split("\n"):
                if "CPU die temperature" in line:
                    temp = float(line.split(":")[1].strip().replace(" C", ""))
                    return temp
        except:
            return 0.0
        return 0.0

    def get_gpu_usage(self):
        """Get GPU usage from Metal Performance HUD"""
        try:
            result = subprocess.run(
                [
                    "sudo",
                    "powermetrics",
                    "--samplers",
                    "gpu_power",
                    "-i",
                    "1",
                    "-n",
                    "1",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            for line in result.stdout.split("\n"):
                if "GPU Busy" in line:
                    usage = float(line.split(":")[1].strip().replace("%", ""))
                    return usage
        except:
            return 0.0
        return 0.0

    def get_process_info(self):
        """Get wheel trading process information"""
        processes = []
        for proc in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_percent"]
        ):
            try:
                if "python" in proc.info["name"].lower():
                    cmdline = " ".join(proc.cmdline())
                    if (
                        "wheel" in cmdline
                        or "run.py" in cmdline
                        or "orchestrate" in cmdline
                    ):
                        processes.append(
                            {
                                "pid": proc.info["pid"],
                                "name": proc.info["name"],
                                "cpu": proc.info["cpu_percent"],
                                "memory": proc.info["memory_percent"],
                                "cmdline": cmdline[:50] + "..."
                                if len(cmdline) > 50
                                else cmdline,
                            }
                        )
            except:
                pass
        return processes

    def update_metrics(self):
        """Update all metrics"""
        while self.running:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
                cpu_avg = sum(cpu_percent) / len(cpu_percent)

                # Memory metrics
                memory = psutil.virtual_memory()

                # GPU metrics (simplified, actual would need Metal API)
                gpu_usage = self.get_gpu_usage()

                # Temperature
                temp = self.get_cpu_temp()

                with self.data_lock:
                    self.cpu_history.append(cpu_avg)
                    self.memory_history.append(memory.percent)
                    self.gpu_history.append(gpu_usage)
                    self.temp_history.append(temp)

            except Exception as e:
                print(f"Error updating metrics: {e}")

            time.sleep(1)

    def draw_bar(self, value, max_value, width, color=None):
        """Draw a horizontal bar"""
        filled = int((value / max_value) * width)
        bar = "█" * filled + "░" * (width - filled)
        if color:
            return color + bar + TERM.normal
        return bar

    def draw_sparkline(self, data, width):
        """Draw a sparkline graph"""
        if not data:
            return " " * width

        chars = "▁▂▃▄▅▆▇█"
        min_val = min(data) if data else 0
        max_val = max(data) if data else 100
        range_val = max_val - min_val if max_val != min_val else 1

        # Sample data to fit width
        if len(data) > width:
            step = len(data) / width
            sampled = [data[int(i * step)] for i in range(width)]
        else:
            sampled = list(data) + [data[-1] if data else 0] * (width - len(data))

        sparkline = ""
        for val in sampled:
            normalized = (val - min_val) / range_val
            idx = int(normalized * (len(chars) - 1))
            sparkline += chars[idx]

        return sparkline

    def display(self):
        """Display the monitoring dashboard"""
        with TERM.fullscreen(), TERM.hidden_cursor():
            while self.running:
                print(TERM.home + TERM.clear)

                # Header
                print(TERM.bold_cyan("🖥️  M4 Pro Performance Monitor - Wheel Trading"))
                print(TERM.cyan("─" * TERM.width))
                print()

                with self.data_lock:
                    # CPU Performance
                    cpu_current = self.cpu_history[-1] if self.cpu_history else 0
                    cpu_color = (
                        TERM.red
                        if cpu_current > 80
                        else TERM.yellow
                        if cpu_current > 60
                        else TERM.green
                    )

                    print(TERM.bold("CPU Usage:"), f"{cpu_current:.1f}%")
                    print(self.draw_bar(cpu_current, 100, 40, cpu_color))
                    print("History:", self.draw_sparkline(self.cpu_history, 50))
                    print()

                    # Memory Usage
                    mem_current = self.memory_history[-1] if self.memory_history else 0
                    mem_info = psutil.virtual_memory()
                    mem_color = (
                        TERM.red
                        if mem_current > 80
                        else TERM.yellow
                        if mem_current > 60
                        else TERM.green
                    )

                    print(
                        TERM.bold("Memory:"),
                        f"{mem_current:.1f}% ({mem_info.used / 1024**3:.1f}GB / {mem_info.total / 1024**3:.1f}GB)",
                    )
                    print(self.draw_bar(mem_current, 100, 40, mem_color))
                    print("History:", self.draw_sparkline(self.memory_history, 50))
                    print()

                    # GPU Usage
                    gpu_current = self.gpu_history[-1] if self.gpu_history else 0
                    gpu_color = (
                        TERM.red
                        if gpu_current > 80
                        else TERM.yellow
                        if gpu_current > 60
                        else TERM.green
                    )

                    print(TERM.bold("GPU (Metal):"), f"{gpu_current:.1f}%")
                    print(self.draw_bar(gpu_current, 100, 40, gpu_color))
                    print("History:", self.draw_sparkline(self.gpu_history, 50))
                    print()

                    # Temperature
                    temp_current = self.temp_history[-1] if self.temp_history else 0
                    temp_color = (
                        TERM.red
                        if temp_current > 80
                        else TERM.yellow
                        if temp_current > 65
                        else TERM.green
                    )

                    print(
                        TERM.bold("CPU Temperature:"),
                        temp_color(f"{temp_current:.1f}°C"),
                    )
                    print(self.draw_bar(temp_current, 100, 40, temp_color))
                    print()

                # Wheel Trading Processes
                print(TERM.cyan("─" * TERM.width))
                print(TERM.bold("Wheel Trading Processes:"))
                processes = self.get_process_info()

                if processes:
                    print(f"{'PID':<8} {'CPU%':<8} {'MEM%':<8} {'Command'}")
                    for proc in processes:
                        cpu_color = (
                            TERM.red
                            if proc["cpu"] > 50
                            else TERM.yellow
                            if proc["cpu"] > 25
                            else TERM.normal
                        )
                        print(
                            f"{proc['pid']:<8} {cpu_color}{proc['cpu']:<8.1f}{TERM.normal} {proc['memory']:<8.1f} {proc['cmdline']}"
                        )
                else:
                    print(TERM.dim("No wheel trading processes running"))

                print()

                # System Info
                print(TERM.cyan("─" * TERM.width))
                print(TERM.bold("System Info:"))
                print(f"Cores: {psutil.cpu_count(logical=False)}P+4E (12 total)")
                print(f"Load Average: {', '.join(f'{x:.2f}' for x in os.getloadavg())}")
                print(f"Uptime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Performance Tips
                if cpu_current > 80:
                    print(TERM.yellow("\n⚠️  High CPU usage detected. Consider:"))
                    print("   - Reducing parallel operations")
                    print("   - Checking for runaway processes")

                if mem_current > 80:
                    print(TERM.yellow("\n⚠️  High memory usage detected. Consider:"))
                    print("   - Clearing DuckDB cache")
                    print("   - Restarting orchestrator")

                if temp_current > 75:
                    print(TERM.yellow("\n⚠️  High temperature detected. Consider:"))
                    print("   - Increasing TG Pro fan speed")
                    print("   - Reducing workload")

                print(TERM.dim("\nPress Ctrl+C to exit"))

                time.sleep(2)

    def run(self):
        """Run the monitor"""
        # Start metrics collection thread
        metrics_thread = threading.Thread(target=self.update_metrics, daemon=True)
        metrics_thread.start()

        try:
            self.display()
        except KeyboardInterrupt:
            self.running = False
            print(TERM.normal + "\nMonitoring stopped.")


def main():
    """Main entry point"""
    print("Starting M4 Pro Performance Monitor...")
    print("Note: Some metrics require sudo access for accuracy.")
    print()

    monitor = M4ProMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
