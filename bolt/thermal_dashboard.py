#!/usr/bin/env python3
"""
Real-time Thermal Dashboard for M4 Pro
Advanced thermal monitoring dashboard with visual analytics and alerts
"""

import asyncio
import json
import logging
import time
from typing import Any

try:
    import blessed

    HAS_BLESSED = True
except ImportError:
    HAS_BLESSED = False

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import contextlib

from .hardware.performance_monitor import get_performance_monitor
from .thermal_monitor import (
    PerformanceLevel,
    ThermalMonitor,
    ThermalReading,
    ThermalState,
    get_thermal_monitor,
)

logger = logging.getLogger(__name__)


class ThermalDashboard:
    """Real-time thermal monitoring dashboard"""

    def __init__(self, thermal_monitor: ThermalMonitor | None = None):
        self.thermal_monitor = thermal_monitor or get_thermal_monitor()
        self.performance_monitor = get_performance_monitor()

        self.is_running = False
        self.dashboard_task: asyncio.Task | None = None

        # Dashboard configuration
        self.refresh_rate = 1.0  # seconds
        self.alert_threshold = 80.0  # °C
        self.chart_width = 60
        self.chart_height = 15

        # Initialize terminal if available
        if HAS_BLESSED:
            self.term = blessed.Terminal()
        else:
            self.term = None
            logger.warning("Blessed not available - using basic text output")

        logger.info("🖥️ Thermal Dashboard initialized")

    async def start(self):
        """Start the thermal dashboard"""
        if self.is_running:
            logger.warning("Thermal dashboard already running")
            return

        # Start thermal monitoring if not already running
        if not self.thermal_monitor.is_monitoring:
            await self.thermal_monitor.start_monitoring()

        self.is_running = True

        if self.term and HAS_BLESSED:
            self.dashboard_task = asyncio.create_task(self._run_interactive_dashboard())
        else:
            self.dashboard_task = asyncio.create_task(self._run_text_dashboard())

        logger.info("🖥️ Thermal dashboard started")

    async def stop(self):
        """Stop the thermal dashboard"""
        self.is_running = False
        if self.dashboard_task:
            self.dashboard_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.dashboard_task
        logger.info("🖥️ Thermal dashboard stopped")

    async def _run_interactive_dashboard(self):
        """Run interactive blessed-based dashboard"""
        with self.term.fullscreen(), self.term.hidden_cursor():
            while self.is_running:
                try:
                    await self._render_interactive_frame()
                    await asyncio.sleep(self.refresh_rate)
                except Exception as e:
                    logger.error(f"Dashboard rendering error: {e}")
                    await asyncio.sleep(5.0)

    async def _run_text_dashboard(self):
        """Run simple text-based dashboard"""
        while self.is_running:
            try:
                await self._render_text_frame()
                await asyncio.sleep(self.refresh_rate)
            except Exception as e:
                logger.error(f"Text dashboard error: {e}")
                await asyncio.sleep(5.0)

    async def _render_interactive_frame(self):
        """Render interactive dashboard frame"""
        if not self.term:
            return

        print(self.term.home + self.term.clear)

        # Header
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        header = f"🌡️ M4 Pro Thermal Monitor - {timestamp}"
        print(self.term.bold_cyan(header))
        print(self.term.cyan("═" * len(header)))
        print()

        # Get current data
        current_reading = self.thermal_monitor.get_current_reading()
        thermal_report = self.thermal_monitor.get_thermal_report()
        performance_dashboard = self.performance_monitor.get_real_time_dashboard()

        if not current_reading:
            print(self.term.red("No thermal data available"))
            return

        # Temperature display
        await self._render_temperature_section(current_reading)
        print()

        # Thermal trends
        await self._render_trend_section(thermal_report.get("thermal_trend", {}))
        print()

        # Performance section
        await self._render_performance_section(current_reading, performance_dashboard)
        print()

        # Alerts and recommendations
        await self._render_alerts_section(thermal_report.get("recommendations", []))
        print()

        # System status
        await self._render_system_status(current_reading, thermal_report)

        # Footer
        print(
            self.term.dim(
                f"\nPress Ctrl+C to exit | Monitoring interval: {self.refresh_rate:.1f}s"
            )
        )

    async def _render_text_frame(self):
        """Render simple text dashboard frame"""
        current_reading = self.thermal_monitor.get_current_reading()
        if not current_reading:
            print("No thermal data available")
            return

        timestamp = time.strftime("%H:%M:%S")
        print(f"\n[{timestamp}] Thermal Status:")
        print(f"  CPU: {current_reading.cpu_temp:.1f}°C")
        print(f"  GPU: {current_reading.gpu_temp:.1f}°C")
        print(f"  P-cores: {current_reading.p_core_temp:.1f}°C")
        print(f"  E-cores: {current_reading.e_core_temp:.1f}°C")
        print(f"  State: {current_reading.thermal_state.value}")
        print(f"  Performance: {current_reading.performance_level.value}")

        if current_reading.throttling_active:
            print("  ⚠️ THROTTLING ACTIVE")

        # Show alerts for high temperatures
        max_temp = max(
            current_reading.cpu_temp,
            current_reading.gpu_temp,
            current_reading.p_core_temp,
            current_reading.e_core_temp,
        )
        if max_temp > self.alert_threshold:
            print(f"  🚨 HIGH TEMPERATURE ALERT: {max_temp:.1f}°C")

    async def _render_temperature_section(self, reading: ThermalReading):
        """Render temperature section with visual bars"""
        if not self.term:
            return

        print(self.term.bold("🌡️ Temperature Readings"))
        print()

        # Temperature data
        temps = [
            ("CPU", reading.cpu_temp),
            ("GPU", reading.gpu_temp),
            ("P-cores", reading.p_core_temp),
            ("E-cores", reading.e_core_temp),
        ]

        for name, temp in temps:
            if temp == 0:
                continue

            # Color coding
            if temp >= 90:
                color = self.term.red
            elif temp >= 80:
                color = self.term.yellow
            elif temp >= 70:
                color = self.term.cyan
            else:
                color = self.term.green

            # Temperature bar
            bar_width = 40
            temp_ratio = min(temp / 100.0, 1.0)
            filled = int(temp_ratio * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)

            print(
                f"{name:>8}: {color}{temp:5.1f}°C{self.term.normal} {color}{bar}{self.term.normal}"
            )

        # Thermal state indicator
        state_color = self._get_thermal_state_color(reading.thermal_state)
        print(
            f"\nThermal State: {state_color}{reading.thermal_state.value.upper()}{self.term.normal}"
        )

        # Performance level
        perf_color = self._get_performance_level_color(reading.performance_level)
        print(
            f"Performance:   {perf_color}{reading.performance_level.value.upper()}{self.term.normal}"
        )

        if reading.throttling_active:
            print(
                f"Status:        {self.term.blink_red}THROTTLING ACTIVE{self.term.normal}"
            )

    async def _render_trend_section(self, trend_data: dict[str, Any]):
        """Render thermal trend section"""
        if not self.term or not trend_data:
            return

        print(self.term.bold("📈 Thermal Trends (5 min)"))
        print()

        cpu_trend = trend_data.get("cpu_trend", 0)
        gpu_trend = trend_data.get("gpu_trend", 0)

        # Trend indicators
        cpu_indicator = self._get_trend_indicator(cpu_trend)
        gpu_indicator = self._get_trend_indicator(gpu_trend)

        print(f"CPU Trend:  {cpu_indicator} {cpu_trend:+.1f}°C/min")
        print(f"GPU Trend:  {gpu_indicator} {gpu_trend:+.1f}°C/min")

        # Temperature ranges
        if "max_cpu_temp" in trend_data:
            print(
                f"CPU Range:  {trend_data['avg_cpu_temp']:.1f}°C avg, {trend_data['max_cpu_temp']:.1f}°C max"
            )
        if "max_gpu_temp" in trend_data:
            print(
                f"GPU Range:  {trend_data['avg_gpu_temp']:.1f}°C avg, {trend_data['max_gpu_temp']:.1f}°C max"
            )

        # Sparkline chart for recent history
        await self._render_sparkline_chart()

    async def _render_sparkline_chart(self):
        """Render sparkline chart of recent temperatures"""
        if not self.term:
            return

        # Get recent temperature data
        history = list(self.thermal_monitor.thermal_history)[-self.chart_width :]
        if len(history) < 2:
            return

        cpu_temps = [r.cpu_temp for r in history if r.cpu_temp > 0]
        gpu_temps = [r.gpu_temp for r in history if r.gpu_temp > 0]

        if not cpu_temps and not gpu_temps:
            return

        print()
        print("Temperature History:")

        # CPU sparkline
        if cpu_temps:
            cpu_sparkline = self._create_sparkline(cpu_temps, self.chart_width)
            print(f"CPU: {cpu_sparkline}")

        # GPU sparkline
        if gpu_temps:
            gpu_sparkline = self._create_sparkline(gpu_temps, self.chart_width)
            print(f"GPU: {gpu_sparkline}")

    def _create_sparkline(self, data: list[float], width: int) -> str:
        """Create sparkline representation of data"""
        if not data:
            return " " * width

        chars = "▁▂▃▄▅▆▇█"
        min_val = min(data)
        max_val = max(data)
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

    async def _render_performance_section(
        self, reading: ThermalReading, perf_data: dict[str, Any]
    ):
        """Render performance section"""
        if not self.term:
            return

        print(self.term.bold("⚡ Performance Status"))
        print()

        # Current performance profile
        profile = self.thermal_monitor.current_profile
        print(
            f"CPU Workers:    {profile.max_cpu_workers} ({profile.max_p_core_workers}P + {profile.max_e_core_workers}E)"
        )
        print(f"GPU Workers:    {profile.max_gpu_workers}")
        print(f"Memory Limit:   {profile.memory_limit_gb}GB")
        print(f"Batch Mult:     {profile.batch_size_multiplier:.1f}x")

        # System utilization if available
        if perf_data and "system" in perf_data:
            system = perf_data["system"]
            cpu_util = system.get("cpu_overall_percent", 0)
            mem_util = system.get("memory_percent", 0)
            gpu_util = system.get("gpu_utilization", 0)

            print("\nCurrent Usage:")
            print(f"CPU:            {cpu_util:.1f}%")
            print(f"Memory:         {mem_util:.1f}%")
            print(f"GPU:            {gpu_util:.1f}%")

    async def _render_alerts_section(self, recommendations: list[str]):
        """Render alerts and recommendations section"""
        if not self.term or not recommendations:
            return

        print(self.term.bold("🚨 Alerts & Recommendations"))
        print()

        for rec in recommendations:
            if "CRITICAL" in rec or "🚨" in rec:
                print(self.term.blink_red(rec))
            elif "HIGH" in rec or "🔥" in rec:
                print(self.term.red(rec))
            elif "⚠️" in rec:
                print(self.term.yellow(rec))
            else:
                print(self.term.cyan(rec))

    async def _render_system_status(
        self, reading: ThermalReading, report: dict[str, Any]
    ):
        """Render system status section"""
        if not self.term:
            return

        print(self.term.bold("💻 System Status"))
        print()

        # Monitoring status
        uptime = time.time() - (
            self.thermal_monitor.thermal_history[0].timestamp
            if self.thermal_monitor.thermal_history
            else time.time()
        )
        print(f"Monitoring:     {uptime/60:.1f} minutes")
        print(f"Readings:       {len(self.thermal_monitor.thermal_history)}")

        # Predicted state
        predicted = self.thermal_monitor.predict_thermal_state(minutes_ahead=5)
        if predicted:
            pred_color = self._get_thermal_state_color(predicted)
            print(f"Predicted (5m): {pred_color}{predicted.value}{self.term.normal}")

        # Fan speed if available
        if reading.fan_speed > 0:
            print(f"Fan Speed:      {reading.fan_speed} RPM")

        # Thermal statistics
        stats = report.get("thermal_statistics", {})
        if stats:
            print(
                f"Temp Range:     {stats.get('min_temp', 0):.1f}°C - {stats.get('max_temp', 0):.1f}°C"
            )
            print(f"Average:        {stats.get('avg_temp', 0):.1f}°C")

    def _get_thermal_state_color(self, state: ThermalState):
        """Get color for thermal state"""
        if not self.term:
            return ""

        colors = {
            ThermalState.OPTIMAL: self.term.green,
            ThermalState.WARM: self.term.cyan,
            ThermalState.HOT: self.term.yellow,
            ThermalState.CRITICAL: self.term.red,
        }
        return colors.get(state, self.term.normal)

    def _get_performance_level_color(self, level: PerformanceLevel):
        """Get color for performance level"""
        if not self.term:
            return ""

        colors = {
            PerformanceLevel.MAXIMUM: self.term.green,
            PerformanceLevel.HIGH: self.term.cyan,
            PerformanceLevel.MEDIUM: self.term.yellow,
            PerformanceLevel.LOW: self.term.yellow,
            PerformanceLevel.MINIMAL: self.term.red,
        }
        return colors.get(level, self.term.normal)

    def _get_trend_indicator(self, trend: float) -> str:
        """Get trend indicator symbol"""
        if trend > 2:
            return "🔺"  # Rising fast
        elif trend > 0.5:
            return "📈"  # Rising
        elif trend < -2:
            return "🔻"  # Falling fast
        elif trend < -0.5:
            return "📉"  # Falling
        else:
            return "➡️"  # Stable

    async def generate_thermal_chart(
        self, filename: str = "thermal_chart.png", hours: int = 1
    ):
        """Generate thermal chart image"""
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib not available - cannot generate charts")
            return

        # Get data from the specified time period
        cutoff_time = time.time() - (hours * 3600)
        history = [
            r for r in self.thermal_monitor.thermal_history if r.timestamp > cutoff_time
        ]

        if len(history) < 2:
            logger.warning("Insufficient data for chart generation")
            return

        # Prepare data
        timestamps = [r.timestamp for r in history]
        cpu_temps = [r.cpu_temp for r in history]
        gpu_temps = [r.gpu_temp for r in history]
        p_core_temps = [r.p_core_temp for r in history]
        e_core_temps = [r.e_core_temp for r in history]

        # Convert timestamps to relative minutes
        start_time = timestamps[0]
        time_minutes = [(t - start_time) / 60 for t in timestamps]

        # Create chart
        plt.figure(figsize=(12, 8))

        # Temperature plot
        plt.subplot(2, 1, 1)
        plt.plot(time_minutes, cpu_temps, label="CPU", linewidth=2, color="red")
        plt.plot(time_minutes, gpu_temps, label="GPU", linewidth=2, color="blue")
        plt.plot(
            time_minutes,
            p_core_temps,
            label="P-cores",
            linewidth=1,
            color="orange",
            alpha=0.7,
        )
        plt.plot(
            time_minutes,
            e_core_temps,
            label="E-cores",
            linewidth=1,
            color="green",
            alpha=0.7,
        )

        # Thermal thresholds
        plt.axhline(
            y=75, color="yellow", linestyle="--", alpha=0.5, label="Warm threshold"
        )
        plt.axhline(
            y=85, color="orange", linestyle="--", alpha=0.5, label="Hot threshold"
        )
        plt.axhline(
            y=95, color="red", linestyle="--", alpha=0.5, label="Critical threshold"
        )

        plt.title("M4 Pro Thermal History")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Thermal state plot
        plt.subplot(2, 1, 2)
        thermal_states = [r.thermal_state for r in history]
        state_values = [self._thermal_state_to_value(state) for state in thermal_states]

        plt.plot(time_minutes, state_values, linewidth=2, color="purple")
        plt.fill_between(time_minutes, state_values, alpha=0.3, color="purple")

        plt.title("Thermal State Over Time")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Thermal State")
        plt.yticks([0, 1, 2, 3], ["Optimal", "Warm", "Hot", "Critical"])
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"📊 Thermal chart saved to {filename}")

    def _thermal_state_to_value(self, state: ThermalState) -> int:
        """Convert thermal state to numeric value for plotting"""
        mapping = {
            ThermalState.OPTIMAL: 0,
            ThermalState.WARM: 1,
            ThermalState.HOT: 2,
            ThermalState.CRITICAL: 3,
        }
        return mapping.get(state, 0)

    async def export_thermal_data(
        self, filename: str = "thermal_data.json", hours: int = 24
    ):
        """Export thermal data to JSON file"""
        cutoff_time = time.time() - (hours * 3600)
        history = [
            r for r in self.thermal_monitor.thermal_history if r.timestamp > cutoff_time
        ]

        # Convert to serializable format
        data = {
            "export_timestamp": time.time(),
            "export_hours": hours,
            "readings_count": len(history),
            "thermal_readings": [
                {
                    "timestamp": r.timestamp,
                    "cpu_temp": r.cpu_temp,
                    "gpu_temp": r.gpu_temp,
                    "p_core_temp": r.p_core_temp,
                    "e_core_temp": r.e_core_temp,
                    "ambient_temp": r.ambient_temp,
                    "thermal_state": r.thermal_state.value,
                    "performance_level": r.performance_level.value,
                    "fan_speed": r.fan_speed,
                    "throttling_active": r.throttling_active,
                }
                for r in history
            ],
            "summary_statistics": self._calculate_summary_statistics(history),
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"📄 Thermal data exported to {filename}")

    def _calculate_summary_statistics(
        self, history: list[ThermalReading]
    ) -> dict[str, Any]:
        """Calculate summary statistics for thermal data"""
        if not history:
            return {}

        cpu_temps = [r.cpu_temp for r in history if r.cpu_temp > 0]
        gpu_temps = [r.gpu_temp for r in history if r.gpu_temp > 0]

        return {
            "monitoring_duration_hours": (history[-1].timestamp - history[0].timestamp)
            / 3600,
            "cpu_temperature": {
                "min": min(cpu_temps) if cpu_temps else 0,
                "max": max(cpu_temps) if cpu_temps else 0,
                "avg": sum(cpu_temps) / len(cpu_temps) if cpu_temps else 0,
            },
            "gpu_temperature": {
                "min": min(gpu_temps) if gpu_temps else 0,
                "max": max(gpu_temps) if gpu_temps else 0,
                "avg": sum(gpu_temps) / len(gpu_temps) if gpu_temps else 0,
            },
            "thermal_state_distribution": {
                state.value: len([r for r in history if r.thermal_state == state])
                for state in ThermalState
            },
            "throttling_events": len([r for r in history if r.throttling_active]),
        }


class ThermalAlertSystem:
    """Thermal alert system with notification support"""

    def __init__(self, thermal_monitor: ThermalMonitor):
        self.thermal_monitor = thermal_monitor
        self.alert_history = []
        self.alert_thresholds = {
            "temperature_warning": 80.0,
            "temperature_critical": 90.0,
            "trend_warning": 3.0,  # °C/min
            "consecutive_warnings": 3,
        }

        # Register thermal callback
        thermal_monitor.register_thermal_callback(self._on_thermal_change)

        self.consecutive_warnings = 0
        self.last_alert_time = 0
        self.alert_cooldown = 60  # seconds

    async def _on_thermal_change(
        self, current: ThermalReading, previous: ThermalReading
    ):
        """Handle thermal state changes for alerts"""
        current_time = time.time()

        # Check for temperature alerts
        max_temp = max(
            current.cpu_temp, current.gpu_temp, current.p_core_temp, current.e_core_temp
        )

        alert_sent = False

        # Critical temperature alert
        if max_temp >= self.alert_thresholds["temperature_critical"]:
            await self._send_alert(
                "CRITICAL",
                f"Critical temperature reached: {max_temp:.1f}°C",
                priority="critical",
            )
            alert_sent = True

        # Warning temperature alert
        elif max_temp >= self.alert_thresholds["temperature_warning"]:
            self.consecutive_warnings += 1
            if (
                self.consecutive_warnings
                >= self.alert_thresholds["consecutive_warnings"]
            ):
                await self._send_alert(
                    "WARNING",
                    f"High temperature sustained: {max_temp:.1f}°C",
                    priority="warning",
                )
                alert_sent = True
                self.consecutive_warnings = 0
        else:
            self.consecutive_warnings = 0

        # Thermal state change alert
        if current.thermal_state != previous.thermal_state:
            if current.thermal_state in [ThermalState.HOT, ThermalState.CRITICAL]:
                await self._send_alert(
                    "STATE_CHANGE",
                    f"Thermal state changed to {current.thermal_state.value}",
                    priority="warning"
                    if current.thermal_state == ThermalState.HOT
                    else "critical",
                )
                alert_sent = True

        # Throttling alert
        if current.throttling_active and not previous.throttling_active:
            await self._send_alert(
                "THROTTLING", "System thermal throttling activated", priority="critical"
            )
            alert_sent = True

        if alert_sent:
            self.last_alert_time = current_time

    async def _send_alert(self, alert_type: str, message: str, priority: str = "info"):
        """Send thermal alert"""
        current_time = time.time()

        # Implement cooldown to prevent spam
        if (
            current_time - self.last_alert_time < self.alert_cooldown
            and priority != "critical"
        ):
            return

        alert = {
            "timestamp": current_time,
            "type": alert_type,
            "message": message,
            "priority": priority,
        }

        self.alert_history.append(alert)

        # Log the alert
        log_func = logger.critical if priority == "critical" else logger.warning
        log_func(f"🚨 THERMAL ALERT [{alert_type}]: {message}")

        # Here you could add additional notification methods:
        # - Email notifications
        # - Slack/Discord webhooks
        # - System notifications
        # - SMS alerts for critical issues

        await self._log_alert_to_file(alert)

    async def _log_alert_to_file(self, alert: dict[str, Any]):
        """Log alert to file for persistence"""
        try:
            import json
            from pathlib import Path

            alert_file = Path("thermal_alerts.jsonl")
            with open(alert_file, "a") as f:
                f.write(json.dumps(alert) + "\n")
        except Exception as e:
            logger.error(f"Failed to log alert to file: {e}")


# Global dashboard instance
_thermal_dashboard = None


def get_thermal_dashboard() -> ThermalDashboard:
    """Get global thermal dashboard instance"""
    global _thermal_dashboard
    if _thermal_dashboard is None:
        _thermal_dashboard = ThermalDashboard()
    return _thermal_dashboard


if __name__ == "__main__":

    async def demo():
        """Demonstrate thermal dashboard"""
        print("🖥️ Starting Thermal Dashboard Demo")

        # Initialize components
        thermal_monitor = get_thermal_monitor()
        dashboard = get_thermal_dashboard()
        ThermalAlertSystem(thermal_monitor)

        try:
            # Start monitoring and dashboard
            await thermal_monitor.start_monitoring()
            await dashboard.start()

            # Let it run for a while (in real usage, this would run indefinitely)
            await asyncio.sleep(30)

        except KeyboardInterrupt:
            print("\n🛑 Dashboard demo stopped by user")
        finally:
            await dashboard.stop()
            await thermal_monitor.stop_monitoring()

    asyncio.run(demo())
