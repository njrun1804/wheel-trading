#!/usr/bin/env python3
"""
Thermal Monitoring and Performance Optimization for M4 Pro
Comprehensive thermal management system for preventing throttling during trading operations
"""

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class ThermalState(Enum):
    """Thermal state levels"""

    OPTIMAL = "optimal"  # < 65°C
    WARM = "warm"  # 65-75°C
    HOT = "hot"  # 75-85°C
    CRITICAL = "critical"  # > 85°C


class PerformanceLevel(Enum):
    """Performance throttling levels"""

    MAXIMUM = "maximum"  # 100% performance
    HIGH = "high"  # 80% performance
    MEDIUM = "medium"  # 60% performance
    LOW = "low"  # 40% performance
    MINIMAL = "minimal"  # 20% performance


@dataclass
class ThermalReading:
    """Single thermal reading with metadata"""

    timestamp: float
    cpu_temp: float
    gpu_temp: float
    ambient_temp: float
    p_core_temp: float
    e_core_temp: float
    thermal_state: ThermalState
    performance_level: PerformanceLevel
    fan_speed: int = 0
    throttling_active: bool = False


@dataclass
class ThermalThresholds:
    """Thermal management thresholds"""

    optimal_max: float = 65.0  # Below this is optimal
    warm_max: float = 75.0  # Above this is warm
    hot_max: float = 85.0  # Above this is hot
    critical_max: float = 95.0  # Above this is critical

    # Hysteresis to prevent oscillation
    hysteresis: float = 3.0

    # Performance reduction thresholds
    throttle_start: float = 80.0  # Start throttling
    throttle_aggressive: float = 88.0  # Aggressive throttling


@dataclass
class PerformanceProfile:
    """Performance profile for different thermal states"""

    max_cpu_workers: int
    max_p_core_workers: int
    max_e_core_workers: int
    max_gpu_workers: int
    batch_size_multiplier: float
    polling_interval: float
    memory_limit_gb: float


class ThermalMonitor:
    """Advanced thermal monitoring and performance optimization for M4 Pro"""

    def __init__(self, max_history: int = 3600):  # 1 hour of data
        self.max_history = max_history
        self.thermal_history = deque(maxlen=max_history)
        self.thresholds = ThermalThresholds()

        # Monitoring state
        self.is_monitoring = False
        self.monitor_task: asyncio.Task | None = None
        self.monitor_thread: threading.Thread | None = None
        self.data_lock = threading.Lock()

        # Performance management
        self.current_profile = self._get_performance_profile(ThermalState.OPTIMAL)
        self.thermal_callbacks: list[Callable] = []
        self.performance_callbacks: list[Callable] = []

        # Thermal prediction
        self.thermal_trend_window = 10  # seconds
        self.prediction_accuracy = deque(maxlen=100)

        # Performance profiles for different thermal states
        self.performance_profiles = {
            ThermalState.OPTIMAL: PerformanceProfile(
                max_cpu_workers=8,
                max_p_core_workers=8,
                max_e_core_workers=4,
                max_gpu_workers=4,
                batch_size_multiplier=1.0,
                polling_interval=1.0,
                memory_limit_gb=18.0,
            ),
            ThermalState.WARM: PerformanceProfile(
                max_cpu_workers=6,
                max_p_core_workers=6,
                max_e_core_workers=4,
                max_gpu_workers=3,
                batch_size_multiplier=0.8,
                polling_interval=0.8,
                memory_limit_gb=16.0,
            ),
            ThermalState.HOT: PerformanceProfile(
                max_cpu_workers=4,
                max_p_core_workers=4,
                max_e_core_workers=3,
                max_gpu_workers=2,
                batch_size_multiplier=0.6,
                polling_interval=0.5,
                memory_limit_gb=14.0,
            ),
            ThermalState.CRITICAL: PerformanceProfile(
                max_cpu_workers=2,
                max_p_core_workers=2,
                max_e_core_workers=2,
                max_gpu_workers=1,
                batch_size_multiplier=0.4,
                polling_interval=0.3,
                memory_limit_gb=12.0,
            ),
        }

        logger.info("🌡️ Thermal Monitor initialized for M4 Pro")

    def _get_performance_profile(
        self, thermal_state: ThermalState
    ) -> PerformanceProfile:
        """Get performance profile for thermal state"""
        return self.performance_profiles.get(
            thermal_state, self.performance_profiles[ThermalState.OPTIMAL]
        )

    async def start_monitoring(self, interval: float = 1.0):
        """Start thermal monitoring"""
        if self.is_monitoring:
            logger.warning("Thermal monitoring already running")
            return

        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info("🌡️ Thermal monitoring started")

    async def stop_monitoring(self):
        """Stop thermal monitoring"""
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitor_task
        logger.info("🌡️ Thermal monitoring stopped")

    async def _monitoring_loop(self, interval: float):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                reading = await self._take_thermal_reading()

                with self.data_lock:
                    self.thermal_history.append(reading)

                # Trigger callbacks for thermal state changes
                await self._handle_thermal_state_change(reading)

                # Trigger performance adjustments
                await self._handle_performance_adjustment(reading)

                # Log warnings for concerning temperatures
                await self._log_thermal_warnings(reading)

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Thermal monitoring error: {e}")
                await asyncio.sleep(5.0)

    async def _take_thermal_reading(self) -> ThermalReading:
        """Take comprehensive thermal reading"""
        timestamp = time.time()

        # Default values
        cpu_temp = 0.0
        gpu_temp = 0.0
        ambient_temp = 0.0
        p_core_temp = 0.0
        e_core_temp = 0.0
        fan_speed = 0

        try:
            # Get detailed thermal data using powermetrics
            proc = await asyncio.create_subprocess_exec(
                "sudo",
                "powermetrics",
                "--samplers",
                "smc,thermal",
                "-n",
                "1",
                "-i",
                "100",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                output = stdout.decode()

                # Parse thermal data
                for line in output.split("\n"):
                    line = line.strip()

                    if "CPU die temperature" in line:
                        cpu_temp = self._extract_temperature(line)
                    elif "GPU die temperature" in line:
                        gpu_temp = self._extract_temperature(line)
                    elif "P-core" in line and "temp" in line.lower():
                        p_core_temp = max(p_core_temp, self._extract_temperature(line))
                    elif "E-core" in line and "temp" in line.lower():
                        e_core_temp = max(e_core_temp, self._extract_temperature(line))
                    elif "Ambient" in line and "temp" in line.lower():
                        ambient_temp = self._extract_temperature(line)
                    elif "Fan" in line and ("RPM" in line or "speed" in line):
                        fan_speed = self._extract_number(line)

        except Exception as e:
            logger.debug(f"powermetrics thermal reading failed: {e}")

        # Fallback to simpler methods if powermetrics fails
        if cpu_temp == 0.0:
            cpu_temp = await self._get_cpu_temp_fallback()

        # Determine thermal state
        max_temp = max(cpu_temp, gpu_temp, p_core_temp, e_core_temp)
        thermal_state = self._determine_thermal_state(max_temp)

        # Determine performance level
        performance_level = self._determine_performance_level(thermal_state, max_temp)

        # Check for throttling
        throttling_active = await self._detect_throttling()

        return ThermalReading(
            timestamp=timestamp,
            cpu_temp=cpu_temp,
            gpu_temp=gpu_temp,
            ambient_temp=ambient_temp,
            p_core_temp=p_core_temp,
            e_core_temp=e_core_temp,
            thermal_state=thermal_state,
            performance_level=performance_level,
            fan_speed=fan_speed,
            throttling_active=throttling_active,
        )

    def _extract_temperature(self, line: str) -> float:
        """Extract temperature from powermetrics line"""
        try:
            # Look for temperature patterns
            parts = line.split()
            for i, part in enumerate(parts):
                if "C" in part or "°C" in part:
                    temp_str = part.replace("C", "").replace("°", "").replace(":", "")
                    return float(temp_str)
                elif part.replace(".", "").isdigit() and i < len(parts) - 1:
                    if "C" in parts[i + 1] or "°C" in parts[i + 1]:
                        return float(part)
        except (ValueError, IndexError):
            pass
        return 0.0

    def _extract_number(self, line: str) -> int:
        """Extract number from line"""
        try:
            import re

            numbers = re.findall(r"\d+", line)
            return int(numbers[0]) if numbers else 0
        except (ValueError, IndexError):
            return 0

    async def _get_cpu_temp_fallback(self) -> float:
        """Fallback method to get CPU temperature"""
        try:
            # Try alternative method
            proc = await asyncio.create_subprocess_exec(
                "sysctl",
                "-n",
                "machdep.xcpm.cpu_thermal_state",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                # This gives thermal state, not temperature
                # Estimate temperature based on thermal state
                thermal_state = int(stdout.decode().strip())
                return 50.0 + (thermal_state * 10.0)  # Rough estimate

        except Exception:
            pass

        return 0.0

    def _determine_thermal_state(self, max_temp: float) -> ThermalState:
        """Determine thermal state based on temperature"""
        if max_temp >= self.thresholds.critical_max:
            return ThermalState.CRITICAL
        elif max_temp >= self.thresholds.hot_max:
            return ThermalState.HOT
        elif max_temp >= self.thresholds.warm_max:
            return ThermalState.WARM
        else:
            return ThermalState.OPTIMAL

    def _determine_performance_level(
        self, thermal_state: ThermalState, max_temp: float
    ) -> PerformanceLevel:
        """Determine performance level based on thermal state and temperature"""
        if thermal_state == ThermalState.CRITICAL:
            return PerformanceLevel.MINIMAL
        elif thermal_state == ThermalState.HOT:
            if max_temp > self.thresholds.throttle_aggressive:
                return PerformanceLevel.LOW
            else:
                return PerformanceLevel.MEDIUM
        elif thermal_state == ThermalState.WARM:
            return PerformanceLevel.HIGH
        else:
            return PerformanceLevel.MAXIMUM

    async def _detect_throttling(self) -> bool:
        """Detect if system is currently throttling"""
        try:
            # Check CPU frequency to detect throttling
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                # If current frequency is significantly below max, might be throttling
                if cpu_freq.current < cpu_freq.max * 0.8:
                    return True

            # Check thermal pressure
            proc = await asyncio.create_subprocess_exec(
                "sysctl",
                "-n",
                "machdep.xcpm.cpu_thermal_state",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                thermal_state = int(stdout.decode().strip())
                return thermal_state > 0  # Non-zero indicates thermal pressure

        except Exception:
            pass

        return False

    async def _handle_thermal_state_change(self, reading: ThermalReading):
        """Handle thermal state changes"""
        if len(self.thermal_history) < 2:
            return

        previous_reading = self.thermal_history[-2]

        if reading.thermal_state != previous_reading.thermal_state:
            logger.info(
                f"🌡️ Thermal state changed: {previous_reading.thermal_state.value} -> {reading.thermal_state.value}"
            )

            # Trigger callbacks
            for callback in self.thermal_callbacks:
                try:
                    await callback(reading, previous_reading)
                except Exception as e:
                    logger.error(f"Thermal callback error: {e}")

    async def _handle_performance_adjustment(self, reading: ThermalReading):
        """Handle performance adjustments based on thermal state"""
        new_profile = self._get_performance_profile(reading.thermal_state)

        if new_profile != self.current_profile:
            logger.info(f"🔧 Performance profile changed: {reading.thermal_state.value}")
            self.current_profile = new_profile

            # Trigger performance callbacks
            for callback in self.performance_callbacks:
                try:
                    await callback(new_profile, reading)
                except Exception as e:
                    logger.error(f"Performance callback error: {e}")

    async def _log_thermal_warnings(self, reading: ThermalReading):
        """Log thermal warnings and recommendations"""
        max_temp = max(
            reading.cpu_temp, reading.gpu_temp, reading.p_core_temp, reading.e_core_temp
        )

        if reading.thermal_state == ThermalState.CRITICAL:
            logger.critical(
                f"🚨 CRITICAL TEMPERATURE: {max_temp:.1f}°C - Emergency throttling active"
            )
        elif reading.thermal_state == ThermalState.HOT:
            logger.warning(
                f"🔥 HIGH TEMPERATURE: {max_temp:.1f}°C - Performance throttling active"
            )
        elif (
            reading.thermal_state == ThermalState.WARM and len(self.thermal_history) > 1
        ):
            # Only warn if temperature is rising
            prev_temp = max(
                self.thermal_history[-2].cpu_temp, self.thermal_history[-2].gpu_temp
            )
            if max_temp > prev_temp + 2:
                logger.warning(f"🌡️ Temperature rising: {max_temp:.1f}°C")

        if reading.throttling_active:
            logger.warning("⚠️ System throttling detected - reducing workload")

    def register_thermal_callback(self, callback: Callable):
        """Register callback for thermal state changes"""
        self.thermal_callbacks.append(callback)

    def register_performance_callback(self, callback: Callable):
        """Register callback for performance profile changes"""
        self.performance_callbacks.append(callback)

    def get_current_reading(self) -> ThermalReading | None:
        """Get most recent thermal reading"""
        with self.data_lock:
            return self.thermal_history[-1] if self.thermal_history else None

    def get_thermal_trend(self, minutes: int = 5) -> dict[str, float]:
        """Get thermal trend over specified time period"""
        if not self.thermal_history:
            return {}

        cutoff_time = time.time() - (minutes * 60)
        recent_readings = [r for r in self.thermal_history if r.timestamp > cutoff_time]

        if len(recent_readings) < 2:
            return {}

        # Calculate temperature trends
        cpu_temps = [r.cpu_temp for r in recent_readings if r.cpu_temp > 0]
        gpu_temps = [r.gpu_temp for r in recent_readings if r.gpu_temp > 0]

        return {
            "cpu_trend": self._calculate_trend(cpu_temps),
            "gpu_trend": self._calculate_trend(gpu_temps),
            "max_cpu_temp": max(cpu_temps) if cpu_temps else 0,
            "max_gpu_temp": max(gpu_temps) if gpu_temps else 0,
            "avg_cpu_temp": sum(cpu_temps) / len(cpu_temps) if cpu_temps else 0,
            "avg_gpu_temp": sum(gpu_temps) / len(gpu_temps) if gpu_temps else 0,
            "readings_count": len(recent_readings),
        }

    def _calculate_trend(self, temperatures: list[float]) -> float:
        """Calculate temperature trend (°C per minute)"""
        if len(temperatures) < 2:
            return 0.0

        # Simple linear regression
        n = len(temperatures)
        x = list(range(n))
        y = temperatures

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        return slope * 60  # Convert to per minute

    def predict_thermal_state(self, minutes_ahead: int = 5) -> ThermalState | None:
        """Predict thermal state based on current trends"""
        trend = self.get_thermal_trend(minutes=5)
        current_reading = self.get_current_reading()

        if not current_reading or not trend:
            return None

        # Predict temperature based on trend
        cpu_predicted = current_reading.cpu_temp + (trend["cpu_trend"] * minutes_ahead)
        gpu_predicted = current_reading.gpu_temp + (trend["gpu_trend"] * minutes_ahead)

        max_predicted = max(cpu_predicted, gpu_predicted)
        return self._determine_thermal_state(max_predicted)

    def get_thermal_report(self) -> dict[str, Any]:
        """Generate comprehensive thermal report"""
        current_reading = self.get_current_reading()
        trend = self.get_thermal_trend(minutes=10)
        predicted_state = self.predict_thermal_state(minutes_ahead=5)

        if not current_reading:
            return {"error": "No thermal data available"}

        # Calculate thermal statistics
        recent_readings = list(self.thermal_history)[-60:]  # Last minute
        if recent_readings:
            temps = [
                max(r.cpu_temp, r.gpu_temp, r.p_core_temp, r.e_core_temp)
                for r in recent_readings
            ]
            thermal_stats = {
                "avg_temp": sum(temps) / len(temps),
                "max_temp": max(temps),
                "min_temp": min(temps),
                "temp_range": max(temps) - min(temps),
            }
        else:
            thermal_stats = {}

        return {
            "current_reading": {
                "timestamp": current_reading.timestamp,
                "cpu_temp": current_reading.cpu_temp,
                "gpu_temp": current_reading.gpu_temp,
                "p_core_temp": current_reading.p_core_temp,
                "e_core_temp": current_reading.e_core_temp,
                "thermal_state": current_reading.thermal_state.value,
                "performance_level": current_reading.performance_level.value,
                "throttling_active": current_reading.throttling_active,
            },
            "thermal_trend": trend,
            "predicted_state": predicted_state.value if predicted_state else None,
            "thermal_statistics": thermal_stats,
            "current_profile": {
                "max_cpu_workers": self.current_profile.max_cpu_workers,
                "max_p_core_workers": self.current_profile.max_p_core_workers,
                "max_gpu_workers": self.current_profile.max_gpu_workers,
                "memory_limit_gb": self.current_profile.memory_limit_gb,
                "batch_size_multiplier": self.current_profile.batch_size_multiplier,
            },
            "recommendations": self._generate_recommendations(current_reading, trend),
        }

    def _generate_recommendations(
        self, reading: ThermalReading, trend: dict[str, float]
    ) -> list[str]:
        """Generate thermal management recommendations"""
        recommendations = []

        if reading.thermal_state == ThermalState.CRITICAL:
            recommendations.extend(
                [
                    "🚨 IMMEDIATE ACTION REQUIRED: Reduce workload immediately",
                    "🔧 Stop non-essential trading processes",
                    "🌪️ Increase fan speed to maximum",
                    "❄️ Consider external cooling solution",
                ]
            )
        elif reading.thermal_state == ThermalState.HOT:
            recommendations.extend(
                [
                    "🔥 Reduce parallel processing",
                    "⏸️ Pause heavy computations",
                    "🌪️ Increase fan speed",
                    "📊 Monitor temperature closely",
                ]
            )
        elif reading.thermal_state == ThermalState.WARM:
            if trend.get("cpu_trend", 0) > 2:  # Rising fast
                recommendations.extend(
                    [
                        "📈 Temperature rising rapidly",
                        "🔧 Consider reducing batch sizes",
                        "⏱️ Increase polling intervals",
                    ]
                )

        if reading.throttling_active:
            recommendations.append(
                "⚠️ System throttling detected - performance is being limited"
            )

        # Performance optimization recommendations
        if reading.performance_level != PerformanceLevel.MAXIMUM:
            recommendations.append(
                f"🎯 Performance limited to {reading.performance_level.value} level"
            )

        return recommendations


# Global thermal monitor instance
_thermal_monitor = None


def get_thermal_monitor() -> ThermalMonitor:
    """Get global thermal monitor instance"""
    global _thermal_monitor
    if _thermal_monitor is None:
        _thermal_monitor = ThermalMonitor()
    return _thermal_monitor


class ThermalOptimizer:
    """Thermal-aware performance optimizer for trading operations"""

    def __init__(self, thermal_monitor: ThermalMonitor):
        self.thermal_monitor = thermal_monitor
        self.optimization_history = deque(maxlen=100)

        # Register callbacks
        thermal_monitor.register_thermal_callback(self._on_thermal_change)
        thermal_monitor.register_performance_callback(self._on_performance_change)

    async def _on_thermal_change(
        self, current: ThermalReading, previous: ThermalReading
    ):
        """Handle thermal state changes"""
        if current.thermal_state.value != previous.thermal_state.value:
            await self._optimize_for_thermal_state(current.thermal_state)

    async def _on_performance_change(
        self, profile: PerformanceProfile, reading: ThermalReading
    ):
        """Handle performance profile changes"""
        await self._apply_performance_profile(profile, reading)

    async def _optimize_for_thermal_state(self, thermal_state: ThermalState):
        """Optimize system for specific thermal state"""
        optimizations = []

        if thermal_state == ThermalState.CRITICAL:
            optimizations.extend(
                [
                    "Stopping non-essential processes",
                    "Reducing to minimal performance mode",
                    "Clearing memory caches",
                    "Pausing background tasks",
                ]
            )
        elif thermal_state == ThermalState.HOT:
            optimizations.extend(
                [
                    "Reducing parallel workers",
                    "Increasing task intervals",
                    "Limiting GPU usage",
                    "Reducing batch sizes",
                ]
            )
        elif thermal_state == ThermalState.WARM:
            optimizations.extend(
                [
                    "Slightly reducing parallelism",
                    "Optimizing memory usage",
                    "Balancing CPU/GPU loads",
                ]
            )

        for optimization in optimizations:
            logger.info(f"🎯 Thermal optimization: {optimization}")

        self.optimization_history.append(
            {
                "timestamp": time.time(),
                "thermal_state": thermal_state.value,
                "optimizations": optimizations,
            }
        )

    async def _apply_performance_profile(
        self, profile: PerformanceProfile, reading: ThermalReading
    ):
        """Apply performance profile to system"""
        logger.info(
            f"🔧 Applying performance profile for {reading.thermal_state.value} state"
        )
        logger.info(f"   Max CPU workers: {profile.max_cpu_workers}")
        logger.info(f"   Max P-core workers: {profile.max_p_core_workers}")
        logger.info(f"   Max GPU workers: {profile.max_gpu_workers}")
        logger.info(f"   Memory limit: {profile.memory_limit_gb}GB")
        logger.info(f"   Batch size multiplier: {profile.batch_size_multiplier}")

    def get_optimization_recommendations(self) -> list[str]:
        """Get current optimization recommendations"""
        current_reading = self.thermal_monitor.get_current_reading()
        if not current_reading:
            return ["No thermal data available"]

        recommendations = []

        # System-level recommendations
        if current_reading.thermal_state == ThermalState.CRITICAL:
            recommendations.extend(
                [
                    "🚨 Emergency thermal management active",
                    "🔴 Consider shutting down non-essential services",
                    "❄️ Implement aggressive cooling measures",
                    "⏸️ Pause trading operations if necessary",
                ]
            )
        elif current_reading.thermal_state == ThermalState.HOT:
            recommendations.extend(
                [
                    "🔥 Implement thermal throttling",
                    "🔧 Reduce computational load",
                    "📊 Monitor thermal trends closely",
                    "⚡ Optimize power-hungry operations",
                ]
            )

        # Performance recommendations
        profile = self.thermal_monitor.current_profile
        recommendations.extend(
            [
                f"🎯 Optimal worker count: {profile.max_cpu_workers} CPU, {profile.max_gpu_workers} GPU",
                f"💾 Memory limit: {profile.memory_limit_gb}GB",
                f"📦 Batch size: {profile.batch_size_multiplier}x standard",
            ]
        )

        return recommendations


if __name__ == "__main__":

    async def demo():
        """Demonstrate thermal monitoring system"""
        print("🌡️ M4 Pro Thermal Monitor Demo")
        print("=" * 50)

        monitor = get_thermal_monitor()
        ThermalOptimizer(monitor)

        # Start monitoring
        await monitor.start_monitoring(interval=2.0)

        try:
            # Monitor for 30 seconds
            for i in range(15):
                reading = monitor.get_current_reading()
                if reading:
                    print(f"\n📊 Sample {i+1}:")
                    print(f"  CPU: {reading.cpu_temp:.1f}°C")
                    print(f"  GPU: {reading.gpu_temp:.1f}°C")
                    print(f"  P-cores: {reading.p_core_temp:.1f}°C")
                    print(f"  E-cores: {reading.e_core_temp:.1f}°C")
                    print(f"  State: {reading.thermal_state.value}")
                    print(f"  Performance: {reading.performance_level.value}")
                    if reading.throttling_active:
                        print("  ⚠️ Throttling active")

                # Show trend every 5 samples
                if i % 5 == 4:
                    trend = monitor.get_thermal_trend()
                    if trend:
                        print("\n📈 Thermal Trend (5 min):")
                        print(f"  CPU trend: {trend['cpu_trend']:+.1f}°C/min")
                        print(f"  GPU trend: {trend['gpu_trend']:+.1f}°C/min")
                        print(
                            f"  Max temps: CPU {trend['max_cpu_temp']:.1f}°C, GPU {trend['max_gpu_temp']:.1f}°C"
                        )

                await asyncio.sleep(2)

            # Generate final report
            print("\n📋 Final Thermal Report:")
            report = monitor.get_thermal_report()
            print(json.dumps(report, indent=2))

        finally:
            await monitor.stop_monitoring()

    asyncio.run(demo())
