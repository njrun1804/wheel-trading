"""
Memory Pressure Monitor - Real-time system memory monitoring and pressure detection

Provides continuous monitoring of system memory usage with intelligent pressure detection,
trend analysis, and proactive alerts for the trading system memory manager.
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import psutil

logger = logging.getLogger(__name__)


class PressureLevel(Enum):
    """Memory pressure levels"""

    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MemoryReading:
    """Single memory usage reading"""

    timestamp: float
    system_percent: float
    available_gb: float
    used_gb: float
    buffers_gb: float
    cached_gb: float
    swap_used_gb: float = 0.0
    swap_total_gb: float = 0.0


@dataclass
class PressureStats:
    """Pressure monitoring statistics"""

    readings_count: int = 0
    pressure_events: int = 0
    critical_events: int = 0
    emergency_events: int = 0
    last_pressure_time: float = 0
    max_pressure_seen: float = 0
    average_pressure_1min: float = 0
    average_pressure_5min: float = 0
    trend_slope: float = 0  # Pressure trend over last minute


class PressureMonitor:
    """
    Real-time memory pressure monitoring with trend analysis

    Features:
    - Continuous memory usage tracking
    - Pressure level classification
    - Trend analysis for predictive alerts
    - Callback system for pressure events
    - Historical data for analysis
    """

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

        # Configuration
        self.monitor_interval = 2.0  # Check every 2 seconds
        self.history_size = 300  # Keep 10 minutes of history (300 * 2s)

        # Pressure thresholds
        self.thresholds = {
            PressureLevel.LOW: 0.70,  # 70%
            PressureLevel.MEDIUM: 0.80,  # 80%
            PressureLevel.HIGH: 0.85,  # 85%
            PressureLevel.CRITICAL: 0.90,  # 90%
            PressureLevel.EMERGENCY: 0.95,  # 95%
        }

        # History tracking
        self.readings: deque = deque(maxlen=self.history_size)
        self.stats = PressureStats()

        # Thread management
        self.monitor_thread: threading.Thread | None = None
        self.running = False
        self.lock = threading.RLock()

        # Callbacks
        self.pressure_callbacks: dict[PressureLevel, list[Callable]] = {
            level: [] for level in PressureLevel
        }

        # State tracking
        self.current_level = PressureLevel.NORMAL
        self.last_level = PressureLevel.NORMAL
        self.consecutive_readings = 0  # Consecutive readings at current level

        logger.info("PressureMonitor initialized")

    def start(self):
        """Start monitoring thread"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True, name="MemoryPressureMonitor"
            )
            self.monitor_thread.start()
            logger.info("Memory pressure monitoring started")

    def stop(self):
        """Stop monitoring thread"""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Memory pressure monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Take reading
                reading = self._take_reading()
                if reading:
                    self._process_reading(reading)

                time.sleep(self.monitor_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)  # Longer sleep on error

    def _take_reading(self) -> MemoryReading | None:
        """Take a single memory reading"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            return MemoryReading(
                timestamp=time.time(),
                system_percent=memory.percent / 100.0,
                available_gb=memory.available / (1024**3),
                used_gb=memory.used / (1024**3),
                buffers_gb=getattr(memory, "buffers", 0) / (1024**3),
                cached_gb=getattr(memory, "cached", 0) / (1024**3),
                swap_used_gb=swap.used / (1024**3),
                swap_total_gb=swap.total / (1024**3),
            )
        except Exception as e:
            logger.error(f"Failed to take memory reading: {e}")
            return None

    def _process_reading(self, reading: MemoryReading):
        """Process a new memory reading"""
        with self.lock:
            # Store reading
            self.readings.append(reading)
            self.stats.readings_count += 1

            # Update stats
            self._update_stats(reading)

            # Determine pressure level
            new_level = self._classify_pressure(reading.system_percent)

            # Handle level changes
            if new_level != self.current_level:
                self._handle_level_change(new_level, reading)
            else:
                self.consecutive_readings += 1

            # Check for sustained pressure (5+ consecutive readings)
            if self.consecutive_readings >= 5 and self.current_level in [
                PressureLevel.HIGH,
                PressureLevel.CRITICAL,
                PressureLevel.EMERGENCY,
            ]:
                self._handle_sustained_pressure(reading)

    def _update_stats(self, reading: MemoryReading):
        """Update monitoring statistics"""
        pressure = reading.system_percent

        # Track maximum pressure
        if pressure > self.stats.max_pressure_seen:
            self.stats.max_pressure_seen = pressure

        # Calculate moving averages
        if len(self.readings) >= 30:  # 1 minute of data
            recent_1min = [r.system_percent for r in list(self.readings)[-30:]]
            self.stats.average_pressure_1min = np.mean(recent_1min)

            # Calculate trend (slope over last minute)
            times = np.array(range(len(recent_1min)))
            if len(recent_1min) > 5:
                self.stats.trend_slope = np.polyfit(times, recent_1min, 1)[0]

        if len(self.readings) >= 150:  # 5 minutes of data
            recent_5min = [r.system_percent for r in list(self.readings)[-150:]]
            self.stats.average_pressure_5min = np.mean(recent_5min)

    def _classify_pressure(self, system_percent: float) -> PressureLevel:
        """Classify pressure level based on system usage"""
        if system_percent >= self.thresholds[PressureLevel.EMERGENCY]:
            return PressureLevel.EMERGENCY
        elif system_percent >= self.thresholds[PressureLevel.CRITICAL]:
            return PressureLevel.CRITICAL
        elif system_percent >= self.thresholds[PressureLevel.HIGH]:
            return PressureLevel.HIGH
        elif system_percent >= self.thresholds[PressureLevel.MEDIUM]:
            return PressureLevel.MEDIUM
        elif system_percent >= self.thresholds[PressureLevel.LOW]:
            return PressureLevel.LOW
        else:
            return PressureLevel.NORMAL

    def _handle_level_change(self, new_level: PressureLevel, reading: MemoryReading):
        """Handle pressure level change"""
        old_level = self.current_level
        self.last_level = old_level
        self.current_level = new_level
        self.consecutive_readings = 1

        logger.info(
            f"Memory pressure changed: {old_level.value} → {new_level.value} "
            f"({reading.system_percent:.1%})"
        )

        # Update event counters
        if new_level in [PressureLevel.MEDIUM, PressureLevel.HIGH]:
            self.stats.pressure_events += 1
            self.stats.last_pressure_time = reading.timestamp
        elif new_level == PressureLevel.CRITICAL:
            self.stats.critical_events += 1
        elif new_level == PressureLevel.EMERGENCY:
            self.stats.emergency_events += 1

        # Trigger callbacks
        self._trigger_callbacks(new_level, reading)

        # Notify memory manager
        if new_level.value in ["medium", "high", "critical", "emergency"]:
            self.memory_manager.handle_pressure(reading.system_percent)

    def _handle_sustained_pressure(self, reading: MemoryReading):
        """Handle sustained high pressure"""
        logger.warning(
            f"Sustained {self.current_level.value} pressure detected "
            f"({self.consecutive_readings} consecutive readings)"
        )

        # Additional actions for sustained pressure
        if (
            self.current_level == PressureLevel.CRITICAL
            and self.consecutive_readings >= 10
        ):
            # 20 seconds of critical pressure
            logger.critical(
                "Extended critical pressure - triggering aggressive cleanup"
            )
            self.memory_manager._handle_emergency()

    def _trigger_callbacks(self, level: PressureLevel, reading: MemoryReading):
        """Trigger registered callbacks for pressure level"""
        callbacks = self.pressure_callbacks.get(level, [])
        for callback in callbacks:
            try:
                callback(level, reading)
            except Exception as e:
                logger.error(f"Pressure callback error: {e}")

    def register_callback(self, level: PressureLevel, callback: Callable):
        """Register callback for specific pressure level"""
        self.pressure_callbacks[level].append(callback)

    def get_current_reading(self) -> MemoryReading | None:
        """Get the most recent memory reading"""
        with self.lock:
            return self.readings[-1] if self.readings else None

    def get_pressure_level(self) -> float:
        """Get current pressure level as float (0.0-1.0)"""
        reading = self.get_current_reading()
        return reading.system_percent if reading else 0.0

    def get_pressure_classification(self) -> PressureLevel:
        """Get current pressure classification"""
        return self.current_level

    def get_trend_analysis(self) -> dict[str, float]:
        """Get memory usage trend analysis"""
        with self.lock:
            if len(self.readings) < 10:
                return {"trend": "insufficient_data"}

            # Analyze recent trend
            recent_readings = list(self.readings)[-60:]  # Last 2 minutes
            pressures = [r.system_percent for r in recent_readings]
            times = np.array(range(len(pressures)))

            # Calculate trend slope
            slope, intercept = np.polyfit(times, pressures, 1)

            # Classify trend
            if abs(slope) < 0.001:  # Less than 0.1% change per reading
                trend = "stable"
            elif slope > 0.002:  # More than 0.2% increase per reading
                trend = "rising_fast"
            elif slope > 0.001:  # 0.1-0.2% increase per reading
                trend = "rising"
            elif slope < -0.002:  # More than 0.2% decrease per reading
                trend = "falling_fast"
            elif slope < -0.001:  # 0.1-0.2% decrease per reading
                trend = "falling"
            else:
                trend = "stable"

            # Predict future pressure
            future_pressure = intercept + slope * len(pressures) * 30  # 1 minute ahead

            return {
                "trend": trend,
                "slope": slope,
                "current_pressure": pressures[-1],
                "predicted_1min": min(1.0, max(0.0, future_pressure)),
                "variance": np.var(pressures),
                "readings_analyzed": len(pressures),
            }

    def get_stats(self) -> dict[str, any]:
        """Get comprehensive monitoring statistics"""
        with self.lock:
            current_reading = self.get_current_reading()
            trend = self.get_trend_analysis()

            return {
                "current": {
                    "level": self.current_level.value,
                    "pressure": current_reading.system_percent
                    if current_reading
                    else 0,
                    "available_gb": current_reading.available_gb
                    if current_reading
                    else 0,
                    "consecutive_readings": self.consecutive_readings,
                },
                "stats": {
                    "readings_count": self.stats.readings_count,
                    "pressure_events": self.stats.pressure_events,
                    "critical_events": self.stats.critical_events,
                    "emergency_events": self.stats.emergency_events,
                    "max_pressure_seen": self.stats.max_pressure_seen,
                    "avg_pressure_1min": self.stats.average_pressure_1min,
                    "avg_pressure_5min": self.stats.average_pressure_5min,
                },
                "trend": trend,
                "thresholds": {
                    level.value: threshold
                    for level, threshold in self.thresholds.items()
                },
                "history_size": len(self.readings),
            }

    def get_historical_data(self, minutes: int = 5) -> list[MemoryReading]:
        """Get historical readings for specified time period"""
        with self.lock:
            if not self.readings:
                return []

            cutoff_time = time.time() - (minutes * 60)
            return [r for r in self.readings if r.timestamp >= cutoff_time]

    def is_trending_up(self, threshold: float = 0.001) -> bool:
        """Check if memory usage is trending upward"""
        return self.stats.trend_slope > threshold

    def is_pressure_sustained(
        self, level: PressureLevel, min_readings: int = 5
    ) -> bool:
        """Check if pressure at specified level is sustained"""
        return self.current_level == level and self.consecutive_readings >= min_readings

    def predict_pressure_breach(
        self, target_level: PressureLevel, minutes_ahead: int = 5
    ) -> dict[str, any]:
        """Predict if pressure will breach target level within time window"""
        trend = self.get_trend_analysis()

        if trend["trend"] == "insufficient_data":
            return {"prediction": "insufficient_data"}

        current_pressure = trend["current_pressure"]
        slope = trend["slope"]
        target_threshold = self.thresholds[target_level]

        # Calculate readings needed to reach threshold
        readings_to_breach = (
            (target_threshold - current_pressure) / slope if slope > 0 else float("inf")
        )
        time_to_breach = (
            readings_to_breach * self.monitor_interval / 60
        )  # Convert to minutes

        will_breach = (
            slope > 0 and time_to_breach <= minutes_ahead and time_to_breach > 0
        )

        return {
            "prediction": "will_breach" if will_breach else "no_breach",
            "target_level": target_level.value,
            "current_pressure": current_pressure,
            "target_threshold": target_threshold,
            "estimated_time_minutes": time_to_breach if will_breach else None,
            "trend_slope": slope,
            "confidence": min(1.0, abs(slope) * 1000),  # Simple confidence metric
        }


# Global instance
_pressure_monitor: PressureMonitor | None = None


def get_pressure_monitor(memory_manager=None) -> PressureMonitor:
    """Get or create the global pressure monitor"""
    global _pressure_monitor
    if _pressure_monitor is None and memory_manager:
        _pressure_monitor = PressureMonitor(memory_manager)
    return _pressure_monitor
