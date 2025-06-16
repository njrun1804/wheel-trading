"""
Memory Pressure Monitor - Unified pressure detection and response.

Consolidates 5 different pressure monitoring implementations.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import psutil

logger = logging.getLogger(__name__)


class MemoryPressure(Enum):
    """Memory pressure levels."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PressureThresholds:
    """Memory pressure thresholds."""

    moderate: float = 0.70  # 70% usage
    high: float = 0.85  # 85% usage
    critical: float = 0.95  # 95% usage


class PressureMonitor:
    """
    Monitor system memory pressure and trigger responses.
    """

    def __init__(self, thresholds: PressureThresholds | None = None):
        self._lock = threading.Lock()
        self.thresholds = thresholds or PressureThresholds()
        self._callbacks: list[Callable[[MemoryPressure], None]] = []

        self._monitoring = False
        self._monitor_thread = None
        self._interval = 5.0  # seconds

    @property
    def current_pressure(self) -> MemoryPressure:
        """Get current memory pressure level."""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent / 100.0

        if usage_percent >= self.thresholds.critical:
            return MemoryPressure.CRITICAL
        elif usage_percent >= self.thresholds.high:
            return MemoryPressure.HIGH
        elif usage_percent >= self.thresholds.moderate:
            return MemoryPressure.MODERATE
        else:
            return MemoryPressure.LOW

    def add_callback(self, callback: Callable[[MemoryPressure], None]):
        """Add pressure change callback."""
        with self._lock:
            self._callbacks.append(callback)

    def start_monitoring(self):
        """Start background pressure monitoring."""
        with self._lock:
            if not self._monitoring:
                self._monitoring = True
                self._monitor_thread = threading.Thread(
                    target=self._monitor_loop, daemon=True
                )
                self._monitor_thread.start()
                logger.info("Memory pressure monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring."""
        with self._lock:
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=1.0)
                logger.info("Memory pressure monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        last_pressure = MemoryPressure.LOW

        while self._monitoring:
            try:
                current = self.current_pressure

                if current != last_pressure:
                    self._notify_callbacks(current)
                    last_pressure = current

                time.sleep(self._interval)

            except Exception as e:
                logger.error(f"Error in pressure monitoring: {e}")
                time.sleep(self._interval)

    def _notify_callbacks(self, pressure: MemoryPressure):
        """Notify all callbacks of pressure change."""
        with self._lock:
            for callback in self._callbacks:
                try:
                    callback(pressure)
                except Exception as e:
                    logger.error(f"Error in pressure callback: {e}")
