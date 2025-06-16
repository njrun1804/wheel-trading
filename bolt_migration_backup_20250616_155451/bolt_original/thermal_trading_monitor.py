"""Thermal management specifically optimized for trading workloads on M4 Pro.

This module provides intelligent thermal monitoring and adaptive performance scaling
to maintain sustained high performance during trading hours while preventing
thermal throttling that could impact execution timing.
"""

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .hardware.hardware_state import get_hardware_state
from .thermal_monitor import ThermalMonitor

logger = logging.getLogger(__name__)


class TradingPerformanceMode(Enum):
    """Performance modes for trading operations."""

    MAXIMUM = "maximum"  # Full performance, accept thermal risk
    BALANCED = "balanced"  # Balanced performance and thermals
    CONSERVATIVE = "conservative"  # Thermal-safe, reduced performance
    EMERGENCY = "emergency"  # Minimal operations only


@dataclass
class ThermalThresholds:
    """Temperature thresholds for different trading scenarios."""

    # CPU temperature thresholds (Celsius)
    cpu_normal_max: float = 75  # Normal operations
    cpu_throttle_start: float = 80  # Begin throttling
    cpu_emergency: float = 90  # Emergency shutdown

    # GPU temperature thresholds (Celsius)
    gpu_normal_max: float = 70  # Normal GPU operations
    gpu_throttle_start: float = 75  # Begin GPU throttling
    gpu_emergency: float = 85  # Emergency GPU shutdown

    # Market hours vs. off-hours thresholds
    market_hours_buffer: float = 5  # Extra conservative during market

    # Sustained load thresholds (for longer operations)
    sustained_cpu_max: float = 70  # For >5 minute operations
    sustained_gpu_max: float = 65  # For >5 minute GPU operations


@dataclass
class ThermalState:
    """Current thermal state of the system."""

    cpu_temperature: float = 0.0
    gpu_temperature: float = 0.0
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    mode: TradingPerformanceMode = TradingPerformanceMode.BALANCED
    throttling_active: bool = False
    minutes_until_throttle: int | None = None


@dataclass
class TradingThermalConfig:
    """Configuration for trading thermal management."""

    enable_predictive_throttling: bool = True
    enable_market_hours_mode: bool = True
    enable_adaptive_batch_sizing: bool = True
    enable_gpu_throttling: bool = True

    # Market hours (UTC) - adjust for your timezone
    market_open_hour: int = 14  # 9:30 AM EST = 14:30 UTC
    market_open_minute: int = 30
    market_close_hour: int = 21  # 4:00 PM EST = 21:00 UTC
    market_close_minute: int = 0

    # Thermal history tracking
    temperature_history_minutes: int = 30
    thermal_trend_samples: int = 10


class TradingThermalManager:
    """Thermal management system optimized for sustained trading performance."""

    def __init__(self, config: TradingThermalConfig | None = None):
        """Initialize trading thermal manager.

        Args:
            config: Thermal management configuration
        """
        self.config = config or TradingThermalConfig()
        self.thresholds = ThermalThresholds()
        self.hw = get_hardware_state()
        self.thermal_monitor = ThermalMonitor()

        # State tracking
        self.current_state = ThermalState()
        self.thermal_history: list[ThermalState] = []
        self.throttle_callbacks: list[
            Callable[[bool, TradingPerformanceMode], None]
        ] = []
        self.mode_change_callbacks: list[Callable[[TradingPerformanceMode], None]] = []

        # Performance adjustments
        self.original_batch_sizes = {}
        self.original_worker_counts = {}

        # Monitoring
        self.monitoring_active = False
        self.last_throttle_event = None

        logger.info("Trading thermal manager initialized")

    async def start_monitoring(self):
        """Start thermal monitoring for trading operations."""
        if self.monitoring_active:
            return

        # Start the underlying thermal monitor
        await self.thermal_monitor.start_monitoring(
            cpu_temp_threshold=self.thresholds.cpu_throttle_start,
            gpu_temp_threshold=self.thresholds.gpu_throttle_start,
            callback=self._thermal_callback,
        )

        # Start trading-specific monitoring loop
        self.monitoring_active = True
        asyncio.create_task(self._trading_monitor_loop())

        logger.info("Trading thermal monitoring started")

    async def stop_monitoring(self):
        """Stop thermal monitoring."""
        self.monitoring_active = False
        await self.thermal_monitor.stop_monitoring()

        # Restore original performance settings
        await self._restore_performance_settings()

        logger.info("Trading thermal monitoring stopped")

    async def _trading_monitor_loop(self):
        """Main monitoring loop for trading thermal management."""
        while self.monitoring_active:
            try:
                # Get current thermal state
                thermal_data = await self.thermal_monitor.get_current_state()
                self._update_thermal_state(thermal_data)

                # Record thermal history
                self._record_thermal_history()

                # Check for predictive throttling
                if self.config.enable_predictive_throttling:
                    await self._check_predictive_throttling()

                # Adjust performance based on market hours
                if self.config.enable_market_hours_mode:
                    self._adjust_for_market_hours()

                # Check for mode changes needed
                new_mode = self._determine_optimal_mode()
                if new_mode != self.current_state.mode:
                    await self._change_performance_mode(new_mode)

                # Sleep until next check
                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error in trading thermal monitor loop: {e}")
                await asyncio.sleep(30)  # Longer sleep on error

    def _thermal_callback(self, thermal_data: dict):
        """Handle thermal events from the underlying monitor."""
        self._update_thermal_state(thermal_data)

        cpu_temp = thermal_data.get("cpu_temperature", 0)
        gpu_temp = thermal_data.get("gpu_temperature", 0)

        # Emergency handling
        if (
            cpu_temp >= self.thresholds.cpu_emergency
            or gpu_temp >= self.thresholds.gpu_emergency
        ):
            asyncio.create_task(self._handle_thermal_emergency())

        # Normal throttling
        elif (
            cpu_temp >= self.thresholds.cpu_throttle_start
            or gpu_temp >= self.thresholds.gpu_throttle_start
        ):
            if not self.current_state.throttling_active:
                asyncio.create_task(self._enter_thermal_throttle())

        # Recovery
        elif (
            cpu_temp <= self.thresholds.cpu_normal_max
            and gpu_temp <= self.thresholds.gpu_normal_max
        ) and self.current_state.throttling_active:
            asyncio.create_task(self._exit_thermal_throttle())

    def _update_thermal_state(self, thermal_data: dict):
        """Update current thermal state."""
        self.current_state.cpu_temperature = thermal_data.get("cpu_temperature", 0)
        self.current_state.gpu_temperature = thermal_data.get("gpu_temperature", 0)
        self.current_state.cpu_utilization = thermal_data.get("cpu_utilization", 0)
        self.current_state.gpu_utilization = thermal_data.get("gpu_utilization", 0)
        self.current_state.timestamp = datetime.now()

    def _record_thermal_history(self):
        """Record thermal state in history."""
        # Add current state to history
        self.thermal_history.append(
            ThermalState(
                cpu_temperature=self.current_state.cpu_temperature,
                gpu_temperature=self.current_state.gpu_temperature,
                cpu_utilization=self.current_state.cpu_utilization,
                gpu_utilization=self.current_state.gpu_utilization,
                timestamp=datetime.now(),
                mode=self.current_state.mode,
                throttling_active=self.current_state.throttling_active,
            )
        )

        # Trim history to configured length
        cutoff_time = datetime.now() - timedelta(
            minutes=self.config.temperature_history_minutes
        )
        self.thermal_history = [
            state for state in self.thermal_history if state.timestamp > cutoff_time
        ]

    async def _check_predictive_throttling(self):
        """Check if predictive throttling should be applied."""
        if len(self.thermal_history) < self.config.thermal_trend_samples:
            return

        # Analyze temperature trend
        recent_states = self.thermal_history[-self.config.thermal_trend_samples :]

        # Calculate temperature trend (degrees per minute)
        time_span = (
            recent_states[-1].timestamp - recent_states[0].timestamp
        ).total_seconds() / 60
        cpu_trend = (
            recent_states[-1].cpu_temperature - recent_states[0].cpu_temperature
        ) / time_span
        gpu_trend = (
            recent_states[-1].gpu_temperature - recent_states[0].gpu_temperature
        ) / time_span

        # Predict time to throttling thresholds
        cpu_minutes_to_throttle = None
        gpu_minutes_to_throttle = None

        if cpu_trend > 0.5:  # Rising at least 0.5°C/min
            cpu_headroom = (
                self.thresholds.cpu_throttle_start - self.current_state.cpu_temperature
            )
            cpu_minutes_to_throttle = max(0, cpu_headroom / cpu_trend)

        if gpu_trend > 0.5:  # Rising at least 0.5°C/min
            gpu_headroom = (
                self.thresholds.gpu_throttle_start - self.current_state.gpu_temperature
            )
            gpu_minutes_to_throttle = max(0, gpu_headroom / gpu_trend)

        # Determine earliest throttling time
        minutes_until_throttle = None
        if cpu_minutes_to_throttle is not None and gpu_minutes_to_throttle is not None:
            minutes_until_throttle = min(
                cpu_minutes_to_throttle, gpu_minutes_to_throttle
            )
        elif cpu_minutes_to_throttle is not None:
            minutes_until_throttle = cpu_minutes_to_throttle
        elif gpu_minutes_to_throttle is not None:
            minutes_until_throttle = gpu_minutes_to_throttle

        self.current_state.minutes_until_throttle = minutes_until_throttle

        # Apply predictive throttling if needed
        if minutes_until_throttle is not None and minutes_until_throttle < 5:
            logger.warning(
                f"Predictive throttling: {minutes_until_throttle:.1f} minutes to thermal limit"
            )
            if not self.current_state.throttling_active:
                await self._enter_thermal_throttle()

    def _adjust_for_market_hours(self):
        """Adjust thermal thresholds based on market hours."""
        if not self.config.enable_market_hours_mode:
            return

        now = datetime.now()
        market_open = now.replace(
            hour=self.config.market_open_hour,
            minute=self.config.market_open_minute,
            second=0,
            microsecond=0,
        )
        market_close = now.replace(
            hour=self.config.market_close_hour,
            minute=self.config.market_close_minute,
            second=0,
            microsecond=0,
        )

        is_market_hours = market_open <= now <= market_close

        if is_market_hours:
            # More conservative thresholds during market hours
            self.thresholds.cpu_throttle_start = 75  # Lower threshold
            self.thresholds.gpu_throttle_start = 70
        else:
            # Normal thresholds outside market hours
            self.thresholds.cpu_throttle_start = 80
            self.thresholds.gpu_throttle_start = 75

    def _determine_optimal_mode(self) -> TradingPerformanceMode:
        """Determine optimal performance mode based on current conditions."""
        cpu_temp = self.current_state.cpu_temperature
        gpu_temp = self.current_state.gpu_temperature

        # Emergency mode
        if (
            cpu_temp >= self.thresholds.cpu_emergency
            or gpu_temp >= self.thresholds.gpu_emergency
        ):
            return TradingPerformanceMode.EMERGENCY

        # Conservative mode
        elif (
            cpu_temp >= self.thresholds.cpu_throttle_start
            or gpu_temp >= self.thresholds.gpu_throttle_start
        ):
            return TradingPerformanceMode.CONSERVATIVE

        # Balanced mode
        elif (
            cpu_temp >= self.thresholds.cpu_normal_max - 5
            or gpu_temp >= self.thresholds.gpu_normal_max - 5
        ):
            return TradingPerformanceMode.BALANCED

        # Maximum mode
        else:
            return TradingPerformanceMode.MAXIMUM

    async def _change_performance_mode(self, new_mode: TradingPerformanceMode):
        """Change performance mode and apply settings."""
        old_mode = self.current_state.mode
        self.current_state.mode = new_mode

        logger.info(f"Performance mode changed: {old_mode.value} -> {new_mode.value}")

        # Apply mode-specific settings
        if new_mode == TradingPerformanceMode.MAXIMUM:
            await self._apply_maximum_performance()
        elif new_mode == TradingPerformanceMode.BALANCED:
            await self._apply_balanced_performance()
        elif new_mode == TradingPerformanceMode.CONSERVATIVE:
            await self._apply_conservative_performance()
        elif new_mode == TradingPerformanceMode.EMERGENCY:
            await self._apply_emergency_performance()

        # Notify callbacks
        for callback in self.mode_change_callbacks:
            try:
                callback(new_mode)
            except Exception as e:
                logger.error(f"Error in mode change callback: {e}")

    async def _apply_maximum_performance(self):
        """Apply maximum performance settings."""
        # Restore original settings
        await self._restore_performance_settings()

        # Set maximum batch sizes
        os.environ["GPU_BATCH_SIZE"] = "4096"
        os.environ["CPU_BATCH_SIZE"] = "2048"
        os.environ["MAX_WORKERS"] = "8"
        os.environ["GREEKS_UPDATE_INTERVAL"] = "5"  # 5 second updates

    async def _apply_balanced_performance(self):
        """Apply balanced performance settings."""
        os.environ["GPU_BATCH_SIZE"] = "2048"
        os.environ["CPU_BATCH_SIZE"] = "1024"
        os.environ["MAX_WORKERS"] = "6"
        os.environ["GREEKS_UPDATE_INTERVAL"] = "10"  # 10 second updates

    async def _apply_conservative_performance(self):
        """Apply conservative performance settings."""
        os.environ["GPU_BATCH_SIZE"] = "1024"
        os.environ["CPU_BATCH_SIZE"] = "512"
        os.environ["MAX_WORKERS"] = "4"
        os.environ["GREEKS_UPDATE_INTERVAL"] = "20"  # 20 second updates

        # Reduce precision for calculations
        os.environ["OPTIONS_PRECISION"] = "reduced"

    async def _apply_emergency_performance(self):
        """Apply emergency performance settings."""
        os.environ["GPU_BATCH_SIZE"] = "256"
        os.environ["CPU_BATCH_SIZE"] = "128"
        os.environ["MAX_WORKERS"] = "2"
        os.environ["GREEKS_UPDATE_INTERVAL"] = "60"  # 1 minute updates

        # Minimal calculations only
        os.environ["DISABLE_ADVANCED_GREEKS"] = "1"
        os.environ["DISABLE_MONTE_CARLO"] = "1"
        os.environ["OPTIONS_PRECISION"] = "minimal"

    async def _restore_performance_settings(self):
        """Restore original performance settings."""
        settings_to_clear = [
            "GPU_BATCH_SIZE",
            "CPU_BATCH_SIZE",
            "MAX_WORKERS",
            "GREEKS_UPDATE_INTERVAL",
            "OPTIONS_PRECISION",
            "DISABLE_ADVANCED_GREEKS",
            "DISABLE_MONTE_CARLO",
        ]

        for setting in settings_to_clear:
            os.environ.pop(setting, None)

    async def _enter_thermal_throttle(self):
        """Enter thermal throttling mode."""
        if self.current_state.throttling_active:
            return

        self.current_state.throttling_active = True
        self.last_throttle_event = datetime.now()

        logger.warning(
            "Entering thermal throttling",
            extra={
                "cpu_temp": self.current_state.cpu_temperature,
                "gpu_temp": self.current_state.gpu_temperature,
                "mode": self.current_state.mode.value,
            },
        )

        # Notify callbacks
        for callback in self.throttle_callbacks:
            try:
                callback(True, self.current_state.mode)
            except Exception as e:
                logger.error(f"Error in throttle callback: {e}")

    async def _exit_thermal_throttle(self):
        """Exit thermal throttling mode."""
        if not self.current_state.throttling_active:
            return

        self.current_state.throttling_active = False

        logger.info(
            "Exiting thermal throttling",
            extra={
                "cpu_temp": self.current_state.cpu_temperature,
                "gpu_temp": self.current_state.gpu_temperature,
                "throttle_duration_minutes": (
                    (datetime.now() - self.last_throttle_event).total_seconds() / 60
                    if self.last_throttle_event
                    else 0
                ),
            },
        )

        # Notify callbacks
        for callback in self.throttle_callbacks:
            try:
                callback(False, self.current_state.mode)
            except Exception as e:
                logger.error(f"Error in throttle callback: {e}")

    async def _handle_thermal_emergency(self):
        """Handle thermal emergency situation."""
        logger.critical(
            "THERMAL EMERGENCY - Reducing system load immediately",
            extra={
                "cpu_temp": self.current_state.cpu_temperature,
                "gpu_temp": self.current_state.gpu_temperature,
            },
        )

        # Force emergency mode
        await self._change_performance_mode(TradingPerformanceMode.EMERGENCY)

        # Force garbage collection
        import gc

        gc.collect()

        # Notify callbacks of emergency
        for callback in self.throttle_callbacks:
            try:
                callback(True, TradingPerformanceMode.EMERGENCY)
            except Exception as e:
                logger.error(f"Error in emergency callback: {e}")

    def register_throttle_callback(
        self, callback: Callable[[bool, TradingPerformanceMode], None]
    ):
        """Register callback for thermal throttling events.

        Args:
            callback: Function called with (is_throttling, performance_mode)
        """
        self.throttle_callbacks.append(callback)

    def register_mode_change_callback(
        self, callback: Callable[[TradingPerformanceMode], None]
    ):
        """Register callback for performance mode changes.

        Args:
            callback: Function called with new performance mode
        """
        self.mode_change_callbacks.append(callback)

    async def get_thermal_report(self) -> dict:
        """Get comprehensive thermal report for monitoring."""
        return {
            "current_state": {
                "cpu_temperature": self.current_state.cpu_temperature,
                "gpu_temperature": self.current_state.gpu_temperature,
                "cpu_utilization": self.current_state.cpu_utilization,
                "gpu_utilization": self.current_state.gpu_utilization,
                "mode": self.current_state.mode.value,
                "throttling_active": self.current_state.throttling_active,
                "minutes_until_throttle": self.current_state.minutes_until_throttle,
            },
            "thresholds": {
                "cpu_normal_max": self.thresholds.cpu_normal_max,
                "cpu_throttle_start": self.thresholds.cpu_throttle_start,
                "gpu_normal_max": self.thresholds.gpu_normal_max,
                "gpu_throttle_start": self.thresholds.gpu_throttle_start,
            },
            "performance_headroom": {
                "cpu_headroom_degrees": max(
                    0,
                    self.thresholds.cpu_throttle_start
                    - self.current_state.cpu_temperature,
                ),
                "gpu_headroom_degrees": max(
                    0,
                    self.thresholds.gpu_throttle_start
                    - self.current_state.gpu_temperature,
                ),
                "estimated_sustained_minutes": self._estimate_sustained_performance_time(),
            },
            "thermal_history_minutes": len(self.thermal_history),
            "last_throttle_event": self.last_throttle_event.isoformat()
            if self.last_throttle_event
            else None,
        }

    def _estimate_sustained_performance_time(self) -> int:
        """Estimate minutes of sustained performance available."""
        if len(self.thermal_history) < 3:
            return 60  # Default estimate

        # Calculate temperature trends
        recent_states = (
            self.thermal_history[-5:]
            if len(self.thermal_history) >= 5
            else self.thermal_history
        )
        time_span = (
            recent_states[-1].timestamp - recent_states[0].timestamp
        ).total_seconds() / 60

        if time_span == 0:
            return 60

        cpu_trend = (
            recent_states[-1].cpu_temperature - recent_states[0].cpu_temperature
        ) / time_span
        gpu_trend = (
            recent_states[-1].gpu_temperature - recent_states[0].gpu_temperature
        ) / time_span

        # Calculate time to threshold for each component
        cpu_headroom = (
            self.thresholds.cpu_throttle_start - self.current_state.cpu_temperature
        )
        gpu_headroom = (
            self.thresholds.gpu_throttle_start - self.current_state.gpu_temperature
        )

        cpu_time = cpu_headroom / max(cpu_trend, 0.1) if cpu_trend > 0 else 120
        gpu_time = gpu_headroom / max(gpu_trend, 0.1) if gpu_trend > 0 else 120

        return max(0, min(int(min(cpu_time, gpu_time)), 120))  # Cap at 2 hours


# Example usage
async def example_thermal_trading_usage():
    """Example of thermal management for trading."""

    # Initialize thermal manager
    config = TradingThermalConfig(
        enable_predictive_throttling=True,
        enable_market_hours_mode=True,
        market_open_hour=14,  # 9:30 AM EST
        market_close_hour=21,  # 4:00 PM EST
    )

    thermal_manager = TradingThermalManager(config)

    # Register callbacks
    def handle_throttling(is_throttling: bool, mode: TradingPerformanceMode):
        if is_throttling:
            print(f"THROTTLING ACTIVATED - Mode: {mode.value}")
        else:
            print("Throttling deactivated")

    def handle_mode_change(mode: TradingPerformanceMode):
        print(f"Performance mode changed to: {mode.value}")

    thermal_manager.register_throttle_callback(handle_throttling)
    thermal_manager.register_mode_change_callback(handle_mode_change)

    try:
        # Start monitoring
        await thermal_manager.start_monitoring()

        # Simulate trading operations
        for i in range(60):  # Run for 10 minutes
            # Get thermal report
            report = await thermal_manager.get_thermal_report()

            print(
                f"Minute {i}: CPU {report['current_state']['cpu_temperature']:.1f}°C, "
                f"GPU {report['current_state']['gpu_temperature']:.1f}°C, "
                f"Mode: {report['current_state']['mode']}"
            )

            if report["current_state"]["minutes_until_throttle"]:
                print(
                    f"  -> Throttling predicted in {report['current_state']['minutes_until_throttle']:.1f} minutes"
                )

            await asyncio.sleep(10)  # 10 second intervals

    finally:
        await thermal_manager.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(example_thermal_trading_usage())
