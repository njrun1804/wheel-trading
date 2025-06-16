#!/usr/bin/env python3
"""
Unified Memory Pressure Handler for Bolt System - M4 Pro Optimization
Orchestrates memory pressure across all components to maintain <4GB total usage

Features:
1. Cross-component memory pressure coordination
2. Intelligent cache eviction strategies
3. Dynamic memory allocation adjustment
4. Emergency memory cleanup triggers
5. Real-time memory monitoring and alerts
6. Predictive memory pressure detection
"""

import asyncio
import contextlib
import gc
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import psutil

from .database_memory_optimizer import get_database_memory_manager
from .gpu_memory_optimizer import get_gpu_memory_manager

# Import our optimized memory managers
from .optimized_memory_manager import get_optimized_memory_manager

logger = logging.getLogger(__name__)


class MemoryPressureLevel(Enum):
    """Memory pressure levels."""

    LOW = "low"  # < 70% of target
    MODERATE = "moderate"  # 70-85% of target
    HIGH = "high"  # 85-95% of target
    CRITICAL = "critical"  # > 95% of target


@dataclass
class MemoryPressureEvent:
    """Memory pressure event."""

    timestamp: float
    level: MemoryPressureLevel
    total_memory_mb: float
    target_memory_mb: float
    component_usage: dict[str, float]
    action_taken: str
    memory_freed_mb: float
    duration_ms: float


@dataclass
class ComponentMemoryProfile:
    """Memory profile for a component."""

    name: str
    current_mb: float
    peak_mb: float
    budget_mb: float
    pressure_threshold_mb: float
    cleanup_callback: Callable[[bool], float] | None
    priority: int  # 1-10, higher = more important


class MemoryPressurePredictor:
    """Predicts memory pressure based on usage patterns."""

    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self._memory_history: deque = deque(maxlen=history_size)
        self._lock = threading.Lock()

    def record_usage(self, memory_mb: float):
        """Record memory usage measurement."""
        with self._lock:
            self._memory_history.append((time.time(), memory_mb))

    def predict_pressure_in_minutes(
        self, minutes: int = 5
    ) -> tuple[float, MemoryPressureLevel]:
        """Predict memory usage and pressure level in N minutes."""
        with self._lock:
            if len(self._memory_history) < 10:
                return 0.0, MemoryPressureLevel.LOW

            # Calculate growth rate
            recent_history = list(self._memory_history)[-10:]
            time_span = recent_history[-1][0] - recent_history[0][0]
            memory_growth = recent_history[-1][1] - recent_history[0][1]

            if time_span <= 0:
                return recent_history[-1][1], MemoryPressureLevel.LOW

            # Growth rate in MB per second
            growth_rate = memory_growth / time_span

            # Predict usage
            current_usage = recent_history[-1][1]
            predicted_usage = current_usage + (growth_rate * minutes * 60)

            # Determine pressure level
            if predicted_usage > 3840:  # 95% of 4GB
                level = MemoryPressureLevel.CRITICAL
            elif predicted_usage > 3400:  # 85% of 4GB
                level = MemoryPressureLevel.HIGH
            elif predicted_usage > 2800:  # 70% of 4GB
                level = MemoryPressureLevel.MODERATE
            else:
                level = MemoryPressureLevel.LOW

            return predicted_usage, level

    def get_memory_trend(self) -> str:
        """Get memory usage trend description."""
        with self._lock:
            if len(self._memory_history) < 5:
                return "insufficient_data"

            recent = list(self._memory_history)[-5:]
            memory_values = [m[1] for m in recent]

            # Simple trend analysis
            if memory_values[-1] > memory_values[0] * 1.1:
                return "increasing"
            elif memory_values[-1] < memory_values[0] * 0.9:
                return "decreasing"
            else:
                return "stable"


class UnifiedMemoryPressureHandler:
    """Unified memory pressure handler for all bolt components."""

    def __init__(self, target_memory_mb: float = 4096):
        self.target_memory_mb = target_memory_mb

        # Pressure thresholds
        self.pressure_thresholds = {
            MemoryPressureLevel.LOW: target_memory_mb * 0.70,  # 2.8GB
            MemoryPressureLevel.MODERATE: target_memory_mb * 0.85,  # 3.4GB
            MemoryPressureLevel.HIGH: target_memory_mb * 0.95,  # 3.8GB
            MemoryPressureLevel.CRITICAL: target_memory_mb * 1.0,  # 4.0GB
        }

        # Component managers
        self.memory_manager = get_optimized_memory_manager()
        self.gpu_manager = get_gpu_memory_manager()
        self.database_manager = get_database_memory_manager()

        # Component profiles
        self.component_profiles: dict[str, ComponentMemoryProfile] = {}
        self._register_components()

        # Pressure prediction
        self.predictor = MemoryPressurePredictor()

        # Event tracking
        self.pressure_events: list[MemoryPressureEvent] = []
        self.max_events = 100

        # Monitoring
        self._monitoring_active = False
        self._monitor_task = None
        self._monitoring_interval = 5.0  # Check every 5 seconds

        # Locks
        self._main_lock = threading.RLock()
        self._cleanup_lock = threading.Lock()

        logger.info(
            f"Unified Memory Pressure Handler initialized (target: {target_memory_mb:.0f}MB)"
        )

    def _register_components(self):
        """Register all components with their memory profiles."""
        # Optimized memory manager components
        self.component_profiles["agents"] = ComponentMemoryProfile(
            name="agents",
            current_mb=0,
            peak_mb=0,
            budget_mb=1200,  # 1.2GB
            pressure_threshold_mb=1000,
            cleanup_callback=lambda force: self.memory_manager.cleanup_component(
                "agents", force
            ),
            priority=7,
        )

        self.component_profiles["einstein"] = ComponentMemoryProfile(
            name="einstein",
            current_mb=0,
            peak_mb=0,
            budget_mb=800,  # 0.8GB
            pressure_threshold_mb=600,
            cleanup_callback=lambda force: self.memory_manager.cleanup_component(
                "einstein", force
            ),
            priority=6,
        )

        self.component_profiles["database"] = ComponentMemoryProfile(
            name="database",
            current_mb=0,
            peak_mb=0,
            budget_mb=1000,  # 1.0GB
            pressure_threshold_mb=800,
            cleanup_callback=lambda force: self.database_manager.cleanup_connections(
                force
            ),
            priority=8,
        )

        self.component_profiles["gpu"] = ComponentMemoryProfile(
            name="gpu",
            current_mb=0,
            peak_mb=0,
            budget_mb=600,  # 0.6GB
            pressure_threshold_mb=480,
            cleanup_callback=lambda force: self.gpu_manager.cleanup_memory(force),
            priority=5,
        )

        self.component_profiles["cache"] = ComponentMemoryProfile(
            name="cache",
            current_mb=0,
            peak_mb=0,
            budget_mb=320,  # 0.32GB
            pressure_threshold_mb=250,
            cleanup_callback=lambda force: self.memory_manager.cleanup_component(
                "cache", force
            ),
            priority=3,
        )

        self.component_profiles["other"] = ComponentMemoryProfile(
            name="other",
            current_mb=0,
            peak_mb=0,
            budget_mb=180,  # 0.18GB
            pressure_threshold_mb=150,
            cleanup_callback=lambda force: self.memory_manager.cleanup_component(
                "other", force
            ),
            priority=2,
        )

    def get_current_memory_usage(self) -> float:
        """Get current total memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def get_memory_pressure_level(
        self, memory_mb: float | None = None
    ) -> MemoryPressureLevel:
        """Get current memory pressure level."""
        if memory_mb is None:
            memory_mb = self.get_current_memory_usage()

        for level in [
            MemoryPressureLevel.CRITICAL,
            MemoryPressureLevel.HIGH,
            MemoryPressureLevel.MODERATE,
            MemoryPressureLevel.LOW,
        ]:
            if memory_mb >= self.pressure_thresholds[level]:
                return level

        return MemoryPressureLevel.LOW

    def update_component_usage(self):
        """Update component memory usage estimates."""
        # Get component usage from managers
        memory_report = self.memory_manager.get_memory_report()
        gpu_report = self.gpu_manager.get_memory_report()
        db_report = self.database_manager.get_memory_report()

        # Update component profiles
        for component_name, component_data in memory_report.get(
            "components", {}
        ).items():
            if component_name in self.component_profiles:
                profile = self.component_profiles[component_name]
                profile.current_mb = component_data.get("allocated_mb", 0)
                profile.peak_mb = max(profile.peak_mb, profile.current_mb)

        # Update GPU usage
        if "gpu" in self.component_profiles:
            profile = self.component_profiles["gpu"]
            profile.current_mb = gpu_report["memory_stats"]["allocated_mb"]
            profile.peak_mb = max(profile.peak_mb, profile.current_mb)

        # Update database usage
        if "database" in self.component_profiles:
            profile = self.component_profiles["database"]
            profile.current_mb = db_report["estimated_usage_mb"]
            profile.peak_mb = max(profile.peak_mb, profile.current_mb)

    def handle_memory_pressure(
        self, level: MemoryPressureLevel, force: bool = False
    ) -> float:
        """Handle memory pressure at the specified level."""
        start_time = time.time()
        initial_memory = self.get_current_memory_usage()

        logger.warning(
            f"Handling memory pressure: {level.value} ({initial_memory:.1f}MB)"
        )

        total_freed = 0.0
        actions_taken = []

        with self._cleanup_lock:
            # Update component usage
            self.update_component_usage()

            # Determine cleanup strategy based on pressure level
            if level == MemoryPressureLevel.CRITICAL:
                # Emergency cleanup - all components
                actions_taken.append("emergency_cleanup")
                for profile in sorted(
                    self.component_profiles.values(), key=lambda x: x.priority
                ):
                    if profile.cleanup_callback:
                        try:
                            freed = profile.cleanup_callback(True)
                            total_freed += freed
                            logger.info(
                                f"Emergency cleanup {profile.name}: freed {freed:.1f}MB"
                            )
                        except Exception as e:
                            logger.error(
                                f"Emergency cleanup failed for {profile.name}: {e}"
                            )

                # Force garbage collection
                collected = gc.collect()
                logger.info(f"Emergency GC collected {collected} objects")

            elif level == MemoryPressureLevel.HIGH:
                # Aggressive cleanup - prioritize low-priority components
                actions_taken.append("aggressive_cleanup")
                cleanup_priority = sorted(
                    self.component_profiles.values(), key=lambda x: x.priority
                )

                for profile in cleanup_priority:
                    if profile.current_mb > profile.pressure_threshold_mb:
                        if profile.cleanup_callback:
                            try:
                                freed = profile.cleanup_callback(True)
                                total_freed += freed
                                logger.info(
                                    f"Aggressive cleanup {profile.name}: freed {freed:.1f}MB"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Aggressive cleanup failed for {profile.name}: {e}"
                                )

            elif level == MemoryPressureLevel.MODERATE:
                # Selective cleanup - components over threshold
                actions_taken.append("selective_cleanup")
                for profile in self.component_profiles.values():
                    if profile.current_mb > profile.pressure_threshold_mb:
                        if profile.cleanup_callback:
                            try:
                                freed = profile.cleanup_callback(False)
                                total_freed += freed
                                logger.info(
                                    f"Selective cleanup {profile.name}: freed {freed:.1f}MB"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Selective cleanup failed for {profile.name}: {e}"
                                )

            # Cache-specific cleanup
            if level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
                actions_taken.append("cache_cleanup")
                try:
                    freed = self.memory_manager.cleanup_component("cache", True)
                    total_freed += freed
                    logger.info(f"Cache cleanup: freed {freed:.1f}MB")
                except Exception as e:
                    logger.error(f"Cache cleanup failed: {e}")

        # Final memory check
        final_memory = self.get_current_memory_usage()
        actual_freed = initial_memory - final_memory
        duration_ms = (time.time() - start_time) * 1000

        # Record pressure event
        event = MemoryPressureEvent(
            timestamp=start_time,
            level=level,
            total_memory_mb=initial_memory,
            target_memory_mb=self.target_memory_mb,
            component_usage={
                name: profile.current_mb
                for name, profile in self.component_profiles.items()
            },
            action_taken=", ".join(actions_taken),
            memory_freed_mb=actual_freed,
            duration_ms=duration_ms,
        )

        self.pressure_events.append(event)
        if len(self.pressure_events) > self.max_events:
            self.pressure_events = self.pressure_events[-self.max_events :]

        logger.info(
            f"Memory pressure handled: {initial_memory:.1f}MB -> {final_memory:.1f}MB "
            f"(freed {actual_freed:.1f}MB) in {duration_ms:.1f}ms"
        )

        return actual_freed

    def predict_and_handle_pressure(self):
        """Predict future memory pressure and handle proactively."""
        # Predict memory usage in 2 minutes
        predicted_usage, predicted_level = self.predictor.predict_pressure_in_minutes(2)

        # If we predict moderate or higher pressure, act now
        if predicted_level in [MemoryPressureLevel.MODERATE, MemoryPressureLevel.HIGH]:
            current_level = self.get_memory_pressure_level()

            if predicted_level.value != current_level.value:
                logger.info(
                    f"Predicted memory pressure: {predicted_level.value} in 2 minutes "
                    f"(current: {current_level.value})"
                )

                # Preemptive cleanup at lower level
                preemptive_level = (
                    MemoryPressureLevel.MODERATE
                    if predicted_level == MemoryPressureLevel.HIGH
                    else MemoryPressureLevel.LOW
                )
                self.handle_memory_pressure(preemptive_level)

    async def start_monitoring(self):
        """Start background memory pressure monitoring."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started unified memory pressure monitoring")

    async def stop_monitoring(self):
        """Stop background memory pressure monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task

        logger.info("Stopped unified memory pressure monitoring")

    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                current_memory = self.get_current_memory_usage()
                current_level = self.get_memory_pressure_level(current_memory)

                # Record usage for prediction
                self.predictor.record_usage(current_memory)

                # Update component usage
                self.update_component_usage()

                # Handle current pressure
                if current_level in [
                    MemoryPressureLevel.MODERATE,
                    MemoryPressureLevel.HIGH,
                    MemoryPressureLevel.CRITICAL,
                ]:
                    self.handle_memory_pressure(current_level)

                # Predictive handling
                self.predict_and_handle_pressure()

                # Periodic logging
                if time.time() % 60 < self._monitoring_interval:  # Log once per minute
                    trend = self.predictor.get_memory_trend()
                    logger.info(
                        f"Memory Status: {current_memory:.1f}MB / {self.target_memory_mb:.1f}MB "
                        f"({current_level.value}, trend: {trend})"
                    )

                await asyncio.sleep(self._monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory pressure monitoring error: {e}")
                await asyncio.sleep(self._monitoring_interval)

    def get_pressure_report(self) -> dict[str, Any]:
        """Get comprehensive memory pressure report."""
        current_memory = self.get_current_memory_usage()
        current_level = self.get_memory_pressure_level(current_memory)
        predicted_usage, predicted_level = self.predictor.predict_pressure_in_minutes(5)

        # Update component usage
        self.update_component_usage()

        return {
            "timestamp": time.time(),
            "current_memory_mb": current_memory,
            "target_memory_mb": self.target_memory_mb,
            "usage_percent": (current_memory / self.target_memory_mb) * 100,
            "current_pressure_level": current_level.value,
            "predicted_usage_mb": predicted_usage,
            "predicted_pressure_level": predicted_level.value,
            "memory_trend": self.predictor.get_memory_trend(),
            "pressure_thresholds": {
                level.value: threshold
                for level, threshold in self.pressure_thresholds.items()
            },
            "components": {
                name: {
                    "current_mb": profile.current_mb,
                    "budget_mb": profile.budget_mb,
                    "peak_mb": profile.peak_mb,
                    "usage_percent": (profile.current_mb / profile.budget_mb) * 100,
                    "over_threshold": profile.current_mb
                    > profile.pressure_threshold_mb,
                    "priority": profile.priority,
                }
                for name, profile in self.component_profiles.items()
            },
            "recent_pressure_events": len(
                [e for e in self.pressure_events if time.time() - e.timestamp < 3600]
            ),
            "total_pressure_events": len(self.pressure_events),
            "recommendations": self._get_recommendations(),
        }

    def _get_recommendations(self) -> list[str]:
        """Get memory optimization recommendations."""
        recommendations = []
        current_memory = self.get_current_memory_usage()
        current_level = self.get_memory_pressure_level(current_memory)

        if current_level == MemoryPressureLevel.CRITICAL:
            recommendations.append(
                "CRITICAL: Memory usage exceeds target - immediate cleanup required"
            )
        elif current_level == MemoryPressureLevel.HIGH:
            recommendations.append("HIGH: Memory usage approaching critical levels")
        elif current_level == MemoryPressureLevel.MODERATE:
            recommendations.append("MODERATE: Memory usage elevated - consider cleanup")

        # Check components over threshold
        over_threshold = [
            name
            for name, profile in self.component_profiles.items()
            if profile.current_mb > profile.pressure_threshold_mb
        ]
        if over_threshold:
            recommendations.append(
                f"Components over threshold: {', '.join(over_threshold)}"
            )

        # Check memory trend
        trend = self.predictor.get_memory_trend()
        if trend == "increasing":
            recommendations.append("Memory usage trending upward - monitor closely")

        # Check recent pressure events
        recent_events = len(
            [e for e in self.pressure_events if time.time() - e.timestamp < 3600]
        )
        if recent_events > 5:
            recommendations.append(
                f"High pressure event frequency: {recent_events} events in last hour"
            )

        if not recommendations:
            recommendations.append("Memory usage is within acceptable levels")

        return recommendations

    def force_emergency_cleanup(self) -> float:
        """Force emergency cleanup of all components."""
        logger.critical("Forcing emergency memory cleanup")
        return self.handle_memory_pressure(MemoryPressureLevel.CRITICAL, force=True)

    def shutdown(self):
        """Shutdown the memory pressure handler."""
        if self._monitoring_active:
            asyncio.create_task(self.stop_monitoring())

        # Final cleanup
        self.force_emergency_cleanup()
        logger.info("Unified Memory Pressure Handler shutdown complete")


# Global instance
_pressure_handler: UnifiedMemoryPressureHandler | None = None


def get_memory_pressure_handler() -> UnifiedMemoryPressureHandler:
    """Get the global memory pressure handler."""
    global _pressure_handler
    if _pressure_handler is None:
        _pressure_handler = UnifiedMemoryPressureHandler()
    return _pressure_handler


# Convenience functions
def handle_memory_pressure(level: MemoryPressureLevel = None) -> float:
    """Handle memory pressure."""
    handler = get_memory_pressure_handler()
    if level is None:
        current_memory = handler.get_current_memory_usage()
        level = handler.get_memory_pressure_level(current_memory)
    return handler.handle_memory_pressure(level)


def get_memory_pressure_report() -> dict[str, Any]:
    """Get memory pressure report."""
    return get_memory_pressure_handler().get_pressure_report()


def force_emergency_cleanup() -> float:
    """Force emergency memory cleanup."""
    return get_memory_pressure_handler().force_emergency_cleanup()


async def start_memory_monitoring():
    """Start global memory pressure monitoring."""
    await get_memory_pressure_handler().start_monitoring()


async def stop_memory_monitoring():
    """Stop global memory pressure monitoring."""
    await get_memory_pressure_handler().stop_monitoring()


if __name__ == "__main__":
    # Test the unified memory pressure handler
    print("Testing Unified Memory Pressure Handler...")

    handler = get_memory_pressure_handler()

    # Get initial report
    report = get_memory_pressure_report()
    print("Initial Memory Status:")
    print(
        f"  Usage: {report['current_memory_mb']:.1f}MB / {report['target_memory_mb']:.1f}MB ({report['usage_percent']:.1f}%)"
    )
    print(f"  Pressure Level: {report['current_pressure_level']}")
    print(f"  Trend: {report['memory_trend']}")

    # Test pressure handling
    freed = handle_memory_pressure(MemoryPressureLevel.MODERATE)
    print(f"  Memory freed: {freed:.1f}MB")

    # Component status
    print("\nComponent Status:")
    for name, stats in report["components"].items():
        print(
            f"  {name}: {stats['current_mb']:.1f}MB / {stats['budget_mb']:.1f}MB ({stats['usage_percent']:.1f}%)"
        )

    # Recommendations
    print("\nRecommendations:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")

    print("Test completed successfully!")
