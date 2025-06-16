"""
Graceful Degradation Manager

Manages graceful degradation of system capabilities when resources are
constrained or components fail, ensuring the system continues to operate
at reduced capacity rather than failing completely.
"""

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DegradationLevel(Enum):
    """Levels of system degradation."""

    NORMAL = "normal"  # Full functionality
    REDUCED = "reduced"  # Some features disabled
    MINIMAL = "minimal"  # Core functionality only
    EMERGENCY = "emergency"  # Survival mode only


class CapabilityType(Enum):
    """Types of system capabilities that can be degraded."""

    AGENT_CONCURRENCY = "agent_concurrency"
    GPU_ACCELERATION = "gpu_acceleration"
    MEMORY_INTENSIVE = "memory_intensive"
    NETWORK_OPERATIONS = "network_operations"
    DISK_OPERATIONS = "disk_operations"
    BACKGROUND_TASKS = "background_tasks"
    LOGGING_DETAIL = "logging_detail"
    CACHING = "caching"
    MONITORING = "monitoring"
    ANALYTICS = "analytics"


@dataclass
class CapabilityConfig:
    """Configuration for a system capability."""

    name: str
    capability_type: CapabilityType
    priority: int  # 1-10, higher = more important
    normal_level: float = 1.0  # Full capability (100%)
    reduced_level: float = 0.7  # Reduced capability (70%)
    minimal_level: float = 0.3  # Minimal capability (30%)
    emergency_level: float = 0.1  # Emergency capability (10%)
    can_disable: bool = True  # Can be completely disabled

    def get_level(self, degradation: DegradationLevel) -> float:
        """Get capability level for degradation level."""
        if degradation == DegradationLevel.NORMAL:
            return self.normal_level
        elif degradation == DegradationLevel.REDUCED:
            return self.reduced_level
        elif degradation == DegradationLevel.MINIMAL:
            return self.minimal_level
        elif degradation == DegradationLevel.EMERGENCY:
            return self.emergency_level
        else:
            return 0.0


@dataclass
class DegradationState:
    """Current state of system degradation."""

    level: DegradationLevel = DegradationLevel.NORMAL
    active_since: float = field(default_factory=time.time)
    reason: str = ""
    affected_capabilities: set[str] = field(default_factory=set)
    triggered_by: str | None = None
    auto_recovery_enabled: bool = True


class GracefulDegradationManager:
    """Manages graceful degradation of system capabilities."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.GracefulDegradationManager")

        # State management
        self.current_state = DegradationState()
        self.degradation_history: list[DegradationState] = []
        self._lock = threading.RLock()

        # Capability configurations
        self.capabilities: dict[str, CapabilityConfig] = {}
        self._setup_default_capabilities()

        # Degradation triggers and callbacks
        self.degradation_triggers: dict[DegradationLevel, list[Callable]] = {
            level: [] for level in DegradationLevel
        }
        self.recovery_callbacks: list[
            Callable[[DegradationLevel, DegradationLevel], None]
        ] = []

        # Auto-recovery settings
        self.auto_recovery_enabled = True
        self.recovery_check_interval = 30.0  # seconds
        self.recovery_threshold_time = (
            120.0  # seconds to wait before attempting recovery
        )

        # Monitoring
        self._monitoring_task: asyncio.Task | None = None
        self._monitoring = False

        # Statistics
        self.stats = {
            "total_degradations": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "time_in_levels": {level.value: 0.0 for level in DegradationLevel},
            "last_level_change": time.time(),
        }

    def _setup_default_capabilities(self):
        """Setup default system capabilities."""

        default_capabilities = [
            CapabilityConfig(
                name="agent_pool",
                capability_type=CapabilityType.AGENT_CONCURRENCY,
                priority=9,
                normal_level=1.0,  # 8 agents
                reduced_level=0.75,  # 6 agents
                minimal_level=0.5,  # 4 agents
                emergency_level=0.25,  # 2 agents
                can_disable=False,
            ),
            CapabilityConfig(
                name="gpu_acceleration",
                capability_type=CapabilityType.GPU_ACCELERATION,
                priority=6,
                normal_level=1.0,  # Full GPU
                reduced_level=0.7,  # Reduced GPU usage
                minimal_level=0.3,  # Minimal GPU
                emergency_level=0.0,  # CPU only
                can_disable=True,
            ),
            CapabilityConfig(
                name="memory_caching",
                capability_type=CapabilityType.CACHING,
                priority=5,
                normal_level=1.0,  # Full caching
                reduced_level=0.6,  # Reduced cache size
                minimal_level=0.2,  # Minimal cache
                emergency_level=0.0,  # No caching
                can_disable=True,
            ),
            CapabilityConfig(
                name="background_monitoring",
                capability_type=CapabilityType.MONITORING,
                priority=4,
                normal_level=1.0,  # Full monitoring
                reduced_level=0.5,  # Reduced frequency
                minimal_level=0.2,  # Basic monitoring
                emergency_level=0.0,  # No monitoring
                can_disable=True,
            ),
            CapabilityConfig(
                name="detailed_logging",
                capability_type=CapabilityType.LOGGING_DETAIL,
                priority=3,
                normal_level=1.0,  # DEBUG level
                reduced_level=0.7,  # INFO level
                minimal_level=0.4,  # WARNING level
                emergency_level=0.1,  # ERROR level only
                can_disable=False,
            ),
            CapabilityConfig(
                name="performance_analytics",
                capability_type=CapabilityType.ANALYTICS,
                priority=2,
                normal_level=1.0,  # Full analytics
                reduced_level=0.5,  # Basic analytics
                minimal_level=0.0,  # No analytics
                emergency_level=0.0,  # No analytics
                can_disable=True,
            ),
            CapabilityConfig(
                name="network_operations",
                capability_type=CapabilityType.NETWORK_OPERATIONS,
                priority=7,
                normal_level=1.0,  # All network ops
                reduced_level=0.8,  # Essential only
                minimal_level=0.5,  # Critical only
                emergency_level=0.2,  # Minimal network
                can_disable=False,
            ),
        ]

        for capability in default_capabilities:
            self.capabilities[capability.name] = capability

    def register_capability(self, capability: CapabilityConfig):
        """Register a new capability for degradation management."""
        with self._lock:
            self.capabilities[capability.name] = capability
            self.logger.info(f"Registered capability: {capability.name}")

    def register_degradation_trigger(
        self, level: DegradationLevel, callback: Callable[[], None]
    ):
        """Register a callback for when degradation reaches a specific level."""
        self.degradation_triggers[level].append(callback)

    def register_recovery_callback(
        self, callback: Callable[[DegradationLevel, DegradationLevel], None]
    ):
        """Register a callback for degradation level changes."""
        self.recovery_callbacks.append(callback)

    async def trigger_degradation(
        self,
        level: DegradationLevel,
        reason: str,
        triggered_by: str | None = None,
        force: bool = False,
    ) -> bool:
        """Trigger system degradation to specified level."""

        with self._lock:
            current_level = self.current_state.level

            # Check if degradation is necessary
            if not force and level.value <= current_level.value:
                self.logger.info(
                    f"Degradation to {level.value} not needed (current: {current_level.value})"
                )
                return False

            # Record previous state
            if current_level != level:
                self.degradation_history.append(self.current_state)

                # Update time tracking
                current_time = time.time()
                time_diff = current_time - self.stats["last_level_change"]
                self.stats["time_in_levels"][current_level.value] += time_diff
                self.stats["last_level_change"] = current_time

            # Create new degradation state
            new_state = DegradationState(
                level=level,
                active_since=time.time(),
                reason=reason,
                triggered_by=triggered_by,
                auto_recovery_enabled=self.auto_recovery_enabled,
            )

            self.logger.warning(
                f"Triggering degradation: {current_level.value} -> {level.value} (reason: {reason})"
            )

            # Apply degradation
            try:
                affected_capabilities = await self._apply_degradation(level)
                new_state.affected_capabilities = affected_capabilities

                self.current_state = new_state
                self.stats["total_degradations"] += 1

                # Execute degradation triggers
                for trigger in self.degradation_triggers.get(level, []):
                    try:
                        if asyncio.iscoroutinefunction(trigger):
                            await trigger()
                        else:
                            trigger()
                    except Exception as e:
                        self.logger.error(f"Degradation trigger failed: {e}")

                # Notify recovery callbacks
                for callback in self.recovery_callbacks:
                    try:
                        callback(current_level, level)
                    except Exception as e:
                        self.logger.error(f"Recovery callback failed: {e}")

                self.logger.info(
                    f"Degradation applied successfully: {len(affected_capabilities)} capabilities affected"
                )
                return True

            except Exception as e:
                self.logger.error(f"Failed to apply degradation: {e}", exc_info=True)
                return False

    async def _apply_degradation(self, level: DegradationLevel) -> set[str]:
        """Apply degradation settings to all capabilities."""

        affected_capabilities = set()

        for capability_name, capability in self.capabilities.items():
            try:
                new_level = capability.get_level(level)
                await self._apply_capability_degradation(capability, new_level)
                affected_capabilities.add(capability_name)

            except Exception as e:
                self.logger.error(
                    f"Failed to degrade capability {capability_name}: {e}"
                )

        return affected_capabilities

    async def _apply_capability_degradation(
        self, capability: CapabilityConfig, level: float
    ):
        """Apply degradation to a specific capability."""

        capability_type = capability.capability_type

        if capability_type == CapabilityType.AGENT_CONCURRENCY:
            # Adjust agent pool size
            import os

            base_agents = int(os.environ.get("BOLT_BASE_AGENTS", "8"))
            new_agent_count = max(1, int(base_agents * level))
            os.environ["BOLT_MAX_AGENTS"] = str(new_agent_count)
            self.logger.info(f"Adjusted agent count to {new_agent_count}")

        elif capability_type == CapabilityType.GPU_ACCELERATION:
            # Adjust GPU usage
            if level == 0.0:
                os.environ["BOLT_DISABLE_GPU"] = "true"
                os.environ["BOLT_CPU_ONLY"] = "true"
                self.logger.info("Disabled GPU acceleration")
            else:
                os.environ["BOLT_GPU_MEMORY_FRACTION"] = str(level)
                self.logger.info(f"Reduced GPU usage to {level:.1%}")

        elif capability_type == CapabilityType.CACHING:
            # Adjust caching behavior
            if level == 0.0:
                os.environ["BOLT_DISABLE_CACHE"] = "true"
                self.logger.info("Disabled caching")
            else:
                os.environ["BOLT_CACHE_SIZE_FRACTION"] = str(level)
                self.logger.info(f"Reduced cache size to {level:.1%}")

        elif capability_type == CapabilityType.LOGGING_DETAIL:
            # Adjust logging level
            if level >= 0.8:
                log_level = "DEBUG"
            elif level >= 0.6:
                log_level = "INFO"
            elif level >= 0.3:
                log_level = "WARNING"
            else:
                log_level = "ERROR"

            os.environ["BOLT_LOG_LEVEL"] = log_level
            self.logger.info(f"Adjusted logging level to {log_level}")

        elif capability_type == CapabilityType.MONITORING:
            # Adjust monitoring frequency
            if level == 0.0:
                os.environ["BOLT_DISABLE_MONITORING"] = "true"
                self.logger.info("Disabled monitoring")
            else:
                # Increase monitoring interval (reduce frequency)
                base_interval = 5.0  # seconds
                new_interval = base_interval / level
                os.environ["BOLT_MONITORING_INTERVAL"] = str(new_interval)
                self.logger.info(f"Adjusted monitoring interval to {new_interval:.1f}s")

        elif capability_type == CapabilityType.MEMORY_INTENSIVE:
            # Adjust memory-intensive operations
            os.environ["BOLT_MEMORY_CONSERVATIVE"] = "true" if level < 0.8 else "false"
            os.environ["BOLT_BATCH_SIZE_FRACTION"] = str(level)
            self.logger.info(f"Adjusted memory usage to {level:.1%}")

        elif capability_type == CapabilityType.ANALYTICS:
            # Adjust analytics and performance tracking
            if level == 0.0:
                os.environ["BOLT_DISABLE_ANALYTICS"] = "true"
                self.logger.info("Disabled analytics")
            else:
                os.environ["BOLT_ANALYTICS_SAMPLE_RATE"] = str(level)
                self.logger.info(f"Reduced analytics sampling to {level:.1%}")

    async def attempt_recovery(self) -> bool:
        """Attempt to recover from current degradation level."""

        with self._lock:
            current_level = self.current_state.level

            if current_level == DegradationLevel.NORMAL:
                self.logger.debug("System already at normal level")
                return True

            if not self.current_state.auto_recovery_enabled:
                self.logger.info("Auto-recovery disabled for current degradation")
                return False

            # Check if enough time has passed
            time_since_degradation = time.time() - self.current_state.active_since
            if time_since_degradation < self.recovery_threshold_time:
                self.logger.debug(
                    f"Recovery threshold not met ({time_since_degradation:.1f}s < {self.recovery_threshold_time}s)"
                )
                return False

            self.logger.info(f"Attempting recovery from {current_level.value}")

            try:
                # Determine target recovery level
                target_level = self._determine_recovery_level(current_level)

                if target_level == current_level:
                    self.logger.info("No recovery possible at this time")
                    return False

                # Check if system conditions allow recovery
                if await self._check_recovery_conditions(target_level):
                    success = await self._execute_recovery(target_level)

                    if success:
                        self.stats["successful_recoveries"] += 1
                        self.logger.info(
                            f"Recovery successful: {current_level.value} -> {target_level.value}"
                        )
                    else:
                        self.stats["failed_recoveries"] += 1
                        self.logger.warning(
                            f"Recovery failed: {current_level.value} -> {target_level.value}"
                        )

                    return success
                else:
                    self.logger.info("System conditions not suitable for recovery")
                    return False

            except Exception as e:
                self.logger.error(f"Recovery attempt failed: {e}", exc_info=True)
                self.stats["failed_recoveries"] += 1
                return False

    def _determine_recovery_level(
        self, current_level: DegradationLevel
    ) -> DegradationLevel:
        """Determine appropriate recovery level."""

        if current_level == DegradationLevel.EMERGENCY:
            return DegradationLevel.MINIMAL
        elif current_level == DegradationLevel.MINIMAL:
            return DegradationLevel.REDUCED
        elif current_level == DegradationLevel.REDUCED:
            return DegradationLevel.NORMAL
        else:
            return current_level

    async def _check_recovery_conditions(self, target_level: DegradationLevel) -> bool:
        """Check if system conditions allow recovery to target level."""

        try:
            # Check system resources
            import psutil

            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1.0)

            # Resource thresholds for recovery levels
            thresholds = {
                DegradationLevel.NORMAL: {"memory": 75.0, "cpu": 70.0},
                DegradationLevel.REDUCED: {"memory": 85.0, "cpu": 80.0},
                DegradationLevel.MINIMAL: {"memory": 90.0, "cpu": 90.0},
                DegradationLevel.EMERGENCY: {"memory": 95.0, "cpu": 95.0},
            }

            threshold = thresholds.get(target_level, {"memory": 75.0, "cpu": 70.0})

            if memory.percent > threshold["memory"]:
                self.logger.info(
                    f"Memory usage too high for recovery: {memory.percent:.1f}% > {threshold['memory']}%"
                )
                return False

            if cpu_percent > threshold["cpu"]:
                self.logger.info(
                    f"CPU usage too high for recovery: {cpu_percent:.1f}% > {threshold['cpu']}%"
                )
                return False

            # Additional checks could include:
            # - GPU memory availability
            # - Disk space
            # - Network connectivity
            # - Error rates

            return True

        except Exception as e:
            self.logger.error(f"Failed to check recovery conditions: {e}")
            return False

    async def _execute_recovery(self, target_level: DegradationLevel) -> bool:
        """Execute recovery to target level."""

        try:
            current_level = self.current_state.level

            # Apply recovery (reverse degradation)
            affected_capabilities = await self._apply_degradation(target_level)

            # Update state
            recovery_state = DegradationState(
                level=target_level,
                active_since=time.time(),
                reason=f"Recovery from {current_level.value}",
                triggered_by="auto_recovery",
                auto_recovery_enabled=self.auto_recovery_enabled,
            )
            recovery_state.affected_capabilities = affected_capabilities

            # Update time tracking
            current_time = time.time()
            time_diff = current_time - self.stats["last_level_change"]
            self.stats["time_in_levels"][current_level.value] += time_diff
            self.stats["last_level_change"] = current_time

            self.degradation_history.append(self.current_state)
            self.current_state = recovery_state

            # Notify callbacks
            for callback in self.recovery_callbacks:
                try:
                    callback(current_level, target_level)
                except Exception as e:
                    self.logger.error(f"Recovery callback failed: {e}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to execute recovery: {e}", exc_info=True)
            return False

    async def force_recovery_to_normal(self) -> bool:
        """Force recovery to normal operation level."""
        self.logger.info("Forcing recovery to normal level")
        return await self.trigger_degradation(
            DegradationLevel.NORMAL,
            "Forced recovery to normal",
            triggered_by="manual",
            force=True,
        )

    async def start_monitoring(self):
        """Start automatic recovery monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self.logger.info("Started degradation monitoring")
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """Stop automatic recovery monitoring."""
        self._monitoring = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task
            self._monitoring_task = None

        self.logger.info("Stopped degradation monitoring")

    async def _monitoring_loop(self):
        """Main monitoring loop for automatic recovery."""
        try:
            while self._monitoring:
                if self.auto_recovery_enabled:
                    await self.attempt_recovery()

                await asyncio.sleep(self.recovery_check_interval)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Monitoring loop error: {e}", exc_info=True)

    def get_current_capability_levels(self) -> dict[str, float]:
        """Get current levels for all capabilities."""
        current_level = self.current_state.level

        return {
            name: capability.get_level(current_level)
            for name, capability in self.capabilities.items()
        }

    def get_degradation_stats(self) -> dict[str, Any]:
        """Get degradation statistics."""
        with self._lock:
            current_time = time.time()
            current_level = self.current_state.level

            # Update current level time
            time_diff = current_time - self.stats["last_level_change"]
            current_level_time = (
                self.stats["time_in_levels"][current_level.value] + time_diff
            )

            return {
                "current_level": current_level.value,
                "time_at_current_level": current_time - self.current_state.active_since,
                "degradation_reason": self.current_state.reason,
                "auto_recovery_enabled": self.auto_recovery_enabled,
                "affected_capabilities": list(self.current_state.affected_capabilities),
                "capability_levels": self.get_current_capability_levels(),
                "statistics": {
                    **self.stats,
                    "time_in_levels": {
                        **self.stats["time_in_levels"],
                        current_level.value: current_level_time,
                    },
                },
                "degradation_history_count": len(self.degradation_history),
            }


# Global degradation manager
_degradation_manager: GracefulDegradationManager | None = None


def get_degradation_manager() -> GracefulDegradationManager:
    """Get or create the global degradation manager."""
    global _degradation_manager
    if _degradation_manager is None:
        _degradation_manager = GracefulDegradationManager()
    return _degradation_manager
