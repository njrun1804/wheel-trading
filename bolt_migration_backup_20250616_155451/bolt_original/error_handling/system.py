"""
Bolt Error Handling System Integration

Provides a unified interface for all error handling components including
circuit breakers, recovery mechanisms, resource guards, and diagnostics.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from .circuit_breaker import CircuitBreakerManager, get_circuit_breaker
from .diagnostics import DiagnosticCollector, DiagnosticReport
from .exceptions import BoltException, wrap_exception
from .recovery import ErrorRecoveryManager, RecoveryConfiguration
from .resource_guards import (
    ResourceGuardManager,
    setup_default_guards,
)


@dataclass
class ErrorHandlingConfig:
    """Configuration for the error handling system."""

    enable_circuit_breakers: bool = True
    enable_resource_guards: bool = True
    enable_recovery_manager: bool = True
    enable_diagnostics: bool = True

    # Recovery settings
    max_retry_attempts: int = 3
    retry_delay_base: float = 1.0
    retry_delay_max: float = 60.0

    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: float = 60.0

    # Resource guard settings
    memory_threshold_high: float = 85.0
    cpu_threshold_high: float = 85.0
    gpu_threshold_high: float = 85.0

    # Monitoring settings
    health_check_interval: float = 300.0  # 5 minutes
    resource_check_interval: float = 10.0  # 10 seconds

    # Callbacks
    on_error_callback: Callable[[BoltException], None] | None = None
    on_recovery_callback: Callable[[BoltException, bool], None] | None = None


class BoltErrorHandlingSystem:
    """Unified error handling system for Bolt."""

    def __init__(self, config: ErrorHandlingConfig | None = None):
        self.config = config or ErrorHandlingConfig()
        self.logger = logging.getLogger(f"{__name__}.BoltErrorHandlingSystem")

        # Initialize components
        self.circuit_breaker_manager: CircuitBreakerManager | None = None
        self.resource_guard_manager: ResourceGuardManager | None = None
        self.recovery_manager: ErrorRecoveryManager | None = None
        self.diagnostic_collector: DiagnosticCollector | None = None

        # State tracking
        self.is_initialized = False
        self.is_monitoring = False
        self._monitoring_tasks: list[asyncio.Task] = []

        # Statistics
        self.stats = {
            "errors_handled": 0,
            "recoveries_attempted": 0,
            "recoveries_successful": 0,
            "circuit_breakers_opened": 0,
            "resource_guards_triggered": 0,
            "system_uptime": time.time(),
        }

        # Component health tracking
        self.component_health = {
            "circuit_breakers": True,
            "resource_guards": True,
            "recovery_manager": True,
            "diagnostics": True,
        }

    async def initialize(self) -> bool:
        """Initialize the error handling system."""
        if self.is_initialized:
            return True

        try:
            self.logger.info("Initializing Bolt error handling system...")

            # Initialize circuit breaker manager
            if self.config.enable_circuit_breakers:
                try:
                    self.circuit_breaker_manager = CircuitBreakerManager()
                    self.logger.info("Circuit breaker manager initialized")
                except Exception as e:
                    self.logger.error(
                        f"Failed to initialize circuit breaker manager: {e}"
                    )
                    self.component_health["circuit_breakers"] = False

            # Initialize resource guard manager
            if self.config.enable_resource_guards:
                try:
                    self.resource_guard_manager = setup_default_guards()

                    # Configure resource guard thresholds
                    if "memory" in self.resource_guard_manager.guards:
                        memory_guard = self.resource_guard_manager.guards["memory"]
                        memory_guard.thresholds.high_threshold = (
                            self.config.memory_threshold_high
                        )

                    if "cpu" in self.resource_guard_manager.guards:
                        cpu_guard = self.resource_guard_manager.guards["cpu"]
                        cpu_guard.thresholds.high_threshold = (
                            self.config.cpu_threshold_high
                        )

                    if "gpu" in self.resource_guard_manager.guards:
                        gpu_guard = self.resource_guard_manager.guards["gpu"]
                        gpu_guard.thresholds.high_threshold = (
                            self.config.gpu_threshold_high
                        )

                    self.logger.info("Resource guard manager initialized")
                except Exception as e:
                    self.logger.error(
                        f"Failed to initialize resource guard manager: {e}"
                    )
                    self.component_health["resource_guards"] = False

            # Initialize recovery manager
            if self.config.enable_recovery_manager:
                try:
                    recovery_config = RecoveryConfiguration(
                        max_retry_attempts=self.config.max_retry_attempts,
                        retry_delay_base=self.config.retry_delay_base,
                        retry_delay_max=self.config.retry_delay_max,
                        circuit_breaker_threshold=self.config.circuit_breaker_failure_threshold,
                        circuit_breaker_timeout=self.config.circuit_breaker_timeout,
                    )
                    self.recovery_manager = ErrorRecoveryManager(recovery_config)

                    # Register components with recovery manager
                    self.recovery_manager.register_components(
                        memory_manager=self.resource_guard_manager.get_guard("memory")
                        if self.resource_guard_manager
                        else None
                    )

                    self.logger.info("Recovery manager initialized")
                except Exception as e:
                    self.logger.error(f"Failed to initialize recovery manager: {e}")
                    self.component_health["recovery_manager"] = False

            # Initialize diagnostic collector
            if self.config.enable_diagnostics:
                try:
                    self.diagnostic_collector = DiagnosticCollector()
                    self.logger.info("Diagnostic collector initialized")
                except Exception as e:
                    self.logger.error(f"Failed to initialize diagnostic collector: {e}")
                    self.component_health["diagnostics"] = False

            self.is_initialized = True
            self.logger.info("Bolt error handling system initialized successfully")

            # Start monitoring if any components are available
            if any(self.component_health.values()):
                await self.start_monitoring()

            return True

        except Exception as e:
            self.logger.error(
                f"Failed to initialize error handling system: {e}", exc_info=True
            )
            return False

    async def start_monitoring(self):
        """Start system monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.logger.info("Starting error handling system monitoring")

        # Start resource guard monitoring
        if self.resource_guard_manager and self.component_health["resource_guards"]:
            try:
                await self.resource_guard_manager.start_all_monitoring()
                self.logger.info("Resource guard monitoring started")
            except Exception as e:
                self.logger.error(f"Failed to start resource guard monitoring: {e}")

        # Start periodic health checks
        if self.diagnostic_collector and self.component_health["diagnostics"]:
            health_check_task = asyncio.create_task(self._health_check_loop())
            self._monitoring_tasks.append(health_check_task)

        # Start system statistics collection
        stats_task = asyncio.create_task(self._stats_collection_loop())
        self._monitoring_tasks.append(stats_task)

    async def stop_monitoring(self):
        """Stop system monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        self.logger.info("Stopping error handling system monitoring")

        # Cancel monitoring tasks
        for task in self._monitoring_tasks:
            task.cancel()

        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)

        self._monitoring_tasks.clear()

        # Stop resource guard monitoring
        if self.resource_guard_manager:
            try:
                await self.resource_guard_manager.stop_all_monitoring()
            except Exception as e:
                self.logger.error(f"Failed to stop resource guard monitoring: {e}")

    async def handle_error(
        self, error: Exception, context: dict[str, Any] | None = None
    ) -> tuple[bool, Any | None]:
        """Handle an error through the complete error handling pipeline."""

        # Convert to BoltException if needed
        if not isinstance(error, BoltException):
            error = wrap_exception(error, context=context)

        self.stats["errors_handled"] += 1

        # Call error callback if configured
        if self.config.on_error_callback:
            try:
                self.config.on_error_callback(error)
            except Exception as callback_error:
                self.logger.warning(f"Error callback failed: {callback_error}")

        recovery_successful = False
        recovery_result = None

        # Attempt recovery if recovery manager is available
        if self.recovery_manager and self.component_health["recovery_manager"]:
            try:
                self.stats["recoveries_attempted"] += 1
                (
                    recovery_successful,
                    recovery_result,
                ) = await self.recovery_manager.handle_error(error, context)

                if recovery_successful:
                    self.stats["recoveries_successful"] += 1

            except Exception as recovery_error:
                self.logger.error(f"Recovery failed: {recovery_error}", exc_info=True)

        # Call recovery callback if configured
        if self.config.on_recovery_callback:
            try:
                self.config.on_recovery_callback(error, recovery_successful)
            except Exception as callback_error:
                self.logger.warning(f"Recovery callback failed: {callback_error}")

        return recovery_successful, recovery_result

    @asynccontextmanager
    async def protected_operation(
        self,
        operation_name: str,
        circuit_breaker: bool = True,
        resource_monitoring: bool = True,
    ):
        """Context manager for protected operations with full error handling."""

        circuit_breaker_instance = None
        if circuit_breaker and self.circuit_breaker_manager:
            circuit_breaker_instance = get_circuit_breaker(operation_name)

        start_time = time.time()

        try:
            # Check circuit breaker
            if circuit_breaker_instance and not circuit_breaker_instance.is_available():
                raise BoltException(
                    f"Circuit breaker open for operation: {operation_name}"
                )

            # Check resource availability
            if resource_monitoring and self.resource_guard_manager:
                critical_resources = (
                    self.resource_guard_manager.get_critical_resources()
                )
                if critical_resources:
                    raise BoltException(
                        f"Critical resource constraints: {', '.join(critical_resources)}"
                    )

            yield

            # Record success
            if circuit_breaker_instance:
                # This would be handled by the circuit breaker's context manager
                pass

        except Exception as e:
            # Handle the error through our pipeline
            await self.handle_error(
                e, {"operation": operation_name, "duration": time.time() - start_time}
            )
            raise

    async def get_system_health(self) -> dict[str, Any]:
        """Get comprehensive system health information."""

        health_info = {
            "timestamp": time.time(),
            "uptime": time.time() - self.stats["system_uptime"],
            "initialized": self.is_initialized,
            "monitoring": self.is_monitoring,
            "component_health": self.component_health.copy(),
            "statistics": self.stats.copy(),
        }

        # Add component-specific health information
        if self.circuit_breaker_manager and self.component_health["circuit_breakers"]:
            health_info[
                "circuit_breakers"
            ] = self.circuit_breaker_manager.get_all_stats()

        if self.resource_guard_manager and self.component_health["resource_guards"]:
            health_info["resource_guards"] = self.resource_guard_manager.get_all_stats()
            health_info[
                "system_health_score"
            ] = self.resource_guard_manager.get_system_health_score()

        if self.recovery_manager and self.component_health["recovery_manager"]:
            health_info["recovery_stats"] = self.recovery_manager.get_recovery_stats()

        return health_info

    async def run_diagnostics(self) -> DiagnosticReport | None:
        """Run comprehensive system diagnostics."""
        if not self.diagnostic_collector:
            return None

        try:
            return await self.diagnostic_collector.collect_full_diagnostic()
        except Exception as e:
            self.logger.error(f"Diagnostic collection failed: {e}", exc_info=True)
            return None

    async def emergency_shutdown(self) -> bool:
        """Emergency shutdown of the error handling system."""
        try:
            self.logger.critical("Emergency shutdown initiated")

            # Stop monitoring
            await self.stop_monitoring()

            # Apply emergency measures on all resource guards
            if self.resource_guard_manager:
                for guard in self.resource_guard_manager.guards.values():
                    try:
                        usage = await guard.check_usage()
                        await guard.apply_emergency_measures(usage)
                    except Exception as e:
                        self.logger.error(
                            f"Emergency measures failed for {guard.name}: {e}"
                        )

            # Force close all circuit breakers
            if self.circuit_breaker_manager:
                self.circuit_breaker_manager.force_close_all()

            self.logger.critical("Emergency shutdown completed")
            return True

        except Exception as e:
            self.logger.error(f"Emergency shutdown failed: {e}", exc_info=True)
            return False

    async def _health_check_loop(self):
        """Periodic health check loop."""
        try:
            while self.is_monitoring:
                try:
                    if self.diagnostic_collector:
                        # Run lightweight health checks
                        health_checks = (
                            await self.diagnostic_collector.health_checker.run_all_checks()
                        )

                        # Check for critical issues
                        critical_issues = [
                            hc for hc in health_checks if hc.status == "critical"
                        ]
                        if critical_issues:
                            self.logger.warning(
                                f"Critical health issues detected: {len(critical_issues)}"
                            )

                            # Auto-trigger recovery for some issues
                            for issue in critical_issues:
                                if issue.name == "system_resources":
                                    # Trigger resource guard emergency measures
                                    if self.resource_guard_manager:
                                        critical_resources = (
                                            self.resource_guard_manager.get_critical_resources()
                                        )
                                        if critical_resources:
                                            self.logger.warning(
                                                f"Auto-triggering emergency measures for: {critical_resources}"
                                            )

                except Exception as e:
                    self.logger.error(f"Health check error: {e}")

                await asyncio.sleep(self.config.health_check_interval)

        except asyncio.CancelledError:
            pass

    async def _stats_collection_loop(self):
        """Periodic statistics collection loop."""
        try:
            while self.is_monitoring:
                try:
                    # Update component health status
                    for component, manager in [
                        ("circuit_breakers", self.circuit_breaker_manager),
                        ("resource_guards", self.resource_guard_manager),
                        ("recovery_manager", self.recovery_manager),
                        ("diagnostics", self.diagnostic_collector),
                    ]:
                        if manager is None:
                            self.component_health[component] = False
                        # Additional health checks could be added here

                except Exception as e:
                    self.logger.error(f"Stats collection error: {e}")

                await asyncio.sleep(60.0)  # Update every minute

        except asyncio.CancelledError:
            pass


# Global error handling system instance
_error_handling_system: BoltErrorHandlingSystem | None = None


def get_error_handling_system(
    config: ErrorHandlingConfig | None = None,
) -> BoltErrorHandlingSystem:
    """Get or create the global error handling system."""
    global _error_handling_system

    if _error_handling_system is None:
        _error_handling_system = BoltErrorHandlingSystem(config)

    return _error_handling_system


async def initialize_error_handling(config: ErrorHandlingConfig | None = None) -> bool:
    """Initialize the global error handling system."""
    system = get_error_handling_system(config)
    return await system.initialize()


# Convenience functions for common operations


async def handle_error(
    error: Exception, context: dict[str, Any] | None = None
) -> tuple[bool, Any | None]:
    """Handle an error through the global error handling system."""
    system = get_error_handling_system()
    if not system.is_initialized:
        await system.initialize()

    return await system.handle_error(error, context)


def protected_operation(operation_name: str, **kwargs):
    """Decorator/context manager for protected operations."""
    system = get_error_handling_system()
    return system.protected_operation(operation_name, **kwargs)


async def get_system_health() -> dict[str, Any]:
    """Get system health from the global error handling system."""
    system = get_error_handling_system()
    return await system.get_system_health()


async def run_diagnostics() -> DiagnosticReport | None:
    """Run diagnostics through the global error handling system."""
    system = get_error_handling_system()
    return await system.run_diagnostics()
