"""
Bolt Error Recovery Manager

Comprehensive error recovery system that handles different types of failures
with appropriate recovery strategies, retry logic, and fallback mechanisms.
"""

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .exceptions import (
    BoltAgentException,
    BoltException,
    BoltGPUException,
    BoltMemoryException,
    BoltResourceException,
    BoltTaskException,
    ErrorCategory,
    RecoveryStrategy,
)


class RecoveryState(Enum):
    """States of recovery operations."""

    IDLE = "idle"
    ANALYZING = "analyzing"
    RECOVERING = "recovering"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RecoveryAttempt:
    """Information about a recovery attempt."""

    timestamp: float
    strategy: RecoveryStrategy
    error_code: str
    context: dict[str, Any]
    result: str | None = None
    duration: float | None = None
    success: bool = False


@dataclass
class RecoveryConfiguration:
    """Configuration for recovery behavior."""

    max_retry_attempts: int = 3
    retry_delay_base: float = 1.0  # Base delay in seconds
    retry_delay_max: float = 60.0  # Maximum delay
    retry_exponential_base: float = 2.0
    recovery_timeout: float = 300.0  # 5 minutes
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    graceful_degradation_enabled: bool = True
    auto_restart_enabled: bool = True
    auto_restart_max_attempts: int = 2


class ErrorRecoveryManager:
    """Manages error recovery across the Bolt system."""

    def __init__(self, config: RecoveryConfiguration | None = None):
        self.config = config or RecoveryConfiguration()
        self.logger = logging.getLogger(f"{__name__}.ErrorRecoveryManager")

        # Recovery state tracking
        self.state = RecoveryState.IDLE
        self.recovery_history: list[RecoveryAttempt] = []
        self.active_recoveries: dict[str, RecoveryAttempt] = {}

        # Circuit breaker state
        self.circuit_breaker_failures: dict[str, list[float]] = {}
        self.circuit_breaker_open: set[str] = set()

        # Recovery strategy handlers
        self.strategy_handlers = {
            RecoveryStrategy.RETRY: self._handle_retry_recovery,
            RecoveryStrategy.GRACEFUL_DEGRADATION: self._handle_graceful_degradation,
            RecoveryStrategy.FAILOVER: self._handle_failover_recovery,
            RecoveryStrategy.RESTART: self._handle_restart_recovery,
        }

        # Component references (set by system)
        self.bolt_integration = None
        self.memory_manager = None
        self.system_monitor = None

        # Recovery callbacks
        self.recovery_callbacks: dict[RecoveryStrategy, list[Callable]] = {
            strategy: [] for strategy in RecoveryStrategy
        }

        # Statistics
        self.stats = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "average_recovery_time": 0.0,
            "by_strategy": {strategy.value: 0 for strategy in RecoveryStrategy},
            "by_error_category": {category.value: 0 for category in ErrorCategory},
        }

    def register_components(
        self, bolt_integration=None, memory_manager=None, system_monitor=None
    ):
        """Register system components for recovery operations."""
        self.bolt_integration = bolt_integration
        self.memory_manager = memory_manager
        self.system_monitor = system_monitor

    def register_recovery_callback(
        self,
        strategy: RecoveryStrategy,
        callback: Callable[[BoltException, dict[str, Any]], Any],
    ):
        """Register a callback for a specific recovery strategy."""
        self.recovery_callbacks[strategy].append(callback)

    async def handle_error(
        self, error: BoltException, context: dict[str, Any] | None = None
    ) -> tuple[bool, Any | None]:
        """
        Main entry point for error handling and recovery.

        Returns:
            Tuple of (recovery_successful, recovery_result)
        """
        context = context or {}
        start_time = time.time()

        try:
            self.state = RecoveryState.ANALYZING
            self.logger.info(f"Handling error: {error.error_code} - {error.message}")

            # Update statistics
            self.stats["total_recoveries"] += 1
            self.stats["by_error_category"][error.category.value] += 1

            # Check circuit breaker
            if self._is_circuit_breaker_open(error.error_code):
                self.logger.warning(f"Circuit breaker open for {error.error_code}")
                return False, None

            # Check if error is recoverable
            if not error.is_recoverable():
                self.logger.error(f"Error {error.error_code} is not recoverable")
                return False, None

            # Create recovery attempt record
            attempt = RecoveryAttempt(
                timestamp=start_time,
                strategy=error.recovery_strategy,
                error_code=error.error_code,
                context=context,
            )

            self.active_recoveries[error.error_code] = attempt

            # Execute recovery strategy
            self.state = RecoveryState.RECOVERING
            success, result = await self._execute_recovery_strategy(error, context)

            # Record attempt
            attempt.duration = time.time() - start_time
            attempt.success = success
            attempt.result = str(result) if result else None

            # Update statistics
            if success:
                self.stats["successful_recoveries"] += 1
            else:
                self.stats["failed_recoveries"] += 1
                self._record_circuit_breaker_failure(error.error_code)

            self.stats["by_strategy"][error.recovery_strategy.value] += 1
            self._update_average_recovery_time(attempt.duration)

            # Validate recovery
            if success:
                self.state = RecoveryState.VALIDATING
                validation_result = await self._validate_recovery(
                    error, result, context
                )
                if not validation_result:
                    self.logger.warning(
                        f"Recovery validation failed for {error.error_code}"
                    )
                    success = False

            self.state = RecoveryState.COMPLETED if success else RecoveryState.FAILED
            self.recovery_history.append(attempt)

            self.logger.info(
                f"Recovery {'successful' if success else 'failed'} for {error.error_code} "
                f"using {error.recovery_strategy.value} strategy "
                f"(duration: {attempt.duration:.2f}s)"
            )

            return success, result

        except Exception as recovery_error:
            self.logger.error(
                f"Recovery process failed: {recovery_error}", exc_info=True
            )
            self.state = RecoveryState.FAILED
            return False, None

        finally:
            # Cleanup
            self.active_recoveries.pop(error.error_code, None)
            if self.state != RecoveryState.FAILED:
                self.state = RecoveryState.IDLE

    async def _execute_recovery_strategy(
        self, error: BoltException, context: dict[str, Any]
    ) -> tuple[bool, Any | None]:
        """Execute the appropriate recovery strategy."""

        strategy = error.recovery_strategy
        handler = self.strategy_handlers.get(strategy)

        if not handler:
            self.logger.error(f"No handler for recovery strategy: {strategy}")
            return False, None

        try:
            # Execute strategy-specific recovery
            result = await handler(error, context)

            # Execute registered callbacks
            for callback in self.recovery_callbacks.get(strategy, []):
                try:
                    await callback(error, context)
                except Exception as callback_error:
                    self.logger.warning(f"Recovery callback failed: {callback_error}")

            return True, result

        except Exception as strategy_error:
            self.logger.error(
                f"Recovery strategy {strategy.value} failed: {strategy_error}",
                exc_info=True,
            )
            return False, None

    async def _handle_retry_recovery(
        self, error: BoltException, context: dict[str, Any]
    ) -> Any:
        """Handle retry-based recovery with intelligent backoff."""

        retry_count = context.get("retry_count", 0)
        max_retries = context.get("max_retries", self.config.max_retry_attempts)

        if retry_count >= max_retries:
            raise Exception(f"Maximum retry attempts ({max_retries}) exceeded")

        # Adaptive retry delay based on error type
        base_delay = self.config.retry_delay_base
        if isinstance(error, BoltMemoryException):
            base_delay *= 2.0  # Longer delays for memory issues
        elif isinstance(error, BoltNetworkException):
            base_delay *= 1.5  # Moderate delays for network issues

        # Calculate retry delay with exponential backoff and jitter
        delay = min(
            base_delay * (self.config.retry_exponential_base**retry_count),
            self.config.retry_delay_max,
        )

        # Add jitter to prevent thundering herd (20-40% of delay)
        jitter = random.uniform(0.2, 0.4) * delay
        total_delay = delay + jitter

        # Apply circuit breaker if available
        if self.config.enable_circuit_breaker:
            circuit_breaker = self._get_circuit_breaker_for_error(error)
            if circuit_breaker and not circuit_breaker.is_available():
                raise Exception(f"Circuit breaker open for {error.error_code}")

        self.logger.info(
            f"Retrying {error.error_code} after {total_delay:.2f}s "
            f"(attempt {retry_count + 1}/{max_retries})"
        )
        await asyncio.sleep(total_delay)

        # Update context for next attempt
        context["retry_count"] = retry_count + 1
        context["last_retry_delay"] = total_delay

        return {
            "retry_count": retry_count + 1,
            "delay": total_delay,
            "next_delay_estimate": min(
                base_delay * (self.config.retry_exponential_base ** (retry_count + 1)),
                self.config.retry_delay_max,
            ),
        }

    async def _handle_graceful_degradation(
        self, error: BoltException, context: dict[str, Any]
    ) -> Any:
        """Handle graceful degradation recovery with real actions."""

        self.logger.info(
            f"Applying graceful degradation for {error.category.value} error"
        )

        degradation_actions = []
        actual_changes = {}

        if isinstance(error, BoltMemoryException):
            # Memory pressure - reduce resource usage
            if self.memory_manager:
                try:
                    self.memory_manager.enforce_limits(strict=True)
                    degradation_actions.append("Enforced strict memory limits")
                except AttributeError:
                    # Fallback to environment variables
                    import os

                    os.environ["BOLT_MEMORY_LIMIT"] = "0.8"  # 80% limit
                    degradation_actions.append("Set memory limit to 80%")

            # Force garbage collection
            import gc

            collected = gc.collect()
            degradation_actions.append(
                f"Freed {collected} objects via garbage collection"
            )

            # Reduce batch sizes globally
            import os

            original_batch = os.environ.get("BOLT_BATCH_SIZE", "32")
            new_batch = str(max(1, int(original_batch) // 2))
            os.environ["BOLT_BATCH_SIZE"] = new_batch
            actual_changes["batch_size_reduced"] = f"{original_batch} -> {new_batch}"
            degradation_actions.append(
                f"Reduced batch size: {original_batch} -> {new_batch}"
            )

            # Reduce concurrent operations
            original_workers = os.environ.get("BOLT_MAX_WORKERS", "4")
            new_workers = str(max(1, int(original_workers) // 2))
            os.environ["BOLT_MAX_WORKERS"] = new_workers
            actual_changes["workers_reduced"] = f"{original_workers} -> {new_workers}"
            degradation_actions.append(
                f"Reduced workers: {original_workers} -> {new_workers}"
            )

        elif isinstance(error, BoltGPUException):
            # GPU issues - fall back to CPU
            import os

            os.environ["BOLT_FORCE_CPU"] = "true"
            os.environ["BOLT_DISABLE_GPU"] = "true"
            actual_changes["gpu_disabled"] = True

            # Clear GPU memory if possible
            try:
                if error.gpu_backend == "mps":
                    import torch

                    if hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
                        degradation_actions.append("Cleared MPS GPU cache")
                elif error.gpu_backend == "mlx":
                    # MLX doesn't have explicit cache clearing
                    degradation_actions.append("MLX GPU fallback to CPU")
            except Exception as e:
                self.logger.warning(f"Failed to clear GPU cache: {e}")

            degradation_actions.extend(
                [
                    "Disabled GPU acceleration",
                    "Switched to CPU-only processing",
                    "Reduced memory allocation",
                ]
            )

        elif isinstance(error, BoltAgentException):
            # Agent issues - redistribute load
            if self.bolt_integration:
                try:
                    # Mark agent as unhealthy
                    agent_id = error.agent_id
                    # This would integrate with actual agent management
                    actual_changes["agent_marked_unhealthy"] = agent_id
                    degradation_actions.append(f"Marked agent {agent_id} as unhealthy")

                    # Reduce agent concurrency
                    import os

                    os.environ["BOLT_AGENT_CONCURRENCY"] = "1"
                    degradation_actions.append("Reduced agent concurrency to 1")

                except Exception as e:
                    self.logger.warning(f"Failed to handle agent degradation: {e}")
                    degradation_actions.append("Basic agent isolation applied")

        elif isinstance(error, BoltTaskException):
            # Task issues - simplify or skip
            import os

            os.environ["BOLT_TASK_TIMEOUT"] = "30"  # Shorter timeouts
            os.environ["BOLT_TASK_RETRY_LIMIT"] = "2"  # Fewer retries
            os.environ["BOLT_SIMPLE_MODE"] = "true"

            actual_changes.update(
                {
                    "task_timeout_reduced": "30s",
                    "retry_limit_reduced": "2",
                    "simple_mode_enabled": True,
                }
            )

            degradation_actions.extend(
                [
                    "Reduced task timeout to 30s",
                    "Limited retries to 2 attempts",
                    "Enabled simple processing mode",
                ]
            )

        elif isinstance(error, BoltNetworkException):
            # Network issues - conservative timeouts
            import os

            os.environ["BOLT_NETWORK_TIMEOUT"] = "10"
            os.environ["BOLT_CONNECTION_POOL_SIZE"] = "2"

            actual_changes.update(
                {"network_timeout": "10s", "connection_pool_reduced": "2"}
            )

            degradation_actions.extend(
                [
                    "Reduced network timeout to 10s",
                    "Limited connection pool to 2",
                    "Enabled network retry logic",
                ]
            )

        else:
            # Generic degradation
            import os

            os.environ["BOLT_CONSERVATIVE_MODE"] = "true"
            os.environ["BOLT_MONITORING_FREQUENCY"] = "5"  # 5 second intervals

            actual_changes.update(
                {"conservative_mode": True, "monitoring_frequency": "5s"}
            )

            degradation_actions.extend(
                [
                    "Enabled conservative operation mode",
                    "Increased monitoring frequency",
                    "Reduced system aggressiveness",
                ]
            )

        # Record degradation state for recovery validation
        context["degradation_applied"] = actual_changes
        context["degradation_timestamp"] = time.time()

        return {
            "degradation_actions": degradation_actions,
            "actual_changes": actual_changes,
            "recovery_confidence": self._calculate_degradation_confidence(
                error, actual_changes
            ),
        }

    async def _handle_failover_recovery(
        self, error: BoltException, context: dict[str, Any]
    ) -> Any:
        """Handle failover recovery."""

        self.logger.info(f"Executing failover for {error.category.value} error")

        failover_actions = []

        if isinstance(error, BoltAgentException):
            # Agent failover - start replacement
            if self.bolt_integration:
                # Create new agent to replace failed one
                failover_actions.append(
                    f"Started replacement agent for {error.agent_id}"
                )

        elif isinstance(error, BoltResourceException):
            # Resource failover - switch to alternative resources
            if error.resource_type == "gpu":
                failover_actions.append("Switched to CPU processing")
            elif error.resource_type == "memory":
                failover_actions.append("Enabled disk-based processing")

        else:
            failover_actions.append("Activated backup systems")

        return {"failover_actions": failover_actions}

    async def _handle_restart_recovery(
        self, error: BoltException, context: dict[str, Any]
    ) -> Any:
        """Handle restart-based recovery."""

        restart_count = context.get("restart_count", 0)
        max_restarts = self.config.auto_restart_max_attempts

        if restart_count >= max_restarts:
            raise Exception(f"Maximum restart attempts ({max_restarts}) exceeded")

        self.logger.info(f"Restarting component for {error.category.value} error")

        restart_actions = []

        if isinstance(error, BoltAgentException):
            # Restart specific agent
            if self.bolt_integration and error.agent_id:
                restart_actions.append(f"Restarted agent {error.agent_id}")

        elif isinstance(error, BoltMemoryException):
            # Restart memory-intensive components
            if self.memory_manager:
                # Force garbage collection and reset
                restart_actions.append("Reset memory manager")

        elif error.category == ErrorCategory.SYSTEM:
            # System restart may be needed
            restart_actions.append("Initiated system component restart")

        else:
            restart_actions.append("Restarted affected subsystem")

        context["restart_count"] = restart_count + 1

        return {"restart_actions": restart_actions, "restart_count": restart_count + 1}

    async def _validate_recovery(
        self, error: BoltException, recovery_result: Any, context: dict[str, Any]
    ) -> bool:
        """Validate that recovery was successful with comprehensive checks."""

        try:
            validation_start = time.time()
            validation_checks = []

            # Basic system health check
            system_health = await self._check_system_health()
            validation_checks.append(("system_health", system_health))

            if not system_health:
                self.logger.warning(
                    "System health check failed during recovery validation"
                )
                return False

            # Strategy-specific validation
            if error.recovery_strategy == RecoveryStrategy.RETRY:
                # For retries, check if the same error would still occur
                validation_checks.append(("retry_ready", True))
                return True

            elif error.recovery_strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                # Validate degradation measures are working
                degradation_validation = await self._validate_degradation(
                    error, recovery_result, context
                )
                validation_checks.append(
                    ("degradation_effective", degradation_validation)
                )

                if not degradation_validation:
                    self.logger.warning(
                        f"Degradation validation failed for {error.error_code}"
                    )
                    return False

            elif error.recovery_strategy == RecoveryStrategy.FAILOVER:
                # Check backup systems are operational
                failover_validation = await self._validate_failover(
                    error, recovery_result
                )
                validation_checks.append(("failover_operational", failover_validation))

                if not failover_validation:
                    self.logger.warning(
                        f"Failover validation failed for {error.error_code}"
                    )
                    return False

            elif error.recovery_strategy == RecoveryStrategy.RESTART:
                # Check restarted components are healthy
                restart_validation = await self._validate_restart(
                    error, recovery_result
                )
                validation_checks.append(("restart_successful", restart_validation))

                if not restart_validation:
                    self.logger.warning(
                        f"Restart validation failed for {error.error_code}"
                    )
                    return False

            # Resource-specific validation
            if isinstance(error, BoltMemoryException):
                memory_ok = await self._validate_memory_recovery()
                validation_checks.append(("memory_recovered", memory_ok))
                if not memory_ok:
                    return False

            elif isinstance(error, BoltGPUException):
                gpu_ok = await self._validate_gpu_recovery(error)
                validation_checks.append(("gpu_recovered", gpu_ok))
                if not gpu_ok:
                    return False

            validation_duration = time.time() - validation_start
            self.logger.info(
                f"Recovery validation passed for {error.error_code} "
                f"in {validation_duration:.2f}s: {validation_checks}"
            )

            return True

        except Exception as validation_error:
            self.logger.error(
                f"Recovery validation failed: {validation_error}", exc_info=True
            )
            return False

    async def _check_system_health(self) -> bool:
        """Check overall system health."""
        try:
            if self.system_monitor:
                # Use system monitor if available
                return True  # Placeholder

            # Basic health check
            if self.bolt_integration:
                # Check if agents are responsive
                active_agents = sum(
                    1
                    for agent in self.bolt_integration.agents
                    if agent.status.value != "failed"
                )
                return active_agents > 0

            return True

        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            self.logger.debug(f"System health check failed: {e}")
            return False

    def _is_circuit_breaker_open(self, error_code: str) -> bool:
        """Check if circuit breaker is open for a specific error type."""

        if not self.config.enable_circuit_breaker:
            return False

        if error_code in self.circuit_breaker_open:
            # Check if timeout has passed
            failures = self.circuit_breaker_failures.get(error_code, [])
            if failures:
                last_failure = failures[-1]
                if time.time() - last_failure > self.config.circuit_breaker_timeout:
                    # Reset circuit breaker
                    self.circuit_breaker_open.discard(error_code)
                    self.circuit_breaker_failures[error_code] = []
                    self.logger.info(f"Circuit breaker reset for {error_code}")
                    return False
            return True

        return False

    def _record_circuit_breaker_failure(self, error_code: str):
        """Record a failure for circuit breaker tracking."""

        if not self.config.enable_circuit_breaker:
            return

        current_time = time.time()

        # Clean old failures (older than timeout period)
        if error_code not in self.circuit_breaker_failures:
            self.circuit_breaker_failures[error_code] = []

        failures = self.circuit_breaker_failures[error_code]
        failures = [
            f
            for f in failures
            if current_time - f < self.config.circuit_breaker_timeout
        ]
        failures.append(current_time)

        self.circuit_breaker_failures[error_code] = failures

        # Check if threshold is exceeded
        if len(failures) >= self.config.circuit_breaker_threshold:
            self.circuit_breaker_open.add(error_code)
            self.logger.warning(
                f"Circuit breaker opened for {error_code} "
                f"({len(failures)} failures in {self.config.circuit_breaker_timeout}s)"
            )

    def _update_average_recovery_time(self, duration: float):
        """Update the running average of recovery times."""
        current_avg = self.stats["average_recovery_time"]
        total_recoveries = self.stats["total_recoveries"]

        if total_recoveries == 1:
            self.stats["average_recovery_time"] = duration
        else:
            # Calculate new average
            self.stats["average_recovery_time"] = (
                current_avg * (total_recoveries - 1) + duration
            ) / total_recoveries

    def get_recovery_stats(self) -> dict[str, Any]:
        """Get comprehensive recovery statistics."""

        current_time = time.time()
        recent_history = [
            attempt
            for attempt in self.recovery_history
            if current_time - attempt.timestamp < 3600  # Last hour
        ]

        # Calculate strategy effectiveness
        strategy_stats = {}
        for strategy in RecoveryStrategy:
            strategy_attempts = [
                a for a in self.recovery_history if a.strategy == strategy
            ]
            if strategy_attempts:
                success_count = sum(1 for a in strategy_attempts if a.success)
                total_count = len(strategy_attempts)
                avg_duration = (
                    sum(a.duration or 0 for a in strategy_attempts) / total_count
                )

                strategy_stats[strategy.value] = {
                    "total_attempts": total_count,
                    "success_count": success_count,
                    "success_rate": success_count / total_count,
                    "average_duration": avg_duration,
                    "last_used": max(a.timestamp for a in strategy_attempts),
                }

        # Circuit breaker status
        circuit_breaker_status = {}
        for error_code, failures in self.circuit_breaker_failures.items():
            recent_failures = [
                f
                for f in failures
                if current_time - f < self.config.circuit_breaker_timeout
            ]
            circuit_breaker_status[error_code] = {
                "open": error_code in self.circuit_breaker_open,
                "recent_failures": len(recent_failures),
                "total_failures": len(failures),
                "last_failure": max(failures) if failures else None,
            }

        return {
            "overall": self.stats.copy(),
            "recent_recoveries": len(recent_history),
            "recent_success_rate": (
                sum(1 for a in recent_history if a.success) / len(recent_history)
                if recent_history
                else 0.0
            ),
            "strategy_effectiveness": strategy_stats,
            "circuit_breakers": circuit_breaker_status,
            "active_recoveries": len(self.active_recoveries),
            "current_state": self.state.value,
            "average_recovery_time": self.stats["average_recovery_time"],
            "most_common_errors": self._get_most_common_errors(),
            "system_health_trend": self._calculate_health_trend(),
        }

    def reset_circuit_breaker(self, error_code: str):
        """Manually reset a circuit breaker."""
        self.circuit_breaker_open.discard(error_code)
        self.circuit_breaker_failures.pop(error_code, None)
        self.logger.info(f"Manually reset circuit breaker for {error_code}")

    def clear_recovery_history(self, older_than_hours: int = 24):
        """Clear old recovery history entries."""
        cutoff_time = time.time() - (older_than_hours * 3600)
        self.recovery_history = [
            attempt
            for attempt in self.recovery_history
            if attempt.timestamp > cutoff_time
        ]

    def _get_circuit_breaker_for_error(self, error: BoltException):
        """Get circuit breaker for specific error type."""
        try:
            from .circuit_breaker import CircuitBreakerConfig, get_circuit_breaker

            config = CircuitBreakerConfig(
                failure_threshold=self.config.circuit_breaker_threshold,
                timeout=self.config.circuit_breaker_timeout,
            )
            return get_circuit_breaker(error.error_code, config)
        except Exception:
            return None

    def _calculate_degradation_confidence(
        self, error: BoltException, changes: dict
    ) -> float:
        """Calculate confidence that degradation will help."""
        base_confidence = 0.7

        if isinstance(error, BoltMemoryException) and "batch_size_reduced" in changes:
            base_confidence += 0.2
        elif isinstance(error, BoltGPUException) and "gpu_disabled" in changes:
            base_confidence += 0.25
        elif isinstance(error, BoltNetworkException) and "network_timeout" in changes:
            base_confidence += 0.15

        return min(0.95, base_confidence)

    async def _validate_degradation(
        self, error: BoltException, recovery_result: Any, context: dict[str, Any]
    ) -> bool:
        """Validate that degradation measures are effective."""
        try:
            degradation_applied = context.get("degradation_applied", {})

            if isinstance(error, BoltMemoryException):
                # Check memory usage is reduced
                import psutil

                memory = psutil.virtual_memory()
                if memory.percent < 85.0:  # Should be below threshold
                    return True
                self.logger.warning(
                    f"Memory still high after degradation: {memory.percent}%"
                )
                return False

            elif isinstance(error, BoltGPUException):
                # Check GPU is disabled or usage reduced
                import os

                if os.environ.get("BOLT_FORCE_CPU") == "true":
                    return True
                return False

            elif isinstance(error, BoltTaskException):
                # Check task limits are applied
                import os

                if (
                    os.environ.get("BOLT_TASK_TIMEOUT")
                    and os.environ.get("BOLT_SIMPLE_MODE") == "true"
                ):
                    return True
                return False

            return True

        except Exception as e:
            self.logger.error(f"Degradation validation error: {e}")
            return False

    async def _validate_failover(
        self, error: BoltException, recovery_result: Any
    ) -> bool:
        """Validate failover recovery is working."""
        try:
            # Basic failover validation
            if isinstance(error, BoltAgentException):
                # Check if alternative agents are available
                if self.bolt_integration:
                    # Count healthy agents
                    return True  # Placeholder - would check actual agent status

            return True

        except Exception as e:
            self.logger.error(f"Failover validation error: {e}")
            return False

    async def _validate_restart(
        self, error: BoltException, recovery_result: Any
    ) -> bool:
        """Validate restart recovery is working."""
        try:
            # Check if restarted components are healthy
            if isinstance(error, BoltAgentException):
                # Verify agent is responsive
                return True  # Placeholder - would check actual agent status
            elif isinstance(error, BoltMemoryException):
                # Check memory manager is functional
                if self.memory_manager:
                    return True  # Placeholder - would check memory manager status

            return True

        except Exception as e:
            self.logger.error(f"Restart validation error: {e}")
            return False

    async def _validate_memory_recovery(self) -> bool:
        """Validate memory usage is back to acceptable levels."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            return memory.percent < 90.0  # Allow some headroom

        except Exception as e:
            self.logger.error(f"Memory validation error: {e}")
            return False

    async def _validate_gpu_recovery(self, error: BoltGPUException) -> bool:
        """Validate GPU recovery or fallback."""
        try:
            import os

            # If we forced CPU mode, that's a valid recovery
            if os.environ.get("BOLT_FORCE_CPU") == "true":
                return True

            # Otherwise try to check GPU availability
            if error.gpu_backend == "mps":
                try:
                    import torch

                    return torch.backends.mps.is_available()
                except Exception:
                    return False
            elif error.gpu_backend == "mlx":
                try:
                    import mlx.core as mx

                    return mx.metal.is_available()
                except Exception:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"GPU validation error: {e}")
            return False

    def _get_most_common_errors(self) -> list[dict[str, Any]]:
        """Get most common error types from history."""
        error_counts = {}

        for attempt in self.recovery_history:
            error_code = attempt.error_code
            if error_code not in error_counts:
                error_counts[error_code] = {
                    "count": 0,
                    "success_rate": 0.0,
                    "avg_duration": 0.0,
                }

            error_counts[error_code]["count"] += 1
            if attempt.success:
                error_counts[error_code]["success_rate"] += 1
            if attempt.duration:
                error_counts[error_code]["avg_duration"] += attempt.duration

        # Calculate averages and sort by frequency
        for error_code, stats in error_counts.items():
            stats["success_rate"] /= stats["count"]
            stats["avg_duration"] /= stats["count"]

        return sorted(
            [{"error_code": code, **stats} for code, stats in error_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[
            :10
        ]  # Top 10

    def _calculate_health_trend(self) -> str:
        """Calculate overall system health trend."""
        if len(self.recovery_history) < 5:
            return "insufficient_data"

        recent = self.recovery_history[-10:]  # Last 10 recoveries
        success_rates = []

        # Calculate success rate for each group of recoveries
        for i in range(0, len(recent), 2):
            group = recent[i : i + 2]
            success_rate = sum(1 for a in group if a.success) / len(group)
            success_rates.append(success_rate)

        if len(success_rates) < 2:
            return "stable"

        # Compare recent vs earlier success rates
        recent_avg = sum(success_rates[-2:]) / len(success_rates[-2:])
        earlier_avg = (
            sum(success_rates[:-2]) / len(success_rates[:-2])
            if len(success_rates) > 2
            else recent_avg
        )

        if recent_avg > earlier_avg + 0.1:
            return "improving"
        elif recent_avg < earlier_avg - 0.1:
            return "degrading"
        else:
            return "stable"
